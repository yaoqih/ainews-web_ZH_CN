---
companies:
- google
- gemini
- huggingface
- anthropic
- deepseek
- zhipu-ai
- tsinghua
- nvidia
date: '2024-06-25T07:02:13.519492Z'
description: '最新版本的 **Chrome Canary** 浏览器现已包含 **Gemini Nano** 的功能标志（feature flag），提供
  Prompt API 和设备端优化指南。该模型分为 Nano 1 和 Nano 2 两个版本，参数量分别为 **18亿 (1.8B)** 和 **32.5亿 (3.25B)**，在性能上与
  Gemini Pro 相比表现不俗。其基础模型和指令微调模型的权重已被提取并上传至 **HuggingFace**。


  在 AI 模型发布方面，**Anthropic** 推出了 **Claude 3.5 Sonnet**，其在部分基准测试中超越了 **GPT-4o**，运行速度是
  Opus 的两倍，并提供免费试用。**DeepSeek-Coder-V2** 在 HumanEval 测试中达到 **90.2%**，在 MATH 测试中达到
  **75.7%**，性能超过了 GPT-4-Turbo-0409；该模型参数量高达 **2360亿 (236B)**，支持 **12.8万 (128K)** 上下文长度。由**智谱
  AI/清华大学**开发的 **GLM-0520** 在编程和综合基准测试中名列前茅。**英伟达 (NVIDIA)** 发布了 **Nemotron-4 340B**，这是一个用于合成数据生成的开源模型系列。


  研究亮点包括：**TextGrad**，一个基于文本反馈实现自动微分的框架；**PlanRAG**，一种迭代式的“先规划再 RAG”决策技术；一篇关于 **goldfish
  loss**（金鱼损失）以减轻大语言模型记忆问题的论文；以及一种用于语言模型智能体（agents）的树搜索算法。'
id: 8e70e6f1-f2bd-4ccd-923a-0f2d00840700
models:
- gemini-nano
- gemini-pro
- claude-3.5-sonnet
- gpt-4o
- deepseek-coder-v2
- glm-0520
- nemotron-4-340b
- gpt-4-turbo-0409
original_slug: ainews-gemini-nano-50-90-of-gemini-pro-100ms
people:
- adcock_brett
- dair_ai
- lmsysorg
title: Gemini Nano：性能达 Gemini Pro 的 50-90%，推理延迟低于 100ms，支持端侧运行，现已在 Chrome Canary 浏览器中上线。
topics:
- model-quantization
- prompt-api
- optimization
- model-weights
- benchmarking
- code-generation
- math
- synthetic-data
- automatic-differentiation
- retrieval-augmented-generation
- mitigating-memorization
- tree-search
- inference-time-algorithms
---

<!-- buttondown-editor-mode: plaintext -->**window.ai.createTextSession() 就足够了**

> 2024年6月21日至6月24日的 AI 新闻。我们为您检查了 7 个 subreddit、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**415** 个频道和 **5896** 条消息）。预计节省阅读时间（按 200wpm 计算）：**660 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

最新的 [Chrome Canary](https://www.google.com/intl/en_au/chrome/canary/) 现在通过 feature flag 支持 Gemini Nano：

- Gemini Nano 的 Prompt API [chrome://flags/#prompt-api-for-gemini-nano](chrome://flags/#prompt-api-for-gemini-nano)
- 设备上的优化指南 (Optimization guide on device) [chrome://flags/#optimization-guide-on-device-model](chrome://flags/#prompt-api-for-gemini-nano)
- 导航至 [chrome://components/](chrome://components/) 并查找 Optimization Guide On Device Model；点击“检查更新”以开始下载。


您现在可以通过控制台访问该模型：`http://window.ai.createTextSession()`

 
![image.png](https://assets.buttondown.email/images/35e5c81e-fb99-46d5-9996-c335a5b4aae9.png?w=960&fit=max)
 

Nano 1 和 2 在 4-bit 量化下分别具有 1.8B 和 3.25B 参数，相对于 Gemini Pro 表现不错：

 
![image.png](https://assets.buttondown.email/images/8592d84c-0fa3-4fac-909d-ad06593c05a7.png?w=960&fit=max)
 


您应该看看这个关于它运行速度的[现场演示](https://x.com/mortenjust/status/1805190952358650251)。
 
![image.png](https://assets.buttondown.email/images/5dce2799-dc79-4533-b7eb-82c3724c0df7.png?w=960&fit=max)
 

最后，[基础模型和指令微调（instruct-tuned）模型的权重](https://x.com/reach_vb/status/1805226216997200145)已经被提取并发布到 HuggingFace。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 综述

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 模型发布与基准测试**

- **Anthropic Claude 3.5 Sonnet**：[@adcock_brett](https://twitter.com/adcock_brett/status/1804908080829726790) 指出 Anthropic 推出了 Claude 3.5 Sonnet，这是一款在某些基准测试中**超越 GPT-4o** 的升级模型。对于开发者来说，它的**速度是 Opus 的 2 倍，而价格仅为 Anthropic 之前顶级模型的 1/5**。对于消费者，它可以**完全免费试用**。[@lmsysorg](https://twitter.com/lmsysorg/status/1804967083358523559) 报告称 Claude 3.5 Sonnet 在 **Coding Arena 中已攀升至第 4 位，接近 GPT-4-Turbo 的水平**。它现在是**编程领域的顶级开放模型**。它在 Hard Prompts 中排名第 11，在综合通用问题中排名第 20。
- **DeepSeek-Coder-V2**：[@dair_ai](https://twitter.com/dair_ai/status/1804922107870036049) 指出 DeepSeek-Coder-V2 在代码和数学生成任务上可与闭源模型竞争。根据其报告，它在 **HumanEval 上达到 90.2%，在 MATH 上达到 75.7%**，高于 GPT-4-Turbo-0409 的表现。包括 **16B 和 236B 参数的模型，具有 128K 上下文长度**。
- **GLM-0520**：[@lmsysorg](https://twitter.com/lmsysorg/status/1804967083358523559) 报告称来自智谱 AI/清华大学的 GLM-0520 表现令人印象深刻，在**编程排名第 9，综合排名第 11**。国产 LLM 正变得比以往任何时候都更具竞争力！
- **Nemotron 340B**：[@dl_weekly](https://twitter.com/dl_weekly/status/1804951560356503900) 报告称 NVIDIA 发布了 Nemotron-4 340B，这是一个**开源模型系列，开发者可以使用它来生成合成数据，用于训练大型语言模型**。

**AI 研究论文**

- **TextGrad**: [@dair_ai](https://twitter.com/dair_ai/status/1804922109543477361) 指出 TextGrad 是一个通过 **LLM 提供的文本反馈进行反向传播，实现自动微分** 的新框架。这改进了单个组件，且自然语言有助于优化计算图。
- **PlanRAG**: [@dair_ai](https://twitter.com/dair_ai/status/1804922113985245385) 报道称 PlanRAG 通过一种名为 **迭代式先规划后 RAG (iterative plan-then-RAG)** 的新技术增强了决策能力。它包含两个步骤：1) LLM 通过检查数据模式和问题来生成决策计划；2) 检索器生成用于数据分析的查询。最后一步检查是否需要新计划进行进一步分析，并迭代之前的步骤或对数据做出决策。
- **Mitigating Memorization in LLMs**: [@dair_ai](https://twitter.com/dair_ai/status/1804922115637875086) 指出这篇论文提出了一种对 next-token prediction 目标的改进，称为 **goldfish loss，旨在帮助缓解对记忆训练数据的逐字生成**。
- **Tree Search for Language Model Agents**: [@dair_ai](https://twitter.com/dair_ai/status/1804922123896713254) 报道称这篇论文为 LM Agent 提出了一种 **推理时树搜索算法，以进行探索并实现多步推理**。该算法在交互式 Web 环境中进行了测试，并应用于 GPT-4o，显著提升了性能。

**AI 应用与演示**

- **Wayve PRISM-1**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908105815212100) 报道称 Wayve AI 推出了 PRISM-1，这是一种 **基于视频数据的 4D 场景（空间 3D + 时间）重建模型**。此类突破对于自动驾驶的发展至关重要。
- **Runway Gen-3 Alpha**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908283334959538) 指出 Runway 展示了 Gen-3 Alpha，这是一款可以 **根据文本提示词和图像生成 10 秒视频** 的新 AI 模型。这些人物角色 100% 由 AI 生成。
- **Hedra Character-1**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908305703227797) 报道称 Hedra 发布了 Character-1，这是一款可以将 **图像转换为唱歌肖像视频** 的新基础模型。其公开预览版 Web 应用可以生成长达 30 秒的具有表现力的说话、唱歌或说唱角色。
- **ElevenLabs Text/Video-to-Sound**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908328088227923) 指出 ElevenLabs 发布了一款全新的 **开源文本和视频转音效应用及 API**。开发者现在可以构建根据文本提示词生成音效或为无声视频添加声音的应用。

**梗与幽默**

- **Gilded Frogs**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1804981653808275800) 将 "Gilded Frogs" 定义为 **积累了巨大财富并佩戴奢华珠宝** 的青蛙，包括金链、镶嵌宝石的手镯和戒指，全身覆盖着钻石、红宝石和蓝宝石。
- **Llama.ttf**: [@osanseviero](https://twitter.com/osanseviero/status/1804883653085769960) 指出 Llama.ttf 是一个 **既是字体也是 LLM** 的文件。TinyStories (15M) 作为一个字体 🤯 字体引擎运行 LLM 的推理。这是将本地 LLM 推向极致的体现。
- **VCs Funding GPT Wrapper Startups**: [@abacaj](https://twitter.com/abacaj/status/1804976343471284326) 发布了一张梗图，调侃 **VC 资助 GPT 套壳 (wrapper) 初创公司**。
- **Philosophers vs ML Researchers**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1804986384022966385) 发布了一张梗图，对比了 **哲学家与 ML 研究员发表论文的数量**。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Stable Diffusion / AI 图像生成**

- **Pony Diffusion 模型给用户留下深刻印象**：在 /r/StableDiffusion 中，用户正在[发掘 Pony Diffusion 模型的各项功能和创意潜力](https://www.reddit.com/r/StableDiffusion/comments/1dmejz5/now_i_get_why_people_like_pony_so_much/)，发现它使用起来既有趣又令人耳目一新。一些人[承认低估了 Pony 的职责履行和 Prompt 遵循能力](https://www.reddit.com/r/StableDiffusion/comments/1dmhmdz/turns_out_im_the_immature_one/)。目前有用户请求提供[深入的 Pony 教程](https://www.reddit.com/r/StableDiffusion/comments/1dmisnf/are_there_any_thorough_tutorials_for_pony/)，以帮助生成理想的全年龄段动漫/漫画风格图像，同时避免产生意外的 NSFW 内容。

- **新技术与模型更新**：用户正在分享 [ComfyUI 中的背景替换、重照明和合成工作流](https://www.reddit.com/r/StableDiffusion/comments/1dn748i/background_replacement_relighting_and_composit_in_comfyui/)，并演示了[在 adetailer 模型中使用 [SEP] Token 处理多个 Prompt 的方法](https://www.reddit.com/r/StableDiffusion/comments/1dmwqqb/did_you_know_you_can_use_sep_to_use_multiple_prompts_for_adetailer_just_make_sure_you_get_the_order_right/)。[SD.Next 发布公告](https://www.reddit.com/r/StableDiffusion/comments/1dmox7j/sdnext_release_20240623/)强调了 10 多项改进，如量化 T5 编码器支持、PixArt-Sigma 变体、HunyuanDiT 1.1 以及针对低 VRAM GPU 的效率升级。[sd-scripts 现已支持训练 Stable Diffusion 3 模型](https://www.reddit.com/r/StableDiffusion/comments/1dmp2xx/sdscripts_finally_supports_to_train_sd3/)。

- **创意应用与模型对比**：尼古拉·特斯拉博物馆的一个展览展出了 [118 件使用 Stable Diffusion 创作的 AI 辅助艺术作品](https://www.reddit.com/r/StableDiffusion/comments/1dmj35z/elementally_ai/)，突显了 AI 在专业社区之外的应用。新的 LoRA 模型不断发布，例如用于北欧风格肖像的 [Aether Illustration](https://www.reddit.com/r/StableDiffusion/comments/1dmjk2x/aether_illustration_new_nordic_style_illustration/) 以及[适用于 SDXL 的黑白插画风格模型](https://www.reddit.com/r/StableDiffusion/comments/1dn74k4/black_white_illustration_style_lora_sdxl_civitai_link_in_comments/)。一项[针对 "woman lying on grass"（躺在草地上的女人）Prompt 的多种模型对比](https://www.reddit.com/r/StableDiffusion/comments/1dmt3c4/woman_lying_on_grass_comparasion_sd3_vs_sdxl_vs_sdxl_turbo_vs_dreamshaper_xl_lighting_vs_juggernaut_x_vs_stable_cascade_vs_epicrealism_5_vs_sd_15_vs_midjourney_vs_dalle_3_vs_adobe_firefly/)引发了关于各模型性能表现的热议。

- **许可协议讨论**：用户发现 [Stable Cascade 的初始权重曾以 MIT 许可证发布了约 4 天](https://www.reddit.com/r/StableDiffusion/comments/1dn6yjp/stable_cascade_weights_were_actually_mit_licensed/)，随后才更改为更严格的许可证，这暗示了 MIT 许可版本具有商业化使用的潜力。这导致人们纷纷下载该特定版本。

**ChatGPT / AI 助手**

- **AI 生成的游戏令用户惊叹**：在 /r/ChatGPT 中，AI 助手 Claude [在聊天界面内创建了一个可玩的 3D 第一人称射击游戏](https://www.reddit.com/r/ChatGPT/comments/1dmejz5/claude_made_me_a_3d_firstperson_shooter_touchscreen_game_right_in_the_chat_interface_in_the_game_you_shoot_happy_emojis_at_sad_monsters_to_make_them_happy_by_the_way_the_ridiculous_idea_for_a_game_is_claudes/)。这个游戏的内容是向悲伤的怪物发射快乐表情符号使它们变开心，这完全是 Claude 自己的主意。这被视为一个突破性时刻，意味着 AI 现已能与初级人类游戏开发者竞争。用户非常欣赏 Claude 这种可爱且充满希望的创意。

- **模型性能与基准测试**：根据最近发布的结果，[Claude 3.5 Sonnet 在 MMLU-Pro 等多项基准测试中位居榜首](https://www.reddit.com/r/ChatGPT/comments/1dmd0km/claude_35_sonnet_take_the_top_spot_on_mmlupro_plus_new_sonnet_35_benchmarks_that_recently_came_out/)。

- **通过知识集成改进聊天机器人**：在 /r/singularity 中，一位用户对[大型 AI 公司尚未将聊天机器人连接到维基百科等知识库或 WolframAlpha 等工具](https://www.reddit.com/r/singularity/comments/1dmnflb/chatgpt_can_be_much_better_easily_imo/)以提高事实、数学、物理等方面的准确性感到惊讶。他们认为底层技术已经存在，只是需要集成，尽管语言模型可能仍面临一些根本性的局限。

---

# AI Discord 回顾

> **特别说明**：正如我们在发布前沿模型时所做的那样，我们展示了在 Claude 3.5、Claude 3 和 GPT4o 上运行相同提示词（Prompts）的输出差异。

## Claude 3 Sonnet

**1. LLM 性能基准测试与进展**

- Meta 的 [Llama 3 模型](https://lmsys.org/blog/2024-05-08-llama3/)在 ChatbotArena 等排行榜上迅速攀升至榜首，表现优于 GPT-4-Turbo 和 Claude 3 Opus，正如[此讨论](https://discord.com/channels/879548962464493619/879548962464493622/1253795430014259370)中所述。
- 讨论了 [IBM 的 Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 和 [DeepSeek 的 DeepSeek-V2 236B 模型](https://huggingface.co/deepseek-ai/DeepSeek-V2)等新模型，后者在[某些基准测试中表现优于 GPT-4](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296)。
- 然而，人们对[某些基准测试持怀疑态度](https://discord.com/channels/729741769192767510/755950983669874798/1254766616986783835)，并呼吁可靠的来源来设定现实的评估标准。

**2. 高效 LLM 训练与推理技术**

- [DeepSpeed 的 ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 被提及，[有望将 GPU 上大型模型训练的通信开销降低 4 倍](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)。
- 讨论了 [vAttention 系统](https://arxiv.org/abs/2405.04437)，用于在不使用 PagedAttention 的情况下[动态管理 KV-cache 以实现高效推理](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)。
- [QServe 的 W4A8KV4 量化](https://arxiv.org/abs/2405.04532)被重点介绍，作为一种[提升 GPU 上云端 LLM 服务性能](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)的技术。
- 提到了 [Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/) 等技术，用于[探索并行 Token 解码以降低推理延迟](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)。

**3. 开源 AI 框架与社区努力**

- 讨论了 [Axolotl 项目](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)，旨在[支持多种数据集格式](https://discord.com/channels/1104757954588196865/1104757955204743201/1253827044463083582)，用于指令微调（Instruction Tuning）和 LLM 预训练。
- [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 被提及为吴恩达（Andrew Ng）关于[构建 Agentic RAG 系统的新课程](https://discord.com/channels/1059199217496772688/1187460979064324127/1254141690714001508)提供支持。
- 介绍了开源的 [RefuelLLM-2 模型](https://huggingface.co/refuelai/Llama-3-Refueled)，声称是[处理“枯燥数据任务”的最佳选择](https://discord.com/channels/1179035537009545276/1179035537529643040/1253788511379787947)。
- 关于 [Modular 的 Mojo](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo) 的讨论强调了其在 Python 集成和 AI 扩展（如 *bfloat16*）方面的潜力，详见[此处](https://discord.com/channels/1087530497313357884/1098713601386233997/1253826170680643594)。

**4. 多模态 AI 与生成式建模**

- 提到了 [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 和 [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) 模型，它们分别专注于聊天交互和编程能力，详见[此处讨论](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014)。
- 重点介绍了 [Phi 3 模型](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)，根据[此 Reddit 链接](https://discord.com/channels/1110598183144399058/1110598183144399061/1253787640977817662)，该模型通过 WebGPU 将强大的 AI 聊天机器人引入浏览器。
- 提到结合 Pixart Sigma、SDXL 和 PAG 旨在实现 DALLE-3 级别的输出，并有可能通过微调（Fine-tuning）进一步优化，详见[此讨论](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014)。
- [此对话](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014)中还提到了开源的 [IC-Light 项目](https://github.com/lllyasviel/IC-Light)，专注于改进图像重照明（Relighting）技术。

## Claude 3.5 Sonnet

1.  **LLM 性能与基准测试（Benchmarking）争论**：
  
  - 像 [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) 和 [Hermes 2 Theta Llama-3 70B](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k) 这样的新模型因其性能而备受关注。然而，社区对 AI 基准测试和排行榜的怀疑日益增加，呼吁建立更具公信力的评估方法。
  - 多个 Discord 社区的讨论涉及了比较 GPT-4、Claude 以及新型开源替代方案的挑战，强调了实际性能与基准测试分数之间的复杂性。

2.  **优化 LLM 训练与推理**：
  
  - 社区正在分享提高 LLM 效率的技术，例如 [量化方法（quantization methods）](https://github.com/ggerganov/llama.cpp/pull/7931) 以及针对 [AMD GPUs](https://github.com/ROCm/xformers) 等特定硬件的优化。
  - 降低计算成本引起了极大兴趣，讨论范围从 [VRAM 优化](https://huggingface.co/mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF/tree/main) 到用于更高效推理的新颖架构。

3.  **开源 AI 开发与协作**：
  
  - 像 [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 和 [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 这样的项目正在促进社区驱动的 AI 工具和框架开发。
  - 模型和工具的开源化已成趋势，如 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 和 [Llamafile](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7) 所示，这鼓励了更多人参与到 AI 开发中。

4.  **多模态 AI 与创意应用**：
  
  - 各个 Discord 社区的讨论突显了对处理文本、图像以及潜在视频的多模态模型日益增长的兴趣，[Stable Artisan](https://bit.ly/4aiVy6C) 等项目正将这些能力带给更广泛的受众。
  - 目前正在进行结合不同模型和技术以达到 DALL-E 3 级别输出的实验，展示了社区驱动的推进生成式 AI 能力的方法。

5.  **AI 集成与实际应用**：
  
  - 多个社区正在探索将 AI 集成到日常工具中的方法，从 [基于浏览器的模型](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) 到用于媒体创作的 Discord 机器人。
  - 人们越来越关注如何让 AI 在特定任务中更易于获取且更有用，这体现在各个 Discord 频道关于代码生成、数据分析和创意应用的讨论中。

## Claude 3 Opus

1.  **LLM 性能与基准测试 (Benchmarking)**：
  
  - 关于 [Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)、[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) 和 [Claude 3.5 Sonnet](https://www.anthropic.com/index/claude-3-5-sonnet) 等模型在各种基准测试和排行榜上表现的讨论。
  - 对某些基准测试（如 [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)）的质疑，并呼吁建立更具公信力的评估标准。

2.  **优化 LLM 训练与推理**：
  
  - 高效训练技术，如 [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) 和 [Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)，以及通过 [vAttention](https://arxiv.org/abs/2405.04437) 和 [QServe](https://arxiv.org/abs/2405.04532) 实现的推理优化。
  - 关于量化方法（如 [W4A8KV4](https://arxiv.org/abs/2405.04532)）及其对模型性能和资源需求影响的讨论。

3.  **开源 AI 框架与协作**：
  
  - 涉及 [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)、[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 和 [Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo) 等开源框架的更新与协作。
  - 推出新的开源模型，例如 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 和 [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)。

4.  **多模态 AI 与生成模型**：
  
  - 多模态 AI 的进展，包括 [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) 和 [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) 等模型。
  - 生成式建模的创新，例如用于浏览器端聊天机器人的 [Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)，以及结合多种技术实现 [DALLE-3](https://openai.com/dall-e-3/)\-级别的输出。
  - 图像重光照（Image Relighting）领域的开源努力，如 [IC-Light](https://github.com/lllyasviel/IC-Light) 项目。

5.  **AI 伦理、法律与问责**：
  
  - 关于 AI 生成内容伦理影响的讨论，如 [Perplexity AI 涉嫌剽窃](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/) 事件。
  - 对 AI 模型发表不准确或诽谤性言论相关法律风险的担忧，如 [Perplexity AI 案例](http://archive.today/GNgAe) 所强调的。
  - 关于科技公司使用开放数据集的问责制以及 [“AI 数据洗白”](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/) 做法的辩论。

## GPT4T (gpt-4-turbo-2024-04-09)

**1. **AI 硬件在进化，但成本激增**：

- **Command R (34b) Q4_K_S** 等 AI 模型对 VRAM 的需求引发了关于转向 **EXL2**（一种更节省 VRAM 的格式）的讨论。[NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/) 因高昂成本仍难以触及。

**2. **优化成为 AI 工具的核心**：

- 利用**量化技术**优化模型性能，提到了 ROCm 版本的 [xformers](https://github.com/ROCm/xformers) 和 [flash-attention](https://github.com/ROCm/flash-attention) 以提高效率。在 **Llama-2** 模型中实现的 **PyTorch 增强**带来了显著的性能提升。

**3. **AI 在多方面应用中取得突破**：

- 由 **OpenRouter** 开发的 AI 驱动阅读辅助工具 [Pebble](https://pebble.study/) 展示了 AI 在增强阅读理解方面的新颖用途。**Cohere 的 API** 支持多步工具调用（Multi-step tool use），显著扩大了功能范围，并辅以关于 [多步工具调用](https://docs.cohere.com/docs/multi-step-tool-use) 的详细文档。

**4. **以 AI 为中心的教育与就业正在扩张**：

- **MJCET** 在特伦甘纳邦成立了首个 AWS Cloud Club，旨在向学生普及 AWS 技术，并举办了由 **AWS Community Hero Mr. Faizal Khan** 参加的庆祝活动。通过 [活动链接](https://meetu.ps/e/NgmgX/14DgQ2/i) 报名。此外，Hamel Dan 的 **LLM101n 课程**提供了以 AI 为灵感的教育内容，教授从零开始构建 LLM。

**5. **其他 AI 进展反映了广泛影响**：

- **Intel** 从 AWS 撤退，令 AI 社区对资源分配感到困惑。**Claude Sonnet 3.5** 在编程任务中的表现赢得赞誉，展示了 AI 在技术应用方面的进步。出人意料的是，**Mozilla 的 Llamafile** 进军 Android 领域，承诺提供创新的移动端解决方案。

## GPT4O (gpt-4o-2024-05-13)

### **模型优化与 LLM 创新**

- **DeepSeek 和 Sonnet 3.5 主导基准测试**：**DeepSeek** 模型以其快速的性能和编码能力给社区留下了深刻印象，在某些情况下超越了 **GPT-4** ([DeepSeek 公告](https://chat.deepseek.com/coder))。同样，**Claude 3.5 Sonnet** 在编码任务中表现优于 **GPT-4o**，这已通过 LMSYS 排行榜排名和实际使用得到验证 ([Claude 推文](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413))。
- **ZeRO++ 和 PyTorch 加速 LLM 训练**：**ZeRO++** 将大模型训练中的通信开销降低了 4 倍，而新的 PyTorch 技术将 Llama-2 的推理速度提高了 10 倍，这些技术被封装在 [GPTFast 软件包](https://github.com/MDK8888/GPTFast)中，优化了其在 A100 或 H100 GPU 上的使用 ([ZeRO++ 教程](https://www.deepspeed.ai/tutorials/zeropp/))。

### **开源进展与社区努力**

- **Axolotl 和 Modular 鼓励社区贡献**：Axolotl 宣布集成了 ROCm 分支版本的 [xformers](https://github.com/ROCm/xformers) 以支持 AMD GPU，Modular 用户讨论了为 LLVM 和 CUTLASS 贡献学习材料 ([相关指南](https://pikuma.com/blog/understanding-computer-cache))。
- **Featherless.ai 和 LlamaIndex 扩展功能**：Featherless.ai 是一个以 Serverless 方式运行公共模型的新平台，其发布引起了广泛关注 ([Featherless](https://featherless.ai))。**LlamaIndex** 现在支持通过 StabilityAI 生成图像，增强了其面向 AI 开发者的工具包 ([LlamaIndex-StabilityAI](https://t.co/a7F0gv4tpi))。

### **AI 生产与实际应用**

- **MJCET 的 AWS Cloud Club 启动**：MJCET 的 AWS Cloud Club 正式成立，促进了 AWS 实操培训和职业发展计划 ([AWS 活动](https://meetu.ps/e/NgmgX/14DgQ2/i))。
- **OpenRouter 在实际应用中的使用**：**JojoAI** 因其主动助手能力而受到关注，它利用 DigiCord 等集成，表现优于 ChatGPT 和 Claude 等竞争模型 ([JojoAI 网站](https://www.digicord.site))。

### **运营挑战与支持查询**

- **安装和兼容性问题困扰用户**：在 Windows 上设置 xformers 等库的困难引发了关于兼容性的讨论，建议倾向于使用 Linux 以获得更稳定的运行 ([Unsloth 故障排除](https://github.com/unslothai/unsloth/issues/243))。
- **额度和支持问题**：Hugging Face 和 Predibase 社区的许多成员面临服务额度丢失和账单查询问题，显示出对改进客户支持系统的需求 ([Predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253822951149797488))。

### **新兴技术与未来方向**

- **发布新的 AI 模型和集群**：**AI21 的 Jamba-Instruct**（具有 256K 上下文窗口）和 **NVIDIA 的 Nemotron 4** 突出了在处理大规模企业文档方面的突破 ([Jamba-Instruct](https://openrouter.ai/models/ai21/jamba-instruct), [Nemotron-4](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct))。
- **多模态融合（Multi Fusion）与量化技术**：关于多模态模型中早期融合与后期融合优缺点的讨论，以及量化技术的进步，突出了在降低 AI 模型推理成本和提高效率方面的持续研究 ([Multi Fusion](https://arxiv.org/abs/2406.09406))。

---

# 第 1 部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**虚拟现实中选择 Juggernaut 还是 SD3 Turbo？**：虽然 *Juggernaut Lightning* 因其在非编程创意场景中的写实性而受到青睐，但 *SD3 Turbo* 的讨论热度并不高，这表明模型的选择受到具体语境和目标的影响。

**PyTorch 用户的量子飞跃**：建议优先投资 *PyTorch* 和 *HuggingFace* 等库，而非 *sklearn* 等陈旧工具。使用 *bitsandbytes* 和 4-bit quantization（4位量化）等精度修改技术，可以帮助在受限的硬件上加载模型。

**元模型合并与共情演化**：*Open Empathic* 项目正在通过 YouTube 贡献的电影场景分类进行扩展。同时，关于 *UltraChat* 和 *Mistral-Yarn* 的合并策略引发了辩论，其中 *mergekit* 和 *frankenMoE finetuning* 被提及为改进 AI 模型的重要技术。

**增强型软件与服务**：一系列贡献相继涌现，包括 *Mistroll 7B v2.2* 的发布、用于 *Stable Diffusion* 的简单微调工具、使用 *PyQt* 和 *Whisper* 开发的媒体转文本 GUI，以及用于 Serverless 模型调用的新 AI 平台 [Featherless.ai](https://featherless.ai)。

**追求 AI 推理的启示**：揭示 LLM 推理最新进展的计划正在酝酿中，重点考察 *Understanding the Current State of Reasoning with LLMs* ([arXiv 链接](https://arxiv.org/abs/2206.07682)) 以及 [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning) 及其同名[备用仓库链接](https://github.com/luban-agi/Awesome-LLM-reasoning)等资源。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI 预览引发热议**：在视频拍摄公告发布后，一名成员因期待 **Unsloth AI** 的发布而分享了一段[临时录像](https://discord.com/channels/1179035537009545276/1179035537529643040/1253726564500111381)，以等待早期访问权限。为了提高清晰度，建议将缩略图从 "csv -> unsloth + ollama" 修改为 "csv -> unsloth -> ollama"，并为新手添加解释性文本。
- **大显存引发大讨论**：一段 [YouTube 视频](https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&t=679)展示了 **Phison** 的 PCIe-NVMe 卡作为一种惊人的 1Tb VRAM 解决方案，引发了关于其对性能影响的讨论。同时，Fimbulvntr 成功将 **Llama-3-70b** 扩展到 64k 上下文，以及关于 VRAM 扩展的辩论，凸显了对大模型容量的持续探索。
- **LLM 的升级与情感**：**Ollama** 更新定于周一或周二发布，承诺支持 CSV 文件。同时，Sebastien 开发的 **emotional llama model**（旨在促进 AI 对情感的更好理解）已在 [Ollama](https://ollama.com/sebdg/emotional_llama) 和 [YouTube](https://www.youtube.com/watch?v=ZJKglSWgD0w) 上线。
- **解决安装与兼容性问题**：从在 Windows 上通过 conda 为 Unsloth 安装 xformers 的困难，到确保 Google Colab 笔记本中初始设置单元格的正确执行，成员们交流了克服软件挑战的技巧。GPU Cloud (NGC) 容器设置讨论以及 CUDA 和 PyTorch 版本限制问题，也通过使用不同的容器和共享 Dockerfile 配置得到了解决。
- **思考合作伙伴关系与 AI 集成**：一篇题为《[Apple 与 Meta 合作伙伴关系：iPhone 生成式 AI 的未来](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/)》的博客引起了社区的兴趣，讨论集中在生成式 AI 在移动设备中的战略影响和潜在集成挑战。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **机器人警示**：有人分享了一个用于集成 Gemini 和 StabilityAI 服务的 Discord 机器人，但成员们对该链接的安全性和上下文表示担忧。
- **Civitai 因许可担忧下架 SD3**：Civitai 移除 SD3 资源引发了激烈讨论，这表明此举是为了预先规避法律问题。
- **低配运行 Stable Diffusion**：讨论了在低规格 GPU 上运行 Stable Diffusion 的技术（如利用 *automatic1111*），并权衡了旧款 GPU 与 **RTX 4080** 等新型号的效率。
- **训练难题与技巧**：社区成员寻求训练模型以及克服 **VRAM** 限制和有问题的元数据等错误的建议，一些人建议使用 **ComfyUI** 和 **OneTrainer** 等专门工具进行更高级的管理。
- **模型兼容性困惑**：讨论强调了 **SD 1.5** 和 **SDXL** 等模型与 **ControlNet** 等插件保持一致的必要性；类型不匹配会导致性能下降和错误。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUTLASS 与 CUDA 协作号召**：受一段分享的 [Tensor Cores YouTube 演讲](https://youtu.be/hQ9GPnV0-50?feature=shared) 启发，用户表达了组建 **CUTLASS** 工作组的兴趣。此外，通过分享 [缓存功能入门指南](https://pikuma.com/blog/understanding-computer-cache)，加深了对 **CPU** 缓存的见解，强调了其对程序员的重要性。
- **浮点数与精度风险**：**FP8** 转换中的精度损失引起了关注，促使大家分享了用于理解 **IEEE** 舍入约定以及使用 **tensor scaling** 抵消损失的资源。对于探索 **Quantization** 的用户，推荐了一系列论文和教育内容，包括 [Quantization 详解](https://youtu.be/0VdNflU08yA) 和 [高级 Quantization](https://youtu.be/1u9xUK3G4VM)。
- **INT4 与 QLoRA 爱好者讨论**：在对比 **INT4 LoRA 微调**与 **QLoRA** 的讨论中，有人指出 **QLoRA** 包含的 **CUDA dequant kernel (axis=0)** 保持了质量和速度，特别是与在大序列上使用 **tinnygemm** 的方案相比。
- **神经网络优化**：**Bitnet tensors** 与 **AffineQuantizedTensor** 的集成引发了辩论，考虑了用于指定打包维度的特殊布局。为了协助调试 **Bitnet tensor** 问题，[CoffeeVampire3 的 GitHub](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py) 和 [PyTorch ao 库教程](https://github.com/pytorch/ao/issues/426) 被列为首选资源。
- **扩展系统稳定性的策略**：**多节点设置优化**和集成 **FP8 matmuls** 的策略是对话的核心，旨在解决性能挑战和训练稳定性，特别是 **H100 GPU** 与 **A100** 相比显示出的问题。此外，还为即将在 **Lambda 集群**上进行的大规模语言模型训练做了准备，重点关注效率和稳定性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**显存紧缺与高昂价格**：工程师们强调了在处理像 **Command R (34b) Q4_K_S** 这样的大型模型时的 VRAM 瓶颈，建议使用 **EXL2** 作为更节省 VRAM 的格式。对于重负载的 AI 工作，以海量内存著称的 [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/) 对大多数人来说在财务上仍然遥不可及，暗示需要数千美元的投资。

**LLM 推理的量子飞跃**：用户对 **Hermes 2 Theta Llama-3 70B** 模型印象深刻，该模型以其显著的 token 上下文限制和创意优势而闻名。关于 LLM 缺乏时间意识的讨论促使人们提到了 **Hathor Fractionate-L3-8B**，因为它在输出 tensors 和 embeddings 保持非量化时表现出色。

**酷炫装备与热门芯片**：在硬件战场上，使用 **P40 GPUs** 运行 **Codestral** 显示出功率利用率激增至 12 tokens/second。与此同时，**iPad Pro** 的 16GB RAM 是否能处理 AI 模型引发了争论，针对 4000 系列 GPU 缺乏 NVlink 的情况，有人提出了使用 **DX 或 Vulkan** 实现多 GPU 支持的构想。

**补丁与插件**：**LLaMa library** 因模型预期的 tensor 数量不匹配导致的错误而令用户困扰，而 **deepseekV2** 面临加载困难，可能通过更新到 **V0.2.25** 来修复。人们对一种假设的全能模型运行器充满热情，该运行器可以处理包括 text-to-speech 和 text-to-image 在内的各种 Huggingface 模型。

**模型工程与谜团**：名字奇特的 **Llama 3 CursedStock V1.8-8B** 模型因其独特的性能（尤其是在创意内容生成方面）激起了人们的好奇心。还有关于 **Multi-model sequence map** 的讨论，它允许数据在多个模型之间流动，而最新的量化 **Qwen2 500M** 模型因其能在性能较低的设备（甚至是 **Raspberry Pi**）上运行而引起轰动。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Siri 与 ChatGPT 的奇特组合**：用户对 Siri 与 ChatGPT 的集成感到困惑，共识是 ChatGPT 充当了 Siri 的增强功能，而非核心集成。Elon Musk 的批评性言论进一步引发了关于该话题的讨论。
- **Claude 在编程上完胜 GPT-4o**：**Claude 3.5 Sonnet** 因在编程任务中优于 **GPT-4o** 的表现而受到赞誉，用户强调了 Claude 在 GPT-4o 碰壁的领域取得了成功。有效性是通过实际使用和 LMSYS 排行榜上的位置来衡量的，而不仅仅是基准测试分数。
- **持久的 LLM 个人助理之梦**：人们对定制和维护语言模型（如 **Sonnet 3.5 或 Gemini 1.5 Pro**）以作为基于个人文档训练的个性化工作机器人的可能性表示热忱，引发了关于 LLM 长期和专业化应用的讨论。
- **GPT-4o 的上下文窗口困扰**：用户在 **GPT-4o** 遵循复杂 prompt 指令及处理长文档的能力限制方面感到吃力。建议使用 Gemini 和 Claude 等替代方案，以便在更大的 token 窗口中获得更好的表现。
- **DALL-E 与 Midjourney 的艺术对决**：服务器上正在展开一场关于 DALL-E 3 和 Midjourney 生成 AI 图像能力的辩论，特别是在绘画风格的艺术作品领域，一些人表现出对前者独特艺术风格的偏好。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 陷入抄袭风波**：[Wired 报道](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/)了 Perplexity AI 因抓取网站而涉嫌违反政策的行为，其聊天机器人将一起罪行错误地归咎于一名警察，引发了关于 AI 摘要不准确所带来的法律影响的辩论。
- **对 Claude 3.5 Sonnet 的反应褒贬不一**：据 [Forbes](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/) 报道，**Claude 3.5 Sonnet** 的发布既因其强大的能力获得掌声，也因其表现得过于谨慎而令人沮丧；同时，用户在 Pro 搜索结果中遇到的不一致性导致了对 Perplexity 服务的不满。
- **关于 Apple 和 Boeing 困境的独家报道**：Apple 的 AI 在欧洲面临限制，而 Boeing 的 Starliner 遭遇重大挑战，这些信息在 Perplexity 上传播，并附有指向相关问题的文章直链（[Apple Intelligence Isn't](https://www.perplexity.ai/page/Apple-Intelligence-Isnt-KJfiVRPEQMmkim0gv7Xh7w), [Boeing’s Starliner Stuck](https://www.perplexity.ai/page/Boeings-Starliner-Stuck-lIlR4mleQUK1Q0kahpVwRQ)）。
- **Perplexity API 的难题**：Perplexity API 社区讨论了 **LLama-3-70B** 在处理长 Token 序列时可能触发的审核机制或技术错误，并提出了关于通过 API 限制链接摘要和引用中时间过滤的问题，详见 [API 参考文档](https://docs.perplexity.ai/reference/post_chat_completions)。
- **社区融合以提升参与度**：OpenAI 社区的一条消息强调了对可分享线程的需求，以促进更好的协作；同时，由 Perplexity AI 制作的 [YouTube 视频](https://www.youtube.com/embed/xUsxGDrwzls)预告了诸如 **Starliner 困境和 OpenAI 最新动态**等多样化的教育性话题。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

### **数据集去重效率提升**：**Rensa** 凭借 Rust 的 FxHash、LSH 索引和即时排列技术，在数据集去重方面实现了比 **datasketch** 快 [2.5-3 倍的速度提升](https://github.com/beowolx/rensa)。

### **模型越狱曝光**：一篇[《金融时报》文章](https://on.ft.com/45ByjEj)强调了黑客通过“越狱” AI 模型来揭示缺陷，同时 GitHub 上的贡献者分享了一个 [“smol q* 实现”](https://github.com/EveryOneIsGross/ganymede)以及像 [llama.ttf](https://fuglede.github.io/llama.ttf/) 这样伪装成字体文件的创新 LLM 推理引擎项目。

### **关于模型参数的热烈辩论**：在 *[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1254053586997088310)* 频道中，讨论范围从 [TinyStories-656K](https://huggingface.co/raincandy-u/TinyStories-656K) 令人惊讶的故事生成能力，到关于 70B+ 参数模型在通用性能上大幅飞跃的断言。

### **数据集合成与分类增强**：成员们分享了一个用于协作跟踪数据集的 [Google Sheet](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAz_CREYkzsdG2NSf-KY/edit#gid=0)，探索了使用 Hermes RAG 格式的改进，并深入研究了用于科学和教学目的的 [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF?row=0) 和 [ft-instruction-synthesizer-collection](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main) 等数据集。

### **AI 安全模型审查与课程**：*#[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296)* 频道内容丰富，从 **Gemini** 和 **OpenAI** 具备脱敏能力的安全性模型，到 Karpathy 的 [LLM101n 课程](https://github.com/karpathy/LLM101n)发布，鼓励工程师构建一个讲故事的 LLM。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SLURM 与 Jupyter 的连接问题**：工程师在通过 Jupyter Notebook 连接 SLURM 管理的节点时遇到问题，称错误可能源于 SLURM 的限制。一位用户反映，即使 GPU 规格配置正确，在训练开始前控制台也会出现 'kill' 消息。
- **PyTorch 提升 Llama-2 性能**：PyTorch 团队实施了相关技术，将 Llama-2 的推理速度提高了多达十倍；这些增强功能封装在 [GPTFast package](https://github.com/MDK8888/GPTFast) 中，需要 A100 或 H100 GPU。
- **AI 模型的伦理与共享**：关于在官方渠道之外分发 Mistral 等专有 AI 模型的伦理和实际考量的严肃对话，强调了对法律问题的担忧以及透明度的重要性。
- **理解 AI 模型变体**：用户讨论了确定 AI 模型是 GPT-4 还是其他变体的方法，包括检查知识截止日期、延迟差异和网络流量分析。
- **LingOly 挑战赛介绍**：新的 LingOly 基准测试旨在评估 LLM 在涉及语言谜题的高级推理方面的表现。在展示的一千多个问题中，顶级模型的准确率低于 50%，这表明对当前架构构成了严峻挑战。
- **ARDiT 的文本转语音创新**：一期 [播客节目](https://youtu.be/lj2y5hE04XI?t=4585) 探讨了使用 SAEs 进行模型编辑，灵感来自 [MEMIT paper](https://arxiv.org/pdf/2210.07229.pdf) 及其 [源代码](https://github.com/kmeng01/memit) 中详述的方法，表明该技术具有广泛的应用前景。
- **思考多模态架构的最优性**：对话探讨了像 Chameleon 这样的早期融合模型在多模态任务中是否优于后期融合方法。焦点在于早期融合在图像 Token 化过程中通用性与视觉敏锐度损失之间的权衡。
- **Intel 撤出 AWS 实例**：Intel 正在停止 gpt-neox 开发团队利用的 AWS 实例，引发了关于计算资源的高性价比方案或替代性手动解决方案的讨论。
- **执行错误：NCCL 后端**：工程师报告在尝试使用 gpt-neox 在 A100 GPU 上训练模型时，持续遇到 NCCL 后端挑战，这一问题在各种 NCCL 和 CUDA 版本中（无论是否使用 Docker）都一致存在。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Character.AI 攻克大规模推理难题**：Character.AI 的 Noam Shazeer 通过 [推理过程的优化](https://research.character.ai/optimizing-inference/) 阐明了对 AGI 的追求，强调了他们每秒处理超过 20,000 条推理查询的能力。
- **收购新闻：OpenAI 欢迎 Rockset**：[OpenAI 已收购 Rockset](https://x.com/deedydas/status/1804185430897889427)，这是一家擅长混合搜索架构的公司，拥有向量 (FAISS) 和关键词搜索等解决方案，加强了 OpenAI 的 RAG 套件。
- **Karpathy 助力 AI 教育**：Andrej Karpathy 播下了雄心勃勃的新课程 "LLM101n" 的种子，该课程将深入探讨从零开始构建类似 ChatGPT 的模型，继承了传奇课程 CS231n 的衣钵。
- **LangChain 澄清资金使用情况**：Harrison Chase 针对 LangChain 将风险投资用于产品开发而非推广的质疑做出了回应，详情见 [推文](https://x.com/hwchase17/status/1804166140773691837)。
- **Murati 预告 GPT 的下一次飞跃**：OpenAI 的 Mira Murati 向爱好者们透露了一个时间表，暗示可能在约 1.5 年内发布下一个 GPT 模型，同时讨论了 AI 给创意和生产性行业带来的巨大变化，详见 [YouTube 视频](https://www.youtube.com/watch?v=yUoj9B8OpR8)。
- **Latent Space 关于招聘 AI 专业人士的见解**：新的一期 "Latent Space Podcast" 播客分解了招聘 AI 工程师的艺术与科学，引导听众了解招聘流程和防御性 AI 工程策略，其中包含来自 @james_elicit 和 @*adamwiggins* 的见解，可在 [此页面](https://x.com/latentspacepod/status/1804269727482810386) 查看，并在 Hacker News 上引起热议。
- **探索 YAML 新领域**：对话展示了开发一种基于 YAML 的 DSL 用于 Twitter 管理以增强帖子分析，并提到了 Zoho Social 的全面功能；对于类似的尝试，Anthropic 建议采用 XML 标签，一个 [GitHub 仓库](https://github.com/go-go-golems/go-emrichen) 展示了在 Go 语言中使用 LLM 成功设计 YAML 模板语言的案例。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **LLVM 的身价**：分享了一篇估算 LLVM 项目成本的文章，详细说明了 1.2k 名开发者产出了 6.9M 行代码，估算成本为 5.3 亿美元。克隆和检出 LLVM 是了解其开发成本的一部分。
- **安装故障与求助**：强调了在 22.04 上安装 Mojo 的问题，提到了所有 devrel-extras 测试均失败；这一棘手情况导致排查工作暂停。另外，Mojo 开发过程中的 segmentation faults 令人沮丧，促使一名用户悬赏 10 美元的 OpenAI API key 以寻求解决其关键问题的帮助。
- **关于 Caching 和 Prefetching 性能的讨论**：深入探讨了 caching 和 prefetching，重点关注正确应用及潜在陷阱。分享的见解包括：如果 prefetching 利用不当，可能会对性能产生负面影响；建议使用 `vtune` 等分析工具针对 Intel 缓存进行优化，尽管 Mojo 目前不支持编译时缓存大小获取。
- **改进提案与 Nightly Mojo 版本**：记录了关于 Mojo 文档改进的建议以及在 Mojo 中受控隐式转换的提案。新的 nightly Mojo 编译器发布以及 MAX repo 更新的消息引发了关于开发工作流和生产力的讨论。
- **数据标注与集成见解**：一个新的数据标注平台倡议收到了关于使用 [Haystack](https://haystack.deepset.ai/) 等工具进行自动化的痛点和成功经验的反馈。ERP 集成的潜力（由手动数据录入挑战和 PDF 处理引发）也是一个焦点，表明了简化数据管理工作流的趋势。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Weta 与 Stability AI 的新变局**：**Weta Digital** 和 **Stability AI** 领导层变动的消息引发了一波讨论，重点关注这些大洗牌的影响，并质疑任命背后的动机。一些讨论指向了 **Sean Parker** 并分享了相关文章，链接了一篇 [路透社关于 Stability AI 的文章](https://www.reuters.com/technology/artificial-intelligence/stability-ai-appoints-new-ceo-information-reports-2024-06-21/)。
- **Llama 3 蓄势待发**：对于 **Llama 3** 的硬件规格展现出的惊人性能，大家感到非常兴奋，其表现可能超越 **GPT-4O** 和 **Claude 3** 等竞争对手模型。参与者分享了在高级配置下“每秒 1 到 2 个 token”的预计吞吐量。
- **Glaze 与 Nightshade 的保护悖论**：一场关于 **Glaze** 和 **Nightshade** 等程序保护艺术家权利能力有限的严肃对话展开。怀疑者指出，后来者总能找到绕过此类保护的方法，从而给艺术家提供了潜在的虚假希望。
- **多模态模型——重复的突破？**：该社区研究了一篇关于多模态模型的新论文，提出了所谓的进步是否有意义的疑问。该论文提倡在多种模态上进行训练以增强通用性，但参与者批评这种反复出现的“突破”叙事缺乏实质性的新意。
- **测试极限：Diffusion Models 的承诺与局限**：lucidrains 分享的一个 **GitHub** 仓库对 diffusion models 进行了深入探讨，讨论了 **EMA (Exponential Moving Average)** 模型更新（[GitHub 上的 Diffusion Models](https://github.com/lucidrains/ema-pytorch)）及其在图像修复中的应用，尽管有证据表明像 Glaze 这样的保护措施会被持续绕过。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **新成员欢迎**：新成员加入了以 Cohere 为中心的 Discord，在共享见解和[工具使用文档](https://docs.cohere.com/docs/tool-use)的引导下，学习如何将 Cohere 模型连接到外部应用程序。
- **对 BitNet 实用性的质疑**：在关于 BitNet 未来的辩论中，有人指出它需要从头开始训练，且未针对现有硬件进行优化，导致 Mr. Dragonfox 对其商业实用性表示担忧。
- **Cohere 的能力与贡献**：在 Microsoft 的 [AutoGen framework](https://github.com/microsoft/autogen/pull/3004/files) 集成了 Cohere 客户端后，社区内呼吁 Cohere 团队在项目推进中提供进一步支持。
- **AI 爱好者渴望多语言扩展**：确认了 Cohere 模型理解并响应多种语言（包括中文）的能力，并引导感兴趣的各方查阅[文档](https://docs.cohere.com/docs/tool-use)和 GitHub [notebook 示例](https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Tool_Use.ipynb)以了解更多信息。
- **开发者办公时间与多步创新**：Cohere 宣布了即将举行的开发者办公时间，重点介绍 **Command R 家族的工具使用能力**，并提供关于[多步工具使用](https://docs.cohere.com/docs/multi-step-tool-use)的资源，以利用模型执行复杂的任务序列。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **关于上下文和 Token 的混淆**：用户报告了关于 Agent 中 **max tokens** 和上下文窗口集成的混淆，特别是 LangChain 未遵循 **Pydantic** 模型的验证。指出上下文窗口或最大 Token 计数应同时包含输入和生成的 Token。
- **LangChain 学习与实现查询**：关于 **LangChain** 学习曲线的讨论非常热烈，成员们分享了诸如 [Grecil 的个人旅程](https://corrective-rag.streamlit.app)等包含教程和文档的资源。同时，关于 **ChatOpenAI** 与 **Huggingface** 模型的辩论突显了在各种场景下的性能差异和适配问题。
- **使用 LangChain 增强 PDF 问答**：分享了使用 LangChain 从 PDF 生成问答对的详细指南，并参考了 GitHub 上的 [#17008](https://github.com/langchain-ai/langchain/issues/17008) 等问题以获取进一步指导。还讨论了使用 **Llama2** 作为 LLM 的调整，强调了自定义 `QAGenerationChain`。
- **从零到 RAG 英雄**：成员们展示了他们为金融文档构建无代码 **RAG 工作流**的经验，并分享了一篇详细介绍该过程的[文章](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888)。讨论还集中在一个自定义的 [Corrective RAG app](https://corrective-rag.streamlit.app) 和 **Edimate**（一个 AI 驱动的视频创作工具，在此[演示](https://x.com/dswharshit/status/1805203856088834428)），这标志着在线学习的未来。
- **AI 框架评估视频**：针对正在评估用于应用程序集成（包括 GPT-4o 等模型）的 AI 框架的工程师，分享了一个 [YouTube 视频](https://youtu.be/uG0cs8AlnHw)，敦促开发者考虑关于特定应用中 AI 框架的必要性和选择的关键问题。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Jamba Instruct 拥有巨大的上下文窗口**：[AI21 的 Jamba-Instruct 模型](https://openrouter.ai/models/ai21/jamba-instruct)已推出，展示了高达 **256K 的上下文窗口**，非常适合处理企业场景中的海量文档。
- **Nemotron 4 凭借合成数据生成引起关注**：[NVIDIA 发布的 Nemotron-4-340B-Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) 专注于通过其新的聊天模型为英语应用生成合成数据。
- **JojoAI 升级为主动型助手**：JojoAI 通过成为一个可以设置提醒的主动型助手而脱颖而出，它采用了 DigiCord 集成，使其有别于 ChatGPT 或 Claude 等竞争对手。可以在 [JojoAI 网站](https://www.digicord.site)上进行体验。
- **Pebble 开创性的阅读辅助工具**：由 OpenRouter 提供支持，并结合了 **Mistral 8x7b** 和 **Gemini** 的 **Pebble** 工具正式亮相，它为增强网页内容的阅读理解和记忆提供了资源。感谢 OpenRouter 团队的支持，详见 [Pebble](https://pebble.study/)。
- **技术社区应对环境和技术问题**：讨论指出，使用像 Nemotron 340b 这样的模型会带来环境足迹方面的担忧，因此推荐使用更小的模型以实现高效和环保。社区还处理了一些实际事务，例如解决 Claude 自托管端点消失的问题，赞扬 Sonnet 3.5 的编程能力，处理 OpenRouter 的速率限制（rate limits），以及就处理暴露的 API 密钥的最佳实践提供建议。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **本地 LLM 进入 OS 模式**：OpenInterpreter 社区一直在讨论通过命令 `interpreter --local --os` 在 OS 模式下使用**本地 LLM**，但对其性能水平存在担忧。
- **桌面端惊喜与 GitHub 荣耀**：OpenInterpreter 团队正在推广即将推出的**桌面应用程序**，它将提供与 GitHub 版本不同的独特体验，并鼓励用户加入[等待名单](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com)。与此同时，该项目庆祝获得了 **50,000 个 GitHub stars**，并预示即将发布重大公告。
- **模型基准测试讨论**：**Codestral 和 Deepseek 模型**引起了关注，Codestral 超过了内部基准测试，而 Deepseek 的快速表现给用户留下了深刻印象。关于未来优化 `interpreter --deepseek` 命令的呼声很高。
- **跨平台 Poetry 表现**：使用 **Poetry** 而非 `requirements.txt` 进行依赖管理一直是一个有争议的话题，一些工程师指出了它在不同操作系统上的缺点，并主张使用 **conda** 等替代方案。
- **社区赞誉与担忧**：虽然社区对支持（特别是对初学者的支持）充满热情和感激，但对 **01 设备**的发货延迟也存在不满，这凸显了社区情绪与产品交付预期之间的平衡。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**指令合成（Instruction Synthesizing）大获全胜**：新分享的 [Hugging Face 仓库](https://huggingface.co/instruction-pretrain/instruction-synthesizer) 强调了 **Instruction Pre-Training** 的潜力，提供了涵盖 40 多个任务的 2 亿个合成配对，为寻求在监督多任务预训练中突破瓶颈的 AI 从业者提供了一种稳健的多任务学习方法。

**将 DeBERTa 与 Flash 结合？**：关于将 **DeBERTa** 与 **Flash Attention 2** 结合的可能性引起了好奇，向对新颖模型架构协同效应感兴趣的 AI 工程师提出了潜在实现方案的问题。

**修复与权宜之计**：从使用移动设备解决 **Maven 课程平台** 空白页问题，到解决 **braintrust** 内内核重启后的权限错误，实际的故障排除仍然是社区讨论的主要内容。

**额度风波持续**：关于 Huggingface 和 Predibase 等平台上服务额度丢失的持续报告引发了成员间的支持和对各自计费支持部门的转介。其中包括一条提示：**Predibase 额度在 30 天后过期**，建议工程师密切关注过期日期以最大化额度使用。

**训练错误与过拟合疑问**：运行 **Axolotl 训练命令** ([Modal FTJ](https://modal.com/ftj)) 时的错误以及对 **LORA 过拟合**（“训练损失显著低于验证损失”）的担忧是主要的痛点，展示了 AI 工程师对警惕模型监控实践的需求。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LightningAI 与 LlamaIndex 联手**：LightningAI 的 RAG 模板为多文档 Agentic RAGs 提供了 [简易设置](https://t.co/2NLH7zuZS6)，提升了 AI 开发效率。此外，LlamaIndex 与 [StabilityAI](https://t.co/a7F0gv4tpi) 的集成现在支持图像生成，扩展了 AI 开发者的能力。
- **利用 LlamaIndex 定制复杂性**：使用 LlamaIndex 进行开发的开发者可以使用有向无环图（DAGs）定制 text-to-SQL 流水线，如本 [功能概述](https://t.co/fiS0kfj8rk) 中所述。同时，为了更好的财务分析，可以利用 Hanane Dupouy 的 [教程幻灯片](https://t.co/lHsThk9IOU) 使用 CRAG 技术来提高检索质量。
- **使用 Mlflow 微调 RAGs**：为了提高 RAGs 的回答准确性，将 LlamaIndex 与 [Mlflow](https://t.co/fo8XxMTO93) 集成提供了一种系统化的方法来管理关键参数和评估方法。
- **LlamaIndex 中的深度查询格式化与并行执行**：成员们讨论了 LlamaIndex 的查询响应模式，如 **Refine** 和 **Accumulate**，以及利用 **OLLAMA_NUM_PARALLEL** 进行并发模型执行；文档解析和 Embedding 不匹配也是技术建议的主题。
- **通过 MLflow 和 LLMs 简化 ML 工作流**：Ankush K Singal 的一篇 [Medium 文章](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437) 强调了通过 LlamaIndex 实际集成 MLflow 和 LLMs 以简化 ML 工作流。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 与 LLAMA 参数大对决**：来自 Meta 的消息来源指出，**Gemini 1.5 Pro** 的参数量少于 **LLAMA 3 70B**，引发了关于 MoE 架构对推理过程中参数计数影响的讨论。
- **GPT-4 的秘诀还是蒸馏后的力量**：社区讨论了 **GPT-4T/o** 是早期融合模型还是大型前代模型的蒸馏版本，显示出对其基础架构理解的分歧。
- **多模态训练困境**：成员们强调了后训练多模态模型的困难，理由是跨不同数据模态转移知识的挑战。这些挣扎表明，对于增强原生多模态系统的复杂性，大家已达成普遍共识。
- **探究 Nous 与索尼的骚动**：Nous Research 成员对 @sonymusic 的一次戏谑询问引发了困惑与兴趣的交织，触及了 AI 在法律和创新领域的作用。
- **AI 排行榜上可疑的指标**：**AlpacaEval 排行榜** 的合法性受到质疑，工程师们在某个模型声称击败了 **GPT-4** 且更具成本效益后，对偏见指标提出了质疑。这引发了关于该领域性能排行榜可靠性的讨论。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **ROCm 分支加入竞争**：为了使用某些功能，建议工程师使用 [xformers](https://github.com/ROCm/xformers) 和 [flash-attention](https://github.com/ROCm/flash-attention) 的 **ROCm 分支版本**，并指出硬件支持特别针对 MI200 和 MI300 GPU，且需要 ROCm 5.4+ 和 PyTorch 1.12.1+。
- **奖励模型被认为不适合数据生成**：共识是 Reward Model 在生成数据方面效率不高，因为其主要设计目的是对数据质量进行分类，而非生产数据。
- **合成标准化考试题目**：分享了一个通过合成 **SAT**、**GRE** 和 **MCAT** 题目来改进针对小型模型的 AGI 评估的想法，并提议加入 **LSAT** 题目。
- **神秘的 Epoch 保存怪癖**：训练 Epoch 似乎以随机间隔保存，这种行为被认为是不寻常的，但社区对此并不陌生。这可能与训练过程中的 steps 计数器有关。
- **数据集格式入门与 MinHash 加速**：一位成员寻求关于 **llama2-13b** 数据集格式的建议，另一位成员讨论了使用 **JSONL** 格式化 **Alpaca** 数据集。此外，分享了一个名为 **Rensa** 的快速 MinHash 实现，用于数据集去重，声称比同类库快 2.5-3 倍，其 GitHub 仓库已开放供社区参与 ([Rensa on GitHub](https://github.com/beowolx/rensa))。
- **提示词结构剖析与镜像**：对 Axolotl 代码库中 `prompt_style` 的澄清揭示了不同的提示词格式化策略，重点介绍了 **INSTRUCT**、**CHAT** 和 **CHATML** 在交互用途上的差异。示例展示了如何使用 `ReflectAlpacaPrompter` 根据指定样式自动构建提示词结构 ([更多信息见 Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4809da1a-b260-413e-bdbe-8b82397846e6))。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile 升级**：[Llamafile v0.8.7](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7) 已发布，具有**更快的量化（quant）操作**和**错误修复**，并传闻即将适配 Android。
- **即将举行的全球 AI 活动**：旧金山正准备迎接 **World's Fair of AI** 和 **AI Quality Conference**，社区领袖将出席。同时 [Mozilla Nightly Blog](https://blog.nightly.mozilla.org/2024/06/24/experimenting-with-ai-services-in-nightly/) 暗示了可能集成 llamafile 以提供 AI 服务。
- **Mozilla Nightly 博客讨论 Llamafile**：Nightly 博客详细介绍了由 llamafile 驱动的本地 AI 聊天服务的实验，预示着更广泛的采用和用户可访问性。
- **成功在 Colab 上执行 Llamafile**：演示了在 Google Colab 上成功运行 llamafile，并提供了一个[供他人参考的模板](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g)。
- **内存管理器改版连接 Cosmos 与 Android**：Cosmopolitan 项目的一个重要 [GitHub commit](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca) 翻新了内存管理器，实现了对 Android 的支持，并引发了通过 Termux 运行 llamafile 的兴趣。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ORPO 缺失的一环**：**Torchtune** 目前不支持 ORPO 训练选项，尽管 DPO 可以使用已记录的[训练配方](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9)。社区成员提到可以参考 [ORPO/DPO 混合数据集](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)。
- **Epochs 固定为单一设置**：在 **Torchtune** 上进行多数据集训练时，目前不允许为每个数据集设置不同的 Epochs。用户应使用 *ConcatDataset* 合并数据集，但同样的 Epochs 数量将适用于所有数据集。
- **是否使用 ChatML**：工程师们讨论了在 **Llama3** 模型中使用 ChatML 模板的有效性，对比了使用 instruct tokenizer 和特殊标记的方法，以及不包含这些元素的基座模型方法，并引用了 [Mahou-1.2-llama3-8B](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B) 和 [Olethros-8B](https://huggingface.co/lodrick-the-lafted/Olethros-8B) 等模型。
- **微调 Phi-3 需要调整**：针对微调 **Phi-3 模型**（如 Phi-3-Medium-4K-Instruct）的任务，建议修改 tokenizer 并在 Torchtune 中添加自定义构建函数以实现兼容性。
- **系统提示词：在 Phi-3 中巧妙实现**：尽管 **Phi-3** 未针对系统提示词（System Prompts）进行优化，但用户可以通过将系统提示词置于用户消息之前来绕过此限制，并调整 tokenizer 配置中的特定 [flag](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128) 以促进微调。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **条件编码难题**：在关于 **tinygrad** 的讨论中，使用类似 `condition * a + !condition * b` 的条件操作来简化 WHERE 函数的做法受到了质疑，因为这可能会导致 *NaNs* 问题。
- **Tinygrad 中的 Intel 探索**：关于 **tinygrad** 对 **Intel 支持** 的查询显示，虽然 **opencl** 是一个可用选项，但该框架至今尚未集成 XMX 支持。
- **周一会议要点**：**tinygrad** 的 **0.9.1 版本**发布已列入下周一会议议程，重点包括 *tinybox* 更新、新 profiler、运行时改进、`Tensor._tri`、llama 推理加速，以及针对 *uop matcher 速度*和 *unet3d* 改进的悬赏任务。
- **Tinygrad 增加缓冲区视图切换功能**：**tinygrad** 的一次提交引入了一个用于切换缓冲区视图（buffer view）的新 flag，该更改已通过 [GitHub Actions 运行](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120)得到证实。
- **Lazy.py 逻辑备受关注**：一位工程师在修改 **tinygrad** 内部的 `lazy.py` 后，发现 process replay 的结果好坏参半，寻求进一步澄清，这表明需要更深入的调查或同行评审。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Claude Sonnet 3.5 性能惊艳**：一位工程师分享了在 Websim 中使用 **Claude Sonnet 3.5** 的经验，赞扬了它的**速度、创造力**和**智能**。他们对“在新标签页中生成”功能特别感兴趣，并尝试通过玩转标志性时尚品牌的配色方案来进行感官互动，正如其[分享的推文](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413)所示。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AWS Cloud Club 在 MJCET 启动**：MJCET 成立了 **Telangana 首个 AWS Cloud Club**，这是一个旨在为学生提供 **Amazon Web Services** 资源和经验的社区，为他们的科技行业职业生涯做准备。
- **与 AWS 专家共同参与云技术精通活动**：一场庆祝 AWS Cloud Club 启动的开幕活动将于 **2024 年 6 月 28 日**举行，届时将邀请 **AWS Community Hero Faizal Khan 先生**出席。感兴趣的人士可以通过[活动链接](https://meetu.ps/e/NgmgX/14DgQ2/i)进行 RSVP。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区长时间没有动态，请告知我们，我们将将其移除。

---

# PART 2: 渠道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1253795430014259370)** (715 messages🔥🔥🔥):

- **Juggernaut Lightning vs SD3 Turbo**：一位成员建议使用 Juggernaut Lightning，认为作为基础模型，它比 SD3 Turbo “真实得多”。另一位成员提到 Juggernaut 更适合角色扮演和创意，而不是编程和智能。
- **初学者帮助**：一位 ML 初学者就项目应使用的库寻求建议，并收到了使用 PyTorch（因其广泛的神经网络支持）和 HuggingFace（用于加载预训练模型）的建议。另一位成员建议避免使用像 sklearn 这样过时的库。
- **模型加载问题**：一位成员在有限的硬件上加载大型 AI 模型时面临挑战，并收到了使用量化技术提高性能的指导。建议包括安装 bitsandbytes 库，以及修改模型加载配置以利用 4-bit 精度的指令。
- **AI 内容创作工具**：讨论了生成类似于 Vidalgo 的 AI 生成视频的复杂性，指出虽然生成文本和音频很直接，但创建小型动态视频具有挑战性。建议使用 RunwayML 和 Capcut 等工具进行视频编辑和素材图片处理。
- **协作项目和模型更新**：成员们分享了与各种 AI 模型相关的经验和项目，包括一个训练用于通过 Xbox 控制器输入玩游戏的模型，以及一个用于预处理大型图像数据集的工具包。此外，还讨论了几个模型的持续工作、即将到来的更新及其潜在应用。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.continue.dev/how-to-use-continue">🧑‍🎓 如何使用 Continue | Continue</a>：在编码时通过 Continue 使用 LLM</li><li><a href="https://en.wikipedia.org/wiki/Chess_notation">国际象棋记谱法 - 维基百科</a>：未找到描述</li><li><a href="https://bhosmer.github.io/mm/ref.html">mm ref</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=_mkyL0Ww_08">Anthropic 震撼的新模型打破了软件行业！Claude 3.5 Sonnet 疯狂的编程能力</a>：跟我学 AI：https://www.skool.com/natural20/about 加入我的社区和课堂学习 AI，为新世界做好准备。#ai #openai #llm</li><li><a href="https://www.swebench.com/">SWE-bench</a>：未找到描述</li><li><a href="https://huggingface.co/briaai/RMBG-1.4">briaai/RMBG-1.4 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-141b-A35b">alignment-handbook/recipes/zephyr-141b-A35b at main · huggingface/alignment-handbook</a>：将语言模型与人类及 AI 偏好对齐的稳健方案 - huggingface/alignment-handbook</li><li><a href="https://en.wikipedia.org/wiki/Apple_M1#GPU">Apple M1 - 维基百科</a>：未找到描述</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta">alignment-handbook/recipes/zephyr-7b-beta at main · huggingface/alignment-handbook</a>：将语言模型与人类及 AI 偏好对齐的稳健方案 - huggingface/alignment-handbook</li><li><a href="https://huggingface.co/papers/2310.16944">论文页面 - Zephyr: Direct Distillation of LM Alignment</a>：未找到描述</li><li><a href="https://huggingface.co/chat?model=HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://github.com/abi/screenshot-to-code">GitHub - abi/screenshot-to-code: 拖入截图并将其转换为整洁的代码 (HTML/Tailwind/React/Vue)</a>：拖入截图并将其转换为整洁的代码 (HTML/Tailwind/React/Vue) - abi/screenshot-to-code</li><li><a href="https://github.com/simpler-env/SimplerEnv">GitHub - simpler-env/SimplerEnv: 在常见设置（如 Google Robot, WidowX+Bridge）的模拟环境中评估和复现真实世界的机器人操作策略（如 RT-1, RT-1-X, Octo）</a>：在常见设置（如 Google Robot, WidowX+Bridge）的模拟环境中评估和复现真实世界的机器人操作策略（如 RT-1, RT-1-X, Octo） - simpler-env/SimplerEnv</li><li><a href="https://huggingface.co/blog?tag=rlhf">Hugging Face – 博客</a>：未找到描述</li><li><a href="https://huggingface.co/posts/nroggendorff/357091156426242">Hugging Face 上的 @nroggendorff："@osanseviero 该你了"</a>：未找到描述</li><li><a href="https://youtu.be/udPY5rQVoW0">运行神经网络版本的 GTA V：GAN Theft Auto</a>：GAN Theft Auto 是一个重构了《侠盗猎车手 5》(Grand Theft Auto 5) 环境的生成对抗网络。它是使用基于 NVIDIA Ga... 的 GameGAN 分支创建的。</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/tAEVgAa4CZw?si=_1ThlIeIQyJAGpze">手势绘图应用演示 | Python OpenCV & Mediapipe</a>：在此视频中，我演示了使用 Python 结合 OpenCV 和 Mediapipe 开发的手势绘图应用。该应用允许你使用手势在屏幕上绘图...</li><li><a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1">stabilityai/stable-video-diffusion-img2vid-xt-1-1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main">microsoft/Phi-3-mini-4k-instruct-gguf at main</a>：未找到描述</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">使用 llama3 的 RAG 聊天机器人</a>：未找到描述</li><li><a href="https://huggingface.co/Azazelle/L3-RP_io/tree/main">Azazelle/L3-RP_io at main</a>：未找到描述</li><li><a href="https://www.vidalgo.tech/">Vidalgo - 一键生成竖屏视频</a>：体验 Vidalgo 带来的轻松视频创作！我们的平台让你只需点击一下，即可为 TikTok、YouTube Shorts 和 Instagram Reels 制作出色的竖屏视频。今天就开始创作吧...</li><li><a href="https://huggingface.co/stabilityai/stablelm-zephyr-3b">stabilityai/stablelm-zephyr-3b · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/huggingface/">Hugging Face</a>：构建未来的 AI 社区。Hugging Face 拥有 227 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a hre

<li><a href="https://tenor.com/view/toy-story-woody-buzz-lightyear-funny-gif-13488605">玩具总动员 Woody GIF - 玩具总动员 Woody Buzz Lightyear - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Azaz">azaz (Z)</a>：未找到描述</li><li><a href="https://github.com/huggingface/alignment-handbook">GitHub - huggingface/alignment-handbook: Robust recipes to align language models with human and AI preferences</a>：用于将语言模型与人类及 AI 偏好对齐的鲁棒配方 - huggingface/alignment-handbook</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/agent#transformers.Agent">Agents &amp; Tools</a>：未找到描述</li><li><a href="https://github.com/beowolx/rensa">GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets</a>：基于 Rust 的高性能 MinHash 实现，带有 Python 绑定，用于大规模数据集的高效相似度估计和去重 - beowolx/rensa</li><li><a href="https://stackoverflow.com/questions/7821661/how-to-write-code-to-autocomplete-words-and-sentences">如何编写代码来实现单词和句子的自动补全？</a>：我想编写在 Linux 终端中执行自动补全的代码。代码应按如下方式工作。它有一个字符串列表（例如 &quot;hello, &quot;hi&quot;, &quot;how a...</li><li><a href="https://github.com/minimaxir/textgenrnn">GitHub - minimaxir/textgenrnn: Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code.</a>：只需几行代码，即可在任何文本数据集上轻松训练属于你自己的、具有任何规模和复杂度的文本生成神经网络。 - minimaxir/textgenrnn</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.</a>：通过提供一套与平台无关的可定制 Pipeline 处理模块，将数据处理从脚本编写的疯狂中解放出来。 - huggingface/datatrove</li><li><a href="https://github.com/not-lain/loadimg">GitHub - not-lain/loadimg: a python package for loading images</a>：一个用于加载图像的 Python 包。通过在 GitHub 上创建账户来为 not-lain/loadimg 的开发做出贡献。</li><li><a href="https://tenor.com/view/vaas-far-cry3-that-is-crazy-gif-26006603">Vaas Far Cry3 GIF - Vaas Far Cry3 That Is Crazy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/vikhyatk/status/1804672473172210106">来自 vik (@vikhyatk) 的推文</a>：让 Claude 帮我制作了一个酷炫的新蒸汽波风格主页……我应该切换过去吗？</li><li><a href="https://x.com/vikhyatk/status/1804673335437254721">来自 vik (@vikhyatk) 的推文</a>：“让它变得更好”</li><li><a href="https://we.tl/t-3ZjcQJIKA2">sonnet_shooter.zip</a>：通过 WeTransfer 发送了 1 个文件，这是在全球范围内发送文件的最简单方式</li><li><a href="https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py#L705">huggingface_hub/src/huggingface_hub/hub_mixin.py 位于 main 分支 · huggingface/huggingface_hub</a>：Huggingface Hub 的官方 Python 客户端。 - huggingface/huggingface_hub</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bnetfp/is_the_p40_the_most_costeffective_way_to_run/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://wandb.ai/vanpelt/m1-benchmark/reports/Can-Apple-s-M1-help-you-train-models-faster-cheaper-than-NVIDIA-s-V100---VmlldzozNTkyMzg">Apple 的 M1 能否比 NVIDIA 的 V100 更快、更便宜地训练模型？</a>：在本文中，我们分析了在 M1 Mac Mini 和 Nvidia V100 上进行 Tensorflow 训练的运行时间、能耗和性能。</li><li><a href="https://github.com/maxmelichov/Text-To-speech">GitHub - maxmelichov/Text-To-speech: Roboshaul</a>：Roboshaul。通过在 GitHub 上创建账户来为 maxmelichov/Text-To-speech 的开发做出贡献。</li><li><a href="http://www.roboshaul.com/">Robo-Shaul 项目</a>：Robo-Shaul 竞赛是一项 2023 年的比赛，旨在克隆 Shaul Amsterdamski 的声音。结果都在这里。</li><li><a href="https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/">介绍 Mac 上的加速 PyTorch 训练</a>：通过与 Apple 的 Metal 工程团队合作，我们很高兴地宣布支持 Mac 上的 GPU 加速 PyTorch 训练。到目前为止，Mac 上的 PyTorch 训练仅利用 CPU，但...</li><li><a href="https://www.nature.com/articles/s41467-024-46631-y">自然语言中大脑嵌入与人工上下文嵌入的对齐指向共同的几何模式 - Nature Communications</a>：在这里，作者利用下额叶回的神经活动模式和大型语言模型（LLM）嵌入，提供了证据...</li>

nce for a common neural code for language processing.</li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1254401005286592604)** (3 messages):

- **编写 Self-Attention 和 Multi-Head Attention 代码**：一位成员分享了他们的博客文章链接，详细介绍了如何从零开始实现 Self-Attention 和 Multi-Head Attention。该博文解释了 Attention 在 Transformer 架构中对于理解句子中单词关系以进行准确预测的重要性。[点击此处阅读全文。](https://ash-01xor.github.io/blog/posts/Attention/)
- **对博文的兴趣**：另一位成员对关于 Attention 机制的博文表示了兴趣，并简单地回复了 "Yes I am interested" 以确认参与。
- **Tree-Sitter S-expression 的挑战**：一位成员提到了他们在处理 Tree-Sitter S-expression 时面临的挑战，称其为“一件苦差事”。这暗示了在他们目前的工作中，解析或处理这些表达式存在困难。

**提及的链接**：[Ashvanth.S Blog - Wrapping your head around Self-Attention, Multi-head Attention](https://ash-01xor.github.io/blog/posts/Attention/)：未找到描述。

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)** (5 messages):

- **在 SD3 中实现 RMSNorm 层**：一位成员提到为 Q 和 K 输入实现了一个可选的 **RMSNorm 层**，参考了 [SD3 论文](https://arxiv.org/pdf/2403.03206)。目前没有提供关于此实现的更多细节。
- **LLMs 与拒绝机制**：分享了一篇关于 **LLM 拒绝/安全** 的博文，强调 *拒绝是由 residual stream 中的单一方向介导的*。完整的解释和更多见解可以在 [已发布在 arXiv 上的论文](https://arxiv.org/abs/2406.11717) 中找到。
- **Florence-2 视觉基础模型**：视觉基础模型 **Florence-2** 的摘要已发布在 [arXiv](https://arxiv.org/abs/2311.06242) 上。Florence-2 在各种计算机视觉和视觉语言任务中使用统一的基于 prompt 的表示，并利用了拥有 54 亿个标注的大型数据集。
- **Facebook AI Twitter 链接**：分享了一个与 **Facebook AI** 相关的 Twitter 链接，未提供额外背景。[Twitter 链接](https://twitter.com/FacebookAIslop)
- **wLLama 测试页面**：分享了一个 **wLLama 基础示例** 页面的链接，演示了模型 completions 和 embeddings。用户可以测试模型、输入本地文件，并计算文本 embeddings 之间的余弦距离 [wLLama 基础示例](http://wllama-basic-example.glitch.me/)。
  

**提及的链接**：

- [wllama.cpp demo](http://wllama-basic-example.glitch.me/)：未找到描述。
- [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)：我们介绍了 Florence-2，这是一种新型视觉基础模型，具有统一的、基于 prompt 的表示，适用于各种计算机视觉和视觉语言任务。虽然现有的视觉大模型表现出色...
- [Refusal in LLMs is mediated by a single direction — AI Alignment Forum](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction.)：这项工作是 Neel Nanda 在 ML Alignment & Theory Scholars Program - 2023-24 冬季队列中的研究项目的一部分，由...共同指导。
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)：对话式大语言模型针对指令遵循和安全性进行了微调，使得模型能够服从良性请求但拒绝有害请求。虽然这种拒绝行为被广泛...

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1253878207875125328)** (12 messages🔥):

- **Mistroll 7B 2.2 版本发布**：一位成员分享了 [Mistroll-7B-v2.2 模型](https://huggingface.co/BarraHome/Mistroll-7B-v2.2)，该模型使用 Unsloth 和 Huggingface 的 TRL 库，训练速度提升了 2 倍。该实验旨在修复模型中的错误行为，并优化专注于数据工程和评估性能的训练流水线。
- **Stable Diffusion 训练器代码分享**：在 [GitHub](https://github.com/CodeExplode/MyTrainer) 上分享了一个用于实验的简单 Stable Diffusion 1.5 微调器（Finetuner）。这段“非常简陋”的代码使用了 Diffusers，旨在帮助用户探索微调。
- **媒体转文本软件发布**：由一位成员开发，该软件使用 PyQt 作为 GUI，OpenAI Whisper 作为 STT，将媒体文件转换为文本，支持本地和 YouTube 视频转录。可在 [GitHub](https://github.com/yjg30737/whisper_transcribe_youtube_video_example_gui) 上获取。
- **SimpleTuner 增强功能**：分享了重构并增强的 SimpleTuner EMA 支持，现在兼容 SD3 和 PixArt 训练，支持 CPU offload 和跳步（step-skipping）。更改内容可以在 [GitHub](https://github.com/bghira/SimpleTuner/pull/521/files) 上查看。
- **Featherless.ai - 新 AI 平台**：一位成员介绍了 [Featherless.ai](https://featherless.ai)，这是一个可以即时以 Serverless 方式运行 Huggingface 公共模型的平台。他们每周上线 100 多个模型，目标是覆盖所有 HF 公共模型，并邀请用户试用该服务并提供反馈。
  

**提到的链接**：

- [BarraHome/Mistroll-7B-v2.2 · Hugging Face](https://huggingface.co/BarraHome/Mistroll-7B-v2.2)：未找到描述
- [Linear Regression From Scratch In Python](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed)：学习如何使用纯 Python 从零实现线性回归。包括代价函数、梯度下降算法、模型训练等……
- [GitHub - CodeExplode/MyTrainer: A simple Stable Diffusion 1.5 Finetuner for experimentation](https://github.com/CodeExplode/MyTrainer)：一个用于实验的简单 Stable Diffusion 1.5 微调器 - CodeExplode/MyTrainer
- [GitHub - yjg30737/pyqt-assistant-v2-example: OpenAI Assistant V2 Manager created with PyQt (focused on File Search functionality)](https://github.com/yjg30737/pyqt-assistant-v2-example)：使用 PyQt 创建的 OpenAI Assistant V2 管理器（专注于文件搜索功能） - yjg30737/pyqt-assistant-v2-example
- [GitHub - yjg30737/whisper_transcribe_youtube_video_example_gui: GUI Showcase of using Whisper to transcribe and analyze Youtube video](https://github.com/yjg30737/whisper_transcribe_youtube_video_example_gui)：使用 Whisper 转录和分析 Youtube 视频的 GUI 展示 - yjg30737/whisper_transcribe_youtube_video_example_gui
- [EMA: refactor to support CPU offload, step-skipping, and DiT models | pixart: reduce max grad norm by default, forcibly by bghira · Pull Request #521 · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/pull/521/files>)：未找到描述
- [CaptionEmporium/coyo-hd-11m-llavanext · Datasets at Hugging Face](https://huggingface.co/datasets/CaptionEmporium/coyo-hd-11m-llavanext)：未找到描述
- [Featherless - Serverless LLM](https://featherless.ai/)：Featherless - 最新的 LLM 模型，Serverless 架构，随取随用。
- [Featherless AI - Run every 🦙 AI model & more from 🤗 huggingface | Product Hunt](https://www.producthunt.com/posts/featherless-llm)：Featherless 是一个使用 Hugging Face 最新开源 AI 模型的平台。面对每天数百个新模型，你需要专门的工具来紧跟热潮。无论你的使用场景如何……

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1253848266756460545)** (5 messages):

- **Chad 计划讨论 LLM 推理**：一位成员宣布计划在下周六讨论“使用 LLM 进行推理（reasoning with LLMs）”，并获得了热烈支持。他觉得对这个话题最有信心，因此选择了它而不是 Triton。
- **准备“了解 LLM 推理的现状”**：Chad 表示他将从论文 *Understanding the Current State of Reasoning with LLMs* [arXiv 链接](https://arxiv.org/abs/2206.07682)开始，并参考了一篇详细的 Medium 文章 [文章链接](https://medium.com/@isamu-website/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0)。
- **探索 Awesome-LLM-Reasoning 仓库**：他提到正在深入研究 [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning) 仓库以及另一个同名仓库 [备选仓库链接](https://github.com/luban-agi/Awesome-LLM-reasoning)，以探索 LLM 在逻辑方面的现状。
- **提到综述论文**：Chad 计划阅读 *Natural Language Reasoning, A Survey* [综述 PDF](https://arxiv.org/pdf/2303.14725) 的开头部分，并参考 GPT-4 发布后发表的论文 [GPT-4 研究链接](https://openai.com/index/gpt-4-research/)。
- **寻求长期规划论文**：他表示有兴趣了解优秀的 LLM 长期规划（long-term planning）论文，特别是那些专注于渗透测试（pentesting）的论文。
  

**提到的链接**：

- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)：研究表明，扩大语言模型规模可以按预期提高在各种下游任务上的性能和样本效率。而本文讨论的是一种不可预测的现象，即我们……
- [Understanding the Current State of Reasoning with LLMs](https://medium.com/@isamu-website/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0)：本文的目标是通过浏览 Awesome-LLM-Reasoning 和 Awesome-LLM-reasoning 的仓库来了解当前的……

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1253912311622664213)** (9 messages🔥):

- **OCR 模型的性价比**：成员们正在寻求高性价比、且能输出 JSON 格式的 OCR 模型推荐。这突显了对高成本效益 AI 解决方案的持续追求。
- **面部稳定、发型变化的视频**：一段显示“面部几乎保持不变但发型不断变化”的模型视频引发了好奇，大家想知道是哪个模型实现了这一效果。视频可以在[这里](https://x.com/gunsnrosesgirl3/status/1804462040871801220?s=46)找到。
- **不支持的图像类型 RuntimeError**：一位用户遇到了 *"RuntimeError: Unsupported image type, must be 8bit gray or RGB image."*。这发生在为人脸识别进行图像编码的过程中，用户提供了代码用于调试。

**提到的链接**：[来自 Science girl (@gunsnrosesgirl3) 的推文](https://x.com/gunsnrosesgirl3/status/1804462040871801220?s=46)：使用 AI 进行的时尚演变

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages):

capetownbali: 让我们知道你在 Llama 上的微调（fine tuning）进展如何！

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1254221908036026501)** (2 messages):

- **重定向到 diffusion-discussions 频道**：一位用户建议，“你最好在[这里](https://discord.com/channels/879548962464493619/1019883044724822016)询问”以进一步讨论相关话题。
- **询问音频转换模型**：一位成员询问是否有音频到音频（audio-to-audio）转换的模型，特别是从乌尔都语/印地语到英语，这表明了对多语言处理能力的需求。

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1253788511379787947)** (376 messages🔥🔥):

- **Cossale 迫切期待 Unsloth 的发布**：他们请求了早期访问权限，并被 theyruinedelise 告知视频将于次日拍摄。与此同时，他们可以观看一个临时的[录像](https://discord.com/channels/1179035537009545276/1179035537529643040/1253726564500111381)。
- **关于缩略图和流程图的反馈**：Cossale 建议修改缩略图以提高清晰度，促使 theyruinedelise 将其从 "csv -> unsloth + ollama" 更新为 "csv -> unsloth -> ollama"。他们还建议在 Logo 下方为初级用户添加描述性文字。
- **关于巨量 VRAM 的讨论令人印象深刻**：成员们讨论了 Phison 令人印象深刻的 PCIe-NVMe 卡，其表现为 1Tb VRAM，并影响了性能。Fimbulvntr 分享了一个 [YouTube 视频](https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&t=679) 来解释这项技术。
- **围绕扩展 LLM 的兴奋**：Fimbulvntr 成功将 Llama-3-70b 的上下文扩展到 64k，而 iron_bound 辩论了 VRAM 扩展对性能的影响。对话涉及了各种大模型更新及其潜在影响。
- **社区即将发布的版本和资源**：Theyruinedelise 宣布 Ollama 更新定于周一或周二，将包含 CSV 文件支持。此外，Sebastien 微调的情感 Llama 模型及其相关资源现已在 [Ollama](https://ollama.com/sebdg/emotional_llama) 和 [YouTube](https://www.youtube.com/watch?v=ZJKglSWgD0w) 上可用。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>：未找到描述</li><li><a href="https://app.uniswap.org/explore/tokens/ethereum/0xfaca6611fca6de09f726b8a0a1448253b6f748e5">Get DOLPHIN on Uniswap</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1805265568658076069">Tweet from Unsloth AI (@UnslothAI)</a>：明天我们将在 @aiDotEngineer 世界博览会上发放我们的新贴纸！🦥 请在 6 月 25 日上午 9 点加入我们，届时我们将举办关于 LLM 分析与技术、Ollama 支持等的研讨会...</li><li><a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>：低秩自适应（LoRA）是一种流行的针对大语言模型的参数高效微调方法。在本文中，我们分析了 LoRA 中实现的低秩更新的影响。我们的研究结果表明...</li><li><a href="https://x.com/Nottlespike/status/1805054022661087657?t=oJn25UE9vTesxym63ToA1A&amp;s=09">Tweet from Kearm (@Nottlespike)</a>：http://x.com/i/article/1805030133478350848</li><li><a href="https://x.com/dudeman6790/status/1805108449581072710">Tweet from RomboDawg (@dudeman6790)</a>：发布 Replete-Coder-Qwen2-1.5b，这是一个无审查的 1.5b 模型，在 100 多种编程语言中具有良好的代码性能，提供开源数据、权重、训练代码，并可完全在移动平台上使用...</li><li><a href="https://www.youtube.com/watch?v=ZJKglSWgD0w">Emotions in AI: Fine-Tuning, Classifying, and Reinforcement Learning</a>：在本视频中，我们将探索使用 Unsloth 和 Ollama 创建 LLM 微调数据集，以训练一个专门的情感检测模型。你...</li><li><a href="https://www.youtube.com/watch?v=dik_wnOE4dk">Tell 'im 'e's dreamin'</a>：电影《城堡》（The Castle）中的一些片段。</li><li><a href="https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&amp;t=679">AI and Unified Memory Architecture: Is it in the Hopper? Is it Long on Promise, Short on Delivery?</a>：坐下来，放松，享受 Wendell 漫谈的舒缓声音。本集重点关注 MI 300a/x 和 Nvidia Grace Hopper。尽情享受吧！...</li><li><a href="https://cloud.llamaindex.ai">LlamaCloud</a>：未找到描述</li><li><a href="https://tenor.com/view/noice-nice-click-gif-8843762">Noice Nice GIF - Noice Nice Click - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/sebdg/SebLLama-Notebooks/tree/main/Emotions">SebLLama-Notebooks/Emotions at main · sebdg/SebLLama-Notebooks</a>：通过在 GitHub 上创建账户来为 sebdg/SebLLama-Notebooks 的开发做出贡献。</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>：用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产机器学习流水线。- GitHub - Unstructured-IO/unstructured</li><li><a href="https://github.com/datamllab/LongLM">GitHub - datamllab/LongLM: [ICML'24 Spotlight] LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning</a>：[ICML'24 Spotlight] LLM 也许是 LongLM：无需微调即可自我扩展 LLM 上下文窗口 - datamllab/LongLM</li><li><a href="https://colab.research.google.com/drive/1lq043a_zdssGBWJakckyy3yrbNSqqixP#scrollTo=ekOmTR1hSNcr">Google Colab</a>：未找到描述</li><li><a href="https://ollama.com/sebdg/emotional_llama">sebdg/emotional_llama</a>：介绍 Emotional Llama，这是为 Ollama Discord 频道的直播活动练习而微调的模型。旨在理解并响应广泛的情感。</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Qwen-1.5b">Replete-AI/Replete-Coder-Qwen2-1.5b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Replete-AI/Adapter_For_Replete-Coder-Qwen-1.5b/">Replete-AI/Adapter_For_Replete-Coder-Qwen2-1.5b · Hugging Face</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1253862702561103904)** (108 条消息🔥🔥):

- **罗技鼠标与 ChatGPT 套壳应用**: 一位成员讨论了如何使用罗技鼠标配合一个“酷炫”的 ChatGPT 套壳应用，该应用能够编程处理基础查询，例如总结和重写文本。他们分享了一个链接来展示该设置的 UI。


<div class="linksMentioned"><p><strong>提到的链接</strong>:</p><ul><li><a href="https://arxiv.org/abs/2401.11817">Hallucination is Inevitable: An Innate Limitation of Large Language Models</a>: 幻觉已被广泛认为是大型语言模型 (LLM) 的一个显著缺点。已有许多研究试图减少幻觉的程度。这些努力……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dl8guc/hf_eng_llama_400_this_summer_informs_how_to_run/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: PygmalionAI's large-scale inference engine</a>: PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号，为 PygmalionAI/aphrodite-engine 的开发做出贡献。</li><li><a href="https://link.springer.com/article/10.1007/s10676-024-09775-5">ChatGPT is bullshit - Ethics and Information Technology</a>: 最近，人们对大型语言模型产生了浓厚的兴趣：这些机器学习系统可以生成类人文本和对话。这些系统的应用一直受到持续存在的……困扰。</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1253789943264972892)** (228 条消息🔥🔥):

- **Windows 上 Xformers 的安装困扰**：一位用户在通过 conda 设置 Unsloth 时，在 Windows 上安装 xformers 遇到困难，遇到了 "PackagesNotFoundError"。 [另一位用户](https://anaconda.org) 建议这些挑战可能源于平台兼容性，从而引发了关于 Unsloth 是否在 Linux 上运行效果更好的讨论。
- **在 Colab 中导入 FastLanguageModel 出现问题**：用户报告了在 Unsloth 的 Google Colab 笔记本中导入 `FastLanguageModel` 时出现的问题。建议的解决方法是确保所有初始单元格（特别是安装 Unsloth 的单元格）都已正确执行。
- **结果因 Token 过期而异**：一位用户通过更换 Google 账号解决了问题，发现 Colab secrets 中过期的 Token 导致了问题，特别是在访问数据集和下载模型方面。
- **使用 Huggingface Token**：一位用户发现添加 Huggingface Token 修复了访问问题，这引发了困惑，因为这些模型本应是公开的。普遍观点认为，这可能是由于 Huggingface 访问的不一致性造成的。
- **使用 Docker 和 Jupyter 运行 Unsloth**：讨论了在 NVIDIA GPU Cloud (NGC) 容器上设置 Unsloth 的问题，并指出了特定 CUDA 和 PyTorch 版本的兼容性问题。解决方案包括尝试不同的容器以及仔细安装 xformers 和 bitsandbytes 等依赖项，用户还分享了他们的 Dockerfile 配置。
  

**提到的链接**：

- [PyTorch Release 24.05 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html#rel-24-05)：未找到描述
- [Home](https://github.com/unslothai/unsloth/wiki)：微调 Llama 3, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth)：未找到描述
- [GitHub - srush/Triton-Puzzles: Puzzles for learning Triton](https://github.com/srush/Triton-Puzzles/)：学习 Triton 的谜题。通过在 GitHub 上创建账号为 srush/Triton-Puzzles 的开发做出贡献。
- [Sao10K/Claude-3-Opus-Instruct-15K · Datasets at Hugging Face](https://huggingface.co/datasets/Sao10K/Claude-3-Opus-Instruct-15K)：未找到描述
- [I got unsloth running in native windows. · Issue #210 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/210)：我在原生 Windows（无 WSL）上运行了 Unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 Deepspeed。我有一个完整的安装教程，我本想在这里写下来，但我现在在用手机...
- [Google Colab breaks · Issue #243 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/243)：在 Colab 上使用 A100 GPU 尝试从 Unsloth 导入 FastLanguageModel 时遇到以下错误。由于以下错误，无法导入 transformers.integrations.peft...
- [Google Colab](https://colab.research.google.com/drive/19ScqSD6-p9NBrpyq5XzVwayhpNnn7YYf?usp=sharing)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing])：未找到描述
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBVEnvCDJv)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBV)：未找到描述
- [CUDA_VISIBILE_DEVICES not functioning · Issue #660 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/660)：当我尝试使用 4xA100 GPU 进行监督微调时看到了错误消息。所以免费版不能在多 GPU 上使用吗？RuntimeError: Error: 超过 1 个 GPU 具有大量的 VRAM 使用...
- [unsloth/unsloth/models/llama.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py)：微调 Llama 3, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
- [Package repository for pytorch :: Anaconda.org](https://conda.anaconda.org/pytorch)：未找到描述
- [Package repository for nvidia :: Anaconda.org](https://conda.anaconda.org/nvidia)：未找到描述
- [Package repository for xformers :: Anaconda.org](https://conda.anaconda.org/xformers)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1254452617564782723)** (1 messages):

- **关于 Apple 与 Meta 合作伙伴关系的博客引发讨论**：一位 AI 爱好者分享了一篇题为 [Apple and Meta Partnership: The Future of Generative AI in iPhones](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/) 的博客文章。该文章讨论了将生成式 AI 模型集成到 Apple 的 AI 系统中的影响、益处和挑战，引发了人们对科技格局潜在影响的关注。

**提及的链接**：[Apple and Meta Partnership: The Future of Generative AI in iPhones](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/)：最近关于 Apple 与 Meta 等 AI 公司就将生成式 AI 模型集成到 iPhone 的 Apple AI 系统中进行合作的讨论引起了极大兴趣。本文...

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1253786569949708398)** (583 messages🔥🔥🔥):

- **Discord 机器人广告引发争议**：一名成员分享了一个机器人链接，声称它集成了 Gemini 用于聊天辅助，并集成了 StabilityAI 用于 text-to-image 生成。其他人批评该链接缺乏上下文，并存在潜在的安全问题。
- **Civitai 与 SD3 的许可风波**：关于 Civitai 因许可问题移除 SD3 资源的讨论非常激烈。一位成员认为这是为了应对潜在的法律问题，而其他人则认为这种辩解值得怀疑。
- **在低端 GPU 上运行 Stable Diffusion**：多位成员讨论了在低配置机器上运行 Stable Diffusion 的挑战。建议包括使用 automatic1111 以及调整 steps 和 resolution 等设置，并就旧款 GPU 与 RTX 4080 等新款 GPU 的效能进行了辩论。
- **训练与技术讨论**：成员们就训练模型和处理错误（包括元数据和 VRAM 分配问题）寻求建议。建议加入特定的训练服务器或使用 ComfyUI 和 OneTrainer 等工具进行更好的管理。
- **模型集成的误解**：用户讨论了不同模型架构之间的兼容性问题，特别是 SD 1.5、SDXL 和 ControlNet 模块之间。强调了将模型类型与其相应的扩展插件匹配的重要性，以避免错误并提高性能。
  

**提及的链接**：

- [无标题](https://www.youtube.co)：未找到描述
- [Discord - Group Chat That’s All Fun & Games](https://dsc.gg/vexel)：Discord 是玩游戏、与朋友闲逛或建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。
- [Green Code](https://www.youtube.com/@Green-Code)：01001000 01101001 00100001 00100000 01001001 00100000 01101101 01100001 01101011 01100101 00100000 01110110 01101001 01100100 01100101 01101111 01110011 00100000 01100001 01100010 01101111 01110101 01...
- [Stable Diffusion 3 Medium - a Hugging Face Space by stabilityai](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium)：未找到描述
- [Alfredo Canziani](https://www.youtube.com/@alfcnz)：音乐、数学和从零开始的 deep learning
- [FIFO Diffusion 测试](https://www.youtube.com/watch?v=bRbIUf2XIII)：正如标题所言，使用 FIFO Diffusion 渲染。在 4090 上两个剪辑总共渲染了 4.5 小时。有点差强人意。会再试一次。
- [使用 Mad Scientist 节点的进阶风格迁移](https://youtu.be/ewKM7uCRPUg?si=9xtX87QB8_-3F19i)：我们正在讨论进阶风格迁移、Mad Scientist 节点以及使用 CosXL-edit 的 Img2Img。升级 IPAdapter 扩展以使用所有...
- [Hot Sweating GIF - Hot Sweating Melting - Discover & Share GIFs](https://tenor.com/view/hot-sweating-melting-burning-donald-duck-gif-12424818824830848928)：点击查看 GIF
- [Well, This Is Shit](https://open.spotify.com/track/0dyz56yOkAvjJMAJ5IxfiE?si=caaca21651c44ccc)：Thomas Benjamin Wild Esq · 歌曲 · 2021
- [lllyasviel/sd-controlnet-canny at main](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/main)：未找到描述
- [下载最新的 NVIDIA 官方驱动程序](http://www.nvidia.com/Download/index.aspx)：下载最新的 NVIDIA 官方驱动程序
- [PyTorch](https://pytorch.org)
- [美学列表](https://aesthetics.fandom.com/wiki/List_of_Aesthetics#Aesthetics_by_Type)：如果你在识别自己的审美或创建情绪板（moodboard）方面需要帮助，请随时在讨论选项卡中提问...
- [lllyasviel/sd_control_collection at main](https://huggingface.co/lllyasviel/sd_control_collection/tree/main)：未找到描述

- [TypeError: list indices must be integers or slices, not str](https://stackoverflow.com/questions/32554527/typeerror-list-indices-must-be-integers-or-slices-not-str)：我有两个列表，想把它们合并成一个数组并最终放入 csv 文件。我该如何避免这个错误：def fill_csv(self, array_urls, array_dates, csv_file_path): ...
- [stable-diffusion-webui/requirements_versions.txt at master · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/requirements_versions.txt)：Stable Diffusion web UI。通过在 GitHub 上创建账号，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
- [0002 - Pony - v3.1alt | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/471439/0002-pony)：与 0001 的区别：更高的饱和度、更明亮、更具动态感，使用了我训练的 2 个 LoRA：- QEW: [https://civitai.com/models/470285/qew-quasarca](https://civitai.com/models/470285/qew-quasarca)...
- [GitHub - Nerogar/OneTrainer: OneTrainer is a one-stop solution for all your stable diffusion training needs.](https://github.com/Nerogar/OneTrainer)：OneTrainer 是满足你所有 Stable Diffusion 训练需求的一站式解决方案。- Nerogar/OneTrainer
- [Feature request: Option to run CodeFormer and/or GFPGAN automatically again after upscale · Issue #1151 · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/1151)：你的功能请求是否与某个问题相关？请描述。我注意到 GFPGAN 和 CodeFormer 似乎在 upscale 发生之前运行，这导致分辨率有些模糊...
- [[Feature Request]: Offline Mode · Issue #11518 · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11518)：是否存在关于此问题的现有议题？我已搜索了现有议题并检查了最近的 build/commit。你的功能会实现什么？提供一个选项来下载所有可能需要的请求文件...
- [GitHub - lucidrains/mmdit: Implementation of a single layer of the MMDiT, proposed in Stable Diffusion 3, in Pytorch](https://github.com/lucidrains/mmdit)：在 PyTorch 中实现 Stable Diffusion 3 中提出的 MMDiT 单层 - lucidrains/mmdit
- [ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com](https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436)：购买 ABS Aquilon Aqua 游戏电脑 - Windows 11 家庭版 - Intel Core i7 第 14 代 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI 驱动性能 - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...
- [Don't ask to ask, just ask](https://dontasktoask.com/)：未找到描述
- [Civitai Link | One-click install Stable Diffusion models](https://civitai.com/product/link)：直接从 Civitai 下载任何模型到你的 Stable Diffusion 实例。
- [Update on SD3 on Civitai | Civitai](https://civitai.com/articles/5840/update-on-sd3-on-civitai)：标准免责声明：本帖不构成法律建议。你如何与 SAI 及其产品互动取决于你自己。你应该寻求自己的...
- [Stable Diffusion 3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3)：未找到描述
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)：未找到描述
- [SD3 IS HERE!! ComfyUI Workflow.](https://youtu.be/Di1KqPXxx2Y?si=NXtsTleKTijMBVJV)：SD3 终于支持 ComfyUI 了！Topaz Labs: [https://topazlabs.com/ref/2377/HOW](https://topazlabs.com/ref/2377/HOW) 如何支持我的频道 - 通过加入我的 Patreon 来支持我：[https://www.patreon.co](https://www.patreon.co)...
- [Deep Learning Fundamentals - Lightning AI](https://lightning.ai/courses/deep-learning-fundamentals/)：Deep Learning Fundamentals 是一门关于使用现代开源技术栈学习深度学习的免费课程。
- [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)：未找到描述
- [HuggingFace](https://www.youtube.com/@HuggingFace)：HuggingFace 的使命是通过开源和开放科学，一次一个 commit 地解决自然语言处理（NLP）问题。我们的 YouTube 频道提供有关 Machine Learning 的教程和视频...
- [Whose art is this, really? Inside Canadian artists’ fight against AI](https://www.thestar.com/news/canada/whose-art-is-this-really-inside-canadian-artists-fight-against-ai/article_54b0cb5c-7d67-5663-a46a-650b462da1ad.html)：视觉艺术家的作品正在网上被收集，并被用作计算机模仿的素材。当多伦多的 Sam Yang 向一个 AI 平台投诉时，他收到了一封他认为旨在嘲弄他的电子邮件...

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1253951731210125353)** (17 条消息🔥):

- **初学者询问工作组贡献方式**：一名新成员询问如何为工作组做贡献，想知道仅关注 GitHub 仓库是否足够，还是存在更正式的方法。
- **复杂 Kernel 中的寄存器使用**：一名成员分享了针对每个线程使用过多寄存器的 Kernel 的调试策略，建议注释掉部分代码或在 Nsight Compute 中检查 SASS。
- **宣布成立 CUTLASS 工作组**：一名成员提议组建一个工作组来创建 CUTLASS 的学习材料，邀请感兴趣的人加入，并建议通过观看 [关于 Tensor Cores 的 YouTube 演讲](https://youtu.be/hQ9GPnV0-50?feature=shared) 进行准备。
- **CPU 缓存见解**：一名成员分享了一份 [以 CPU 为中心的计算机缓存指南](https://pikuma.com/blog/understanding-computer-cache)，强调了理解缓存对程序员的重要性。
  

**提到的链接**：

- [第 23 讲：Tensor Cores](https://youtu.be/hQ9GPnV0-50?feature=shared)：幻灯片：https://drive.google.com/file/d/18sthk6IUOKbdtFphpm_jZNXoJenbWR8m/view?usp=drive_link
- [探索缓存内存的真实工作原理](https://pikuma.com/blog/understanding-computer-cache)：尽管我们经常听到 L1、L2、缓存块大小等术语，但大多数程序员对缓存究竟是什么了解有限。这是一份适合初学者的缓存工作原理入门读物。

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1253916276393250886)** (4 条消息):

- **INT4 LoRA 微调 vs QLoRA**：用户询问了 **INT4 LoRA 微调** 与 **QLoRA** 在精度和速度方面的差异。另一名成员解释说，带有 HQQ 的 QLoRA 涉及冻结的量化权重，不使用 tinnygemm，并且由于 tinnygemm 在处理长序列时效率低下，因此在 *torch.matmul* 的同时使用反量化（dequantizing）。
- **QLoRA 的性能与速度**：提到 **QLoRA** 保持了良好的质量和快速的性能，特别是在实现了 CUDA 反量化 Kernel (axis=0) 的情况下。另外提到了一项贡献，即用户创建了一个用于 int4 的融合 GEMM，这对于固定序列长度的训练非常有效，提供了最快的解决方案。

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1254356369302360109)** (1 条消息):

- **使用 NVIDIA 工具测量带宽、吞吐量和延迟**：一名成员分享了一份详细的 [GitHub 指南](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter09)，介绍如何使用 NVIDIA 工具测量 **带宽 (bandwidth)、吞吐量 (throughput) 和延迟 (latency)**。该指南提供了逐步说明，有助于更好的性能分析和优化。

**提到的链接**：[Guide-NVIDIA-Tools/Chapter09 at main · CisMine/Guide-NVIDIA-Tools](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter09)：通过在 GitHub 上创建账号来为 CisMine/Guide-NVIDIA-Tools 的开发做出贡献。

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1254368386608271361)** (1 条消息):

- **具备 AI 和 CUDA 技能的实习寻求者**：一名来自 **越南 (VietNam)** 的成员正在寻求 AI 和 CV 领域的远程实习，重点是 CUDA 优化。他们分享了自己的经验和两个 GitHub 仓库：[Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C) 和 [Guide-NVIDIA-Tools](https://github.com/CisMine/Guide-NVIDIA-Tools)。

**提到的链接**：[GitHub - CisMine/Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C)：通过在 GitHub 上创建账号来为 CisMine/Parallel-Computing-Cuda-C 的开发做出贡献。

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1254771364288532520)** (3 条消息):

- **寻求 AI/ML 基础知识**：一名成员征求在 Coursera 等平台上学习 **AI/ML** 基础知识的优质课程推荐。另一名成员询问了他们的编程、计算机科学或数学背景，以便推荐合适的资源。

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1253934851313107067)** (28 条消息🔥):

- **讨论了 FP8 转换中的精度损失**：成员们讨论了 **PyTorch 如何遵循 IEEE 惯例**进行 FP8 转换中的舍入，解决了精度损失问题，并建议对 Tensor 进行缩放（scaling）可以最大限度地减少这种损失。一位成员提到，缩放确保了更有效地利用 GPU 的数值范围 ([link](https://github.com/pytorch/pytorch/blob/f42d5b6dca75ee020355fc75532347ca2734b117/c10/util/Float8_e4m3fn.h#L46))。
- **浮点精度详解**：浮点精度问题是一个热门话题，一位成员分享了 [floating-point-gui.de](https://floating-point-gui.de/) 作为了解数值输出中意外精度误差的资源。
- **FP8 精度的缩放**：几位成员辩论了如何确定 Tensor 转换为 FP8 的 **scaling factors**（缩放因子），一些人建议基于 min/max 值或其他指标，以避免溢出（overflow）和欠载（underflow） ([link](https://gist.github.com/drisspg/64600f98c4a0cb41917afe81e757469e))。
- **分享 Quantization 学习资源**：对于那些希望更好理解 Quantization 的人，成员们推荐了各种资源，包括一个 [GitHub 论文列表](https://github.com/cuda-mode/awesomeMLSys) 和教学 YouTube 视频（[Quantization 详解](https://youtu.be/0VdNflU08yA) 和 [高级 Quantization](https://youtu.be/1u9xUK3G4VM)）。
- **FP8 缩放更新**：一位成员提到了 PyTorch 的最新更新，现在支持 FP8 转换的 **row-wise scaling**（逐行缩放），并暗示即将发布供社区讨论的帖子。
  

**提到的链接**：

- [Scaled_FP8.md](https://gist.github.com/drisspg/64600f98c4a0cb41917afe81e757469e): GitHub Gist：即时分享代码、笔记和片段。
- [Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training](https://youtu.be/0VdNflU08yA?feature=shared): 在这段视频中，我将介绍并解释 Quantization：我们将首先简要介绍整数和浮点数的数值表示...
- [Lecture 7 Advanced Quantization](https://youtu.be/1u9xUK3G4VM?feature=shared): 幻灯片: https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0
- [GitHub - cuda-mode/awesomeMLSys: An ML Systems Onboarding list](https://github.com/cuda-mode/awesomeMLSys): 一个 ML Systems 入门列表。通过在 GitHub 上创建账户为 cuda-mode/awesomeMLSys 的开发做出贡献。
- [pytorch/c10/util/Float8_e4m3fn.h at f42d5b6dca75ee020355fc75532347ca2734b117 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/f42d5b6dca75ee020355fc75532347ca2734b117/c10/util/Float8_e4m3fn.h#L46): Python 中具有强大 GPU 加速的 Tensor 和动态神经网络 - pytorch/pytorch
- [Float stored in 8 bits - ONNX 1.17.0 documentation](https://onnx.ai/onnx/technical/float8.html#cast): 未找到描述
- [The Floating-Point Guide - What Every Programmer Should Know About Floating-Point Arithmetic](https://floating-point-gui.de/): 未找到描述
- [Visualising ML number formats](https://thecharlieblake.co.uk/visualising-ml-number-formats): 机器学习数值格式的可视化 —— 我在网上找不到任何好的机器学习数值格式可视化，所以我决定自己做一个。它是交互式的，希望...
- [float8_experimental/float8_experimental/float8_utils.py at d4ade877dff327ea7f51e91f7cc218ae956e8cfd · pytorch-labs/float8_experimental](https://github.com/pytorch-labs/float8_experimental/blob/d4ade877dff327ea7f51e91f7cc218ae956e8cfd/float8_experimental/float8_utils.py#L142): 此仓库包含实验性的 PyTorch 原生 float8 训练 UX - pytorch-labs/float8_experimental
- [float8_experimental/test/test_base.py at d4ade877dff327ea7f51e91f7cc218ae956e8cfd · pytorch-labs/float8_experimental](https://github.com/pytorch-labs/float8_experimental/blob/d4ade877dff327ea7f51e91f7cc218ae956e8cfd/test/test_base.py#L86-L111): 此仓库包含实验性的 PyTorch 原生 float8 训练 UX - pytorch-labs/float8_experimental

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1254565529515982928)** (18 条消息🔥):

- **Valorant 账号因关联作弊者被封禁**：一位用户的社交好友因与作弊者组队，其 Valorant 账号被封禁 180 天。*"我建议她去找支持团队，但她现在非常焦虑，所以我觉得值得在这里提一下。"*
- **对账号封禁的焦虑**：这位朋友非常焦虑，在寻求进一步帮助前仅等待了支持团队一小时。*"我告诉她先等等。"*
- **提供的地区和详情**：用户提到受影响的朋友位于加利福尼亚州，玩的是 Valorant。*"她在加州，她刚刚告诉我的。"*
- **支持查询的回复**：一位回复者提到可能会调查此问题，但也指出可能无能为力。*"我觉得答案是‘真的没办法’，哈哈。"*
- **回放审查与合理的封禁**：对方保证会查看回放，以确保封禁是合理的。*"他们会查看回放并做出适当的封禁处理！"*

---

### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1254763687831535616)** (2 条消息):

- **运行 `torchao_int4_demo.py` 产生无意义输出**：一名成员报告在尝试运行 `torchao_int4_demo.py` 时得到了无意义的输出，例如 *"Unterscheidung Hinweis Unterscheidung Einzeln Unterscheidung Unterscheidung ..."*。他们提到唯一的改动是 *"设置了 `compile=None`码"*，并向另一名成员寻求帮助。后者询问该问题是否在所有模型上都会出现，并建议尝试使用 `'axis=0'`。

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1253786744155930655)** (465 条消息🔥🔥🔥):

- **NCCL 初始化计划**：一名成员提议使用 **MPI** 来初始化 **NCCL**，并在 **MPI** 不可用时回退到文件系统或 TCP Sockets。他们的目标是将 GPU 计算保留在 CUDA 中，以确保稳定性和性能。
- **H100 与 A100 训练稳定性**：成员们讨论了 **H100 GPU** 与 **A100 GPU** 相比在训练中的不稳定性，**H100** 在约 28K 步时会出现梯度“爆炸”。有人建议将计算复制到 GPU 以避免此问题。
- **CUDA 与多节点设置**：在测试使用 **MPI**、**slurm** 和 TCP Sockets 等不同方法的多节点设置方面投入了大量精力。讨论内容包括确保所有节点协同工作且无显著开销所需的改进。
- **集成 FP8 Matmuls**：一名成员描述了集成 **FP8 matmuls** 的过程，并观察到性能略有提升。他们分享了与 FP8 Tensor Cores 相关的详细挑战和策略，以及优化重缩放（rescaling）和转置（transposing）操作的方法。
- **集群训练准备**：讨论了在新的 **Lambda 集群**上尝试训练大语言模型的计划，旨在更快地完成重要的训练里程碑。这包括确保成本效益以及验证不同硬件设置下训练运行的稳定性。

**提到的链接**：

- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)：大语言模型是在海量的网络抓取数据上训练的，这些数据通常是非结构化的、有噪声的且措辞不当。当前的缩放定律（Scaling Laws）表明，从这类数据中学习需要大量的……
- [WIP Distribution Visualisation to help with FP8 work & beyond by ademeure · Pull Request #618 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/618)：尚未准备好集成 / 仍然非常粗糙，有一堆未解决的问题，我不确定代码应该放在哪里等：需要找到一种方法，减少这些生成代码对主代码库的污染……
- [Socket server/client interface by chinthysl · Pull Request #633 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/633/files#diff-6b403830ffefa78e4ce238b4ab24bb0624dfede82fe2a5214ee63e2cfda07a19)：用于利用 PR #632 中分布式接口的虚拟 PR。
- [FlexNet 11.19.5 build on Visual Studio 2015](https://community.flexera.com/t5/FlexNet-Publisher-Forum/FlexNet-11-19-5-build-on-Visual-Studio-2015/m-p/306967)：大家好，我正尝试用 FlexNet 11.19.5 构建我的应用。我遇到了一些编译器问题（Visual Studio 2015）：c:\\program files (x86)\\windows kits\\8.1\\include\\shared\\ws2def.h(100): warning C4005: '...
- [MPI/TCP/FS for NCCL-init by gordicaleksa · Pull Request #632 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/632)：与其在训练期间混合使用 NCCL 和 Open MPI，不如过渡到仅使用 NCCL。据我所知，这样做没有缺点，它们是等效的，且在速度方面……

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1254425625394020453)** (2 messages):

- **PCIe 限制讨论**：成员们讨论了 **PCIe 在通信方面的功耗、重量和引脚限制**。一位成员指出，不开发低规格产品的主要原因是专注于销售利润更高的高端服务器。
- **目标锁定大客户**：另一位成员推测，公司主要针对 **云 GPU 提供商等大客户**。这与其目前追求收入最大化的产品策略相一致。

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1254075604400341034)** (25 messages🔥):

- **调试 Bitnet Tensor 问题**：成员们在运行一个可训练网络时遇到了 Bitnet tensors 的问题，由于维度无法被 4 整除而触发错误。分享的错误回溯显示，`AssertionError` 是由于 Bitnet 调度尝试执行不支持的 `aten.to.dtype_layout` 操作引起的。
- **更新的测试脚本和仓库链接**：更新后的测试脚本已链接至 [CoffeeVampir3 的 GitHub](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py) 以使用新的库路径。CoffeeVampir3 还在此分享了主仓库链接 [here](https://github.com/CoffeeVampir3/ao-bitnet/tree/main)。
- **Affine Quantization 讨论**：Vayuda 和 Jerry 讨论了将 Bitnet tensors 集成到 AffineQuantizedTensor 中的可能性，考虑为打包张量（packed tensors）创建一个新的布局，以指示当前打包的维度。Jerry 强调 bit (uint1) tensors 应该保持独立，但要与仿射量化张量兼容。
- **寻求帮助并请求最小复现**：Marksaroufim 请求一个最小复现示例（minimal reproducible example）来调试 Bitnet tensors 中的 dtype 转换问题。CoffeeVampir3 提供了测试脚本的链接以方便调试过程。
- **新教程和 Tensor Subclassing 想法**：Marksaroufim 建议在 [PyTorch ao 库](https://github.com/pytorch/ao/issues/426)上发布新教程，强调该库在处理量化优化器和 kv caches 方面的潜力。Gau.nernst 和 Vayuda 讨论了 fp5 缺乏进展的情况，以及将 8-bit Adam 与 tensor subclasses 集成的潜在兴趣。

**提到的链接**：[The next tutorials · Issue #426 · pytorch/ao](https://github.com/pytorch/ao/issues/426)：根据我们的 README.md，torchao 是一个用于在 PyTorch 工作流中创建和集成高性能自定义数据类型布局的库。到目前为止，我们在构建原始数据类型方面做得很好...

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1253787640977817662)** (312 条消息🔥🔥):

- **GPU VRAM 限制模型能力**：讨论强调了在 VRAM 有限的 GPU 上加载大型模型（如 **Command R (34b) Q4_K_S**）的局限性，这会导致 token 上下文窗口缩减并阻碍可用性。多位成员建议研究 **EXL2** 等对模型 VRAM 效率更高的替代格式。
- **对服务器设置和 headless 运行的兴趣**：用户表示有兴趣在远程服务器和 headless（无头）设置上运行 **LM Studio**，以实现更好的硬件利用率。建议包括探索用于服务器设置的 **llama.cpp**，并指出 **LM Studio** 目前不支持直接的远程或 headless 操作。
- **以文本到文本为主及模型自定义**：成员们讨论了 **LM Studio** 仅能处理文本到文本交互的局限性，不支持图像生成或文本转语音（TTS）功能。一些用户提到了 **SillyTavern** 等替代前端，但也承认其侧重于 RP/角色扮演，强调了对更多样化选项的需求。
- **优化 P40 GPU 的散热**：分享了关于 GPU 散热的故障排除技巧，特别是针对 P40 GPU。用户指出了充足散热方案的重要性，并分享了制作定制风道以更有效地管理 GPU 温度的经验。
- **探索用于编程的各种语言模型**：讨论涉及寻找用于编程任务的最佳语言模型，提到了 **Codestral 22B** 等模型。成员们强调了模型大小和量化的重要性，建议在特定硬件限制下使用 **Q5** 或 **Q6** 量化版本以获得最佳性能。
  

**提到的链接**：

- [README.md · artificialguybr/ColoringBookRedmond-V2 at main](https://huggingface.co/artificialguybr/ColoringBookRedmond-V2/blob/main/README.md)：未找到描述
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)：本文介绍了 MCT Self-Refine (MCTSr) 算法，这是一种将大语言模型 (LLMs) 与蒙特卡洛树搜索 (MCTS) 创新结合的算法，旨在增强在复杂数学问题中的表现...
- [GitHub: Let’s build from here](https://github.com/)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献、管理 Git 仓库、像专家一样评审代码、跟踪错误和功能...
- [bartowski/Codestral-22B-v0.1-GGUF · Hugging Face](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF)：未找到描述
- [Confused Computing GIF - Confused Computing Counting - Discover & Share GIFs](https://tenor.com/view/confused-computing-counting-math-problems-gif-14678592)：点击查看 GIF
- [configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md)：LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
- [GitHub - theroyallab/tabbyAPI: An OAI compatible exllamav2 API that's both lightweight and fast](https://github.com/theroyallab/tabbyAPI)：一个兼容 OAI 的 exllamav2 API，既轻量又快速 - theroyallab/tabbyAPI

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1253881658881609759)** (116 条消息🔥🔥):

- **Hermes 2 Theta Llama-3 令用户惊叹**：成员们称赞了 **Hermes 2 Theta Llama-3 70B** 模型，因其能够记住高达 19k tokens 的上下文并有效地遵循指令。一位成员分享说，由于其在角色扮演场景中的深度推理和创意能力，这可能是他们目前心目中的顶级模型。[Hermes 2 Theta Llama-3](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k)。
- **DeepSeek Coder V2 受到欢迎**：用户讨论了 **DeepSeek Coder V2** 模型的性能和 Prompt 问题，建议使用特定的 Prompt 预设以避免出现意外的中文输出。一位用户强调了该模型在 C# 编程任务中如何超越了 GPT4o。[DeepSeek Coder V2](https://chat.deepseek.com/coder)。
- **Llama 3 CursedStock 模型引发关注**：成员们对 **Llama 3 CursedStock V1.8-8B** 异常的命名和性能表示好奇和有趣，分享说它通过合并 uncensored 模型而名副其实。此外，还讨论了它在特定故事写作和创意内容生成等利基角色中的表现。[Llama-3 CursedStock V1.8-8B](https://huggingface.co/PJMixers/LLaMa-3-CursedStock-v1.8-8B)。
- **对 LLM 时间感知能力的担忧**：关于 LLM 无法处理需要时间感知（temporal awareness）和因果推理任务的辩论。用户承认当前 AI 的局限性，强调需要专门的硬件来实现真正的通用人工智能（AGI）。
- **量化模型的实验**：用户分享了使用不同量化模型（如 Q6_K_L 和 Q8）的经验，并指出某些版本在处理大上下文大小时存在问题。他们还讨论了保持输出张量（output tensors）和嵌入（embeddings）不进行量化以获得更好性能的潜在好处，特别是对于 **Hathor Fractionate-L3-8B** 模型。[Hathor Fractionate-L3-8B](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05)。
  

**提到的链接**：

- [DeepSeek](https://chat.deepseek.com/coder): 与 DeepSeek AI 聊天。
- [PrunaAI/cognitivecomputations-Dolphin-2.9.1-Phi-3-Kensho-4.5B-GGUF-smashed · Hugging Face](https://huggingface.co/PrunaAI/cognitivecomputations-Dolphin-2.9.1-Phi-3-Kensho-4.5B-GGUF-smashed): 未找到描述
- [cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B · Hugging Face](https://huggingface.co/cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B): 未找到描述
- [mradermacher/New-Dawn-Llama-3-70B-32K-v1.0-GGUF · Hugging Face](https://huggingface.co/mradermacher/New-Dawn-Llama-3-70B-32K-v1.0-GGUF): 未找到描述
- [meta-llama/Meta-Llama-3-8B · Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B): 未找到描述
- [PJMixers/LLaMa-3-CursedStock-v1.8-8B · Hugging Face](https://huggingface.co/PJMixers/LLaMa-3-CursedStock-v1.8-8B?not-for-all-audiences=true): 未找到描述
- [Flash Thumbs Up GIF - Flash Thumbs Up Way To Go - Discover & Share GIFs](https://tenor.com/view/flash-thumbs-up-way-to-go-good-gif-22860767): 点击查看 GIF
- [GitHub - RJ-Flash/AGI-Project: The AGI Project aims to develop an Artificial General Intelligence (AGI) system capable of understanding, learning, and applying knowledge across a wide range of tasks at a level comparable to human intelligence. Our goal is to create a system that can perform any intellectual task that a human being can do, with the ability to learn and adapt.](https://github.com/RJ-Flash/AGI-Project): AGI 项目旨在开发一个通用人工智能（AGI）系统，能够跨广泛任务理解、学习和应用知识，达到与人类智能相当的水平。我们的目标是创建一个能够执行人类可以完成的任何智力任务，并具有学习和适应能力的系统。
- [mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF at main](https://huggingface.co/mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF/tree/main): 未找到描述
- [Nitral-AI/Hathor_Fractionate-L3-8B-v.05 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05): 未找到描述
- [bartowski/Hathor_Stable-L3-8B-v0.5-GGUF · Hugging Face](https://huggingface.co/bartowski/Hathor_Stable-L3-8B-v0.5-GGUF): 未找到描述
- [TheDrummer (Drummer)](https://huggingface.co/TheDrummer/): 未找到描述
- [mradermacher/Halu-8B-Llama3-Blackroot-GGUF · Hugging Face](https://huggingface.co/mradermacher/Halu-8B-Llama3-Blackroot-GGUF/): 未找到描述
- [mradermacher/Mistral-7B-Erebus-v3-i1-GGUF · Hugging Face](https://huggingface.co/mradermacher/Mistral-7B-Erebus-v3-i1-GGUF/): 未找到描述
- [OpenPipe/Hermes-2-Theta-Llama-3-70B-32k · Hugging Face](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k): 未找到描述

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1253883521303314532)** (4 条消息):

- **DeepseekV2 聊天加载问题**：一位用户提到 **deepseekV2** 无法加载进行聊天。另一位用户指出需要 **V0.2.25** 版本，并且“自动更新目前已损坏”。
- **多模型序列提案**：一名成员建议为 **Multi-model** 设置增加一个功能，即“为模型构建序列图”，允许一个模型将信息输入到两个并行模型中，然后再汇总到最终模型。
- **Ubuntu LM Studio 网络错误**：在 Ubuntu 22.04 上使用 **LM Studio** 尝试在 Hugging Face 上搜索模型时出现“网络错误”。然而，该成员指出在 Mac M1 上仍然正常工作，且该问题是在注释掉 AnythingLLM Web 服务器使用的 3001 端口的 ser2net 配置文件后出现的。

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1254859928183636009)** (9 条消息🔥):

- **估算 AI 配置成本难倒用户**：一名成员询问配置一台具有 GPT 或 Bard 性能的机器的预算。回复指出成本极高，可能需要数千美元，具体取决于配置，对于普通用户来说并不现实。
- **NVIDIA DGX GH200 受到关注**：分享了 [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/) 的链接，并指出 OpenAI 正在使用它，其特点是拥有超大内存容量，旨在处理 TB 级模型。另一名成员幽默地评论说，这种配置超出了大多数人的预算。

**提到的链接**：[NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/)：面向新兴 AI 的海量内存超级计算

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1253836060459274280)** (18 条消息🔥):

- **NVlink 的缺失限制了 4000 系列 GPU**：一名成员质疑 4000 系列 GPU 中 NVlink 的缺失是否会阻碍将多个 GPU 用于 AI 目的。他们还询问了是否可以使用 **DX 或 Vulkan 多 GPU 功能**作为替代方案。
- **Proxmox 配置下 Nvidia P40 的性能**：一位用户讨论了他们在运行 Proxmox 和 Debian 的服务器中使用两块 Nvidia P40 的新配置。他们注意到在使用 Codestral 进行全 GPU 卸载（offload）时，功耗显著飙升，达到了 12 tokens/second。
- **ROCm 6.1.3 支持多 GPU**：据分享，AMD 发布了 **ROCm 6.1.3**，现在支持高端 RDNA3 显卡的多 GPU 协作。
- **关于 iPad Pro 16GB RAM 的辩论**：关于 **iPad Pro** 的 16GB RAM 版本对于运行大型 AI 模型是否必要存在争论。一名成员强调量化模型可以装入他们 RTX 4070 Ti Super 的 16GB 显存中，但不确定这是否适用于 Apple 的硬件。
- **Corsair 电源和存储购买咨询**：一位用户询问以 266 欧元购买 **Corsair AX1600i** 以及以 668 欧元购买 4 块 **Exos Enterprise 18TB** 硬盘是否值得，但未收到具体反馈。

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1253935277769228300)** (3 条消息):

- **Llama.cpp 模型加载错误**：一名成员报告了在加载 **Blombert 3B f16 gguf** 模型时出现 **“张量数量错误”**（wrong number of tensors）的问题，错误信息为 `'done_getting_tensors: wrong number of tensors; expected 356, got 291'`。另一名成员建议该错误是由于 **llama.cpp** 版本与 LM Studio 不兼容导致的。
- **上下文长度排障建议**：讨论了 **Blombert 3B** 等大型模型的一个常见问题，将错误归因于上下文长度不匹配。建议的可能解决方案是：*“不断调低上下文长度，直到它不再‘发疯’。”*

---

### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/)** (1 条消息):

cdrivex4: 是的，好。听起来很有趣。

---

### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1254635814118490143)** (1 条消息):

- **Qwen2 500M 模型量化更新**：**Qwen2 500M** 模型的最新量化版本已发布。这些模型针对**快速生成**进行了优化，甚至可以部署在像 **Raspberry Pi** 这样的轻量级计算设备上。在此处探索模型：[here](https://huggingface.co/lmstudio-community/Qwen2-500M-Instruct-GGUF)。

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1254225652538933379)** (12 messages🔥):

- **模型加载问题困扰用户**：一位用户在使用 LMS 配合 batch script 加载模型时遇到困难，但最终获得了成功。他们请求对其 batch script 提供反馈，以检查是否存在错误或优化空间。
- **LMStudio 并非开源**：一位用户询问 LMStudio 是否开源以及是否可以扩展。另一位成员澄清它不是开源的，这导致该用户考虑开发自己的工具来实现所需功能。
- **全能模型运行器的构想**：讨论涉及到了对一个能够运行 Huggingface 上各种模型（包括 text to speech、text to image 等）的程序的渴望。目前尚未发现现有的解决方案，但人们对这类项目表现出了兴趣。

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1253792638432182303)** (276 messages🔥🔥):

- **对 GPT-5 的期待升温**：用户对 OpenAI 延迟推出功能表示沮丧，语音模式和 GPT-4 Vision 被反复提及为逾期未至。一位成员表示：*"事到如今我甚至不在乎它什么时候发布了，发布了我会用，但也就那样吧，当然这只是我个人的看法。"*
- **Siri 与 ChatGPT 集成之争**：关于 ChatGPT 是否集成到 Siri 中存在困惑，一位成员澄清道：“不，它更像是一个额外奖励，并不是那种互相依赖的深度集成”。Elon Musk 对此集成的批评也引发了讨论。
- **Claude 与 ChatGPT 性能对比**：许多用户讨论了 **Claude 3.5 Sonnet** 优于 **GPT-4o** 的表现，尤其是在编程方面，有人说：“同样的事情我在 4o 上试过但失败了，Claude 3.5 成功完成了，甚至做得更多”。基准测试和 Claude 的 “artifacts” 等特定功能被频繁提及作为证据。
- **AI 模型经济学与 Token 限制**：讨论强调了各种 AI 模型的对比，包括 **Claude 的 200k tokens** 对比 **ChatGPT 的 GPT-4 128k** 以及 Plus 用户的 32k。一位用户指出：“Claude 3.5 Sonnet 已登上 LMSYS 排行榜”，强调了实际表现优于纯粹的基准测试。
- **LLM 的持久化用例**：一位用户询问如何创建一个基于个人文档训练的持久化 LLM，问道：“有没有办法让这些 LLM（如 Sonnet 3.5 或 Gemini 1.5 Pro 等）高度专注于特定领域，并作为我个人的工作机器人来使用？”这引发了人们对定制化、长期 AI 应用潜力的极大兴趣。

**提及的链接**：

- [Wired: AI startup Perplexity is 'BS machine'](https://www.youtube.com/watch?v=MFdjEW8_SUg)：Wired 全球编辑总监 Katie Drummond 加入 'Squawk Box'，讨论该杂志对 AI 搜索初创公司 Perplexity 的调查。
- [Computer Stick Man GIF - Computer Stick Man Table Flip - Discover & Share GIFs](https://tenor.com/view/computer-stick-man-table-flip-look-at-you-9gag-gif-26257332)：点击查看 GIF

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1253827661143212032)** (29 messages🔥):

- **GPT-4o 连接问题已解决**：多位用户报告在 GPT-4o 上遇到错误提示：*"连接 worker 时发生错误"*，但在短时间内已恢复。一位用户确认：*"对我来说现在恢复正常了。"*
- **屏幕共享功能尚无 ETA**：一位用户询问屏幕共享功能的可用性，另一位用户回答目前还没有预计到达时间（ETA）。
- **GPT-4o 的 Prompt 遵循问题**：用户讨论了 GPT-4o 无法持续遵循特定 Prompt 格式和指令的问题。例如，尽管明确要求使用 HTML，它仍经常输出 Markdown；并且在处理结构化评审指令时，它会错误地一次性评审整个文档。
- **ChatGPT 性能缓慢及崩溃**：用户在使用 ChatGPT 时经历了性能缓慢和频繁崩溃。一位用户评论道：*"是的，我这里也经常崩溃。"*
- **文档长度与 GPT 上下文窗口限制**：一位拥有 1200 页文档的用户在让 GPT 准确处理内容时遇到困难。另一位用户解释说，ChatGPT 的 context window 不足以处理如此庞大的文档，并推荐使用 Gemini 和 Claude 等具有更大 Token 窗口的工具。

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1253787018446504020)** (53 messages🔥):

- **成员讨论背景移除的限制**：一位成员提到 *DALL-E 仅编辑其自身生成的图像*，而 ChatGPT 虽然提供了一些图像编辑功能（如为任务生成 Python 脚本），但在 *background removal*（背景移除）方面表现挣扎。另一位成员建议尝试使用 *online services* 进行背景移除。
- **热切期待 Sora 发布**：一位用户对 Sora 的发布表示兴奋并询问更新。另一位成员分享说目前还没有时间表，但提供了 [在服务器上生成的 Sora 视频](https://discord.com/channels/974519864045756446/1238697668646010910/1240586938273103894) 的链接。
- **利用 AI 创作奇幻电影情节**：一位成员兴奋地分享了他们使用 ChatGPT 开发的 *fantasy movie ideas*，其中包括对 *The Wizard of Oz* 的重新构思。他们讨论了使用 *DALL-E* 将这些构思视觉化。
- **排除 ChatGPT 功能故障**：用户正在排查 ChatGPT 的 *image background removal* 能力，指出虽然它会尝试进行基础编码，但在处理使用 *Deeplab model* 等更复杂的任务时会遇到 *memory allocation issues*（内存分配问题）。讨论还包括通过调整 *custom instructions* 来修改行为的见解。
- **交互式提示词与优化响应**：一位成员分享了一个用于在预算内组装 PC 的详细交互式提示词，另一位成员则在寻求有关 *cryptocurrency* 相关提示词的建议。此外，人们对改进 MyGPT 提示词以提高响应准确性和可靠性表现出浓厚兴趣，特别是在提取主题和处理上传文件方面。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1253787018446504020)** (53 messages🔥):

- **背景移除：梦想还是现实？**：成员们讨论了让 ChatGPT 执行图像背景移除的尝试。尽管 ChatGPT 生成了尝试执行此操作的脚本，但由于在使用高级机器学习工具时出现内存分配问题，结果并不一致。
- **对 Sora 发布的期待日益增长**：新用户对 Sora 的发布表示兴奋和迫切。一位成员分享了一个 Sora 活动的视频链接，在服务器上引起了一些轰动。
- **DALL-E 与 Midjourney 在艺术作品上的对比**：成员们辩论了 DALL-E 3 与 Midjourney 在创建 AI 图像（尤其是绘画风格图像）方面的效果。个人偏好倾向于 DALL-E 3，因为它具有特定的艺术风格。
- **奇幻电影与提示词编写**：一位用户分享了他们使用 ChatGPT 创作电影构思的经验，特别是对 *The Wizard of Oz* 的重新构思。他们寻求关于完善提示词以生成更准确、更生动图像的建议。
- **交互式 PC 组装提示词**：一位成员展示了一个创意交互式提示词，旨在帮助用户在指定预算内组装 PC，其中结合了对廉价组件的网页搜索，并使用 Python 跟踪项目进度。

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1253789650942951627)** (381 条消息🔥🔥):

- **Wired 抨击 Perplexity 抄袭**：Wired 的一篇文章指责 Perplexity AI “秘密抓取”网站，违反了其自身政策。用户对此进行了讨论，一些人认为考虑到 AI 在数据摘要方面的普遍做法，这种强烈抵制有些过度了 ([来源](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/))。
- **关于 AI 摘要的法律视角**：Redditor 们讨论了 AI 摘要文章不准确以及可能发表诽谤性言论的法律风险。Wired 的一项观察指出，尽管提供了来源链接，Perplexity 的聊天机器人仍错误地将一起犯罪归咎于一名警官 ([存档链接](http://archive.today/GNgAe))。
- **Claude 3.5 Sonnet 推出**：Perplexity Pro 会员注意到最近新增了 Claude 3.5 Sonnet 模型。初步反应称赞了它的能力，但一些用户批评它过于谨慎且限制较多 ([福布斯文章](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/))。
- **用户不满与平台可靠性**：几位用户报告了 Perplexity 的问题，包括 Pro 搜索结果的不一致以及移动端 App 的登录问题。一位用户对 Claude 3.5 Sonnet 的功能和限制级别表示了强烈不满。
- **Pro 搜索与模型使用见解**：讨论揭示了对 Pro 搜索有效性变化和来源限制的挫败感，用户认为 Perplexity 优先考虑合作伙伴关系而非核心改进。一位用户指出，与竞争对手相比，Claude 的 API 订阅提供了更多价值 ([相关视频](https://www.youtube.com/watch?v=iDlM0cYS9Zs))。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/">Why Perplexity’s Cynical Theft Represents Everything That Could Go Wrong With AI</a>：这是这个关键时刻的完美案例研究：AI 的好坏取决于监管它的人。</li><li><a href="https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/">Perplexity Plagiarized Our Story About How Perplexity Is a Bullshit Machine</a>：专家们对于这家 AI 驱动的搜索初创公司的做法是否会使其面临从侵权到诽谤的法律指控意见不一——但有人说原告将会有强有力的证据...</li><li><a href="https://tenor.com/view/just-when-i-thought-i-was-out-they-pull-me-back-in-michael-corleone-al-pacino">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=iDlM0cYS9Zs">I installed Android on Rabbit R1 &amp; Made it Useful</a>：我成功地在 Rabbit R1 上安装了 Android 13，这让设备变得更有用了！让我可以下载 App、发送消息等等。这里是...</li><li><a href="https://x.com/roramora0/status/1804604063922655743">来自 Cubicle e/acc (@roramora0) 的推文</a>：@dwarkesh_sp Dwarkesh，如果你与 Anthropic 有联系，请通知他们，他们的 recaptcha-en.js 文件存在安全漏洞，允许使用 js 代码模拟鼠标动作。这允许...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://tenor.com/view/just-when-i-thought-i-was-out-they-pull-me-back-in-michael-corleone-al-pacino-the-godfather-gif-19249100">Just When I Thought I Was Out They Pull Me Back In GIF</a>：点击查看 GIF</li><li><a href="https://www.goodnewsnetwork.org/robot-mimics-human-sense-of-touch-to-better-sort-through-litter/">Robot Mimics Human Sense of Touch to Better Sort Through Litter</a>：作者解释说，人类的触觉具有多层感官知觉，包括对不同温度感觉的变化。</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo?hl=en">Perplexity - AI Companion</a>：浏览时随时提问</li><li><a href="https://www.goodnewsnetwork.org/robot-mimics-human-sense-of-touc">Robot Mimics Human Sense of Touch to Better Sort Through Litter</a>：作者解释说，人类的触觉具有多层感官知觉，包括对不同温度感觉的变化。</li><li><a href="http://archive.today/GNgAe">Perplexity Plagiarized Our Story About How Perplexity Is a Bullshit M…</a>：未找到描述</li></ul></div>

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1253940127986745426)** (12 条消息🔥):

- **探索 Apple AI 在欧洲推迟**：成员们分享了一个讨论 **Apple 的 AI 功能及其在欧洲地区的限制** 的页面。更多详情请查看 [Apple Intelligence Isn't](https://www.perplexity.ai/page/Apple-Intelligence-Isnt-KJfiVRPEQMmkim0gv7Xh7w)。
- **Perplexity 搜索与学习**：多位成员分享了他们在 Perplexity AI 上的独特搜索，展示了其在学习和信息获取方面的多样化用途。值得关注的搜索包括 [AI 改进](https://www.perplexity.ai/search/AI-Y9Ao26a2SquKulTrvmGfLg#0) 和 [语言探索](https://www.perplexity.ai/search/let-words-spray-vxBv1ca.QbmnB.oMSR.2Jw) 等主题。
- **波音 Starliner 问题**：两位成员重点介绍了 Perplexity AI 上一篇关于 **波音 Starliner 面临挑战** 的文章。通过 [Boeing’s Starliner Stuck](https://www.perplexity.ai/page/Boeings-Starliner-Stuck-lIlR4mleQUK1Q0kahpVwRQ) 阅读更多内容。
- **OpenAI 社区消息**：一条社区消息建议成员确保他们的主题帖是可分享的，以提高社区参与度。在此阅读完整建议 [此处](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **YouTube 教育内容**：Perplexity AI 分享了一个即将发布的 YouTube 视频，暗示了诸如 **Starliner 问题、Apple AI 在欧洲、OpenAI 的收购** 等重要话题。在此观看预告 [此处](https://www.youtube.com/embed/xUsxGDrwzls)。

**提及的链接**：[YouTube](https://www.youtube.com/embed/xUsxGDrwzls)：未找到描述

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1253805803740336249)** (12 条消息🔥):

- **寻找项目创意**：一位用户正在寻求使用 API 构建的 **有趣项目**，以及用于了解 *目前正在做什么以及什么是可能的* 资源。
- **LLama-3-70B API 上下文长度困惑**：一位用户指出，当总 Token 超过约 1642 时会出现 **连接错误**，而另一位用户则报告成功发送了近 3000 个 Token 的请求。怀疑可能是 **审核触发（moderation trigger）或技术问题**。
- **Perplexity 摘要导航超链接**：当要求 Perplexity 通过链接总结网页时，它会通过提供的链接中的超链接进行导航。用户正在寻找一种将摘要限制在初始 URL 的方法。
- **关于 API 中引用时间过滤器的查询**：一位用户询问 API 在线模型是否有 **引用的时间过滤器**，并指出存在一些未公开的请求参数。该用户没有 Beta 测试权限，但已提出申请。

**提及的链接**：[Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)：未找到描述

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1254516705778995200)** (20 messages🔥):

- **Rensa 提升数据集去重效率**：一名成员介绍了 **Rensa**，这是一个由 **Rust 编写并带有 Python 绑定的高性能 MinHash 实现**，展示了 FxHash、LSH 索引和即时排列（on-the-fly permutations）等特性。他们声称其速度比 **datasketch** 快 2.5-3 倍，并在 [GitHub](https://github.com/beowolx/rensa) 上进行了分享。
- **Claude 对《网络寓言》（The Cyberiad）的奇怪反应**：成员们讨论了 AI 模型 Claude 在被问及《网络寓言》时产生十四行诗中断的现象。一位参与者分享了导致此现象的 Prompt，并指出 *"<meta_start>"</meta_start>* 可能是一个 Glitch Token。
- **分享 Glitch Token 研究**：在关于 Claude 行为的讨论中，一名成员分享了关于 Glitch Token 的 **arXiv 论文**以供进一步阅读：[论文 1](https://arxiv.org/pdf/2404.09894) 和 [论文 2](https://arxiv.org/pdf/2405.05417)。
- **Sonnet 在技术话题上的抵触**：一名成员观察到该 AI 模型经常拒绝与技术新闻和 Machine Merging 相关的请求。另一名成员幽默地评论道，模型对 AI 相关问题的敏感度似乎有所提高。
- **对 ChatGPT 论文的批判性观点**：分享了一个对“ChatGPT is bullshit”论文的批评链接，反驳了论文中关于 LLM 产生误导性且对真相漠不关心的输出的观点。该评论可在 [Substack](https://spacedogchronicles.substack.com/p/nothing-is-an-absolute-reality-all?r=2cp9ad) 上阅读。
  

**提到的链接**：

- [Nothing is an absolute reality, all is permitted](https://spacedogchronicles.substack.com/p/nothing-is-an-absolute-reality-all?r=2cp9ad)：机器学习时代的真相是什么？
- [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa)：由 Rust 编写并带有 Python 绑定的高性能 MinHash 实现，用于大规模数据集的高效相似度估计和去重 - beowolx/rensa

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1253821344097898671)** (9 messages🔥):

- **黑客越狱 AI 模型**：分享了一条关于黑客“越狱”强大 AI 模型以突出其缺陷的推文。详细文章可以在[这里](https://on.ft.com/45ByjEj)找到。
- **GitHub 上的 smol q* 实现**：提到了一个 GitHub 仓库 [ganymede](https://github.com/EveryOneIsGross/ganymede)，这是一个“q* 的小型实现”。对于那些对使用 Qwen 0.5B 进行简陋 q* 实现感兴趣的人来说，这是一个资源。
- **用“Claude 玩意儿”制作的游戏**：一名成员分享了他们制作的游戏链接，可在 [Replit](https://replit.com/@0xjzy/SkeletalExpensiveEmbeds) 上运行。
- **字体中的 LLM 推理**：介绍了 [llama.ttf](https://fuglede.github.io/llama.ttf/)，这是一个既是字体文件，又是大语言模型和推理引擎的项目。其原理是使用 HarfBuzz 的 Wasm Shaper 进行字体整形，从而在字体文件内实现复杂的 LLM 功能。
- **mautonomy 分享的推文链接**：分享了一个 Twitter 链接，未提供额外上下文。推文可以在[这里](https://twitter.com/agi2025/status/1798905521334010193?s=19)找到。
  

**提到的链接**：

- [llama.ttf](https://fuglede.github.io/llama.ttf/)：暂无描述
- [来自 Financial Times (@FT) 的推文](https://x.com/FT/status/1804009458613326282)：黑客在全球范围内“越狱”强大的 AI 模型，以突出其缺陷 https://on.ft.com/45ByjEj
- [SkeletalExpensiveEmbeds](https://replit.com/@0xjzy/SkeletalExpensiveEmbeds)：在浏览器中实时运行 Python 代码。使用 Replit（一个强大的 IDE、编译器和解释器）在线编写和运行 50 多种语言的代码。
- [GitHub - EveryOneIsGross/ganymede: smol implementation of q\*](https://github.com/EveryOneIsGross/ganymede)：q* 的小型实现。通过在 GitHub 上创建账号来为 EveryOneIsGross/ganymede 的开发做出贡献。

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296)** (278 条消息🔥🔥):

- **分享了 TheBloke 服务器链接**：一位用户询问 TheBloke 服务器的链接，另一位成员回复了 [Discord 邀请链接](https://discord.gg/VpFvs9cU)。
- **AI 回复中的安全模型**：讨论强调了 **Gemini** 以及可能的 **OpenAI** 中的安全模型会检查回复，并可能对其进行删减或拒绝。一位用户指出：“即便你可以对其进行越狱（jailbreak），如果消息无法逃脱安全过滤，你仍然无法看到它。”
- **Karpathy 的新课程**：一位用户指出了 Karpathy 的新课程 [LLM101n: Let's build a Storyteller](https://github.com/karpathy/LLM101n)，最初将其误认为是 micrograd 仓库。
- **Hermes 2 Pro 70b 格式问题**：用户报告了 Hermes-2-Theta-Llama-3-70B 模型回复以 "<|end_header_id|>" 开头的问题，并被建议改用 llama3 instruct 格式。
- **Replete-Coder 发布**：宣布了一个新模型 Replete-Coder-Qwen2-1.5b，在 100 种编程语言的 HumanEval 测试中获得了 35 分。更多细节在[一条推文](https://x.com/dudeman6790/status/1805108449581072710)中分享。
  

**提到的链接**：

- [PromptIde](https://ide.x.ai/)：未找到描述
- [Dołącz do serwera TheBloke AI na Discordzie!](https://discord.gg/VpFvs9cU)：用于讨论和支持 AI 大语言模型（LLM）以及通用 AI。 | 23728 名成员
- [BigCodeBench Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)：未找到描述
- [Xiaojie Xiaocat GIF - Xiaojie Xiaocat - Discover & Share GIFs](https://tenor.com/view/xiaojie-xiaocat-gif-18274610571282727560)：点击查看 GIF
- [Skeleton Skeletons GIF - Skeleton Skeletons Skull - Discover & Share GIFs](https://tenor.com/view/skeleton-skeletons-skull-skulls-jellymid-gif-25125581)：点击查看 GIF
- [来自 RomboDawg (@dudeman6790) 的推文](https://x.com/dudeman6790/status/1805108449581072710)：发布 Replete-Coder-Qwen2-1.5b，这是一个无审查的 1.5b 模型，在 100 多种编程语言中具有良好的编码性能，提供开源数据、权重、训练代码，并完全可在移动平台上使用...
- [Cool Beans GIF - Cool Beans Thumbsup - Discover & Share GIFs](https://tenor.com/view/cool-beans-thumbsup-gif-13344631)：点击查看 GIF
- [GitHub - karpathy/LLM101n: LLM101n: Let's build a Storyteller](https://github.com/karpathy/LLM101n)：LLM101n: Let's build a Storyteller。通过在 GitHub 上创建账号为 karpathy/LLM101n 的开发做出贡献。
- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1805259592806678699)：今天早上，美国唱片工业协会（RIAA）代表环球、华纳和索尼，对 Suno 和 Udio 提起了版权侵权诉讼。
- [来自 Keyon Vafa (@keyonV) 的推文](https://fxtwitter.com/keyonV/status/1803838591371555252)：新论文：如何判断 Transformer 是否具有正确的模型世界？我们训练了一个 Transformer 来预测纽约市出租车的行驶方向。该模型表现良好，能够找到新路径之间的最短路径...
- [Announcing PromptIDE](https://x.ai/blog/prompt-ide)：未找到描述

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1254053586997088310)** (15 条消息🔥):

- **Tiny Stories 模型以其紧凑的体积令人印象深刻**：讨论集中在最小的 LLM 上，其中一个显著的亮点是 [TinyStories-656K](https://huggingface.co/raincandy-u/TinyStories-656K) 模型，它仅有 600k 个参数。这个轻量级模型能够利用 llama 架构生成连贯的故事。
- **大型模型表现出更优越的性能**：成员们讨论了大型模型的有效性，指出良好的通用性能始于 3B 左右的参数，在 7B-8B 模型中可以看到显著改进。对于顶级性能，70B+ 参数的模型被视为基准。
- **自主 Agent**：关于像 Claude 这样的文本预测器是否有潜力执行与有意识的人类相当的任务存在辩论，一些人断言自主、自我改进的 Agent 已经触手可及。
- **AI 趣闻**：由 Claude 创建的一个幽默的 greentext 故事强调了其创造性文本生成的能力，展示了先进的文本预测能力并娱乐了用户。

**提到的链接**：[raincandy-u/TinyStories-656K · Hugging Face](https://huggingface.co/raincandy-u/TinyStories-656K)：未找到描述

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1254368727064117278)** (12 条消息🔥):

- **在 Google Sheets 中追踪数据集生成**：一位成员分享了一个 [Google Sheet](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0) 用于追踪数据集生成的领域，通过注明兴趣、潜在文档来源和目标规模来鼓励参与。这旨在简化数据集创建流程。
- **Huggingface 聊天模板简化了文档输入**：成员们讨论了通过文档输入字段增强 Huggingface 聊天模板，推广用于标准元数据的 Hermes RAG 格式。这种修改通过使用 jinja 模板和 XML 进行格式化，使得将文档集成到模型输入中变得**极其容易**。
- **AllenAI 引用分类提示词**：分享了一个由 AllenAI 提供的有趣的引用分类提示词，可能对 `academic papers` 类别有用。这个基于 YAML 的提示词有助于将引用分类为 "Background"、"Extends"、"Uses"、"Motivation"、"CompareOrContrast" 以及 "FutureWork"。
- **SciRIFF 数据集**：小组讨论了 [SciRIFF 数据集](https://huggingface.co/datasets/allenai/SciRIFF?row=0)，该数据集包含 13.7 万个指令遵循（instruction-following）演示，用于理解五个领域的科学文献。该数据集附带各种配置以及相应的 [GitHub repo](https://github.com/allenai/SciRIFF) 用于代码、模型训练和评估。
- **Instruction-pretrain 数据集**：一位成员重点介绍了 [ft-instruction-synthesizer-collection](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main)，指出它是完全 RAG 格式的，并认为尽管它主要是多选题而非自由形式，但可能很**有趣**。他们考虑了进行数据增强的可能性，以使该数据集适应各种用途。
  

**提到的链接**：

- [instruction-pretrain/ft-instruction-synthesizer-collection at main](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main)：未找到描述
- [allenai/SciRIFF · Datasets at Hugging Face](https://huggingface.co/datasets/allenai/SciRIFF?row=0)：未找到描述
- [RAG Data Synthesis](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0)：Sheet1 领域、课程文件、来源/链接、HF repo、规模（行数）、状态、负责人、审核人、审核笔记、网络搜索、Wikipedia、代码库、学术论文、书籍、金融、SEC 备案等，1000，Agent s...

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 条消息):

teknium: [https://twitter.com/hamish_kerr/status/1804352352511836403](https://twitter.com/hamish_kerr/status/1804352352511836403)

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1253814084449734792)** (114 条消息🔥🔥):

- **SLURM 节点问题**：一位用户报告通过 Jupyter Notebook 连接到由 SLURM 管理的节点时，在训练阶段遇到错误，可能是由于 SLURM 限制导致的。他们提到在控制台测试时，尽管正确指定了 GPU 使用，但在开始训练前收到了 "kill" 消息。
- **PyTorch 加速 Llama-2**：PyTorch 团队发布了将 Llama-2 推理速度提高 10 倍的技术，并在[博客文章](https://pytorch.org/blog/accelerating-generative-ai-2/)中分享。一位用户开发了一个 pip 包 [GPTFast](https://github.com/MDK8888/GPTFast)，将这些技术应用于所有 HF 模型，并寻求 A100 或 H100 GPU 集群的使用权限。
- **开源 AI 模型问题**：围绕在官方渠道之外分享 Mistral 等专有 AI 模型的伦理和实用性展开了讨论。用户强调了此类行为的法律和道德影响，并强调了 AI 开发中问责制和透明度的必要性。
- **模型延迟分析**：用户讨论了确定 AI 模型是 GPT-4 还是其他变体的方法，建议包括检查知识截止日期（knowledge cutoffs）和分析延迟差异。还提出了通过嗅探网络流量来识别 API 调用中使用的模型。
- **LingOly 基准测试讨论**：讨论了一个名为 LingOly 的新基准测试，该测试通过低资源语言的语言谜题评估 LLM 的高级推理能力。该基准测试包含 1,133 个问题，顶级模型的准确率低于 50%，因其挑战性和潜在的记忆化（memorization）问题而受到关注。
  

**提到的链接**：

- [LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages](https://arxiv.org/abs/2406.06196v2)：在本文中，我们介绍了 LingOly 基准测试，这是一个用于评估 LLM 高级推理能力的新型基准测试。通过具有挑战性的语言奥林匹克谜题，我们评估了 (i) 能力...
- [Virus Computer GIF - Virus Computer Hello Your Computer Has Virus - Discover & Share GIFs](https://tenor.com/view/virus-computer-hello-your-computer-has-virus-meme-memes-gif-20233783)：点击查看 GIF
- [examples/examples/benchmarks/bert at main · mosaicml/examples](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert)：快速且灵活的参考基准测试。通过在 GitHub 上创建账户为 mosaicml/examples 的开发做出贡献。
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/.)：这篇文章是关于如何使用纯原生 PyTorch 加速生成式 AI 模型的系列博客的第二部分。我们很高兴能分享广泛新发布的 PyTorch 性能...
- [GitHub - AnswerDotAI/bert24](https://github.com/AnswerDotAI/bert24)：通过在 GitHub 上创建账户为 AnswerDotAI/bert24 的开发做出贡献。

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1253799610372329492)** (155 条消息🔥🔥):

- **TTS 论文引入 ARDiT**：关于[一篇新 TTS 论文](https://arxiv.org/abs/2406.05551)的讨论，强调了 ARDiT 在零样本（zero-shot）文本转语音中的潜力。一位成员评论道：“有很多想法可以应用到其他地方。”
- **探索多目标损失**：关于在神经网络训练中强制执行帕累托改进（Pareto improvements）的激烈辩论，重点关注多维目标。一位成员[分享了关于多目标优化的见解](https://en.wikipedia.org/wiki/Multi-objective_optimization)，另一位成员总结道：“可能必须选择一小部分权重（例如范数权重和偏置），这些权重在不同的帕累托版本之间变化，并共享其余部分。”
- **优化中的平方投票法**：引用[平方投票法](https://en.wikipedia.org/wiki/Quadratic_voting)（quadratic voting）作为平衡竞争性人类价值观并将其整合到多目标优化中的方法。对话围绕在机器学习模型中使用平方投票法的可行性和影响展开。
- **多任务学习中的争议**：一位成员推荐了一篇论文，该论文揭示了专门的多任务优化方法相比传统方法没有显著优势（[点击此处阅读](https://arxiv.org/abs/2209.11379)）。另一位成员[强调了一项后续研究](https://arxiv.org/abs/2312.06134)，讨论了数据不平衡任务集合中的优化动态。
- **AE 中的潜空间正则化**：一个线程讨论了如何在自动编码器（autoencoder）嵌入中引入噪声，建议直接向编码输出添加高斯噪声。成员们就正则化和批归一化（batch normalization）的必要性进行了辩论，以防止嵌入无限制地缩放。
  

**提到的链接**：

- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366): Maximum Manifold Capacity Representations (MMCR) 是一种最近的多视图自监督学习 (MVSSL) 方法，其表现与其它领先的 MVSSL 方法相当或更优。MMCR 引起关注的原因在于它...
- [HyperZ$\\cdot$Z$\\cdot$W Operator Connects Slow-Fast Networks for Full Context Interaction](https://arxiv.org/abs/2401.17948): Self-attention 机制利用大型隐式权重矩阵，通过基于点积的激活函数（仅包含极少数可训练参数）进行编程，从而实现长序列建模。在本文中...
- [Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546): 解决大型语言模型 (LLMs) 安全风险的一种方法是从其训练数据中审查危险知识。虽然这删除了显式信息，但隐式信息仍可能保留...
- [Quadratic voting - Wikipedia](https://en.wikipedia.org/wiki/Quadratic_voting): 未找到描述
- [4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities](https://arxiv.org/abs/2406.09406): 目前的多模态和多任务基础模型（如 4M 或 UnifiedIO）展示了令人期待的结果，但在实践中，它们接受多样化输入和执行多样化任务的开箱即用能力受到限制...
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782): 虽然深度学习和深度强化学习 (RL) 系统在图像分类、游戏博弈和机器人控制等领域展示了令人印象深刻的结果，但数据效率仍然是一个...
- [Multi-objective optimization - Wikipedia](https://en.wikipedia.org/wiki/Multi-objective_optimization): 未找到描述
- [VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models](https://arxiv.org/abs/2406.13362): 随着最近大型语言模型的成功，视觉语言模型 (VLMs) 得到了快速发展。然而，目前很少有尝试将高效的线性 Recurrent Neural Networks 结合进来...
- [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399): Transformers 在算术任务上的糟糕表现似乎在很大程度上源于它们无法跟踪长数字串中每个数字的准确位置。我们修复了...
- [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/abs/2010.05874): 包含数十甚至数百种语言的大规模多语言模型给多任务优化带来了巨大挑战。虽然应用一种语言无关的优化程序是通用做法...
- [Toward Infinite-Long Prefix in Transformer](https://arxiv.org/abs/2406.14036v1): 提示 (Prompting) 和基于上下文的微调方法（我们称之为 Prefix Learning）已被提出，用于增强语言模型在各种下游任务上的性能，其效果可与全参数...
- [Order Matters in the Presence of Dataset Imbalance for Multilingual Learning](https://arxiv.org/abs/2312.06134): 在本文中，我们实证研究了多任务学习的优化动态，特别关注那些主导具有显著数据不平衡的任务集的动态。我们提出了一个简单的...
- [Tweet from François Fleuret (@francoisfleuret)](https://x.com/francoisfleuret/status/1804873919653957733): 一份小报告！
- [Do Current Multi-Task Optimization Methods in Deep Learning Even Help?](https://arxiv.org/abs/2209.11379): 最近的研究为深度多任务模型提出了一系列专门的优化算法。人们通常声称这些多任务优化 (MTO) 方法产生的解决方案优于...
- [Autoregressive Diffusion Transformer for Text-to-Speech Synthesis](https://arxiv.org/abs/2406.05551): 音频语言模型最近成为各种音频生成任务的一种有前景的方法，它依赖音频分词器 (audio tokenizers) 将波形编码为离散符号序列。音频分词器...
- [Grokking of Hierarchical Structure in Vanilla Transformers](https://arxiv.org/abs/2305.18741): 对于人类来说，语言的产生和理解对句子的层级结构非常敏感。在自然语言处理领域，过去的研究质疑了神经序列模型在多大程度上能有效...
- [Why Momentum Really Works](https://distill.pub/2017/momentum/): 我们通常将带有动量 (momentum) 的优化想象成一个球滚下山坡。这并没有错，但故事远不止于此。
- [no title found](https://aligniverse.streamlit.app/): 未找到描述

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1254263108042489967)** (10 messages🔥):

- **Epoch 重新探讨机器学习中的算力权衡**：成员们讨论了 [Epoch AI 的博客文章](https://epochai.org/blog/trading-off-compute-in-training-and-inference#monte-carlo-tree-search)，该文章探讨了如何平衡训练和推理过程中的算力。一位成员指出：*“将推理算力提高 1-2 个数量级，有可能节省约 1 个 OOM 的训练算力。”*
- **关于 Neural Redshifts 的论文引起关注**：成员们分享了 [一篇关于 Neural Redshifts 的论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.pdf)，并指出初始化可能比研究人员通常认为的更为重要。有人评论道：*“初始化比研究人员所认为的要有趣得多。”*
- **AI 公案（Koans）引发笑声与启迪**：一段关于 AI 公案的幽默交流被分享，并链接到了一个 [黑客笑话集](http://www.catb.org/esr/jargon/html/koans.html)。插图包含了一个关于新手和资深黑客的轶事，展示了*“重启试试”*如何出人意料地解决问题。
  

**提到的链接**：

- [Trading Off Compute in Training and Inference](https://epochai.org/blog/trading-off-compute-in-training-and-inference#monte-carlo-tree-search)：我们探索了几种在训练或推理上投入更多资源之间进行权衡的技术，并描述了这种权衡的特性。我们概述了对 AI 发展的一些影响...
- [Some AI Koans](http://www.catb.org/esr/jargon/html/koans.html)：未找到描述

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1253931388760232018)** (3 messages):

- **播客中探讨使用 SAEs 进行模型编辑**：一位成员引用了一段 [播客视频](https://youtu.be/lj2y5hE04XI?t=4585)，讨论了使用 **SAEs** 进行模型编辑的潜力，特别是使用 **MEMIT 论文**中非精选（non-cherrypicked）的编辑列表来评估有效性。他们链接了 [MEMIT 论文](https://arxiv.org/pdf/2210.07229.pdf) 及其 [源代码](https://github.com/kmeng01/memit) 以供进一步探索。
- **对字典学习（dictionary learning）实证评估的兴趣**：一位成员询问是否有推荐的论文，对受 **dictionary learning** 发现的特征影响的模型行为进行实证评估。这表明研究重点在于通过结构化特征操纵来理解模型引导（model steering）的实证方法。
  

**提到的链接**：

- [Ep 14 - Interp, latent robustness, RLHF limitations w/ Stephen Casper (PhD AI researcher, MIT)](https://youtu.be/lj2y5hE04XI?t=4585)：我们采访了 Stephen Casper，朋友们都叫他 “Cas”。Cas 是 MIT 计算机科学（EECS）系的博士生，就职于 Algorithmic Ali...
- [Mass Editing Memory in a Transformer](https://memit.baulab.info/)：通过直接计算参数变化，在 GPT 中更新数千条记忆。

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1254766616986783835)** (6 messages):

- **本地模型注册简化**：一位用户询问是否可以在不修改 `lm_eval/models/__init__.py` 的情况下在本地注册模型。另一位用户解释了 `register_model` 的用法，并提供了一个代码片段，展示了如何通过包装模块（wrapper module）来实现这一目标。
- **指出 Commit 中的破坏性变更**：一个添加了 `tokenizer logs info` 的提交无意中破坏了主分支。用户指出了错误的导入路径问题，并请求进行热修复（hotfix）。
- **请求并应用热修复**：另一位用户引导大家关注一个提议的 [hotfix](https://github.com/EleutherAI/lm-evaluation-harness/pull/2015)，并请人进行测试。确认后，他们承认该修复解决了问题。

**提到的链接**：[add tokenizer logs info (#1731) · EleutherAI/lm-evaluation-harness@536691d](https://github.com/EleutherAI/lm-evaluation-harness/commit/536691da2444bd35b76d3f9c9527126273a63251)：\* add tokenizer logs info

- 增加无 tokenizer 的情况
- 更新 lm_eval/logging_utils.py

Co-authored-by: Hailey Schoelkopf &lt;[65563625+haileyschoelkopf@users.noreply.github.com](mailto:65563625+haileyschoelkopf@users.noreply.github.com)&gt;

- U...

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1254207280333983855)** (2 条消息):

- **关于最佳多模态 LLM 架构的辩论**：一名成员质疑像 Chameleon 这样的早期融合（early fusion）模型是否优于在将图像输入 LLM 上下文之前使用视觉编码器的方法。他们担心每种方法可能并不在所有任务中都具有绝对优势，而是取决于具体任务。
- **早期融合中的视觉敏锐度权衡**：他们指出早期融合可能在通用性方面表现更好；然而，据闻该模型在视觉敏锐度（visual acuity）方面存在困难。这是由于图像 Tokenization 过程压缩了图像信息，导致与使用视觉编码器的 patch embedding 相比，清晰度有所损失。

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1254196252451803168)** (3 条消息):

- **Intel 撤回 AWS 实例，考虑替代方案**：*“Intel 正在撤回我们的 AWS 实例，所以我认为我们要么为这些实例支付少量费用，要么切换到手动触发的免费 GitHub runners。”* 目前尚未提到最终决定。
- **A100 GPU 上的 NCCL 后端问题**：尝试在内部 **A100 GPU** 上使用 **gpt-neox** 训练模型时遇到了 NCCL 后端问题。该问题在不同版本的 **NCCL** 和 **CUDA** 中持续存在，无论是否使用 Docker。

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1253822664632438887)** (133 条消息🔥🔥):

- **Noam Shazeer 谈论 Character.AI 的推理优化**：Noam Shazeer 的一篇新[博客文章](https://research.character.ai/optimizing-inference/)讨论了 Character.AI 如何通过优化推理过程来实现 AGI。文章强调了他们在处理每秒超过 20,000 次推理查询方面所做的努力。
- **OpenAI 收购 Rockset**：[OpenAI 已收购 Rockset](https://x.com/deedydas/status/1804185430897889427) 以增强其检索增强生成 (RAG) 能力。Rockset 成立于 2016 年，其团队在构建向量搜索 (FAISS) 和关键词搜索等混合搜索解决方案方面拥有深厚的专业知识。
- **Karpathy 宣布新课程**：[Karpathy 计划推出一门雄心勃勃的 “LLM101n” 课程](https://x.com/miru_why/status/1804205538798182780?s=46&t=90xQ8sGy63D2OtiaoGJuww)，旨在从零开始构建类似 ChatGPT 的模型，类似于他著名的 CS231n 课程。
- **回应 LangChain 融资争议**：[LangChain 的 Harrison Chase](https://x.com/hwchase17/status/1804166140773691837) 澄清说，他们的资金完全专注于产品开发，而不是赞助活动或广告，以此回应有关其使用风险投资资金的批评。
- **Mira Murati 暗示 GPTnext**：Mira Murati 暗示下一个主要的 GPT 模型可能会在 [1.5 年内发布](https://www.youtube.com/watch?v=yUoj9B8OpR8)，并讨论了 AI 工具为各个领域的创造力和效率带来的巨大变革。
  

**提到的链接**：

- [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news)：未找到描述
- [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-new)：未找到描述
- [Character.AI 的 AI 推理优化](https://research.character.ai/optimizing-inference/)：在 Character.AI，我们正致力于实现 AGI。在未来的状态中，大语言模型 (LLMs) 将增强日常生活，提供商业生产力和娱乐，并帮助人们……
- [Multi 博客 – Multi 加入 OpenAI](https://multi.app/blog/multi-is-joining-openai)：最近，我们越来越多地问自己应该如何与计算机协作。不是在计算机上或使用计算机，而是真正地与计算机协作。与 AI 协作。我们认为这是最重要的……
- [Olympia | 优于 ChatGPT](https://olympia.chat)：通过价格合理的 AI 驱动顾问来发展您的业务，这些顾问是业务战略、内容开发、营销、编程、法律战略等方面的专家。
- [Hamel Husain (@HamelHusain) 的推文](https://x.com/HamelHusain/status/1804301841666314501)：在我在实际场景中遇到的大多数案例中，“AI Engineer” 这个头衔是有害的。我在下面的视频中解释了原因。引用 Hugo Bowne-Anderson (@hugobowne) 的《AI Engineer 数据素养鸿沟》🎙……
- [AI 无处不在：改变我们的世界，赋能人类](https://www.youtube.com/watch?v=yUoj9B8OpR8)：达特茅斯工程学院举办了一场与校友、OpenAI 首席技术官 Mira Murati 的独家对话。她讨论了人工智能……
- [Sully (@SullyOmarr) 的推文](https://x.com/sullyomarr/status/1803779798658859067?s=46)：介绍 Otto —— 一种与 AI Agents 交互和协作的新方式 —— 使用表格！现在你可以让数百个 Agent 同时为你工作。

- [来自 Harrison Chase (@hwchase17) 的推文](https://x.com/hwchase17/status/1804166140773691837): @levelsio 我们所有的资金都投入到了核心团队中，用于帮助构建 LangChain、LangSmith 以及其他相关产品。我们确实有一项政策，即不提供金钱赞助活动，更不用说...
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1804899734475374857): Artifacts 专业技巧：如果你在使用 NPM 模块时遇到不支持的库错误，只需让 Claude 改用 cdnjs 链接，它应该就能正常工作。
- [来自 nano (@nanulled) 的推文](https://x.com/nanulled/status/1804638164923167057): 100倍检查的数据训练，然后... 它真的成功了，而且确实能对模式进行推理。我简直不敢相信。
- [来自 Morten Just (@mortenjust) 的推文](https://x.com/mortenjust/status/1805190952358650251?s=46&t=90xQ8sGy63D2OtiaoGJuww): 速度很快。Chrome 在我的笔记本上本地运行 Gemini。只需 2 行代码。
- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/andrewcurran_/status/1805259592806678699?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天早上，RIAA 代表 Universal、Warner 和 Sony 对 Suno 和 Udio 提起了版权侵权诉讼。
- [来自 Ammaar Reshi (@ammaar) 的推文](https://x.com/ammaar/status/1803914672091074726): 带有 Artifacts 的 Claude Sonnet 3.5 还可以播放声音！利用 @elevenlabs API，它创建了一个功能齐全的 AI 音效生成器应用，我所做的只是粘贴了 API 文档。我惊呆了...
- [未找到标题](https://news.ycombinator.com/item?id=40739982): 未找到描述
- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/andrewcurran_/status/1805259592806678699?s=46&t=90xQ8sGy63D2Oti): 今天早上，RIAA 代表 Universal、Warner 和 Sony 对 Suno 和 Udio 提起了版权侵权诉讼。
- [来自 Morten Just (@mortenjust) 的推文](https://x.com/mortenjust/status/1805190952358650251?s=46&t=90xQ8s): 速度很快。Chrome 在我的笔记本上本地运行 Gemini。只需 2 行代码。
- [来自 TestingCatalog News 🗞 (@testingcatalog) 的推文](https://x.com/testingcatalog/status/1805288828938195319?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 它要来了 🔥
- [来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1805226216997200145?s=46&t=90xQ8sGy63D2OtiaoGJuww): 等等，什么？已经有人从 Chrome 中提取了 Gemini Nano 的权重并分享到了 Hub 上 ⚡ > 看起来是运行在 tf-lite 上的 4-bit 版本 (?) > 基础 + 指令微调适配器。义务性披露...
- [来自 Deedy (@deedydas) 的推文](https://x.com/deedydas/status/1804185430897889427): OpenAI 刚刚收购了 Rockset 以增强 RAG 能力。Rockset 由前 Facebook 团队于 2016 年创立，该团队曾构建了 RocksDB（Google LevelDB 的一个分支，由 Jeff Dean 本人编写的可嵌入式 NoSQL 数据库）。...
- [来自 Bilawal Sidhu (@bilawalsidhu) 的推文](https://x.com/bilawalsidhu/status/1804255144835457503?s=46&t=90xQ8sGy63D2OtiaoGJuww): 哇。Stability AI 的新任 CEO 是 Prem Akkaraju，他是传奇视觉特效工作室 Weta Digital 的前 CEO。SVD 本可以与 Runway/Luma 竞争，但他们错失了机会。事实上，Luma AI 已经...
- [来自 Bilawal Sidhu (@bilawalsidhu) 的推文](https://x.com/bilawalsidhu/status/1804255144835457503?s=46&t=90xQ8s): 哇。Stability AI 的新任 CEO 是 Prem Akkaraju，他是传奇视觉特效工作室 Weta Digital 的前 CEO。SVD 本可以与 Runway/Luma 竞争，但他们错失了机会。事实上，Luma AI 已经...
- [来自 miru (@miru_why) 的推文](https://x.com/miru_why/status/1804205538798182780?s=46&t=90xQ8sGy63D2OtiaoGJuww): 看起来 @karpathy 正在筹划一门类似于 cs231n 的完整课程 “LLM101n”，涵盖如何从零开始构建类似 ChatGPT 的模型 https://github.com/karpathy/LLM101n。非常有野心！
- [来自 Robert Graham 𝕏 (@ErrataRob) 的推文](https://x.com/erratarob/status/1804018865145315529?s=46&t=90xQ8sGy63D2OtiaoGJuww): NVIDIA 正处于 Sun Microsystems 在互联网泡沫早期的地位。Sun 拥有领先的 Web 服务器、最聪明的工程师以及行业内最高的声望。如果你...
- [来自 jason liu (@jxnlco) 的推文](https://x.com/jxnlco/status/1804601597353226738): 这看起来像是编造的。如果你构建过 MLE 系统。我不相信 Chaining 和 Agent 不仅仅是一个 Pipeline。MLE 从未构建过容错系统？
- [来自 Mira Murati (@miramurati) 的推文](https://x.com/miramurati/status/1804567253578662264?s=46&t=90xQ8sGy63D2OtiaoGJuww): 在 OpenAI，我们致力于推进科学理解，以帮助改善人类福祉。我们正在构建的 AI 工具，如 Sora, GPT-4o, DALL·E 和 ChatGPT，从技术角度来看令人印象深刻...
- [来自 Andrej Karpathy (@karpathy) 的推文](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 本周在旧金山举行的 @aiDotEngineer 世界博览会 🔥 https://www.ai.engineer/worldsfair。想起了我最近一次演讲的第一张幻灯片：“以防万一你还在好奇……不，这不正常……”

- [GitHub - admineral/Reactor: 早期 Alpha 版：使用 Codesandbox 的 Sandpack 进行 React 代码编辑和实时预览对话。Vercel ai SDK RSC GenUI](https://github.com/admineral/Reactor): 早期 Alpha 版：使用 Codesandbox 的 Sandpack 进行 React 代码编辑和实时预览对话。Vercel ai SDK RSC GenUI - admineral/Reactor
- [Reddit - 深入探讨任何事物](https://www.reddit.com/r/LocalLLaMA/comments/1dmt6oy/two_quotes_from_anthropics_product_lead_on_claude/): 未找到描述
- [GitHub - beowolx/rensa: Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大规模数据集的高效相似度估算和去重](https://github.com/beowolx/rensa): Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大规模数据集的高效相似度估算和去重 - beowolx/rensa

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1253830564629184625)** (3 条消息):

- **关于招聘 AI 工程师的新播客发布！**：Latent Space Podcast 发布了名为“如何招聘 AI 工程师”的新剧集，包含来自 @james_elicit 和 @*adamwiggins* 的客座文章和奖励播客。该剧集涵盖了一系列主题，包括“定义招聘流程”、“防御性 AI 工程”以及“防御性 AI 工程的技术选择”[详情点击此处](https://x.com/latentspacepod/status/1804269727482810386)。
- **播客也登上了 Hacker News**：除了直接链接外，还提到该播客正在 [Hacker News](https://news.ycombinator.com/) 上进行讨论。未提供更多细节。

**提到的链接**：[来自 Latent Space Podcast (@latentspacepod) 的推文](https://x.com/latentspacepod/status/1804269727482810386): 🆕如何招聘 AI 工程师，来自 @james_elicit 和 @*adamwiggins* 的罕见客座文章（及奖励播客）！涵盖：- 定义招聘流程 - 作为混沌媒介的防御性 AI 工程 - 技术选择...

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1253801612745506840)** (72 条消息🔥🔥):

- **录制权限待 World's Fair 之后确定**：一位成员询问另一位成员是否可以录制会议，并承诺在 World's Fair 之后再上传。对方通过大拇指表情符号表示同意。
- **开发 Twitter 管理应用**：一位成员讨论了使用 Twitter API 为 Twitter 管理应用创建基于 YAML 的 DSL，旨在生成更好的社交帖子分析。他们就添加更多功能的重要性寻求反馈，并分享了详细的 YAML 代码段。
- **参考 Zoho Social 获取灵感**：一位成员建议参考 Zoho Social 的功能来构建 Twitter 分析应用。他们提供了一个 [Zoho Social 链接](https://www.zoho.com/social/features.html)，详细介绍了调度、监控和分析社交媒体帖子等各种功能。
- **Anthropic 的 XML 标签建议**：提到 Anthropic 建议在某些功能中使用 XML 标签，并链接到了相关的[文档](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)。
- **LLM 生成 YAML 项目的成功案例**：随后讨论了 LLM 在生成基于 YAML 的项目中的实用性，一位成员分享了他们使用 LLM 在 Go 中创建 YAML 模板语言实现的经验，并指向了他们的 [GitHub 仓库](https://github.com/go-go-golems/go-emrichen)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://gist.github.com/wesen/9b2baa6cf5ed4a137adccd3e7c70c41c">ai-workshop.md</a>: GitHub Gist: 立即分享代码、笔记和片段。</li><li><a href="https://www.zoho.com/social/features.html">Zoho Social - 功能</a>: Zoho Social 的功能将告诉你为什么它是目前你能买到的最好的社交媒体营销软件。</li><li><a href="https://github.com/go-go-golems/go-emrichen">GitHub - go-go-golems/go-emrichen: Go 语言实现的 YAML 模板语言 emrichen</a>: Go 语言实现的 YAML 模板语言 emrichen - go-go-golems/go-emrichen</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1253826170680643594)** (62 条消息🔥🔥):

- **估算 LLVM 的成本**：Curiosity.fan 分享了一篇[估算 LLVM 成本的文章](https://chriscummins.cc/2019/llvm-cost/)，该文章得出结论，1200 名开发者产出了 690 万行代码，估算成本为 5.3 亿美元。讨论内容包括克隆和检出 LLVM 项目以了解其开发成本。
- **Mojo 安装问题**：Darinsimmons 分享了他在全新安装的 22.04 系统和 Mojo 的 nightly 版本中遇到的挫折，表示包括 blog 2406 在内的所有 devrel-extras 测试均未通过。他计划暂时离开电脑休息一下，稍后再解决这个问题。
- **关于 LLVM 和 Mojo 的互动讨论**：[EuroLLVM 2024 演讲](https://youtu.be/y85-1g39X3E?si=N7ZEMxJgWBBwD22x)等视频增强了人们对 LLVM 和 Mojo 的兴趣，用户表达了他们的热情，并计划深入研究 MLIR 和 LLDB 扩展。
- **文档导航困惑**：用户讨论了由于 Mojo 的 nightly 和 stable 文档之间缺乏清晰区分而导致的困惑。有人建议为 stable 和 nightly 版本维护独立的文档集，以提高清晰度。
- **对 Mojo Stencil 操作的好奇**：Benny.n 表现出对探索 Mojo 算法库中 `stencil` 函数的兴趣，推测其用于降低维度。他还计划重新实现 autotune 功能，使超参数评估在编译时更加高效。
  

**提到的链接**：

- [2024 EuroLLVM - How Slow is MLIR](https://www.youtube.com/watch?v=7qvVMUSxqz4)：2024 年欧洲 LLVM 开发者大会。演讲者：Mehdi Amini, Jeff Niu。幻灯片：https://llvm.org/devm...
- [stencil | Modular Docs](https://docs.modular.com/mojo/stdlib/algorithm/functional/stencil)：stencilrank Int, stencilaxis Int, type fn(StaticIntTuple[$1]) capturing -> Tuple[StaticIntTuple[$1], StaticIntTuple[$1]], mapstrides Int) capturing -> Int, loadfn fn[Int capturing -> SIMD$4, ...
- [2024 EuroLLVM - Mojo debugging: extending MLIR and LLDB](https://youtu.be/y85-1g39X3E?si=N7ZEMxJgWBBwD22x)：2024 年欧洲 LLVM 开发者大会。Mojo 调试：扩展 MLIR 和 LLDB。演讲者：Walter Erquinigo, Billy Zhu。
- [2024 EuroLLVM - Efficient Data-Flow Analysis on Region-Based Control Flow in MLIR](https://www.youtube.com/watch?v=vvVR3FyU9TE)：2024 年欧洲 LLVM 开发者大会。MLIR 中基于 Region 的控制流的高效数据流分析。演讲者：Weiwei ...
- [Estimating the Dollar Cost of LLVM](https://chriscummins.cc/2019/llvm-cost/)：全职极客和研究生的个人网站，热衷于开发优秀的软件。
- [mojo/examples/reduce.mojo at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/examples/reduce.mojo)：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 开发做出贡献。
- [mojo/docs/changelog.md at 1b79ef249f52163b0bafbd10c1925bfc81ea1cb3 · modularml/mojo](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25)：Mojo 编程语言更新日志。

---

### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1254604078236045398)** (1 条消息):

- **Modular 发布新视频**：**Modular** 刚刚发布了一个新的 [YouTube 视频](https://www.youtube.com/watch?v=AEnvEpQm9zg)，标题为 " - YouTube"。该视频的描述目前未定义。

**提到的链接**：[\- YouTube](https://www.youtube.com/watch?v=AEnvEpQm9zg)：未找到描述。

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1253804704555733055)** (5 条消息):

- **构建一个新的数据标注平台**：一位成员就构建一种不同类型的数据标注平台征求反馈，询问了最常见的数据标注类型、使用的方法、痛点、人工干预以及自动化解决方案的潜在成本。
- **产品图像标注的痛点**：一位成员讨论了产品图像和元数据的标注，强调了歧义性和所需手动工作量等痛点。他们表示，如果自动化产品具有成本效益且可靠，他们愿意使用。
- **PDF 的手动标注**：另一位成员分享了他们在 PDF 手动数据标注方面的经验，并提到尝试为自动化而微调模型。他们强调 [Haystack](https://haystack.deepset.ai/) 是他们探索过的一种工具，并强调了 PDF 数据提取和标注准确性的重要性，特别是对于 ERP 集成而言。
- **对 ERP 集成的兴趣**：原作者对反馈表示感谢，并注意到将他们的标注平台与 ERP 系统集成的可能性，这是受关于 quickbooks 和手动数据录入的见解启发而产生的。

**提到的链接**：[Haystack | Haystack](https://haystack.deepset.ai/)：Haystack，可组合的开源 AI 框架

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1253795498612097115)** (51 条消息🔥):

- **CONTRIBUTING.md 缺少测试说明**：一位用户注意到 Mojo 仓库中的 `CONTRIBUTING.md` 文件没有说明如何在提交 PR 之前运行所有测试。他们建议添加这些说明，并在此处链接了相关文档 [here](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md)。
- **Mojo 的 control-flow.ipynb 出现错误**：一位用户报告在运行 `control-flow.ipynb` 中的代码片段时出现 SIGSEGV 错误。另一位用户无法复现该问题，并建议更新到最新的 nightly 版本并更改类型作为可能的修复方案。
- **Mojo 的 staticmethod.ipynb 出现问题**：有报告称在 `staticmethod.ipynb` 中涉及从值中销毁字段时出现错误。尽管进行了更新，问题仍然存在，导致用户考虑提交 GitHub issue 以寻求进一步帮助。
- **提供 OpenAI API key 以寻求帮助**：一位遇到关键问题的用户提供了一个价值 10 美元的 OpenAI API key 作为奖励，以激励他人帮助解决问题，这凸显了社区精神和问题的紧迫性。他们强调该问题具有阻塞性，并提供了 GitHub issue [链接](https://github.com/modularml/mojo/issues/3102)。
- **Mojo 的开发和 Docker 支持**：讨论包括在 dev containers 中运行 Mojo 的设置，并附带了示例项目的链接，如 [benz0li/mojo-dev-container](https://github.com/benz0li/mojo-dev-container) 以及官方的 modular Docker 容器示例 [此处](https://github.com/modularml/mojo/tree/main/examples/docker)。用户分享了他们对这些环境的偏好和经验。

**提到的链接**：

- [YouTube](https://www.youtube.com/watch?v=)：未找到描述
- [2024 EuroLLVM - Mojo debugging: extending MLIR and LLDB](https://www.youtube.com/watch?v=y85-1g39X3E)：2024 年欧洲 LLVM 开发者会议。演讲者：Walter Erquinigo, Billy Zhu —— Mojo 调试：扩展 MLIR 和 LLDB。
- [thatstoasty - Overview](https://github.com/thatstoasty)：thatstoasty 有 19 个公开仓库。在 GitHub 上关注他们的代码。
- [mojo/stdlib/docs/development.md at main · modularml/mojo](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
- [mojo/examples/docker at main · modularml/mojo](https://github.com/modularml/mojo/tree/main/examples/docker)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
- [[BUG] LSP & Mojo crashes when using Python.evaluate in a certain way · Issue #3102 · modularml/mojo](https://github.com/modularml/mojo/issues/3102)：Bug 描述：以某种方式使用 Python.evaluate 时 LSP 和 Mojo 崩溃。我希望它能向我展示代码的问题所在，而不是直接崩溃。复现步骤包括相关的...
- [GitHub - benz0li/mojo-dev-container: Multi-arch (linux/amd64, linux/arm64/v8) Mojo dev container](https://github.com/benz0li/mojo-dev-container)：多架构 (linux/amd64, linux/arm64/v8) Mojo dev container - benz0li/mojo-dev-container
- [Modular Inc](https://github.com/modular)：Modular 是一个集成、可组合的工具套件，可简化您的 AI 基础设施，让您的团队能够更快地开发、部署和创新。- Modular Inc

---

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1254449303972483212)** (58 条消息🔥🔥):

- **关于 `prefetch` 和 `PrefetchOptions` 的帮助**：一位成员就 `prefetch` 和 `PrefetchOptions` 寻求指导，指出在紧接着读取数据之前使用 `PrefetchOptions().for_write().low_locality().to_instruction_cache()` 会获得意想不到的加速。另一位成员确认，预取通常仅在 `N` 较大时有益，因为较小的 `N` 可能会适得其反。
- **缓存性能与预取**：成员们讨论了通过 `profiler` 了解缓存活动的重要性，因为错误使用手动预取可能会降低性能。他们强调阅读相关的手册，如 [Intel HPC tuning manual](https://link.to/manual)，以进一步深入了解预取机制。
- **指令缓存 vs 数据缓存**：澄清了获取到指令缓存（`icache`）也会影响指令和数据共享的 `L2` 缓存。由于结构化缓存管理的差异，这可能会导致意想不到的加速。
- **向量化/并行化调用中的函数内联**：讨论指出，内联函数通常会带来向量化/并行化操作的性能提升，因为 `outlined` 函数很少会被自动向量化。
- **优化工具**：为了缓存大小优化和其他性能原因，推荐使用 Intel 的 `vtune` 或 AMD 的 `AMD uProf` 等工具。Mojo 目前缺乏编译时缓存大小获取功能，而这是避免 `false sharing` 等问题所必需的。
  

**提到的链接**：

- [Prefetching - Algorithmica](https://en.algorithmica.org/hpc/cpu-cache/prefetching/)：未找到描述
- [PREFETCHW — Prefetch Data Into Caches in Anticipation of a Write](https://www.felixcloutier.com/x86/prefetchw)：未找到描述
- [PREFETCHh — Prefetch Data Into Caches](https://www.felixcloutier.com/x86/prefetchh)：未找到描述

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1253799832485892167)** (21 条消息🔥):

- **Nightly MAX 仓库落后于 Mojo**：一位成员注意到 nightly/max 仓库已经快一周没有更新了。另一位成员解释说，发布 MAX 的 nightly 构建的 CI 出现了问题，目前正在修复中。
- **新的 Mojo Nightly 版本发布**：发布了新的 nightly Mojo 编译器版本。用户可以更新到 `2024.6.2205` 和 `2024.6.2305`，详情见 [raw diffs](https://github.com/modularml/mojo/compare/bc3546a57e101fe0eb990bc15e96dad2b39e1aaf...40dc6b31bcaf1deb7032b2dff10ac80c068f9a3d) 和 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **受控隐式转换提案**：讨论透露，将隐式转换改为 opt-in（选择性开启）的提案来自 Modular。计划是使用一个 `decorator`，仅在合理的地方启用它。
- **排查 input() 函数中的段错误**：一位用户在 `input()` 函数中调整缓冲区大小时遇到段错误（segmentation fault）并寻求帮助。另一位用户建议这可能与一个关于无符号整数转换的 [现有 bug](https://github.com/modularml/mojo/issues/3065) 有关。
- **外部表情符号已可以使用**：一位成员庆祝 Discord 中现在可以使用外部表情符号了。他们对这一新功能表示兴奋。
  

**提到的链接**：

- [gojo/input.mojo at input · thatstoasty/gojo](https://github.com/thatstoasty/gojo/blob/input/input.mojo#L58)：将 Golang 标准库移植到 Mojo 的实验。 - thatstoasty/gojo
- [mojo/stdlib/docs/development.md at main · modularml/mojo](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md)：Mojo 编程语言。在 GitHub 上为 modularml/mojo 的开发做贡献。
- [[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo](https://github.com/modularml/mojo/issues/3065)：Bug 描述。在 Discord 讨论后迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同情况下表现不同...

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1253788315237351476)** (102 条消息🔥🔥):

- **Weta Digital 领导层变动引发反响**：讨论围绕 Weta Digital 及其新任 CEO 展开，提到了 Sean Parker，并推测这一决定更像是一场出售。*"来自 Weta Digital 的 Prem Akkaraju 吗"*，同时还提到了对公司可能面临的骚扰问题的沮丧情绪。
- **Stability AI 新任 CEO 与行业内幕**：分享了一篇关于 Stability AI 任命新 CEO 的 Reuters 文章，人们对领导层变动背后的动机表示怀疑。一位成员强调 *"对于那些不想为这些小丑支付 400 美元订阅费的人"*，并分享了一个 [Reuters 链接](https://www.reuters.com/technology/artificial-intelligence/stability-ai-appoints-new-ceo-information-reports-2024-06-21/)。
- **Llama 3 硬件推荐引起关注**：分享了在 12 通道 AMD 服务器上运行 Q6 Llama 400 的规格以及成本估算，引发了对潜在性能的兴奋。设定了 *"在此配置下每秒 1 到 2 个 tokens"* 的预期，并引发了关于其与 GPT-4O 和 Claude 3 对比的预测。
- **关于 Meta 模型推测的辩论**：用户讨论了 Meta 405B 模型的预期能力及其潜在的训练改革。评论包括对 8B 和 70B 等模型更新权重的期待，以及诸如 *"Meta 没有为 Llama 3 发布论文"* 之类的观察。
- **探索 EMA 和模型蒸馏的进展**：用户讨论了在 diffusers 中实现 EMA 模型更新的方法（由 lucidrains 在 [GitHub](https://github.com/lucidrains/ema-pytorch) 上分享），以及它们在特定项目中的适用性。还分析了训练数据集中多个 caption 的价值以及 text embeddings 的细微差别，并考虑了它们对模型训练和性能的影响。
  

**提到的链接**：

- [将活体神经元连接到计算机](https://youtu.be/c-pWliufu6U)：使用下方链接中的代码 thoughtemporium 可享受 Incogni 年度计划专属 4 折优惠：https://incogni.com/thoughtemporium____________________________...
- [呼吁为个人助手构建开源多模态模型 | LAION](https://laion.ai/notes/open-gpt-4-o/)：<p>OpenAI 最近推出的 GPT-4-OMNI 等技术再次展示了强大的多模态模型在积极转型方面可能具有的潜力...
- [ema: offload to cpu, update every n steps by bghira · Pull Request #517 · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/pull/517/files>)：未找到描述
- [Neuroplatform - FinalSpark](https://finalspark.com/neuroplatform/)：未找到描述

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1253894530336559276)** (27 messages🔥):

- **Glaze 团队对新攻击论文的评论**：Glaze 团队回应了关于对抗性扰动的新论文，承认了论文的发现，并讨论了他们使用作者代码进行的测试。他们强调了“噪声上采样（noisy upscaling）”方法及其对 Diffusion 模型的依赖（类似于 [DiffPure](https://arxiv.org/abs/2205.07460)），用于去除图像中的伪影。
- **对 Glaze/Nightshade 有效性的怀疑**：成员们对那些相信 Glaze 或 Nightshade 能保护其艺术作品的艺术家表示怀疑和遗憾。他们强调了后发者在规避这些保护措施方面具有不可避免的优势，以及由此给艺术家带来的虚假希望。
- **关于多模态模型的新论文**：讨论了一篇关于 [多模态模型](https://arxiv.org/abs/2406.09406) 的新论文，指出其在多种模态和任务上进行训练以提高模型通用性的努力。然而，成员们觉得这类论文总是在没有实质性新结果的情况下反复宣称取得了突破。
- **关于图像修复 Diffusion 模型的讨论**：对图像修复工具进行了详细询问，[Robert Hoenig](https://arxiv.org/search/?query=Hoenig) 讨论了他们在实验中使用 [超分辨率对抗防御](https://github.com/aamir-mustafa/super-resolution-adversarial-defense) 以及在特定图像分辨率上进行训练的情况。测试显示，Glaze 的保护措施被一致绕过。
  

**提到的链接**：

- [来自 François Fleuret (@francoisfleuret) 的推文](https://x.com/francoisfleuret/status/1804873919653957733)：一个小报告！
- [4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities](https://arxiv.org/abs/2406.09406)：当前的多模态和多任务基础模型（如 4M 或 UnifiedIO）展示了令人期待的结果，但在实践中，它们开箱即用的接受多样化输入和执行多样化任务的能力有限...
- [KalMamba: Towards Efficient Probabilistic State Space Models for RL under Uncertainty](https://arxiv.org/abs/2406.15131)：概率状态空间模型（SSMs）对于从高维、部分信息中进行强化学习（RL）至关重要，因为它们为控制提供了简明的表示。然而，它们缺乏...
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794)：我们推出了用于语言模型的 DataComp (DCLM)，这是一个用于受控数据集实验的测试平台，旨在改进语言模型。作为 DCLM 的一部分，我们提供了一个包含 240T token 的标准化语料库...
- [Glaze - v2.1 更新](https://glaze.cs.uchicago.edu/update21.html)：未找到描述

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1253902101382303775)** (117 条消息🔥🔥):

- **新成员导览 Discord 和 Cohere 频道**：多位新成员加入了 Discord，其中包括受 Varun 邀请的成员。频道内提供了关于如何使用该平台的建议，介绍了特定频道的使用方法，并分享了 [tool use documentation](https://docs.cohere.com/docs/tool-use) 链接，以帮助理解如何将 Cohere 模型连接到外部工具。
- **关于 BitNet 和模型 Quantization 的讨论**：成员们辩论了 BitNet 的可行性和未来应用，指出 BitNet 尚未针对当前硬件进行优化，且需要从头开始训练。Mr. Dragonfox 详细阐述了为什么 BitNet 目前在商业用途上不切实际，提到了其缺乏硬件支持以及训练需求效率低下的问题。
- **对新 AI 模型和传闻的关注**：一位成员表达了对 Cohere 发布新模型的期待，类似于 Meta、OpenAI 和 Anthropic 最近的更新。此外，还有关于 Anthropic 最新模型 Claude-3.5-Sonnet 的推测，并讨论了模型中单语义性（monosemanticity）的扩展，并链接了相关论文。
- **关于 Cohere 多语言能力的讨论**：一位用户询问 Cohere 是否可以用中文等其他语言回答。Nick_Frosst 确认了这一能力，并引导用户查看 [documentation](https://docs.cohere.com/docs/tool-use) 和 [notebook example](https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Tool_Use.ipynb)，以了解如何使用 Cohere 模型实现工具调用（tool use）。
  

**提到的链接**：

- [abideen/Bitnet-Llama-70M · Hugging Face](https://huggingface.co/abideen/Bitnet-Llama-70M)：未找到描述
- [Bonjour Bonjour Mon Amor GIF - Bonjour Bonjour mon amor Bonjour mon cher - Discover & Share GIFs](https://tenor.com/view/bonjour-bonjour-mon-amor-bonjour-mon-cher-bon-matin-bonjours-gif-11477332989234919415)：点击查看 GIF
- [Tool Use with Cohere's Models - Cohere Docs](https://docs.cohere.com/docs/tool-use)：未找到描述
- [Login | Cohere](https://coral.cohere.com/?_gl=1*db2k2l*_gcl_au*MTE5MDQyODEyMC4xNzE5MDcxNDg5*_ga*NzUxMTg0MTI2LjE3MTkwNzE0ODk.*_ga_CRGS116RZS*MTcxOTA5OTEwMC40LjEuMTcxOTA5OTEwMy41Ny4wLjA.)：Cohere 通过一个易于使用的 API 提供对高级 Large Language Models 和 NLP 工具的访问。免费开始使用。
- [Add support for BitnetForCausalLM (new model / new datatype) by Eddie-Wang1120 · Pull Request #7931 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/7931)：自报审查复杂度：低、中、高。我已阅读贡献指南。PR 介绍：此 PR 旨在支持 BitnetFor...

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1254548505964449876)** (10 条消息🔥):

- **Microsoft AutoGen 添加 Cohere 客户端**：一位贡献者分享了在 AutoGen 中添加 Cohere 客户端的 [GitHub pull request](https://github.com/microsoft/autogen/pull/3004/files)。用户们表示非常兴奋，称“太酷了，感谢添加客户端支持！”
- **呼吁 Cohere 团队参与**：一位成员澄清该贡献并非出自其手，并向社区贡献者喊话。另一位成员请求 Cohere 团队协助进一步的实现，“我们希望 Cohere 团队能帮助我们完成 CohereClient 的实现。”

**提到的链接**：[Cohere Client by Hk669 · Pull Request #3004 · microsoft/autogen](https://github.com/microsoft/autogen/pull/3004/files)：为什么需要这些更改？为了增强 AutoGen 对非 OpenAI 模型的支持。Command 系列模型包括 Command、Command R 和 Command R+。它们共同构成了文本生成...

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1254845382656004157)** (1 messages):

- **Cohere Developer Office Hours 公告**: *"欢迎明天参加我们的 Cohere Developer Office Hours！"* Cohere 的一位高级产品经理将共同主持本次会议，讨论 **Command R 系列的 tool use 能力**，重点关注 **Cohere API 中的 multi-step tool use**。
- **Multi-step Tool Use 详细概览**: Cohere 分享了 multi-step tool use 的概览，它 *"允许 Cohere 的模型调用外部工具：搜索引擎、API、函数、数据库等。"* 欲了解更多信息，请参阅 Cohere 文档和博客文章 ([multi-step tool use](https://docs.cohere.com/docs/multi-step-tool-use), [Command R+](https://cohere.com/blog/multi-step-tool-use))。
  

**提到的链接**:

- [加入 Cohere 社区 Discord 服务器！](https://discord.gg/s3pcZTyPgD?event=1248301309233336350): Cohere 社区服务器。欢迎来这里聊聊 Cohere API、LLM、Generative AI 以及相关的一切。 | 17232 名成员
- [使用 Cohere 自动化复杂的业务工作流：Multi-Step Tool Use 实战](https://cohere.com/blog/multi-step-tool-use): 企业正越来越多地采用 AI 来增强业务工作流。配备了外部工具的 AI 模型具有简化业务运营的潜力。在 Cohere，我们很高兴分享...
- [Multi-step Tool Use (Agents)](https://docs.cohere.com/docs/multi-step-tool-use): 未找到描述

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1253801659675578378)** (100 messages🔥🔥):

- **Max Tokens 和 Pydantic 验证困扰用户**: 用户讨论了关于 Agent 的 max tokens 和 context window 的困惑，以及 LLM 不遵循 **Pydantic** 验证的问题。*"context window 或 max token 始终包含完整的输入 token 加上生成的 token。"*
- **LangChain 教程与资源**: 几位用户表示学习 **LangChain** 存在困难，特别是在构建 chatbot 和处理对话偏离方面。[Grecil](https://corrective-rag.streamlit.app) 分享了学习 LangChain 的个人历程，并提供了教程和文档链接。
- **使用多个聊天模型和 API**: 用户辩论了 **ChatOpenAI** 与来自 **Huggingface** 的开源模型在不同场景下的性能问题和应用。一位用户询问了如何处理 Excel 文件的 RAG，暗示了对 **LangChain** 支持各种数据格式通用性的担忧。
- **在 Chain 中处理消息历史和元数据**: 用户寻求关于实现和调试 **RunnableWithMessageHistory** 以及在文档检索器中加入元数据的帮助。*"如何添加包含此 chain 中检索到的文档/分块的元数据。"*
- **Streamlit 应用托管讨论**: 讨论了 **Streamlit** 应用中的资源管理和并发问题，包括嵌入 API key 和同时处理多个用户。"是的，Streamlit 会处理这些。一旦你关闭标签页，你的实例和你上传的文件就会被删除。"
  

**提到的链接**:

- [simplememory](https://pypi.org/project/simplememory/): Agent 记忆框架
- [Reflexion - LangGraph](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/#revision): 未找到描述
- [NVIDIA NIMs | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/chat/nvidia_ai_endpoints/#example-usage-within-a-conversation-chains>): langchain-nvidia-ai-endpoints 包包含用于在 ... 上使用模型构建应用程序的 LangChain 集成
- [构建聊天机器人 | 🦜️🔗 Langchain](https://js.langchain.com/v0.2/docs/tutorials/chatbot/#managing-conversation-history>): 概览
- [构建聊天机器人 | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/tutorials/chatbot/#managing-conversation-history>): 本指南假设你熟悉以下概念：
- [构建聊天机器人 | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>): 本指南假设你熟悉以下概念：
- [TiDB Vector | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/vectorstores/tidb_vector/#using-as-a-retriever>): TiDB Cloud 是一款全面的数据库即服务 (DBaaS) 解决方案，提供专用和无服务器选项。TiDB Serverless 正在将内置向量搜索集成到 MySQL 领域...
- [Introduction - LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-6-customizing-state.): 未找到描述
- [无标题](https://corrective-rag.streamlit.app): 未找到描述

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1254216522109423647)** (21 条消息🔥):

- **使用 LangChain 从 PDF 生成问答对**：一位用户请求了使用 LangChain 从 PDF 生成问题和答案的代码。Python 代码涉及使用 `PyPDFLoader` 加载 PDF，将其拆分为块（chunks），使用 `OpenAIEmbeddings` 创建嵌入（embeddings），并设置 `RetrievalQA` 链。
- **链接来自 GitHub 的 issue**：提供的代码参考了几个 GitHub issues，例如 [这一个](https://github.com/langchain-ai/langchain/issues/17008)，用于指导如何从 PDF 生成问答对。
- **使用 Llama2 作为 LLM**：另一位用户请求修改代码以使用 Llama2 作为 LLM。更新后的指令建议初始化 `LlamaCpp` 并使用 `prompt_template` 设置 `QAGenerationChain`。
- **遍历文本以生成问答对**：最后，提供了关于如何遍历来自 PDF 的文本块，并使用 `QAGenerationChain` 生成问答对的指令。这种方法确保了从文档中生成多个问答对。
  

**提到的链接**：

- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/20406>).): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/20406>)): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/10395>).): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/17008>)): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/4950>).): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1254034783869206558)** (5 条消息):

- **金融文档的无代码 RAG 工作流**：一位成员分享了一篇关于使用 Flowise 设计用于金融文档分析的检索增强生成（RAG）应用程序的[文章](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888)。主要特性包括使用 Redis 实现的嵌入缓存（embedding cache）以及用于语义搜索的 Qdrant。
- **从零开始实现线性回归**：另一位成员发布了一篇[文章](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed)，详细介绍了如何用 Python 从头实现线性回归。该教程避免使用 scikit-learn 等机器学习包，而是专注于核心概念。
- **Corrective RAG 应用**：一位成员提供了他们在 Streamlit 上的 [Corrective RAG 应用](https://corrective-rag.streamlit.app)链接。
- **Edimate：AI 驱动的教育视频**：一位成员介绍了 Edimate，这是一个能在约三分钟内生成教育视频的工具。他们分享了一个[演示](https://x.com/dswharshit/status/1805203856088834428)，展示了其通过创建引人入胜的动画视频来改变在线学习（e-learning）的潜力。
- **LLM 的回归测试**：一篇信息丰富的帖子链接到了一个关于使用开源工具对 LLM 进行回归测试的[代码教程](https://www.evidentlyai.com/blog/llm-regression-testing-tutorial)。该教程涵盖了创建黄金数据集（golden datasets）、评估响应变化以及使用 Evidently Python 库来评估 LLM 输出。

**提到的链接**：

- [来自 Harshit Tyagi (@dswharshit) 的推文](https://x.com/dswharshit/status/1805203856088834428)：如何用 AI 重新定义在线学习？这是我在教育科技领域工作近十年后的疑问。答案是按需生成视频/课程来解释任何主题……
- [关于 LLM 回归测试的教程](https://www.evidentlyai.com/blog/llm-regression-testing-tutorial)：在本教程中，你将学习如何系统地检查 LLM 输出的质量。你将处理诸如答案内容、长度或语调变化等问题，并了解哪些方法可以检测到这些问题……
- [用 Python 从零开始实现线性回归](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed)：学习如何在纯 Python 中从头实现线性回归。包括损失函数、梯度下降算法、模型训练等……
- [金融文档的轻松无代码 RAG 工作流：实现嵌入缓存和聊天……](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888)：在快速发展的金融数据分析领域，无需广泛的开发即可利用先进技术的力量……
- [未找到标题](https://corrective-rag.streamlit.app)：未找到描述

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1254886695481114694)** (1 条消息):

- **决定使用哪种 AI 框架？先问这些关键问题**：一位成员分享了一个[关于 AI 框架考量的 YouTube 视频](https://youtu.be/uG0cs8AlnHw)。视频讨论了开发者在将 GPT-4o 等 AI 工具集成到应用中之前应该询问的基本问题。

**提到的链接**：[你的应用真的需要 AI 框架或 GPT-4o 吗？](https://youtu.be/uG0cs8AlnHw)：所以，你想把 AI 集成到你的产品中，对吧？别急，没那么快！有了 GPT-4o、Gemini、Claude、Mistral 等模型以及各种框架……

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1254860159113756702)** (1 条消息):

- **AI21 推出 Jamba-Instruct：** Jamba-Instruct 是 AI21 推出的指令微调变体，专为企业级用途量身定制，拥有令人印象深刻的 **256K 上下文窗口 (context window)**，可处理大型文档。点击[此处](https://openrouter.ai/models/ai21/jamba-instruct)查看更多详情。
- **NVIDIA 发布 Nemotron 4 340B Instruct：** Nemotron-4-340B-Instruct 是一款聊天模型，专注于英语应用的**合成数据生成 (synthetic data generation)**。点击[此处](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct)了解更多。
  

**提到的链接**：

- [AI21: Jamba Instruct by ai21](https://openrouter.ai/models/ai21/jamba-instruct)：Jamba-Instruct 模型由 AI21 Labs 推出，是其混合 SSM-Transformer Jamba 模型的指令微调变体，专门针对企业应用进行了优化。- 256K 上下文窗口...
- [NVIDIA Nemotron-4 340B Instruct by nvidia](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct)：Nemotron-4-340B-Instruct 是一款针对合成数据生成优化的英语聊天模型。该大语言模型 (LLM) 是 Nemotron-4-340B-Base 的微调版本，专为单轮对话设计...

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1253865624073932801)** (7 条消息):

- **JojoAI 转型为主动助手**：一名成员将 **JojoAI** 转型为能够执行*设置提醒*等功能的主动助手。他们强调，与 ChatGPT 或 Claude 不同，JojoAI 使用 DigiCord 集成在特定时间提醒用户 [JojoAI 网站](https://www.digicord.site)。
- **Pebble：AI 阅读理解工具**：一款名为 **Pebble** 的 AI 驱动阅读理解工具已上线，旨在帮助用户记住网页上的信息。开发者使用了 OpenRouter 配合 **Mistral 8x7b** 和 **Gemini**，并对 OpenRouter 团队的支持表示感谢 [Pebble](https://pebble.study/)。
- **使用 OpenRouter 修改 MoA 项目**：一位贡献者修改了 **MoA 项目** 以使用 OpenRouter，并添加了一个带有 API 端点的服务器，并创建了一个用于操作的 GUI。该项目已在 [GitHub](https://github.com/timothelaborie/MoA-Openrouter/blob/main/gui.ipynb) 上发布。
  

**提到的链接**：

- [Pebble](https://pebble.study/)：未找到描述
- [DigiCord](https://www.digicord.site)：有史以来最强大的 AI 驱动 Discord 机器人！
- [MoA-Openrouter/gui.ipynb at main · timothelaborie/MoA-Openrouter](https://github.com/timothelaborie/MoA-Openrouter/blob/main/gui.ipynb)：结合了 MoA 但使用 Openrouter。通过在 GitHub 上创建账户来为 timothelaborie/MoA-Openrouter 的开发做出贡献。

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1253793170710462559)** (106 条消息🔥🔥):

- **Nemotron 340b 的环境影响受到质疑**：*“Nemotron 340b 绝对是你所能使用的对环境最不友好的模型之一。”* 讨论继续进行，对比建议认为 Gemini Flash 和其他更小、更便宜的模型是合成数据生成的更好替代方案。
- **Claude 自我审核端点问题已修复**：*“看起来 Claude 的自我审核端点消失了？”* 在标记了 404 错误后，修复程序迅速实施，问题已得到解决。
- **Sonnet 3.5 的编程能力受到称赞**：一位用户分享了使用 Sonnet 3.5 进行编程的积极体验，称其令人印象深刻，并指向了一个结合检索增强生成 (RAG) 的[真实世界演示](https://simonwillison.net/2024/Jun/21/search-based-rag/)。
- **OpenRouter 速率限制和额度说明**：*“如何提高特定 LLM 的速率限制？”* 共享了关于速率限制 (rate limits) 和额度 (credits) 的文档，解释了如何通过 API 请求检查余额和使用情况。
- **处理泄露的 API 密钥**：*“嘿，我像个白痴一样，在直播中展示了一个新创建的 API 密钥，结果有人用了它。”* 建议是禁用而不是删除受损的密钥，以便更好地追踪任何不当使用。
  

**提到的链接**：

- [Transforms | OpenRouter](https://openrouter.ai/docs/transforms)：转换数据以供模型消费
- [Limits | OpenRouter](https://openrouter.ai/docs/limits#rate-limits-and-credits-remaining)：设置模型使用限制
- [Building search-based RAG using Claude, Datasette and Val Town](https://simonwillison.net/2024/Jun/21/search-based-rag/)：检索增强生成 (RAG) 是一种为基于 LLM 构建的系统添加额外“知识”的技术，使其能够针对其训练数据中未包含的自定义信息回答问题...

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1253828783019003924)** (85 条消息🔥🔥):

- **OS 模式下的本地 LLMs？**: 一位成员询问是否可以在 OS 模式下使用本地 LLMs。另一位成员确认道：“是的！但这些模型的性能不是很好……”并提供了命令 `interpreter --local --os`。
- **桌面端 App 的高级体验**: 一位成员询问了桌面端 App 与 GitHub 版本之间的区别。Mikebirdtech 强调：“桌面端 App 将是体验 Open Interpreter 的一种非常酷的方式”，并建议加入 [桌面端 App 等候名单](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com)。
- **达成 GitHub Star 里程碑**: Killianlucas 兴奋地宣布该项目在 GitHub 上已达到 **50,000 stars**，称其为社区的一项巨大成就。他提到很快会发布一个重大的服务器公告。
- **Codestral 和 Deepseek 模型热度**: 几位成员讨论了最近发布的 Deepseek 和 Codestral 模型，Killianlucas 指出：“Codestral……击败了我们所有的内部基准测试……”，并因其速度而青睐 Deepseek，提到即将发布一个带有优化过的 `interpreter --deepseek` 命令的更新。
- **Ollama 连接问题**: Arsaboo 在使用 OI 界面连接到另一台计算机上托管的 Ollama 时遇到了问题。多位成员建议了各种修复和故障排除步骤，包括更改 API base URLs 和使用代理，但都没有最终解决问题。
  

**提到的链接**:

- [未找到标题](http://192.168.2.162:11434): 未找到描述
- [Open Interpreter v0.3 第 2 部分](https://www.youtube.com/live/7lyw8V1PK3s?si=XT3DgJNTb7vQfpdM&t=9772): 0:00 - 设置；6:10 - 调试 `interpeter --os`；8:01 - 使用 Cursor 辅助调试；19:38 - 聊天；22:24 - Sonnet 的回答优于 4o；29:00 - 修复 bash...
- [open-interpreter/interpreter/terminal_interface/profiles/defaults/codestral.py at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/codestral.py): 一个用于计算机的自然语言界面。通过在 GitHub 上创建账户，为 OpenInterpreter/open-interpreter 的开发做出贡献。
- [由 MikeBirdTech 将 Vision 模型更新为 gpt-4o · Pull Request #1318 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/pull/1318): 描述你所做的更改：gpt-4-vision-preview 已弃用，应更新为 gpt-4o https://platform.openai.com/docs/deprecations/2024-06-06-gpt-4-32k-and-vision-preview-models ...
- [在不同机器上将 Open Interpreter 与 Ollama 配合使用 · Issue #1157 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/issues/1157#issuecomment-2184982086>): 描述 Bug：我正尝试在另一台运行 Ollama 的计算机上使用 OI。我使用的命令是：`interpreter -y --context_window 1000 --api_base http://192.168.2.162:11434/api/generate` -...
- [Open Interpreter - 桌面端 App](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com): 申请 Open Interpreter 桌面端 App 的早期访问权限。
- [Google Colab](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g)): 未找到描述

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1253843036362903614)** (17 messages🔥):

- **Poetry vs requirements.txt 引发辩论**：成员们讨论了使用 **Poetry** 替代传统 `requirements.txt` 文件的优缺点。一位成员强调了 Poetry 的确定性构建（deterministic builds）和管理便捷性，而另一位成员指出它在跨平台管理上可能存在困难，并建议将 **conda** 作为替代方案。
- **分享 01 安装文档**：一位成员分享了在不同操作系统上安装 01 的[设置链接](https://01.openinterpreter.com/getting-started/setup)。另一位成员表达了挫败感，称其在某些平台上“尚无法运行”。
- **Windows 安装挑战**：讨论强调了在 Windows 上使用 **Poetry** 和 **venv** 等工具管理依赖项与使用 **conda** 相比的困难。尽管一位用户断言 Poetry 和 venv 在 Windows 上运行良好，但另一位用户指出非 01 软件包经常出现失败。
- **社区情绪**：一位成员表达了强烈的正面情感，称这个 Discord 社区是他们的最爱。其他人讨论了 01 light 的入门友好性，开发者指出当前版本需要技术知识，但未来的版本旨在变得更加易于使用。
- **发货时间表的沮丧**：成员们对 01 设备的发货时间表表示担忧。一位用户提到了反复的延迟，而另一位用户则针对被视为误导的信息为时间表进行了辩护。
  

**提到的链接**：

- [Poetry - Python dependency management and packaging made easy](https://python-poetry.org/)：未找到描述
- [Setup - 01](https://01.openinterpreter.com/getting-started/setup)：未找到描述

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1254330956882640936)** (5 messages):

- **来自 Techfren 社区的有趣缩略图**：一位成员分享了一个 [YouTube 直播视频](https://youtube.com/live/8TN8tzkyB50?feature=share)，并注意到由 Techfren 社区的 Flashwebby 制作的有趣缩略图。另一位成员评论说很喜欢这个缩略图，这促使原成员分享了他们对视频的轻松贡献。
- **Amoner 使用 AI 混音 "The Wheels on the Bus"**：一位成员展示了一个 [YouTube 视频](https://www.youtube.com/watch?v=a-Jlq0iX898&t=47s&ab_channel=Amoner)，重点介绍了使用 Suno 和 Luma 技术混音的 "The Wheels on the Bus"。视频描述强调了创新性地使用 GenAI 技术来创建下一代音乐和视觉效果。

**提到的链接**：[AI Remix: The Wheels on the Bus | Next-Gen Music & Visuals by Suno & LumaLabs](https://www.youtube.com/watch?v=a-Jlq0iX898&t=47s&ab_channel=Amoner)：通过这款创新的 AI 生成混音，体验前所未有的 'The Wheels on the Bus'！利用最新的 GenAI 技术，我们与 S... 合作。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1253816131815346207)** (33 条消息🔥):

- **探索用于多任务学习的 Instruction Pre-Training**：一名成员分享了一个关于 **Instruction Pre-Training** 的 [Hugging Face 仓库](https://huggingface.co/instruction-pretrain/instruction-synthesizer)，该方法通过指令-响应对增强原始语料库，用于有监督的多任务预训练。该方法已在 40 多个任务类别中有效合成了 2 亿个配对。
- **带有 Flash Attention 2 的 DeBERTa**：一位用户询问是否有人知道 **使用 Flash Attention 2 的 DeBERTa 实现**，表示有兴趣将这两项技术结合起来。
- **Maven 课程平台上的空白页问题**：多位用户在尝试访问 Maven 上的课程时遇到了空白页，引发了关于故障排除以及尝试联系 Maven 支持的讨论。一个临时的变通方法是在移动设备上访问课程。
- **运行 AI 应用研讨会**：与会者讨论了即将举行的旧金山活动 [AI Engineer World’s Fair](https://www.ai.engineer/worldsfair)，其中包括关于使用模板快速部署 AI 应用的研讨会。几位成员表示有兴趣在活动中见面。
- **为什么公司更倾向于 Fine-tuning 而非 RAG**：有一场关于为什么招聘广告通常寻求 Fine-tuning 经验而非检索增强生成（RAG）的讨论。有人建议，公司旨在降低 LLM 成本，这使得 Fine-tuning 成为一项有价值的技能。
  

**提到的链接**：

- [AI Mathematical Olympiad - Progress Prize 1 | Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/512844)：未找到描述
- [Welcome to Outlines! - Outlines 〰️](https://outlines-dev.github.io/outlines/welcome/)：使用 LLM 进行结构化文本生成
- [instruction-pretrain/instruction-synthesizer · Hugging Face](https://huggingface.co/instruction-pretrain/instruction-synthesizer)：未找到描述
- [Instruction Pre-Training: Language Models are Supervised Multitask Learners](https://arxiv.org/abs/2406.14491)：无监督多任务预训练一直是近期语言模型（LMs）成功背后的关键方法。然而，有监督多任务学习仍具有巨大潜力，随着规模的扩大……
- [无标题](https://maven.com/parlance-labs/fine-tuning/1/home)：未找到描述
- [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa)：Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大规模数据集的高效相似度估算和去重 - beowolx/rensa
- [AI Engineer World's Fair](https://www.ai.engineer/worldsfair)：加入 2,000 名由 AI 增强并利用 AI 进行构建的软件工程师。2024 年 6 月 25 日至 27 日，旧金山。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/)** (1 条消息):

christopher_39608: 有趣的帖子：

[https://x.com/rasbt/status/1805217026161401984](https://x.com/rasbt/status/1805217026161401984)

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1253805483987570872)** (6 条消息):

- **额度缺失与故障排除**：一位用户报告称，“我还没有收到额度”，并被建议如果他们正确填写了表格，请联系账单部门。他们被告知向账单部门发送电子邮件，并附上注册日期证明、HF 用户名和电子邮件。
- **及时的客服响应**：另一位面临同样问题的个人直接在频道中提到了他们的 HF 用户名和电子邮件。他们收到了快速回复，建议他们联系账单部门以获得进一步协助，并确认已将收据发送至提供的电子邮件。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1253980281635868774)** (3 条消息):

- **报告 Mixtral 8x22 的模板损坏**：一位用户询问了关于 **Mixtral 8x22** 模板损坏的问题，并标记了两名成员寻求帮助以解决该问题。
- **通过 VS Code 扩展使用 Replicate 额度**：据分享，**Replicate 额度**可以通过名为 **continue.dev** 的 VS Code 扩展来使用。该扩展的功能类似于 **GitHub Copilot**，使用 Replicate APIs，并提供 **@docs 功能**以便在本地与 Replicate 文档进行交互。

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1253820800776147055)** (1 条消息):

- **用户因额度缺失感到沮丧**：一名用户报告在登录并添加账单信用卡后未看到其额度。他们分享了组织 ID *be7114fc-9d79-475a-a258-ddbda1553c9a* 以寻求帮助。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/)** (1 条消息):

jxnlco: nah

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1254483892149293156)** (3 条消息):

- **Subprocess.CalledProcessError 困扰训练过程**：一名用户报告了一个错误：*subprocess.CalledProcessError: Command '['/usr/bin/python3', '-m', 'axolotl.cli.train', '/content/qlora.yml']' returned non-zero exit status 1*，这表明运行 Axolotl 的训练命令时出现了问题。
- **LORA 过拟合担忧**：另一名用户询问，即使在使用 **LORA** 时，训练损失显著低于验证损失是否预示着过拟合。该问题反映了用户在微调模型时对过拟合的普遍担忧。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1254604590226210826)** (1 条消息):

- **请求协助解决 .yml 和数据集中的错误**：一位成员就遇到的错误寻求帮助。他们附上了 .yml 文件和数据集以提供上下文，并提到在此次 FTJ 中使用了 Modal，对任何提供的支持表示感谢。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/)** (1 条消息):

mgrcic: 也可以在 [https://www.youtube.com/watch?v=QUXQNi6jQ30](https://www.youtube.com/watch?v=QUXQNi6jQ30) 查看

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1253805778351947838)** (3 条消息):

- **Dan 澄清额度问题**：一名用户因尚未收到额度而寻求帮助。Dan 询问该用户是否在截止日期前注册并回复了表单，并表示如果提供电子邮件地址，他可以检查发送到平台的数据。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1254293680638922752)** (2 条消息):

- **用户标签和代码占据了聊天内容**：由于出现了类似 `<@466291653154439169>` 的用户标签以及 `tyagi-dushyant1991-e4d1a8` 和 `williambarberjr-b3d836` 这样的代码，看起来成员们正在分享唯一标识符或代码。目前没有关于这些标签用途或目的的进一步上下文。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1253818157441749175)** (25 条消息🔥):

- **部分用户缺失额度**：包括 **xyz444139**、**nima01258** 和 **claudio_08887** 在内的几位成员报告称，尽管按照流程操作，但仍未收到额度。**ankrgyl** 通过检查电子邮件记录、确认权限并在适当情况下发放额度来解决这些问题。
- **重启 kernel 后权限问题得到解决**：**claudio_08887** 在运行评估示例时遇到了 “User does not have permissions to create a project within this org” 错误。在重启 kernel 后问题得到解决，这表明它可能是一个临时性问题。
- **braintrust 缺乏直接的微调功能**：当被问及使用 braintrust 微调 Huggingface 模型的教程时，**ankrgyl** 澄清说 braintrust 可以协助评估微调后的模型，但本身不具备内置的微调功能。
- **客户反馈受到赞赏和鼓励**：**lapuerta91** 表达了对产品的赞赏，**ankrgyl** 对此表示感谢，并邀请其就潜在的改进提供进一步反馈。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253822951149797488)** (13 条消息🔥):

- **Predibase 额度在 30 天后过期**：一名用户询问 **Predibase 额度是否在月底过期**。随后得到了确认，并附带了参考链接，说明 **额度在发放 30 天后过期**。
- **新用户额度协助**：一名新用户注意到可用额度仅显示为 $25。**Predibase support** 建议直接私信或发送电子邮件至 [support@predibase.com](mailto:support@predibase.com) 寻求帮助。
- **企业级功能**：讨论中提到了 Predibase 的 **enterprise tier**（企业级），称其为 **生产规模的应用** 提供功能。建议对该级别感兴趣的用户联系支持团队。

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1254141690714001508)** (5 条消息):

- **LightningAI 的 RAG 模板简化了 AI 开发**：LightningAI 提供了用于开发和共享传统 ML 及 GenAI 应用的工具，正如 [Jay Shah 的模板](https://t.co/2NLH7zuZS6)所示，该模板用于构建多文档 Agentic RAG。此模板允许开箱即用的配置，以简化开发流程。
- **使用 DAG 实现可定制的 Text-to-SQL**：现有的 Text-to-SQL 模块在生产环境中使用时，通常需要自定义编排和 Prompt 调整。llama_index 的一个[被低估的特性](https://t.co/fiS0kfj8rk)是它支持这些高级 LLM 定制的能力。
- **用于更好财务分析的 Corrective RAG**：由 Yan 等人描述的 CRAG 技术可以评估检索质量，并在知识库不足时使用网络搜索作为备份上下文。Hanane Dupouy 的[教程幻灯片](https://t.co/lHsThk9IOU)为实现这种高级 RAG 技术提供了详细指导。
- **使用 Mlflow 进行 RAG 参数调优**：管理 RAG 从分块（chunking）到索引（indexing）的众多参数对于回答的准确性至关重要，因此必须有一套系统的跟踪和评估方法。将 llama_index 与 [Mlflow](https://t.co/fo8XxMTO93) 集成，通过定义适当的评估指标和数据集，有助于实现这一目标。
- **LlamaIndex 通过 StabilityAI 集成图像生成功能**：create-llama 中的新功能现在支持使用 [StabilityAI](https://t.co/a7F0gv4tpi) 生成图像。这一集成扩展了 LlamaIndex 为 AI 开发者提供的能力。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1253885329572892813)** (70 条消息🔥🔥):

- **LlamaIndex 查询响应模式详解**：成员们讨论了 LlamaIndex 中的各种查询响应模式，例如 **Refine**、**Compact**、**Tree Summarize** 和 **Accumulate**。每种模式使用不同的策略来增量生成和优化响应，或通过树状总结生成响应（[来源](https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode)）。
- **在 LlamaIndex 中使用 OLLAMA_NUM_PARALLEL**：一位成员询问了如何使用 **OLLAMA_NUM_PARALLEL** 在 LlamaIndex 中并发运行多个模型。据指出，这似乎只需要设置一个环境变量，目前 LlamaIndex 不需要进行任何更改。
- **文档解析问题**：有人提出 LlamaIndex 网站上的某些文档页面无法正确渲染。以 .md 结尾的链接被指出是原因所在，官方计划更新这些页面（[示例链接](https://docs.llamaindex.ai/en/stable/community/full_stack_projects/)）。
- **关于向量数据库中自定义相似度评分的讨论**：一位成员询问如何在 LlamaIndex 中使用 Weaviate 或 Elasticsearch 定义自定义相似度评分。建议在向量数据库层面实现此功能，因为 LlamaIndex 是对这些库的封装，并不直接支持自定义检索器（retrievers）。
- **PGVectorStore 中的 Embedding 维度不匹配**：一位成员在使用 **bge-small embedding** 模型与 **PGVectorStore** 时遇到了维度不匹配的问题，该模型需要 384 维的 Embedding，而默认值为 1536。建议调整 `embed_dim` 参数并确保使用正确的 Embedding 模型。

**提到的链接**：

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)：大型语言模型（LLMs）展示了令人印象深刻的能力，但也面临幻觉、知识过时以及推理过程不透明、不可追溯等挑战。检索增强...
- [Anthropic | LlamaIndex.TS](https://ts.llamaindex.ai/modules/llms/available_llms/anthropic)：用法说明
- [Full Stack Projects - LlamaIndex](https://docs.llamaindex.ai/en/stable/community/full_stack_projects/)：无描述
- [Full-Stack Web Application - LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/apps/)：无描述
- [Query Engine with Pydantic Outputs - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/query_engine/pydantic_query_engine/#query-engine-with-pydantic-outputs)：无描述
- [Index - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode)：无描述

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1254781955933999154)** (1 messages):

- **MLflow 与 LLMs 结合 LlamaIndex 的指南**：分享了一个关于使用 **LlamaIndex** 集成 **MLflow** 和 **LLMs** 的 [Medium 文章](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437) 链接。该文章由 Ankush K Singal 撰写，旨在“解锁机器学习的效率”。

**提到的链接**：[Unlocking Efficiency in Machine Learning: A Guide to MLflow and LLMs with LlamaIndex Integration](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437)：Ankush k Singal

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1253860614942752838)** (17 messages🔥):

- **Gemini 1.5 Pro 的参数量少于 LLAMA 3 70B**：一位拥有“Meta 可靠消息源”的成员声称 *“Gemini 1.5 Pro 的参数量比 LLAMA 3 70B 少。”* 这引发了关于架构差异的讨论，特别是 MoE (Mixture of Experts) 如何影响推理时的激活参数量。
- **GPT-4 中的 Early fusion 技术**：关于 GPT-4T/o 是蒸馏模型还是利用了 Early fusion 技术存在争论。一位成员建议 *“GPT4 o 只是 Early fusion 的 GPT4”*，而另一位则认为它涉及像 *“GPT4-omni”* 这样的大型模型蒸馏而成。
- **多模态模型后训练的难度**：针对 Gemini Ultra 和 GPT4-o 等多模态模型的后训练（post-training）展开了讨论，强调了模态迁移中的挑战。有人指出 *“原生多模态模型的后训练非常困难，而且跨模态的迁移似乎很小。”*
- **Multi 加入 OpenAI，停止应用服务**：根据一篇 [博客文章](https://multi.app/blog/multi-is-joining-openai)，曾旨在将桌面计算重新构想为原生多人协作模式的 Multi 正在加入 OpenAI。Multi 将于 2024 年 7 月 24 日停止服务，一位成员评论道 *“OpenAI 正在开启大肆收购模式”。*

**提到的链接**：[Multi Blog – Multi is joining OpenAI](https://multi.app/blog/multi-is-joining-openai) ：最近，我们越来越多地问自己应该如何与计算机协同工作。不是在计算机上或使用计算机，而是真正地与计算机协同。与 AI 一起。我们认为这是最重要的……

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1253943008655577090)** (20 messages🔥):

- **错误代码的价值**：成员们辩论了在训练中包含错误代码的重要性。有人表示，*“带有错误的代码是必要的，这样它才能理解如何修复错误”*，而另一位则强调 *“坏数据需要置于某种语境中，使其明显是坏的。”*
- **AI 数据集中的风险规避**：讨论了使用开放数据集的高风险。一位成员指出，*“现在的赌注太高了……人们第无数次过滤 CommonCrawl”*，这在很大程度上是由于对法律问题和舆论反弹的担忧。
- **伦理与许可证问题**：对话涵盖了许可证条款的不一致性。一位成员幽默地评论道，*“你就是不能上传并用你自己的数据训练，哈哈”*，指出了对限制性许可证的实际规避。
- **高风险数据类型**：Natolambert 指出，与其它类型的数据相比，视频和图像数据集具有更高的风险。他们还表示需要更快地改进合成数据（synthetic data）方案，暗示了目前的局限性。
- **相关文章链接**：讨论中包含了一篇由 dn123456789 分享的 [2022 年关于 AI 数据洗白的文章](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/)，该文章强调了科技公司如何规避责任。这引发了关于当前 AI 实践中数据集伦理现状惨淡的评论。
  

**提到的链接**：

- [AI Data Laundering: How Academic and Nonprofit Researchers Shield Tech Companies from Accountability - Waxy.org](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/)：开发 AI 的科技公司正将数据收集和训练外包给学术/非营利研究小组，从而使他们免于潜在的问责和法律责任。
- [AI Data Laundering: How Academic and Nonprofit Researchers Shield Tech Companies from Accountability - Waxy.org](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-compa)：开发 AI 的科技公司正将数据收集和训练外包给学术/非营利研究小组，从而使他们免于潜在的问责和法律责任。

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1253805819179569195)** (13 messages🔥):

- **Sony Music 对阵 Nous Research：** 一名 Nous Research 成员在 X 上艾特了 @sonymusic，并质问：“nouse research 到底是谁？”。这引发了好奇心，似乎将关于 AI 创新和潜在法律纠纷的讨论混为一谈。
- **预先发出的停止侵权函（Cease and Desist）笑话：** 一位成员开玩笑说，尽管从未训练过音频模型，但还是解锁了“超稀有的‘预先停止侵权函’成就”，为法律担忧增添了幽默感。
- **Claude 3.5 阴谋论：** 有人分享了一个幽默的阴谋论，称“Claude 3.5 并不是真的，只是把 Claude 3 的‘我很聪明’向量调高了而已”，表达了对模型改进的怀疑。
- **OpenAI 模糊的道歉：** Mira Murati 在 X 上的帖子谈到了 OpenAI 的使命、Sora 和 GPT-4o 等工具，以及在创造创新 AI 与管理其影响之间取得平衡。尽管她做了详细解释，但一位成员评论说，这种道歉“显然无法让任何人满意”。
- **Hugging Face 访问风波：** Hugging Face 某个模型页面上的公告称，由于冲突，他们将暂停新的下载访问请求，理由是认为 Hugging Face “反复滥用‘贡献者公约行为准则’（Contributor Covenant Code of Conduct）”，并优先考虑商业化而非社区福祉。
  

**提到的链接**：

- [CausalLM/14B-DPO-alpha · Hugging Face](https://huggingface.co/CausalLM/14B-DPO-alpha)：未找到描述
- [来自 Nous Research (@NousResearch) 的推文](https://x.com/nousresearch/status/1804219649590276404?s=46)：呃，嘿 @sonymusic，nouse research 到底是谁
- [来自 Tsarathustra (@tsarnick) 的推文](https://x.com/tsarnick/status/1803901130130497952)：Mira Murati：GPT-3 是幼儿水平，GPT-4 是聪明的高中生水平，而将在一年半内发布的下一代将达到博士水平
- [来自 emozilla (@theemozilla) 的推文](https://x.com/theemozilla/status/1804220182237495461?s=46)：我解锁了超稀有的“预先停止侵权函”成就（注：我从未训练过任何音频模型）引用 Nous Research (@NousResearch) 呃，嘿 @sonymusic，到底谁是...
- [来自 Mira Murati (@miramurati) 的推文](https://x.com/miramurati/status/1804567253578662264)：在 OpenAI，我们致力于推进科学理解，以帮助改善人类福祉。我们正在构建的 AI 工具，如 Sora、GPT-4o、DALL·E 和 ChatGPT，从技术角度来看令人印象深刻...

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014)** (9 条消息🔥):

<ul>
  <li><strong>互联网流量与内容质量</strong>：一位成员建议，如果内容真的很棒，人们就会点击并探索。然而，他们指出，如果内容平庸，本来就不值得获得太多流量。</li>
  <li><strong>农夫与羊问题的笑话</strong>：有人分享了一条幽默的推文，扩展了“一个农夫和一只羊的问题”，暗示“羊也可以划船”。完整的推文可以在<a href="https://x.com/_arohan_/status/1804661929694446065">这里</a>查看。</li>
  <li><strong>Gemini 1.5 的炫耀资本</strong>：提到了一款更新的 Gemini 模型，据报道该模型未能进入 I/O 演讲。关于此事的推文可以在<a href="https://x.com/an1lam/status/1792397828733776026">这里</a>找到。</li>
  <li><strong>Anthropic 的 AI 视频</strong>：Anthropic 一直在 YouTube 上分享关于 AI 性格和可解释性等主题的视频。值得关注的视频有 <a href="https://www.youtube.com/watch?v=iyJj9RxSsBY">《AI 的性格应该是怎样的？》</a> 和 <a href="https://www.youtube.com/watch?v=sQar5NNGbw4">《扩展可解释性》</a>。</li>
  <li><strong>对 AI 内容的褒贬不一</strong>：一些成员觉得 AI 相关内容的某些部分很无聊，或者没有预想的那么有趣。尽管有这些批评，大家仍然希望继续制作此类内容。</li>
</ul>

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/_arohan_/status/1804661929694446065">rohan anil (@_arohan_) 的推文</a>：抱歉我必须分享这个。你要知道羊也可以划船！</li><li><a href="https://x.com/an1lam/status/1792397828733776026!">Stephen Malina (@an1lam) 的推文</a>：不敢相信这居然没进 I/O 演讲！更新后的 Gemini 正在通过。</li><li><a href="https://www.youtube.com/watch?v=iyJj9RxSsBY">AI 的性格应该是怎样的？</a>：你如何赋予 AI 助手性格？那到底意味着什么？以及你为什么要这样做？在这次对话中，Stuart Ritchie (Re...</li><li><a href="https://www.youtube.com/watch?v=sQar5NNGbw4">扩展可解释性</a>：科学与工程是密不可分的。我们的研究人员反思了科学与工程进步之间的紧密关系，并讨论了技术...</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1254198777745641603)** (3 条消息):

- **吃吧，小猪们**：一位用户分享了消息 *"eat up piggies"*。在没有进一步解释的情况下，其语境仍不明确。
- **模型中心即将到来**：另一条消息简单地写道 *"model hubs soon 🤗"*。这暗示了与模型中心（model hubs）相关的即将到来的开发或发布。
- **表达困惑**：Nathan Lambert 表达了 *"This makes no sense in so lost"*（这毫无意义，我完全迷失了）的情绪。这表明对之前的消息存在一些困惑或误解。

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1253808200164315196)** (4 条消息):

- **Mixture of Agents 模型引发关注**：一位成员分享了一条关于 Mixture of Agents 模型在 AlpacaEval 排行榜上表现最强的推文，声称它比 GPT-4 便宜 25 倍且更胜一筹。另一位成员认为这很“愚蠢”，质疑该排行榜的合法性，据称该排行榜包含偏见指标。
- **对 Alpaca Eval 的怀疑**：几位成员对 Alpaca Eval 排行榜表示怀疑，指出它可能包含偏见或夸大的性能指标。一位成员直言不讳地表示：“他们在排行榜中加入了各种垃圾（slop）”，并自称为“Alpaca Eval 黑粉”。

**提到的链接**：[Kyle Corbitt (@corbtt) 的推文](https://x.com/corbtt/status/1804199596656410987)：很高兴被正式认可为 AlpacaEval 排行榜上最强的模型。 🙂 [https://tatsu-lab.github.io/alpaca_eval/](https://tatsu-lab.github.io/alpaca_eval/) 引用 Kyle Corbitt (@corbtt) 非常激动地宣布我们的...

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1253827044463083582)** (33 messages🔥):

- **使用 ROCm 分支版本**：成员们讨论了在某些功能下需要使用 [xformers](https://github.com/ROCm/xformers) 和 [flash-attention](https://github.com/ROCm/flash-attention) 的 ROCm 分支版本。一位用户确认 flash-attention 的支持需要 ROCm 5.4+、PyTorch 1.12.1+ 以及 MI200 和 MI300 GPU。
- **Reward Model 对数据生成效果不佳**：简短的交流得出结论，Reward Model 在生成数据方面不值得使用，因为它主要用于数据质量分类。
- **提升 AGI Eval**：一位用户提到计划合成 SAT、GRE 和 MCAT 题目，以潜在地提升小模型的 AGI 评估，并建议也加入 LSAT 题目。
- **Epoch 保存问题**：一位用户报告了训练过程中 Epoch 保存的问题，保存点似乎不一致（如 1.05 Epoch，然后又回到 0.99 Epoch）。这被认为是一个已知但奇特的行为，可能与 steps 计数器有关。
- **在 AMD 上进行 Finetuning**：有人提出了关于在 AMD 硬件上进行 Finetuning 的问题，回复指出 Eric 对此有经验，但尚未确认这是否是一个简单的过程。
  

**提到的链接**：

- [GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention](https://github.com/ROCm/flash-attention)：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 ROCm/flash-attention 的开发做出贡献。
- [GitHub - ROCm/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.](https://github.com/ROCm/xformers)：可黑客化且优化的 Transformers 构建模块，支持组合式构建。 - ROCm/xformers

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages):

lore0012: 我不再遇到这个问题了。

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1253830860449382578)** (4 messages):

- **微调 Qwen2 7b 时的 HeaderTooLarge 错误**：一位成员在运行 `CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess axolotl/ben_configs/qwen2_first.yaml` 时遇到了 `safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge`。该错误发生在尝试加载 checkpoint shards 时。
- **Qwen2 7b 模型的本地目录问题**：当将 `base_model` 设置为 Hugging Face 仓库时，微调配置可以正常工作，但指向本地目录（`/large_models/base_models/llm/Qwen2-7B`）时则失败。即使该文件夹是挂载的 NFS，失败依然存在。
- **对 NVIDIA Megatron-LM Bug 的沮丧**：一位用户在花了一周时间尝试让 megatron-lm 运行并遇到无数错误后表示沮丧。所面临问题的一个例子可见 [GitHub Issue #866](https://github.com/NVIDIA/Megatron-LM/issues/866)，其中讨论了 `convert.py` 脚本中 parser 参数的问题。

**提到的链接**：

- [[BUG] the argument of parser.add_argument is wrong in tools/checkpoint/convert.py · Issue #866 · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/issues/866)：描述 Bug [https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/convert.py#L115](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/convert.py#L115) 必须是 'choices=['GPT', 'BERT'],' 而不是 'choice=['GPT', 'BER...

---

### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1254518443789648024)** (5 messages):

- **新手询问数据集的适用性**：一位正在尝试使用 **axolotl** 微调 **llama2-13b** 的新成员询问了关于数据集格式和内容的问题。他们问道：“这里是询问数据集格式和内容的合适地方吗？”
- **'Alpaca' 数据集的格式示例**：另一位成员分享了一个使用 **JSONL** 进行 **Alpaca** 微调的数据集案例。他们提供了详细的示例，包括指令（instructions）、输入模式（input patterns）和预期输出（expected outputs），并质疑 LLM 是否能泛化诸如 "move to the left" 和 "move a little to the left" 之类的命令。
- **介绍用于高性能 MinHash 的 Rensa**：一位成员兴奋地介绍了他们的侧边项目 **Rensa**，这是一个使用 Rust 编写并带有 Python 绑定的高性能 MinHash 实现。他们声称在数据集去重（deduplication）等任务中，它比 `datasketch` 等现有库快 2.5-3 倍，并分享了其 [GitHub 链接](https://github.com/beowolx/rensa)以寻求社区反馈和贡献。

**提到的链接**：[GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa)：使用 Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大型数据集的高效相似度估计和去重 - beowolx/rensa

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1254711001174245438)** (5 messages):

- **Axolotl 代码库中 Prompt Style 的解释**：关于 `prompt_style` 的查询引出了一段解释，说明它指定了与语言模型交互时 prompt 的格式化方式，从而影响响应的性能和相关性。文中详细列举了 `INSTRUCT`、`CHAT` 和 `CHATML` 等示例，以说明针对不同交互类型的各种 prompt 结构化策略。
- **ReflectAlpacaPrompter 使用示例**：`ReflectAlpacaPrompter` 类示例强调了不同的 `prompt_style` 值（如 "instruct" 和 "chat"）如何决定生成的 prompt 结构。`match_prompt_style` 方法被用于根据选定的样式设置 prompt 模板。

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4809da1a-b260-413e-bdbe-8b82397846e6))：更快地理解代码。

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1254906057256468573)** (1 messages):

- **Llamafile v0.8.7 发布并带来升级**：[Llamafile v0.8.7](https://discord.com/channels/1089876418936180786/1182689832057716778/1254823644320763987) 发布，具有**更快的量化操作**和**错误修复**。文中还提到了 Android 版本的暗示。
- **旧金山举办重大 AI 活动**：**World's Fair of AI** 和 **AI Quality Conference** 将有著名的社区成员出席。提供了 [World's Fair of AI](https://www.ai.engineer/worldsfair) 和 [AI Quality Conference](https://www.aiqualityconference.com/) 的链接。
- **Firefox Nightly AI 服务实验**：Firefox Nightly 用户可以通过正在进行的实验访问可选的 AI 服务。详情可以在 [Nightly 博客](https://discord.com/channels/1089876418936180786/1254858795998384239)中查看。
- **最新的 ML Paper Picks 已发布**：一位社区成员分享了[最新的 ML Paper Picks](https://discord.com/channels/1089876418936180786/1253145681338830888)。
- **预约即将到来的 7 月 AI 活动**：活动包括 [Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134)、[AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249) 以及 [AutoFIx by Sentry.io](https://discord.com/events/1089876418936180786/1245836053458190438)。

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1253796478535860266)** (31 条消息🔥):

- **Llamafile 帮助命令问题**：一位用户报告运行 `llamafile.exe --help` 返回空输出，并询问这是否为已知问题。聊天中没有进一步的讨论或提供的解决方案。
- **在 Google Colab 上运行 Llamafile**：一位用户在最初的困惑后，成功在 Google Colab 上运行了 llamafile，并分享了[他们的示例链接](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g)。
- **Llamafile 重新打包问题**：一位用户对重新打包 llamafile 时的磁盘空间需求表示担忧，建议能够指定不同的提取和重新打包位置。由于 llamafile 文件巨大，这引发了关于是否需要通过环境变量或标志位指定位置的讨论。
- **Cosmopolitan 的新内存管理器**：分享了一个讨论重写内存管理器以支持 Android 的 [GitHub 提交](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca)，并引发了对通过 Termux 在 Android 上运行 llamafile 的兴趣。
- **Mozilla Nightly 博客提到 Llamafile**：[Nightly 博客](https://blog.nightly.mozilla.org/2024/06/24/experimenting-with-ai-services-in-nightly/)提到了 llamafile，并提供了切换 Firefox 配置以启用本地 AI 聊天的指南。这让社区感到兴奋，并建议为新用户提供更清晰的说明。
  

**提到的链接**：

- [未找到标题](http://localhost:8080`): 未找到描述
- [来自 Dylan Freedman (@dylfreed) 的推文](https://x.com/dylfreed/status/1803502158672761113)：新的开源 OCR 模型刚刚发布！微软的这款模型具有我在任何开源模型中见过的最佳文本识别能力，在手写识别方面表现出色。它还能处理各种...
- [Mozilla Builders](https://future.mozilla.org/builders/): 未找到描述
- [发布 llamafile v0.8.7 · Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7)：此版本包含针对量化（quants）的重要性能增强。293a528 Arm 上针对旧版和 k-quants 的性能改进 (#453) c38feb4 针对 i-quants 优化了矩阵乘法...
- [重写内存管理器 · jart/cosmopolitan@6ffed14](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca)：Actually Portable Executable 现在支持 Android。Cosmo 旧的 mmap 代码需要 47 位地址空间。新实现非常通用，支持较小的地址空间（例如...
- [ggerganov - 概览](https://github.com/ggerganov/)：我喜欢大的 .vimrc，我不撒谎。ggerganov 拥有 71 个仓库。在 GitHub 上关注他们的代码。
- [Google Colab](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g): 未找到描述
- [功能请求：支持 Florence-2 视觉模型 · Issue #8012 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/8012)：功能描述：需要支持 Florence-2 系列视觉模型。动机：一个 400M 的模型在基准测试中击败了 15-16B 参数的模型？可能实现：无回复。

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1253791496432517293)** (24 messages🔥):

- **DPO 训练选项可用；ORPO 尚未支持**：当被问及使用 Torchtune 进行 DPO 和 ORPO 训练的选项时，一名成员分享了一个 [ORPO/DPO 数据集](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)，并提到目前尚未支持 ORPO，而 DPO 已经有一个[可用的 recipe](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9)。另一位成员确认了这一点，并补充说 ORPO 需要独立于 supervised fine-tuning 进行实现。
- **多数据集训练及 Epoch 限制**：一名成员询问了关于在多个数据集上训练以及为每个数据集设置不同 epoch 的问题，并被建议使用 *ConcatDataset*。会议强调了不支持为每个数据集设置不同的 epoch。
- **关于在 Llama3 中使用 ChatML 模板的辩论**：关于在 Llama3 中使用 ChatML 模板的讨论正在进行中，涉及 [Mahou-1.2-llama3-8B](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B) 和 [Olethros-8B](https://huggingface.co/lodrick-the-lafted/Olethros-8B)。参与者辩论了使用 instruct tokenizer 和不带特殊 token 的 base model 与使用 ChatML 相比是否合适。
- **Phi-3 模型微调的可行性**：回答了关于使用 torchtune 微调 Phi-3-Medium-4K-Instruct 模型可行性的查询。建议更新 tokenizer 并在 torchtune 中添加自定义构建函数以实现兼容性，如果需要，可以通过将 system prompts 添加到用户消息前部来包含它们。
- **在 Phi-3 中使用 System Prompts 的说明**：有人指出 Phi-3 模型可能没有针对 system prompts 进行优化，但用户仍然可以像往常一样将 system prompts 添加到用户消息前部，以便在 Phi-3 上进行微调。提到 tokenizer 配置中的一个[特定标志](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128)，用于允许使用 system prompt。
  

**提到的链接**：

- [lodrick-the-lafted/Olethros-8B · Hugging Face](https://huggingface.co/lodrick-the-lafted/Olethros-8B)：未找到描述
- [flammenai/Mahou-1.2-llama3-8B · Hugging Face](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B)：未找到描述
- [microsoft/Phi-3-mini-4k-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)：未找到描述
- [torchtune/torchtune/models/phi3/_sentencepiece.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128.)：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k)：未找到描述
- [torchtune/recipes/configs/llama2/7B_lora_dpo.yaml at f200da58c8f5007b61266504204c61a171f6b3dd · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9)：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/html/2404.14219v1#S2))：未找到描述
- [microsoft/Phi-3-mini-4k-instruct · System prompts ignored in chat completions](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/51#665f24e07a329f831b1e3e4e.)：未找到描述
- [microsoft/Phi-3-medium-4k-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)：未找到描述
- [config.json · microsoft/Phi-3-medium-4k-instruct at main](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json)：未找到描述

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1253788818042126418)** (8 messages🔥):

- **WHERE 函数说明**：一名成员询问是否可以使用 `condition * a + !condition * b` 等条件操作来简化 WHERE 函数，随后有人指出 *NaNs* 可能会是一个问题。
- **Intel 支持咨询**：有人询问 tinygrad 对 **Intel 支持**的情况。另一名成员回答可以使用 **opencl**，但目前还不支持 XMX。
- **周一会议概览**：即将于太平洋时间周一上午 9:40 举行的会议重点议题包括 *tinybox* 的更新、新的 profiler、运行时增强以及 **0.9.1 版本发布**计划。具体议程涵盖了 `Tensor._tri` 增强、llama cast 加速，并提到了关于 *uop matcher 速度*改进和 *unet3d* 的 bounty 悬赏。
- **线性代数函数的未来**：一位用户询问了 tinygrad 是否有计划实现通用的线性代数函数，如行列式计算或矩阵分解。*在提取的消息中未给出具体答复。*

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1254621018971050006)** (2 messages):

- **tinygrad 中标记 Buffer view 选项**：分享了一个引入 flag 使 buffer view 在 tinygrad 中变为可选的 commit。该 commit 信息为 *"make buffer view optional with a flag"*，并提供了相关的 [GitHub Actions 运行记录](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120)。
- **lazy.py 的更改引发关注**：一名成员质疑自己是否操作不当，因为他们对 `lazy.py` 的修改导致了正向（好）和负向（坏）的进程重放 (process replay) 输出。他们正在寻求对这种意外行为的解释，暗示其修改可能存在潜在问题。

**提到的链接**：[make buffer view optional with a flag · tinygrad/tinygrad@bdda002](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120)：你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - make buffer view optional with a flag · tinygrad/tinygrad@bdda002

---

### **LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1254510317266796731)** (1 messages):

- **Claude Sonnet 3.5 在 Websim 中表现惊艳**：一名成员在 Websim 中测试 **Claude Sonnet 3.5**，并对该模型的“速度、创意和智能”印象深刻。他们强调了“在新标签页生成”等功能，并分享了尝试用不同标志性时尚品牌的配色方案来“催眠”自己的经历。[Twitter 链接](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413)。

**提到的链接**：[来自 Rob Haisfield (robhaisfield.com) (@RobertHaisfield) 的推文](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413)：我正在“测试” Sonnet 3.5 @websim_ai + 新功能（主要是“在新标签页生成”）。我被这个模型的速度、创意和智能惊呆了 🫨😂 实验室亮点...

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1254828730174406738)** (1 messages):

- **MJCET 启动 AWS Cloud Club**：我们很高兴地宣布，MJCET 在特伦甘纳邦 (Telangana) 启动了第一个 **AWS Cloud Club**！这个充满活力的社区提供资源、培训和 Amazon Web Services (AWS) 的实践经验，为成员提供科技行业职业所需的必备技能。
- **AWS Hero 独家开幕活动**：欢迎参加 2024 年 6 月 28 日上午 10 点至 12 点在 Block 4 研讨厅举行的 AWS Cloud Club MJCET 盛大开幕仪式，届时将有 AWS Community Hero **Faizal Khan 先生**出席。请通过此 [meetup 链接](https://meetu.ps/e/NgmgX/14DgQ2/i) RSVP 以确认出席。

**提到的链接**：[AWS Cloud Clubs MJCET 开幕式，2024 年 6 月 28 日周五上午 10:00 | Meetup](https://meetu.ps/e/NgmgX/14DgQ2/i)：**加入我们，参加 AWS Cloud Club MJCET 的盛大开幕式！** 我们很高兴宣布在 MJCET 启动我们的 AWS Cloud Club 活动！快来探索这个世界吧。

---

---

---

---

---

{% else %}

> 完整的频道细分内容已因邮件长度而截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}