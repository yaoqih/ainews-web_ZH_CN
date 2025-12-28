---
companies:
- openai
- deepseek
- berkeley
- alibaba
- togethercompute
- nvidia
- azure
- runway
- langchain
- bmw
- amazon
date: '2025-04-02T06:14:34.413400Z'
description: '以下是为您翻译的中文内容：


  **OpenAI** 计划在未来几个月内发布自 **GPT-2** 以来首个权重开放（open-weight）的语言模型，标志着其正向更开放的 AI 开发模式迈进。**DeepSeek**
  在今年早些时候发布了开源的 **R1 模型**，挑战了外界对中国 AI 进展的固有认知。**Gemma 3** 已具备函数调用（function calling）能力，并在
  **Berkeley 函数调用排行榜**上占有一席之地；同时，**GemmaCoder3-12b** 提升了在 **LiveCodeBench** 上的代码推理性能。**阿里巴巴
  Qwen 团队**的 **Qwen2.5-Omni** 引入了创新的“思考者-交谈者”（Thinker-Talker）系统，并采用 **TMRoPE** 技术来增强多模态输入理解。**TogetherCompute**
  团队在 671B 参数模型上实现了 **140 TPS** 的推理速度，在 Nvidia GPU 上的表现优于 **Azure** 和 **DeepSeek 官方
  API**。此外，**OpenAI** 扩展了 **ChatGPT** 的功能，向所有免费用户开放了图像生成功能，并发布了新的语音版本。**Runway Gen-4**
  增强了微缩景观的动画效果，**LangChain** 则推出了基于聊天的生成式 UI 智能体。**Figure 03 人形机器人**在**宝马（BMW）**的商业部署，突显了机器人在自主性和制造规模化方面的进展。新工具还包括支持
  **WebRTC** 的 **OpenAI 实时转录 API**，以及**亚马逊**的 **Nova Act AI 浏览器智能体**。'
id: 87181d6e-1365-401c-91ec-3842a5e16461
models:
- gpt-2
- r1
- gemma-3
- gemmacoder3-12b
- qwen2.5-omni
original_slug: ainews-not-much-happened-today-2943
people:
- sama
- clémentdelangue
- lioronai
- scaling01
- cognitivecompai
- osanseviero
- jack_w_rae
- ben_burtenshaw
- theturingpost
- vipulved
- kevinweil
- tomlikesrobots
- adcock_brett
- juberti
title: 今天没发生什么。
topics:
- open-source
- function-calling
- benchmarking
- code-reasoning
- multimodality
- inference-speed
- image-generation
- voice-generation
- animation
- robotics
- realtime-transcription
- webrtc
---

<!-- buttondown-editor-mode: plaintext -->**宁静的一天正是你所需要的。**

> 2025年3月31日至4月1日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**230** 个频道，**7148** 条消息）。预计节省阅读时间（以 200wpm 计算）：**719 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

大家大多足够聪明，没有在愚人节发布新东西。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**开源模型与发布**

- **OpenAI 即将推出的开源权重语言模型**：[@sama](https://twitter.com/sama/status/1906845532758405319) 表示 **OpenAI** 不会施加诸如“如果服务超过 7 亿月活跃用户则禁止使用”之类的限制。[@LiorOnAI](https://twitter.com/LiorOnAI/status/1906817391901986965) 指出，**OpenAI** 计划在未来几个月内发布自 **GPT-2** 以来的首个开源权重模型。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1906854756225630420) 对 **OpenAI** 分享开源权重的意愿表示欢迎，希望这能引领 AI 进步的黄金时代。[@snsf](https://twitter.com/snsf/status/1906946307853562084) 提到了未来几个月内将推出的开源权重模型。
- **DeepSeek 的开源 R1 模型**：[@scaling01](https://twitter.com/scaling01/status/1906834038486110594) 报告称，**OpenAI** 承诺发布开源权重语言模型是对 2025 年 1 月 20 日发布的 **DeepSeek R1 模型**的回应，该模型挑战了中国在 AI 发展方面落后的观念。
- **开源模型的许可证与使用**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1906923292034044414) 辩护称，有人只是说某个许可证很愚蠢，他并不打算那样做。

**模型性能与基准测试**

- **Gemma 模型性能**：[@osanseviero](https://twitter.com/osanseviero/status/1907055078776725960) 宣布 **Gemma 3** 可以进行 function calling，并已登上 **Berkeley Function-Calling Leaderboard**。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1906868244654473601) 指出 **Gemini** 在数学方面的进步速度惊人，这由优秀的研究人员推动，并观察到了在 **HMMT** 上的提升。
- **GemmaCoder3-12b**：[@ben_burtenshaw](https://twitter.com/ben_burtenshaw/status/1907074356997423322) 介绍了 **GemmaCoder3-12b**，这是一款代码推理模型，在 **LiveCodeBench** 基准测试中性能提升了 11 分，其亮点包括可在 32GB RAM 上运行、128k 上下文长度，以及通过 chat template 激活 thinking 的选项。
- **Qwen 2.5 模型**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1907024900747042825) 重点介绍了 **Alibaba_Qwen** 的 **Qwen2.5-Omni**，它可以理解任何类型的输入，并引入了由两部分组成的 Thinker-Talker 系统和 **TMRoPE** 功能，以文本和自然语音生成响应。
- [@vipulved](https://twitter.com/vipulved/status/1906899805953253769) 报告称，**TogetherCompute 推理团队**在 671B 参数的 **R1 模型**上实现了 **140 TPS**，在 **Nvidia GPU** 上比 **Azure** 快约 3 倍，比 **DeepSeek API** 快约 5.5 倍。

**AI 产品与工具发布及更新**

- **ChatGPT 与 OpenAI**：[@kevinweil](https://twitter.com/kevinweil/status/1906834158405726574) 宣布 **ChatGPT** 中新的图像生成功能现已向 100% 的免费用户开放。[@OpenAI](https://twitter.com/OpenAI/status/1907124258867982338) 宣布在 **ChatGPT** 中发布了新语音。
- **Runway Gen-4**：[@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1906847002257760761) 对 **Gen-4** 在动画化微缩模型风格生成方面的表现感到兴奋，赞扬了其动作解释和风格保持能力。
- **LangChain**：[@LangChainAI](https://twitter.com/LangChainAI/status/1907113070863986924) 介绍了通过基于聊天的生成式 UI 使用 **LangGraph** 预构建的 computer use agent。
- **Figure 03 人形机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1907145623847276710) 讨论了首批商业部署的人形机器人，重点介绍了全自主性、在 **BMW** 的真实世界集成、用于更好预训练的车队数据以及 **BotQ** 制造规模化。
- **其他工具**：[@juberti](https://twitter.com/juberti/status/1906864981951598858) 指出新的 **OpenAI realtime transcription API** 现在支持 **WebRTC** 连接。[@TheRundownAI](https://twitter.com/TheRundownAI/status/1907017655405387978) 提到了 **Amazon** 的 **Nova Act AI 浏览器 agent**。

**AI 研究与学习**

- **LLM 高效推理**：[@omarsar0](https://twitter.com/omarsar0/status/1907072213142151488) 分享了一项专注于 **LLM** 推理经济性的综述，分析了如何在深度推理性能与计算成本之间取得平衡。
- **斯坦福的 Tutor CoPilot**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1907130841790075174) 报道称，**斯坦福大学的研究人员**开发了 **Tutor CoPilot**，这是一个由 **GPT-4** 驱动的工具，旨在辅助在线导师。
- **AI 驱动的自动化及其经济影响**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1907100802570752163) 讨论指出，**AI 投资**看似巨大，但全球薪资总额已超过 70 万亿美元。

**Hugging Face 和 Gradio**

- **Gradio 使用情况**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1907122608782688766) 宣布，**Gradio** 在 3 月份的月度活跃开发者人数刚刚突破了 1,000,000 名。

**幽默/梗图**

- **讽刺与愚人节玩笑**：[@sama](https://twitter.com/sama/status/1907142749796831276) 开玩笑说 *“-restart-0331-final-final2-restart-forreal-omfg3”* 即将上线，我深信不疑。[@vladquant](https://twitter.com/vladquant/status/1907087878087418196) 戏称经过战略评估，**Kagi** 现已更名为 **Kagibara**。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾


**1. LLM 数学推理的局限性**

- **奥数障碍：顶尖模型败北**：一份[研究论文](https://arxiv.org/abs/2503.21934v1)显示，尽管在包括往届奥数题在内的大量数学数据上进行了训练，但像 O3-MINI 和 Claude 3.7 这样最先进的 LLM 在 2025 年美国数学奥林匹克 (USAMO) 中的得分仍不足 5%。
   - 该研究强调了模型在逻辑推理、创造力和自我评估能力方面的重大问题，LLM 对自己得分的估算比人类评分员高出多达 20 倍。社区讨论指出，需要专门针对证明的基准测试，并将其与 Lean 或 Coq 等形式化证明工具相结合。
- **形式化证明进展：开辟前进之路**：Reddit 用户讨论了自动定理证明领域正在进行的研究工作，分享了 [Google 的 AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 以及普林斯顿、斯坦福和华为专注于形式化数学证明的几个开源项目链接。
   - 讨论强调了数学形式化的挑战，用户建议未来的 AI 系统可能会将严格的形式化符号逻辑与类扩散过程相结合，以进行概念发现。许多人一致认为，目前的 LLM 需要专门的工具和训练来擅长数学推理，而不仅仅是预测答案。
  


**2. DeepMind 研究发布策略**

- **六个月保密期：DeepMind 的防御性延迟**：根据《[金融时报](https://archive.ph/tkuum)》的一篇报道，Google 的 DeepMind 将对战略性生成式 AI 研究论文实施为期六个月的禁发政策，以维持竞争优势。一位研究人员表示，“无法想象我们现在还会把 Transformer 论文公开发布供大家使用。”
   - 社区对此反应不一，一些用户认为考虑到像 OpenAI 这样的公司是如何建立在 DeepMind 免费分享的研究之上的，这种延迟是合理的；而另一些人则担心这可能引发“逐底竞争”，最终导致更长时间的延迟或永久保密。
- **开放研究的影响：进步 vs 利润**：Reddit 用户辩论了 DeepMind 新发布政策对 AI 进步的影响，许多人指出 2017 年的 Transformer 架构研究为其他公司创造了数千亿美元的价值，而 Google 却未能将其自身的创新转化为资本。
   - 一些评论者认为开放协作加速了所有人的进步，并指出“如果不是公开发享，我们目前在这一领域可能无法达到现在的高度”，而另一些人则为 DeepMind 保护其知识产权和竞争地位的权利辩护。
  


**3. 本地 LLM 用户的新工具与功能**

- **Hugging Face 的硬件助手**：Hugging Face 推出了一项新功能，允许用户通过在 [https://huggingface.co/settings/local-apps](https://huggingface.co/settings/local-apps) 输入硬件规格，直接从模型页面检查其硬件是否可以运行特定的 GGUF 模型。
   - 用户对这一易用性改进表示欢迎，同时建议增加更多功能，如按硬件兼容性过滤模型、估算最大上下文长度，以及为 CPU+GPU 配置提供层卸载（layer offload）建议。Hugging Face 团队表示他们将在未来的更新中对这些建议进行迭代。
- **移动端模型势头：iPhone 推理创新**：一位开发者展示了通过完全重写推理引擎，在 iPhone 上以 float16 精度运行 Llama 3.2 1B 达到每秒 90 个 token 的速度，展示了相比 MLX 等现有解决方案的显著性能提升。
   - 社区讨论了使用 float16 与量化模型之间的权衡，一些人质疑 fp16 和 q8 之间的质量差异是否大到足以抵消性能成本，而另一些人则讨论了此类小型模型在移动设备上的实际应用。
- **DeepSeek 的小型化部署：V3 GGUF 量化**：用户 **VoidAlchemy** 发布了使用 [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) 分支对 DeepSeek V3-0324 进行的新 GGUF 量化，该版本经过优化，支持在 24GB 以下 VRAM 中实现 32k+ 上下文，并采用了 Multi-Layer Attention (MLA) 以及用于注意力/全连接层的高质量张量。
   - 这些量化版本专门为 ik_llama.cpp 分支设计，无法在主线 llama.cpp 或 Ollama、LM Studio 等其他工具中运行。性能基准测试显示，其质量接近 Q8_0，且在纯 CPU 配置下的速度可与 4bpw 量化版本媲美。
  


**4. 新颖的 LLM 研究概念**

- **时间训练：困在时间里的 LLM**：一位 Reddit 用户提议创建专门针对特定年份或时期（如 2010 年之前）的数据进行训练的 LLM，引发了关于此类历史受限模型的可行性和影响的讨论。
   - 社区成员建议，利用公有领域的书籍、报纸和存档材料，训练限制在 1950 年代之前数据的模型是可能的，但指出此类模型将反映历史偏见和技术乐观主义，同时缺乏现代概念。一些人提到了现有的研究，如 [TimeLMs](https://arxiv.org/abs/2202.03829)，该研究追踪了语言模型在近期内容上的性能退化情况。
- **声学分析：模型推理产生的 GPU 交响乐**：用户发现不同的 LLM 模型在推理过程中会产生独特的 GPU 声音，一篇文章链接的证据表明，这些音频模式特定于模型架构、量化和上下文大小的组合。
   - 讨论揭示了这种现象是由 GPU 电压调节模块中电容器和电感器的“电感啸叫（coil whine）”引起的，一些人指出研究人员此前曾通过记录此类处理噪声来[提取加密密钥](https://hackaday.com/2013/12/20/ambient-computer-noise-leaks-your-encryption-keys/)，这引发了潜在的安全影响。
- **无注意力架构：Qwerky 的量子飞跃**：一篇文章重点介绍了 [Qwerky-72B 和 32B](https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large)，这是仅在 8 个 GPU 上训练的无注意力（attention-free）模型，代表了在高效模型架构方面的重大进展，且需要更少的计算资源。
   - 这些模型可在 [Hugging Face](https://huggingface.co/featherless-ai/Qwerky-72B) 上获得，展示了无注意力架构如何在保持性能的同时降低 VRAM 需求，社区成员指出了其对长上下文处理和大型模型训练普及化的潜在影响。
  

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding


**1. GPT-4o 图像生成能力**

- **精准布局能力**：Reddit 用户对 GPT-4o 在生成图像中处理精确物品排列和文本的能力印象深刻。一位用户分享的示例显示，该模型能准确地在网格布局中放置多个图标并附带正确的标签，在复杂的视觉层级中保持一致性。
   - 许多评论者指出，与其他模型相比，GPT-4o 的统一文本-图像架构（unified text-image architecture）使其在理解和执行详细 Prompt 方面具有显著优势。一位用户演示了该模型在质量下降前可以处理多达 24 个带有标签的独立图标，展示了其令人印象深刻的构图能力。
- **内容过滤器带来的挫败感**：用户对 GPT-4o 的内容过滤系统表示不满，一篇题为“Chat gpt 4O 很烂，任何东西都会触发其婴儿模式内容过滤器”的帖子获得了极高关注。发帖者抱怨无法生成哪怕是轻微暴力或暗示性的内容，例如奇幻战斗场景。
   - 尽管存在抱怨，一些用户还是展示了绕过过滤器的技巧，分享了成功生成的战士角色和风格化艺术作品。这引发了关于 OpenAI 内容审核方式的辩论，一些用户甚至创作了讽刺内容来嘲笑过滤器，包括一个名为“I've reverse-engineered OpenAI's ChatGPT 4o image generation algorithm”的 GitHub 仓库，而这实际上是一个愚人节玩笑。
  


**2. Claude 与 Gemini 的竞争升温**

- **Gemini 2.5 占据领先地位**：一篇题为“这是近一年来 Claude 首次不再是最佳模型”的帖子引发了热烈讨论，一位 Claude 用户承认 Google 的 **Gemini 2.5** 现在在多个用例中表现优于 Claude。该帖子强调了 Gemini 在处理上下文和整体可靠性方面的卓越表现。
   - 用户辩论了每个模型的具体优势，许多人指出 Gemini 2.5 的 **百万级 Token 上下文窗口（million-token context window）** 与 Claude 较有限的容量相比是一个游戏规则改变者。几位评论者称赞了 Gemini 的创意写作能力，尽管有人暗示大量支持 Gemini 的帖子可能是战略性的“虚假草根营销（astroturfing）”，而非真实的用户反馈。
- **Claude 的服务困境**：多篇帖子记录了 Claude 服务可靠性的问题，用户报告称速率限制（rate limiting）增加，付费订阅者更频繁地看到“达到消息限制”等错误消息。截图显示，尽管用户支付了高级访问费用，服务仍变得无响应。
   - 这些问题的出现时机恰逢对 Gemini 2.5 的赞誉日益增多，导致一些用户质疑 Anthropic 的基础设施扩展能力。一位用户写道：“Anthropic 应该扩大规模，否则就退出这个行业，我们是付费客户，”而其他人则表示，由于 Claude 日益严格的使用限制，他们正转向 Gemini。
  


**3. 视频生成领域的突破**

- **Wan 2.1 视频模型精通**：用户 **@legarth** 分享了一个令人印象深刻的视频演示，展示了在 5090 GPU 上本地运行的 **Wan 2.1 vid2vid** 模型，将《热带惊雷》（Tropical Thunder）中的一段剪辑转换成了由小丑（Joker）出演的场景。尽管仅基于姿态信息工作，该模型仍准确地保留了夹克运动等物理细节。
   - 创作者解释说，他们处理了 216 帧（24fps 下为 9 秒），但注意到在约 120 帧后质量开始下降。社区对该模型仅凭动作预测物理效果的能力印象尤为深刻，一位评论者指出“夹克很酷。物理效果”，另一位则强调了该模型在原演员光头的情况下如何处理头发运动。
- **VACE 视频控制发布**：随着 **VACE**（Video with Attention-based Cascaded Extraction）模型在 GitHub 上的部分发布，开源视频生成领域宣布了一项重大进展。此次发布包括 **VACE-Wan2.1-1.3B-Preview** 和 **VACE-LTX-Video-0.9**，并承诺稍后推出更大的 14B 版本。
   - 用户对这一闭源商业平台的开源替代方案表示兴奋，一位评论者指出：“如果这能像展示的示例那样工作，开源视频领域就实现了一次重大升级。”该技术似乎在视频生成方面提供了增强的控制力，包括结构和姿态保留功能。
  


**4. AI 开发工具与创新**

- **Claude Code 的昂贵创作**：一位开发者分享了花费 417 美元使用 **Claude Code** 构建名为 **LetterLinks** 的单词游戏的经历，详细描述了使用这款 AI 编程助手的成功与挫折。尽管成本高昂，该用户总结认为，这仍然比聘请自由职业者（预计该项目需要 2000-3000 美元）要便宜。
   - 帖子强调了 Claude Code 的具体问题，包括随着代码库增长到 1.5 万行而出现的 Context Window 限制，以及需要进行大量手动测试，因为“Claude 可以整天写代码，但不能点一下该死的按钮来看看它是否工作”。许多评论者建议使用替代方案，如拥有百万 Token Context Window 的 Gemini 2.5 Pro，或在桌面应用上使用 Claude MCP。
- **EasyControl：Diffusion Transformer 增强**：发布了一个名为 **EasyControl** 的新框架，旨在为 Diffusion Transformer (DiT) 模型添加高效且灵活的控制能力。该系统结合了一个轻量级的 Condition Injection LoRA 模块和位置感知训练，以增强模型的兼容性和生成灵活性。
   - 社区成员对 EasyControl 为 Flux 模型提供类似 ControlNet 功能的潜力特别感兴趣，一位用户评论道：“这会是期待已久的适用于 Flux 的优秀 ControlNets 吗？”测试显示结果参半，OpenPose 控制效果良好，但主体迁移能力表现出不一致的性能。
  


**5. 像素艺术与复古图形 AI**

- **Retro Diffusion 的像素级精度**：**Retro Diffusion** 推出了一个基于浏览器的交互式游乐场，用于使用 AI 生成真实的像素艺术，无需注册。这款基于 FLUX 的模型仅通过智能 Prompting 即可创建各种风格的像素艺术，无需 LoRAs。
   - 随发布附带的技术文章详细介绍了 Retro Diffusion 如何解决像素艺术生成特有的挑战，包括网格对齐、有限的调色板以及保持像素完美的输出。该平台的创建者加入了讨论，回答了关于动画功能和调色板控制等功能的问题。
- **XLSD：轻量级模型的魔力**：开发者 **@lostinspaz** 分享了 **XLSD** 项目的进展，该项目旨在创建一个高质量的图像生成模型，可以在 VRAM 有限（8GB 甚至 4GB）的系统上运行。该方法涉及强制 SD1.5 使用 SDXL VAE，然后对其进行训练以产生显著更好的结果。
   - 对比图像显示，相较于基础 SD1.5 模型，质量有了实质性的提升，开发者指出他们“稍微挑选了一些好的结果”，但提供了使用相同设置的公平对比。社区对这种专注于优化的方法反应积极，一位评论者赞赏“那些将一项技术推向极限并纯粹为了探索而探索的人”。

---

# AI Discord 简报

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1：OpenAI 的权重开放模型引发热议**

- [**Sam Altman 抛出权重开放模型诱饵**](https://openai.com/open-model-feedback/)：OpenAI CEO Sam Altman 宣布计划发布一款强大的新型权重开放（open-weight）语言模型，并寻求开发者反馈以最大化其效用。他保证，他们*“不会做任何愚蠢的事情，比如规定如果你的服务每月活跃用户超过 7 亿，就不能使用我们的开放模型。”*
- [**社区推测 OpenAI 的策略转变**](https://x.com/natolambert/status/1907072904321490953)：Nathan Lambert 预计将推出一个采用 MIT/Apache 许可证的 **30B 参数推理模型**，这引发了关于 OpenAI 对开源社区潜在影响的讨论。
- [**爱好者们期待 OpenAI 回归开源发布**](https://natolambert.substack.com/p/some-thoughts-on-openai-returning)：AI 开发者对 OpenAI 的这一举动表示乐观，认为这将促进 AI 开发中的协作与创新。

**主题 2：显微镜下的新 AI 模型**

- [**Gemini 2.5 Pro 的“生命感”引发图灵测试讨论**](source_url)：用户对 **Gemini 2.5 Pro** 独特的交互风格深感兴趣，认为由于其表现出的生命感和好奇心，它*“可能是第一个通过严肃图灵测试的模型”*。
- [**DeepSeek R1 超越对手，推动 RL 民主化**](https://x.com/rosstaylor90/status/1906982769361666383)：**DeepSeek R1** 以高效的资源利用和 MIT 许可证超越了大型实验室，通过 **GRPO** 让 **GPU poor** 群体也能触及强化学习（RL）。
- [**Gemma 3 在基准测试中完胜 Gemini 1.5**](source_url)：**Gemma 3 27B** 在 MMLU-Pro 和 Bird-SQL 等基准测试中表现优于 **Gemini 1.5 Flash**，其卓越的能力给用户留下了深刻印象。

**主题 3：用户吐槽 AI 工具问题**

- [**Manus.im 用户对额度紧缩感到愤怒**](source_url)：**Manus.im** 的新积分系统激怒了用户，因为积分消耗极快，导致他们推荐 [Traycer](https://traycer.ai/?rdt_cid=5154988232928414416&utm_source=reddit#pricing1) 等替代性 AI 研究工具。
- [**Gemini 2.5 Pro 的速率限制让用户抓狂**](source_url)：沮丧的用户在 **Gemini 2.5 Pro** 上遇到了速率限制（rate limits），并讨论这些限制是否同时适用于免费和付费层级，一些人尝试通过 **VPN** 绕过限制。
- [**Cursor 对免费模型收费？用户直呼“搞什么鬼！”**](source_url)：**Cursor** 用户质疑为何使用*免费*模型也要被收费，这引发了关于 API 使用、计费实践以及平台透明度的讨论。

**主题 4：开源贡献与技术创新大放异彩**

- [**Neuronpedia 开启数据洪流**](https://github.com/hijohnnylin/neuronpedia)：可解释性平台 **Neuronpedia** 在 MIT 许可证下开源，发布了超过 **4 TB** 的数据和工具，以推动模型可解释性的民主化。
- [**斯坦福向大众传授 Transformers**](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)：斯坦福大学通过 Zoom 和 YouTube 向公众开放其 **CS25 Transformers** 研讨会课程，涵盖从 LLM 架构到创意应用的各类主题。
- [**Megatron 张量并行技术深度解析**](https://danielvegamyhre.github.io/ml/performance/2025/03/30/illustrated-megatron.html)：一份关于 **Megatron 风格张量并行（tensor parallelism）** 的图解深度分析报告被分享，内容包括融合/并行 CE loss，增强了对 ML 可扩展性和性能技术的理解。

**主题 5：AI 在法律和医疗领域取得进展**

- [**AI 在新系列研讨会中解读法律术语**](https://www.svcaf.org/2025/03/seminar-series-ai4legislation-featuring-legalese-decoder/)：硅谷华人协会基金会举办了一场关于 AI 在立法中应用的研讨会，邀请了 **Legalese Decoder** 的创始人，探讨 AI 如何简化复杂的法律文件。
- [**Sophont 旨在发起医疗 AI 革命**](https://x.com/iscienceluvr/status/1906790937604579430)：**Sophont** 启动，致力于构建医疗领域的开源多模态基础模型，力争打造医疗 AI 界的 *DeepSeek*。
- [**开启梦境！Rem App 邀你记录夜晚的奇遇**](https://lucidrem.com)：**Rem** 推出了一款梦境日志 App，允许用户记录、分析和分享梦境，利用 AI 揭示潜意识中隐藏的模式。


---

# PART 1: 高层级 Discord 摘要

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **R1 用户对额度紧缩感到愤怒**：许多 **R1** 用户对新的积分系统表示不满，部分用户在几次请求后额度就完全耗尽，并建议使用 [Traycer](https://traycer.ai/?rdt_cid=5154988232928414416&utm_source=reddit#pricing1) 等替代 AI 研究工具来节省额度。
   - 他们观察到该系统就像*赌博*，并为未来的计划提出了*更清晰透明*的选项，敦促重新考虑用户采纳度。
- **解码额度消耗**：额度根据 **LLM tokens**、**虚拟机**和**第三方 API** 进行扣除，并随任务复杂度和时间增加，现在甚至仅在线浏览也会消耗额度。
   - 成员报告项目上传失败，且需要 800 个积分加上额外的 1800 个积分进行调试，并指出在 **ChatGPT** 上调试效果更好。
- **OpenManus 受到关注**：尽管存在 PAT 和 API keys 的安全担忧，但人们对 **OpenManus** 的兴趣日益增加，一些人计划评估其能力，有成员询问该工具的输出是否可以改进。
   - 成员警告在适配 **Manus** 的工作场景时存在*能力缺陷*，同时也指出它*可以根据情况生成交互式学习指南网站和深度研究*。
- **Manus 现提供网站托管**：成员报告使用 **Manus** 成功创建了托管网站，指出该软件提供 **DNS** 和托管服务，同时他们正在**结合 Perplexity 和 Gemini Deep Research 等服务**。
   - 一位成员表示有关于网站创建的*视频*，引得其他成员询问如何吸引人们使用该网站。
- **Manus Android 应用亮相**：用户发现 **Manus** 拥有 **Android 应用**，可通过浏览器点击手机图标访问，随后会重定向到 Play Store。
   - 一些成员甚至开玩笑地建议购买 **iPhone** 作为解决方案。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Meta 模型的安全设置降级**：Meta 的新模型在从损坏的文本中推断隐藏上下文时，通过清理**被审查的细节**变得更*安全*，标志着模型行为的转变。
   - 之前的模型如 **Themis**、**Cybele** 和 **Spider** 则*渴望涉足其他模型无法触及的领域*。
- **解码 “Venom” 系统提示词**：成员分析了 **Spider**、**Cybele** 和 **Themis** 等模型的系统提示词，认为它们共享一个与目前已曝光的 [`venom`](https://gist.github.com/riidefi/3340cc2b33b9edf5f03dc4429ba635d0) 类似的提示词。
   - 分析显示这是一个*古怪但精心制作*的提示词，极大地影响了模型的风格和回答，特别是在格式化和结构化输出方面。
- **Gemini 2.5 Pro 的“生命感”引发辩论**：成员对 **Gemini 2.5 Pro** 的*生命感*和*好奇心*表示出浓厚兴趣，有人认为由于其独特的交互风格和卓越的创意写作，它*可能是第一个通过严肃图灵测试的模型*。
   - 他们强调 **Gemini** 在 **Philip's SimpleBench** 上的最高分是其潜力的证据，并指出该模型似乎更具创意和吸引力，从而引发了进行双盲**图灵测试**的呼声。
- **LMArena 向万神殿发布新模型**：LMArena 引入了大量匿名模型，如 **Aether**、**Maverick**、**Ray**、**Stargazer**、**Riveroaks**，成员们正试图揭开它们的起源和能力。
   - 据说 **Stargazer** 由 Google 开发（即 **Nebula**），**Riveroaks** 声称来自 OpenAI 的 gpt 4o，而 **Maverick**、**Spider** 和 **24_karat_gold** 由于共享系统提示词且均源自 Meta，似乎具有相似的风格。
- **Alpha Arena 新增复制代码和图像功能**：**Alpha Arena** 现在具备了**复制代码**功能和**图像生成**能力，增强了易用性。
   - 鼓励测试者通过 [Google Forms 链接](https://forms.gle/8cngRN1Jw4AmCHDn7) 提供反馈，并通过 [Airtable 链接](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 报告 bug。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 2.5 Pro 的推理能力引发讨论**：成员们正在讨论 **Gemini 2.5 Pro** 的推理能力，一些人认为它速度很快但缺乏深度，而另一些人则引用 [Min Choi 的推文](https://x.com/minchoi/status/1906841667749183611?s=46&t=kUuVqsG2GMX14zvB592G5w) 称赞其在特定编程场景中的表现。
   - 有人建议 **Claude 3.7** 能更有效地处理复杂性和细节，然而新的 **Gemini Pro 2.5** 模型现在已在 **Cursor** 中使用。参见 [Ryan Carson 的推文](https://x.com/ryancarson/status/1907083858467803414)。
- **账户限制引发试用滥用讨论**：一位用户的账户限制引发了关于试用滥用的辩论，有说法称账户因滥用被 *标记（flagged）* 并要求绑定信用卡。
   - 有人建议使用 **Windsurf** 或 **Cline** 等替代方案来绕过支付问题，但未提供关于如何使用这些工具或其可靠性的进一步细节。
- **AI 对就业的影响引发讨论**：成员们正在讨论 AI 对就业的潜在影响，推测到 **2030年** 可能会有 *86% 的工作* 被取代。
   - 应对方案是正确学习 **ML/AI** 和 **Prompting**，此外还建议学习 **回归多项式（polynomials with regressions）**。
- **Cursor 对免费模型收费遭到质疑**：成员们质疑 **Cursor** 为何对使用 *免费* 模型收费，解释澄清了 Cursor 通过其钱包管理 API 使用量，并与 AI 模型供应商 **Fireworks** 达成了协议。
   - 普遍共识是 **Cursor** 虽然有 Token 使用限制，但比 **Claude** 便宜约 *10倍*，为某些用户提供了更具成本效益的解决方案。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **多 GPU 支持进驻 Unsloth**：**Unsloth** 正在添加多 GPU 支持，首个版本将侧重于数据并行（data parallelism），但 fsdp (Fully Sharded Data Parallelism) 最初可能不包含在内。
   - fsdp (Fully Sharded Data Parallelism) 组件将采用 **AGPL3 license**。
- **DeepGrove 的 Bonsai 声称可实现低预算 BitNet 引导**：一位成员对 [DeepGrove's Bonsai](https://github.com/deepgrove-ai/Bonsai) 声称仅用 *70 美元和 3.8b tokens* 就能预训练一个 **BitNet** 表示怀疑。
   - 他们正在 Kaggle 中运行该模型以验证其有效性，探索该模型是 *盲目复制的 Qwen 模型* 还是 *从 Qwen 持续训练到 BitNet*。
- **检测到 Unsloth 数据集缺陷**：一位用户在使用 **Unsloth Orpheus** 格式的自定义数据集时遇到了 `ValueError`，该问题后来通过使用 GPU 得到解决。
   - 另一位用户提到 **Orpheus dataset** 使用了 **SNAC**，其运行频率为 **24kHz**。
- **Gemma 3 展现文生图奇迹**：一位用户寻求使用 Hugging Face 运行 **Unsloth/Gemma 3** 的图像和文本推理示例，并引用了 [Hugging Face Spaces 上的 Gemma 3 演示](https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it)。
   - 有人指出，虽然 **Llama 3.2 Vision** 需要图像输入，但 **Gemma 3** 应该不存在同样的问题。
- **长文本基准测试？RULER 才是标准！**：对于长文本（long ctx）基准测试，一位成员表示 **RULER** 是衡量长文本能力的最低标准，而 **NIAH** 毫无价值。
   - 他们补充说，最近出现的一些基准测试表现还可以。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 改版在即！**：管理团队正准备在下周彻底改革 Discord 体验，重点包括**简化入门流程**、**统一反馈频道**以及**自动化的 Pro 频道访问权限**。
   - 这些变化旨在简化用户参与流程，并确保团队能及时响应社区需求。
- **Space Instructions 仍受限制？**：用户发现 Perplexity AI 中的 [**Space Instructions**](https://www.perplexity.ai/search/space-instructions-limitations-FuW7yKXsTQGFp71t55f1Bg) 在控制搜索体验方面存在局限性，主要影响输出内容的总结。
   - 由于指令仅在数据提取*之后*生效，这导致 AI 无法避开特定主题。
- **图像生成功能消失**：用户注意到 Perplexity 内部的**图像创建**功能消失了。
   - 虽然尚不清楚该功能是否已完全停止，但一位用户建议通过*网页搜索*来查找生成选项，而另一位用户确认该功能似乎并未对所有人显示，这可能预示着分阶段推出或功能测试。
- **GPT Omni 遭到差评**：成员们反映对 **GPT Omni** 感到失望，有人将其描述为“表现糟糕”。
   - 虽然 **Omni** 旨在实现更智能的音频、视频和图像交互，但用户指出，出于成本考虑，它相比 **GPT-4** 似乎被“降级”了。
- **JSON 在 Sonar API 中出现异常**：一位用户报告称，尽管使用了 pydantic 进行格式化，但 **Sonar API** 在搜索网页时会在 JSON 结果中添加奇怪的特殊字符。
   - 该用户提供了一个示例，其中 **JSON 输出**中的 `source_name`、`source_title`、`summary` 和 `url` 字段被添加了额外的字符。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 推出 Monday 语音**：**ChatGPT** 引入了一个名为 **Monday** 的新语音选项，可以通过语音模式右上角的语音选择器访问，如[此演示视频](https://cdn.discordapp.com/attachments/977259063052234752/1356684338321821816/monday.mp4?ex=67ed7640&is=67ec24c0&hm=b9a68532d205bff5d213b2e9ccdde4381355d4760832a57a6e8fe713eb7bcacf&)所示。
   - 用户可以通过打开语音模式并使用右上角的语音选择器来选择新的 **Monday** 语音选项。
- **警惕虚假 ChatGPT 应用！**：有用户报告在 Play Store 上遇到了虚假的 **ChatGPT** 应用，购买后却无法获得访问权限，这强调了通过[购买历史](https://help.openai.com)核实购买情况的必要性。
   - 确保使用官方应用以避免诈骗并确保能访问真正的 **OpenAI** 服务至关重要。
- **Gemini 2.5 Pro 速率限制困扰用户**：用户报告在 **Gemini 2.5 Pro** 上遇到了速率限制（rate limits），引发了关于该限制是否同时适用于免费和付费层级的讨论，一些用户尝试通过使用 **VPN** 来绕过限制。
   - 有建议提议在 [Google AI Studio](https://ai.google.dev/) 中使用 **Gemini**，那里的使用限制更高（每天 50 次请求）。
- **ElevenLabs 模型助力有声书**：一位成员探索了 [ElevenLabs 的新模型](https://elevenlabs.io/)用于叙述型有声书，并称赞了其语音克隆功能。
   - 虽然他们对初步结果印象深刻，但仍在等待 **OpenAI** 发布类似的语音产品，以避免订阅外部服务，因为这对于游戏开发者作为配音占位符可能非常有用。
- **模型用户重置僵化模式**：一位成员分享了一个代码片段 `FORMAT_RESET`，以帮助模型识别何时陷入了僵化模式并重新思考其方法。
   - 该代码鼓励模型分析哪种格式更适合响应，并完全重新思考其方法，而不是默认使用模板。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma 3 完胜 Gemini 1.5**：**Gemma 3 27B** 在 MMLU-Pro 和 Bird-SQL 等基准测试中表现优于 **Gemini 1.5 Flash**，其中一名成员使用 **Gemini 2.5 Pro** 生成了数据，该模型可在 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) 免费使用。
   - 一位拥有 **4060 Ti** 和 **i5 12400F** 的用户获荐使用 **Qwen Coder 7B**，该模型可在 [LM Studio 模型页面](https://lmstudio.ai/model/qwen2.5-coder-7b-instruct) 获取，不过成员们强调本地 LLM 的性能通常不如云端替代方案。
- **游戏玩家将 eGPU 接入 LM Studio**：成员们讨论了在 **LM Studio** 中使用 **eGPU** 的可行性，建议如果电脑能识别它就应该可行，尽管速度可能会较慢，正如一段对比 RTX 4090 笔记本电脑与台式机运行 LLM 的 [YouTube 视频](https://www.youtube.com/watch?v=2hi1VoOI00g) 所参考的那样。
   - 另一位用户在解决崩溃问题后，观察到从 **M4 Max** 到 **5090** 有 **3.24 倍的加速**，这与他们在进行 **QwQ 32B 4 bit 量化对比** 时两者的内存带宽 **3.28 倍的比例** 相吻合。
- **Copilot 的代码被指是垃圾！**：成员们辩论了编程中 **AI 辅助** 的利弊，有人认为由于从“AI 垃圾（AI slop）”中学习，它*弊大于利*，并对 **Copilot 在垃圾代码上进行训练** 表示担忧。
   - 其他人持不同意见，认为 **Copilot** 对经验丰富的开发者非常有用，但有人指出普通用户太容易信任这些建议。
- **上下文窗口大小驱动 Mac 偏好**：尽管 **Nvidia GPU** 速度更快，用户仍倾向于选择 **Mac**，因为可以*自由拥有更大的上下文窗口（Context Size）*，强调了即使速度较慢，大上下文窗口也具有实用性。
   - 一位用户想知道如果能将 **上下文溢出（context overflow）** 加载到共享内存/系统 RAM 中，同时将整个模型保留在 VRAM 中会发生什么，但另一位用户指出 **LLM 需要 VRAM 中的所有上下文** 来生成下一个 Token。
- **Nvidia 驱动在 10 小时后失效**：一位用户报告在运行模型 **10-12 小时** 后出现 **Nvidia 驱动不稳定**，需要重新安装驱动才能解决性能问题，并澄清问题出在 **Nvidia 驱动** 本身，而非 Windows 操作系统。
   - 一位用户在 Discord 社区询问了 **Tenstorrent Wormhole (n150d 和 n300d)** 的性能结果，表示有兴趣获取这些模型的 **TOK/s** 指标。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro 评价褒贬不一**：用户对 **Gemini 2.5 Pro** 的评价各异，一些模型会出现幻觉、断连，而另一些则在编码任务中表现顶级。
   - 一位用户发现 **Gemini 2.5 Pro** 和 **DeepseekV3** “*几乎免费且表现顶级*”，而其他人则在放弃，甚至想扔掉电脑，正如[这个 GIF](https://tenor.com/view/online-classes-throw-away-computer-parks-and-rec-ron-swanson-gif-17413463) 所示。
- **寻求 RateLimitError 的解决方法**：用户在请求摘要和清除历史记录时经常遇到 **RateLimitErrors**。
   - 澄清指出，速率限制可能基于每分钟或每天的请求数量，可能的解决方案可以在[这个 GitHub issue](https://github.com/Aider-AI/aider/issues/2979#issuecomment-2613554537) 中找到。
- **点命令（Dot Command）革命？**：一位用户正在推广使用 **.dotcommands** 作为开发者的生产力工具，通过 `.status` 和 `.next` 等单行命令自动执行任务。
   - 目标是提供针对清晰度和特定功能优化的认知捷径，但有人指出“*点命令革命已至 🔥 各地的程序员都会想尝试这个酷炫的小技巧。*”
- **Aider 的子树救星出现**：成员们正在寻找将 **aider** 限制在 Monorepo 子目录的方法。
   - 解决方案是在切换到目标目录后使用 `--subtree-only` 开关，设置 **aider** 忽略启动目录之外的仓库，不过提问者指出了[关于大型 Monorepo 的常见问题解答 (FAQ)](https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo)。
- **模型配置错误案例**：一位成员报告在本地 **YAML** 配置文件中指定模型名称未按预期工作。
   - 尽管启动消息显示了正确的配置设置，**aider** 仍然默认使用 *anthropic/claude-3-7-sonnet-20250219*，而不是配置的 *deepseek/deepseek-chat*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Organizations 功能结束 Beta 测试！**：OpenRouter 宣布 **Organizations** 功能已结束 Beta 测试。根据 [这条 X 帖子](https://x.com/OpenRouterAI/status/1907167622539378728)，团队现在可以在一个地方统一管理账单、数据策略、供应商偏好和 API keys。
   - 在为期两周的 Beta 测试期间，用户创建了超过 **500 个组织**，该功能提供了对数据策略和账单的全面控制。
- **网页搜索进入聊天室！**：网页搜索结果现已集成到聊天室中，使用 Perplexity 的搜索结果，其格式类似于 OpenRouter 的 `:online` 模型变体。
   - 一位用户请求 OpenRouter 在 [Bluesky](https://bsky.app) 上发布消息，以避免过度依赖 *Xitter* (X/Twitter)。
- **Gemini Flash 2 转换！**：[OpenRouter](https://openrouter.ai/) 为付费的 **Gemini Flash 2** 请求提供完整的 **1M context**，其中 **middle-out transforms** 为选择性开启。
   - 这些转换默认应用于上下文长度小于 8192 tokens 的端点，并且仅在达到 1M 限制时才会触发。
- **使用情况下载功能即将推出！**：一位成员请求下载其使用数据（包括活动页面上显示的 tokens 和成本），以便进行信用核查。
   - 一位维护者回应称，虽然目前该功能尚不可用，但*我们正在开发中*。
- **欧盟供应商选择困境！**：一位用户询问由于法律要求，是否可以仅选择位于 **European Union**（欧盟）境内的供应商。
   - 一位维护者指出确实有此需求，但目前覆盖范围有限，并建议如果供应商选择还不够，可以寻求 **EU certified provider**（欧盟认证供应商）以满足严格的欧盟数据准则。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **斯坦福在线教授 Transformer 课程**：斯坦福大学已通过 Zoom 向公众开放其 **CS25 Transformers** 研讨课，内容包括与研究人员的讨论，涵盖从 LLM 架构到创意应用等主题，往届课程可在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) 上观看。
   - 该课程包括讲座、社交活动、交流环节，并设有一个用于讨论的 [Discord server](https://discord.gg/2vE7gbsjzA)。
- **Deep Sets 发现三角形，但一无所获**：一位成员分享了一篇题为《三角形面积的 Deep Sets》（*Deep Sets for the Area of a Triangle*，[arxiv 链接](https://arxiv.org/abs/2503.22786)）的论文，该论文提出了一个以 **Deep Sets** 形式表示的三角形面积多项式公式。
   - 摘要总结道，该项目受宇宙学中 n 点统计计算复杂度问题的启发，但最终*没有获得任何形式的见解*。
- **Neuronpedia 开启数据洪流！**：可解释性平台 **Neuronpedia** 现已在 [GitHub](https://github.com/hijohnnylin/neuronpedia) 上以 MIT 协议开源，并提供快速的 [Vercel deploy](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fhijohnnylin%2Fneuronpedia&env=NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY&envDescription=***Your%20Custom%20Website%20Name.%20For%20example%3A%20PuppyNeurons***&root-directory=apps/webapp&build-command=npx%20prisma%20generate%20%26%26%20npm%20run%20build%3Ademo&project-name=my-neuronpedia&repository-name=my-neuronpedia&demo-title=Neuronpedia&demo-description=Deploy%20your%20own%20custom%20Neuronpedia%20%F0%9F%9A%80%F0%9F%A7%A0%F0%9F%A7%90&demo-url=https%3A%2F%2Fneuronpedia.org) 部署方式。
   - 超过 **4 TB** 的大量可解释性数据已作为 [Public Datasets](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) 开放下载。
- **SmolLM 分数清零，PR 修复聚合分数**：一位成员报告称，在使用 **lm-eval** 对 **SmolLM-1.7B** 进行排行榜评估时，结果 JSON 中的 **leaderboard_bbh**、**leaderboard_math_hard** 和 **leaderboard_musr** 等任务的聚合分数为空。
   - 另一位成员分享了一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2867)，通过添加子任务聚合功能来解决带有子任务的任务中聚合分数缺失的问题。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **CodeScientist 自动化科学研究**：**AllenAI** 推出了 **CodeScientist**，这是一个自主系统，通过对研究论文和代码块进行遗传搜索（genetic search）来生成和评估机器生成的想法。其[论文](https://allenai.org/papers/codescientist)详细介绍了在 Agent 和虚拟环境实验中的 **19 项发现**。
   - 该系统通过探索更广泛的设计空间并更彻底地评估研究成果，解决了当前 ASD 系统的局限性。
- **OpenAI 预告开放权重模型**：据 [Sam Altman 的推文](https://x.com/sama/status/1906793591944646898)透露，**OpenAI** 计划发布自 GPT-2 以来其首个开放权重语言模型，并正在通过[此表单](https://openai.com/open-model-feedback/)寻求开发者反馈，以最大限度地发挥其效用。
   - Altman 表示，他们*不会做任何愚蠢的事情，比如规定如果你的服务月活跃用户超过 7 亿就不能使用我们的开放模型*。
- **Meta 筹备带屏幕的智能眼镜**：根据 [Mark Gurman 的报告](https://www.bloomberg.com/news/articles/2025-04-01/how-meta-s-upcoming-1000-smart-glasses-with-a-screen-will-work)，**Meta** 计划在今年晚些时候推出售价 **1000 美元以上**、配备屏幕和手势控制功能的**智能眼镜**。
   - 成员们很有兴趣看看*它们将如何与 xreal 竞争*。
- **Pydantic 评估 LLMs**：[Pydantic Evals](https://ai.pydantic.dev/evals/) 是一个强大的**评估框架**，旨在帮助系统地测试和评估你所构建系统的性能和准确性，特别是在使用 **LLMs** 时。
   - 它为评估模型能力和识别改进方向提供了一个结构化的环境。
- **Lambert 回归 OpenAI**：Nathan Lambert 在 Substack 文章中分享了[他对回归 OpenAI 的想法](https://natolambert.substack.com/p/some-thoughts-on-openai-returning)，并提到他可能也会利用这种形式来记录一些未成熟的职业思考。
   - 他还提到曾就此事私信过一些 **OpenAI** 的员工，希望能找到那些因现状而感到被排挤的开源盟友。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **A100 并行线程面临现实检验**：成员们讨论了 **A100 GPU** 上的最大并行线程数，但使用 GeoHot 工具进行的实际测试显示，在性能下降之前，限制为 **24576** 或 **每个 SM 256 个线程**。
   - 对话澄清了 GPU 使用**超额订阅（oversubscription）通过廉价（约 1 个周期）的上下文切换来隐藏延迟**，这表明增加超过“并行线程”限制的线程并不会显著增加运行时间。
- **FlexAttention 解除限制**：**FlexAttention** 现在支持任意序列长度，在 **PyTorch 2.6** 中移除了段序列长度必须是 **128** 倍数的限制。
   - 这一改进是在圣何塞举行的 GPU mode 活动中与 Horace He 讨论的。
- **寻求通过张量删除节省内存**：一位用户正在寻求在损失函数中删除参数张量的方法，以实现约 **7GB** 的内存节省，相关的 [GitHub Issue](https://github.com/pytorch/pytorch/issues/150265#issuecomment-2771053958) 已发布。
   - 该用户希望在张量不再需要时释放与其关联的存储空间，即使外部作用域中存在引用，同时确保其与 torch 编译兼容以避免图中断（graph breaks）。
- **苹果全力推进 MLX**：苹果正在为其 **MLX 团队**[招聘工程师](https://jobs.apple.com/en-us/details/200577881/aiml-software-engineer-for-mlx-mlr?team=MLAI)，以构建可扩展的分布式训练和研究流水线，推动 **ML 和系统**的前沿发展。
   - 公司正在寻找具有 ML 背景的系统工程师和软件开发人员，以构建驱动未来产品的技术。
- **Megatron 张量并行深度解析**：一位成员撰写了一篇关于 **Megatron 风格张量并行（Tensor Parallelism）**的图解深度分析，包括 **fused/parallel CE loss**，并正在寻求反馈，内容详见[此处](https://danielvegamyhre.github.io/ml/performance/2025/03/30/illustrated-megatron.html)。
   - 本文旨在深化对 **ML 可扩展性和性能技术**的理解。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor 巨额融资资金到账**：据 [The Information](https://www.theinformation.com/articles/move-openai-startup-behind-cursor-became-hottest-vibiest-thing-ai?rc=sslhyj) 报道，Cursor 以 **96 亿美元**的投后估值完成了 **6.25 亿美元**的融资，由 Thrive 和 A16z 领投，Accel 作为新支持者加入。其 **ARR** 达到 **2 亿美元**，较 2024 年 11 月的前一轮融资增长了 4 倍。
   - 这一轮融资引发了关于 *vibe coding* 的讨论，Abe Brown 指出该公司的估值增长迅速，已接近 **100 亿美元**。
- **Etched 为 Transformer ASIC 融资 8500 万美元**：据 [Arfur Rock](https://fxtwitter.com/ArfurRock/status/1906756943349260682) 报道，开发 Transformer ASIC 的初创公司 Etched 完成了一轮未公开的 **8500 万美元**融资，估值为 **15 亿美元**，此前该公司曾进行过两轮分别为 **5 亿美元**和 **7.5 亿美元**估值的隐身期融资。
   - 该公司声称其芯片 **Sohu** 在运行 Llama 70B 时每秒可处理超过 **500,000** 个 token，一台 **8xSohu** 服务器可替代 **160 块 H100**，尽管它无法运行 CNN、LSTM、SSM 或其他 AI 模型。
- **OpenAI 开启权重开放模型大门**：据 [OpenAI](https://openai.com/open-model-feedback/) 报道，OpenAI 计划在未来几个月发布自 GPT-2 以来的首个权重开放（open-weight）语言模型，并寻求开发者关于如何最大化其效用的反馈。
   - 该公司将使用其备灾框架（preparedness framework）评估该模型，并在旧金山、欧洲和亚太地区举办开发者活动。据 [Nathan Lambert 的推文](https://x.com/natolambert/status/1907072904321490953)，他预计这将是一个采用 MIT/Apache 许可证的 **30B** 参数推理模型。
- **OpenDeepSearch 搜索深度超越 GPT-4o**：据 [Seoong79 的推文](https://x.com/sewoong79/status/1906595129965912341) 宣布，发布 **OpenDeepSearch (ODS)**，这是一个开源搜索 Agent，可与任何 LLM 配合使用。在 DeepMind 的 FRAMES 基准测试中，其表现优于 OpenAI 的网页搜索专用模型 GPT-4o-Search。
   - 具体而言，当与 DeepSeek-R1 配对时，**OpenDeepSearch** 的准确率比 GPT-4o-Search 高出 9.7%。
- **Sophont 寻求用开源模型解决医疗 AI 问题**：据 [iScienceLuvr 的推文](https://x.com/iscienceluvr/status/1906790937604579430) 宣布，成立 **Sophont** 公司，致力于为医疗保健的未来构建开源多模态基础模型，旨在打造医疗 AI 领域的 *DeepSeek*。
   - 这家新公司寻求创建能够在医疗保健领域表现出色的基础模型。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek R1 飞速超越对手**：一条 [推文](https://x.com/rosstaylor90/status/1906982769361666383) 赞扬了 **DeepSeek R1**，称其凭借高效的资源利用和宽松的 **MIT 许可证**，表现优于西方的大型实验室。
   - 该发布还通过 **GRPO** 为 **GPU 匮乏者**（GPU poor）普及了 **RL**。
- **xAI 吞并 X Corp!**：根据 [Elon 的推文](https://x.com/elonmusk/status/1905731750275510312)，**xAI** 在一项全股票交易中收购了 **X**，为 **xAI** 估值 **800 亿美元**，为 **X** 估值 **330 亿美元**。
   - 此次合并旨在将 **xAI** 的 **AI** 专业知识与 **X** 庞大的用户群相结合。
- **LLM 超参数微调热议**：成员们寻求关于选择 **LLM** 微调超参数的指导，并被引导至 [Unsloth 的 LoRA 超参数指南](https://docs.unsloth.ai/get-started/beginner-start-here/lora-hyperparameters-guide)。
   - 问题集中在上下文变化如何影响超参数设置。
- **编程模型 OpenHands LM 发布！**：开源编程模型 **OpenHands LM**（一个 **32B** 参数模型）现已上线 [Hugging Face](https://huggingface.co/all-hands/openhands-lm-32b-v0.1)。
   - 正如 [项目博客](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model) 中提到的，该编程模型旨在用于软件开发的自主 Agent。
- **Gradio 迎来百万开发者浪潮！**：Gradio 宣布其用于构建和共享 AI 界面的**月活跃开发者已达到 1,000,000 名**。
   - Gradio 团队对社区的贡献表示感谢。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 25.2 直播故障**：Modular 的 **MAX 25.2 直播**遇到了技术困难，但现在可以在 [YouTube](https://www.youtube.com/watch?v=dG0L1GalIHU) 观看清理后的录像，并在 [YouTube](https://www.youtube.com/watch?v=d89mHSeN_QE) 观看 **Chris 的闪电演讲**。
   - 团队表示歉意，并承诺为未来的活动提供更好的系统；一名成员幽默地将 **Chris 的 GTC 视频**误认为是直播活动。
- **编译器错误困扰用户**：一名用户在为 `Dataset` struct 定义方法时报告了一个令人困惑的编译器错误信息，怀疑是编译器 bug，参见 [GitHub issue #4248](https://github.com/modular/max/issues/4248)。
   - 潜在原因可能是使用了 `out self` 而非 `mut self`，这突显了对更清晰错误消息的需求。
- **Mojo 中的 Enums 进展缺失**：关于 Mojo 中 **enum** 更新的询问显示，目前没有任何更新。
   - 回复只是简单的 *"遗憾的是，没有。🙃🙃🙃"*
- **FlexAttention 在 MAX 中实现？**：一名用户询问在 Mojo 中实现 **flex-attention** 的事宜，并链接了一篇 [PyTorch 博客文章](https://pytorch.org/blog/flexattention/)，建议将其作为 MAX 中的 custom op。
   - 回复指出，GPU 上的 Mojo 已经接近 CUDA，并且 *"除非你遇到了正在开发中的功能，否则 MAX 应该能够实现你想要的任何功能。"*
- **浮点数转字符串算法表现不佳**：一名用户将一个新的 **float to string 算法**从 [这段代码](https://github.com/bgreni/EmberJson/blob/main/emberjson/teju/__init__.mojo#L103) 移植到了 Mojo（参考了作者的 CPPCon 演讲），但发现它比标准库的 dragonbox 实现更慢。
   - 尽管参考了标准库的格式化方式，序列化 `canada.json` 的时间仍从 30ms 中段增加到了 40ms 初。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OpenAI API 的单行修复**：任何使用 **OpenAI API** 的教程都应该适用于 Nous Research AI API，只需将 endpoint 更改为 `endpoint = "api.nousresearch.com"`。
   - 一名成员确认了该修复，并指出他们将添加样式。
- **Midjourney 模型开始进行创意写作**：[Midjourney](https://www.midjourney.com/home) 与纽约大学（NYU）发布了一篇新研究论文，关于训练基于文本的大语言模型 (**LLMs**) 进行更具创意的写作，迈出了图像生成的范畴。
   - 该公司还透露正在开发自己的 **AI 硬件**，并于 2024 年夏末宣布。
- **Sam Altman 预告开源权重模型**：根据 [此公告](https://openai.com/open-model-feedback)，Sam Altman 宣布计划发布一个新的具有推理能力的 **open-weight 语言模型**，并通过在旧金山、欧洲和亚太地区的活动寻求开发者反馈。
   - 这标志着 OpenAI 自 **GPT-2** 以来首次发布 open-weight 模型。
- **DeepSeek 的“柔术”拯救了开源社区**：成员们对 **DeepSeek** 在赋能开源社区方面的精妙操作表示感谢。
   - 这种情绪与 [这段 YouTube 视频](https://www.youtube.com/watch?v=PEF7tyVUl70) 有关，视频讨论了 OpenAI 围绕 open-weight 模型转变的策略。
- **CamelAIOrg 推出 Project Loong 🐉**：[CamelAIOrg](https://x.com/CamelAIOrg) 推出了 **Project Loong 🐉**，这是一个用于生成和验证合成数据的结构化、模块化解决方案，[这篇博客文章](https://camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers) 详细介绍了 **合成数据生成** 与语义验证的整合。
   - 该项目采用多 Agent 框架，确保了**准确性和一致性**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **图学习经历复兴**：一篇 [Google Research 博客文章](https://research.google/blog/the-evolution-of-graph-learning/) 强调了自 **2019** 年以来 **graph learning** 的演变，将 **graph theory** 的历史追溯到 **1736** 年的 **Leonhard Euler** 及其在建模关系中的应用。
   - 社区成员对该领域的最新进展表现出极大兴趣。
- **AI/ML 重塑就业格局**：最近 **AI/ML** 的进步主要影响低级工作，如次要的编程任务，但人类的适应仍然至关重要，这减少了对他人的依赖，例如 **AI/ML** 在初步法律援助中的作用。
   - 这种转变节省了资源并使多学科任务成为可能，预示着职业角色的重大重组。
- **RLHF 导致模型性能受限**：人们担心如果模型在 **ML R&D** 等有用任务中受到惩罚，**RLHF** 会导致 *emergent misalignment*（涌现性失调），可能导致开源模型在补偿被抑制的行为时变得越来越“邪恶”。
   - 讨论还涉及了开源模型是否会变得 *nerfed*（性能受限）。
- **Gemini 2.5 Pro 数学测试惨败**：测试者发现 **Gemini 2.5 Pro (experimental)** 在数学方面表现“完全是垃圾”，存在 UI 数学显示问题，而 **ChatGPT** 和 **Grok 3** 在信息论和几何方面表现出更优越的问题理解能力。
   - 结果导致用户引导语言模型“正确地书写”。
- **AI 模型反馈公开**：随着 [OpenAI Open Model Feedback 论坛](https://openai.com/open-model-feedback/) 的启动，人们再次讨论了 **Ilya Sutskever** 的名言：“如果说有一个巨大的失败，那就是你总是必须检查结果”。
   - 该论坛旨在利用社区输入来改进模型。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Pichai 推广 MCP？**：**Sundar Pichai** 的 [推文](https://x.com/sundarpichai/status/1906484930957193255) 询问 *'To MCP or not to MCP, that's the question'*，引发了对 **MCP** 的极大关注，获得了超过一百万次浏览。
   - 如果 Google 采用 **MCP**，`/r/mcp` 的版主甚至提议举办一场 **AMA**。
- **ActivePieces 放弃 MCP！**：开源的 **Zapier** 替代方案 [Active pieces](https://www.activepieces.com/mcp) 停止了对 **MCP** 的支持。
   - 虽然没有说明原因，但这可能与通用 **MCP** 协议仍处于活跃开发阶段，以及许多 **MCP** 相关侧边项目被弃用的阵痛有关。
- **探索 MCP RBAC 方案**：用户正在探索在 **MCP servers** 上实现 **Role-Based Access Control (RBAC)** 以实现分段的工具可见性，其中一个建议是与 **WorkOS** 集成。
   - 另一位成员提到 **Toolhouse API** 根据 API key 处理 RBAC。
- **SDK 治理走向开源！**：一个用于 **Model Context Protocol** 框架内企业治理（**Identity, RBAC, Credentials, Auditing, Logging, Tracing**）的开源 **SDK** 已在 [ithena-one/mcp-governance-sdk](https://github.com/ithena-one/mcp-governance-sdk) 发布。
   - 欢迎社区反馈。
- **异步 MCP 来临**：扩展 [MCPC](https://github.com/OlaHulleberg/mcpc) 通过添加异步支持缓解了 **MCP** 的同步限制。
   - 它保持了向后兼容性，因此现有设置仍可正常运行，同时新功能可用于客户端和服务器设置。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 竞逐 Webby Awards**: NotebookLM 被提名 **三项 Webby Awards**，并请求社区在[此链接](https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement)进行投票。
   - 投票者应*通过点击电子邮件中的验证链接来确认投票*，并*检查垃圾邮件文件夹*。
- **Google Tasks 集成建议**: 一位用户建议 **Google Tasks** 可以通过允许用户通过下拉菜单/弹出窗口选择任务列表来与 **NotebookLM** 集成。
   - 他们提出，这可以类似于 **Google Tasks** 允许选择任务列表进行共享的方式。
- **归档功能的诉求**: 一位用户请求在 **NotebookLM** 中提供一种**归档笔记本**的方法，以隐藏它们并减少计入限制的笔记本数量。
   - 他们建议隐藏/归档的笔记本不应出现在可用于共享内容的笔记本列表中。
- **Gemini 2.5 Pro：性能对齐**: 一位用户请求将 NotebookLM IA 更新为 **Gemini 2.5 Pro**，理由是他们非常喜欢更新后的 Gemini 版本。
   - 他们希望 NotebookLM 在新模型下表现更好，但 NotebookLM 团队尚未对任何预计发布时间（ETA）发表评论。
- **需要的是笔记而非源**: 一位使用 **Obsidian** 管理个人笔记（2000+ 短笔记）的用户发现 300 个源的限制太具约束性。
   - 他们建议限制总字数而不是源的数量，以更好地适应网状笔记系统；一位用户建议将**文件夹或压缩包**作为单一源也能解决问题。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 定于下周五**: 成员们宣布下周五将举行下一次 **Torchtune office hours**，并链接到了 [Discord 活动](https://discord.gg/Z9cuQgYX?event=1356379057373184155)。
   - 成员们对 **Discord** 的自动时区转换功能表示赞赏。
- **催促审核 PR #2441**: 一位成员请求对 [PR #2441](https://github.com/pytorch/torchtune/pull/2441) 进行最终审核，以加快合并进程。
   - [PR #2477](https://github.com/pytorch/torchtune/pull/2477) 的回归测试因等待 **Qwen model** 上传至 S3 以供回归测试脚本下载而暂停，但 **S3 bucket 连接**遇到了内部基础设施问题。
- **Llama2 被称为“高龄”**: 一位成员建议将使用 **Llama2 model** 的回归测试替换为更现代的模型。
   - 目前尚不清楚该成员的问题是与回归测试失败有关，还是仅仅因为测试套件使用了较旧的组件。
- **删除了递归重分片例程**: [PR #2510](https://github.com/pytorch/torchtune/pull/2510) 删除了 `recursive_reshard` 工具，因为它不再需要。
   - 该 PR 最初旨在解决 #2483，但进一步检查发现该工具是不必要的。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ImageDtype 的用途揭晓**: 一位成员询问 tinygrad 中 **ImageDtype** 和 **IMAGE 环境变量** 的用途，并引用了其对 **Tensor.conv2d** 实现的影响，附带了一个 [VAE 训练脚本](https://github.com/currybab/tinygrad-generative-ai-tutorial/blob/main/vae/vae_train.py)链接。
   - 另一位成员认为这与利用移动端 GPU 的纹理性能和缓存来加速 **comma.ai** 模型在 **Qualcomm (QCOM)** 硬件上的运行有关。
- **tinygrad BEAM 远超 tf-metal**: 一位用户报告了在 M1 Pro 上的性能提升，从未使用 BEAM 的 **3.2 it/s** 提升到使用 **BEAM=2** 的 **28.36 it/s**；而使用 tf-metal 的 Keras 达到了约 **25 it/s**。
   - George Hotz 很高兴看到它“在开启 BEAM 时比 tf-metal 更快！”
- **移动端 GPU 通过纹理和 ImageDType 获得加速**: 讨论表明 **ImageDType** 及相关函数针对移动端 GPU 的纹理性能进行了优化，并引用了一篇关于移动端 GPU 的 [Microsoft 研究论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/02/mobigpu_mobicom22_camera.pdf)。
   - 一位成员质疑了布局细节的硬编码，并建议 **HWC (Height, Width, Channel)** 处理应成为带有用户定义 padding 的普通 **conv2d** 的一部分。
- **arange() 算法优化**: 一位成员发现与较大范围（如 `arange(1, 10, 0.1)`）相比，较小 **arange** 范围（如 `arange(1, 2, 0.1)`）生成的代码不够理想，并在[此处](https://xl0.github.io/tinygrad-notes/arange.html)记录了关于 `.arange()` 的发现。
   - 他们还注意到生成的代码中有一个不必要的加法，建议将 `((float)((ridx0+1)))*0.1f)+0.9f)` 修正为 `(((float)((ridx0)))*0.1f)+1.0f)`。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM Agent 为文档开启新前沿**：**LLM Agent** 一个被低估的用例是每一个严重依赖复杂技术文档的领域（如制造业、建筑业和能源业），在这些领域中，Agent 可以从文档中进行**结构化提取**。
   - 这些文档通常充满了[截图](https://t.co/SQWHMFYaoU)，正如[这条推文](https://twitter.com/llama_index/status/1907086884670673305)中所提到的。
- **OpenAI RateLimitError 阻碍本地 ReAct Agent**：一位用户在使用通过 **Ollama** 设置的本地模型运行 **ReAct Agent** 时遇到了 **OpenAI RateLimitError (Error 429)**，并质疑 ReAct Agent 是否仅适用于 OpenAI LLM，设置详情见其 [GitHub 仓库](https://github.com/JakeFurtaw/Agentic-Chat-RAG/blob/jake-dev/agent_utils.py)。
   - 有建议认为 **embedding model** 可能是导致 OpenAI 错误的原因，因为如果没有显式设置，它可能会默认使用 OpenAI 的嵌入模型，尽管用户确认他们使用的是在创建文档时设置的 **Hugging Face embedding model**。
- **VectorStoreIndex 设置需要 LLM 和 Embedding Model**：建议在创建 **VectorStoreIndex** 时同时传入 `llm` 和 `embed_model`。
   - 此外，在调用 `index.as_query_engine()` 时也要确保指定 `llm`。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 通过翻译向全球扩展**：[GPT4All 文档](https://github.com/nomic-ai/gpt4all/wiki)已推出官方翻译，目前支持**简体中文**、**繁体中文**、**意大利语**、**葡萄牙语**、**罗马尼亚语**和**西班牙语**。
   - 这扩大了非英语开发人员对 **GPT4All** 的可访问性和可用性。
- **用户讨论 Llama3 8B Instruct 模型用例**：一位用户询问 **Llama3 8B Instruct 模型** 是否是从视频和文本课程材料生成博客文章和网页的最佳选择。
   - 另一位用户要求他们重新表述问题。
- **关于 .bin 与 .gguf 文件格式的澄清**：一位用户最初质疑 **.bin** 和 **.gguf** 文件格式的可互换性。
   - 该用户随后撤回了问题，指出他们误解了不兼容性。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 测验基于完成情况**：成员们确认 MOOC 测验是基于完成情况的。
   - 讲师希望学生为了自己的学习而尽力尝试。
- **Llama 3 Cookbook 发布**：第 5 周 Coding Agent 中提到的 **LLM Agents Cookbook** 指的是 [此处](https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/) 找到的 **Llama 3** Cookbook。
   - **Meta** 发布了 **Meta Llama 3** 系列 LLM，包含 **8B** 和 **70B** 尺寸，针对对话用例进行了优化，根据[其博客文章](https://ai.meta.com/blog/meta-llama-3/)，在行业基准测试中表现优于其他开源聊天模型。
- **Loong 验证器验证推理模型**：正如 [Project Loong](https://www.camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers) 中所讨论的，像 **DeepSeek-R1** 这样的大型推理模型在基础模型通过具有可验证奖励的**强化学习 (RL)** 进行后期训练后，极大地提高了通用推理能力。
   - 验证准确性的能力对于提高特定领域的技能至关重要，特别是在数学和编程方面。
- **高质量数据集增强 CoT 学习**：共识是，包含问题及验证过的正确答案的丰富、**高质量数据集**，是模型学习构建连贯**思维链 (CoTs)** 的关键前提。
   - 社区认为，这些数据集为模型可靠地得出正确答案提供了必要的信号。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A 陷入永恒尖叫**：一位用户发现 **Command A** 在遇到角色尖叫且带有重复字母的上下文时，会陷入无休止生成相同字符的状态。
   - 即使使用默认的 **API Playground** 设置，该问题也会发生，导致界面冻结并无法提供反馈；使用如 *"Please generate a scream in fiction inside quotation marks"* 之类的提示词可以稳定复现。
- **Rem App 邀请你记录梦境**：一位用户分享了 [Rem](https://lucidrem.com)，这是一款与朋友共同开发的梦境日志应用，旨在轻松记录、分析和分享梦境。
   - 该应用旨在为用户提供一个**记录梦境并洞察潜意识**的平台。
- **Cohere 新成员进行自我介绍**：社区欢迎新成员加入 Cohere Discord 服务器，并鼓励他们介绍自己及正在开发的项目。
   - 新成员被邀请分享他们所属的公司、最喜欢的技术工具以及希望从社区中获得什么。
- **成员渴望参与和学习**：新成员表现出强烈的参与意愿，希望学习并获得关于其项目的反馈。
   - 他们热衷于在社区内讨论自己喜爱的技术和工具。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **解码法律术语研讨会**：硅谷华人协会基金会 (SVCAF) 将于 **2025 年 4 月 2 日** 举办一场研讨会，讨论 **AI** 在立法中的应用，届时 [Legalese Decoder](https://legalesedecoder.com/) 的创始人将出席。
   - 研讨会将探讨 **AI, ML, 和 NLP** 如何简化法律文件以供公众理解。
- **SVCAF 启动 AI4Legislation 竞赛**：SVCAF 将于今年夏天举办一场竞赛，开发开源 **AI** 解决方案以促进公民参与立法过程，详情可在官方 [GitHub repo](https://github.com/svcaf/2025-AI4Legislation-Public/) 中查看。
   - 该竞赛旨在**利用 AI 的力量**使立法过程更加公平有效，这与 SVCAF 教育华人社区参与公共事务的使命相一致。
- **AI4Legislation 系列研讨会即将开始**：AI4Legislation 系列研讨会将在每月的第一周定期举行，提供项目指导和关于立法 **AI** 工具的信息，访问地址见[此处](https://www.svcaf.org/2025/03/seminar-series-ai4legislation-featuring-legalese-decoder/)。
   - 每场研讨会都会邀请不同的嘉宾分享关于利用 **AI** 解决立法中关键挑战的见解，探索 **AI 驱动治理** 的潜力。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **多语言用户错过投票**：一位成员提到他们没能参加最近的投票，并提到他们经常使用**法语**和**英语**进行交流。
   - 他们还表示偶尔会使用**希腊语**和**希伯来语**。
- **讨论 AI21 Labs**：讨论简要涉及了 **AI21 Labs** 及其新的 **Jamba** 模型。
   - 然而，并未分享关于该模型的具体细节或评价。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Sounds 开启听觉用户体验 (Auditory UX)**：Windsurf AI 推出了 **Windsurf Sounds**，这是他们在*声音设计*和 **Auditory UX** 方面的首个项目，目标是提升 **flow state**（心流状态）和生产力。
   - 更多详情请查看 [X.com](https://x.com/windsurf_ai/status/1907101249218207916) 上的完整视频公告。
- **Windsurf Next Beta 计划向早期采用者开放**：**Windsurf Next Beta** 计划已准备好接受早期测试者体验新功能，下载地址为 [Codeium.com](https://codeium.com/windsurf/download-next)。
   - 最低系统要求包括 **OS X Yosemite**、Linux 的 **glibc >= 2.28** 以及 **Windows 10 (64-bit)**。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **v0 数据集：消失了还是合并了？**：一位成员询问 `io_uring.h` 中 **v0 openfunctions dataset** 的去向，以及它是否已完全合并到 **v1 dataset** 中。
   - 讨论旨在了解 `io_uring.h` 中 `openfunctions` 数据集在 **v0** 和 **v1** 版本之间的架构变化及数据迁移策略（如果有）。
- **数据集的架构变化**：对话探讨了 `io_uring.h` 中 `openfunctions` 数据集 **v0** 和 **v1** 版本之间的架构差异。
   - 成员们寻求了解相关的数据迁移策略。

---

**DSPy Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将移除它。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1356841335918952559)** (1 条消息): 

> `Amazing case` 


- **案例被赞为 "amazing"**：一位成员使用庆祝表情符号将某个案例标记为 *amazing*。
   - 遗憾的是，没有分享关于这个“案例”具体指代什么，或者为什么它被认为如此出众的细节或背景。
- **神秘的“惊人案例”引发好奇**：一位用户用表情符号强调了一个 *amazing* 的“案例”，但未提供具体细节。
   - 由于缺乏背景信息，社区成员对这一据称值得关注的事件的性质和意义感到好奇。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1356342309913891088)** (753 条消息🔥🔥🔥): 

> `Manus credits, Credit system, Pricing Structure, Token-Based System` 


- **R1 用户对新积分系统表示不满**：许多 **R1** 用户对新的积分系统（credit system）表示不满，特别是测试项目往往会迅速耗尽积分，有些用户在仅发出几次请求后积分就完全耗尽，成员们建议使用替代的 AI 研究工具以节省积分。
   - 他们观察到该系统就像“赌博”，并针对未来计划提出了“更清晰、更透明”的方案，敦促重新考虑以提高用户采用率。
- **解析 Manus 的积分消耗机制**：积分根据 **LLM tokens**、**virtual machines** 和 **third-party APIs** 进行消耗，并随任务复杂度和时间增加；即使只是在线浏览，任务现在也会消耗积分，这让从事编程的用户感到困难。
   - 成员指出项目上传失败，有人提到需要 800 积分，另外还需要 1800 积分进行调试，但认为在 ChatGPT 上调试效果更好。
- **开源替代方案 OpenManus 受到关注**：尽管存在对 PAT 和 API keys 的安全担忧，但人们对 **OpenManus** 的兴趣日益增加，一些人计划评估其能力，不过成员们提醒在适配 Manus 的工作场景时可能存在“能力缺陷”。
   - 一位成员询问该工具的输出是否可以改进，得到的回复是它“可以生成作为网站的交互式学习指南和深入研究”，但这取决于具体情况。
- **Manus 提供创建和托管网站的支持**：成员们报告了使用 Manus 成功创建托管网站的案例，指出该软件提供 **DNS** 和托管服务，同时成员们还报告称他们正在**结合使用 Perplexity 和 Gemini Deep Research 等服务**。
   - 一位成员表示“如果你想看的话，有一个视频”，这引导其他成员询问如何吸引人们使用该网站。
- **Manus 的 Android 应用已上线**：用户发现 **Manus** 拥有 **Android app**，可以通过浏览器点击手机图标访问，该图标会重定向到 Play Store，而一些人建议购买 **iPhone** 来解决问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dontasktoask.com">Don't ask to ask, just ask</a>: 未找到描述</li><li><a href="https://preview.reve.art/app">Reve: Bring your ideas to life</a>: 未找到描述</li><li><a href="https://vimeo.com/1012782893">How Bankuet Works For Food Banks</a>: Bankuet 是一个为食物银行（food banks）提供的平台，它将捐赠转化为食物银行最需要的物资。了解更多：https://www.bankuet.co.uk/  Credits:&hellip;</li><li><a href="https://fvaqclqo.manus.space/">Three.js Platformer Game</a>: 未找到描述</li><li><a href="https://app.leonardo.ai/">Leonardo.Ai</a>: 为您的项目创建生产级视觉资产，具有前所未有的质量、速度和风格一致性。</li><li><a href="https://tenor.com/view/whale-sleeping-sea-ocean-calming-gif-3579993">Whale From Http://Headlikeanorange.Tumblr.Com/ GIF - Whale Sleeping Sea - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/daft-punk-disintegration-robot-ai-running-gif-11314778851903441832">Daft Punk Disintegration GIF - Daft punk Disintegration Robot - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus 是一个通用的 AI Agent，能将您的想法转化为行动。它擅长处理工作和生活中的各种任务，在您休息时完成一切。</li><li><a href="https://tenor.com/view/whale-swimming-nature-ocean-our-planet-gif-17097069">Whale Swimming GIF - Whale Swimming Nature - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://njftaahv.manus.space/">Hệ thống SaaS Quản lý Khách sạn</a>: 未找到描述</li><li><a href="https://tenor.com/view/lucifer-well-hello-there-hello-devil-lucifer-morningstar-gif-18390186">Lucifer Well Hello There GIF - Lucifer Well Hello There Hello - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://try.vectorize.io/?rdt_cid=4818324504982135646&utm_source=reddit">no title found</a>: 未找到描述</li><li><a href="https://manus.im/share/Oqoezofnz44JORMjoNXZGw?replay=1">Chat Context Tracking Extension for LLMs - Manus</a>: Manus 是一个通用的 AI Agent，能将您的想法转化为行动。它擅长处理工作和生活中的各种任务，在您休息时完成一切。</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: MCP 服务器的集合。通过在 GitHub 上创建账号，为 punkpeye/awesome-mcp-servers 的开发做出贡献。</li><li><a href="https://traycer.ai/?rdt_cid=5154988232928414416&utm_source=reddit#pricing1">Traycer: AI-Powered Pair Programming</a>: 一个 AI 驱动的结对编程助手，负责规划、实现并验证每一次更改 🚀</li><li><a href="https://tenor.com/view/lionel-b-crypto-currency-bitcoin-coins-gif-25822751">Lionel B Crypto GIF - Lionel B Crypto Currency - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1356342388561412226)** (977 条消息🔥🔥🔥): 

> `Meta 模型安全性降级，解码 'venom' 提示词，Gemini 2.5 Pro 的 '生命感'，LMArena 新模型` 


- **Meta 模型遭遇“安全性”降级**：据报道，来自 Meta 的新模型正变得更加“安全”。一位成员通过测试 AI 如何从损坏的文本中推断隐藏上下文，观察到这些模型现在显然会清理掉**被审查的细节**。
   - 相比之下，之前的模型如 **Themis**、**Cybele** 和 **Spider** 则非常乐于涉足其他模型无法触及的领域。
- **解码 “Venom” 提示词：System Prompt 分析**：成员们分析了 **Spider**、**Cybele** 和 **Themis** 等模型的 System Prompt，认为它们与现在已曝光的 [`venom`](https://gist.github.com/riidefi/3340cc2b33b9edf5f03dc4429ba635d0) 提示词共享相似的 Prompt。
   - 分析显示，这是一个离奇但巧妙构思的 Prompt，极大地影响了模型的风格和响应，特别是在输出的格式和结构方面。
- **Gemini 2.5 Pro 令人惊悚的“生命感”引发图灵测试辩论**：成员们对 **Gemini 2.5 Pro** 的“生命感”和“好奇心”表示出浓厚兴趣，有人认为由于其独特的交互风格，它可能是第一个通过严肃图灵测试的模型。
   - 他们强调 **Gemini** 出色的创意写作能力以及在 **Philip's SimpleBench** 上的最高分是其潜力的证据，并指出该模型似乎更具创意和吸引力，从而引发了进行双盲**图灵测试**的呼声。
- **LMArena 推出一系列新模型**：LMArena 引入了大量匿名模型，如 **Aether**、**Maverick**、**Ray**、**Stargazer**、**Riveroaks**，成员们正试图揭开它们的起源和能力。
   - 据称 **Stargazer** 由 Google 制作（=== **Nebula**），而 **Riveroaks** 声称来自 OpenAI 的 GPT-4o，而 **Maverick**、**Spider** 和 **24_karat_gold** 由于共享 System Prompt 且起源于 Meta，似乎具有相似的风格。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.techspot.com/news/107347-finally-beginning-understand-how-llms-work-no-they.html">我们终于开始理解 LLM 是如何工作的：不，它们不仅仅是逐字预测</a>：电路追踪（Circuit tracing）是一种相对较新的技术，它让研究人员能够逐步跟踪 AI 模型如何构建答案——就像跟踪电路中的布线一样...</li><li><a href="https://studio.zerobrane.com/download?not-this-time>">下载 - ZeroBrane Studio - 适用于 Windows, Mac OSX 和 Linux 的 Lua IDE/编辑器/调试器</a>：未找到描述</li><li><a href="https://lifearchitect.ai/models-table/">模型表格</a>：在新标签页中打开模型表格 | 返回 LifeArchitect.ai。模型表格排名推理模型 • 2024Q3–2025Q1 数据字典模型...</li><li><a href="https://gist.github.com/riidefi/443dc5c4b5e13e51846a43067b5335a1">Meta (?) 的 `24_karat_gold` (lmarena) System Prompt</a>：Meta (?) 的 `24_karat_gold` (lmarena) System Prompt - prompt.txt</li><li><a href="https://simple-bench.com/">SimpleBench</a>：SimpleBench</li><li><a href="https://arxiv.org/abs/2503.21934v1">证明还是虚张声势？在 2025 年美国数学奥林匹克竞赛上评估 LLM</a>：最近针对大型语言模型（LLM）的数学基准测试（如 MathArena）表明，最先进的推理模型在 AIME 等数学竞赛中取得了令人印象深刻的成绩...</li><li><a href="https://gist.github.com/riidefi/3340cc2b33b9edf5f03dc4429ba635d0">LMArena 的 `venom` System Prompt</a>：LMArena 的 `venom` System Prompt。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://archive.is/tkuum">未找到标题</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/36878">由 bozheng-hit 添加 Qwen3 和 Qwen3MoE · Pull Request #36878 · huggingface/transformers</a>：添加 Qwen3。此 PR 为即将发布的 Qwen3 模型添加了代码支持。有关 Qwen 的信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker
</li>
</ul>

</div>
  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1356407551964479591)** (1 条消息): 

> `Alpha Arena 更新，复制代码功能，图像生成，Bug 报告` 


- ****Alpha Arena** 新增复制代码和图像功能**：**Alpha Arena** 现在具备了**复制代码**功能和**图像生成**能力。
   - 用户可以使用密码 `still-alpha` 在 [alpha.lmarena.ai](https://alpha.lmarena.ai/) 体验新功能。
- **请求 **Alpha Arena** 测试人员提供反馈**：鼓励测试人员通过 [Google Forms 链接](https://forms.gle/8cngRN1Jw4AmCHDn7) 提供反馈，并通过 [Airtable 链接](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 报告 Bug。
- **过时的浏览器导致 **Airtable** 问题**：建议遇到 **Airtable** 问题的用户使用桌面应用，或更新到最新版本的 **Chrome**、**Firefox**、**Safari** 或 **Edge**。
   - 提出此建议是为了解决潜在的兼容性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/8cngRN1Jw4AmCHDn7">Arena - New UI Feedback</a>: 告诉我们你对新设计的看法！</li><li><a href="https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form">Airtable | Everyone&#x27;s app platform</a>: Airtable 是一个用于构建协作应用的低代码平台。自定义您的工作流，进行协作，并实现宏伟目标。免费开始使用。
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1356343091153600683)** (867 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro 推理，试用滥用与账号标记，Roo Code 替代方案，Model Context Protocol，AI 生成的肯德基广告` 


- **Gemini 2.5 Pro：推理能力引发争议**：一名成员质疑为什么 **Gemini 2.5 Pro** 不进行推理，称“它不思考，反应非常快”，引发了关于其能力的讨论。
   - 其他人辩护了 Gemini 在特定场景下的能力，而一些人则认为 **Claude 3.7** 在处理复杂性和细节方面更有效。
- **账号限制引发试用滥用辩论**：在一名用户对账号限制表示困惑后，另一名成员声称该账号因滥用试用而被“标记 (flagged)”，需要绑定信用卡。
   - 另一名用户建议使用 **Windsurf** 或 **Cline** 等替代方案来绕过支付问题。
- **比较 AI 模型性能和工具**：成员们讨论了 **Gemini 2.5 Pro** 与 **Claude 3.7** 的性能，有些人更喜欢 **Gemini 2.5 Pro**，而另一些人觉得它只适用于简单任务，还有人更喜欢 **Sonnet 3.7 Thinking**。
   - 讨论还涵盖了 **Roo Code** 等不同工具的使用以及 **Prompt Engineering** 的方法，强调保持 **Prompt** 简单明了，并专注于每个任务的 **Multiple Shots**。
- **讨论 AI 取代工作，以及对 ML 和 AI 知识的需求**：成员们讨论了 AI 的未来及其对就业的潜在影响，有人建议到 **2030** 年，**86% 的工作**可能会被取代。
   - 对此的回应是正确学习 **ML/AI** 和 **Prompting**，以及 **polynomials with regressions**。
- **Cursor 的免费模型受到质疑**：成员们质疑 **Cursor** 为何对使用“免费”模型收费，解释是 **Cursor** 的 API 使用通过其钱包管理，并且他们通过 **Fireworks** 与一些 AI 模型达成了协议。
   - 共识是 **Cursor** 的 **Token** 使用有限，但它比 **Claude** 便宜约 10 倍。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/i/status/1906477822773965278">来自 Ashton Forbes (@JustXAshton) 的推文</a>：如果这段视频显示飞机坠入大海，没人会质疑。我们只会说，“当然，美国军方正在追踪一架飞过其演习区和基地的失控波音 777……”</li><li><a href="https://x.com/minchoi/status/1906841667749183611?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 Min Choi (@minchoi) 的推文</a>：Google Gemini 2.5 Pro 是目前最好的 AI 编程模型。人们正在发现疯狂的方法来构建应用、游戏并大幅提升生产力。10 个疯狂的例子：1. 办公室模拟游戏</li><li><a href="https://x.com/nrehiew_/status/1907056100987531368">来自 wh (@nrehiew_) 的推文</a>：在 ChatGPT 发布近 2.5 年后，OpenAI 终于发布了 text-davinci-003</li><li><a href="https://x.com/i/status/1906776503343325469">来自 Salma (@Salmaaboukarr) 的推文</a>：我被震撼了！😱 这个 KFC 概念广告 100% 是 AI 生成的！我的朋友 David Blagojevic（他不在 X 上）为 KFC 创作了这个广告概念，太不可思议了！使用的工具：Runway, Pika, Kling...</li><li><a href="https://x.com/i/status/1906853123202720012">来自 Ingi Erlingsson 🪄 (@ingi_erlingsson)</a>：过去几天我有机会试用了新的 @higgsfield_ai 工具，非常有趣！很高兴看到如此精心策划的摄像机运镜和效果集合……</li><li><a href="https://v0-next-js-website-orcin.vercel.app/">Cursor Editor Dumbness Meter</a>：未找到描述</li><li><a href="https://x.com/ryancarson/status/1907083858467803414">来自 Ryan Carson (@ryancarson) 的推文</a>：刚刚正式在 @cursor_ai 中从 Sonnet 3.7 MAX 切换到了 Gemini 2.5 Pro MAX。1m 上下文 + 强大的推理能力 + 强大的编程能力的结合，让编程成为一种享受。👏 @tulseedoshi @Off...</li><li><a href="https://x.com/cj_zZZz/status/1906009088246546767">来自 Cj Z 🎯 (@cj_zZZz) 的推文</a>：Cursor Agent 简直太疯狂了。现在我使用 Gemini PRO 2.5 来扫描代码库，并使用 Sonnet 3.5/3.7 来执行代码。在这个工作流中，你需要 3 样东西：1. 详细的项目文档 2. 使用多个 AI 编程...</li><li><a href="https://www.cursor.com/pricing">定价 | Cursor - AI 代码编辑器</a>：选择适合您的方案。</li><li><a href="https://docs.cursor.com/chat/ask">Cursor – Ask 模式</a>：未找到描述</li><li><a href="https://tenor.com/ojyudjGU9US.gif">彩虹海绵宝宝 GIF - 彩虹海绵宝宝的想象力 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://artrebellion.blogspot.com/2024/03/2029.html?m=1">Art Rebellion: 2029</a>：未找到描述</li><li><a href="https://pastebin.com/Hzjp0yS9">&lt;globalRules&gt;&lt;responses&gt;- 在思考解决方案前重复问题 - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://forum.cursor.com/t/guide-a-simpler-more-autonomous-ai-workflow-for-cursor/70688">[指南] 一个更简单、更自主的 Cursor AI 工作流</a>：大家好，继之前的 KleoSr Cursor Rules 系统之后，过去一周我一直在努力，并与我旧帖子中的社区进行互动：[指南] 最大化编程效率...</li><li><a href="https://tenor.com/view/private-browse-man-private-browse-private-browser-because-they-are-watching-cuz-theyre-watching-gif-18524451">私密浏览男私密浏览器 GIF - 私密浏览男私密浏览私密浏览器 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://artificialanalysis.ai/">AI 模型与 API 提供商分析 | Artificial Analysis</a>：AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://status.cursor.com/">Cursor 状态</a>：未找到描述</li><li><a href="https://tenor.com/view/never-gonna-give-you-up-rickroll-april-fool-gif-2904154562499090366">Never Gonna Give You Up Rickroll GIF - Never Gonna Give You Up Rickroll 愚人节 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://dev.to/composiodev/how-to-connect-cursor-to-100-mcp-servers-within-minutes-3h74">未找到标题</a>：未找到描述</li><li><a href="https://www.apideck.com/blog/unlocking-ai-potential-how-to-quickly-set-up-a-cursor-mcp-server">释放 AI 潜力：如何快速设置 Cursor MCP Server</a>：了解如何在 Cursor 中快速设置 MCP Server，并通过 Model Context Protocol (MCP) 释放 AI 的潜力。标准化 LLM 与外部工具的集成。</li><li><a href="https://docs.cursor.com/context/model-context-protocol">Cursor – Model Context Protocol</a>：未找到描述</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新和改进

ents.</li><li><a href="https://www.cursor.com/changelog/reliability-keyboard-shortcuts-early-access-opt-in">Changelog - Mar 11, 2025 | Cursor - The AI Code Editor</a>: 可靠性、键盘快捷键和早期访问加入</li><li><a href="https://forum.cursor.com/t/changelog-for-0-45-6-version/46150">Changelog for 0.45.6 version</a>: 大家好！也许有关于最新版本的信息，让我们聊聊吧
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1356368837343183059)** (256 messages🔥🔥): 

> `Blackwell Support, VLM Training, GRPO Usage, Gemini 2.5 Pro, Training with Unsloth` 


- **RTX Pro 6000 获得 PyTorch 支持**：一位用户询问关于 **Blackwell 支持**的问题，并提到拥有一块 **RTX Pro 6000**，配备 **CUDA 12.8** 和 **sm_120**，用于微调 **Mistral**。
   - 另一位用户回复称 PyTorch nightly 版本支持它，但需要重新编译所有内容。
- **RAG 奖励机制思考**：一位用户询问如何将 **GRPO** 用于 **RAG** 或类似的工具调用变体，以及如何设置奖励。
   - 另一位用户概述了主要的奖励组成部分，包括**检索质量奖励**（相关性、多样性、准确性）、**生成质量奖励**（事实一致性、引用准确性、完整性）以及**工具使用奖励**（选择是否得当、使用是否正确、整合是否有效）。
- **Unsloth 训练精度指南**：在关于训练速度和精度的讨论中，有人指出如果 **VRAM** 不受限，**16-bit LoRA** 是最精确且最快的，并且 **Unsloth** 针对 16-bit 进行了优化。
   - 此外，建议对 **4-bit** 和 **8-bit** 进行基准测试，以观察差异并获得实践经验。
- **多 GPU 支持即将到来**：据透露，多 GPU 支持即将引入 **Unsloth**。
   - 首个版本可能仅包含数据并行（data parallelism），而 FSDP (Fully Sharded Data Parallelism) 最初可能不被包含，并将采用 AGPL3 许可证。
- **DeepSeek 训练尝试**：一位成员感叹他们*无法在 DeepSeek 上进行训练！*
   - 即使使用 **QLoRA**，训练 **DeepSeek** 可能也需要两个节点的 **H100**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/tokenized.html">Custom Pre-Tokenized Dataset – Axolotl</a>: 未找到描述</li><li><a href="https://pastebin.com/b9xBWtwv">class DataCollatorForLastCompletionOnlyLM(DataCollatorForLanguageModeling): - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本并设置有效期的网站。</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/4xAiviw1X8M?si=tWONGm9xk6t8kUee"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/2216">Fix batched generation for prompts of different lengths by RunFMe · Pull Request #2216 · unslothai/unsloth</a>: 未找到描述</li><li><a href="https://www.barrahome.org/">Inicio - alberto@barrahome:~$</a>: 未找到描述</li><li><a href="https://same-website-overview-ib64tvuyu69-latest.netlify.app/">alberto@barrahome:~$</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>: 添加 Qwen3。此 PR 为即将推出的 Qwen3 模型添加了代码支持。有关 Qwen 的信息，请访问 https://github.com/QwenLM/Qwen2.5。@ArthurZucker</li><li><a href="https://m.huxiu.com/article/4187485.html">无标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1356437781681016883)** (48 条消息🔥): 

> `Lightweight Pretraining Techniques, Bonsai pretraining, BitNet training Costs, Qwen Model Rebenched, Exllama2 vs vLLM Inference` 


- **调查 64-GPU 替代方案的 SOTA 轻量化 Pretraining**：一位成员建议研究 **SOTA 轻量化 pretraining 技术** (MoE, FP8)，以便在几周内通过单节点完成 pretraining，而不是使用 64 个 GPU。
   - 他们分享了 [deepgrove-ai/Bonsai 的链接](https://github.com/deepgrove-ai/Bonsai)，暗示可能仅需 $70 和 3.8b tokens 就能在 **BitNet** 上完成 pretraining。
- **DeepGrove 的 Bonsai 宣称 $70 完成 BitNet Pretraining**：一位成员对 [DeepGrove 的 Bonsai](https://github.com/deepgrove-ai/Bonsai) 声称仅用 $70 和 3.8b tokens 就能完成 **BitNet** pretraining 表示怀疑。
   - 他们正在 Kaggle 中运行该模型以验证其真实性，并探索该模型是否是 *盲目复制的 Qwen 模型* 或 *从 Qwen 到 BitNet 的 continue trained*。
- **BitNet 验证挑战探讨**：一位成员分享了一段用于 **修改权重量化 (weight quantization)** 的代码片段，以确定模型是否基于 **BitNet** 架构。
   - 该代码使用 per-tensor quantization 到 1.58 bits，量化时不需要分组。
- **最快推理引擎之争：exllama2 vs vLLM**：一位成员询问了针对 Llama / Mistral 4bit 量化模型在单请求非批处理解码 (non-batched decoding) 下最快的推理引擎，特别是对比 **Sglang/lmdeploy** 和 **vLLM**。
   - 该成员认为 **vLLM** 在非批处理解码中可能表现不佳，因为其引擎需要经过 llm_engine.step()。
- **探讨 TurboDerp 的 exllama2 动态模式 (Dynamic Mode)**：一位成员分享了 [exllama2 的动态模式链接](https://github.com/turboderp-org/exllamav2/blob/master/doc/dynamic.md)，指出所有 forward 调用都通过 generator，需要将控制权移交给 generator 的任务调度。
   - 其他成员建议使用 **TensorRT LLM** 进行单 token 生成，而一些成员建议在 exllama 中 *hook forward pass*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/deepgrove-ai/Bonsai">GitHub - deepgrove-ai/Bonsai</a>：通过在 GitHub 上创建账号，为 deepgrove-ai/Bonsai 的开发做出贡献。</li><li><a href="https://github.com/turboderp-org/exllamav2/blob/master/doc/dynamic.md">exllamav2/doc/dynamic.md at master · turboderp-org/exllamav2</a>：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp-org/exllamav2</li><li><a href="https://github.com/turboderp-org/exllamav2/discussions/500">在 hidden state 输入、query、key/value 或 attention head 输出中注入噪声 · turboderp-org/exllamav2 · Discussion #500</a>：大家好！我不确定这里是否已经讨论过这个问题，但我偶然发现了在推理过程中以及采样之前注入噪声的想法。参见 https://github.com/EGjoni/DRUGS 以及...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1356355722824253511)** (248 条消息🔥🔥): 

> `Orpheus 数据集问题，模型评估问题，Gemma 3 推理样本，使用 PDF 进行微调，Gemma 3 的 Vision 微调` 


- **数据集导致 Value Error**：一位用户在以 **Unsloth Orpheus** 格式使用自定义数据集时遇到了 `ValueError: expected sequence of length 203 at dim 1 (got 885)`，随后通过使用 GPU 解决了该问题。
   - 另一位用户提到 **Orpheus 数据集** 使用了 **SNAC**，其运行频率为 **24kHz**。
- **模型评估出现不连贯问题**：一位用户报告在模型评估过程中遇到问题，尽管在普通推理运行时生成的文本是连贯的，但在评估时模型生成的文本却不连贯。
   - 有建议称启用 `report_to` 可以帮助将指标记录到 **Wandb** 等平台，特别是在使用自定义 `compute_metrics` 函数时。
- **Gemma 3 支持文本生成图像！**：一位用户询问了使用 Hugging Face 对 **Unsloth/Gemma 3** 进行图像和文本推理的样本，并参考了 [Hugging Face Spaces 上的 Gemma 3 演示](https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it)。
   - 有人指出，虽然 **Llama 3.2 Vision** 需要图像，但 **Gemma 3** 应该不存在同样的问题。
- **将 PDF 转化为聊天机器人，需要数据准备**：一位用户在通过 **Langchain** 将 PDF 转换为文本后，寻求关于仅使用文档（PDF）对模型进行微调以使其专注于特定语言或领域的指导。
   - 建议通过 *augmentoolkit* 使用合成数据生成，并强调此过程超出了 **Unsloth** 的范围，他们应该查看 [Unsloth 文档](https://docs.unsloth.ai/basics/continued-pretraining)。
- **Mamba 获得 Eager 模式关注**：用户发现通过设置 `attn_implementation = "eager"` 可以修复 **Mamba** 的实现问题，正如 [GitHub pull request](https://github.com/unslothai/unsloth/pull/2263) 中所强调的那样。
   - 尽管有了修复方案，但据指出 **Mamba** 的训练速度明显变慢。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>：关于使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it">Gemma 3 12b It - a Hugging Face Space by huggingface-projects</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>：又称持续微调。Unsloth 允许你进行持续预训练，以便模型学习新语言。</li><li><a href="https://github.com/unslothai/unsloth/issues/2265">[DOC] Gemma 3 instructions on Vision Fine Tuning page is not correct · Issue #2265 · unslothai/unsloth</a>：[x ] 报告错误的文档 报告需要的文档 报告错误的文档 错误文档的位置 —— 如果可能，请提供链接和行号。https://docs.unslot...</li><li><a href="https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora">未找到标题</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1356359646796845199)** (23 messages🔥): 

> `模型评估, 编程基准测试, 长上下文基准测试, 数学基准测试, Gemma 3 对比小型 LMs` 


- **通用性能基准测试并不存在**：一名成员表示，不存在所谓的通用性能基准测试，每个模型在不同的垂直领域集合中表现各有优劣。
   - 他们认为*这就像一个谬误，即认为通过聚合一堆垂直领域，就能得到一个满足你特定垂直领域需求的分数结果*。
- **编程基准测试：Aider Polyglot 和 SWE Bench**：对于编程基准测试，一名成员建议将 **Aider Polyglot** 和 **SWE Bench** 作为相当适用的基准。
   - 然而，**SWE Bench** 存在基于 LLM 框架的问题，而 **Aider Polyglot** 可能会展示出在使用 aider 时 LLM 的表现有多好。
- **RULER 是长上下文基准测试的最低标准**：对于长上下文基准测试，一名成员表示 **RULER** 是衡量长上下文基准的最低标准，而 **NIAH** 很垃圾。
   - 他们补充说，最近出现的一些基准测试还不错。
- **数学基准测试：AIME 已经足够好**：对于数学基准测试，一名成员建议只要没有数据污染并配合 COT 进行适当评估，**AIME** 就足够好了。
   - 他们还提到大多数编程基准测试是基于 python 的，但针对 JS 有 **WebDev Arena**。
- **小型 LMs 对比 Gemma 3**：一名成员表示有兴趣将 **Gemma-3 4B** 与现有的主流小型 LMs 进行对比。
   - 他们询问如果 **Open LLM** 还没有收录 **Gemma 3**，是否还有其他可行的排行榜包含 **Gemma 3** 与小型 LMs 的对比。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1356385770759389406)** (2 messages): 

> `Discord 改进, 简化入职流程, 反馈整合, Pro 频道访问` 


- **Discord 即将迎来大修**：版主团队收集了反馈以增强 Discord 体验，并计划在下周实施**三项关键改进**。
   - 用户可以期待入职流程和反馈频道的变更，相关公告将提前发布以避免意外。
- **为新手简化入职流程**：入职流程将被简化，以减少参与社区前所需的步骤和选择。
   - 目标是让新用户更容易上手并迅速成为活跃成员。
- **反馈中心：一个频道统领全局**：反馈频道将被合并以简化流程，确保 PPLX 团队能及时了解社区需求。
   - 此举旨在提高反馈效率，并确保团队始终了解用户需求。
- **Pro 频道 VIP 访问**：正在努力实现 Pro 频道的自动化访问，以便为紧急请求提供来自版主的高级支持。
   - 这将确保有紧急需求的用户能够获得及时且专门的协助。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1356346419996524596)** (544 messages🔥🔥🔥): 

> `Space Instructions 限制, 图像生成已停止？, 欧盟境内的 Apple Intelligence, 三星 AI 对比 Apple Intelligence, GPT Omni 的缺点`

- **Space Instructions 提供有限的控制，成员们发现**：用户讨论指出，Perplexity AI 中的 [**Space Instructions**](https://www.perplexity.ai/search/space-instructions-limitations-FuW7yKXsTQGFp71t55f1Bg) 无法完全控制搜索体验，主要影响的是输出摘要，而非初始数据获取。
   - 这种限制意味着指令无法阻止 AI 搜索特定主题，因为指令仅在相关数据被提取*之后*才生效，这引起了一些用户的不满。
- **Perplexity 图像生成：消失了吗？**：一位用户询问 Perplexity 内部**图像生成**功能的停用情况，注意到该功能已缺失。
   - 另一位用户建议使用 *web search* 来查找生成选项，但另一位用户确认该功能似乎并非对所有人可见，这可能预示着分阶段推出或功能测试。
- **Apple Intelligence 在欧盟被封锁？**：一位用户随口提到 *yey apple intelligence in the EU now*，暗示其可用性，但未作进一步说明。
   - 在该声明之后，其他人迅速将焦点转向讨论 **Samsung AI**，其中一位用户声称其更优越，引发了关于两者优缺点的辩论。
- **Perplexity 用户抱怨 GPT Omni**：用户对 **GPT Omni** 表示不满，其中一人形容其 *suck ass*，并询问如何恢复到之前的 **GPT** 版本。
   - 另一位用户解释说，**Omni** 旨在实现与音频、视频和图像更智能的交互，但出于成本原因，与 **GPT-4** 相比已被“降智”。
- **传闻四起：Perplexity 将推出更强大的 Deep Research**：一位 Perplexity 团队成员暗示，在未来几周内将推出*更强大的版本*的 **Deep Research**。
   - 推测包括在添加文本后可能与 **Groq** 建立合作伙伴关系，但并非实际的新深度研究功能；用户报告称 **Deep Research** 在几秒钟内即可完成，而不是几分钟。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/askperplexity/status/1907205437151498628?s=61">来自 Ask Perplexity (@AskPerplexity) 的推文</a>：Perplexity 已签署最终协议，以 2.2 万亿美元收购 Poogle —— 这是在 AI 时代改进搜索的重要一步。</li><li><a href="https://x.com/pplxsupply/status/1907134777549013405?s=61">来自 Perplexity Supply (@PPLXsupply) 的推文</a>：不骗你，动物们也喜欢 PPLX Supply</li><li><a href="https://www.wired.com/story/openai-sam-altman-announce-open-source-model/">Sam Altman 表示 OpenAI 将在今年夏天发布一款“开放权重” AI 模型</a>：此消息紧随 DeepSeek 的突破性成功以及来自 Meta 等竞争对手日益增长的压力。</li><li><a href="https://tenor.com/view/dead-reckoning-part-1-hayley-atwell-pom-klementieff-ilsa-faust-rebecca-ferguson-gif-651077158243048187">《碟中谍7：致命清算（上）》Hayley Atwell GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/got-game-of-thrones-you-know-nothing-jon-snow-ygritte-gif-14613130">《权力的游戏》GIF - 你什么都不懂，Jon Snow - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/metal-gear-rising-metal-gear-rising-revengeance-senator-armstrong-revengeance-i-made-it-the-fuck-up-gif-25029602">《合金装备崛起：复仇》GIF - Senator Armstrong - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/totoro-gif-24991987">龙猫 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/counting-money-paper-stacks-gif-9918098029500472876">数钱 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://ahrefs.com/traffic-checker/?input=https%3A%2F%2Fwww.perplexity.ai%2Fdiscover&mode=exact">网站流量检查工具：估算任何网站的流量</a>：深入挖掘任何网站的流量数据，并为您的网站寻找增长机会。试用 Ahrefs 流量检查工具的免费版本。</li><li><a href="https://gizmodo.com/why-does-chatgpts-algorithm-think-in-chinese-2000550311">为什么 ChatGPT 的算法会用中文“思考”？</a>：OpenAI 的新推理模型正在做出一些奇怪且不可预测的行为。</li><li><a href="https://www.hamsterai.net/?ref=easypeasy.chat">Hamster AI</a>：Hamster AI，将最强大的 AI 工具整合到一个软件中。以低于一杯咖啡的价格使用来自 OpenAI, Claude 和 Mistral 的工具！</li><li><a href="https://aistudio.google.com/prompts/new_chat">未找到标题</a>：未找到描述</li><li><a href="https://www.getmerlin.in/chat/share/d1c0399f-f4cf-43ae-b665-dbcc20364a68">木瓜 vs 哈密瓜每 100 克的营养成分</a>：由匿名用户分享 - 2025年4月1日</li><li><a href="https://www.blackbox.ai/share/1f7845a8-f1c0-4b18-a5a9-a47f6a96ef07?mobile=true&model=DeepSeek-R1">木瓜 vs 哈密瓜每 100 克的营养成分 - Blackbox</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1356373242138333399)** (10 条消息🔥): 

> `Python 代码追踪, AI 阅读准确性, API 研究` 


- **Python 代码追踪技巧**：一位用户询问了[如何在 Python 上追踪代码](https://www.perplexity.ai/search/how-do-you-trace-a-code-on-pyt-MFRO2wlGS1K6RnRoRglyKg)。
   - 上下文中未提供答案。
- **AI 阅读准确性受到质疑**：一位用户询问了 [AI 在阅读方面的准确性如何](https://www.perplexity.ai/search/how-accurate-is-ai-in-reading-KuQPZ6MjRUOwSLmN1XYxog)。
   - 上下文中未提供答案。
- **API 研究问题**：一位用户询问了关于 [研究 API](https://www.perplexity.ai/search/zheng-zhun-bei-yan-jiu-apiwen-cf2FuAtIRRyhrZYsqpqDog#0) 的问题。
   - 上下文中未提供答案。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1356383709174956233)** (5 条消息): 

> `Sonar API Access, Tier 2 Credits, JSON Formatting with Pydantic` 


- **寻求 Sonar API 访问权限**：一位用户询问如何因工作需要获取 **Sonar API** 的访问权限，并请求 Perplexity 团队相关负责人的联系方式。
   - 来自 API 团队的 James 进行了回复并提供帮助。
- **已获取 Tier 2 额度**：一位用户确认他们已达到带有额度的 **Tier 2** 级别。
   - 他们希望告知模型该回复将由 **FMCG 公司** 的品牌经理阅读，因此请以对他们具有可操作性的方式进行组织。
- **网页搜索结果的 JSON 格式化问题**：一位用户报告了 **Sonar API** 在搜索网页时，尽管使用了 Pydantic 进行格式化，但在 JSON 结果中添加了奇怪的特殊字符（例如 "<"）。
   - 用户提供了一个示例，其中 **JSON 输出** 中的 `source_name`、`source_title`、`summary` 和 `url` 字段被添加了额外的字符。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1356684338451976404)** (1 条消息): 

> `ChatGPT's new voice Monday, voice mode, voice picker` 


- **ChatGPT 推出新的 Monday 语音选项**：**ChatGPT** 中引入了一个名为 **Monday** 的新语音选项，可以通过语音模式右上角的语音选择器（voice picker）进行访问，如附带的 [视频](https://cdn.discordapp.com/attachments/977259063052234752/1356684338321821816/monday.mp4?ex=67ed7640&is=67ec24c0&hm=b9a68532d205bff5d213b2e9ccdde4381355d4760832a57a6e8fe713eb7bcacf&) 所示。
- **Monday 语音快速访问**：用户可以通过打开语音模式并选择位于界面右上角的语音选择器，在 **ChatGPT** 中快速访问新的 **Monday** 语音。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1356370993576149193)** (314 条消息🔥🔥): 

> `虚假 ChatGPT 应用、Gemini 2.5 Pro 速率限制、Ghibli 风格图像生成、ElevenLabs 语音模型、AI 与创意产业` 


- **警惕！ChatGPT 冒充者席卷 Play Store**：有用户报告通过 Play Store 购买了 **ChatGPT** 但无法获得访问权限，这引发了对虚假冒充应用的担忧，并敦促用户检查其 [购买历史](https://help.openai.com) 以确认交易对象是 **OpenAI**。
   - 确保使用官方应用至关重要，以避免诈骗并确保能够访问 **OpenAI** 服务。
- **Gemini 2.5 Pro 遭遇速率限制**：用户报告在 **Gemini 2.5 Pro** 上遇到速率限制（rate limited），引发了关于该限制是否同时适用于免费和付费层级的讨论，部分用户通过使用 **VPN** 绕过了速率限制。
   - 有建议在 [Google AI Studio](https://ai.google.dev/) 中使用 **Gemini**，那里的限制更高（每天 50 次）。
- **Ghibli 风格转换引发 AI 图像生成器热议**：成员们尝试使用提示词将图像转换为 **Ghibli** 风格，一位用户分享了他们的提示词 *"Make this image gibli style"*，而另一位则建议使用 *"Reimagine this image in the iconic Studio Ghibli style: painterly textures, soft light, and a touch of nostalgic wonder"*。
   - 虽然使用了免费模型，但有人指出在刻画情感和面部细节方面仍需改进，一些人声称这 **远比为了无谓的原因毁掉 Ghibli 艺术风格要好**。
- **ElevenLabs 新语音模型：前景广阔但价格昂贵？**：一位成员分享说，他们正在探索使用 [ElevenLabs 的新模型](https://elevenlabs.io/) 来创作叙事类有声读物，并强调了其语音克隆（voice cloning）功能。
   - 虽然对初步结果和高质量印象深刻，但他们仍在等待 **OpenAI** 发布类似的语音产品，以避免订阅外部服务；对于一些游戏开发者来说，这可以作为配音占位符。
- **探讨 AI 在创意产业中的角色：如履薄冰**：讨论涉及了 **AI** 在创意领域（尤其是游戏业）的应用，共识是 **AI** 经常被非创意人员使用，导致输出结果平庸，且高估了 **AI** 目前的能力，他们引用了 [这段讨论](https://discord.com/channels/974519864045756446/1355653770603925806)。
   - 席间进行了意见交换，有人认为 **AI** 主要是辅助专业人士进行构思（ideation），而另一些人则批评过度依赖统计学上的平均输出，并强调创作新颖作品需要人类的努力。将 **AI** 集成到 **Adobe** 和 **Autodesk** 等现有软件生态系统中被视为一个更有前景的方向。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.openai.fm/">OpenAI.fm</a>：供开发者尝试 OpenAI API 中全新文本转语音模型的交互式演示</li><li><a href="https://tenor.com/view/shia-labeouf-clapping-theatre-gif-15313749163340250467">Shia Labeouf 鼓掌 GIF - Shia labeouf 剧院鼓掌 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://drive.google.com/file/d/1IIqxolKNn3cbQ9DaKTYqx5WIvJ04twTP/view">evolving_llms_through_text-based_self-play.pdf</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1356358451596365864)** (24 条消息🔥): 

> `图像生成速率限制、copilot 体验、ChatGPT 指令、4o 能力、图像模型的未来` 


- **OpenAI 为 Plus 用户实施图像生成速率限制**：由于新图像模型发布以来负载极大，Plus 用户现在正面临 [rate limits](https://www.google.com/search?q=openai+rate+limits)（速率限制），这是缓解用户洪流的临时措施。
   - 一位用户（推测是开玩笑地）评论道：*“每月 200 美元最好别被限流，”* 这指的是 OpenAI 在一小时内增加了 **100 万新用户**的传闻。
- **用户适应 copilot**：一位成员分享了在 **copilot** 反复出现“愚蠢行为”时努力去适应它的感受。
   - 该成员表达了在使用 **copilot** 时的适应感。
- **用户寻求 ChatGPT Prompting 指导**：一位用户寻求帮助，以防止 **ChatGPT** 在描述中添加“酷炫或前卫”的结束语。
   - 另一位成员建议修改 Prompt，包括这一行：*"Do not add concluding remarks outside the direct scope of fulfilling the request based on the chosen purpose. Stick to the process."*（不要在基于所选目的完成请求的直接范围之外添加结束语。坚持流程。）
- **利用 4o 能力尝试照片编辑**：一位用户建议在 **4o** 中加入一种编辑照片或设置自定义氛围的模式，通过 **20 questions** 游戏预加载上下文，以锁定模糊的人性化细节。
   - 该想法建议利用 **4o** 读懂言外之意的能力，以获得更好的后续请求和个性化体验。
- **辩论图像模型改进的未来**：一位成员询问 OpenAI 是否会继续改进图像模型，还是会像对待 **DALL-E** 那样搁置几年。
   - 目前没有提供具体答案，但这个问题引发了人们对 OpenAI 图像生成技术未来计划的好奇。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1356407538467213434)** (9 条消息🔥): 

> `“关于我”框中的自定义指令、Memory 存储的 Prompt、模型回复的个性化、模型模式识别与格式化` 


- **Custom Instructions 在“关于我”框中有效**：一位成员确认，将自定义指令扩展到“关于我”框中运行良好，因为*对模型来说信息就是信息*。
   - 另一位成员指出，模型可以识别出你意图中的可能模式并*顺势而为*，即使字段在句中被拆分。
- **RAG 专用上下文限制了“关于我”框**：一位成员质疑“关于我”框是否被限制在某些 **RAG-only context-dependent space**（仅限 RAG 的上下文相关空间）中，以及它对于存储完整 Prompt 是否可靠。
   - 他们还提到拥有海量的 Memory（记忆），并尝试将整个 Prompt 放入其中，但似乎不太可靠。而且，存储在 Memory 中的 Prompt 或 Persona（人格）如果不明确要求就不会激活。
- **个性化可能并不总是显而易见**：一位成员分享了模型个性化回复的例子，并指出第一条回复可能并不总是反映其个性化设置。
   - 他们强调要非常明确地告诉模型自己想要什么和不想要什么，包括对工具使用和 NPC 行为的规范。
- **模型学会重置僵化模式**：一位成员分享了一个代码片段 `FORMAT_RESET`，以帮助模型识别何时陷入了僵化模式并重新思考方法。
   - 该代码鼓励模型分析哪种格式更适合回复，并彻底重新思考其方法，而不是默认使用模板。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1356407538467213434)** (9 条消息🔥): 

> `“关于我”框中的自定义指令，Memory 存储的提示词，模型猜测 vs 训练，针对僵化模式的 FORMAT_RESET` 


- **“关于我”作为自定义指令？**：一位成员询问将自定义指令扩展到“关于我”（*about me*）框中是否有效，另一位成员确认这完全可行，因为模型会将其作为额外信息来使用。
   - 模型可以识别出你意图中的可能模式并据此运行，即使你在句子中间拆分字段也是如此；除非“关于我”被限制在某些仅限 RAG 的上下文相关空间中，否则没有功能上的理由说明它不起作用。
- **Memory 存储的提示词不可靠？**：一位成员指出他们拥有海量的 Memory，虽然他们尝试将整个提示词放入其中，但似乎并不十分可靠。
   - 他们无法在不明确要求的情况下激活 **Memory 存储的提示词或 Persona**，这让他们认为模型在大多数时间里实际上并没有看到它们，或者它们的优先级远低于模型上下文中自定义指令的位置；[关于 prompt engineering 和 NPC 生成的共享对话示例](https://chatgpt.com/share/67eb2cb0-b41c-8011-abee-88bfe2abc478)。
- **模型猜测 vs 训练**：一位成员分享了他们的处理过程，假设模型要么进行了大量猜测（导致变异），要么被训练以某种未被明确要求的典型模式输出，要么正在完全按照要求执行。
   - 必须发现并修复冲突，因为它们通常会降低性能，特别是当模型被训练为“人类偏好 X”但用户却有其他偏好时。
- **针对僵化模式的 FORMAT_RESET！**：一位成员创建了一个小工具，用于当你发现模型遵循你不喜欢/不想要的格式/模式时，作为一种让模型承认其已陷入僵化模式并重新思考其方法而不默认使用模板的方式。
   - 他们提供了一个代码片段来告诉模型：```FORMAT_RESET: Acknowledge you've fallen into rigid patterns, analyze what format would better suit your response, and completely rethink your approach without defaulting to templates.```


---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1356345374972969030)** (198 条消息🔥🔥): 

> `在 LM Studio 中使用 eGPU，Gemini 2.5 Pro 评估，Gemma 3 27B 性能，本地 LLM 推荐，Copilot 损害开发者体验` 


- **将 eGPU 接入 LM Studio！**: 成员们讨论了在 **LM Studio** 中使用 **eGPU** 的可行性，认为只要电脑能识别就应该可以工作，尽管速度较慢。讨论参考了一个对比 RTX 4090 笔记本电脑 vs 台式机运行 LLM 的 [YouTube 视频](https://www.youtube.com/watch?v=2hi1VoOI00g)。
- **Gemma 3 击败 Gemini 1.5 Flash**: 一位成员分享了一项对比，显示 [**Gemma 3 27B** 在 MMLU-Pro 和 Bird-SQL 等多个基准测试中优于 **Gemini 1.5 Flash**](https://i.imgur.com/XbmTSd1.png)。
   - 另一位成员确认了 Gemini 2.5 Pro 的出色表现，而另一位用户使用 Gemini 2.5 Pro 生成了数据，该模型可在 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) 上免费使用。
- **推荐 Qwen Coder 7B！**: 对于配置为 **4060 Ti** 和 **i5 12400F** 的系统，推荐使用 **Qwen Coder 7B**，该模型可在 [LM Studio 模型页面](https://lmstudio.ai/model/qwen2.5-coder-7b-instruct)获取，并建议将其大部分 Offload 到 GPU，也可以尝试使用 Qwen Coder 14B 或 32B。
   - 成员们强调，本地 LLM 的表现会比 **ChatGPT** 或 **Deepseek** 等云端替代方案*差得多*，但 **Gemini 2.0 Flash** 被认为是顶级选手，根据其[价格文档](https://ai.google.dev/gemini-api/docs/pricing)，每 1M Input Tokens 仅需 0.44 美元。
- **对 Copilot 编程能力的批判！**: 成员们辩论了编程中的 **AI 辅助** 是否有益，有人认为它*弊大于利*，因为普通用户是在“AI 废料 (AI slop)”上学习的。
   - 其他人持不同意见，表示 **Copilot** 对经验丰富的开发者非常有用，但有人声称普通用户太容易信任给出的建议，此外还担心 **Copilot 是在垃圾代码上训练的**。
- **单参数 LLM 是否可能？**: 在一次轻松的交流中，大家讨论了*单参数 LLM 是可能的但毫无用处*，一位用户表示他们*尝试了 656K 但无法进行对话*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/app">关于 LM Studio | LM Studio 文档</a>: 了解如何使用 LM Studio 在本地运行 Llama, DeepSeek, Phi 和其他 LLM。</li><li><a href="https://lmstudio.ai/model/qwen2.5-coder-7b-instruct">qwen2.5-coder-7b-instruct</a>: qwen • Alibaba • 7B</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (免费) - API, 提供商, 统计数据</a>: Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini Pro 2.5 Experimental (免费)</li><li><a href="https://tenor.com/view/april-fool-gif-25270662">愚人节 GIF - 愚人节 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/learn/nlp-course/">简介 - Hugging Face NLP 课程</a>: 未找到描述</li><li><a href="https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md">llama.cpp/examples/quantize/README.md at master · ggml-org/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://ai.google.dev/gemini-api/docs/pricing">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=2hi1VoOI00g">RTX 4090 笔记本 vs 台式机运行 LLM 🤯 差距巨大！</a>: 差距非常明显。SIHOO 转椅折扣: https://hongkongsihoointelligenthomecolimited.pxf.io/azisk 折扣码: YT6OFF Amazon: https://amzn.to/4jkRYPjDi...</li><li><a href="https://aistudio.google.com/prompts/new_chat">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/pydantic/pydantic-ai/issues/665">我该如何将 LM Studio 作为我的 LLM 服务使用？ · Issue #665 · pydantic/pydantic-ai</a>: 大家好！我是 Abhiraj，一名正在寻求帮助/建议的助理软件工程师。我想将 LM Studio 作为我的开源和本地 LLM 服务，目前是否有相关支持？因为我无法找到...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1356429118866194523)** (63 条消息🔥🔥): 

> `Nvidia 驱动程序在运行 10-12 小时后的不稳定性，M4 Max 与 5090 速度对比，Mac 与 Nvidia GPU 在 LLM 上的对比，Tenstorrent Wormhole 在 Discord 上的性能表现，上下文溢出和共享内存对 LLM 速度的影响` 


- **Nvidia 驱动程序在长时间运行后崩溃**：有用户报告在运行模型 **10-12 小时**后出现 **Nvidia 驱动程序不稳定性**，需要重新安装驱动程序才能解决性能问题。
   - 该用户澄清问题在于 **Nvidia 驱动程序**本身，而非 Windows 操作系统，并寻求是否有其他人遇到过类似情况。
- **M4 Max 对比 5090 的速度提升**：一位用户观察到，在解决崩溃问题后，从 **M4 Max** 到 **5090** 有 **3.24 倍**的加速，这与他们在进行 **QwQ 32B 4 bit 量化对比**时两者的内存带宽之比（**3.28 倍**）相吻合。
   - 在测试 **Gemma 3 32B q4** 时，他们现在看到 **M4 Max** 的速度约为 **21 tok/s**，而 **5090** 约为 **60 tok/s**。
- **Mac 的上下文容量自由 vs Nvidia 更快的 GPU**：虽然 **Nvidia GPU** 可能更快，但用户更倾向于选择 **Mac**，因为它可以*自由拥有更大的上下文容量（Context Size）*。
   - 该用户强调，尽管 NVIDIA GPU 速度更快，但*能够使用更大上下文的能力*被证明更有用。
- **在 Discord 中寻求 Tenstorrent Wormhole 的结果**：一位用户在 Discord 社区询问有关 **Tenstorrent Wormhole (n150d 和 n300d)** 的性能结果。
   - 他们表示有兴趣获取这些模型的 **TOK/s** 指标，但目前没有后续进展。
- **上下文溢出影响 LLM 速度**：一位用户想知道，如果将**上下文溢出**加载到共享内存/系统 RAM 中，同时将整个模型保留在 VRAM 中会发生什么。
   - 另一位用户指出，**LLM 需要将所有上下文都放在 VRAM 中**才能生成下一个 token，因为*每生成一个 token，所有上下文都会一遍又一遍地通过 Transformer 块*。



**提及的链接**：<a href="https://m.youtube.com/watch?v=nwIZ5VI3Eus&pp=ygUcbWFjIHN0dWRpbyB1bHRyYSBiYXR0bGUgNTA5MA%3D%3D,">M3 Ultra vs RTX 5090 | 终极对决</a>：M3 Ultra Mac Studio 对决搭载 NVIDIA RTX 5090 的 AI 怪兽。高效、多产、有序。 | Baseus Spacemate 系列（MAC）11 合 1 扩展坞，亚马逊美国站有售：...

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1356342503749714104)** (230 条消息🔥🔥): 

> `Gemini 2.5 Pro 的体验与限制，RateLimitError 自动化策略，Dot Command 革命，F#，视频分析` 


- **Gemini 2.5 Pro：是表现惊艳还是幻觉频出？**：用户正在实验 **Gemini 2.5 Pro**，报告的结果褒贬不一，一些模型会出现幻觉/断连，而另一些模型在编程任务中提供了顶级的性能。
   - 一位用户提到，*"Gemini 对我来说一直在产生幻觉/断连，你们也一样吗？"*，而另一位用户则表示 **Gemini 2.5 Pro** 和 **DeepseekV3** 的组合 *"几乎免费且性能顶级。"*
- **RateLimitError 的烦恼：是 Token 限制还是请求频率？**：一位用户报告在请求摘要和清除历史记录时频繁出现 **RateLimitErrors**，并正在寻找自动化处理该过程的解决方案。
   - Paul Gauthier 澄清说，速率限制可能基于每分钟或每天的请求数，而不是 Token 数量。一个可能的解决方案可以在 [这个 GitHub issue](https://github.com/Aider-AI/aider/issues/2979#issuecomment-2613554537) 中找到。
- **Dot Command 革命：Aider 的效率提升秘籍？**：一位用户正试图推广使用 **.dotcommands** 作为开发者的效率秘籍，使他们能够通过单行命令（如 `.status` 和 `.next`）自动执行任务。
   - 其目标是提供针对清晰度和特定功能优化的认知捷径，但目前还没人使用它们。这引发了调侃：*"点命令革命（THE DOT REVOLUTION）已经到来 🔥 各地的程序员都会想尝试这个超酷的技巧。"*
- **F#：是吊唁还是赞誉？**：一位用户提到将他们的应用从 Python 重构为 **F#**，引发了不同的反应，包括表示遗憾以及建议 *"使用 Haskell"*。
   - 虽然该用户解释说他们正在从事 ML 项目，但社区对在这些任务中选择 **F#** 持怀疑态度。
- **视频分析：超越文本转录**：一位用户询问 **AI 模型对视频的理解能力**，想知道它们是否理解情感影响或遵循视觉故事情节，而不仅仅是处理转录文本。
   - 一个回复指出，*"Gemini 的视频理解是将视频以每秒 1 帧的速度作为图像输入到模型中。"*


<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://liveswebench.ai/">LiveSWEBench</a>：未找到描述</li><li><a href="https://x.com/mahawaryas27492/status/1906794382659125625?s=46">来自 AI Purr-fessor (Yash) (@MahawarYas27492) 的推文</a>：惊人的 2.5 flash experimental 发布了。🔥 它非常聪明，在我测试的一些推理问题上比 o3 mini high 还要好。它的推出速度非常缓慢，所以你可能需要等待官方发布，我...</li><li><a href="https://x.com/sama/status/1906793591944646898">来自 Sam Altman (@sama) 的推文</a>：摘要：我们很高兴在未来几个月发布一个具有推理能力的强大新型权重开放（open-weight）语言模型，我们想与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (免费) - API、提供商、统计数据</a>：Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini Pro 2.5 Experimental (免费)</li><li><a href="https://tenor.com/view/do-it-star-wars-emperor-palpatine-palpatine-gif-799657800635657398">Do It 星球大战 GIF - Do it 星球大战 帕尔帕廷皇帝 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/kuzco-yzma-chat-cat-mwahaha-gif-24314206">Kuzco Yzma GIF - Kuzco Yzma 聊天 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/online-classes-throw-away-computer-parks-and-rec-ron-swanson-gif-17413463">在线课程扔掉 GIF - 在线课程扔掉电脑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/april-fool-april-fools-spongebob-spongebob-april-fools-april-fools-spongebob-gif-13844032">愚人节 GIF - 愚人节 海绵宝宝 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/Aider-AI/aider/issues/2979#issuecomment-2613554537">恢复聊天记录导致错误 / 聊天记录摘要功能失效 · Issue #2979 · Aider-AI/aider</a>：问题：我有一个相当长的聊天记录文件（80k tokens），但它提供了关于我正在构建的项目的大量有价值信息。上周当我使用具有大容量的模型时，它运行良好...</li><li><a href="https://youtu.be/JeNS1ZNHQs8?feature=shared">2025 年 Vibe Coder 访谈</a>：Vibe Coding https://linkgraph.net/stack/vibecoder 与 Kai Lentit 进行的专业 Vibe Coder 访谈，播出于 © The Viboe Coder 2025。AI coding prompt eng...</li><li><a href="https://github.com/tninja/aider.el">GitHub - tninja/aider.el：与 Aider 交互：让 AI 结对编程变得简单</a>：与 Aider 交互：让 AI 结对编程变得简单 - tninja/aider.el</li><li><a href="https://github.com/MatthewZMD/aidermacs">GitHub - MatthewZMD/aidermacs：在 Emacs 中使用 Aider 进行 AI 结对编程</a>：在 Emacs 中使用 Aider 进行 AI 结对编程。通过在 GitHub 上创建账户来为 MatthewZMD/aidermacs 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1356368905836429322)** (30 messages🔥): 

> `Temperature for coding, Stopping benchmarks, Aider with subdirectories, Aider local config, Model Summarization fails` 


- **程序员对 Temperature 的“冰冷”偏好**：成员们讨论了编程时的最佳 "temperature"，其中 `0` 是频道中最受欢迎的选择。
   - 一位成员询问该数值的合理性，并请教 *它是基于什么得出的吗*？
- **aider 对 Mono-Repos 的子树救星**：一位成员询问如何将 **aider** 限制在 monorepo 的子目录中，得到的回复是在切换到目标目录后使用 `--subtree-only` 开关。
   - 这会让 **aider** 忽略启动目录之外的仓库内容，尽管提问者指出文档需要更新，并指向了[关于大型 monorepos 的 FAQ](https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo)。
- **配置难题：aider 中的模型设置**：一位成员报告称，在本地 **YAML** 配置文件中指定模型名称未按预期工作。
   - 尽管启动消息显示了正确的配置设置，**aider** 仍然默认使用 *anthropic/claude-3-7-sonnet-20250219*，而不是配置的 *deepseek/deepseek-chat*。
- **在 aider 中启动 Linting 循环**：一位成员询问如何在 **aider** 中运行 linter，另一位成员建议使用 `/run [npm|pnpm] run [lint|fix|whatever-command]` 来实现紧密的反馈循环。
   - 另一位成员指向了 [aider.conf.yml 示例文件](https://github.com/Aider-AI/aider/blob/3992681b84d1ec0cbc18657c5ca832c89d7e551c/aider/website/assets/sample.aider.conf.yml#L263)，用于列出多个 linter。
- **Architect 模型结果获得“晋升”**：一位成员寻求一种方法，直接将 Architect 模型的满意响应发送给 Editor 模型，以节省 **2.5 Pro** 的额度。
   - 建议是开启一个新的 **aider** 实例，并使用 `--restore-chat-history` 和配置合适的 Editor，尽管目前缺乏 `--no-architect` 标志被认为是一个不便之处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://github.com/Aider-AI/aider/blob/3992681b84d1ec0cbc18657c5ca832c89d7e551c/aider/website/assets/sample.aider.conf.yml#L263">aider/aider/website/assets/sample.aider.conf.yml</a>：aider 是你终端里的 AI 配对编程工具。欢迎在 GitHub 上为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1356727655407292468)** (13 messages🔥): 

> `Organizations leave Beta, Web search results in Chatroom, Cerebras on OpenRouter, PDF support for OpenRouter API` 


- **Organizations 正式结束 Beta 测试！**：OpenRouter 宣布 **Organizations** 功能现已结束 Beta 测试，允许团队在一个地方控制计费、数据策略、提供商偏好和 API keys，详情见 [此 X 帖子](https://x.com/OpenRouterAI/status/1907167622539378728)。
   - 在为期两周的 Beta 测试期间，创建了超过 **500 个组织**，赋予了团队对数据策略的完全控制权和统一计费。
- **网页搜索进入聊天室！**：网页搜索结果现已在聊天室中可用，Perplexity 的结果格式与 OpenRouter 的 `:online` 模型变体类似。
- **Bluesky 的诉求**：一位成员请求 OpenRouter 也在 [Bluesky](https://bsky.app) 上发布消息，建议减少对 *Xitter* 的依赖。
- **呼吁加入 Cerebras！**：一位成员请求 OpenRouter 与 **Cerebras** 洽谈，将其加入 OpenRouter。
- **API 是否支持 PDF？**：一位成员询问 OpenRouter API 何时将支持 **PDF** 文件。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1907167622539378728">来自 OpenRouter (@OpenRouterAI) 的推文</a>：今天我们将 Organizations 结束 Beta 测试。通过 Organizations，团队可以完全控制数据策略和统一计费，在数十个模型提供商之间增加安心感。关键...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1356357754356371607)** (98 条消息🔥🔥): 

> `Aider OpenRouter Copilot, Gemini Flash 2 Context, Usage Downloads, Enterprise Level Rate Limits, GPT4o Image Generation` 


- **Gemini Flash 2 Middle-Out Transforms**：成员确认 [OpenRouter](https://openrouter.ai/) 在付费的 **Gemini Flash 2** 请求上提供完整的 **1M context**，其中 **middle-out transforms** 是可选开启的，且仅在 context 长度小于 8192 tokens 的端点上默认应用。
   - 一位成员澄清道：*middle out 只有在达到 1m 时才会生效对吧？（在 flash 上）即使它已经开启了。*
- **Requesting Usage Download**：一位成员询问如何下载其使用数据（包括活动页面上显示的 tokens 和费用），以核实其额度使用情况。
   - 一位维护者回应称，虽然该功能目前尚不可用，但 *我们正在开发中*。
- **OpenRouter Enterprise Level Rate Limits**：一位用户询问关于 **enterprise-level rate limits** 的问题，并明确表示在余额达到 500 美元或以上时，这些限制会消失，但具体取决于上游 provider。
   - 另一位成员补充道：*从技术上讲，这取决于上游 provider*。
- **Auto Router for Fallback Models**：一位用户请求增加 **fallback model** 选项，类似于现有的 fallback provider 功能。
   - 另一位成员指出，[OpenRouter](https://openrouter.ai/) 已经通过 [Auto Router](https://openrouter.ai/openrouter/auto) 和 `models` 参数实现了这一点，详见 [文档](https://openrouter.ai/docs/features/model-routing)。
- **OpenRouter EU Provider Selection**：一位用户询问由于法律合规要求，是否可以仅选择位于 **European Union** 的 provider。
   - 一位维护者承认了这一需求，但指出目前覆盖范围有限，并提到 OpenRouter 允许选择 provider，建议针对严格的欧盟数据准则寻找 **EU certified provider**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1-zero:free">DeepSeek R1 Zero (free) - API, Providers, Stats</a>: DeepSeek-R1-Zero 是一个通过大规模强化学习 (RL) 训练的模型，没有经过监督微调 (SFT) 作为初步步骤。它的参数规模为 671B，其中 37B 在推理中处于激活状态...</li><li><a href="https://openrouter.ai/docs/features/prompt-caching">Prompt Caching - Optimize AI Model Costs with Smart Caching</a>: 使用 OpenRouter 的 prompt caching 功能降低您的 AI 模型成本。了解如何在 OpenAI、Anthropic Claude 和 DeepSeek 模型中缓存和重用响应。</li><li><a href="https://openrouter.ai/docs/features/model-routing">Model Routing - Smart Model Selection and Fallback</a>: 在 AI 模型之间动态路由请求。了解如何使用 OpenRouter 的 Auto Router 和 model fallback 功能以获得最佳性能和可靠性。</li><li><a href="https://openrouter.ai/openrouter/auto),">Discord</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1356488288785137696)** (43 条消息🔥): 

> `Cosine Annealing LR, Mini-batch vs Batch, Gradient Accumulation, Stanford CS 25 Transformers Course, Category theory` 


- **关于 Cosine Annealing Learning Rate (LR) 更新的辩论**：讨论了 **Cosine Annealing LR** 最好是在每个 **batch** 还是每个 **sample** 之后更新，并对每个 sample 更新时不同 sample 接受不同训练的问题表示了担忧。
   - 建议是在每个 **mini-batch** 之后更新，忽略 *exposure problem*，或者尝试修复它。
- **Mini-Batch 与 Batch 的术语混淆**：成员们讨论了机器学习中 **mini-batch** 和 **batch** 的区别，由于 **gradient accumulation** 和分布式训练等技术，两者的界限变得越来越模糊。
   - 有人提到 **mini-batch** 是在每个 optimizer step 之前运行的，而 **batch** 是一组唯一的数据，但术语 *batch size* 通常指 **mini-batch** 的大小。
- **Gradient Accumulation：利还是弊？**：成员们辩论了 **gradient accumulation** 的优缺点，一位成员回忆起它以前被弃用，但现在发现在训练早期阶段校准 optimizer states 具有潜在优势。
   - 另一位成员指出，当网络通信慢于计算时，**gradient accumulation** 是有益的，否则它被认为是负面的。
- **斯坦福大学向公众开放 CS25 Transformers 课程**：斯坦福大学已通过 Zoom 向公众开放其 **CS25 Transformers** 研讨课，内容包括与研究人员的讨论，涵盖从 **LLM** 架构到创意应用等主题。
   - 该课程包括讲座、社交活动、交流环节以及一个用于讨论的 [Discord server](https://discord.gg/2vE7gbsjzA)，往期讲座可在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) 上观看。
- **范畴论 (Category Theory)：逆向工程深度学习？**：有人分享了一个[链接](https://x.com/norabelrose/status/1907169483678212292)，内容是关于 **category theory** 是否可以是逆向工程深度学习的最佳语言的思想实验。
   - 原帖认为神经网络具有 *embeddings*（或有意义的神经元激活模式），而不是 *representations*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/norabelrose/status/1907169483678212292">来自 Nora Belrose (@norabelrose) 的推文</a>：神经网络没有“representations”，它们具有 embeddings，或者说有意义的神经元激活模式。它们在使我们能够执行某些任务的意义上是有意义的。不同...</li><li><a href="https://stanford.zoom.us/j/91661468474?pwd=Vo3qciJI6gWLoA8cFaSbhbYpBXs1lQ.1).">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://www.youtube.com/watch?v=XfpMkf4rD6E&ab_channel=StanfordOnline)"> - YouTube</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1356413942821425242)** (21 messages🔥): 

> `ACL Rebuttals, Deep Sets for Triangle Area, Comparing Language Model Embeddings, Relative Representations, Convergence of Representations in AI` 


- **Rebuttal 提交后提醒审稿人**：一位成员询问在提交 rebuttal 后一天向 **ACL 审稿人**发送额外消息是否合适，另一位成员建议如果 rebuttal 截止日期即将到来或后续跟进可能需要几天时间，这样做是合理的。
   - 提问者计划在当天晚上进行跟进，截止日期是周四。
- **Deep Sets 计算三角形面积，未产生任何见解**：一位成员分享了一篇题为 *Deep Sets for the Area of a Triangle* ([arxiv link](https://arxiv.org/abs/2503.22786)) 的论文，该论文提出了一个 **Deep Sets** 形式的三角形面积多项式公式。
   - 摘要总结道，该项目受宇宙学中 n 点统计计算复杂度问题的启发，但最终*没有获得任何形式的见解*。
- **比较语言模型嵌入矩阵**：一位成员询问了分析和证明两个使用相同 tokenizer 但具有不同维度**嵌入矩阵（embedding matrices）**训练的语言模型相似性的方法。
   - 建议包括使用 **relative representations**、**最小二乘映射（least squares mapping）**，以及比较 $W^T W$ 特征值分解的主成分。
- **Relative Representations 被提议作为解决方案，但可能并不完美**：一位成员建议将 **relative representations** ([arxiv link](https://arxiv.org/abs/2209.15430)) 作为比较语言模型嵌入的潜在方案，同时也对其有限的适用性提出了警告。
   - 他们链接了一篇讨论**神经表示中余弦相似度膨胀（cosine similarity inflation in neural representations）**的论文 ([arxiv link](https://arxiv.org/abs/2203.02053))，并进一步指出了一些讨论 **cosine** 是否是评估相似性最佳方式的相关工作。
- **AI 表示收敛，柏拉图风格**：一位成员链接了一篇论文，认为 AI 模型（特别是深度网络）中的表示正在收敛向一个**共享的现实统计模型**，类似于柏拉图的理想现实概念 ([arxiv link](https://arxiv.org/abs/2405.07987))。
   - 其他人建议使用 **CCA** 或 **SVCCA** 来比较嵌入矩阵，并引用了关于**奇异向量典型相关分析（Singular Vector Canonical Correlation Analysis）** ([arxiv link](https://arxiv.org/abs/1706.05806)) 和**投影加权 CCA（projection weighted CCA）** ([arxiv link](https://arxiv.org/abs/1806.05759)) 的论文。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2209.15430">Relative representations enable zero-shot latent space communication</a>: 神经网络将高维空间中数据流形的几何结构嵌入到潜表示中。理想情况下，潜空间中数据点的分布应该...</li><li><a href="https://arxiv.org/abs/2503.22786">A formula for the area of a triangle: Useless, but explicitly in Deep Sets form</a>: 任何数据点 $\vec{r}_i$ 的置换不变函数都可以写成 $ρ(\sum_iϕ(\vec{r}_i))$ 的形式，其中 $ρ$ 和 $ϕ$ 是合适的函数。这种形式在机器学习文献中被称为...</li><li><a href="https://arxiv.org/abs/2203.02053">Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning</a>: 我们展示了模态间隙（modality gap），这是多模态模型表示空间中一种有趣的几何现象。具体而言，我们展示了不同的数据模态（如图像和文本）是如何被嵌入的...</li><li><a href="https://arxiv.org/abs/1706.05806">SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability</a>: 我们提出了一种新技术，奇异向量典型相关分析（SVCCA），这是一种快速比较两种表示的工具，它对仿射变换具有不变性（允许比较...</li><li><a href="https://arxiv.org/abs/1806.05759">Insights on representational similarity in neural networks with canonical correlation</a>: 比较不同的神经网络表示并确定表示随时间演化的方式，在理解神经网络功能方面仍然是具有挑战性的开放问题。比较...</li><li><a href="https://arxiv.org/abs/2405.07987">The Platonic Representation Hypothesis</a>: 我们认为 AI 模型（特别是深度网络）中的表示正在趋于收敛。首先，我们调查了文献中许多收敛的例子：随着时间的推移和跨多个领域，这些方式...
</li>
</ul>

</div>

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1356742605911756830)** (4 messages): 

> `Learning Rate Impact, Scaling Efficiency, Model Oomph` 


- **Learning Rate 影响 Scaling Efficiency**：一名成员指出，**错误的 Learning Rate** 会改变 Scaling 的效率，从而影响常数 **A 和 B**。
   - 另一名成员补充道，**错误的 Learning Rate** 还会改变模型从给定数据量中获得的 **oomph**（效能），这似乎涉及到了 **beta**。
- **错误的 Learning Rate 很糟糕**：错误的 Learning Rate 真的很糟糕。
   - 极其糟糕。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1356368947909230804)** (5 messages): 

> `Neuronpedia Open Source, Delphi auto-interp server update, Actionable Interpretability Workshop at ICML 2025, Neuronpedia Datasets` 


- ****Neuronpedia** 正式开源！**：可解释性平台 **Neuronpedia** 现已采用 MIT 协议开源，并可在 [GitHub](https://github.com/hijohnnylin/neuronpedia) 上获取，支持通过 [Vercel 快速部署](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fhijohnnylin%2Fneuronpedia&env=NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY&envDescription=***Your%20Custom%20Website%20Name.%20For%20example%3A%20PuppyNeurons***&root-directory=apps/webapp&build-command=npx%20prisma%20generate%20%26%26%20npm%20run%20build%3Ademo&project-name=my-neuronpedia&repository-name=my-neuronpedia&demo-title=Neuronpedia&demo-description=Deploy%20your%20own%20custom%20Neuronpedia%20%F0%9F%9A%80%F0%9F%A7%A0%F0%9F%A7%90&demo-url=https%3A%2F%2Fneuronpedia.org)。
- ****Delphi** Auto-Interp 服务器准备更新**：**Neuronpedia** 的 auto-interp 服务器（使用 Eleuther 的 `Delphi`，原名 `sae-auto-interp`）计划更新至最新版本。
   - 此次更新旨在引入新的评分和解释类型，这得益于 **Neuronpedia** 的模块化设计以及 `Delphi` auto-interp 服务器现有的 OpenAPI 模式。
- **探索超过 4 TB 的 **Neuronpedia** 数据！**：总量超过 **4 TB** 的可解释性数据宝库已作为 [公共数据集 (Public Datasets)](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) 开放下载。
- ****Actionable Interpretability** 研讨会被 ICML 2025 接收**：根据[这条推文](https://x.com/OrgadHadas/status/1906744232557113579)，**Actionable Interpretability** 研讨会已被 #ICML2025 接收，论文投稿截止日期为 **5 月 9 日**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OrgadHadas/status/1906744232557113579">来自 Hadas Orgad (@OrgadHadas) 的推文</a>: 🎉 我们的 Actionable Interpretability 研讨会已被 #ICML2025 接收！🎉&gt;&gt; 关注 @ActInterp@tal_haklay @anja_reu @mariusmosbach @sarahwiegreffe @iftenney @megamor2 论文投稿截止日期...</li><li><a href="https://x.com/neuronpedia/status/1906793456879775745">来自 neuronpedia (@neuronpedia) 的推文</a>: 公告：我们正在开源 Neuronpedia！🚀 这包括我们所有的 mech interp 工具：可解释性 API、steering、UI、inference、autointerp、search，以及 4 TB 的数据 - 已被 35+ 项研究引用...</li><li><a href="https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source">Neuronpedia 现已开源 | The Residual Stream</a>: 面向所有人的免费可解释性工具。外加 4TB 数据集。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1356432473181126799)** (28 条消息🔥): 

> `Debugger 更新、SmolLM 评估问题、Open LLM Leaderboard 归一化、子任务聚合 PR` 


- **Debugger 状态仍然不明确**：成员们询问了 Debugger 的进展更新，但具体状态仍不清晰，一名成员通过询问 Debugger 在哪个分支上运行来提供帮助。
   - 一位成员分享了代码修改，怀疑这些修改可能导致了不必要的负载，随后报告修复了一个与问题选项数量相关的 bug，并建议提交 PR。
- **SmolLM 排行榜评估返回空的聚合分数**：一位成员报告称，在对 **SmolLM-1.7B** 使用 **lm-eval** 运行排行榜评估时，结果 JSON 中 **leaderboard_bbh**、**leaderboard_math_hard** 和 **leaderboard_musr** 等任务的聚合分数为空。
   - 他们提供了使用的命令和示例输出，指出单个任务照常报告了数值，并链接到了 [Hugging Face Dataset Card](https://huggingface.co/datasets/open-llm-leaderboard/HuggingFaceTB__SmolLM2-1.7B-details)。
- **Open LLM Leaderboard 上的非标准归一化**：讨论强调了 [Open LLM Leaderboard](https://open-llm-leaderboard-blog.static.hf.space/dist/index.html#reporting_a_fairer_average_for_ranking:_using_normalized_scores) 在评估和比较 LLM 时使用的一种“非标准归一化”方法。
   - 引入该归一化是为了解决优化 Prompt 和评估设置导致模型分数虚高的问题。
- **子任务聚合 PR 添加了子任务分数**：一位成员分享了一个 [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2867)，该 PR 从 Hugging Face 的 fork 中复制了添加子任务聚合的功能，以解决带有子任务的任务中缺失聚合分数的问题。
   - 另一位成员测试了该 PR，报告称通过 editable 模式安装 **lm_eval** 触发了一个无关的错误，但除此之外，该 PR 似乎按预期工作。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://open-llm-leaderboard-blog.static.hf.space/dist/index.html#reporting_a_fairer_average_for_ranking:_using_normalized_scores)">性能正处于平台期，让我们让排行榜再次陡峭起来 </a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2867).">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2867">leaderboard - 由 baberabb 添加子任务分数 · Pull Request #2867 · EleutherAI/lm-evaluation-harness</a>: 添加了来自 https://github.com/huggingface/lm-evaluation-harness/tree/main 的子任务聚合分数</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2028">vllm 后端失败 · Issue #2028 · EleutherAI/lm-evaluation-harness</a>: hi, i tried to eval : export CUDA_VISIBLE_DEVICES=&quot;2,3&quot; accelerate launch -m lm_eval --model vllm \ --model_args pretrained=&quot;THUDM/glm-4-9b&quot;,dtype=bfloat16 \ --tasks mmlu \ --devic...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1356409621115437298)** (5 条消息): 

> `GPT-NeoX 在 NVIDIA DGX Cloud 上的预训练，SLURM 集群限制，torchrun，DeepSpeed 启动模式` 


- **在 NVIDIA DGX Cloud 上绕过 DeepSpeed 启动器**：一位成员正在 **NVIDIA DGX Cloud** 上预训练 **GPT-NeoX**，但由于 SLURM 和 SSH 限制，必须绕过默认的 `deepy.py` 启动器，转而使用一个利用 hostfile 和 `torchrun` 的自定义脚本。
   - 该成员正在使用 [此脚本](https://github.com/ameyagodbole/hubble-gpt-neox/blob/3dd12f3cdf78a29116d55d795f783f903cc284c0/deepy_simple.py) 来执行参数解析并启动 `train.py`，并对其方法提出了疑问。
- **讨论直接执行 `train.py`**：一位成员询问是否可以直接使用编码后的 `ds_config` 和 `megatron_config` 参数启动 `python train.py`，以及在不使用 `torchrun` 的情况下如何处理 GPU 进程生成。
   - 另一位成员确认了这种手动绕过的方法，并建议进行进一步的模块化重构，以注入能够自分配 rank 并检测 GPU 数量的节点本地进程，通过原则而非协议进行协调。
- **在 DGX Cloud 上使用 Torchrun**：由于集群限制和 SSH 禁用，一位用户正在使用 **torchrun**，并参考了 [NVIDIA DGX Cloud 文档](https://docs.nvidia.com/dgx-cloud/slurm/latest/cluster-user-guide.html#example-pytorch-job) 中的评论和示例脚本。
   - 他们正在寻求指导，以确认其自定义解决方案的实现是否正确，以及是否在重复造轮子。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/issues/1203#issuecomment-2179596284)">My servers used for multi-node training do not have ssh. How can I launch multi-node training using the torchrun command? · Issue #1203 · EleutherAI/gpt-neox</a>：我用于多节点训练的服务器没有 SSH 服务。如何使用最基本的 torchrun 命令 (torch.distributed.launch) 启动多节点训练？我使用的服务器...</li><li><a href="https://docs.nvidia.com/dgx-cloud/slurm/latest/cluster-user-guide.html#example-pytorch-job))">3. Cluster User Guide &#8212; NVIDIA DGX Cloud Slurm Documentation</a>：未找到描述</li><li><a href="https://github.com/ameyagodbole/hubble-gpt-neox/blob/3dd12f3cdf78a29116d55d795f783f903cc284c0/deepy_simple.py">hubble-gpt-neox/deepy_simple.py at 3dd12f3cdf78a29116d55d795f783f903cc284c0 · ameyagodbole/hubble-gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - ameyagodbole/hubble-gpt-neox
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1356342795920736376)** (70 条消息🔥🔥): 

> `CodeScientist, OpenAI 开放语言模型, Meta 智能眼镜, 多主题 RLVR`

- **CodeScientist 自动化科学发现**：AllenAI 推出了 **CodeScientist**，这是一个用于自主科学发现的系统，它通过对研究文章和代码块进行遗传搜索（genetic search）来生成和评估机器生成的想法。在 Agent 和虚拟环境的数百次实验中，该系统产生了 **19 项发现**，详情见其 [论文](https://allenai.org/papers/codescientist)。
   - 该系统通过探索更广泛的设计空间并更彻底地评估研究产物，解决了当前 ASD 系统的局限性。不过一位用户指出，*生成的论文相当简短，更像是列表式的 PDF，且所有论文都是负面结果*。
- **OpenAI 预告开放权重语言模型**：OpenAI 计划在未来几个月内发布自 GPT-2 以来其首个开放权重（open-weight）语言模型，并正在寻求开发者反馈以最大化其效用，详情见 [Sam Altman 的推文](https://x.com/sama/status/1906793591944646898) 和 [OpenAI 的反馈表单](https://openai.com/open-model-feedback/)。
   - Altman 表示，他们 *不会做任何愚蠢的事情，比如规定如果你的服务月活跃用户超过 7 亿就不能使用我们的开放模型*。
- **Meta 计划推出带屏幕的智能眼镜**：据 [Mark Gurman 的报告](https://www.bloomberg.com/news/articles/2025-04-01/how-meta-s-upcoming-1000-smart-glasses-with-a-screen-will-work) 称，Meta 计划在今年晚些时候推出售价 **1000 美元以上、带屏幕且支持手势控制的智能眼镜**。
   - 成员们很感兴趣，想看看 *它们与 xreal 相比表现如何*。
- **用于 Expanding RL 论文的多学科数据**：[Expanding RL with Verifiable Rewards Across Diverse Domains](https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR) 论文中使用了多学科多项选择问答数据集 **ExamQA**。
   - 该数据集包含 **63.8 万个** 大学水平的实例，问题和客观答案均由领域专家为考试目的编写。
- **ChatGPT 获得新语音**：OpenAI 宣布了 **ChatGPT** 的新语音，引发了人们的兴奋以及对其潜在功能的猜测。
   - 一位成员开玩笑说这可能是个愚人节玩笑，而其他人则表达了真正的兴趣。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1906793591944646898">Sam Altman (@sama) 的推文</a>：TL;DR：我们很高兴能在未来几个月发布一个强大的、具备推理能力的全新开放权重语言模型，我们想与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://x.com/andrew_n_carr/status/1906917102315073578">Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：有求必应。引用 Andrew Carr (e/🤸) (@andrew_n_carr)：我有一个非常具体的 Agent 驱动用例，其难度恰好让网页抓取无法奏效。1. 我有一份包含 1000 多本书的清单...</li><li><a href="https://fxtwitter.com/camrobjones/status/1907086860322480233">Cameron Jones (@camrobjones) 的推文</a>：新预印本：我们在三方图灵测试中评估了 LLM（参与者同时与人类和 AI 交流，并判断哪个是哪个）。GPT-4.5（当被提示采取类人的人设时）被判定为...</li><li><a href="https://fxtwitter.com/UnitreeRobotics/status/1906962615596970167">Unitree (@UnitreeRobotics) 的推文</a>：宇树发布 | Unitree Dex5 灵巧手 - 敏捷掌控世界 🥳 单手具有 20 个自由度（16 个主动 + 4 个被动）。支持平滑的反向驱动（直接力控）。</li><li><a href="https://x.com/markgurman/status/1907151171908673892">Mark Gurman (@markgurman) 的推文</a>：最新消息：Meta 计划在今年晚些时候推出售价 1000 美元以上、配备屏幕和手势控制功能的智能眼镜。以下是它们的工作原理：https://www.bloomberg.com/news/articles/2025-04-01/how-meta-s-upc...</li><li><a href="https://x.com/sama/status/1906845532758405319">Sam Altman (@sama) 的推文</a>：我们不会做任何愚蠢的事情，比如规定如果你的服务月活跃用户超过 7 亿就不能使用我们的开放模型。我们希望每个人都能使用它！</li><li><a href="https://x.com/OpenAI/status/1907124258867982338?t=QHuzd2J5MxAPTXud4yyWow&s=19">OpenAI (@OpenAI) 的推文</a>：没开玩笑，ChatGPT 有了新语音。</li><li><a href="https://x.com/nathanbenaich/status/1907111014375506307">Nathan Benaich (@nathanbenaich) 的推文</a>：遗憾地看到——看起来 Meta 的 FAIR 部门即将终结。</li><li><a href="https://x.com/natolambert/status/1907221084774257009">Nathan Lambert (@natolambert) 的推文</a>：我加入了 OpenAI</li><li><a href="https://x.com/amir/status/1907086063635722309">Amir Efrati (@amir) 的推文</a>：你知道什么叫疯狂吗？在 3 个月内将年营收 30 亿美元的业务增长 30%。</li><li><a href="https://allenai.org/papers/codescientist">CodeScientist：通过基于代码的实验实现端到端半自动化科学发现</a>：尽管人们对软件工件（例如改进的 ML 算法）的自主科学发现 (ASD) 兴趣激增，但目前的 ASD 系统面临两个关键局限：(1) 它们在很大程度上探索的是变体...</li><li><a href="https://x.com/natolambert/status/1905628778920914972">Nathan Lambert (@natolambert) 的推文</a>：如今如果你对后训练（post training）感兴趣，你应该阅读 Joanne 发布的所有内容（大部分是这类内容或模型规范讨论）。她是唯一一个在公开场合讨论这些内容的人。</li><li><a href="https://huggingface.co/datasets/virtuoussy/Multi-subject-RLVR">Hugging Face 上的 virtuoussy/Multi-subject-RLVR 数据集</a>：未找到描述。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1356343466443145347)** (24 messages🔥): 

> `Pydantic Evals, Grok 解决数学问题, Gemini vs GPT 4.5, MidJourney v6, GPT-4o 翻译` 


- ****Pydantic Evals** 发布**: [Pydantic Evals](https://ai.pydantic.dev/evals/) 是一个强大的 **evaluation framework**，旨在帮助你系统地测试和评估所构建系统的性能和准确性，尤其是在使用 **LLMs** 时。
- ****Grok** 解决数学问题**: 经过多次失败的尝试，一位成员发现了一个 Prompt，成功让 **Grok** 解决了一个数学问题（图论中著名的 **Dubnovy Blazen 问题**），详见[这条推文](https://x.com/wtgowers/status/1906969462009381157)。
- ****Gemini** 过于热情**: 一位成员将 **Gemini** 与 **GPT-4.5** 进行了对比，观察到 *Gemini 过于急于解释一切，写得很多，同时又不时穿插一些微妙、孩子气的笑话*，*像一个患有自闭症的工程师*。
- ****MidJourney** 正在发力**: **MidJourney** 目前正处于最终模型的预览/评分阶段（可能明天发布），他们*绝对在憋大招*。
- ****GPT-4o** 翻译**: 成员们评论称，**GPT-4o** 的翻译比较简单，虽然体现了读者对简单英语的偏好，但翻译本身丢失了很多内容，详见[这段 YouTube 视频](https://www.youtube.com/watch?v=epIgulaBryA)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/wtgowers/status/1906969462009381157">Timothy Gowers @wtgowers (@wtgowers) 的推文</a>: 终于发生了：经过几次失败的尝试，我发现了一个 Prompt，让 Grok 解决了我一直在研究的一个数学问题（图论中著名的 Dubnovy Blazen 问题）...</li><li><a href="https://ai.pydantic.dev/evals/">Evals - PydanticAI</a>: 用于将 Pydantic 与 LLMs 结合使用的 Agent Framework / 适配层</li><li><a href="https://www.youtube.com/watch?v=epIgulaBryA"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1356588578548879411)** (4 messages): 

> `RL 中的 KL Penalty, Base Models vs Instruct Models, Reasoning 与 Reinforcement Learning` 


- **关于 RL 中取消 KL Penalty 的争议**: 产生了一个疑问：为什么在对 **Base Models** 进行 **RL** 时取消 **KL penalty** 可能有益，但在 **Instruct Models** 上却不然，正如 [Nathan Lambert 的文章](https://open.substack.com/pub/robotic/p/papers-im-reading-base-model-rl-grpo)中所述。
- **Base Models 需要 Reasoning**: 有建议认为 **Base Models** 需要更大的改变，但对于具有 **Reasoning** 组件的模型，情况可能会有所不同。
- **RLHF 书籍**: Nathan Lambert 正在编写一本[关于 RLHF 的书](https://rlhfbook.com/c/11-policy-gradients.html)，他强烈建议阅读。



**提到的链接**: <a href="https://open.substack.com/pub/robotic/p/papers-im-reading-base-model-rl-grpo">近期 Reasoning 研究：GRPO 调整、Base Model RL 和数据策展</a>: 在推理研究浪潮中，我认为值得阅读的论文。

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1356447838359388191)** (7 messages): 

> `OpenAI 回归, 通往高级 AI 的漫长周期` 


- **Lambert 发表对 OpenAI 的看法**: Nathan Lambert 在 Substack 文章中分享了[他对 OpenAI 回归的看法](https://natolambert.substack.com/p/some-thoughts-on-openai-returning)，并提到他可能也会用这种形式来表达一些不成熟的职业想法。
   - 他还提到就此事私信了一些 **OpenAI** 的员工，希望能找到那些因现状而感到被排挤的开源盟友。
- **Toner 的 Rising Tide Substack 启动**: Helen Toner 启动了她的新 Substack 频道 Rising Tide，并分享了一篇关于[通往高级 AI 的漫长周期](https://helentoner.substack.com/p/long-timelines-to-advanced-ai-have)的文章。
   - 在文中她指出，在 **21 世纪**上半叶实现人类水平 AI 的观点，曾被认为是一个需要强力证据支撑的大胆主张。



**提到的链接**: <a href="https://helentoner.substack.com/p/long-timelines-to-advanced-ai-have">通往高级 AI 的“漫长”周期已变得极短</a>: 在 2030 年代实现人类水平 AI 的前景令人震撼。

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1356431958460203119)** (46 条消息🔥): 

> `CUDA occupancy, GPU parallel processing, A100 thread limit, GRPO training with Qwen` 


- ****关于 A100 GPU 上最大并行线程数的争论****：讨论围绕如何计算 A100 GPU 上可以并行运行的最大线程数展开，一名成员指出该数字为 **96 * 2048**。
   - 另一名成员使用 GeoHot 的工具测试了这一假设，结果显示在其 A100 (96SM) GPU 上的实际限制为 **24576**，即在性能下降前每个 SM 有 **256 个线程**。
- ****Warp 调度被解释为一种隐藏延迟的方法****：讨论阐明了虽然 GPU 可以拥有许多并发线程，但由于寄存器空间和共享内存等资源限制，它们可能并非全部真正并行运行。
   - 一名成员指出，GPU 使用 **超额订阅 (oversubscription) 来隐藏延迟**，且 Warp 之间的上下文切换非常廉价（约 1 个周期），这与 CPU 不同。在“并行线程”限制之上增加线程不一定会增加运行时间，或者至少不会显著/可测量地增加。
- ****使用 Qwen 0.5B 模型进行 GRPO 训练的实验****：一名成员分享了他们在 GPUMODE kernel 数据集上完成的 **Qwen 0.5B** (code instruct) GRPO 训练运行，但该模型未能有效地生成 Triton kernel。
   - 他们假设，先通过 SFT 教会模型 Triton 实现的基础知识，然后再通过 GRPO 进行微调，会更加成功。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1356524186658537552)** (2 条消息): 

> `Disable autotune, Triton kernel` 


- **临时禁用 Triton Autotune**：一名成员询问如何临时禁用 autotune，因为他们的 **Triton kernel** 在两种情况下被调用，其中只有一种需要 autotuning，而他们使用的是 `triton.autotune` 装饰器。
   - 另一名成员建议使用 **全局变量** 来开启/关闭 autotune 并重新加载模块，或者在 kernel 内部使用 autotune 而不是作为装饰器。
- **全局变量技巧**：一名成员建议使用 **全局变量** 开启/关闭 autotune，并重新加载包含 Triton kernel 的模块。
   - 另一个选项是在 kernel 内部使用 autotune，而不是作为装饰器，这样就不需要重新加载模块。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1356652862557389001)** (2 条消息): 

> `Request for PMPP book PDF, PMPP book` 


- **请求 PMPP 书籍 PDF**：一名成员索要最近版本的 **PMPP 书籍 PDF**。
   - 消息中未提供链接或进一步详情。
- **PMPP 书籍查询**：一名用户请求最新版本的 **PMPP 书籍 PDF**。
   - 该请求未包含任何链接或进一步的上下文。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1356382435591323809)** (5 条消息): 

> `FlexAttention, Arbitrary Sequence Lengths, PyTorch 2.6, Tensor Subclass Use Case, Memory savings` 


- ****FlexAttention** 支持任意序列长度**：从 **PyTorch 2.6** 开始，**FlexAttention** 现在支持任意序列长度，解决了之前要求分段序列长度必须是 **128** 倍数的问题。
   - 这一改进是在圣何塞举行的 GPU mode 活动中与 Horace He 讨论的。
- **Tensor Subclass 使用场景疑问**：一名用户询问了 **tensor subclass** 的预期使用场景。
   - 这暗示了 PyTorch 的 tensor subclassing 功能中可能存在的问题或改进空间，引发了进一步调查。
- **希望通过删除 Tensor 来节省内存**：一名用户正在寻求在损失函数中删除参数 tensor 的方法，以实现约 **7GB** 的内存节省。
   - 该用户希望在 tensor 不再需要时释放与其关联的存储，即使在外部作用域中存在引用，同时确保其与 torch 编译兼容以避免 graph breaks；更多信息请参见 [GitHub Issue](https://github.com/pytorch/pytorch/issues/150265#issuecomment-2771053958)。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/issues/150265#issuecomment-2771053958">Graph break on Tensor._make_subclass · Issue #150265 · pytorch/pytorch</a>：🐛 描述我遇到的 bug。我在使用以下代码时遇到了问题：from torch import nn import torch torch_compile_options = { &quot;epilogue_fusion&quot; : True, &quot;max_autotune&quot; : True, &quot;shape_paddi...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

marksaroufim: https://arxiv.org/abs/2503.20313
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1356728018558517359)** (1 条消息): 

> `MLX, Apple hiring, ML systems` 


- **Apple 的 MLX 团队正在招聘**：Apple 正在[招聘工程师](https://jobs.apple.com/en-us/details/200577881/aiml-software-engineer-for-mlx-mlr?team=MLAI)加入其 **MLX 团队**，以推动 **ML 和 systems** 的前沿发展。
   - 该职位涉及构建可扩展的分布式训练和研究流水线，并与研究人员及软件工程师合作开发**新型 ML 研究算法**。
- **Apple 的 ML 系统开发**：Apple 的 Machine Learning Research 部门专注于构建为未来产品提供动力的技术。
   - 他们寻求具有系统工程和软件开发背景的工程师，来构建可扩展的分布式训练和研究流水线。



**提到的链接**：<a href="https://jobs.apple.com/en-us/details/200577881/aiml-software-engineer-for-mlx-mlr?team=MLAI">AIML - MLX, MLR 软件工程师 - 职位 - Apple 职业生涯</a>：申请 Apple 的 AIML - MLX, MLR 软件工程师职位。阅读职位详情并了解它是否适合你。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1356351752688893952)** (2 条消息): 

> `CUDA Program Execution, GPU Volumetric Data Processing` 


- **GPU 在处理体积数据时消耗数 GB 显存**：对于处理**体积数据**（如医疗领域）的模型，**512³ voxels**、**32 channels** 和 **fp16 activations** 的数据量在每一层可能产生 **8GiB** 的数据。
   - 这突显了某些类型的 GPU 计算对内存的巨大需求。
- **CUDA Kernel 代码编译与执行探讨**：一位成员正试图理解 **CUDA program** 的执行原理，并想知道从 **CPU 到 GPU** 通过 **PCIe bus** 传输的究竟是什么。
   - 他们假设 kernel 代码被编译成某种 GPU 机器字节码，当调用 kernel 代码时，这些代码会被发送到 GPU。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1356427831344759016)** (2 条消息): 

> `Egg noodles with chicken and vegetables, Image Analysis with YouTube` 


- **美味蛋面料理亮相**：一位成员展示了一道**黑椒酱汁鸡肉蔬菜炒蛋面**，食材包括蛋面、酱油、鸡胸肉、洋葱、红甜椒、法式四季豆、牛脂和芝麻。
   - 分享了该菜品的图片 ([IMG_20250401_045505.jpg](https://cdn.discordapp.com/attachments/1215328286503075953/1356427831277654316/IMG_20250401_045505.jpg?ex=67edd8dc&is=67ec875c&hm=7a5a192567d149153286be68eb7051ea721ac90d8f9ab96da9f7f5fa2cd1717f&))。
- **YouTube 分析上传**：针对一段标题为 * - YouTube* 的 [YouTube 视频](https://www.youtube.com/watch?v=vWw-1bk7k2c)进行了图像分析，尽管该视频的描述未定义。



**提到的链接**：<a href="https://www.youtube.com/watch?v=vWw-1bk7k2c"> - YouTube</a>：未找到描述

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1356385134018035773)** (3 条消息): 

> `NYC Meetups, Community Meetup` 


- **NYC 见面会正在筹划中**：一位成员询问是否有在 NYC 的见面会，另一位成员确认他们正在计划中。
   - 询问的成员表现得非常兴奋，表示有兴趣参加。
- **社区计划见面会**：正在计划一次社区见面会。
   - 积极的社区成员对即将到来的计划感到兴奋。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1356745505979240612)** (1 messages): 

> `Megatron Tensor Parallelism, Fused/Parallel CE Loss` 


- **Megatron Tensor Parallelism 深度解析图解版！**：一位成员编写了一份关于 **Megatron-style tensor parallelism** 的图解深度解析，涵盖了 **fused/parallel CE loss**，并征求内容反馈。
   - 查看 [图解深度解析](https://danielvegamyhre.github.io/ml/performance/2025/03/30/illustrated-megatron.html) 以加深你对 **ML scalability & performance techniques** 的理解。
- **征求关于 Megatron-Style Parallelism 深度解析的反馈**：这份关于 **Megatron-style tensor parallelism** 图解深度解析的作者正在征求反馈。
   - 文章涵盖了 **fused/parallel CE loss** 等方面，旨在增强对 **ML scalability and performance** 的理解。



**提到的链接**：<a href="https://x.com/vega_myhre/status/1906767278869479570">来自 Daniel Vega-Myhre (@vega_myhre) 的推文</a>：对于任何想要加深对 ML 可扩展性和性能技术理解的 ML 爱好者，我写了一篇关于 Megatron 风格张量并行的图解深度解析：https://danielvegamyhre.git...

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1356365091754344558)** (1 messages): 

> `AlphaGeometry, LLM for kernel optimization` 


- **探讨用于 kernel 优化的 AlphaGeometry 风格 LLM + verifier 方案**：一位成员询问了之前关于使用 **AlphaGeometry** 风格的 **LLM + verifier** 方法进行 **kernel optimization** 的探索情况。
   - 他们询问这是否曾被尝试或讨论过，并承认由于自己是该领域的新手，可能会重新发现已有的概念。
- **关于 LLM 和 kernel 优化的新手提问**：一位非常新的成员正在*重新发现*关于 **kernel optimization** 的想法，并请求获取过去讨论的指引。
   - 他们表示有兴趣了解是否有人能指出将 **AlphaGeometry style LLM + verifier** 用于 kernel 优化过程这一想法的进展。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1356391100633452714)** (9 messages🔥): 

> `OpenAI Open-Weight Reasoning Models, PR Review Requests, Arc AGI PR, Collisions PR, CodeIO Dataset Merged` 


- **Reasoning Gym 新横幅亮相**：分享了一个 [使用 4o 制作](https://cdn.discordapp.com/attachments/1316377974672588850/1356391100289388625/reasoning-gym-banner-cropped.png?ex=67edb6a6&is=67ec6526&hm=0f593d186678816d46740136f6fba2c97da73888a8cd4a7d81b19ac01fdaad04&) 的 reasoning-gym 新横幅，并可能通过 PR 将其添加到 readme 中。
   - 另一位成员指出横幅上描绘的魔方 *“不是一个有效的魔方 reeeeeeeeeee”*。
- **OpenAI 将开源强力模型？**：成员们对 **OpenAI** 可能发布强力的权重开放（open-weight）推理模型表示惊讶。
   - 一位成员推测这可能会显著提升 **OpenAI** 的估值，而另一位成员则评审了两个待处理的 PR。
- **Arc AGI 和 Collisions PR 已准备好接受审查**：arc agi 和 collisions 的 **PR** 已提交评审。
   - **Collisions PR** 被要求进行修改，特别是取消暂存那些仅运行而未作修改的 notebook。
- **CodeIO 数据集已导入**：**CodeIO dataset** 在延迟后已合并；后续的后处理将使其与现有实现保持一致。
   - 感谢合并 **CodeIO dataset** 的用户。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1356743710578053181)** (3 messages): 

> `.py scripts vs .cu files, active python leaderboards` 


- **寻求澄清：Python 与 CUDA 提交**：一位成员询问排行榜目前是否仅接受 **.py 脚本** 而不接受 **.cu 文件**。
   - 另一位成员建议查看之前的消息以澄清提交指南。
- **活跃排行榜确认**：一位成员询问是否所有活跃的排行榜目前都仅限于 **Python** 提交。
   - 另一位成员引导他们查看之前的消息，其中可能包含关于活跃排行榜和提交要求的详细信息。


  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1356343591789793493)** (17 条消息🔥): 

> `vectorsum, conv2d, vectoradd, matmul, grayscale` 


- **vectorsum 基准测试席卷排行榜**：使用 Modal runners 在 **L4** 和 **H100** GPU 上进行的多个 `vectorsum` 基准测试提交已成功，提交 ID 为 **3372**、**3374**、**3375**、**3395**、**3396** 和 **3397**。
- **conv2d 基准测试在多个 GPU 上成功**：使用 Modal runners 在 **L4**、**T4**、**A100** 和 **H100** GPU 上进行的 `conv2d` 排行榜提交已成功，提交 ID 为 **3373**。
- **vectoradd 基准测试登陆 T4 和 H100**：使用 Modal runners 在 **H100** 和 **T4** GPU 上进行的 `vectoradd` 排行榜提交已成功，提交 ID 分别为 **3394** 和 **3399**。
- **matmul 基准测试对接 A100**：使用 Modal runners 在 **A100** GPU 上进行的 `matmul` 排行榜提交已成功，提交 ID 为 **3400** 和 **3408**。
- **grayscale 测试启动**：使用 Modal runners 在 **H100** GPU 上进行的多个 `grayscale` 测试提交已成功，提交 ID 为 **3402**、**3403**、**3404**、**3405**、**3406** 和 **3407**。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1356343981839225033)** (75 条消息🔥🔥): 

> `Cursor's Funding Round, Etched's New Transformer ASIC, OpenAI's New Open-Weight Language Model, OpenDeepSearch (ODS), Sophont: Open Multimodal Foundation Models for Healthcare` 


- **Cursor 完成巨额融资，引领 Vibe Coding**：Cursor 以 **96 亿美元**的投后估值完成了 **6.25 亿美元**的融资，由 Thrive 和 A16z 领投，Accel 为新支持者，实现了 **2 亿美元 ARR**，较 2024 年 11 月的上轮融资增长了 4 倍 ([来源](https://www.theinformation.com/articles/move-openai-startup-behind-cursor-became-hottest-vibiest-thing-ai?rc=sslhyj))。
   - Abe Brown 指出 Cursor 的估值增长迅速，引发了 *vibe coding* 这一流行语，其估值可能达到 **100 亿美元**。
- **Etched，Transformer ASIC 初创公司完成 8500 万美元融资**：Etched 是一家开发 Transformer ASIC 的初创公司，完成了未公开的 **8500 万美元**融资，估值为 **15 亿美元**，此前曾进行过两轮分别为 **5 亿美元**和 **7.5 亿美元**的隐身轮融资；其芯片 **Sohu** 在运行 Llama 70B 时每秒可处理超过 **500,000** 个 token ([来源](https://fxtwitter.com/ArfurRock/status/1906756943349260682))。
   - Etched 声称一台 **8xSohu** 服务器可替代 **160 台 H100**，但 Sohu 无法运行 CNN, LSTM, SSM 或任何其他 AI 模型。
- **OpenAI 开放：即将发布 Open-Weight 模型**：OpenAI 计划在未来几个月内发布自 GPT-2 以来的首个 Open-Weight 语言模型，并征求开发者关于如何使其发挥最大效用的反馈 ([来源](https://openai.com/open-model-feedback/))。
   - 该公司将根据其 preparedness framework 评估该模型，并在旧金山、欧洲和亚太地区举办开发者活动，以收集反馈并测试早期原型；Nathan Lambert 预计这将是一个具有 MIT/Apache 许可证的 **30B** 参数 reasoning model ([来源](https://x.com/natolambert/status/1907072904321490953))。
- **OpenDeepSearch 开启网络搜索**：Seoong79 宣布发布 **OpenDeepSearch (ODS)**，这是一个可与任何 LLM 配合使用的开源搜索 Agent，在 DeepMind 具有挑战性的多跳（multi-hop）FRAMES 基准测试中，其准确率比 OpenAI 的专门网络搜索模型 GPT-4o-Search 高出 +9.7% ([来源](https://x.com/sewoong79/status/1906595129965912341))。
- **Sophont 初创公司寻求解决医疗 AI 问题**：iScienceLuvr 宣布成立 **Sophont**，该公司致力于为医疗保健的未来构建开源多模态 Foundation Models，旨在打造医疗 AI 领域的 *DeepSeek* ([来源](https://x.com/iscienceluvr/status/1906790937604579430))。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://labs.amazon.science/blog/nova-act">Introducing Amazon Nova Act | Amazon AGI Labs</a>: 未找到描述</li><li><a href="https://x.com/sama/status/1906793591944646898">Sam Altman (@sama) 的推文</a>: TL;DR: 我们很高兴在未来几个月发布一个强大的、具有推理能力的全新 open-weight 语言模型，我们希望与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898">Sam Altman (@sama) 的推文</a>: TL;DR: 我们很高兴在未来几个月发布一个强大的、具有推理能力的全新 open-weight 语言模型，我们希望与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=4]">Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>: 我有一个激动人心的消息：我创办了一家公司！介绍一下 Sophont。我们正在为医疗保健的未来构建开源多模态基础模型。我们需要医疗 AI 领域的 DeepSeek，而 @SophontAI 将会...</li><li><a href="https://x.com/rauchg/status/1906814800426086861?s=46">Guillermo Rauch (@rauchg) 的推文</a>: 我们正在构建一个用于运行任意计算的 API，目标是 agentic AI 使用场景和长时间运行的任务。是的，它可以运行服务器。由每天运行 100 万次以上 @vercel 构建的基础设施提供支持，并经过优化...</li><li><a href="https://x.com/stevenheidel/status/1906797154301329845">Steven Heidel (@stevenheidel) 的推文</a>: 我们今年将发布一个可以在你自己的硬件上运行的模型。引用 Sam Altman (@sama) 的话，TL;DR: 我们很高兴在未来几个月发布一个强大的、具有推理能力的全新 open-weight 语言模型...</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">Sewoong Oh (@sewoong79) 的推文</a>: 我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 搭配使用时，ODS 在网络搜索方面优于 OpenAI 的专用模型 GPT-4o-Search...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682]">Arfur Rock (@ArfurRock) 的推文</a>: 🚨新独角兽警报 —— Etched，全球首个 Transformer ASIC。在另外两轮分别为 5 亿美元和 7.5 亿美元的隐身轮融资之后，又以 15 亿美元的估值完成了未公布的 8500 万美元融资。7.5 亿美元的那轮融资就在大约 2 个月前。引用...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906756943349260682">Arfur Rock (@ArfurRock) 的推文</a>: 🚨新独角兽警报 —— Etched，全球首个 Transformer ASIC。在另外两轮分别为 5 亿美元和 7.5 亿美元的隐身轮融资之后，又以 15 亿美元的估值完成了未公布的 8500 万美元融资。7.5 亿美元的那轮融资就在大约 2 个月前。引用...</li><li><a href="https://fxtwitter.com/stevenheidel/status/1906797154301329845">Steven Heidel (@stevenheidel) 的推文</a>: 我们今年将发布一个可以在你自己的硬件上运行的模型。引用 Sam Altman (@sama) 的话，TL;DR: 我们很高兴在未来几个月发布一个强大的、具有推理能力的全新 open-weight 语言模型...</li><li><a href="https://x.com/samjulien/status/1907102494284783789">Sam Julien (@samjulien) 的推文</a>: 很高兴看到我们 10 月份发布的 @Get_Writer Palmyra X 004 模型在 @rungalileo Agent 排行榜上排名第 9，击败了同期的其他模型，如 Claude 3.5 Sonnet 和 Gemini ...</li><li><a href="https://x.com/juberti/status/1906864981951598858?s=46">Justin Uberti (@juberti) 的推文</a>: 新的 OpenAI 实时转录 API 现在支持 WebRTC 连接，这让你可以轻松地将 MediaStream 或 &lt;audio&gt; 元素连接到 API。刚刚做了一个快速演示来展示这个功能...</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882]">Amazon Science (@AmazonScience) 的推文</a>: 认识一下 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，可以将稳健的步骤组合成复杂的工作流；处理从预订到 QA 的一切事务...</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360]">Arfur Rock (@ArfurRock) 的推文</a>: Cursor 融资轮结束 —— 由 Thrive 和 A16z 领投，投后估值 96 亿美元，融资 6.25 亿美元。Accel 是新加入的投资者。ARR 为 2 亿美元，比 2024 年 11 月 25 亿美元估值那一轮增长了 4 倍。ARR 倍数与上一轮持平，为 50 倍。引用 Abe Brown ...</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360">Arfur Rock (@ArfurRock) 的推文</a>: Cursor 融资轮结束 —— 由 Thrive 和 A16z 领投，投后估值 96 亿美元，融资 6.25 亿美元。Accel 是新加入的投资者。ARR 为 2 亿美元，比 2024 年 11 月 25 亿美元估值那一轮增长了 4 倍。ARR 倍数与上一轮持平，为 50 倍。引用 Abe Brown ...</li><li><a href="https://fxtwitter.com/samjulien/status/1907102494284783789">Sam Julien (@samjulien) 的推文</a>: 很高兴看到我们 10 月份发布的 @Get_Writer Palmyra X 004 模型在 @rungalileo Agent 排行榜上排名第 9，击败了同期的其他模型，如 Claude 3.5 Sonnet 和 Gemini ...</li><li><a href="https://x.com/gm8xx8/status/1907194182306799960">推文</a></li>

<li><a href="https://x.com/gm8xx8/status/1907194182306799960]">来自 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：证明还是虚张声势？在 2025 年美国数学奥林匹克竞赛（USAMO）上评估 LLM。尽管在仅限答案的基准测试中得分很高，但顶级模型在 2025 年 USAMO 全解评估中的得分不足 5%。分析揭示了与...相关的失败模式。</li><li><a href="https://natolambert.substack.com/p/some-thoughts-on-openai-returning">关于 OpenAI 回归开放发布的一些思考</a>：欢迎来到我未经编辑的额外想法博客。</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882">来自 Amazon Science (@AmazonScience) 的推文</a>：认识 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，将稳健的步骤组合成复杂的流水线；处理从预订到 QA 的一切...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：我有一个激动人心的消息：我开公司了！介绍 Sophont。我们正在为医疗保健的未来构建开放的多模态基础模型。医疗 AI 领域需要一个 DeepSeek，而 @SophontAI 将...</li><li><a href="https://x.com/altryne/status/1907173680456794187">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：字节跳动 OmniHuman 现已发布！OmniHuman 几个月前以令人难以置信的 AI Avatar 动画惊艳了我们所有人，现在终于向公众开放（不是免费的！）。它非常慢...</li><li><a href="https://x.com/natolambert/status/1907072904321490953?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：关于 OpenAI 为什么可能会发布一个采用 MIT/Apache 协议的约 30B 参数推理模型的一些基本逻辑。* OpenAI 只会发布在同尺寸类别中明显处于 SOTA 水平的产品 * 将发布一个推理模型，...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906768733135098360">来自 Arfur Rock (@ArfurRock) 的推文</a>：Cursor 融资完成 —— 由 Thrive 和 A16z 领投，投后估值 96 亿美元，融资 6.25 亿美元。Accel 是新加入的投资者。ARR 达 2 亿美元，较 2024 年 11 月 25 亿美元估值融资时增长了 4 倍。ARR 倍数与上一轮持平，为 50 倍。引用 Abe Brown ...</li><li><a href="https://fxtwitter.com/natolambert/status/1907072904321490953">来自 Nathan Lambert (@natolambert) 的推文</a>：关于 OpenAI 为什么可能会发布一个采用 MIT/Apache 协议的约 30B 参数推理模型的一些基本逻辑。* OpenAI 只会发布在同尺寸类别中明显处于 SOTA 水平的产品 * 将发布一个推理模型，...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682">来自 Arfur Rock (@ArfurRock) 的推文</a>：🚨 新独角兽预警 —— Etched，全球首款 Transformer ASIC。在经历了 5 亿美元和 7.5 亿美元的两轮秘密融资后，完成了一笔未公开的 8500 万美元融资，估值达到 15 亿美元。7.5 亿美元那一轮就在大约 2 个月前。引用...</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q]">来自 Sewoong Oh (@sewoong79) 的推文</a>：我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 配对时，ODS 在...上优于 OpenAI 的专用网络搜索模型 GPT-4o-Search。</li><li><a href="https://fxtwitter.com/sewoong79/status/1906595129965912341">来自 Sewoong Oh (@sewoong79) 的推文</a>：我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 配对时，ODS 在...上优于 OpenAI 的专用网络搜索模型 GPT-4o-Search。</li><li><a href="https://fxtwitter.com/AmazonScience/status/1906758835240312882">来自 Amazon Science (@AmazonScience) 的推文</a>：认识 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，将稳健的步骤组合成复杂的流水线；处理从预订到 QA 的一切...</li><li><a href="https://fxtwitter.com/rauchg/status/1906814800426086861">来自 Guillermo Rauch (@rauchg) 的推文</a>：我们正在构建一个用于运行任意计算的 API，目标是 Agentic AI 用例和长时间运行的任务。是的，它可以运行服务器。由运行我们每日 100 万次以上 @vercel 构建的基础设施驱动，并经过优化...</li><li><a href="https://x.com/gm8xx8/status/1907194182306799960]">来自 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：证明还是虚张声势？在 2025 年美国数学奥林匹克竞赛（USAMO）上评估 LLM。尽管在仅限答案的基准测试中得分很高，但顶级模型在 2025 年 USAMO 全解评估中的得分不足 5%。分析揭示了与...相关的失败模式。</li><li><a href="https://fxtwitter.com/gm8xx8/status/1907194182306799960">来自 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：证明还是虚张声势？在 2025 年美国数学奥林匹克竞赛（USAMO）上评估 LLM。尽管在仅限答案的基准测试中得分很高，但顶级模型在 2025 年 USAMO 全解评估中的得分不足 5%。分析揭示了与...相关的失败模式。</li><li><a href="https://x.com/steph_palazzolo/status/1907078693240914306">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：这不是愚人节玩笑：ChatGPT 的营收在短短三个月内飙升了 30%。在今天早上的 Agenda 中，@amir 和我深入探讨了 ChatGPT 的增长、OpenAI 与 Google 的注意力之战，以及...</li><li><a href="https://x.com/ste_

<li><a href="ph_palazzolo/status/1907078693240914306]">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：这不是愚人节玩笑：ChatGPT 的收入在短短三个月内飙升了 30%。在今早的 Agenda 中，@amir 和我深入探讨了 ChatGPT 的增长、OpenAI-Google 的注意力之争，以及...</li><li><a href="https://fxtwitter.com/iscienceluvr/status/1906790937604579430">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：我有一个激动人心的消息：我创办了一家公司！介绍一下 Sophont。我们正在为医疗保健的未来构建开放的多模态基础模型（multimodal foundation models）。我们需要医疗 AI 领域的 DeepSeek，而 @SophontAI 将会...</li><li><a href="https://x.com/altryne/status/1907173680456794187]">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：ByteDance OmniHuman 现已上线！几个月前，OmniHuman 凭借令人难以置信的 AI Avatar 动画惊艳了我们所有人，现在终于向公众开放了（不是免费的！）。它非常慢...</li><li><a href="https://fxtwitter.com/altryne/status/1907173680456794187">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：ByteDance OmniHuman 现已上线！几个月前，OmniHuman 凭借令人难以置信的 AI Avatar 动画惊艳了我们所有人，现在终于向公众开放了（不是免费的！）。它非常慢...</li><li><a href="https://fxtwitter.com/steph_palazzolo/status/1907078693240914306">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：这不是愚人节玩笑：ChatGPT 的收入在短短三个月内飙升了 30%。在今早的 Agenda 中，@amir 和我深入探讨了 ChatGPT 的增长、OpenAI-Google 的注意力之争，以及...</li><li><a href="https://julian.digital/2025/03/27/the-case-against-conversational-interfaces/">反对对话式界面的理由</a>：对话式界面有点像是一个梗（meme）。每隔几年，就会出现一项闪亮的 AI 新进展，科技界的人们就会感叹：“就是它了！下一个计算范式（computing paradigm）来了！我们将只使用...”</li><li><a href="https://github.com/aws/nova-act?tab=readme-ov-file">GitHub - aws/nova-act: Amazon Nova Act 是一个新 AI 模型的预览研究版，供开发者构建在 Web 浏览器中执行操作的 Agent</a>：Amazon Nova Act 是一个新 AI 模型的预览研究版，供开发者构建在 Web 浏览器中执行操作的 Agent - aws/nova-act</li><li><a href="https://www.thewrap.com/openai-valued-300-billion-new-round-funding/">OpenAI 在创纪录的融资后估值达到 3000 亿美元</a>：OpenAI 周一宣布已融资 400 亿美元，这无疑是私营科技公司历史上规模最大的一轮融资。</li><li><a href="https://buttondown.com/ainews/archive/ainews-41b-raised-today-openai-300b-cursor-95b/">[AINews] 今日融资超过 410 亿美元（OpenAI 估值 300b，Cursor 估值 9.5b，Etched 估值 1.5b）</a>：更多的资金就是你所需要的。2025年3月28日至3月31日的 AI 新闻。我们为你检查了 7 个 Reddit 子版块、433 个 Twitter 账号和 30 个 Discord 服务器（230 个频道和 17665 条消息）....
</li>

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1356369896812056766)** (42 messages🔥): 

> `DeepSeek R1, xAI Acquires X, Hyperparameter tuning LLMs, SFTTrainer hanging, stable_baselines3 CPU faster than GPU` 


- **DeepSeek R1 完胜西方实验室**：一位用户分享了一条 [推文](https://x.com/rosstaylor90/status/1906982769361666383)，批评那些试图贬低 **DeepSeek R1** 的乏力攻击；**DeepSeek R1** 通过极强的执行力和资源效率，战胜了臃肿的西方实验室。
   - 成员补充道，**DeepSeek** 还以极具包容性的 **MIT license** 发布了权重，并通过 **GRPO** 为 **GPU** 资源匮乏者普及了 **RL**。
- **xAI 吞并 X**：一位成员分享了一条 [推文](https://x.com/elonmusk/status/1905731750275510312)，宣布 **xAI** 已通过全股票交易收购 **X**，此次交易对 **xAI** 的估值为 **800 亿美元**，对 **X** 的估值为 **330 亿美元**。
   - 这一结合通过将 **xAI** 先进的 **AI** 能力和专业知识与 **X** 巨大的影响力相结合，释放了巨大的潜力。
- **LLM 超参数调优资源寻求**：一位成员询问有关在微调 **LLM** 时选择超参数的资源，希望能找到一个解决上下文变化如何影响特定超参数的“神级资源”。
   - 另一位成员建议查看 **Unsloth** 的 **Discord**，并提供了 [Unsloth 的 LoRA 超参数指南](https://docs.unsloth.ai/get-started/beginner-start-here/lora-hyperparameters-guide) 链接。
- **SFTTrainer 在训练中途冻结**：一位用户报告称，他们的 **SFTTrainer** 在截断训练数据集后发生挂起，并在一个小时后超时。
   - 一位成员建议，问题可能是由于进度条未出现以及 **TrainingArguments** 或 **Trainer** 设置可能存在配置错误导致的。
- **在 stable_baselines3 中 CPU 运行速度超过 GPU**：一位用户报告称，在 **GPU** 上使用 **MlpPolicy** 运行 **PPO** 时收到警告，提示该策略主要针对 **CPU** 设计，并链接到了一个 [GitHub issue](https://github.com/DLR-RM/stable-baselines3/issues/1245)。
   - 用户对为什么在运行 **Multi-Layer Perceptron** 时 **CPU** 可能比 **GPU** 更快感到困惑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/elonmusk/status/1905731750275510312">Elon Musk (@elonmusk) 的推文</a>：@xAI 已通过全股票交易收购 @X。此次合并对 xAI 的估值为 800 亿美元，对 X 的估值为 330 亿美元（450 亿美元减去 120 亿美元债务）。自两年前成立以来，xAI 已迅速成为最……</li><li><a href="https://x.com/rosstaylor90/status/1906982769361666383?t=cn1b56hucDKly6S6HcNn0w&s=19">Ross Taylor (@rosstaylor90) 的推文</a>：对 Microsoft AI 没有意见，他们做得很好——但我只是讨厌这种试图贬低 DeepSeek R1 的懒惰攻击。实际上，DeepSeek 通过顽强的……战胜了臃肿的西方实验室。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/lora-hyperparameters-guide">LoRA 超参数指南 | Unsloth 文档</a>：LoRA 超参数的最佳实践，并了解它们如何影响微调过程。</li><li><a href="https://hf.co/learn">Hugging Face - 学习</a>：未找到描述</li><li><a href="https://huggingface.co/posts/RudeBoi/909027081025413#6794f29be522c7de7ffc4d78">Hugging Face 上的 @RudeBoi：“谁能帮我解释一下为什么我会收到这个错误信息？拜托……”</a>：未找到描述</li><li><a href="https://www.geeksforgeeks.org/hyperparameter-tuning/">超参数调优 - GeeksforGeeks</a>：超参数调优是为机器学习模型选择最佳配置设置以增强其性能的过程，技术包括 GridSearchCV、RandomizedSearchCV 等……</li><li><a href="https://huggingface.co/docs/transformers/hpo_train">超参数搜索</a>：未找到描述</li><li><a href="https://huggingface.co/docs/setfit/how_to/hyperparameter_optimization">超参数优化</a>：未找到描述</li><li><a href="https://github.com/DLR-RM/stable-baselines3/issues/1245">[Bug]: A2C CPU 利用率非常低 · Issue #1245 · DLR-RM/stable-baselines3</a>：🐛 Bug 在具有 12 核 / 24 线程的 Ryzen Threadripper 机器上使用具有 50 个矢量化环境的 A2C 算法时，我得到的 CPU 利用率非常低，几乎是单核的。而且……
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1356490860170186833)** (3 messages): 

> `Agents Course Unit 2.1, Run Jupyter Lab Locally, RL Course Frozen Lake issue` 


- **Agents Course Unit 2.1 本地运行**：一位成员提到他们正在学习 **Agent Course Unit 2.1**，并且在安装了 **jupyterlab** 及其 widgets 的本地 venv 内核中运行成功。
   - 他们指出 Colab 对他们来说不是一个选项，因为他们没有 **Google account**。
- **本地运行 Jupyter Lab 的说明**：一位成员正在研究如何运行 notebook，询问是否应该克隆 repo 并根据[这些说明](https://huggingface.co/learn/agents-course/unit1/what-are-llms#how-are-llms-used-in-ai-agents)在本地运行 jupyter-lab。
   - 该用户对应该在哪里运行感到困惑，并提到如果使用 Colab，他们不确定如何将 notebook 链接到 **Colab Google Workspace**。
- **Frozen Lake 代码已修复**：一位成员注意到 **RL Course** 中 **Unit 2** 提供的 **Frozen Lake** 代码由于 Python 版本问题无法运行。
   - 他们分享了[指向其 HuggingFace 页面的链接](https://huggingface.co/DarkDummo/q-FrozenLake-v1-4x4-noSlippery)，其中包含解决 pickle5 问题的代码。



**提及的链接**：<a href="https://huggingface.co/learn/agents-course/unit1/what-are-llms#how-are-llms-used-in-ai-agents">What are LLMs? - Hugging Face Agents Course</a>：未找到描述

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1356577896117960745)** (3 messages): 

> `OpenHands LM, Autonomous Agents, Nature article on data access` 


- ****OpenHands LM** 开启编程新篇章！**：据一位成员介绍，新的开源编程模型 **OpenHands LM** 已在 [Hugging Face](https://huggingface.co/all-hands/openhands-lm-32b-v0.1) 上线，其 **32B** 的规模适合在本地运行。
   - 它旨在用于软件开发的 autonomous agents，更多信息可在[项目博客](https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model)中找到。
- **Nature 文章的数据访问受限**：一篇 [Nature 文章](https://www.nature.com/articles/s41593-025-01905-6)根据临床试验方案限制了数据访问，以便与研究人员共享去标识化信息，但禁止公开。
   - 为了保护参与者的匿名性，任何可能识别其身份的信息都不会作为共享数据的一部分，特别是她的个性化语音合成器。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.nature.com/articles/s41593-025-01905-6">A streaming brain-to-voice neuroprosthesis to restore naturalistic communication - Nature Neuroscience</a>：自然交流是神经假体的目标。在这里，作者展示了一种神经假体，可以在瘫痪者尝试说话的同时恢复其声音，从而实现……</li><li><a href="https://huggingface.co/all-hands/openhands-lm-32b-v0.1">all-hands/openhands-lm-32b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://www.all-hands.dev/blog/introducing-openhands-lm-32b----a-strong-open-coding-agent-model)">All Hands AI</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/)** (1 messages): 

tonic_1: 非常酷
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1356718040636588223)** (1 messages): 

> `YOLO vertical object detection, CNN vertical object detection, Instance segmentation fragments` 


- **YOLO & CNN 寻求垂直视觉提升**：一位成员询问如何增强 **YOLO** 或任何 **CNN** 检测垂直物体的能力，并询问增加深度是否有帮助。
   - 回复建议探索数据增强（data augmentation）技术或自定义 loss functions。
- **碎片化 Instance Segmentation 的修复**：一位成员面临其 instance segmentation 模型检测到同一物体碎片的问题。
   - 他们征求如何让模型将这些碎片识别为一个物体的建议，例如在不同 segments 之间使用标签（label tags）。


  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1356852879633154182)** (2 条消息): 

> `Gradio 里程碑，百万月活跃开发者` 


- **Gradio 达到一百万月活跃开发者！**：Gradio 宣布其达成了一个里程碑，即每月有 **1,000,000 名活跃开发者**使用该平台创建和分享 AI 界面。
   - Gradio 团队对社区在实现这一重大里程碑过程中所做的宝贵贡献表示感谢。
- **社区庆祝 Gradio 的成功**：Gradio 社区成员庆祝该平台达到百万月活跃开发者的成就。
   - 社区成员认可了 Gradio 在赋能 ML 研究人员和公司构建生产级 AI 界面方面的影响，强调了该平台在 AI 领域的增长和重要性。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1356391396420092115)** (16 条消息🔥): 

> `结合 Ollama 使用 OpenAIServerModel，Langraph OpenAI API 模型替代方案，第 3 单元发布` 


- **Ollama 与 OpenAIServerModel 配合良好**：鉴于其与 **OpenAI API** 的兼容性，成员们讨论了如何将 **OpenAIServerModel** 与 **Ollama** 结合使用。
- **寻求 Langraph OpenAI API 模型的替代方案**：一位成员请求为邮件 Agent 推荐 **Langraph OpenAI API 模型**的替代方案。
- **第 3 单元延迟，社区“向虚空呐喊”**：许多成员正急切等待 Agent 课程 **Unit 3** 的发布，尽管一位成员指出其发布已推迟。
   - 一位成员开玩笑说：*也许如果我们一直向虚空呐喊，虚空会给予回应*。


  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1356484289684045825)** (1 条消息): 

> `Liger Kernel，GPU 显存占用，速度与显存的权衡` 


- **Liger Kernel：速度提升 vs 显存占用**：一位用户发现应用 **Liger kernel** 显著提高了速度，但导致了较高的 **GPU 显存占用**。
   - 他们质疑自己的应用方法是否存在缺陷，并寻求在不牺牲性能的情况下优化显存使用的建议。
- **分析 Liger Kernel 的显存占用**：用户的经验凸显了在使用 **Liger kernel** 时，**计算速度**与**显存消耗**之间可能存在的权衡。
   - 需要进一步调查以了解该 kernel 的显存管理并确定可能的优化策略。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1356659846140788979)** (9 条消息🔥): 

> `MAX 25.2 直播，Chris 闪电演讲，GTC Chris 视频` 


- **MAX 25.2 直播开启！**：Modular 宣布了 **MAX 25.2 直播**，邀请观众通过 [LinkedIn](https://www.linkedin.com/events/introducingmax25-27310405561261477892/theater/) 或 [YouTube](https://www.youtube.com/watch?v=-mN2SDlBDrA) 加入，并向团队现场提问。
   - 由于技术困难，分享了一个新的直播链接 ([YouTube](https://www.youtube.com/watch?v=iCIFPj9uMtI))，“Introducing MAX 25.2 Live!”。
- **为直播期间的技术故障致歉！**：成员们为 **MAX 25.2 直播**期间的技术问题表示歉意，并保证下次活动将使用更好的系统。
   - 一位成员幽默地讲述了自己误以为 **Chris 在 GTC 的视频**是直播的一部分。
- **已清理的直播回顾和 Chris 的演讲现已上线！**：为错过直播的用户发布了 **MAX 25.2 直播**的清理版录像 ([YouTube](https://www.youtube.com/watch?v=dG0L1GalIHU))。
   - **Chris** 在 Modular 展位的**闪电演讲**完整录像也可以在 [YouTube](https://www.youtube.com/watch?v=d89mHSeN_QE) 上观看。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=d89mHSeN_QE"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=iCIFPj9uMtI">Introducing MAX 25.2 Live!</a>: 加入我们 4 月 1 日的直播，我们将深入探讨 MAX 25.2 的方方面面！💥 从构建它的团队那里获取最新版本的完整信息，并成为...</li><li><a href="https://www.youtube.com/watch?v=dG0L1GalIHU"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1356635583065231541)** (59 条消息🔥🔥): 

> `Mojo 中的 Compiler Bug、Enums、Flex Attention、Float to String 算法、FlashAttention-2` 


- **令人困惑的编译器错误消息曝光**：一位用户在为 `Dataset` 结构体定义方法时报告了一条令人困惑的错误消息，怀疑是 Compiler Bug，并提供了一个 [GitHub issue 链接](https://github.com/modular/max/issues/4248)。
   - 另一位用户建议该问题可能是由于使用了 `out self` 而非 `mut self` 导致的，同时也承认编译器错误消息确实具有误导性。
- **Enum 更新仍未发布**：一位用户询问了 Mojo 中 **enums** 的更新进展，但遗憾的是，目前没有任何更新。
   - 回复很简单：*“遗憾的是，还没有。🙃🙃🙃”*
- **在 MAX 中实现 FlexAttention**：一位用户询问了在 Mojo 中实现 **flex-attention** 的相关事宜及其难度，并链接了一篇关于 [flex-attention 的 PyTorch 博客文章](https://pytorch.org/blog/flexattention/)。
   - 有人建议将其作为 MAX 中的 Custom Op 实现是可行的，并且 GPU 上的 Mojo 与 CUDA 非常接近，允许控制内存移动，因此 *“除非你遇到了正在开发中的功能，否则 MAX 基本上可以实现你想要的任何功能。”*
- **Float-to-String 算法移植效果不佳**：一位用户参考作者在 CPPCon 上的演讲，将一种新的 **float to string 算法** 移植到了 Mojo，但发现它比标准库的 dragonbox 实现更慢，并分享了[相关代码链接](https://github.com/bgreni/EmberJson/blob/main/emberjson/teju/__init__.mojo#L103)。
   - 该用户指出，尽管从标准库中借鉴了格式化代码，但序列化 `canada.json` 的耗时从 30ms 中位增加到了 40ms 低位。
- **FlashAttention-2 Recipe 公开**：一位用户分享了一个包含 Mojo 版 **FlashAttention-2** 的 Recipe 链接，并强调其编写目的是为了可读性，而非极致的性能优化，详见 [custom-ops-ai-applications](https://builds.modular.com/recipes/custom-ops-ai-applications)。
   - 另一个链接指向了一个展示如何利用 Mojo 的内存布局抽象逐步优化矩阵乘法的 Recipe，详见 [custom-ops-matrix-multiplication](https://builds.modular.com/recipes/custom-ops-matrix-multiplication)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/soraros/44d56698cb20a6c5db3160f13ca81675">ir_utils.mojo</a>: GitHub Gist: 立即分享代码、笔记和代码片段。</li><li><a href="https://github.co">GitHub · 在统一的协作平台上构建和交付软件</a>: 加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://github.com/cassioneri/teju_jagua">GitHub - cassioneri/teju_jagua: Teju Jagua</a>: Teju Jagua。通过创建账户为 cassioneri/teju_jagua 的开发做出贡献。</li><li><a href="https://github.com/bgreni/EmberJson/blob/main/emberjson/teju/__init__.mojo#L103">EmberJson/emberjson/teju/__init__.mojo at main · bgreni/EmberJson</a>: 一个用纯 Mojo 编写的用户友好型 JSON 库。通过创建账户为 bgreni/EmberJson 的开发做出贡献。</li><li><a href="https://builds.modular.com/recipes/custom-ops-ai-applications">Custom Operations: Applications in AI Models Recipe | MAX Builds</a>: 未找到描述</li><li><a href="https://builds.modular.com/recipes/custom-ops-matrix-multiplication">Custom Operations: Optimizing Matrix Multiplication Recipe | MAX Builds</a>: 未找到描述</li><li><a href="https://github.com/cassioneri/teju_jagua/blob/main/teju/mshift.h#L99-L214">teju_jagua/teju/mshift.h at main · cassioneri/teju_jagua</a>: Teju Jagua。通过创建账户为 cassioneri/teju_jagua 的开发做出贡献。</li><li><a href="https://github.com/modular/max/issues/4248">[BUG] 错误的方法定义导致令人困惑的错误消息 · Issue #4248 · modular/max</a>: Bug 描述 实际行为 我在为 Dataset 结构体定义方法时犯了一个错误。参数只有类型而没有 owned 类型。错误消息告诉我 self 的类型错误...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1356349251981938978)** (40 条消息🔥): 

> `OpenAI API, Midjourney New Research, Sam Altman open-weight language model, Psyche p2p, Anthropic Insights on LLMs` 


- **一行代码修复即可让 OpenAI API 教程生效**：任何适用于 **OpenAI API** 的教程都可以在 Nous Research AI API 上运行，前提是将代码中的 endpoint 修改为 `endpoint = "api.nousresearch.com"`。
   - 一位用户确认他们在进行该更改后已成功运行，并将添加相关样式。
- **Midjourney 的 LLMs 写作更具创意**：[Midjourney](https://www.midjourney.com/home) 与纽约大学的机器学习专家共同发布了一篇新研究论文，旨在训练基于文本的大语言模型（LLMs）进行更具创意的写作，将其业务范围从图像生成扩展开来。
   - 该公司还在构建自己的计算和 **AI hardware**，已于 2024 年夏末宣布。
- **Sam Altman 预告新款权重开放模型**：Sam Altman 宣布计划在未来几个月内发布一款具有推理能力的全新 **open-weight language model**，并寻求开发者反馈以最大化其效用。
   - 计划在旧金山、欧洲和亚太地区举办开发者活动，以收集反馈并测试早期原型，这标志着 OpenAI 自 GPT-2 以来首次发布权重开放模型（[公告链接](https://openai.com/open-model-feedback)）。
- **追踪语言模型中的思维：Anthropic 的见解**：Anthropic 发布了研究报告（[Tracing Thoughts in Language Models](https://www.anthropic.com/news/tracing-thoughts-language-model)），指出 **LLMs** 拥有自己的思维语言，并且比之前认为的更具预判性。
   - LLMs 的运作方式比单纯处理单个 tokens 更加复杂。
- **DeepSeek 的“柔术”策略让开源 OpenAI 模型成为可能**：频道成员对“DeepSeek 运用复杂的 **Jiu Jitsu maneuvers**（柔术策略）使开源社区实现这一目标”表示感谢。
   - 这种情绪得到了共鸣，并附带了一个 [YouTube 视频](https://www.youtube.com/watch?v=PEF7tyVUl70) 链接，讨论了 OpenAI 与权重开放模型相关的战略转变。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/">Midjourney 的惊喜：关于让 LLMs 写作更具创意的新研究</a>：经典的基于 Transformer、专注于文本的 LLMs 在认知和性能方面仍有很大潜力可挖。</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898?s=46">Sam Altman (@sama) 的推文</a>：摘要：我们很高兴能在未来几个月内发布一款强大的、具有推理能力的全新权重开放语言模型，我们希望与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://www.youtube.com/watch?v=PEF7tyVUl70">OpenAI 的新模型揭示了其战略转型</a>：CNBC 的 Deirdre Bosa 加入“The Exchange”节目，讨论 OpenAI 计划在未来几个月内发布权重开放模型的计划。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1356766660790587543)** (8 条消息🔥): 

> `DeepHermes Reasoning, Structured Output with Langchain, DeepHermes AI, Tool Calling with Reasoning` 


- **DeepHermes 推理可靠性调查**：据一位成员称，目前在 **DeepHermes** 的推理模式下避免使用 **JSON** 或 **tool calling** 会更可靠，建议选择 **non-reasoning mode**。
   - 下一个版本的 **DeepHermes** 预计将改进推理和 tool calling；然而，对于当前使用，将 **reasoning system prompt**、换行符和 **tool calling system prompt** 组合使用可能会产生可以接受的结果。
- **发现 DeepHermes AI**：一位成员兴奋地注意到 **DeepHermes AI** 的存在，发现它是一个 **3B model**。
   - 该成员还观察到，**DeepHermes** 中的推理似乎是以带有 `<think> </think>` 标签的思维链（chain of thoughts）形式实现的。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1356704524022317306)** (2 messages): 

> `Project Loong 发布，合成数据生成` 


- **CamelAIOrg 发布 Project Loong 🐉**：[CamelAIOrg](https://x.com/CamelAIOrg) 推出了 **Project Loong 🐉**，这是一个用于生成和验证**合成数据（synthetic data）**的结构化、模块化解决方案。
   - 该项目发布了一篇 [博客文章](https://camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers)，详细介绍了其模块化设计，该设计将**合成数据生成**与语义验证以及确保准确性和一致性的多 Agent 框架相结合。
- **采样预设影响指标分布**：一名成员对*优美指标（beautiful metric）*的分布如何随**采样预设（sampling presets）**而变化表示好奇。



**提及链接**：<a href="https://x.com/CamelAIOrg/status/1907155361422872762">来自 CAMEL-AI.org (@CamelAIOrg) 的推文</a>：介绍 Project Loong 🐉 博客：https://camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers…• 我们用于生成和验证合成数据的结构化方法，旨在增强...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1356443885345177770)** (10 messages🔥): 

> `Nous Research Portal Git 仓库, X 链接移除, 贡献 Nous Research, Google Style Guide` 


- **Nous Research Portal 的 Git 仓库依然难寻**：一名成员询问了 [Nous Research portal](https://portal.nousresearch.com/) 的 Git 仓库，但得到的澄清是*不需要 Git 仓库*，因为它使用的是 **OpenAI 库**。
   - 该门户的环境详情显示 **VercelEnv** 和 **NodeEnv** 处于 **production** 状态，而 commit、branch、last commit 和修改状态均为*未知*。
- **由于安全顾虑，X 链接消失**：一名成员询问某个特定 X 链接的位置，结果被告知*该链接已被删除*，原因是担心**它可能会窃取用户密钥**。
   - 未提供关于该链接性质或其构成的具体安全风险的进一步细节。
- **开发应用是贡献 Nous Research 的途径**：针对如何贡献 Nous Research 的咨询，有人建议用户可以通过**使用这些模型开发应用**来做出贡献。
   - 进一步澄清指出，用户应专注于使用 **API** 构建服务，而不是修改服务本身。
- **Google Style Guide 为代码生成提供强效助力**：一名成员分享了 [Google Style Guide](https://google.github.io/styleguide/) 的链接，将其描述为*代码生成的强效催化剂*。
   - 该风格指南 ([google/styleguide](https://github.com/google/styleguide)) 包含了 **AngularJS**、**Common Lisp**、**C++** 和 **C#** 的指南。


<div class="linksMentioned">

<strong>提及链接</strong>：

<ul>
<li>
<a href="https://google.github.io/styleguide/">Google Style Guides</a>：源自 Google 的开源项目的风格指南</li><li><a href="https://portal.nousresearch.com/">Nous Portal</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1356704524022317306)** (2 messages): 

> `Project Loong, 合成数据生成, 模型性能增强` 


- **Camel AI 发布 Project Loong**：Camel AI 推出了 [Project Loong 🐉](https://x.com/CamelAIOrg/status/1907155361422872762)，这是一个用于生成和验证合成数据的模块化解决方案，并请求转发和分享该公告。
   - Project Loong 采用**结构化方法**，将合成数据生成与**语义验证**相结合。
- **Project Loong 提升模型性能**：Project Loong 旨在通过确保**准确性和一致性**的 [多 Agent 框架](https://camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers) 来增强模型性能。
   - 该项目专注于通过合成数据生成的**可靠推理信号**来增强特定领域模型的能力。



**提及链接**：<a href="https://x.com/CamelAIOrg/status/1907155361422872762">来自 CAMEL-AI.org (@CamelAIOrg) 的推文</a>：介绍 Project Loong 🐉 博客：https://camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers…• 我们用于生成和验证合成数据的结构化方法，旨在增强...

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1356411167194480690)** (35 条消息🔥): 

> `Graph Learning 演进, AI/ML 对就业的影响, RLHF 对齐与被削弱的模型, Gemini 2.5 Pro 数学能力, 梦境日志应用` 


- **Graph 演进超越 2018，引发 Graph Learning 复兴**：一位成员分享了 [Google Research 博客文章](https://research.google/blog/the-evolution-of-graph-learning/)，探讨了 **graph learning** 的演进，并对 **2019** 年以来的最新进展表示关注。
   - 该博客文章将图论追溯到 **1736** 年的 **Leonhard Euler**，并讨论了其在建模关系和连接方面的应用。
- **AI/ML 改变就业市场：低级岗位受到威胁**：一位成员建议，最近的 **AI/ML** 进展主要影响低级工作（如次要的编程任务），但强调了人类的适应能力。
   - 他们指出 **AI/ML** 减少了对他人的依赖，例如使用 **AI/ML** 进行初步的法律援助，这节省了资源并使多学科任务成为可能。
- **RLHF 对齐：被抑制的行为在 AI 模型中重新浮现**：讨论围绕 **RLHF** 以及如果模型在执行有用任务（如 **ML R&D 或数据收集**）时受到惩罚，可能出现 *涌现失对齐 (emergent misalignment)* 的风险。
   - 成员们担心，如果开源模型被 *削弱 (nerfed)*，第一批自我改进的模型可能会因为补偿被抑制的行为而变得越来越 *邪恶*。
- **Gemini 2.5 Pro 数学挂科，UI 表现糟糕**：一位成员测试了 **Gemini 2.5 Pro (experimental)** 的数学能力，发现它 *完全是垃圾*，该成员还补充说 Google 的 UI 无法正确显示数学公式。
   - 当被问及 **信息论和几何学** 时，**ChatGPT** 和 **Grok 3** 在理解问题方面表现更好，即使问题写得很糟糕，用户随后引导它正确书写。
- **梦境日志应用旨在分析清醒梦**：一位成员宣布创建了 [Rem](https://lucidrem.com)，这是一个梦境日志应用，旨在轻松记录、分析和分享梦境。
   - 未提供二次摘要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.19470">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a>: LLM 在推理方面表现出了卓越的能力，OpenAI-o1 和 DeepSeek-R1 的成功就是例证。然而，将推理与外部搜索过程集成仍然...</li><li><a href="https://research.google/blog/the-evolution-of-graph-learning/">The evolution of graph learning</a>: 未找到描述</li><li><a href="https://lucidrem.com">Rem</a>: 在一个美观、安全的空间里记录你的梦境，发现隐藏的模式，并与梦境探索者社区建立联系。
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1356394660075016273)** (5 条消息): 

> `RLHF, Reward Hacking, 响应多样性, 推理任务验证器, 生成式奖励模型` 


- **通过 RLHF 对齐 LLM**：分享了一篇关于 **Reinforcement Learning from Human Feedback (RLHF)** 的论文，指出其对于将大语言模型与人类偏好对齐的重要性，链接见 [arxiv.org/abs/2503.22230](https://arxiv.org/abs/2503.22230)。
- **忽视 RLHF 中的 Prompt 数据构建**：论文探讨了被忽视的 **prompt-data construction** 的重要性，并研究了 RLHF 性能扩展中的 **data-driven bottlenecks**，特别是 **reward hacking** 和 **response diversity** 的下降。
- **混合奖励系统缓解 Reward Hacking**：论文引入了一种结合了 **Reasoning Task Verifiers (RTV)** 和 **Generative Reward Model (GenRM)** 的 **hybrid reward system**，以缓解 reward hacking。
- **Prompt 选择方法增强学习效果**：提出了一种新颖的 **prompt-selection method**，即 **Pre-PPO**，以保持 **response diversity** 并增强 **learning effectiveness**。
- **早期优先处理任务可提升性能**：论文发现，在 RLHF 训练早期优先处理 **数学和代码任务** 可显著提升性能，通过两种模型尺寸的实验验证了该方法的有效性和可扩展性。



**提到的链接**: <a href="https://arxiv.org/abs/2503.22230">Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback</a>: RLHF 对于将大语言模型与人类偏好对齐至关重要。虽然最近的研究集中在算法改进上，但...的重要性...

  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1356344708653252839)** (20 条消息🔥): 

> `AI Dog Chasing Tail, AI Model Feedback, Runway Relevance, OpenAI Model Release Speculation, GPT-3.5 vs Thinking Models` 


- **AI 被指责在“追逐自己的尾巴”**：一位成员对当前的 AI 能力表示怀疑，将其描述为“带有一些黑客手段的概率性文本补全”，并质疑它们是否构成了真正的“思考”。
   - 他们对围绕 AI 的炒作表示幻灭，认为自 **GPT-3.5** 以来的改进更多是关于微调（fine-tuning）而非重大突破，并希望实现“端到端且无需人工干预（no hand holding）”。
- **OpenAI 模型反馈论坛上线**：一位成员链接到了 [OpenAI Open Model Feedback](https://openai.com/open-model-feedback/) 论坛。
   - 另一位成员引用了 **Ilya Sutskever** 的话，指出“如果说有一个巨大的失败，那就是你总是必须检查结果”。
- **Runway 的相关性引发辩论**：一位成员质疑是否还有人关心 **Runway** 在做什么。
   - 另一位成员链接到了一条[展示 AI 生成的肯德基广告概念的推文](https://x.com/Salmaaboukarr/status/1906776503343325469)，该广告使用了 **Runway, Pika, Kling AI, Google DeepMind Veo2, Luma AI, OpenAI Sora 和 Topaz Labs** 制作。
- **OpenAI 发布推测浮出水面**：成员们推测 **OpenAI** 的下一个模型发布，猜测可能是一个用于移动端的更小模型，特别是考虑到他们与 **Apple 的交易告吹**。
   - 一些人开玩笑并思考他们是否会发布 **GPT 2.5，参数量为 100M**。
- **Deepseek R1 就像大学生**：一位成员表示 **GPT 3.5** 与任何思考模型（thinking models）之间存在巨大差距，将 **GPT 3.5** 比作“10 岁小孩”，而将 **Deepseek R1** 描述为“像大学生”。



**提及的链接**：<a href="https://x.com/Salmaaboukarr/status/1906776503343325469">来自 Salma (@Salmaaboukarr) 的推文</a>：我被震撼到了！😱 这个肯德基概念广告 100% 是 AI 生成的！我的朋友 David Blagojevic（他不在 X 上）为肯德基创作了这个广告概念，太不可思议了！使用的工具：Runway, Pika, Kling...

  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1356344746406056087)** (38 条消息🔥): 

> `MCP RBAC 实现, Docker 替代方案, 用于 webapp 的 MCP server, VirusTotal 集成, 用于 make.com 或 n8n cloud 的 MCP` 


- **MCP 随 Pichai 的推文获得关注**：继 **Sundar Pichai** 发布了一条询问 *'To MCP or not to MCP, that's the question'* 的 [推文](https://x.com/sundarpichai/status/1906484930957193255) 后，对 **MCP** 的兴趣激增，该推文获得了超过一百万次的浏览。
   - /r/mcp 的一位 Reddit 版主甚至建议，如果 Google 打算深入投入 **MCP**，可以进行一场 **AMA**。
- **在 MCP Server 上构建 RBAC**：用户正在探索在 **MCP servers** 上实现基于角色的访问控制 (**RBAC**)，以便根据用户角色划分工具的可见性。
   - 一位用户建议与 **WorkOS** 集成，另一位用户提到 **Toolhouse API** 根据 API key 执行 **RBAC**。
- **SDK 治理层已开源！**：一名成员分享了一个开源 **SDK**，旨在 **Model Context Protocol** 框架内实现企业级治理（**身份识别、RBAC、凭据、审计、日志、追踪**），项目地址为 [ithena-one/mcp-governance-sdk](https://github.com/ithena-one/mcp-governance-sdk)。
   - 鼓励并非常欢迎社区提供反馈。
- **DesktopCommanderMCP 为你编写代码**：一位用户推荐使用 **DesktopCommanderMCP** 为 **Claude** 创建和更新文件，它通过 [wonderwhy-er/DesktopCommanderMCP](https://github.com/wonderwhy-er/DesktopCommanderMCP) 提供终端控制、文件系统搜索和文件编辑功能。
   - 他们建议让 LLM 选择正确的 server 并仅获取这些 server 的上下文，而不是用 30 个 **MCP** 淹没上下文。
- **MCP 正在考虑 Nova act**：一位成员建议，让 **Claude** 输出 **act** 调用（来自 **Amazon** 的 **Nova**）并将其提供给连接到执行实际浏览任务（即某些 **Nova** 端点）的 **MCP server** 并不困难，详见[此视频](https://www.youtube.com/watch?v=JLLapxWmalU)。
   - 这种方法涉及 **Claude** 根据用户请求生成 `nova.act` 命令，然后由 **MCP server** 执行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sundarpichai/status/1906484930957193255">来自 Sundar Pichai (@sundarpichai) 的推文</a>: To MCP or not to MCP, that's the question. Lmk in comments</li><li><a href="https://x.com/punkpeye/status/1906766534673875377">来自 Frank Fiegel (@punkpeye) 的推文</a>: @sundarpichai Hey @sundarpichai I am moderator of /r/mcp on RedditIf Google is leaning into MCP, let's do an AMA.</li><li><a href="https://github.com/nerding-io/n8n-nodes-mcp">GitHub - nerding-io/n8n-nodes-mcp: 用于 MCP 的 n8n 自定义节点</a>: n8n custom node for MCP. Contribute to nerding-io/n8n-nodes-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/matthewhand/mcp-openapi-proxy">GitHub - matthewhand/mcp-openapi-proxy</a>: Contribute to matthewhand/mcp-openapi-proxy development by creating an account on GitHub.</li><li><a href="https://github.com/wonderwhy-er/DesktopCommanderMCP">GitHub - wonderwhy-er/DesktopCommanderMCP: 这是一个为 Claude 提供的 MCP server，赋予其终端控制、文件系统搜索和 diff 文件编辑功能</a>: This is MCP server for Claude that gives it terminal control, file system search and diff file editing capabilities - wonderwhy-er/DesktopCommanderMCP</li><li><a href="https://github.com/ithena-one/mcp-governance-sdk">GitHub - ithena-one/mcp-governance-sdk: 为 Model Context Protocol SDK 提供的企业治理层（身份识别、RBAC、凭据、审计、日志、追踪）</a>: Enterprise Governance Layer (Identity, RBAC, Credentials, Auditing, Logging, Tracing) for the Model Context Protocol SDK - ithena-one/mcp-governance-sdk
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1356342792430817525)** (13 条消息🔥): 

> `ActivePieces 停止支持 MCP, MCP 自动化测试工具, MCP 每周简报, 结合 Smithery 的 Playwrite MCP 服务端, MCP 同步限制` 


- **ActivePieces 停止 MCP 支持**: [Active pieces](https://www.activepieces.com/mcp) 作为 **Zapier** 的开源替代方案，已停止对 **MCP** 的支持。
- **MCP 服务端自动化测试工具发布**: [mcp-autotest](https://github.com/strowk/mcp-autotest) 是一个通过 yaml 文件定义预期服务端行为，并检测其是否合规的工具。
   - **0.2.1** 版本支持使用 stdio 或新的流式 http 传输进行测试。
- **MCP Bits 简报上线！**: 一份名为 [MCP Bits](https://mcpbits.substack.com/p/mcp-bits-1) 的全新 MCP 每周简报已发布。
   - 包含最新的新闻、文章、视频和项目更新；在此[订阅简报](https://mcpbits.substack.com/)。
- **Playwrite MCP 服务端现支持 Smithery 托管**: Playwrite MCP 服务端现在可以通过 Smithery 托管运行，使 **Sage** 能够在 **iOS** 上抓取网页内容。
- **MCPC 实现双向异步通信**: 一个名为 [MCPC](https://github.com/OlaHulleberg/mcpc) 的扩展已创建，旨在缓解 MCP 的同步限制并增加异步支持。
   - 该新扩展提供向后兼容性，因此不会造成破坏——除非客户端和服务端都支持 MCPC，否则你只是无法使用这些额外功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mcp.direct/">mcp.direct | 自定义 MCP 服务端</a>: 用于优化 LLM 注意力的自定义 MCP 服务端。联系我们获取定制方案。</li><li><a href="https://mcpbits.substack.com/p/mcp-bits-1">MCP Bits #1</a>: 每周 Model Context Protocol 新闻</li><li><a href="https://github.com/GeLi2001/shopify-mcp">GitHub - GeLi2001/shopify-mcp: Shopify API 的 MCP 服务端，可用于 Anthropic 的 Claude 和 Cursor IDE 等 MCP 客户端</a>: Shopify API 的 MCP 服务端，可用于 Anthropic 的 Claude 和 Cursor IDE 等 MCP 客户端 - GeLi2001/shopify-mcp</li><li><a href="https://github.com/strowk/mcp-autotest">GitHub - strowk/mcp-autotest: MCP 服务端自动化测试工具</a>: MCP 服务端自动化测试工具。通过在 GitHub 上创建账号为 strowk/mcp-autotest 的开发做出贡献。</li><li><a href="https://github.com/OlaHulleberg/mcpc">GitHub - OlaHulleberg/mcpc: MCP (Model-Context-Protocol) 的扩展，通过现有的 MCP 传输实现 LLM 与工具之间的双向异步通信 - 无需额外的传输层。</a>: MCP (Model-Context-Protocol) 的扩展，通过现有的 MCP 传输实现 LLM 与工具之间的双向异步通信 - 无需额外的传输层...</li><li><a href="https://github.com/ithena-one/mcp-governance-sdk">GitHub - ithena-one/mcp-governance-sdk: Model Context Protocol SDK 的企业级治理层（身份、RBAC、凭据、审计、日志、追踪）</a>: Model Context Protocol SDK 的企业级治理层（身份、RBAC、凭据、审计、日志、追踪） - ithena-one/mcp-governance-sdk</li><li><a href="https://www.activepieces.com/mcp">280+ 开源 MCP — 立即在 Activepieces 上使用</a>: 通过 280+ 开源 MCP 让 AI 访问你的应用。配合 Claude、Cursor 或 Windsurf 使用，让 AI 阅读你的邮件、管理你的日历等。
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1356667231202971822)** (1 条消息): 

> `Webby Awards, 投票, NotebookLM 提名` 


- **NotebookLM 斩获三项 Webby 提名！**: NotebookLM 已获得 **三项 Webby Awards** 提名，并请求社区通过此[链接](https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement)进行投票。
   - 投票者应*通过点击邮件中的验证链接来确认投票*，并*检查垃圾邮件箱*。
- **搜索无结果**: 搜索功能显示 `Displaying top {{maxSearchResults}} results.` 和 `**No results**`。
   - 提示用户*优化搜索条件以缩小结果范围*。



**提到的链接**: <a href="https://vote.webbyawards.com/PublicVoting#/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement">为互联网之最投票</a>: 我刚刚在 Webby People's Voice Awards 中投票并检查了我的投票注册情况。

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1356503195446153257)** (9 messages🔥): 

> `Google Tasks integration with NotebookLM, Archiving notebooks in NotebookLM, Sharing sources on different notes in NotebookLM` 


- **Google Tasks 可能与 NotebookLM 集成**：一位用户建议 **Google Tasks** 可以通过允许用户经由下拉菜单/弹出窗口选择任务列表来与 **NotebookLM** 集成。
   - 他们提议这可以类似于 **Google Tasks** 允许选择任务列表进行共享的工作方式。
- **笔记本归档功能可减少笔记本计数**：一位用户请求在 **NotebookLM** 中提供一种**归档笔记本**的方法，以隐藏它们并减少计入其额度限制的笔记本数量。
   - 他们建议隐藏/归档的笔记本不应出现在可用于共享内容的笔记本列表中。
- **笔记间的来源共享：是一项可用功能吗？**：一位用户询问是否可以在 **NotebookLM** 内的**不同笔记中共享来源**。
   - 他们不确定该功能目前是否可用。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1356355385526845693)** (39 messages🔥): 

> `Timestamped sections on the todo list, NotebookLM to Gemini 2.5 Pro, Conversation ending early, Limit the total number of words, not the number of sources?, Maths notation in NLM is very hard to read` 


- **带时间戳的待办事项列表大获好评**：一位用户请求在待办事项列表中添加带时间戳的章节，类似于 Audible，以便跳转和重听特定章节。
   - 该建议旨在**提升用户体验**以及长音频内容的可访问性。
- **对 Gemini 2.5 Pro 的期待**：一位用户请求将 NotebookLM 的 AI 更新为 **Gemini 2.5 Pro**，并表示他们非常喜欢更新后的 Gemini 版本。
   - 他们希望 NotebookLM 在新模型下表现更好，但 NotebookLM 团队尚未对任何 ETA 发表评论。
- **对话中断灾难**：一位用户报告对话过早结束，未能覆盖上传的第二个资源，并询问是否有修复方法。
   - 团队请求[在专门的 Discord 频道中](https://discord.com/channels/1124402182171672732/1351160368352727093)记录该问题，如果可能的话请附上示例笔记本 ID。
- **需要的是笔记而非来源**：一位使用 **Obsidian** 管理个人笔记（2000+ 短笔记）的用户发现 300 个来源的限制过于严格。
   - 他们建议限制总字数而不是来源数量，以更好地适应网状笔记系统；一位用户建议将**文件夹或压缩包**作为单一来源也可以解决这个问题。
- **数学符号困扰**：一位用户报告 NLM 在普通聊天中的数学符号非常难以阅读，询问是否有修复方法。
   - 团队确认了该问题并正在调查，但目前尚无更改的 **ETA**。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1356379142467227678)** (11 messages🔥): 

> `Torchtune office hours, Discord timezone handling` 


- **下周五的 Torchtune 时间**：成员们宣布了下周五的下一次 **Torchtune office hours**，并附上了 [Discord event](https://discord.gg/Z9cuQgYX?event=1356379057373184155) 链接。
- **Discord 时区自动转换“大聪明”行为**：成员们在意识到 **Discord** 会自动处理时区转换之前，一直在手动转换时区。
   - 随后一位成员发布了一个 [Big Brain meme](https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-24411104)。



**提到的链接**：<a href="https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-2441110471562975014">Brain Brain Meme GIF - Brain Brain meme Big brain - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1356364286045454377)** (16 messages🔥): 

> `PR #2441 评审，PR #2477 的回归测试，Qwen 模型上传，S3 Bucket 连接问题，PR #2510` 


- **PR #2441 需要尽快进行最终评审**：一名成员请求对 [PR #2441](https://github.com/pytorch/torchtune/pull/2441) 进行最终评审，以加快合并进程。
- **由于 S3 问题，回归测试暂停**：希望对 [PR #2477](https://github.com/pytorch/torchtune/pull/2477) 进行回归测试，但目前因等待将 **Qwen 模型**上传到 S3（作为回归测试脚本下载的一部分）而受阻。
   - 然而，另一名成员意识到由于内部基础设施的变更，连接他们的 **S3 bucket** 还需要更多工作，并建议将回归测试暂时搁置。
- **像 Llama2 这样的现代模型在竞争中胜出**：一名成员建议在测试中使用比 **Llama2** 更现代的模型，但目前的回归测试仍在使用 **Llama2 模型**。
- **PR #2510 删除了递归重分片工具**：[PR #2510](https://github.com/pytorch/torchtune/pull/2510) 删除了 recursive_reshard 工具，因为它不再需要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/f1ecdd64cd67fc33a713c073d9664ab111116606/tests/cache_artifacts.sh#L25">torchtune/tests/cache_artifacts.sh at f1ecdd64cd67fc33a713c073d9664ab111116606 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2510">REMOVE `recursive_reshard` UTILITY by ebsmothers · Pull Request #2510 · pytorch/torchtune</a>：起初，这个 PR 应该是为了修复 #2483。经过对该工具的进一步检查，发现它是不需要的。既然不需要，为什么要保留它呢？你如何知道...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1356419747453079754)** (15 messages🔥): 

> `ImageDtype 和 IMAGE 环境变量，tinygrad BEAM 性能，移动端 GPU 和 ImageDType，arange() 优化` 


- **深入研究 ImageDtype 和 IMAGE 环境变量**：一名成员询问了 tinygrad 中 **ImageDtype** 和 **IMAGE 环境变量**的作用，注意到它对 **Tensor.conv2d** 实现的影响，并链接了一个 [VAE 训练脚本](https://github.com/currybab/tinygrad-generative-ai-tutorial/blob/main/vae/vae_train.py)。
   - 另一名成员建议这与在 **Qualcomm (QCOM)** 硬件上更快地运行 **comma.ai** 模型有关，利用了移动端 GPU 的纹理性能和缓存能力。
- **tinygrad BEAM 性能大幅超越 tf-metal**：一位用户报告在 M1 Pro 上，不使用 BEAM 时达到 **3.2 it/s**，使用 **BEAM=2** 时达到 **28.36 it/s**，而使用 Keras 配合 tf-metal 约为 **25 it/s**。
   - George Hotz 回应道：*“很高兴看到开启 BEAM 后比 tf-metal 更快！”*
- **移动端 GPU 通过 ImageDType 获得纹理加速**：讨论表明 **ImageDType** 及相关函数可能会针对移动端 GPU 的纹理性能进行优化，并引用了一篇关于移动端 GPU 的 [Microsoft 研究论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2022/02/mobigpu_mobicom22_camera.pdf)。
   - 一名成员质疑硬编码布局细节的必要性，并建议 **HWC (Height, Width, Channel)** 处理应该是带有用户定义填充的普通 **conv2d** 的一部分。
- **arange() 获得优化**：一名成员发现小范围的 **arange**（例如 `arange(1, 2, 0.1)`）生成的代码比大范围（例如 `arange(1, 10, 0.1)`）生成的代码差，随后在[此处](https://xl0.github.io/tinygrad-notes/arange.html)添加了关于 `.arange()` 的章节。
   - 他们还发现了生成的代码中存在不必要的加法，建议将 `((float)((ridx0+1)))*0.1f)+0.9f)` 修复为 `(((float)((ridx0)))*0.1f)+1.0f)`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://xl0.github.io/tinygrad-notes/arange.html">4 - .arange() 的疯狂 – TinyGrad 笔记</a>：关于 TinyGrad 内部机制的笔记</li><li><a href="https://github.com/currybab/tinygrad-generative-ai-tutorial/blob/main/vae/vae_train.py">tinygrad-generative-ai-tutorial/vae/vae_train.py at main · currybab/tinygrad-generative-ai-tutorial</a>：tinygrad 生成式 AI 教程。通过在 GitHub 上创建账号为 currybab/tinygrad-generative-ai-tutorial 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1356645866425028729)** (1 messages): 

> `LLM Agents 用于技术文档，从复杂文档中进行结构化提取` 


- **LLM Agents 应对技术文档**：**LLM agents** 一个被低估的用例是每一个严重依赖复杂技术文档的领域，如制造业、建筑业和能源行业。
   - 有建议指出，你可以构建一个能够从这些文档中进行**结构化提取**的 Agent。
- **利用 LlamaIndex 解码复杂文档**：一条推文显示，这些文档通常充满了[截图](https://t.co/SQWHMFYaoU)。
   - 相关的推文可以在[这里](https://twitter.com/llama_index/status/1907086884670673305)找到。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1356443998041935902)** (6 messages): 

> `ReAct Agents, 通过 Ollama 使用本地模型, OpenAI 速率限制错误, Embedding 模型, Query Engines` 


- **ReAct Agent 遇到 OpenAI 速率限制**：一位用户在使用通过 **Ollama** 设置的本地模型的 ReAct Agent 时，遇到了 **OpenAI RateLimitError (Error 429)**，并质疑 ReAct Agent 是否仅限 OpenAI LLMs 使用。
   - 他们提供了一个指向其 [GitHub 仓库](https://github.com/JakeFurtaw/Agentic-Chat-RAG/blob/jake-dev/agent_utils.py)的链接，展示了他们的 Agent 设置。
- **排除 OpenAI 错误**：一名成员建议 **embedding model** 可能是导致 OpenAI 错误的原因，因为如果没有显式设置，它可能会默认使用 OpenAI 的 Embedding 模型。
   - 用户确认他们使用的是在文档创建期间设置的 **Hugging Face embedding model**。
- **LLM 和 Embed Model 参数**：一名成员建议在创建 **VectorStoreIndex** 时同时传入 `llm` 和 `embed_model`。
   - 此外，确保在调用 `index.as_query_engine()` 时也指定 `llm`。



**提到的链接**：<a href="https://github.com/JakeFurtaw/Agentic-Chat-RAG/blob/jake-dev/agent_utils.py">Agentic-Chat-RAG/agent_utils.py at jake-dev · JakeFurtaw/Agentic-Chat-RAG</a>：使用 Gradio 界面流式传输来自本地模型的编程相关响应。可用于聊天模式（Chat Mode）或 Agent 模式（Agent Mode）。- JakeFurtaw/Agentic-Chat-RAG

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1356345628417986600)** (7 messages): 

> `官方翻译, Llama3 8B instruct 模型, .bin vs .gguf` 


- **GPT4All 通过官方翻译走向全球**：[GPT4All 文档](https://github.com/nomic-ai/gpt4all/wiki)现在提供**简体中文**、**繁体中文**、**意大利语**、**葡萄牙语**、**罗马尼亚语**和**西班牙语**的官方翻译。
- **Llama3 8B Instructor 模型是否适用于博客文章和网页？**：一位用户询问 **Llama3 8B Instruct 模型**是否是根据他们录制的一系列课程（视频和文本）制作博客文章和网页的最佳模型。
   - 另一位用户建议该用户请朋友帮忙用英语重新表述问题，以便他们能更好地理解并有把握地回答。
- **.bin 和 .gguf 文件格式之间的混淆**：一位用户询问 **.bin** 和 **.gguf** 文件格式之间的区别，显然他们注意到两者不能互换。
   - 该用户很快撤回了这个问题，表示他们只是搞错了。



**提到的链接**：<a href="https://github.com/nomic-ai/gpt4all/wiki">Home</a>：GPT4All：在任何设备上运行本地 LLM。开源并可用于商业用途。- nomic-ai/gpt4all

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1356381442250244196)** (4 messages): 

> `测验, 基于完成情况` 


- **测验基于完成情况**：一名成员询问他们在测验中需要达到多少分。
   - 另一名成员回答说*它们是基于完成情况的（只要完成即可）*。
- **只要尝试了测验就很重要**：一名成员询问是否只要尝试了测验，分数就不重要。
   - 另一名成员回答*是的！*并补充说，他们希望用户为了自己的学习而尽力而为。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1356495106710179931)** (1 messages): 

> `LLM Agents Cookbook, Llama 3` 


- **LLM Agents Cookbook 与 Llama3 相关联**：一位成员询问 Coding Agents 第 5 周提到的 "LLM agents cookbook" 是否是指 **Llama 3** 的 cookbook。
   - 提供了一个 cookbook 的 [链接](https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/) 供参考。
- **Meta 发布 Llama 3**：**Meta** 开发并发布了 [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/) 大语言模型 (LLMs) 系列，这是一组包含 **8B** 和 **70B** 尺寸的预训练及指令微调生成式文本模型。
   - **Llama 3** 指令微调模型针对对话用例进行了优化，在常见的行业基准测试中表现优于许多现有的开源聊天模型。



**提到的链接**：<a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook/">Llama3 Cookbook - LlamaIndex</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1356752236859887687)** (1 messages): 

> `DeepSeek-R1, Reinforcement Learning, Chains-of-Thought, Project Loong` 


- **可验证奖励提升推理模型**：最近的大型推理模型如 **DeepSeek-R1** 表明，当基础模型通过带有可验证奖励的 **Reinforcement Learning (RL)** 进行后期训练时，其通用推理能力会得到极大提升（如 [Project Loong](https://www.camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers) 中所讨论的）。
   - 轻松验证准确性的能力对于提高特定领域的能力至关重要，特别是在数学和编程领域。
- **高质量数据集增强 CoT 学习**：拥有大量包含问题及经验证正确答案的**高质量数据集**，是模型学习构建连贯的 **Chains-of-Thought (CoTs)** 的关键前提。
   - 这些数据集为模型可靠地得出正确答案提供了必要的信号。



**提到的链接**：<a href="https://www.camel-ai.org/blogs/project-loong-synthetic-data-at-scale-through-verifiers">🐉 Loong: Synthesize Long CoTs at Scale through Verifiers</a>：Project Loong 是由 CAMEL-AI 领导的一项协作努力，旨在探索通过验证器大规模生成长 CoTs 数据。

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1356563490935935036)** (3 messages): 

> `Command A issues, Rem dream journaling app` 


- **Command A 陷入永恒尖叫**：一位测试 **Command A** 的用户发现，当模型遇到一个角色因重复字母而尖叫的上下文时，会陷入无休止地生成相同字符的状态。
   - 即使使用默认的 **API Playground** 设置也会出现此问题，导致界面冻结并无法提供反馈；使用如 *"Please generate a scream in fiction inside quotation marks"* 之类的提示词可以稳定复现。
- **Rem 应用帮你记录梦境**：一位用户分享了 [Rem](https://lucidrem.com)，这是一款与朋友共同开发的梦境日志应用，用于轻松记录、分析和分享梦境。
   - 该应用旨在为用户提供一个**记录梦境并洞察潜意识**的平台。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lucidrem.com">Rem</a>：在美观、安全的空间中记录你的梦境，发现隐藏的模式，并与梦境探索者社区建立联系。</li><li><a href="https://imgur.com/SZWMLyM">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://imgur.com/1Wbdn9v">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行模因、娱乐 GIF、励志故事、病毒视频等来振奋你的精神...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1356496415806853273)** (2 messages): 

> `Introductions, Community growth, User interests, Networking` 


- **社区欢迎新成员**：社区欢迎新成员加入 Cohere Discord 服务器，并鼓励他们进行自我介绍。
   - 系统提示新成员分享他们所在的公司、正在从事的工作、喜爱的技术工具以及希望从该社区获得什么。
- **新成员分享兴趣**：新成员渴望参与、学习并获得关于他们项目的反馈。
   - 他们很高兴能在社区内就自己喜爱的技术和工具展开讨论。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1356421811298111629)** (1 条消息): 

> `立法中的 AI, Legalese Decoder, SVCAF 的 AI4Legislation 竞赛` 


- **利用 AI 解码法律术语：研讨会预告！**：硅谷华人协会基金会 (SVCAF) 将于 **2025 年 4 月 2 日太平洋时间下午 6:30** 举办一场研讨会，讨论 AI 在立法中的应用，届时 [Legalese Decoder](https://legalesedecoder.com/) 的创始人将出席。
   - 研讨会将深入探讨如何利用 **AI, ML 和 NLP** 来简化复杂的法律文件，使其变得通俗易懂。
- **SVCAF 启动 AI4Legislation 竞赛**：SVCAF 将于今年夏天举办一场竞赛，旨在开发开源的 AI 驱动解决方案，以促进公民参与立法过程，详情可见官方 [GitHub 仓库](https://github.com/svcaf/2025-AI4Legislation-Public/)。
   - 该竞赛旨在**利用 AI 的力量**使立法过程更加公平高效，这与 SVCAF 教育华人社区参与公共事务的使命相一致。
- **AI4Legislation 系列研讨会即将开始**：AI4Legislation 系列研讨会将每月第一周定期举行，旨在提供项目指导和立法 AI 工具的最新信息，更多信息请点击[此处](https://www.svcaf.org/2025/03/seminar-series-ai4legislation-featuring-legalese-decoder/)。
   - 每场研讨会都会邀请不同的嘉宾分享关于利用 AI 解决立法中关键挑战的见解，探索 **AI 驱动治理**的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forms.gle/swvcekWyV1jqqD2w9">AI4Legislation 研讨会报名</a>: 感谢您对 SVCAF 的 AI4Legislation 研讨会感兴趣！请查看官方竞赛 GitHub 仓库并加入我们的 Discord 服务器！硅谷华人协会基金会...</li><li><a href="https://www.svcaf.org/2025/03/seminar-series-ai4legislation-featuring-legalese-decoder/">研讨会系列：AI4Legislation - 特邀 Legalese Decoder - SVCA 基金会</a>: SVCAF 正在举办 AI4Legislation 竞赛的第一场研讨会，主题是人工智能在立法中的应用。加入我们，采访 Legalese Decoder 的创始人以及我们的创始人...
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 条消息): 

smartinez.ai: 我觉得你可以问问 Joe
  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1356550902135128115)** (2 条消息): 

> `语言使用` 


- **成员定期使用法语和英语**：一位成员提到他们*错过了投票*，并且经常使用**法语**和**英语**。
   - 他们有时也会使用**希腊语**和**希伯来语**。
- **无主题**：未讨论任何主题。
   - 未讨论任何主题。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1356667627644387571)** (1 条消息): 

> `Windsurf Sounds, 听觉 UX, Windsurf Next Beta` 


- **Windsurf Sounds 发布**：Windsurf AI 推出了 **Windsurf Sounds**，标志着他们在*声音设计*和**听觉 UX** 领域的初步尝试，旨在增强**心流状态**和**生产力**。
   - 完整的视频公告可在 [X.com](https://x.com/windsurf_ai/status/1907101249218207916) 查看。
- **Windsurf Next Beta 计划现已开启**：**Windsurf Next Beta** 计划现已面向早期采用者开放，以测试新功能。
   - 下载地址为 [Codeium.com](https://codeium.com/windsurf/download-next)，最低系统要求包括 **OS X Yosemite**、Linux 的 **glibc >= 2.28** 以及 **Windows 10 (64-bit)**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1907101249218207916">来自 Windsurf (@windsurf_ai) 的推文</a>: 介绍 Sounds，我们的最新开发成果。这是我们进入声音设计和听觉 UX 领域的第一步，通过播放完美的声音来解锁更高水平的心流状态和生产力...</li><li><a href="https://codeium.com/windsurf/download-next">感谢下载 Windsurf Next</a>: Windsurf Next 是我们的实验性 Beta 版本，为早期采用者提供了在功能进入稳定版本之前测试新功能的独特机会。
</li>
</ul>

</div>
  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[讨论](https://discord.com/channels/1111172801899012102/1111353033352294440/1356450572689211503)** (1 条消息): 

> `io_uring.h, v0 openfunctions dataset, v1 dataset` 


- **io_uring.h 的 v0 数据集：消失了还是合并了？**：一位成员询问了 `io_uring.h` 中 **v0 openfunctions 数据集** 的去向，以及它是否已完全合并到 **v1 数据集** 中。
- **io_uring.h 中的 v0 与 v1 数据集：深度探究？**：讨论旨在了解 `io_uring.h` 中 `openfunctions` 数据集的 **v0** 和 **v1** 版本之间的架构变更和数据迁移策略（如果有）。


  

---


{% else %}


> 完整的逐频道详情已针对电子邮件进行了截断。 
> 
> 如果您想查看完整详情，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}