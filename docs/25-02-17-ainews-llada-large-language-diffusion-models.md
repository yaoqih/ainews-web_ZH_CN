---
companies:
- stepfun-ai
- scale-ai
- cambridge
- llamaindex
date: '2025-02-18T03:27:47.627285Z'
description: '**LLaDA (Large Language Diffusion Model) 8B** 是一款突破性的基于扩散（diffusion-based）的语言模型，其性能可与
  **LLaMA 3 8B** 媲美，而训练所需的 Token 数量仅为后者的七分之一（2 万亿 Token），并消耗了 13 万个 H800 GPU 小时。它通过在扩散过程中预测均匀掩码（masked）的
  Token，引入了一种新颖的文本生成方法，从而实现了多轮对话和指令遵循能力。


  与此同时，**阶跃星辰 (StepFun AI)** 发布了两款重磅模型：**Step-Video-T2V 30B**，这是一款文生视频模型，能够生成高达 **204
  帧**且具有高连贯性和运动质量的视频；以及 **Step-Audio-Chat 132B**，这是一款语音对语音（voice-to-voice）模型。


  此外，诸如 **Scale AI 的 EnigmaEval** 和 **剑桥大学的 ZeroBench** 等极具挑战性的多模态基准测试显示，目前的尖端模型在这些测试中得分竟然为零，凸显了这些任务的巨大难度。业界还注意到扩散模型在语言建模领域的回归——这种此前仅处于推测阶段的架构，如今已成功实现了规模化应用。'
id: 4466a990-74f9-45b5-9f9d-8903f3cf38e0
models:
- llada-8b
- llama-3-8b
- step-video-t2v-30b
- step-audio-chat-132b
- llama-2-7b
original_slug: ainews-llada-large-language-diffusion-models
people:
- arankomatsuzaki
- _akhaliq
- omarsar0
- iscienceluvr
- gallabytes
- maximelabonne
- reach_vb
title: LLaDA：大语言扩散模型
topics:
- diffusion-models
- text-generation
- multimodality
- video-generation
- voice-processing
- benchmarking
- instruction-following
- model-scaling
- gpu-usage
- long-context
- multi-turn-dialogue
---

<!-- buttondown-editor-mode: plaintext -->**中国 AI 就是你所需的一切？**

> 2025年2月14日至2025年2月17日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**211** 个频道，**11039** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1163 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天每个人都能有所收获。

在美国节假日的今晚晚些时候 [Grok 3 预期发布](https://x.com/elonmusk/status/1890958798841389499)之前，今天是值得关注的“小事”之日。要么是：

- **当天发布的大模型**是两款 StepFun 模型：[Step-Video-T2V](https://x.com/bdsqlsz/status/1891348664460714425?s=46)（30B Text to Video，具有显著的连贯性，包括 Sora 都会出错的著名高难度芭蕾舞者动作）和 [Step-Audio-Chat](https://huggingface.co/stepfun-ai/Step-Audio-Chat)（一个 132B 的 voice to voice 模型）。
- 或者你可以关注 Scale AI 的 [EnigmaEval](https://x.com/alexandr_wang/status/1891208692751638939?s=46)（极难的多模态谜题）或剑桥大学的 [ZeroBench]( https://x.com/jrobertsai/status/1891506671056261413?s=46)（100 个手动策划的多步视觉推理问题），**在这些评测中，目前的顶级模型得分均为 0。**
- 或者你可以看看 [Schulman 和 Zoph 关于 post-training 的最新演讲](https://x.com/johnschulman2/status/1891539960743743756)。

但我们选择将今日的头条新闻授予 [LLaDA: Large Language Diffusion Models](https://ml-gsai.github.io/LLaDA-demo/)，这是第一个扩展到足以与 Llama 3 8B 等自回归模型竞争的文本扩散模型。


![image.png](https://assets.buttondown.email/images/007f0b39-2759-4568-93a6-b7c97ff6055a.png?w=960&fit=max)


这是一种“白鲸”式的替代 LLM 架构，此前一直处于推测阶段，直到现在才成功扩展。其核心技巧是调整扩散模型以预测均匀掩码的 token，在扩散过程中生成文本：


![https://ml-gsai.github.io/LLaDA-demo/static/images/diff_normal_150ms.gif](https://ml-gsai.github.io/LLaDA-demo/static/images/diff_normal_150ms.gif)


对于那些习惯了文本从左到右流式输出的人来说，扩散模型似乎不太实用……直到你尝试在预设的文本结构和词汇选择之间进行填充（infill），或者创建连贯的长篇故事结构。LLaDA 可能是更大变革的开端。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

**AI 模型与研究发布**

- **LLaDA (Large Language Diffusion Model) 8B**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891343406334693879) 宣布发布 **LLaDA**，这是一个从零开始训练的 **8B 参数扩散语言模型**。其性能与 **LLaMA3 8B** 相当，但使用的 Token 数量减少了 7 倍（**2T tokens**）。[@_akhaliq](https://twitter.com/_akhaliq/status/1891487050936815693) 强调 **LLaDA** 在文本生成中采用了 **diffusion model** 方法，与传统的从左到右生成方式不同。[@omarsar0](https://twitter.com/omarsar0/status/1891568386494300252) 进一步详细介绍了 **LLaDA** 的特性，指出其具有竞争力的性能、可扩展性、**突破了逆转诅咒 (reversal curse)**，以及具备 **多轮对话和指令遵循** 的能力。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891337383200903625) 强调 **LLaDA-8B** 在标准基准测试中超越了 **Llama-2 7B**，并与 **Llama-3 8B** 表现持平。该模型在 **2.3 万亿 tokens** 上进行了训练，耗费了 **13 万个 H800 GPU 小时**。[@gallabytes](https://twitter.com/gallabytes/status/1891356261582557438) 对其“扩散”属性提出质疑，指出虽然作为一种新方法令人印象深刻，但缺乏 SDE、概率流或噪声。[@maximelabonne](https://twitter.com/maximelabonne/status/1891472925603090497) 对扩散模型在语言领域的回归表示惊讶。
- **Step-Video-T2V 30B**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891330624436220069) 分享了 **Step-Video-T2V** 的开源发布。这是一个 **30B 参数的文本生成视频模型**，能够生成多达 **204 帧** 的视频，并强调其生成的 **高质量视频具有强大的运动动力学和一致的内容**。[@_akhaliq](https://twitter.com/_akhaliq/status/1891353872809025961) 还讨论了 **Step-Video-T2V 技术报告**，详细介绍了视频基础模型的实践、挑战和未来。[@reach_vb](https://twitter.com/reach_vb/status/1891415982003949726) 宣布发布来自 **@StepFun_ai** 的 **30B 文本生成视频模型**，还提到了一个用于**更快推理的 Turbo 模型**，并且该模型采用 **MIT 许可证**。
- **Step-Audio 130B**：[@_akhaliq](https://twitter.com/_akhaliq/status/1891528348590833834) 宣布了 **Step-Audio**，这是一个 **1300 亿参数的多模态 LLM**，用于理解和生成人类语音。[@reach_vb](https://twitter.com/reach_vb/status/1891517368603492697) 针对这个 **132B 参数的端到端语音语言模型 (Speech LM)** 问道：“chat，这是真的吗？”，并将其描述为“**语音输入，语音输出** 🤯”，同时指出其采用 **APACHE 2.0 许可证**。
- **Mistral Saba**：[@sophiamyang](https://twitter.com/sophiamyang/status/1891487141718376580) 宣布了 **Mistral Saba**，这是 **MistralAI** 的首个区域语言模型，拥有 **24B 参数**，在来自 **中东和南亚** 的数据集上训练，支持 **阿拉伯语和印度语系语言**，特别是 **泰米尔语和马拉雅拉姆语**。[@sophiamyang](https://twitter.com/sophiamyang/status/1891488607518462157) 指出该模型可通过 la Plateforme 上的 **API `mistral-saba-2502`** 获取，并提供定制训练服务。

**基准测试与性能**

- **LLaDA 性能**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1891337383200903625) 指出 **LLaDA 8B 在标准零样本/少样本学习任务上超越了 Llama-2 7B**，且表现与 **Llama-3 8B 持平**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1891343406334693879) 强调尽管训练数据较少，**LLaDA** 的性能仍与 **LLaMA3 8B** 旗鼓相当。
- **ZSEval 基准测试**：[@lateinteraction](https://twitter.com/lateinteraction/status/1891288503201259843) 介绍了 **ZSEval**，这是一个新的 **LLM 基准测试**，利用 **多人游戏** 让模型在 **知识、推理和规划** 方面展开竞争，同时还测试了使用 **DSPy 优化** 的 **自我改进能力**。
- **GPQA 分数提升**：[@casper_hansen_](https://twitter.com/casper_hansen_/status/1891518246047686976) 质疑为什么在 **99.9% 为数学内容的数据集上进行 SFT 会将 GPQA 分数** 从 **45.30% 提升至 62.02%**，并询问是基础模型的数学能力较弱，还是 GPQA Diamond 与数学相关。
- **Mistral Saba 的阿拉伯语基准测试表现**：[@sophiamyang](https://twitter.com/sophiamyang/status/1891487842964021305) 提到了 **Mistral Saba 在阿拉伯语基准测试中的强劲表现**。

**工具与库**

- **OmniParser 更新**：[@reach_vb](https://twitter.com/reach_vb/status/1891467489030082875) 宣布了 **Microsoft 对 OmniParser 的静默更新**，这是一款**屏幕解析工具**，并指出其比 **v1 快 60%**，在 **4090** 上具有**亚秒级延迟**。[@mervenoyann](https://twitter.com/mervenoyann/status/1891524621435830700) 强调了改进后且更快的 **OmniParser** 是一款**用于 Web 自动化的突破性截图解析器**，它是开源的，并且可以与 **Qwen2.5VL, DeepSeek R1, 4o/o1/o3 mini** 等模型配合使用。
- **Ollama 模型下载**：[@ollama](https://twitter.com/ollama/status/1891581808451444932) 确认看到来自 **Ollama** 的模型下载流向主要的云托管商，并且 [@ollama](https://twitter.com/ollama/status/1891376667668426933) 确认 **Ollama 是用 Go 编写的**。
- **LangGraph.js 结合 MongoDB**：[@LangChainAI](https://twitter.com/LangChainAI/status/1891505383874625618) 宣布了一场关于使用 **MongoDB 数据平台结合 LangGraph.js** 构建 **JavaScript AI Agent** 的网络研讨会，内容涵盖 **Node.js 集成**和**状态持久化**。
- **Arch AI 原生代理**：[@_akhaliq](https://twitter.com/_akhaliq/status/1891289618978316749) 介绍了 **Arch**，这是一个由 Envoy 贡献者构建的 **Agent 原生 AI 代理 (AI-native proxy)**，具有**边缘护栏 (edge guardrails)、任务路由与 Function Calling 以及可观测性**等功能。

**中国与 DeepSeek 焦点**

- **DeepSeek 的崛起**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891502835910459731) 指出 **DeepSeek 的文锋** 推动了**中国股市增长约 1.3 万亿美元**，并暗示 **DeepSeek 的叙事完成了任何经济刺激措施都无法做到的事情**。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891474971668451679) 描述了 **CCTV 关于杭州科技的报道**，其中包括 **DeepSeek, Unitree, DeepRobotics 和 Game Science**，强调了先进技术与怀旧传播方式之间的对比。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891446214236807451) 将 **DeepSeek** 与**腾讯、小米、华为、阿里巴巴**列为参加北京座谈会的重要科技领袖，**文锋作为唯一的 AI 领域代表**出席。
- **文锋的影响力**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891458990057382384) 表示**文锋将激励一代又一代的亚洲技术宅 (autists)**，并且 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1891466290092478565) 建议**文锋少年感的外貌非常适合改编成动漫/漫画**，甚至比《Dr. Stone》更好。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891365138529194473) 确认**梁文锋会见了习近平**，并注意到他在与党内精英会面时始终如一的西装和姿势。
- **中国科技座谈会**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891550013265752351) 分析称**北京座谈会**具有深刻的象征意义，指出了**文锋和星星**的加入、**百度和字节跳动**的缺席、与**马云**的和解，以及**王沪宁**的主持。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891580287118705040) 提到 **习近平思想** 的作者 **王沪宁** 主持了该科技座谈会。

**Perplexity Deep Research 与使用**

- **Deep Research 发布与使用**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891577500888625520) 表示 **Perplexity 上的 Deep Research 使用量正在飙升 🚀**，占据了搜索 QPS 的很大一部分。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891578863219568952) 宣传 **Perplexity 提供几乎无限的 Deep Research Agent**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891563239240069245) 演示了 **Deep Research 作为小型企业注册法律顾问** 的功能，强调了其相比昂贵的人类顾问的易得性。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891537805387485571) 展示了 **Perplexity Deep Research 作为产品经理进行路线图规划**，展望了公司中的 AI 员工/实习生。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891233048605184371) 展示了 **Perplexity Deep Research 像 Bill Ackman 一样撰写投资备忘录**，并以 **$UBER** 为例。
- **Deep Research 问题与改进**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891535474315149662) 承认 **Perplexity Deep Research 在财务查询方面存在不准确性**，例如过时的比特币价格和公司市值，并承诺进行修复并提供更可靠的数据源。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1891344192707969134) 征求关于 **Deep Research** 的反馈，请求演示需求和痛点，并提到可能支持更长的报告和学术来源。[@hrishioa](https://twitter.com/hrishioa/status/1891350088527835137) 分享了使用 **Deep Research** 的技巧，包括在搜索前通过 **对话** 来提供上下文。[@hrishioa](https://twitter.com/hrishioa/status/1891511609098387487) 提供了在使用和不使用 Prompt 工程的情况下，以及使用 **o1pro** 代替 **o3minihigh** 时 **Deep Research** 的示例。

**AI 与社会、伦理及未来**

- **AI Robotics 突破**：[@adcock_brett](https://twitter.com/adcock_brett/status/1891581168299942256) 宣布 **今年是我们一直期待的 AI Robotics 突破之年**，并表示 **10 年前还为时过早**。[@adcock_brett](https://twitter.com/adcock_brett/status/1891305061365465121) 表达了让 **人形机器人制造并驾驶 eVTOL** 的梦想。[@Meta](https://twitter.com/Yuchenj_UW/status/1891560033193955795) 宣布 **Meta 的 AI 现在可以通过非侵入性脑活动记录以 80% 的准确率读取思想**，强调了其对脑损伤患者的潜力以及神经科学与 AI 之间的协同作用。
- **ASI 与生存风险**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891547149302739202) 断言：“**我的个人 ASI 将根据我的决定给我提供尽可能多的怪物（而且是我想要的口味）。除此之外的一切都是灭绝事件**”。[@sama](https://twitter.com/sama/status/1891533802779910471) 提到，在品味极高的测试者中，“**尝试 GPT-4.5 更多地是一种‘感受到 AGI’的时刻**”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891394128774041712) 调侃说“**ASI 最后的孤注一掷 (Hail Mary)**”拯救了一个崩溃的帝国。
- **法律与服务中的 AI**：[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1891308426841461054) 建议，如果 **Harvey（AI 律师事务所）是法律的未来**，那么自然的结论是创办自己的律师事务所并统治该垂直领域。[@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1891308159530082315) 对律师事务所等服务型公司像软件公司一样提高利润率表示看空。[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1891329620558983633) 表示，由于所有权规则，**“不可能”建立一家既是科技公司又用股权奖励非律师员工的律师事务所**。
- **LLM 中的偏见与观点**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1891234868664668380) 认为 **LLM 是媒体出版商，会有自己的观点**，将偏见视为一种特性而非缺陷，并预测会出现 **“福克斯新闻版和纽约时报版模型”**。
- **AI 安全与伦理**：[@mmitchell_ai](https://twitter.com/mmitchell_ai/status/1891534146213666870) 注意到人们对他们主张“**不应开发完全自主的 AI Agent**”的论文很感兴趣。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891414501011968390) 将“**安全主义 (Safetyism)**”确定为通往成熟人类道路上的三个大过滤器事件之一。
- **AI 开发的未来**：[@omarsar0](https://twitter.com/omarsar0/status/1891570913327374496) 预测我们将不再关心驱动 AI 系统的具体模型，**产品或开发体验将成为获胜的关键因素**。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1891590473153695860) 呼吁回归更 **开放、透明和协作的 AI**，让人想起 **2016-2020 年**。

**幽默/迷因**

- **Grok 3 热度**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1891243031090626776) 开了一个黑色幽默的玩笑：“**一个人为了告诉我们 Grok 3 到底有多好而牺牲了，永远不要忘记**”，获得了超过一百万的曝光量。[@marktenenholtz](https://twitter.com/marktenenholtz/status/1891544044016173475) 鼓励大家为 **Grok 3 的发布**造势，期待一个“**拥有比 GPT-4 大得多的集群的顶尖团队能交付出什么**”。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1891491103674626177) 评论道：“**如果属实，这也不过如此。即便拥有 60k 个 GPU，我也能预见到 Grok 3 甚至无法确保哪怕一天的领先地位**”。
- **ChatGPT-4o 的编程实力**：[@_akhaliq](https://twitter.com/_akhaliq/status/1891296414174490897) 分享了由 **ChatGPT-4o 生成的令人印象深刻的代码**，用于弹球模拟；[@_akhaliq](https://twitter.com/_akhaliq/status/1891249188366701011) 展示了另一个用于旋转 ASCII 球体的 **ChatGPT-4o 代码**，并将其与 [@flavioAd](https://twitter.com/flavioAd/status/1891246340518133912) 的原始测试进行了对比。[@mckbrando](https://twitter.com/mckbrando/status/1891280957568610454) 简单地表示：“**我不知道我们做了什么，但现在的 ChatGPT 感觉变酷了**”。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Zonos：开源权重语音克隆模型**

- **Zonos，一个易于使用的 1.6B 开源权重文本转语音模型，可以从 10 秒的剪辑中创建新语音或克隆语音** ([得分: 330, 评论: 80](https://reddit.com/r/LocalLLaMA/comments/1irhttv/zonos_the_easy_to_use_16b_open_weight/)): **Zonos** 是一个 **1.6B 参数的开源权重文本转语音模型**，可以生成新语音或从短至 **10 秒** 的剪辑中克隆语音。它在 **8GB VRAM** 上运行效率很高，并且可以使用 **Docker 在 Linux 上**轻松设置，同时提供了一个 [Windows 友好分支](https://github.com/sdbds/Zonos-for-windows)（尽管作者未对其进行担保）。该模型及其**混合版本**可在 [Hugging Face](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) 上获取，作者建议使用 **Ocenaudio** 编辑语音样本。
  - 用户讨论了 **Zonos 的技术设置**，包括在 Linux 和 Windows 上使用 **Docker**，以及 Windows 用户需要 **WSL2** 来启用带有 **Nvidia Container Toolkit** 的 Docker。一些用户报告了 **docker-compose.yml** 的问题，并需要调整 **network_mode** 和端口以确保正常运行。
  - 关于 Zonos 与 **ElevenLabs** 相比的**性能和质量**存在争论，一些用户认为 Zonos 不够自然和富有表现力，而另一些人则欣赏其成本效益和语音克隆的潜力。用户注意到 **espeak** 被用于音素化，这可能会影响非英语语言的表现。
  - 讨论强调了 Zonos 的**局限性和潜在改进方向**，例如 30 秒的输出限制，以及在生成的语音中需要更好的重音和情感控制。社区期待未来的更新和改进，一些用户表示有兴趣将 Zonos 用于 **sillytavern** 等应用，并将其与 **Kokoro** 和 **Fairseq** 等其他模型进行比较。


**主题 2. OpenArc Python API 增强 Intel 推理**

- **今天我发布了 OpenArc，这是一个用于在 Intel CPU、GPU 和 NPU 上实现更快推理的 Python 服务 API。低层级、极简依赖，并附带首个用于模型转换的 GUI 工具。** ([Score: 279, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1ir9mcw/today_i_am_launching_openarc_a_python_serving_api/)): **OpenArc** 是一款新推出的轻量级推理引擎，旨在利用 **Transformers** 中的 **Optimum-Intel** 来发挥 **Intel** 硬件加速性能。它具有一个包含四个端点的强类型 API、原生聊天模板，以及一个用于维护暴露参数的 pydantic 模型。OpenArc 面向 **Intel** 加速器和边缘设备的用户，提供了一种低层级的上下文管理方法，并推广了在本地与 LLM 交互的 API 调用设计模式。未来的更新包括 **OpenAI proxy**、**docker compose 示例**以及对多 GPU 执行的支持。
  - **厂商锁定担忧**：用户对厂商锁定表示担忧，强调了在 **Apple、Intel、Nvidia 和 AMD** 等多种硬件上运行模型的重要性。摆脱 GPU 依赖运行 LLM 的能力将促进本地执行，打破 **Nvidia** 在市场上的主导地位。
  - **性能与兼容性**：人们对 **OpenVINO** 与 **llama.cpp** 的性能对比很感兴趣，讨论强调 **OpenVINO** 使用独特的图表示进行量化，这与 **llama.cpp** 没有直接可比性。轶事证据表明 **OpenVINO** 在高精度下速度更快，目前正在努力针对较新模型对其性能进行基准测试。
  - **社区与协作**：几位用户对该项目表示赞赏，并对未来的更新表示期待，特别是用于在低成本 **Intel VPS** 上进行测试的 **Docker compose 示例**。还有人呼吁开展协作，特别是那些针对 **Intel** 平台开发类似项目的用户，其中一位用户特别提到了在多系统模型托管和分布式微调方面的工作。


**主题 3. DeepSeek-R1: MoE 模型 CPU 性能**

- **DeepSeek-R1 纯 CPU 性能 (671B, Unsloth 2.51bit, UD-Q2_K_XL)** ([Score: 120, Comments: 59](https://reddit.com/r/LocalLLaMA/comments/1ir6ha6/deepseekr1_cpuonly_performances_671b_unsloth/)): **DeepSeek-R1** 作为一个 671B 模型，由于其 **MoE 特性**，在纯 CPU 推理性能方面表现出色，测试在 **Xeon w5-3435X** 和 **TR pro 5955wx** 等多种 CPU 上进行。使用 **Unsloth** 的 2.51-bit 量化相比 1.58-bit 提供了更好的质量和速度，在 Xeon w5-3435X 上达到了 **4.86 tok/s**。**kTransformer** 通过混合使用 CPU 和 GPU 提供了近 2 倍的性能提升，尽管它受限于 VRAM 的上下文长度。作者正在寻求更多的 CPU 性能数据，并指出 **STREAM** 基准测试结果低于预期，计划针对 kTransformer v0.3 和 Prompt 处理速率进行更新。
  - **性能对比**：用户正在讨论各种 CPU 的性能，包括 **Xeon w5-3435X**、**Epyc Rome** 和 **Threadripper 5965wx**，重点关注内存带宽和 CCD 的影响。**Xeon Sapphire Rapids** 显示出低于预期的实际带宽，而 **Epyc Rome** 搭配 DDR4 内存提供了良好的性价比，在 STREAM 上达到了约 160GB/s。
  - **优化技术**：优化速度的建议包括尝试线程数、避免 KV cache 量化，以及使用 `--no-mmap --mlock` 等标志来提高 Prompt 处理速度。**CheatCodesOfLife** 提供了针对不同缓存类型和量化设置的详细性能对比。
  - **量化与速度**：**Unsloth** 的 **2.51-bit 量化**方法因其优于 1.58-bit 的性能而受到关注，用户报告了不同配置下的不同速度。值得注意的是，**thereisonlythedance** 提到通过部分卸载（offload）达到了每秒 6 个 token，而其他人则指出了硬件升级对性能提升的影响。


## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Nvidia GPU：10 个月内算力翻倍引发疑问**

- **[Nvidia 算力每 10 个月翻一番](https://i.redd.it/bs0887rrppje1.png)** ([Score: 481, Comments: 37](https://reddit.com/r/OpenAI/comments/1irlie2/nvidia_compute_is_doubling_every_10_months/)): **NVIDIA 的算力**预计每 **10 个月**翻一番，正如一张显示从 **2020 年到 2024 年**跨越 Hopper、Ampere、Volta 和 Pascal 等 GPU 世代的 FLOP/s 指数级增长图表所示。数据表明预计增长率为**每年 2.3 倍**，置信区间在 **2.2 到 2.5 倍**之间。
  - 用户质疑 **NVIDIA 算力预测的准确性**，认为这些数字可能涉及不同精度级别的比较（例如 **FP16 到 FP8**），这可能会夸大感知到的性能提升。人们对所引用的具体 FLOPS 类型感到好奇，假设倾向于可能是 **FP16** 中的 **Tensor FLOPS**。
  - 讨论将增长率与**摩尔定律 (Moore's Law)** 进行了比较，指出虽然摩尔定律涉及晶体管数量，但 NVIDIA 的预测涵盖了性能和生产规模。用户强调该图表反映的是总装机算力，而不仅仅是单个 GPU 的算力，如果生产数量增加，这可能会产生误导。
  - 对于增长率是否会超过需求存在怀疑，评论指出**算力需求**的增长甚至比供应还要快。用户询问了 **Blackwell GPU** 的状态及其维持这种增长的潜力，并反思了视频处理等领域对更多算力的需求。


**主题 2. Video-to-Video AI 的进展：混元 (Hunyuan) 的哈利·波特动漫**

- **[哈利·波特动漫 2024 - 混元 Video to Video](https://v.redd.it/01n918qh1pje1)** ([Score: 683, Comments: 69](https://reddit.com/r/StableDiffusion/comments/1iritgq/harry_potter_anime_2024_hunyuan_video_to_video/)): 该帖子提到了利用**混元 (Hunyuan) Video-to-Video 模型**进行的 **Harry Potter Anime 2024** 转换。然而，帖子正文中没有提供更多细节或背景。
  - 用户对**工作流和渲染细节**表示感兴趣，并要求提供有关 GPU 使用情况和渲染所需时间的信息。**Inner-Reflections** 分享了一个 [Lora 模型](https://civitai.com/models/1132089/flat-color-style?modelVersionId=1315010) 的链接，并提到了 **controlnets** 改进结果的潜力。
  - 用户对**一致性和角色设计**提出了担忧，指出了诸如**罗恩的发色**和缺乏面部表情等问题。**DaddyKiwwi** 和 **FourtyMichaelMichael** 讨论了保持风格一致性的挑战，这可能是由于渲染限制造成的。
  - **Neither_Sir5514** 等人强调了对低帧率 **2D 艺术风格**的偏好，这与不太吸引人的 2.5D 类皮克斯风格形成对比。**Liquidphantom** 补充说，动漫通常在**动作场面为 12fps**，**对话场面为 8fps**，强调了这项技术对动画的适用性。


**主题 3. 开源视频模型 Step-Video-T2V：高需求，高创新**

- **[新开源视频模型：Step-Video-T2V](https://v.redd.it/cr62isaa1qje1)** ([Score: 330, Comments: 63](https://reddit.com/r/StableDiffusion/comments/1irn0eo/new_opensource_video_model_stepvideot2v/)): 该帖子介绍了一个新的**开源视频模型 Step-Video-T2V**。帖子正文中没有提供额外的细节或背景。
  - **VRAM 需求与优化**：**Step-Video-T2V** 模型需要 **80GB VRAM** 才能达到最佳性能，但讨论表明，通过**量化 (quantization) 和优化**，它可能在 **24GB VRAM** 上运行。预计集成到 **Diffusers 库**中将增强优化能力。
  - **技术细节与资源**：该模型被描述为拥有 **300 亿参数**，具有深度压缩 VAE，可实现 **16x16 空间和 8x 时间压缩比**。代码和权重等资源可在 [GitHub](https://github.com/stepfun-ai/Step-Video-T2V) 和 [Hugging Face](https://huggingface.co/stepfun-ai/stepvideo-t2v) 上获得。
  - **VRAM 创新**：一篇链接文章讨论了 **SanDisk 的新 HBF 内存**，该内存可以在 GPU 上实现高达 **4TB 的 VRAM**，这可能有利于读取密集型任务，但由于延迟比 HBM 高，对训练的益处较小。有人猜测 GPU 会采用混合 **HBF/HBM 架构**，以平衡计算和存储需求。


**主题 4. AI Agent Apply Hero：大规模职位申请及其影响**

- **AI Agent Apply Hero 已完成超过 160 万份职位申请** ([分数：301，评论：30](https://reddit.com/r/ClaudeAI/comments/1irqwxn/ai_agent_apply_hero_has_done_over_16m_job/))：**AI Agent Apply Hero** 使用 **Claude** 驱动其模型，已提交了超过 **160 万份职位申请**，展示了 AI 能力的重大进步。该 Reddit 帖子强调了 AI 技术在未来几年的快速发展和潜力。
  - 评论者对 **AI Agent Apply Hero** 的有效性表示怀疑，认为它制造了不必要的干扰，并对其获得工作的成功率提出质疑。**Deep_Area_3790** 和 **DaShibaDoge** 对其给职位申请过程带来的影响以及这 **160 万份申请** 的实际结果表示担忧。
  - 一些用户（如 **EngineeringSmooth398** 和 **Halbaras**）批评了公司繁琐的职位申请流程，认为 AI 工具可能会凸显当前系统的冗余和低效。**literum** 建议未来 AI 可能会更高效地管理职位申请，类似于公司使用 AI 进行简历筛选的方式。
  - **Funny_Ad_3472** 幽默地指出了自动化申请与人为因素之间的脱节，提到由于 AI 自动化，被叫去参加自己都不记得申请过的职位的面试是多么奇怪。


**主题 5. AI 图像修复：放大 Windows XP Bliss 壁纸**

- **[经典的 Windows XP Bliss 壁纸从 800x600 放大至 8K (7680x4320) - 使用 Flux Dev 进行扩图与放大](https://www.reddit.com/gallery/1irhjoy)** ([分数：170，评论：22](https://reddit.com/r/StableDiffusion/comments/1irhjoy/iconic_windows_xp_bliss_wallpaper_from_800x600_to/))：经典的 **Windows XP 'Bliss' 壁纸** 已使用 **Flux Dev** 等 **AI 工具** 从 **800x600** 分辨率放大至 **8K (7680x4320)**。这展示了 AI 在增强经典数字图像方面的应用。
  - 讨论强调了 'Bliss' 图像的**原始拍摄地点**，并指出该地点已经存在许多高质量图像，从而质疑 AI 放大的必要性。**External_Waltz_8927** 等人批评了 AI 生成的 8K 图像缺乏细节，认为不当的设置（如过度的去噪）可能会降低图像质量。
  - **LatentSpacer** 分享了使用 AI 进行图像放大的详细 **workflow**（工作流），包括该过程的 [链接](https://pastebin.com/EK0iuRQu)，并强调了调整去噪和 tile size 等设置的重要性。最终图像分享在[此处](https://freeimage.host/i/2yK0bSa)，并在 **Photopea** 中进行了微调以去除伪影。
  - 对话还提到了在 [archive.org](https://archive.org/details/bliss-600dpi) 上可以找到 **PNG 格式** 的 **存档高清原图**。此外还有关于原片相机 **Mamiya RB67** 及其大底片格式的讨论，这种格式提供了丰富的细节，建议在 AI 提示词中使用它以获得更好的输出质量。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. Grok 3 引发 AI 辩论：马斯克的言论遭遇社区质疑**

- **马斯克吹捧 Grok 3 为“地球上最聪明的 AI”**：埃隆·马斯克预告了即将发布的 **Grok 3**，宣称它是 *“地球上最聪明的 AI”* 并安排了现场演示，这在用户中引发了热议，但也带来了对其实际效用和潜在偏见的怀疑。人们担心 **Grok 3** 过度迎合文化偏好而非真正的智能，与此同时，OpenAI 董事会坚决拒绝了马斯克控制该公司的 **974 亿美元** 报价，并表示 *“OpenAI 不予出售。”* ([彭博社报道](https://www.bloomberg.com/news/articles/2025-02-14/openai-board-rejects-musk-s-97-4-billion-bid-to-control-company?srnd=undefined))
- **用户推测 Grok 3 的能力**：爱好者们正热切期待 **Grok 3**，一位用户乐观地表示：*“如果 Grok 3 强 10 倍，那就全剧终了”*，同时引用了 [Sam Altman 的推文](https://x.com/sama/status/1890816782836904000)，该推文预告了 **ChatGPT-4o** 的改进。然而，一些人担心 **Grok** 可能会根据其编程成为宣传工具，并期待评估其相对于 **O1** 等现有 AI 模型的表现。
- **Grok 潜在的政治倾向引发关注**：用户表示担心由埃隆·马斯克支持的 **Grok** 可能会表现出反映其政治观点的偏见，强调 AI 需要平衡的审查制度并避免政治叙事。讨论集中在 **AI 是否应该影响政治观点**，希望 **Grok 3** 在技术上表现出色，但担心它可能成为宣传工具。

**主题 2. DeepSeek 模型推动性能与伦理讨论**

- **DeepSeek R1 竞逐推理桂冠**：用户们正将 **DeepSeek R1** 与 **O3 Mini High** 进行对比，并倾向于认为 R1 在推理任务上表现更优。[Nebius AI Studio 的一条推文](https://x.com/nebiusaistudio/status/1890397790250893743)指出 **DeepSeek R1** 的性能端点达到了 *60+ tokens/秒*。虽然 **R1** 在推理方面表现出色，但 **O3 Mini High** 在不同场景下的连贯性也受到了关注。
- **DeepSeek R1 的推理 Token 在 OpenRouter 上引发疑问**：**OpenRouter** 的用户观察到免费版 **DeepSeek R1** 模型存在异常，推理 token 意外地出现在响应内容中，而不是独立的字段里。**OpenRouter** 团队正在解决这些问题，以优化开源推理模型的行为，并参考了 [DeepSeek 的 GitHub](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations) 使用指南。
- **关于 DeepSeek 是否为 CCP 项目的辩论**：**LM Studio** 社区就 **DeepSeek** 是否为 CCP 项目展开了轻度辩论，讨论了其资金来源和 AI 的快速进步。有推测认为，AI 发展的飞速节奏可能会带来不可预见的突破；同时，一些用户根据 **Aider** 用户的报告，对 **DeepSeek** 受训练数据和 Web 界面设置影响而可能存在的审查制度表示担忧。

**主题 3. Llama 3.2 挑战 VRAM 限制并催生量化解决方案**

- **Llama 3.2 吞噬 VRAM**：Hugging Face 用户正面临 **Llama 3.2** 模型在本地使用时需要 **20GB VRAM** 的难题，这促使人们探索如 [EasyQZ](https://easyqz.online) 等量化技术，以便在显存有限的 GPU 上运行更大的模型。性能基准测试结果显示，一个 **9B 模型** 的表现出人意料地优于更大规模的模型。
- **Unsloth 优化 Llama 3.2 微调**：**Unsloth AI** 强调其能够对 **Llama 3.2** 进行微调，支持高达 **32k** token 的扩展上下文长度，展示了在处理长序列方面的重大进展。他们还修复了 **Phi-4** 中的错误，并提供了 **Llama 3.2** 和 **Phi-4** 的微调教程（[Unsloth 博客](https://unsloth.ai/blog/llama3)，[Unsloth 博客 Phi-4](https://unsloth.ai/blog/phi4)，[YouTube 教程](https://www.youtube.com/watch?v=JJWvYQdOVOY)）。
- **Torchtune 社区增强 Llama 3.3 Tokenizer**：**Torchtune** 社区就为 **Llama3 tokenizer** 添加“tool”选项达成共识，以确保与“ipython”的向后兼容性，展示了协作改进的成果。此次更新反映了在优化和调整模型以适应多样化应用并保持兼容性方面的持续努力。

**主题 4. RAG 与微调之争加剧：效率与应用重点凸显**

- **微调在性能上压倒 RAG**：**Unsloth AI** Discord 成员发现，在拥有充足训练数据的情况下，微调的表现显著优于 **RAG**，从而在 **LLM** 应用中实现卓越性能。一位成员建议 **RAG** 可能会变得小众化，而微调将成为大多数用例的主流方法。
- **RAG 在大型代码库中面临 Repomap 障碍**：**Aider** 用户讨论了 **RAG** 和 **Repomap** 在大型仓库中的低效问题，观察到随着代码库规模增加，收益递减，并建议使用 **BM25** 和 **RAG** 的混合评分来改进搜索。文中引用了一篇[关于 10k 仓库规模代码库 RAG 的博客文章](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/)，概述了弥补 LLM 上下文缺口的方法。
- **Perplexity Deep Research 挑战传统搜索**：**Perplexity AI** 推出了 **Deep Research**，该功能通过多次搜索和来源分析生成深度研究报告，为免费用户提供**每天 5 次查询**，Pro 用户则为 **500 次**。用户评价指出了其与 **OpenAI** 产品相比的局限性和偶尔出现的“幻觉”问题，但仍对其未来的改进抱有希望（[Perplexity 博客](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)）。

**主题 5. 社区驱动的工具化与优化工作推动 AI 开发进阶**

- **Unsloth 的招聘挑战寻求深度学习问题解决者**：**Unsloth AI** 推出了[五项挑战](https://x.com/danielhanchen/status/1891194528931209644)，为深度学习优化专家提供高达 **50 万美元/年** 的年薪，寻求社区在 **Griffin** 和 **FSDP2** 等领域的贡献。该计划旨在通过开放的社区参与加速关键技术领域的进步。
- **OpenRouter 图表实时数据更新及 Tooltip 扩展发布**：**OpenRouter** 通过 **Google AI 的 Vertex** 增强了其吞吐量和延迟图表的实时更新功能（[OpenRouter 公告](https://x.com/OpenRouterAI/status/1891510121139769542)），并推出了 **Versatile AI Tooltip** Chrome 扩展，以便使用 **OpenRouter** 模型快速处理文本。该扩展专注于高效的摘要和翻译（[Chrome 应用商店](https://chromewebstore.google.com/detail/versatile-ai-tooltip/jdldlpaafnfeggaopbmelkmgagikfekm)）。
- **Cursor IDE 用户创新 MCP Server 配置**：**Cursor IDE** 用户正在积极分享创新的 **MCP Server** 配置，包括 **Brave Search** 和 **Sequential Thinking**（详见[此 YouTube 视频](https://youtu.be/RCFe1L9qm3E?si=V4v8Y8XT1MkhKx_O)），以增强编码工作流和文档处理。据报告，这些配置显著提高了编码过程的可靠性。

---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 发布招聘挑战**：Unsloth 宣布了[五项挑战](https://x.com/danielhanchen/status/1891194528931209644)，为深度学习优化领域的问题解决者提供高达 **50 万美元/年** 的薪资。
   - 公司寻求社区在 **Griffin**、**FSDP2** 和其他技术领域的进步做出贡献。
- **用户对 AI Agents 持怀疑态度**：关于 **AI Agents** 定义的讨论兴起，一些人认为它们仅仅是精心编写的 Prompts，而非突破性技术。
   - 成员们对围绕 AI Agents 的炒作表示担忧，强调实用性比营销语言更重要。
- **CUDA OOM 错误困扰用户**：成员们讨论了在训练过程中遇到 **CUDA Out-of-Memory (OOM)** 错误的问题，这可能是由于数据不平衡导致的。
   - 尽管使用了类似的设置，但即使是数据的差异也可能导致 **VRAM** 负载激增，从而使解决工作变得复杂。
- **Fine Tuning 在性能上优于 RAG**：成员们发现，在拥有足够训练数据的情况下，Fine Tuning 的表现可以显著优于 **RAG**，从而提高 **LLM** 应用的整体性能。
   - 一位成员建议，**RAG** 可能会被委派给利基用途，而 Fine Tuning 将继续占据主导地位。
- **提出 SCION 优化器**：提出了一种新的优化器 **SCION**，它可能在不使用 **Adam** 的情况下训练大型神经网络，在减少内存占用的同时改善结果。
   - 这引发了关于 **SCION** 在模型训练中高效管理计算资源的实际应用的讨论。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布 Deep Research**：Perplexity AI 推出了 **Deep Research** 功能，该功能通过多次搜索和来源分析生成深度研究报告。非订阅用户每天限 **5 次查询**，**Pro 用户**每天限 **500 次**。
   - Deep Research 已在 Web 端上线，并即将登陆 **iOS**、**Android** 和 **Mac**。根据[官方博客](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)介绍，它在“人类最后的考试”（Humanity’s Last Exam）等专家级任务中获得了高分。
- **用户批评 Deep Research 的局限性**：用户正在将 **Perplexity Pro** 与 **You.com** 进行比较，指出 **Deep Research** 存在限制，且偶尔会产生响应**幻觉**（hallucinates），导致部分用户考虑价格更具吸引力的替代方案。
   - 尽管存在担忧，且在 [Reddit](https://www.markdownguide.org/tools/reddit/) 上出现了与 OpenAI 产品的负面对比，但社区成员仍对 Perplexity 未来功能的改进抱有希望。
- **Sonar API 缺失引用问题曝光**：成员们反映 **Sonar API** 未能按预期返回引用（citations），尽管输出内容中包含了参考标识，这引发了对其功能的困惑。
   - 一位用户澄清说，引用是独立的，需要正确的访问权限，详见 [Perplexity 文档](https://docs.perplexity.ai)。
- **DeepSeek 的影响引发辩论**：成员们正在探讨 **DeepSeek** 对 AI 方法论的潜在影响，认可其重塑当前实践的能力，如 [Perplexity AI 页面](https://www.perplexity.ai/search/how-does-deepseek-impact-ai-de-Ks0imm6zSYyr2b53rIkBiQ)所述。
   - 虽然一些人强调了其在 AI 深度方面的优势，但另一些人则对集成难度表示担忧。
- **以太坊 Spectra 升级受到关注**：社区分享并讨论了最近的 **Ethereum Spectra 升级**，指出了其对网络架构的关键改动和增强。
   - 对话集中在对去中心化应用（**dApps**）和更广泛的 Ethereum 生态系统的潜在影响上，这些内容在 [Perplexity AI 页面](https://www.perplexity.ai/page/ethereum-pectra-upgrade-a-comp-.mLTziZzT3Sn5xWmFa50hA)进行了总结。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF 用户应对 Llama 3.2 的 VRAM 需求**：Hugging Face 用户正在讨论本地模型的使用，指出 **Llama 3.2** 模型需要 **20GB VRAM**，并正在探索如 [EasyQZ](https://easyqz.online) 等量化方案，以便在低 VRAM 的 GPU 上运行更大的模型。
   - 用户正在交流各种模型的性能指标，其中一个 **9B 模型**展现出了令人印象深刻的能力，其表现优于许多尺寸更大的模型。
- **Science News Bot 加入战场**：**Science News Bot** 已在 [GitHub](https://github.com/AstraBert/SciNewsBot) 上发布，旨在提供最新的科学进展和研究。
   - **Mini PyTorch** 团队为 PyTorch 爱好者和开发者发布了一个轻量级替代方案，可在 [GitHub](https://github.com/karam-koujan/mini-pytorch) 上获取。
- **多语言 NER 模型展现潜力**：**GLiNER-MoE-MultiLingual** 模型是一个零样本（zero-shot）命名实体识别（NER）模型，展示了先进的 NER 能力。该模型使用 **NOMIC-MOE 架构**训练了一个 epoch，可通过[此处](https://huggingface.co/Mayank6255/GLiNER-MoE-MultiLingual)访问。
   - 优化后的 **SwarmFormer** 现在具备局部窗口注意力（local windowed attention）和 token-to-cluster 门控功能，增强了移动端和边缘环境的计算效率。
- **差分隐私受到关注！**：一位成员发布了关于**差分隐私**（Differential Privacy）的系列博客，首篇入门文章名为《差分隐私！！但为什么？》，点击[此处](https://theailandscape.substack.com/p/differential-privacy-but-why)阅读。
   - 该系列强调了在各种场景下，从数据中提取洞察与保护**个人隐私**之间的关键平衡。
- **HF 课程参与者期待新单元**：Hugging Face Agents 课程的参与者庆祝完成了 **Unit 1** 并获得了证书，而新学习者则在寻求通过 [Hugging Face 官方页面](https://huggingface.co/learn/agents-course/unit1/introduction)访问课程材料的指导。
   - 社区成员提议组建学习小组，奖金材料及后续单元预计将定期发布。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布 1.3.4 补丁**: **Windsurf 1.3.4** 补丁已发布，修复了 Cascade 写入工具中的 bug 并解决了消息取消问题。详情请参阅 [changelog](https://www.codeium.com/changelog)。
   - 关键修复包括更好的 **Cascade Base** 额度处理和更清晰的身份验证登录错误消息。
- **Model Context Protocol 获得 Cascade 支持**: **Cascade** 现在支持 **Model Context Protocol (MCP)**，允许对已配置的 **MCP** 服务器进行工具调用，用户应使用锤子图标进行设置。
   - 此功能适用于所有个人计划，每次 **MCP** 工具调用消耗一个 flow act。
- **JetBrains 扩展用户翘首以盼**: 用户希望更新 JetBrains 扩展，特别是刷新 **DeepSeek** 和 **Gemini 2.0 Flash** 等工具的模型列表。
   - 一名开发者回应称更新即将发布，并强调由于企业用户的需求，Codeium 扩展正在进行持续维护。
- **Windsurf 引发对 Codeium 扩展的不满**: 用户对 Windsurf 掩盖了原始 **Codeium 扩展** 感到不满，感叹两者缺乏功能对等。
   - 一些用户感到非常沮丧，正在寻找 **Qodo** 等替代方案。
- **Cascade 的自动保存：是意外的救星还是阴险的破坏者？**: 报告显示 Cascade 会在不询问的情况下自动应用更改，导致对代码编辑产生困惑。
   - 建议的解决方法是开启新的对话历史记录以避开自动建议。 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen MLX 模型引发困扰**: 用户报告在 Macbook Ultra 上加载 **Qwen MLX 模型** 时出现 `ModuleNotFoundError`，建议更新到 `mlx-engine/0.6.0` 可能会有帮助。
   - 尽管尝试了修复问题（包括删除特定目录），但即使在最新版本中，问题依然存在。
- **RX580 性能乏力**: RX580 GPU 在运行 **LLM** 时面临限制，建议转向支持 **ROCm** 的较新模型以获得更好的性能。
   - 鼓励用户使用 vast.ai 等云服务来评估 GPU 模型的性能，而无需进行大量的期初投资。
- **DeepSeek 被贴上 CCP 项目标签**: 社区对 **DeepSeek** 是否为 CCP 项目及其在 AI 技术快速进步背景下的资金来源进行了轻微辩论。
   - 小组推测 AI 进步的速度可能会带来意想不到的突破。
- **API Key 问题困扰本地服务器用户**: 用户对 **LM Studio** 中的本地服务器 API 需要 **OpenAI API key** 表示沮丧。
   - 聊天参与者建议通过修改代码来有效绕过本地设置的 **API key** 要求。
- **4060 Ti 表现出色**: **RTX 4060 Ti** 在高效处理 AI 模型方面表现出色，特别是在每秒 **token** 数方面。
   - 对比显示 **RTX 3090** 的表现也令人赞赏，特别是运行大型模型时。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 3 热度持续升温**：爱好者们对即将发布的 **Grok 3** 充满期待，并对其能力和性能提升进行了推测；有报道称 Grok 3 已开启早期访问，其中包括名为 *Ara* 的语音模式。
   - 一位用户推测了潜在的改进，称 *“如果 Grok 3 强 10 倍，那就游戏结束了”*，同时引用了 [Sam Altman 的推文](https://x.com/sama/status/1890816782836904000)，该推文预告了 **ChatGPT-4o** 的改进。
- **R1 与 O3 Mini High 争夺推理之王**：用户正在权衡 **DeepSeek R1** 和 **O3 Mini High**，发现 R1 在推理能力上更受青睐；讨论中提到了 [Nebius AI Studio 的推文](https://x.com/nebiusaistudio/status/1890397790250893743)，称 **DeepSeek R1** 的性能端点达到了 *60+ tokens/second*。
   - 反馈表明，虽然 **R1** 在某些推理任务中表现出色，但 **O3 Mini High** 在各种场景下提供了更好的一致性。
- **RAG 在 Repomap 方面遭遇瓶颈**：参与者正在努力解决 **RAG** 和 **Repomap** 在庞大仓库中的效率问题，观察到随着代码库的膨胀，收益正在递减；一个建议是使用 **BM25** 和 **RAG** 的混合评分来增强搜索结果。
   - 讨论涉及了一篇关于 [针对 1 万个仓库规模的代码库构建 RAG](https://www.qodo.ai/blog/rag-for-large-scale-code-repos/) 的博客文章，概述了弥补 LLM 上下文差距的方法。
- **Gemini Flash 2.0 的速度令人惊喜**：轶事证据表明，与 Sonnet 相比，**Gemini Flash 2.0** 提供了更快、更具成本效益的解决方案，且推理和数学技能出奇地强大。
   - 这一发现引发了用户在评估不同 AI 模型的性能和效率时的比较与讨论。
- **Aider 遭遇棘手 Bug**：在最近的更新后，**Aider** 用户报告了一些 Bug，特别是命令解析障碍和 `.dockerfile` 处理问题。
   - 社区还对 **DeepSeek** 中潜在的审查制度表示担忧，这受到训练数据和 Web 界面设置的影响。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 潜在的政治倾向引发辩论**：用户担心由 Elon Musk 支持的 **Grok** 可能会表现出反映其政治观点和在 Twitter 上行为的偏见，强调了 AI 进行合理审查以及避免政治叙事的必要性。
   - 讨论涉及了 **AI 是否应该影响政治观点**，一些人希望 **Grok 3** 能在与现有 AI 的竞争中表现出色，而另一些人则担心它可能会根据其编程成为宣传工具。
- **AI 生成的代码引发安全警报**：用户强调了依赖 **AI 生成的代码** 的风险，因为存在不安全字符串拼接等潜在安全漏洞，强调了开发者进行严格代码审查的重要性。
   - 讨论强调，如果未经人类专家彻底审核，**AI 糟糕的编码实践可能会给应用程序引入漏洞**。
- **GPT Store Action 需要隐私政策**：一名成员报告在尝试在 **GPT Store** 发布时遇到错误，提示 **“Public actions require valid privacy policy URLs”**（公开 Action 需要有效的隐私政策 URL）。
   - 另一名成员建议，填写 Action 中的隐私政策字段可以解决此问题。
- **多 Agent 系统规避 Token 限制**：成员们讨论了 AI 交互中 **token output streams** 的限制，分享了这对其项目影响的挫败感和经验。
   - 有建议称，使用 **multi-agent** 设置可能有助于在生成较长输出时规避 Token 限制，同时注意到 **MCP (Model Context Protocol)** 在简化 API 交互方面的潜力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD 1.7.0 优于 1.10**：用户报告称 **Stable Diffusion 1.7.0** 比 1.10 更高效，具有更快的加载速度和改进的提示词遵循度（prompt adherence）。
   - 相比之下，用户发现 1.10 版本经常生成不令人满意的图像，理由是加载时间更长且结果不稳定。
- **披露最佳 LORA 训练方案**：为了获得更好的 **LORA training** 效果，根据用户的讨论，建议针对目标主体准备约 **50-150 张图像**。
   - 社区还推荐使用 **Koyha LORA** 等工具来简化特定风格的训练过程。
- **本地 AI > 在线 AI？**：用户主张**在本地运行 AI 模型**，强调与在线服务相比，本地具有更大的控制权和自定义空间。
   - 他们建议，为了获得最佳性能，需要配备至少 **8GB VRAM** 的 NVIDIA GPU 和 **32GB RAM**。
- **精通图像生成**：讨论涵盖了有效的图像创建方法，包括使用**区域提示词（regional prompting）**和优化提示词中的词序以获得更好的结果。
   - 强调提示词结构的清晰度是获得卓越图像结果的关键因素。
- **ComfyUI：强大但复杂**：虽然承认 **ComfyUI** 对新用户来说可能比较难上手，但社区认可其在图像生成工作流中的灵活性和广泛功能。
   - **拖放式图像上传**等便捷工具简化了 AI 生成任务。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 性能大幅下滑**：用户报告称 **Claude** 的性能显著变慢，且过去几天其上下文感知能力有所下降，导致一些人认为它被“削弱（nerfed）”了。一位用户建议尝试 Anthropic 新的 **Thinking** 模型，并在模型设置中开启 *Thinking* 以查看 **Claude** 的思考过程，详见[此推文](https://x.com/btibor91/status/1890703686281867744?s=46)。
   - 成员们表示沮丧并开始探索替代模型，并指出根据[此 Reddit 帖子](https://www.reddit.com/r/ClaudeAI/s/VhbuwHrwem)，它可能在没有正式模型更新的情况下经历了内部调整。
- **MCP Server 配置引发创新**：用户分享了各种 **MCP server** 配置，强调使用 **Brave Search** 和 **Sequential Thinking** 来提高性能，详见[此 YouTube 视频](https://youtu.be/RCFe1L9qm3E?si=V4v8Y8XT1MkhKx_O)。据报道，这些配置增强了编码过程的可靠性和文档记录。
   - 一位用户还在寻求 **MCP** 的基础“**hello world**”示例，表明对更易获取资源的需求，另一位用户分享了[关于 Tavily MCP 的链接](https://youtu.be/jUmUxtvZFIE?si=xhso29DQNkY0RFra)。
- **Grok 3 的发布引发好奇**：**Grok 3** 的发布引起了兴趣，用户对其与现有模型的性能对比感到好奇，并有推测称后端更新可能会使其与更新的 **Llama** 版本对齐。
   - 在经历了之前令人失望的发布后，用户表达了测试 **Grok 3** 以验证其能力的渴望。关于推出的新闻在[此推文](https://x.com/kimmonismus/status/1891590879430754550)中进行了讨论。
- **Cursor 的任务处理面临审查**：人们对 **Cursor** 高效管理大量任务和处理大型文件的能力提出了担忧，特别是在 Agent 模式下，即使是拥有无限 token 的用户也是如此。由于这些限制，用户可能需要考虑替代方案。您可以通过[此链接](https://docs.cursor.com/settings/models)查看模型。
   - 一位用户分享了 [PearAI](https://trypear.ai/)，这是一个开源的 AI 驱动代码编辑器，作为一种可能的替代方案。另一位用户分享了[一条测试多个 AI 工具的推文](https://x.com/lukas_troup/status/1889698021371224148?s=46)，用于将简单的 Figma UI 设计转换为代码。
- **免费试用漏洞令 Cursor 用户担忧**：用户担心可能存在的漏洞允许滥用 **Cursor** 的免费试用，引发了关于安全和监管的对话。如[此推文](https://x.com/iBigQiang/status/1890993390172406160)所示，有一个使用 Cloudflare 无限续订 Cursor 的教程和脚本。
   - 社区对这些问题感到沮丧，敦促 **Cursor** 团队对滥用行为保持警惕，一位用户分享了一个用于自动登录 Cursor 的 [GitHub](https://github.com/chengazhen/cursor-auto-free) 仓库。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的实时图表获得更新**：得益于 **Google AI 的 Vertex** 增强功能，OpenRouter 的 throughput（吞吐量）和 latency（延迟）图表现在可以实时更新，详情见其 [公告](https://x.com/OpenRouterAI/status/1891510121139769542)。
   - 他们还发布了 **llms.txt**，这是一个用于与文档对话的全面资源（[llms.txt](https://openrouter.ai/docs/llms.txt) 和 [完整版](https://openrouter.ai/docs/llms-full.txt)）。
- **AI Tooltip 扩展程序诞生**：**Versatile AI Tooltip** Chrome 扩展程序允许用户使用 **OpenRouter** 模型快速处理文本，只需配置 API key 即可。
   - 它专注于以低廉的价格总结文章和翻译文本片段，详情可在 [Chrome Web Store](https://chromewebstore.google.com/detail/versatile-ai-tooltip/jdldlpaafnfeggaopbmelkmgagikfekm) 查看。
- **Toledo1 提供按需 AI 服务**：**Toledo1** 平台提供基于 **按提问付费（pay-per-question）** 模式的私有 AI 助手对话，结合了多个 AI 以确保准确性。
   - 它具有易于安装的客户端搜索功能，通过 [Toledo1](https://toledo1.com/) 提供企业级安全性，且无需订阅费。
- **DeepSeek R1 的推理表现引发关注**：用户注意到 **DeepSeek R1** 免费模型存在不一致性，推理 token（reasoning tokens）意外地出现在响应内容中，而不是作为单独的字段。
   - OpenRouter 团队正在跟踪管理开源推理模型行为的解决方案，使用建议可在 [DeepSeek 的 GitHub](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations) 上找到。
- **数据隐私成为棘手问题**：有人担心强制路由到特定国家可能违反数据保护法，因此主张提供基于区域的路由选项以满足合规性。
   - OpenRouter 承认了这一需求，并正在探索支持欧盟（EU）特定路由的选项，以实现更好的法律合规性，特别是针对目前围绕 [OpenAI SDK](https://openrouter.ai/docs/community/frameworks#using-the-openai-sdk) 的讨论。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **马斯克的 Grok 3 引发争论**：Elon Musk 宣布将发布 **Grok 3** 并安排了现场演示，称其为*“地球上最智能的 AI”*，这引发了关于其是否符合用户实际需求以及其“**based**”特性的复杂反应。现场演示定于**太平洋时间周一晚上 8 点**进行。
   - 一些人担心该模型针对文化偏见进行了过度调整；此外，OpenAI 董事会拒绝了马斯克 **974 亿美元**控制该公司的出价，并表示：*“OpenAI 是非卖品。”* 据 [Bloomberg](https://www.bloomberg.com/news/articles/2025-02-14/openai-board-rejects-musk-s-97-4-billion-bid-to-control-company?srnd=undefined) 报道，董事会强调他们一致拒绝了马斯克破坏竞争的企图。
- **Mistral Saba 旨在打造区域性 AI**：**Mistral Saba** 是一个 24B 参数模型，专为中东和南亚设计，强调通过定制训练能力来提升语言表达、细微差别和文化背景。 [Mistral AI](https://mistral.ai/en/news/mistral-saba) 将其作为服务于特定地理区域、市场和客户的众多定制训练模型之一推出。
   - 这一举措凸显了为特定区域市场量身定制 AI 模型而非依赖通用模型的趋势，展示了区域细微差别在专业化应用中变得越来越重要。
- **腾讯混元 LLM Turbo-S 即将亮相**：腾讯计划公开发布其 **Hunyuan** LLM Turbo-S 和视频生成模型 **HunyuanVideo I2V**，预计于 2025 年第一季度发布，正如 [青龍聖者在推特上发布的消息](https://x.com/bdsqlsz/status/1891507545325404639) 所述。
   - 这一公告反映了腾讯在竞争激烈的 AI 领域（尤其是视频技术领域）加强地位的雄心，有可能重塑媒体和娱乐领域的 AI 应用。
- **GRPO 与 SFT 搭配使用！**：成员们注意到 **GRPO** 并非旨在取代 **SFT**；相反，两者应结合使用以获得最佳效果。[Trelis Research 在推特上发布了关于 GRPO vs SFT 视频的预告](https://x.com/TrelisResearch/status/1890445776137969710)。
   - 这种模型训练的协作方法受到希望通过利用 **GRPO** 和 **SFT** 技术各自优势来最大化模型性能的用户的青睐。
- **LLaDA 挑战自回归模型**：论文介绍了 **LLaDA**，这是一种扩散模型，通过展示其在可扩展性和指令遵循能力方面的潜力，挑战了传统的自回归模型（ARMs），详见 [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)。
   - **LLaDA** 的引入引发了 AI 社区对其作为扩散模型分类的讨论和审查，观察结果指出其缺乏传统的扩散特征。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Data URIs 激发 AI 图像生成讨论**：一位成员探索了训练 AI 使用 **Data URIs** 或 **Base64** 编码生成图像，允许在不使用外部链接的情况下存储图像，但指出这种方法在用户界面中通常显示为 **blob**。
   - 讨论强调了在使用 **Data URIs** 处理 AI 生成图像时，存储便利性与显示质量之间的权衡。
- **神童现象启发 AI 进步**：关于数学和国际象棋等领域常见**神童**的讨论引发了关注，暗示这些领域可能是 AI 取得进步的希望之地。
   - 另一位成员推测**音乐**可能也符合这种模式，并质疑为什么它没有被归类为“简单”领域。
- **NovelAI 揭秘采样器奥秘**：NovelAI 对**采样方法（sampling methods）**的理解得到了进化，因为他们决定提供多种**采样器算法（sampler algorithms）**，从而促使开发人员弄清楚其中涉及的复杂性。
   - 这与其他组织形成了对比，那些组织最初不提供复杂的采样器，因此缺乏投入大量精力去理解它们的动力。
- **重复惩罚（Repetition Penalty）依然棘手**：文本生成中的**重复惩罚**是一种统计方法，用于缓解水平较低的作者常见的过度重复问题。
   - 虽然优秀的作者可以成功地利用重复来达到戏剧效果，但在开发**采样策略（sampler policies）**时，区分好的重复和坏的重复仍然是一个挑战。
- **DEQs 的深度挑战 RNN 的时间维度**：关于 **DEQs** 的讨论强调了它们与循环结构的关系，重点关注**权重共享（weight tying）**的影响以及隐藏层收敛到**不动点（fixed points）**的潜力。
   - 参与者指出，与处理时间变化的 **RNN** 不同，**DEQs** 强调深度，并可以使用隐式微分（implicit differentiation）方法进行反向传播。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes 在推理方面表现出色**：用户报告称 **DeepHermes** 模型比目前的 R1 蒸馏模型（distillates）更好地遵循指令，并被认为是第一个可用的通用推理模型。根据 [VentureBeat](https://venturebeat.com/ai/personalized-unrestricted-ai-lab-nous-research-launches-first-toggle-on-reasoning-model-deephermes-3/) 的报道，一名 DeepHermes-3 用户在 MacBook Pro M4 Max 消费级硬件上报告的处理速度为每秒 28.98 个 tokens。
   - 有建议认为推理任务和数据集可以进一步增强这一性能。
- **呼吁开源旧模型**：针对 **Anthropic** 可能删除 **Claude 3** 且不计划将其开源的决定，人们表达了担忧，尽管该模型已经发布一段时间。有人认为发布旧模型可以建立良好声誉，并有利于公司在开源社区中的名声。
   - 成员们提到了 Hugging Face 首席执行官 Clement Delangue 之前的工作，他在[这段 Youtube 视频](https://www.youtube.com/watch?v=ry_SthLfTa8)中讨论了 OpenAI、DeepSeek 以及他公司的创新。
- **强化学习 (RL) 助力 LLM**：分享了关于 **Reinforcement Learning (RL)** 和奖励机制如何有效增强 LLM 的见解，特别是在对齐（alignment）工具方面。历史资料表明，早在其他公司采用类似策略之前，研究人员就已经意识到了这些技术。
   - Google 首席科学家 Jeff Dean 在[这段视频](https://www.youtube.com/watch?v=v0gjI__RyCY)中讨论了他在 Google 的 25 年历程，从 PageRank 到 AGI。
- **关于模型训练成本的辩论**：一场关于训练 **1B model** 所需成本的讨论浮出水面，完整配置的估计费用从**数千到数万美元**不等。虽然在消费级 GPU 上进行训练是可行的，但数据和架构的进步可能会随着时间的推移影响整体成本效益。
   - 一位参与者提到，通过精细的数据选择和训练策略，有可能以大约 **$200 到 $1,000** 的价格实现一个 **1.5B model**，并链接到了 [1.5-Pints 技术报告](https://arxiv.org/html/2408.03506v1#:~:text=Using%20our%20pre,16K%20context%20window%20version%2C%20which)。
- **LLaDA 扩散模型挑战 LLM**：[LLaDA 论文](https://arxiv.org/abs/2502.09992)介绍了一种扩散模型，通过为概率推理提供一种有原则的生成方法，重新定义了大语言模型 (LLM) 的格局。*LLaDA* 展示了强大的可扩展性，并可与领先的 LLM 竞争。
   - LLaDA 在 In-context learning 和多轮对话中的卓越表现，使其成为 **GPT-4o** 等成熟模型的竞争替代方案。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Splines 在 AI 中展现出潜力**：讨论强调了在 AI 模型中使用 **NURBS** 和 **splines** 进行增强函数逼近并减少过拟合的潜力，这与传统的多项式形成对比。更多研究可见于 [NeuralSVG: An Implicit Representation for Text-to-Vector Generation](https://sagipolaczek.github.io/NeuralSVG/)。
   - 参与者探讨了在开发新的 AI 方法论中光滑度和拓扑结构的重要性，旨在实现更好的泛化能力。
- **UltraMem 提升推理速度**：在论文 [Ultra-Sparse Memory Network](https://arxiv.org/abs/2411.12364) 中详细介绍的 **UltraMem 架构**，利用大规模、超稀疏记忆层来提高推理速度和模型性能。
   - 该架构在保持计算效率的同时，可能比传统的 MoE 方法更具优势。
- **社区寻求论文讨论主持人**：成员们正积极鼓励更多人参与主持论文讨论，以丰富小组内的观点。物流细节在 #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1340233356553617429) 中讨论。
   - 建议包括开发一个带有依赖链接的**经典论文**层次树，以简化理解和知识进阶。
- **Mistral 的 Saba 支持多种语言**：MistralAI 推出了 [Mistral Saba](https://x.com/sophiamyang/status/1891487141718376580)，这是一个拥有 240 亿参数的区域语言模型，在来自中东和南亚的数据集上进行了训练。
   - 该模型在**阿拉伯语**以及**泰米尔语**和**马拉雅拉姆语**等南印度语言方面表现出色，专注于区域语言细微差别。
- **OpenAI 起草机器人伦理指南**：OpenAI 发布了其 [机器人 50 定律](https://sherwood.news/tech/step-aside-asimov-here-are-openais-50-laws-of-robotics/)，定义了安全的 AI 行为和现代 AI 伦理准则，扩展了阿西莫夫最初的三大定律。
   - 该文件旨在使 AI 模型与人类价值观和安全保持一致，为正在进行的 AI 责任讨论做出贡献。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 可能获得 OpenAI 兼容性**：一名成员询问了 **Cohere API** 与 **OpenAI** 的兼容性，得到了反馈并被转介给产品团队，目前尚未给出明确的时间表。
   - 另一名成员正在构建一个兼容各种 AI 模型的**深度研究克隆版**，仅需基础 URL 和模型规范即可进行 LLM 提供商初始化。
- **社区辩论审核模型**：成员们辩论了审核模型的有效性，对 **Llamaguard** 表示不满，并为他们的特定用例寻找更好的替代方案。
   - 一些人正在使用 **OpenAI 的 omni 审核**，但希望 **Cohere** 能尽快发布新模型。
- **排除技术故障**：一名成员报告在全新安装后 **cohere** 包出现 **ModuleNotFoundError**，建议认为是环境问题。
   - 另一名用户报告了登录错误并分享了截图，鉴于错误上下文不明确，期待社区协助。
- **Aya-8b 集成遇到问题**：一名将 **Cohere Aya-8b 模型**集成到其应用程序中的成员遇到了该特定模型版本的 **API 支持**问题。
   - 分享了一个使用 **c4ai-aya-expanse-8b** 模型的请求示例，并参考了相关资源进行排查，而其他成员则在 **Rerank API** 上遇到了**超时问题**。
- **Embed API 速率限制**：Embed API 的生产环境速率限制为**每分钟 2,000 次调用**，无论嵌入的 token 数量多少，这实际上将其限制为**每分钟总共 2,000 个文档**。
   - 用户幽默地测试了 AI 对颜色的偏好，但 AI 表示它**没有个人偏好**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Perplexity 推出 Deep Research Agent**：新的 [Perplexity Deep Research Agent](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) 提供免费层级和 **每月 20 美元** 的专家研究员选项，约三分钟即可生成报告。
   - 用户反馈指出输出尚可，但建议后续功能可以更好地 *“利用深度研究 (leverage deep research)”*，仍有进一步改进的潜力。
- **马斯克预告 Grok 3 发布**：Elon Musk 宣布即将通过现场演示发布 **Grok 3**，并大胆宣称其为 *“地球上最聪明的 AI”*，但外界仍持怀疑态度。
   - 社区成员反应不一，在保持热情的同时，考虑到竞争格局和 **Grok** 之前的性能问题，也表达了审慎态度。
- **StepFun 开源多模态模型**：StepFun 开源了 **Step-Video-T2V**（一个需要 80G VRAM 的 30B 文本转视频模型）以及他们的 **Step-Audio-Chat** 模型，后者已在 [Hugging Face](https://huggingface.co/stepfun-ai/Step-Audio-Chat) 上线。
   - 评估显示该 *音频聊天模型表现远超竞争对手*，突显了在 **multimodal AI** 领域的增强能力。
- **Zed 预测下一次编辑**：Zed 推出了一款名为 **Zeta** 的新 [编辑预测模型](https://zed.dev/blog/edit-prediction)，旨在通过预测代码编辑来提升开发者生产力。
   - 尽管社区赞赏开源举措，但也有人质疑 **Zeta** 在性能上是否能与 **Cursor** 和 **Copilot** 等成熟工具抗衡。
- **Eliza 引发怀旧**：成员们回顾了 **Eliza** 作为早期 AI 治疗师的影响，并讨论了相关的 [播客剧集](https://corecursive.com/eliza-with-jeff-shrager/)。
   - 参与者推测了如果它在今天发布可能产生的影响，并引用了 Gary Marcus 的评论，假设其能获得 **1000 亿美元融资**。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 服务器增强工作流**：成员们讨论了多个 MCP 服务器，包括用于 **顺序思维 (sequential thinking)**、**文件系统访问**、**Web 搜索** 和 **GitHub 仓库评估** 的服务器。
   - 这些工具提供了增强功能，以简化各种任务并提高效率。
- **Glama 网站遭遇故障**：用户报告由于其托管地 **IAD** 地区的网络中断，导致无法访问 *Glama* 网站。
   - 包括加载缓慢和超时在内的问题最终得到了解决。
- **分享 Prompt 工具推荐**：一位用户在 MCP 环境中寻求 Prompt 管理工具的建议；推荐包括 **LLMling**、用于文档的 **MCP-llms-txt** 以及 **Langfuse**。
   - 这些工具旨在协助更好的 Prompting 和工作流维护。
- **SSE Wrappers 开发挑战**：开发者讨论了为 MCP 服务器创建 **SSE wrappers** 的挑战，特别是关于在不修改现有服务器代码的情况下进行通信的问题。
   - 分享了 [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) 和 [mcp-sse](https://github.com/sidharthrajaram/mcp-sse) 等资源，尽管一些人对连接稳定性和中间件集成表示沮丧。
- **Neon 的 MCP 服务器简化数据库管理**：来自 Neon 的 Daniel 介绍了 [Neon 的 MCP 服务器](https://neon.tech/guides/neon-mcp-server)，允许用户通过 [Neon API](https://api-docs.neon.tech/reference/getting-started-with-neon-api) 使用自然语言管理数据库。
   - 这通过使用 Large Language Models (LLMs) 进行交互，简化了数据库工作流。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 可能简化 NumPy**：一位成员推测，通过将工作外包给 **MAX**，用 **Mojo** 编写的 **NumPy** 将比 **C** 语言版本更简单。
   - 另一位成员表示赞同，认为这种方法可以显著提高效率和可维护性。
- **社区关注 Polars**：一位用户请求获取 **Polars** 的 **Discord 链接**，这是一个由 **Rust** 驱动的 DataFrame 库。
   - 快速的响应确认了 **Polars** 在 **Mojo** 社区中的相关性和现有知名度。
- **Mojo 关注 GPU，包括 Metal**：讨论围绕 **Mojo** 为 **GPU** 编译代码的计划展开，并可能支持 **Apple** 的 **Metal API**。
   - 最终策略包括为包括 **Apple Silicon** 在内的各种硬件实现 **MAX 驱动程序**，从而为更广泛的兼容性打开大门。
- **AI 助手加入 Mojo 重构**：一位用户探索了使用 **AI** 将大型 **Python** 项目重构为 **Mojo**，特别是针对小型类，并利用其合理的 **Mojo** 知识。
   - 另一位成员建议尝试 `Gemini 2.0 Flash` 变体，但提醒完全自动化该过程将非常困难。
- **Mojo 需要更清晰的错误提示**：用户强调需要提高 **Mojo 错误消息** 的清晰度，以帮助理解该语言。
   - 一位成员指出，在从其他语言迁移代码时，**Mojo** 的 **borrow checker**（借用检查器）增加了额外的复杂性，通常需要完全重新构建架构。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FSDP 替代 Deepspeed Zero**：一位成员询问如何在 **Torchtune** 中使用 **Deepspeed Zero**，但 *ebsmothers* 澄清说他们使用的是 **FSDP**，它等同于 **ZeRO-3**。
   - 另一位成员建议使用 **distributed recipes**，无需额外设置即可获得与 **Zero** 相关的能力。
- **GRPO PR 需要一些维护**：参与者讨论了 **GRPO PR** 的进展，指出需要单元测试和一个实验性组件文件夹来存放各种数据集（参见 [PR #2398](https://github.com/pytorch/torchtune/pull/2398)）。
   - 重点放在了在进行生成更改时保持向后兼容性。
- **Tool 角色进入 Llama3.3**：大家达成共识，在 **Llama3 tokenizer** 中添加 “tool” 选项，提供与 “ipython” 的向后兼容性（参见 [tokenizer.py](https://github.com/Ankur-singh/torchtune/blob/f386ff99f8c25f758cc1f3c88a28fac03513f497/torchtune/models/llama3_3/_tokenizer.py#L13)）。
   - 贡献者们积极辩论了引入新 tokenizer 构建器的影响，并细致地处理模型版本检查。
- **依赖问题导致开发环境膨胀**：成员们对开发环境中不断增加的可选依赖列表表示担忧，特别是与日志框架相关的依赖。
   - 有人提议对依赖项进行分类，使用户能够仅安装所需的日志框架，从而最大限度地减少不必要的臃肿。
- **裁剪/缩尾处理（Winsorization）挽救 RLHF**：建议在 `rlhf.get_batch_log_probs` 函数中添加裁剪（clipping）或缩尾处理（winsorization）选项，以解决由于对数行为导致的丢弃序列结束（EOS）标记的问题。
   - 这被认为是处理对数概率（log probabilities）方面的一个潜在改进。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor 操作更加灵活**：成员们建议 `Tensor.tril` 和 `triu` 应该接受 `Tensor` 而非 `int` 作为对角线参数，以便在 KV-cache 空间的分块注意力（blockwise attention）自定义 Kernel 中提供更大的灵活性。
   - *psychofauna* 提倡使用 `int | Tensor` 的理想签名，以简化跨维度的多条对角线管理。
- **Tinygrad Bug 修复悬赏启动**：一位用户发布了一个 [Bug 修复悬赏的 PR](https://github.com/tinygrad/tinygrad/pull/9110)，并根据他们对测试失败的分析指出该任务非常简单。
   - 他们认为相关的测试可能代表了一个修复或拼写错误，尚待对近期提交（commits）进行进一步审查。
- **寻找 HEVC 解码器工具**：继 HEVC 解码器之后，一位用户询问了用于命令队列管理的工具，并提到之前使用的 ioctl 嗅探器（sniffer）非常有帮助。
   - *nimlgen* 提到存在用于转储队列的代码，尽管可能需要针对当前用途进行调整。
- **Attention 实现接近完成**：一位成员报告称，通过改编 `extra/models/llama.py` 中的现有代码，并参考顶尖的 Hugging Face 仓库进行评估，已实现了完整的 `com.microsoft.Attention`。
   - 该实现在 **250** 个测试中通过了 **201** 个，失败主要是由于各种量化格式下的数值不准确。
- **Tinychat 针对移动端强化**：一位用户宣布了对 Tinychat 的增强，特别针对移动设备上 WASM 的稳定性，相关更改已记录在他们的 PR 中。
   - 他们目前正在清理代码以准备合并，并欢迎就这些更新进行咨询。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 全面揭秘！**：@mesudarshan 展示了 **LlamaParse** 的功能，强调了其**多种解析模式**以及**解析音频/图像**的能力，并进一步增强了 **JSON 模式**，[详见此处](https://t.co/gFTggKKbaD)。
   - 他在[此链接](https://t.co/YH1g3x5Yfc)中对各项功能进行了*全面分解*。
- **LlamaIndex.TS 瘦身！**：**LlamaIndex.TS** 获得了更新，通过瘦身使其*更易于交付*，极大地提升了可用性，改进详情[见此处](https://t.co/9psfaHw7ZN)。
   - 该版本让开发者能够更快地将 LlamaIndex 集成到基于 Typescript 的 LLM 应用中。
- **使用 LlamaIndex 部署销售外联 Agent**：分享了一个**自动化销售外联 Agent**教程，该 Agent 利用 **@llama_index workflows** 撰写外联邮件并根据回复管理会议日程，演示见[此处](https://t.co/FbmOP69b2F)和[此处](https://t.co/9hHo1XXJBU)。
   - 该 Agent 帮助生成外联邮件并安排会议。
- **LLM 联盟（Consortium）问答！**：Massimiliano Pippi 引入了 **@karpathy** 的 **LLM 联盟**概念实现，允许通过多个 LLM 回答问题并比较答案，在[此处](https://t.co/iIbLjY7K23)和[此处](https://t.co/90tipH1h8P)有更充分的解释。
   - 该项目探索了*协作式 LLM 响应*。
- **Mistral Saba 小模型首次亮相**：**Mistral AI** 推出了 **Mistral Saba**，这是一个专注于阿拉伯语的新模型，**LlamaIndex** 立即提供了支持，用户可以使用 `pip install llama-index-llms-mistralai` 进行安装，[更多信息](https://t.co/bvuwqOWnOB)。
   - 它提供了一个*专门针对阿拉伯语的小模型*。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **寻找 **Nomic Embed Text V2** 源码**：一位成员在寻找 **Nomic Embed Text V2** 的源代码和训练数据，起初未果，直到另一位成员分享了 [GitHub repository](https://github.com/nomic-ai/nomic) 和 [Hugging Face page](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)。
   - 讨论强调了开源项目清晰文档和可发现性的重要性。
- **DeepSeek 模型绑定出现故障**：用户报告在加载 **DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf** 模型时出错，暗示可能是由于过时的 **python bindings** 导致的。
   - 另一位成员建议更新 **llama.cpp** 库以解决冲突。
- **LocalDoc 数据转储？**：用户询问在 GPT4All 中启用 **LocalDoc** 是否会将数据发送到云端，以及在初始使用后如何禁用它。
   - 解决方案包括取消勾选 **Enable Data Lake** 设置以防止数据共享。
- **Code Llama 模板纠纷**：用户请求协助配置 **Code Llama** 模型的聊天模板。
   - 分享了一个消息模板的基础示例，并强调可能需要模型的特定仓库或全名以获得更好的支持。
- **模型工具调用探索**：一位成员询问如何验证模型在执行过程中是否有效地使用了工具。
   - 给出的建议是咨询模型开发者的文档，以确认该模型是否专门针对 tool calling 进行了训练。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **伯克利 RDI 实习招募人才**：伯克利负责任去中心化智能中心 (RDI) 为加州大学伯克利分校的学生提供实习机会，重点关注市场营销和 Web 开发，申请通过 [此链接](https://forms.gle/1Px8Gh9kbBTmx7fg9) 和 `samanthaguo@berkeley.edu` 进行滚动审核。
   - 职责包括制定营销策略、管理社交媒体、增强网站以及为 YouTube 频道制作多媒体内容；优先考虑具备设计工具、GitHub 和多媒体制作技能的人选。
- **CS294/194-280 课程注册指南**：学生可以通过 CalCentral 注册 **CS194-280**（课程编号 **33840**）和 **CS294-280**（课程编号 **33841**），候补名单中的学生可使用 [申请表](https://forms.gle/sfWW8M2w1LDTnQWm9)。
   - 虽然接受逾期注册，但有关课程的问题应提交至 [Edstem](https://edstem.org/us/join/QMmJkA)，而不是通过电子邮件联系工作人员。
- **DeepSeek 推理讨论已安排**：举行了一场关于 **DeepSeek** 推理方法和 **GRPO** 的讨论，探讨如何将 GRPO 风格的推理集成到较小的模型中。
   - 学习小组还在每周论坛中讨论了 **Yu Su 教授** 的 **Lecture 3**。
- **FinRL-DeepSeek 网络研讨会首发 CVaR-PPO 扩展**：**2 月 25 日**举行的 **FinRL-DeepSeek** 网络研讨会将展示 **CVaR-PPO** 的扩展版本，该版本结合了由 **LLMs** 生成的交易建议和风险评估，并提供**开源代码**、**数据**和 **AI 交易 Agent**。
   - 将分享在 **Nasdaq-100** 上的回测结果。
- **Quiz 3 预计很快发布**：成员们询问了 **Quiz 3** 的发布日期，由于日程安排问题，该测验有所延迟。
   - 目前预计将在**下周初**发布。



---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Code Golf 竞赛启动**：成员们讨论了参与 **DSPy code golf**，建议围绕创建简洁的 **DSPy 解决方案**和分享快速技巧（quick hacks）开展竞赛。
   - 一位成员指出这是*花时间在有趣的编程挑战上的绝佳借口*，并链接了 [Omar Khattab (@lateinteraction)](https://x.com/lateinteraction/status/1890442615700545878) 展示 **DSPy** 用法的推文。
- **Qwen 0.5B 处理 JSON 输出**：一位成员寻求关于使用 **Qwen 0.5B** 确保 **多标签分类任务** 输出 **JSON** 的指导，重点关注必要的配置。
   - 另一位成员建议使用 **DSPy** 的 **JSONAdapter** 来进行正确的 **JSON 输出** 格式化，并提供了示例代码。
- **自定义 DSPy 实现出现**：一位成员分享了他们 fork 并定制的 **Parlant** 实现，增加了 **DSPy 支持**，确保了 **JSON 输出** 的正常工作。
   - 他们提供了一个可以集成到 **Parlant server** 中的示例运行，以获得最佳性能。
- **常量参数放置位置探讨**：有人请求澄清常量参数应该包含在 **signature docstring** 中，还是作为流水线设置中的 **InputField**。
   - 讨论包括了对 **mipro** 和 **bootstrapfewshotrandomsearch** 优化过程影响的考虑。
- **MIpro 反馈循环策略**：关于如何向 **MIpro** 传递错误答案反馈的问题被提出，而不仅仅是将其视为一个次要指标。
   - 该咨询寻求有效传达输出错误原因的方法，以提高模型的性能。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **FinRL-DeepSeek 网络研讨会发布**：一场关于 **FinRL-DeepSeek 算法交易** 的网络研讨会已宣布，展示了 **CVaR-PPO** 的扩展，该扩展集成了来自 **LLMs** 的交易建议和风险评估，计划于 **CST 时间 2 月 25 日晚上 8 点** 举行，[注册链接在此](https://melwy.com/registration/webinar)。
   - 研讨会承诺提供 **开源代码**、数据、交易 **Agent** 以及在 **Nasdaq-100** 上的回测结果，强调了可部署 AI **Agent** 的可用性。
- **关于研讨会录像的咨询**：一位成员询问如何获取 **FinRL-DeepSeek 网络研讨会** 的录像。
   - 这表明人们对研讨会内容有浓厚兴趣，特别是对于无法参加直播的人。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1340019471338700873)** (1141 条消息🔥🔥🔥): 

> `Unsloth 招聘挑战, AI 学习资源, 模型微调, 在 Hugging Face 上上传模型, 技术领域的个人经验`

- **Unsloth 招聘挑战赛发布**：Unsloth 发布了五项挑战，为合适的人选提供高达每年 $500K 的潜在薪资，重点考察各技术领域的解题能力。
   - 鼓励参与者解决与 Griffin、FSDP2 以及其他深度学习优化相关的难题，因为公司正寻求推动社区贡献。
- **AI 的有效学习资源**：用户建议了多种学习 AI 的方法，例如阅读源码、学术论文，以及利用 ChatGPT 等工具进行解释。
   - 推荐了包括 Karpathy 课程在内的多种课程，以及处理 AI 复杂课题的实践方法。
- **理解 Hugging Face 上的模型上传**：讨论了在 Hugging Face 上同时上传量化模型和原始模型权重的初衷，部分用户对资源占用表示困惑。
   - 用户注意到模型被修复或增强的情况，并强调优化原始模型是 Unsloth 的首要任务。
- **技术领域的经验与观点**：多位用户分享了他们在技术领域的个人和学术历程，并就热情和兴趣如何影响学习和职业路径发表了见解。
   - 讨论了努力与天赋的价值，突出了个人在工业界和学术界的经历。
- **Kernel 开发与学习 CUDA**：用户表达了对底层 Kernel 开发（特别是 CUDA 和 Triton）的兴趣，分享了教育背景，并讨论了实习和就业前景。
   - 关注点集中在 RL 和 GPU 优化方面的持续努力，一些用户寻求在毕业后转型进入工业界角色。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/phi4">使用 Unsloth 微调 Phi-4</a>：使用 Unsloth 微调 Microsoft 的新 Phi-4 模型！我们还发现并修复了模型中的 4 个 bug。</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度可增加 6 倍！</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">推理 - GRPO & RL | Unsloth 文档</a>：使用 Unsloth 通过 GRPO（强化学习 RL 微调的一部分）训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://x.com/danielhanchen/status/1891194528931209644">Daniel Han (@danielhanchen) 的推文</a>：我们设置了 5 个挑战，如果你获得 47 分，我们将提供 50 万美元/年 + 股权邀请你加入 🦥@UnslothAI！无需经验或 PhD。40 万 - 50 万美元/年：创始工程师（47 分）；25 万 - 3...</li><li><a href="https://huggingface.co/archit11/smollm350m-grpo">archit11/smollm350m-grpo · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=JJWvYQdOVOY">Llama 3.2 微调入门指南（支持 16k, 32k,... 上下文）</a>：学习如何在 Google Colab 中使用 Unsloth 微调 Meta 的 Llama 3.2 模型。本分步教程涵盖：• 从 Hugging Face 选择合适的 Llama 3.2 模型...</li><li><a href="https://x.com/Dorialexander/status/1885844180976431564">Alexander Doria (@Dorialexander) 的推文</a>：如果有人感兴趣，我成功在 Colab 上运行了 @willccbb 脚本的可用版本。将 Llama 1B 替换为 Qwen 0.5b，并使用 vLLM 进行推理。完整训练大约需要 2 小时。https://colab...</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识。</li><li><a href="https://github.com/Zyphra/Zonos/pull/102">文档：由 sardorb3k 在 README 中添加 training.py 使用说明 · Pull Request #102 · Zyphra/Zonos</a>：添加关于训练前提条件和设置的详细章节；记录配置选项和参数；解释 Checkpoint 保存和训练过程；包含自定义训练的示例命令。</li><li><a href="https://github.com/sardorb3k/Zonos">GitHub - sardorb3k/Zonos: Zonos-v0.1 是一款领先的开源权重文本转语音模型，在超过 20 万小时的多样化多语言语音上训练而成，其表现力和质量可与甚至超越顶尖的 TTS 提供商。</a>：Zonos-v0.1 是一款领先的开源权重文本转语音模型，在超过 20 万小时的多样化多语言语音上训练而成，其表现力和质量可与甚至超越顶尖的 TTS...</li><li><a href="https://docs.unsloth.ai/get-started/fi">Unsloth 文档</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1">unsloth/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1/tree/main">unsloth/DeepSeek-R1 的 main 分支</a>：未找到描述</li><li><a href="https://github.com/huggingface/trl/pull/2810">由 willccbb 提交的用于自定义多步 Rollout 的 GRPO 环境（仅限 vLLM）· Pull Request #2810 · huggingface/trl</a>：此 PR 的作用？在 trl/environments 下为 Environment 对象添加了一个协议，该对象封装了 vLLM 的 .generate(...) 以允许自定义 Rollout 逻辑，并向 Trai... 添加了一个可选的 env 字段。
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1340020309582938112)** (114 条消息🔥🔥): 

> `RAG 实现、开源工具、AI Agents、双 CPU 服务器搭建、Project Digits 与内存` 


- **探索 RAG 实现工具**：成员们讨论了实现 **RAG** 的选项，并建议使用 **Llama Index**、**Haystack** 和 **AnythingLLM** 来简化流程。
   - 有人指出，根据业务垂直领域和预期的不同，入门可能需要投入大量精力。
- **AnythingLLM 推广 Unsloth**：用户分享了使用 **AnythingLLM** 的积极体验，强调了它作为 RAG 应用无缝桌面工具的作用。
   - 其创始人因其出色的教育者身份以及在社区内有效推广 **Unsloth** 而受到赞赏。
- **围绕 AI Agents 的质疑**：关于 AI Agents 的真实定义引发了讨论，有观点认为它们可能只是有效的 Prompt，而非突破性技术。
   - 成员们对围绕 AI Agents 的炒作表示担忧，强调了实用性优于营销语言的重要性。
- **双 CPU 服务器配置见解**：一位成员寻求关于构建配备 1TB RAM 的双 CPU 服务器的建议，重点是使用一个 CPU 处理 GPU 任务，另一个处理数据处理。
   - 讨论中提到了维护此类设置的挑战，以及与运行特定 LLMs 相关的隐藏成本。
- **Project Digits 与内存的未来**：参与者讨论了 **Project Digits** 的潜力，该项目将专注于内存容量而非速度，并将其与 Apple 的芯片设计进行了比较。
   - 成员们预计内存技术的未来进步将实现更有效的 LLMs 微调和更广泛的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda">llama.cpp/docs/build.md at master · ggml-org/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/microsoft/markitdown">GitHub - microsoft/markitdown: Python tool for converting files and office documents to Markdown.</a>：用于将文件和 Office 文档转换为 Markdown 的 Python 工具。 - microsoft/markitdown
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1340005702483185674)** (305 条消息🔥🔥): 

> `CUDA OOM 问题、大模型微调、持续预训练、Unsloth 更新、问答模型格式` 


- **排查 CUDA Out of Memory 错误**：成员们讨论了在训练过程中遇到 CUDA Out of Memory (OOM) 错误的情况，特别是在休息一周后，并指出数据不平衡可能是原因之一。
   - 一些人指出，尽管使用了类似的设置，解决 OOM 问题仍然很困难，并指出即使是数据差异也可能导致 VRAM 负载激增。
- **在新数据上微调大模型**：参与者分享了在适配和微调 Llama 和 Mistral 等模型时的经验，有时会遇到意想不到的推理行为。
   - 对话强调了在训练中使用格式良好的数据集的重要性，特别是强调了问答格式的必要性。
- **持续预训练的注意事项**：讨论了为使模型适应新语言或任务而进行的持续预训练过程，并建议检查模型已学习的数据。
   - 成员们引导他人参考相关资源，例如特定的 Unsloth 文档页面，以获取最佳实践指导。
- **Unsloth 仓库更新**：对 Unsloth 仓库最近更新后可能出现的破坏性变更表示担忧，引发了关于同时更新 Unsloth 和 Torch 的讨论，成员们对性能的潜在影响持谨慎态度。
   - 建议实现 Flag 以防止在已有缓存的情况下重复下载模型，从而提升用户体验。
- **为模型训练格式化数据集**：一位用户询问了 Mistral 3 small 对话数据集的正确格式，表示对创建正确的模板感到困惑。
   - 这引发了关于准备微调数据集复杂性的对话，社区为解决模板相关问题提供了支持。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://ollama.com/download/mac">在 macOS 上下载 Ollama</a>: 为 macOS 下载 Ollama</li><li><a href="https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/">🐋 使用 Llama.cpp 在 Open WebUI 运行 DeepSeek R1 Dynamic 1.58-bit</a>: 特别感谢 UnslothAI 的卓越工作！得益于他们的努力，我们现在可以运行完整的 DeepSeek-R1 671B 参数模型的 dynamic 1.58-bit 量化版本（压缩至...）</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: 未找到描述词</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb#scrollTo=KN6nELjXcRez.">Google Colab</a>: 未找到描述词</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing">Google Colab</a>: 未找到描述词</li><li><a href="https://download.pytorch.org/whl/cu124">未找到标题</a>: 未找到描述词</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZ">Google Colab</a>: 未找到描述词</li><li><a href="https://colab.research.google.com/drive/1lXijH5SR5buvuAbWG9ip2TCLlrPSsQ9n?usp=sharing">Google Colab</a>: 未找到描述词</li><li><a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型 (GRPO)</a>: 你现在可以使用 Unsloth 100% 在本地复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth 需求 | Unsloth 文档</a>: 这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb">Qwen2.5-VL/cookbooks/ocr.ipynb at main · QwenLM/Qwen2.5-VL</a>: Qwen2.5-VL 是由阿里云 Qwen 团队开发的多模态大语言模型系列。 - QwenLM/Qwen2.5-VL</li><li><a href="https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-vllm">保存到 VLLM | Unsloth 文档</a>: 为 VLLM 将模型保存为 16bit</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">推理 - GRPO &amp; RL | Unsloth 文档</a>: 使用 Unsloth 通过 GRPO（强化学习 RL 微调的一部分）训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>: 关于对齐 smol 模型的课程。通过在 GitHub 上创建账号为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">安装 + 更新 | Unsloth 文档</a>: 学习在本地或在线安装 Unsloth。</li><li><a href="https://docs.unsloth.ai/basics">推理 - GRPO &amp; RL | Unsloth 文档</a>: 使用 Unsloth 通过 GRPO（强化学习 RL 微调的一部分）训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://github.com/shannonhochkins/ha-component-kit">GitHub - shannonhochkins/ha-component-kit: 为开发者设计，这个强大的包基于 React 构建，为你的 Home Assistant 智能家居仪表盘创建无缝、高度可定制的界面。</a>: 为开发者设计，这个强大的包基于 React 构建，为你的 Home Assistant 智能家居仪表盘创建无缝、高度可定制的界面。 - GitHub - shannonhochki...</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows 安装 | Unsloth 文档</a>: 了解如何在带有或不带有 WSL 的 Windows 上安装 Unsloth。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: 未找到描述词</li><li><a href="https://github.com/woct0rdho/triton-windows">GitHub - woct0rdho/triton-windows: 支持 Windows 的 Triton 语言和编译器分支</a>: 支持 Windows 的 Triton 语言和编译器分支 - woct0rdho/triton-windows</li><li><a href="https://github.com/unslothai/unsloth/issues/1713">Unsloth 覆盖了由 huggingface 库加载的模型的 forward call 函数 · Issue #1713 · unslothai/unsloth</a>: 我正在尝试使用预训练模型作为奖励模型的 GRPO notebook。基本上，我参考了这个链接中的 notebook https://colab.research.google.com/github/unslothai/notebooks/blob/mai...</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>: 又名持续微调 (Continued Finetuning)。Unsloth 允许你进行持续预训练，使模型可以学习一种新语言。</li><li><a href="http://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>: 第一次接触 Unsloth？</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | U</a>

nsloth 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1340512228218241205)** (102 messages🔥🔥): 

> `Fine Tuning vs RAG, Synthetic Data Generation, LLM Training Strategies, Dynamic Data Challenges, Optimized Reasoning Dataset` 


- **Fine Tuning 领先于 RAG**：成员们讨论了在正确执行的情况下，**Fine Tuning** 的表现如何显著优于 **RAG**，并强调了在拥有充足训练数据时性能的提升。
   - 一位成员对 **Fine Tuning** 的未来表示信心，认为它可能成为 **LLM** 应用中的主导方法，而将 **RAG** 边缘化到特定用途。
- **合成数据生成 (Synthetic Data Generation) 见解**：一位成员分享了他们在合成数据生成方面的成功经验，即通过要求 **LLM** 根据给定文档生成问题，进而创建训练数据集。
   - 这种方法被认为有助于保持问题的相关性，同时解决了训练数据的结构和覆盖范围问题。
- **动态数据的挑战**：成员们对频繁变化的数据在仅依赖 **Fine Tuning** 时如何影响模型性能表示担忧，这支撑了 **RAG** 在这些场景中的重要性。
   - 成员们指出，虽然 **Fine Tuning** 表现卓越，但对于快速演变的数据，可能仍需要 **RAG**，需要结合这两种方法以获得最佳结果。
- **LLM 训练策略探索**：讨论涉及了 **Fine Tuning** 中数据质量的重要性，强调收集多样化且冗余的问题以帮助模型更好地理解。
   - 建议调整训练参数（如 Learning Rate 和 Epochs）可以影响模型学习细微数据的效果，特别是对于较小的数据集。
- **发布新的优化推理数据集**：宣布了一个名为 **Optimized_Reasoning** 的新数据集，旨在提高模型的推理能力，同时减少 Token 使用量。
   - 该数据集解决了现代 **LLM** 在处理推理任务时的性能差距，为获得更好的训练结果提供了资源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2307.03172">Lost in the Middle: How Language Models Use Long Contexts</a>: 虽然最近的语言模型能够将长上下文作为输入，但关于它们利用长上下文的效果如何，知之甚少。我们分析了语言模型在两个任务上的表现...</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1irdwy0/new_dataset_release_romboorgoptimized_reasoning/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1340102902487715942)** (109 messages🔥🔥): 

> `Coconut LLM, Chess Move Reasoning, DeepSeek Performance, SCION Optimizer, Paper Review Request` 


- **探索 Coconut LLM 的能力**：成员们讨论了 [Coconut LLM](https://github.com/facebookresearch/coconut) 及其对 GPT-2 的使用，思考更新的模型如何提高推理任务的性能。
   - 一位成员指出，目前的结果尚未反映出重大进展，并推测这可能与 Unsloth 的未来计划有关。
- **国际象棋走法推理中的挑战**：一位成员分享了他们训练国际象棋走法推理模型的经验，强调了 **LLM** 在复杂博弈中难以进行充分预判的问题。
   - 另一位成员指出，有必要构建一个强大的 Reward Function，以在实际应用中正确引导模型的学习。
- **DeepSeek 令人印象深刻的进展**：社区强调了 DeepSeek 团队的高效工作，一位成员报告称，在处理 **GitHub** PR 时，2000 次尝试中成功完成了 337 次。
   - 这种对性能的关注表明，人们对利用此类模型处理复杂编码任务有着持续的兴趣，这与小组的使命一致。
- **SCION 优化器提案**：建议使用一种新的优化器 SCION，它可能在不使用 Adam 的情况下训练大型神经网络，在减少内存占用的同时提高结果。
   - 这引发了关于 SCION 在模型训练中如何高效管理计算资源的实际应用的讨论。
- **寻求 AI 研究论文的反馈**：一位成员请求对其关于扩展 **Mixture of Experts (MoE)** 模型的论文提供反馈，并联系了当地教授寻求帮助。
   - 他们正积极与多位学者进行评审交流，展示了社区在改进研究成果方面的协作方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>: 我们提出了 rStar-Math，证明了小型语言模型 (SLMs) 可以在不依赖优于其的模型进行蒸馏的情况下，媲美甚至超越 OpenAI o1 的数学推理能力。rStar-Math 实现了...</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>: 本文介绍了在极小数据集上过拟合 (overfitting) 预训练大语言模型 (LLMs) 时出现的反直觉泛化结果。在开放式文本生成的设置下，通常认为...</li><li><a href="https://x.com/CevherLIONS/status/1890081275949768955?t=Non-jguA4XQVfb1rxzmg2w&">Tweet from Volkan Cevher (@CevherLIONS)</a>: 🔥 想要在不使用 Adam 的情况下训练大型神经网络，同时占用更少内存并获得更好结果吗？⚡ 看看 SCION：一种利用范数约束 (norm-constrained) 适应问题几何结构的新型优化器...</li><li><a href="https://x.com/CevherLIONS/status/1890081275949768955?t=Non-jguA4XQVfb1rxzmg2w&s=19">Tweet from Volkan Cevher (@CevherLIONS)</a>: 🔥 想要在不使用 Adam 的情况下训练大型神经网络，同时占用更少内存并获得更好结果吗？⚡ 看看 SCION：一种利用范数约束 (norm-constrained) 适应问题几何结构的新型优化器...</li><li><a href="https://github.com/NoahBPeterson/hemingway-GRPO/blob/main/s1_grpo_trainer.py">hemingway-GRPO/s1_grpo_trainer.py at main · NoahBPeterson/hemingway-GRPO</a>: 通过在 GitHub 上创建账号，为 NoahBPeterson/hemingway-GRPO 的开发做出贡献。</li><li><a href="https://github.com/facebookresearch/coconut?tab=readme-ov-file">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>: 训练大语言模型在连续潜空间 (Continuous Latent Space) 中进行推理 - facebookresearch/coconut</li><li><a href="https://github.com/hkust-nlp/CodeIO">GitHub - hkust-nlp/CodeIO: CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction</a>: CodeI/O：通过代码输入输出预测压缩推理模式 - hkust-nlp/CodeIO</li><li><a href="https://github.com/huggingface/trl/issues/2723">[Project] Training Agents with GRPO · Issue #2723 · huggingface/trl</a>: 让我们讨论如何使用 GRPO 训练 Agent。在这里，我将链接与实现该想法所需的各种问题、功能或疑问相关的子议题。</li><li><a href="https://search.brave.com/search?q=train+model+to+play+chess&source=android&summary=1&conversation=d57e09e86a44422daba783">Brave Search</a>: 搜索网页。私密。真正有用的结果、AI 驱动的回答等。全部来自独立的索引。无画像分析，无偏见，无大科技公司。</li><li><a href="https://github.com/huggingface/trl/pull/2810">GRPO Environments for custom multi-step rollouts (vLLM-only) by willccbb · Pull Request #2810 · huggingface/trl</a>: 此 PR 做了什么？在 trl/environments 下为 Environment 对象添加了一个协议，该对象封装了 vLLM 的 .generate(...) 以允许自定义 rollout 逻辑，并向 Trai... 添加了一个可选的 env 字段。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1340013903475900517)** (1 messages): 

> `Deep Research, Query Limits, Platform Availability` 


- **Perplexity 发布 Deep Research**: Deep Research 允许用户通过执行多次搜索、阅读来源并交付综合结果，自动生成深入的研究报告。非订阅用户**免费**使用，限制为**每天 5 次查询**。
   - 它在专家级任务中表现出色，并在 Humanity’s Last Exam 基准测试中获得了高分。
- **Deep Research 现已在 Web 端上线**: Deep Research 从今天起对 Web 用户开放，并将很快推送到 **iOS**、**Android** 和 **Mac**。
   - 建议用户将 [App 更新](https://perplexity.ai)至最新版本以访问此功能。
- **Pro 用户享有更高的查询限制**: 相比非订阅用户的**每天 5 次查询**，Pro 用户可以享受显著更高的限制，即**每天 500 次查询**。
   - 这种分级访问为选择 Pro 订阅的用户提供了更多深入研究的机会。
- **如何有效利用 Deep Research**: 要尝试 Deep Research，用户应在提交查询前从搜索框的模式选择器中选择 “Deep Research”。
   - 有关其功能的详细说明可以在[此处](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)找到。



**提及的链接**: <a href="https://perplexity.ai)">未找到标题</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1340005226597318762)** (756 messages🔥🔥🔥):

> `Perplexity Pro 功能、Deep Research 与 OpenAI 模型对比、Perplexity 使用案例、账单与订阅查询、技术问题及反馈` 


- **Perplexity Pro 与 You.com 的对比**：用户强调了他们在 Perplexity Pro 和 You.com 上的使用体验，特别是关注定价以及对各种 AI models 的访问。
   - 虽然 Perplexity 提供了 Deep Research 能力，但一些用户对 You.com 提供的更实惠且可能无限制的访问模式表现出兴趣。
- **Deep Research 的局限性**：许多用户指出了 Perplexity 的 Deep Research 的局限性，通常将其与 OpenAI 的产品进行对比并认为稍逊一筹，并报告了回复中频繁出现幻觉（hallucinations）的问题。
   - 尽管存在这些问题，但随着 Perplexity 继续开发其功能，用户对未来的改进持乐观态度。
- **Perplexity 的使用案例**：用户提到使用 Perplexity 处理各种任务，如研究、编程辅助（coding assistance）以及管理心理健康对话，展示了其多功能性。
   - Deep Research 和 R1 等功能的加入使其成为许多应用场景下的强大工具，特别是对于学生和专业人士。
- **订阅与账单问题**：一些用户对订阅问题表示担忧，例如无法访问其 Pro roles 或在切换模型时遇到 bug。
   - 关于促销代码和 Pro 功能验证的咨询引发了关于服务可访问性和客户支持的讨论。
- **技术故障与用户反馈**：用户报告了 Perplexity app 中的技术故障，包括语音模式（voice mode）和 Deep Research 功能的问题，并对糟糕的用户体验表示沮丧。
   - 收集了关于 app 整体体验和能力的反馈，表明用户希望改进并更好地整合各项功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.apple.com/documentation/swiftui/picker">Picker | Apple 开发者文档</a>：用于从一组互斥值中进行选择的控件。</li><li><a href="https://developer.apple.com/documentation/swiftui/toggle">Toggle | Apple 开发者文档</a>：在开启和关闭状态之间切换的控件。</li><li><a href="https://www.markdownguide.org/tools/reddit/">Reddit | Markdown 指南</a>：Reddit 是一个支持使用 Markdown 发布内容的流行在线社区。</li><li><a href="https://www.theverge.com/2024/10/2/24260262/ray-ban-meta-smart-glasses-doxxing-privacy">大学生利用 Meta 智能眼镜实时人肉搜索他人</a>：隐私始终是智能眼镜面临的一个主要问题。</li><li><a href="https://home.you.com/articles/youdotcom-is-the-go-to-platform-to-test-and-compare-the-latest-ai-models">在 you.com 一站式访问最强大的 AI 模型</a>：You.com 现在提供首创的自定义模型选择器，让用户可以在一个地方访问、测试和比较大型语言模型 (LLMs)，如 GPT-4、Claude Instant、Gemini Pro 等...</li><li><a href="https://www.theverge.com/2024/10/2/24260262/ray-ba">大学生利用 Meta 智能眼镜实时人肉搜索他人</a>：隐私始终是智能眼镜面临的一个主要问题。</li><li><a href="https://www.cplx.app/">Complexity</a>：每个人都梦寐以求的 Perplexity.ai 增强版。</li><li><a href="https://chromewebstore.google.com/detail/perplexity-mermaid-viewer/dniomjbkoibfmmdjeghickikhdlddhnm">Perplexity Mermaid Viewer - Chrome 网上应用店</a>：检测 Perplexity 上的 Mermaid 图表并在弹出窗口中显示预览的扩展程序。</li><li><a href="https://tenor.com/view/izuku-midoriya-smash-robot-destroy-gif-25269478">Izuku Midoriya GIF - Izuku Midoriya Smash - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://testing.googleblog.com/2007/01/introducing-testing-on-toilet.html">
Google 测试博客：介绍“马桶上的测试”
</a>：未找到描述</li><li><a href="https://x.com/deryatr_/status/1890955299604865366?s=46">来自 Derya Unutmaz, MD (@DeryaTR_) 的推文</a>：为了比较 Perplexity Deep Research (PDR) 和 OpenAI Deep Research (ODR)，我上传了数十个我们确认为在某些免疫细胞亚型中差异表达的基因，然后使用了...</li><li><a href="https://www.youtube.com/watch?v=wU-490g1d3c">苏联剪辑，但效果好得多！！！</a>：由 Overtracker 剪辑，音乐已减速。我不拥有视频或音乐的版权。致谢：ONIMXRU x SMITHMANE "SHADOW" :https://www.youtube.com/watch?v=jq...</li><li><a href="https://www.zdnet.com/article/perplexity-lets-you-try-deepseek-r1-without-the-security-risk/">Perplexity 让你在没有安全风险的情况下尝试 DeepSeek R1</a>：这里有两种在不将数据暴露给国外服务器的情况下尝试 R1 的方法。</li><li><a href="https://deepnewz.com/ai-modeling/openai-updates-gpt-4o-model-extends-knowledge-cutoff-to-june-2024-october-2023-4-6ab7d8ce.">未找到标题</a>：未找到描述
</li>
</ul>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1340021149521940621)** (47 messages🔥): 

> `Perplexity AI 更新, 文化与声音遗产, DeepSeek 对 AI 的影响, Substack 说明, Ethereum Spectra 升级` 


- **Perplexity AI 发布更新**：Perplexity AI 推出了多项更新，包括 Jeep 的车载广告以及由 FBI 揭露的秘密 UFO 办公室，详情见 [YouTube 视频](https://www.youtube.com/embed/gXIdi-b75_k)。
   - 讨论了*文化与声音遗产*，强调了技术与艺术表达之间的相互作用。
- **讨论 DeepSeek 对 AI 的影响**：成员们思考了 [DeepSeek](https://www.perplexity.ai/search/how-does-deepseek-impact-ai-de-Ks0imm6zSYyr2b53rIkBiQ) 对 AI 的影响，探讨了它可能如何重塑当前的方法论。
   - 观点各异，一些人强调其在 AI 深度方面的潜在益处，另一些人则对集成挑战表示担忧。
- **Substack 功能详解**：[一位成员寻求关于 Substack 运作方式的解答](https://www.perplexity.ai/search/how-would-you-explain-substack-lqF9LguMRdK4.cqO5q70Dw)，旨在利用该平台进行内容创作。
   - 讨论强调了 Substack 区别于传统博客的独特功能。
- **重点介绍 Ethereum 的 Spectra 升级**：分享了最近的 [Ethereum Spectra 升级](https://www.perplexity.ai/page/ethereum-pectra-upgrade-a-comp-.mLTziZzT3Sn5xWmFa50hA)，展示了网络的主要变化和改进。
   - 参与者讨论了对 dApps 和更广泛的 Ethereum 生态系统的潜在影响。
- **探索文化与声音遗产**：成员们讨论了最近关于*文化与声音遗产*的分析，相关见解发表在 [Perplexity AI 页面](https://www.perplexity.ai/page/the-cultural-and-sonic-legacy-nEwxLsvRTPi7vbuGmTA_1Q)上。
   - 对话集中在技术进步对文化表达和声音创新的影响。



**提到的链接**：<a href="https://www.youtube.com/embed/gXIdi-b75_k">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1340006375044022303)** (18 messages🔥): 

> `Perplexity API 发票, Deep Research 可用性, Sonar API 引用问题, Perplexity API 中的图像处理, 使用 Deep Research 模型` 


- **关于 Perplexity API 发票的请求**：几位成员对购买 API 额度后未收到客服关于**发票**的回复表示担忧。
   - 一位成员根据另一位用户的建议，分享了通过点击“View Dashboard”并找到“Invoice History”来手动查找发票的方法。
- **关于 API 中 Deep Research 的咨询**：成员们询问 **Deep Research** 功能是否可以通过 API 访问，对其潜在的增强功能表现出兴趣。
   - 其他人确认该功能目前无法通过 API 使用，并建议查阅文档以获取进一步说明。
- **API 响应中缺失引用**：针对 **Sonar API** 未能按预期返回引用的问题提出了担忧，尽管输出中包含了参考资料。
   - 一位用户澄清说引用是单独存在的，并提供了输出结构的示例，强调需要正确访问它们。
- **API 中的图像处理**：一位成员询问了 **Perplexity API** 对图像的支持，结果发现目前仅支持文本输入格式。
   - 这引发了关于完善文档以反映 API 功能和局限性的讨论。
- **在 API 中将 Deep Research 设置为默认**：一位用户寻求关于如何在 API 使用中将 **Deep Research 模型**设置为默认模型的帮助。
   - 回复指出该模型不适用于 API，并引导用户参考文档。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai">未找到标题</a>：未找到描述</li><li><a href="https://sdk.vercel.ai/providers/ai-sdk-providers/perplexity">Perplexity</a>：了解如何将 Perplexity 的 Sonar API 与 AI SDK 结合使用。</li><li><a href="https://xpert.digital/en/ai-search-engine/).">Perplexity Sonar Pro API 作为外部应用和工具中的 AI 搜索引擎 —— 适用于智能应用和定制化搜索</a>：🚀 将 Perplexity Sonar Pro API 集成到您的工具中，实现智能搜索解决方案。🔎 通过基于 AI 的搜索优化您的应用，以获得精确结果。🛠️ 创建...</li><li><a href="https://www.youtube.com/watch?v=ExvP1EIUo1s',">*全新* Perplexity.AI 文档上传</a>：释放 Perplexity.ai 最新文档上传功能的力量。Perplexity.ai 已添加上传文档以辅助回答问题的功能...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1340785146223263845)** (1 条消息): 

> `Goedel theorem prover, GLiNER-MoE-MultiLingual, Kyro-1, Science News Bot, Mini PyTorch` 


- **Goedel theorem prover 发布**: **Goedel theorem prover** 已在 Hugging Face 上推出，展示了其在数学推理方面的独特能力。你可以在[这里](https://huggingface.co/spaces/Tonic/Math)进行探索。
- **GLiNER-MoE-MultiLingual NER 模型**: **GLiNER-MoE-MultiLingual** 是一个零样本（zero-shot）命名实体识别模型，使用 NOMIC-MOE 架构训练了一个 epoch，突显了先进的 NER 能力。更多详情请见[这里](https://huggingface.co/Mayank6255/GLiNER-MoE-MultiLingual)。
- **Kyro-1: Open-Neo 的推理模型**: **Kyro-1** 标志着 Open-Neo 的首个推理模型，在中轻度推理任务中表现出强劲性能。欲了解更多信息，请点击[这里](https://huggingface.co/collections/open-neo/kyro-n1-67ab2e7bbc76a9aab3030c21)查看。
- **Science News Bot 上线了！**: **Science News Bot** 已发布，随时准备提供最新的科学进展和研究。你可以在 [GitHub](https://github.com/AstraBert/SciNewsBot) 上找到该项目。
- **Mini PyTorch 带来简洁性**: **Mini PyTorch** 已发布，为 PyTorch 爱好者和开发者提供了一个轻量级的替代方案。在[这里](https://github.com/karam-koujan/mini-pytorch)探索其功能。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1340015467103784981)** (327 条消息🔥🔥): 

> `Hugging Face Model Usage, Model Performance, Training and Fine-tuning LLMs, Technical Issues and Errors, AI Agent Development` 


- **Hugging Face 模型使用查询**: 用户讨论了在本地使用哪些模型以及特定 Hugging Face 模型的要求，例如 Llama 3.2 模型需要 20GB VRAM。
   - 一些用户探索了量化选项，以便在 VRAM 较低的 GPU 上运行更大的模型。
- **模型性能见解**: 围绕一个 9B 模型性能优于更大模型的讨论展开，强调了其在同尺寸下的出色表现。
   - 交换了模型性能指标，展示了各种模型的能力。
- **训练和微调 LLM**: 有关于使用 AutoTrain 训练 Google Flan-t5-base 模型的咨询，并询问了预期的训练时间。
   - 用户分享了关于微调和量化在实现和效率方面与模型适配器（adapters）对比的见解。
- **技术问题与错误**: 几位用户报告在访问 Hugging Face 网站某些资源时收到 500 内部错误。
   - 分享了通过创建新 Spaces 并重新上传源代码来解决这些问题的建议。
- **AI Agent 开发讨论**: 用户讨论了将 Agentic 代码编辑器集成到 Discord 机器人中的实现，寻求 CLI 集成和可脚本化接口等功能。
   - 推荐了一些满足这些要求且强调易用性、无需复杂界面的工具。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>：未找到描述</li><li><a href="https://easyqz.online">EasyQZ</a>：未找到描述</li><li><a href="https://huggingface.co/tomg-group-umd/huginn-0125">tomg-group-umd/huginn-0125 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo">YuE - innova-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/tasks">Tasks - Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/hmmm-thinking-batman-gif-6153870554148391864">Hmmm Thinking GIF - Hmmm Thinking Batman - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/tree/main">leafspark/Llama-3.2-11B-Vision-Instruct-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/instructor-ai/instructor">GitHub - instructor-ai/instructor: 为 LLM 提供结构化输出</a>：为 LLM 提供结构化输出。通过在 GitHub 上创建账号来为 instructor-ai/instructor 的开发做出贡献。</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct">meta-llama/Llama-3.2-11B-Vision-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/building-preparing-space-takes-forever">构建 / 准备 Space 耗时过长</a>：我的 Space 要么一直处于构建中，要么一直处于准备中：没有任何警告/错误日志或其他通知。多次尝试删除并重新创建仍出现同样的问题。即使切换到付费计划也...</li><li><a href="https://huggingface.co/google/owlv2-base-patch16-ensemble">google/owlv2-base-patch16-ensemble · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/deepbeepmeep/YuEGP">GitHub - deepbeepmeep/YuEGP: YuE: 面向 GPU 资源匮乏者的开源全曲生成基础模型</a>：YuE：面向 GPU 资源匮乏者的开源全曲生成基础模型 - deepbeepmeep/YuEGP</li><li><a href="https://youtu.be/WKTNxaZJf4Y">LLM 编年史 #6.4：结合 ReAct (Reason + Act) 的 LLM Agent</a>：在本集中，我们将介绍 LLM Agent，重点关注那些在提高 LLM 推理能力的同时允许其与外部交互的核心研究...</li><li><a href="https://github.com/SadilKhan/Text2CAD">GitHub - SadilKhan/Text2CAD: [NeurIPS'24 Spotlight] Text2CAD：从初学者到专家级的文本提示生成序列化 CAD 设计</a>：[NeurIPS'24 Spotlight] Text2CAD：从初学者到专家级的文本提示生成序列化 CAD 设计 - SadilKhan/Text2CAD</li><li><a href="https://gist.github.com/Getty/f2a76b52aac1c21aa8b4a92d978c92d9">transcribe.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">Easy_training/Galore+Qlora_With_Multi_GPU_Support at main · rombodawg/Easy_training</a>：通过在 GitHub 上创建账号来为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://pcniki-com.translate.goog/yue/?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=ja&_x_tr_pto=wapp">YuE 的安装方法和基本用法</a>：YuE 是一款为自动生成音乐而开发的尖端开源 AI 模型。它特别擅长根据歌词生成完整的歌曲，能够创作出包含人声和伴奏的、长达数分钟的正规乐曲。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1340357871732265040)** (4 条消息): 

> `NVIDIA Course, Generative AI Tutorials, Differential Privacy` 


- **关于 NVIDIA 课程的咨询**：一位成员询问某个特定项目是否为 **NVIDIA 课程**，以及是否**免费**提供。
   - 未提供关于课程本身的更多具体信息。
- **寻找 Generative AI 资源**：一位新成员（**3D 图形程序员**）正在寻求专注于 **Generative AI** 的课程或教程推荐。
   - 他们特别提到偏好包含模型训练和部署的**实战编程环节**的资源。
- **Differential Privacy 博客系列发布**：一位成员开始撰写一系列关于 **Differential Privacy** 的博客，首篇入门文章名为《Differential Privacy!! But Why?》，可在此处[阅读](https://theailandscape.substack.com/p/differential-privacy-but-why)。
   - 作者强调了在各种背景下，从数据中提取洞察与保护**个人隐私**之间的关键平衡。



**提到的链接**：<a href="https://theailandscape.substack.com/p/differential-privacy-but-why">Differential Privacy!! But Why?</a>：Differential Privacy 启蒙系列博客 #1

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 条消息): 

pablo.ce: https://www.youtube.com/watch?v=03jYz0ijbUU

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1340095440779087882)** (15 条消息🔥): 

> `AI 驱动的测验, 插画文本转视频, GLiNER-MoE-MultiLingual, SwarmFormer 优化, DeepNeo 模型` 


- **AI 测验变革学习方式**：一篇文章讨论了 [EasyQZ](https://medium.com/@visrow/ai-deep-research-and-quiz-platform-easyqz-transforming-learning-with-ai-powered-quizzes-494f289c506d) 如何利用 AI 创建引人入胜的测验，从而彻底改变学习体验。
   - 该平台承诺通过交互式测验形式增强教育评估。
- **动漫主题视频制作简化**：一个名为 [Illustration Text to Video](https://huggingface.co/spaces/Sergidev/Illustration-Text-To-Video) 的新项目使用户能够使用 HunyuanVideo 和流行的 2D 艺术 LORA 创建动漫主题短视频。
   - 分享了一个示例输出视频，展示了该工具在生成动画内容方面的能力。
- **使用 GLiNER-MoE-MultiLingual 进行 Zero-Shot NER**：对 [GLiNER-MoE-MultiLingual](https://huggingface.co/Mayank6255/GLiNER-MoE-MultiLingual) 模型的探索，该模型在仅经过一个训练 epoch 后，就能成功执行多种语言的 Zero-Shot NER 任务。
   - 结果显示，与仅限英文的模型相比，该模型具有竞争力的性能，并分享了优化策略的详细概述。
- **优化 SwarmFormer 以提高效率**：对 [SwarmFormer](https://huggingface.co/Mayank6255/SwarmFormer-small-ef) 的改进包括 local windowed attention 和 token-to-cluster gating，增强了计算效率。
   - 正在进行的实验展示了 FLOPs 的显著降低，使该模型适用于移动和边缘环境。
- **用于推理任务的 DeepNeo 模型**：[DeepNeo](https://huggingface.co/collections/open-neo/deepneo-1-67aea4c0f086ab0f70ed5720) 模型被介绍为一种混合模型，支持普通推理和 Agentic 推理应用。
   - 它从 DeepHermes 中汲取灵感，为推理任务提供了灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Sergidev/Illustration-Text-To-Video">Illustration Text To Video - a Hugging Face Space by Sergidev</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/open-neo/deepneo-1-67aea4c0f086ab0f70ed5720">DeepNeo-1 - a open-neo Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/EarthnDusk/SDXL_To_Diffusers">SDXL To Diffusers - a Hugging Face Space by EarthnDusk</a>: 未找到描述</li><li><a href="https://huggingface.co/Mayank6255/GLiNER-MoE-MultiLingual">Mayank6255/GLiNER-MoE-MultiLingual · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1irdwy0/new_dataset_release_romboorgoptimized_reasoning/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/minchyeom/Starlette-1-3B">minchyeom/Starlette-1-3B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.kaggle.com/code/mayankrakesh121/dynamic-swarmformer">Dynamic SwarmFormer</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自无附加数据源的数据</li><li><a href="https://huggingface.co/Mayank6255/SwarmFormer-small-ef">Mayank6255/SwarmFormer-small-ef · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/louisbrulenaudet/agentic-market-tool">GitHub - louisbrulenaudet/agentic-market-tool: Barebones tool built upon Hugging Face smolagents and Alpaca for financial analysis automation 🤗</a>: 基于 Hugging Face smolagents 和 Alpaca 构建的用于金融分析自动化的基础工具 🤗 - louisbrulenaudet/agentic-market-tool
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 条消息): 

arpitbansal.: 绝对不是正确的频道，甚至不是正确的服务器
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1340186203802177618)** (13 messages🔥): 

> `Alzheimer's detection with MRI, AI controlling applications, Computer Vision Automation Framework, Performance improvements in vision systems, M.A.R.K. AI Assistant` 


- **使用 OASIS 数据集进行阿尔兹海默症检测**：一位成员正在研究**使用 MRI 扫描进行阿尔兹海默症检测**，并在此任务中使用了 **OASIS 数据集**。
   - 他们正在寻求该领域有经验的其他人的指导。
- **AI 模拟应用上的真人操作**：一段演示视频展示了 AI 利用 **computer vision** 与 **Galculator 应用**进行交互，包括映射按钮位置。
   - 该方法使用 Python 中的测试序列来自动化类人操作，展示了其功能。
- **Computer Vision 框架增强**：一个视觉系统正在实施性能增强，旨在从**每 5 秒 1 次预测**提升到**每秒 3 次预测**。
   - 改进包括将图像大小减少 **1.2GB**，使应用程序几乎能够瞬间启动。
- **AI-M.A.R.K. 协作**：成员们讨论了构建 **10 Minute M.A.R.K.** (Management Assistance and Reporting Kinetics) AI，该 AI 可以执行传统上由人类完成的任务。
   - 成员们有兴趣通过 pull requests 分享测试和代码贡献，以增强该框架。
- **分享测试与贡献**：一位成员表示愿意将其测试贡献给 **CVAF 仓库**，并建议可能为他们的更改创建一个单独的 branch。
   - 他们提供了一个 GitHub 仓库链接，展示了他们在该项目上的工作。



**Link mentioned**: <a href="https://github.com/OIEIEIO/CVAF/tree/oieieio">GitHub - OIEIEIO/CVAF at oieieio</a>: Computer-vision based automation framework (name is a WIP) - GitHub - OIEIEIO/CVAF at oieieio

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1340071534479409253)** (5 messages): 

> `Scaling up Test-Time Compute, Multi-Task Text Generation Model Training, Fine-Tuning Qwen Models, Function Calling Model Tuning` 


- **探索 Latent Reasoning 方法**：一位成员询问是否有人尝试过论文 [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://link.to.paper) 中的模型。该方法专注于提高测试阶段的计算效率。
   - *有人尝试过这个模型吗？* 随后引发了对共享经验和技术的呼吁。
- **多任务文本生成寻求指导**：一位成员正在寻求关于训练多任务文本生成模型的指导，用于**翻译**和**写作辅助**等任务。他们对构建数据集的最佳方式表示疑问，并寻求除 **LoRA** 之外的有效微调技术。
   - 成员们受邀对**数据集结构**和**数据源**提出建议，重点是防止训练过程中的*灾难性遗忘 (catastrophic forgetting)*。
- **请求 Qwen 3B 模型审查**：一位成员请求检查重新上传的 Qwen 模型，并询问关于测试原始 **Qwen 3B 模型**的情况。他们分享了一个 [Qwen 2.5 免费 Google Colab](https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing) 链接。
   - 他们还提供了额外的 [Qwen 2.5 notebooks](https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing) 链接，用于对话风格的微调。
- **Function Calling 模型微调**：一位成员提出了关于微调 **function calling 模型**的疑问，即当函数不可用时如何提供通用回答。他们希望在调整模型响应的同时，保持模型调用函数的能力。
   - 对话围绕确保模型在潜在的函数限制下仍能有效服务用户查询展开。



**Link mentioned**: <a href="https://huggingface.co/unsloth/Qwen2.5-3B-bnb-4bit">unsloth/Qwen2.5-3B-bnb-4bit · Hugging Face</a>: no description found

  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1340017334219444267)** (18 条消息🔥): 

> `Token 创建困惑、需要详细视频讲解、Meta LLama 3.2 访问问题、'Think -> Action -> Observation' 循环、微调章节集成至 NLP 课程` 


- **Token 创建困惑**：一位成员对创建 Token 时需要勾选的大量选项感到沮丧，觉得不知所措并寻求帮助。
   - 另一位成员提到他们在 Google Colab 上运行时不需要勾选任何选项，建议简化流程。
- **请求 Dummy Agent 代码的详细视频**：一位成员请求对 “Dummy Agent library” 环节的代码进行 10-15 分钟的视频讲解，认为课程中的解释不够充分。
   - 作为回应，另一位成员建议运行 Notebook 中提供的代码，其中已经包含了进一步的解释。
- **Meta LLama 3.2 模型访问被拒**：一位用户报告说他们申请 Meta LLama 3.2 模型访问权限被拒绝，可能是因为所在国家/地区的原因。
   - 另一位用户确认在美国也被拒绝，建议课程更新以使用其他替代模型。
- **'Think -> Action -> Observation' 循环的益处**：一位成员询问 'Think -> Action -> Observation' 循环的有效性，指出他们的 Agent 只是在重复动作而没有完成任务。
   - 另一位用户回复说，如果使用得当，该循环应该允许自我纠正，并请求提供示例。
- **NLP 课程中新增微调章节**：官方宣布微调章节已集成到 Hugging Face NLP 课程中，引入了测验和互动练习。
   - 分享了新章节的链接，邀请参与者探索增强的学习资源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/nlp-course">nlp-course (Hugging Face NLP 课程)</a>：未找到描述</li><li><a href="https://huggingface.co/blog/smolervlm#smoldocling">SmolVLM 变得更小了 – 推出 256M &amp; 500M 模型！</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1340004941640761434)** (772 条消息🔥🔥🔥): 

> `课程结业证书、课程访问常见问题、提交问题排查、即将推出的单元和内容、社区参与` 


- **完成单元后授予证书**：许多参与者庆祝完成了 Unit 1 并成功获得了课程证书，在社区中分享了他们的成就。
   - 然而，一些用户对如何获取证书以及相应的凭证链接表示困惑。
- **访问课程材料**：新学员询问如何开始课程和访问 Unit 1，许多人被引导至 Hugging Face 官方页面查看说明。
   - 强调该课程是自适应进度的（self-paced），鼓励用户根据自己的方便参与学习。
- **考试和工具的技术问题**：用户报告了提交测验和在课程中使用自定义工具时遇到的问题，表明需要额外支持。
   - 讨论了排查步骤，如重新登录和管理 Hugging Face 账户设置以解决这些问题。
- **社区交流与活动**：参与者表达了合作和交流的渴望，一些人提议组建学习小组进行集体学习。
   - 发布了关于即将举行的 Hackathon 活动的公告，旨在促进社区参与并提供实践机会。
- **课程进度与预期**：关于下一单元时间表的询问显示，奖励材料将很快发布，后续单元预计将定期推出。
   - 参与者对学习更多关于 AI Agent 的知识表现出极大的热情，并被鼓励在课程推进过程中保持耐心和参与度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/unit1/introduction">Introduction to Agents - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://u.cisco.com/paths/288">Cisco U.</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/communication/next-units">When will the next units be published? - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/">Introduction to Agents - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/tutorial">Let’s Create Our First Agent Using smolagents - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://hf.co/settings/tokens.">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/communication/live1">Live 1: How the Course Works and First Q&amp;A - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://apps.apple.com/at/app/ai-academy-deep-learning/id6740095442?l=en-GB">‎Ai Academy : Deep Learning</a>: ‎80 天 AI 精通之路：循序渐进学习深度学习。通过 Day-by-Day Deep Learning 这一动态且交互式的学习应用，按照您自己的节奏掌握人工智能和机器学习...</li><li><a href="https://apps.apple.com/us/app/twitch-live-streaming/id460177396">‎Twitch: Live Streaming</a>: ‎Twitch 是成千上万社区汇聚一堂的地方，为了我们喜爱的博主、热爱的游戏、乐趣、彼此以及任何事物。下载 Twitch，加入数百万人享受直播的行列...</li><li><a href="https://nvda.ws/3WYewvn">How Scaling Laws Drive Smarter, More Powerful AI</a>: AI Scaling Laws 描述了模型性能如何随着训练数据规模、模型参数或计算资源的增加而提升。</li><li><a href="https://huggingface.co/agents-course/notebooks/blob/main/dummy_agent_library.ipynb">dummy_agent_library.ipynb · agents-course/notebooks at main</a>: 未找到描述</li><li><a href="https://steamescommnunity.com/s/1059410958">Steam Gift Activation</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct">meta-llama/Llama-3.3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/theRealProHacker/">theRealProHacker - Overview</a>: 乐于尝试任何事物 UI/UX | AI | NLP | Games | QC - theRealProHacker</li><li><a href="https://github.com/huggingface/agents-course/pull/154">fix: add missing variable declarations by rhanb · Pull Request #154 · huggingface/agents-course</a>: 在本地环境中学习本课程的这一部分时，不清楚变量 SYSTEM_PROMPT 和 prompt 指代的是什么</li><li><a href="https://gist.github.com/jdnichollsc/1301bddd3b958b4f465063d700193212">Data ingestion with Hugging Face datasets</a>: 使用 Hugging Face datasets 进行数据摄取。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1340386662085038082)** (1 条消息): 

> `Distilabel Error, Data Generation with Distilled R1 Model` 


- **Distilabel 中的缓存 AssertionError**: 一位用户在尝试使用 **Distilabel** 从 **蒸馏 R1 模型 (distilled R1 model)** 生成数据时，遇到了一个 **AssertionError**，提示 'Cannot have cache with result_handler not alive'。
   - *已卡住 20 分钟*，他们正在寻求实现方面的帮助并排查该问题。
- **分享代码片段用于调试**: 用户分享了一个代码片段，其中加载了一个数据集，并为模型 ID 为 **DeepSeek-R1-Distill-Qwen-1.5B** 的文本生成设置了 **Pipeline**。
   - 提供的代码包含了 **vLLM** 模型的配置和一个 **TextGeneration** 步骤，旨在利用指定参数生成数据。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1340117500008071240)** (1 条消息): 

> `Windsurf 补丁版本 1.3.4 发布，引入 Model Context Protocol，自定义应用图标功能，消息取消修复，Cascade Base 额度处理` 


- **Windsurf 1.3.4 补丁发布**：**Windsurf 1.3.4** 补丁已发布，解决了与 Cascade 写入工具、消息取消等相关的 Bug。你可以在[这里](https://www.codeium.com/changelog)阅读完整的变更日志（2025年2月14日）。
   - Bug 修复包括对 Cascade Base 额度处理的增强以及改进的身份验证登录错误提示。
- **现已支持 Model Context Protocol**：Cascade 现在支持 **Model Context Protocol (MCP)**，允许用户对配置好的 MCP 服务器进行工具调用（tool calls）。用户只需点击 Cascade 输入工具栏中的锤子图标即可轻松设置 MCP。
   - 此功能适用于所有个人方案，每次 MCP 工具调用消耗一个 flow act。
- **消息取消与身份验证错误修复**：该补丁包含了关于用户消息取消的各项修复，以及针对身份验证入口更好的错误提示。这些改进确保了在处理错误时有更流畅的用户体验。
- **Cascade Base 额度处理改进**：改进了当用户额度较低时 **Cascade Base** 的触发机制，提升了整体可用性。系统能更有效地捕获错误情况，减少用户的困惑。



**提到的链接**：<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf Editor 的最新更新和变更。

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1340111757099794523)** (50 条消息🔥): 

> `JetBrains 扩展更新，Windsurf 与 Codeium 扩展之争，Windsurf 登录问题，Codeium Chrome 扩展查询，VS Code 扩展崩溃` 


- **对 JetBrains 扩展更新的期待**：成员们表达了对 JetBrains 扩展更新的渴望，特别是希望刷新 DeepSeek 和 Gemini 2.0 Flash 等模型的列表，这可能以极小的努力带来巨大的价值。
   - 一份回应指出，JetBrains 的更新确实在计划中，并强调由于企业用户的需求，Codeium 扩展的维护工作一直在持续。
- **关于 Windsurf 与 Codeium 扩展的辩论**：用户对 Windsurf 和 Codeium 扩展之间缺乏功能对等（feature parity）感到沮丧，一些人觉得 Windsurf 已经让原有的扩展显得相形见绌。
   - 有批评指出该扩展似乎被“遗弃”了，相比之下，Qodo 等替代方案被频繁提及并获得好评。
- **用户在 Windsurf 登录时遇到的挑战**：多名用户报告了访问 Windsurf 的问题，包括尝试使用 mcp 文件后的问题以及登录系统宕机。
   - 相关的求助被引导至相应的支持渠道，并强调了 Windsurf 与 Codeium 扩展查询之间的区别。
- **关于 Chrome 扩展使用的疑问**：一名用户询问 Codeium Chrome 扩展是否能协助通过 CodeSignal 的测评，目前该扩展在测评中似乎缺乏支持。
   - 其他人建议查阅官方文档以获取安装和使用指南，并引发了对其性能的进一步询问。
- **VS Code 扩展频繁崩溃**：许多用户报告 VS Code 扩展频繁崩溃，需要重新加载才能恢复功能，这已成为一个持续存在的问题。
   - 类似问题的变体还包括与执行截止时间（execution deadlines）相关的错误消息，引发了对可能修复方案的讨论。



**提到的链接**：<a href="https://codeium.com/chrome_tutorial?">Chrome Tutorial | Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。

  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1340004829233152020)** (580 messages🔥🔥🔥): 

> `Windsurf Performance Issues, Cascade and Debugging, MCP Server Functionality, AI Code Suggestion Behavior, Windsurf Features and Subscriptions` 


- **Windsurf 性能和功能担忧**：用户在最近更新后遇到了显著的性能问题和 AI 回复中的幻觉（hallucinations），导致代码编辑过程中的挫败感和困难。
   - 许多用户表达了回滚到以前版本的愿望，并对 Windsurf 如何有效利用某些模型表示担忧。
- **Cascade 的自动保存行为**：有报告称 Cascade 在没有用户干预的情况下自动应用更改，导致在管理代码编辑时产生困惑。
   - 建议开启新的对话历史记录，以减轻自动建议带来的一些问题。
- **MCP Server 问题**：几位用户遇到了与 MCP Server 相关的错误，特别是集中在配置以及与 Windsurf 的兼容性上。
   - 建议包括从这些服务器中删除不兼容的代码以恢复功能。
- **AI 代码建议行为**：用户对 Claude 和 Cascade 等 AI 在代码修改中采取限制性方法表示沮丧，通常忽略了上下文或现有逻辑。
   - 大家一致认为，AI 倾向于建议调试打印（debugging prints），这会导致代码混乱和误解。
- **用户对订阅模式的参与**：围绕订阅计划展开了讨论，例如根据 Turbo Mode 等可用功能在 Pro 和 Team 计划之间切换的实用性。
   - 用户强调了他们在额度限制（credit limits）方面的体验，引发了对订阅价值与 AI 性能对比的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://itsalin.com/appInfo/?id=pearcleaner">Alin Lupascu</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/advanced">Windsurf - Advanced</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/terminal#turbo-mode">Terminal - Codeium Docs</a>: 未找到描述</li><li><a href="https://codeium.canny.io/feature-requests/p/ask-cascade-about-using-cascade">Ask Cascade about using Cascade | Feature Requests | Codeium</a>: 应该有一个专门的聊天或模式来向 Cascade 询问关于 Cascade 的问题。例如，我想在编写内容时暂时关闭自动补全。</li><li><a href="https://www.pulsemcp.com/posts/newsletter-claude-open-data-1000-mcp-servers-jailbreaking-results">Open Source Claude Data, 1000+ MCP Servers, Jailbreaking Results | PulseMCP</a>: 2025 年 2 月 15 日当周更新：开源 Claude 数据，1000+ MCP Servers，越狱结果</li><li><a href="https://tenor.com/view/monkey-shower-monkey-shower-notfoddy-gif-26328653">Monkey Shower GIF - Monkey Shower Monkey Shower - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://share.cleanshot.com/PQBHx6Kt">CleanShot 2025-02-16 at 22.59.25</a>: 上传到 CleanShot Cloud 的截图</li><li><a href="https://github.com/AiCodingBattle/mdCrawler">GitHub - AiCodingBattle/mdCrawler</a>: 通过在 GitHub 上创建账号来为 AiCodingBattle/mdCrawler 的开发做出贡献。</li><li><a href="https://youtube.com/shorts/vxGs_NxfMYA?feature=shar">Quantum Machine Learning For Fun and Profit</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1340020556212338718)** (290 条消息🔥🔥): 

> `模型加载问题、LLM 资源与建议、硬件兼容性、DeepSeek 讨论、API 与服务器使用` 


- **加载 Qwen MLX 模型的问题**：用户报告在 Macbook Ultra 上加载 Qwen MLX 模型时遇到 `ModuleNotFoundError`，建议更新到 `mlx-engine/0.6.0` 以修复。
   - 一位用户还尝试删除特定目录来解决问题，但确认即使是最新版本问题依然存在。
- **在 GPU 上运行 LLM 的建议**：讨论强调了 RX580 GPU 在运行 LLM 时的局限性，建议用户选择支持 ROCm 的较新显卡以获得更好的性能。
   - 鼓励用户通过 vast.ai 等云服务测试 GPU 模型，以便在投入硬件成本前评估性能。
- **DeepSeek 与社区项目**：社区对 DeepSeek 被标记为 CCP 项目以及在持续的技术发展中潜在的资金来源进行了轻松的讨论。
   - 对话反映了 AI 的快速进步，以及这些进步可能在不久的将来带来的突破。
- **API 与本地服务器可用性**：一位用户对本地服务器 API 需要 OpenAI API key 表示沮丧，引发了关于提高可用性的快速修复和变通方法的讨论。
   - 聊天参与者建议对代码进行简单修改，以有效地绕过本地设置的 API key 要求。
- **集成网页搜索功能**：一位参与者询问了 LM Studio 网页搜索扩展的可用性，强调了其在增强应用程序功能方面的潜在效用。
   - 社区对这类功能表现出兴趣，以增强功能并探索 LM Studio 更多集成化的用途。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/Qwen2-VL-7B-Instruct-GGUF">lmstudio-community/Qwen2-VL-7B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>: 初识 Unsloth？</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">Easy_training/Galore+Qlora_With_Multi_GPU_Support at main · rombodawg/Easy_training</a>: 通过在 GitHub 上创建账户，为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://fixupx.com/snazzylabs/status/1890927763772579977">Quinn Nelson (@SnazzyLabs) 的推文</a>: 开源软件太酷了。这是我的 Rivian R1S 正在由 Comma 3X 自动驾驶——这是一个安装在仪表盘上的设备，它接管了我汽车自动驾驶计算机的信号并插入自己的（信号）...</li><li><a href="https://github.co">GitHub · 在统一的协作平台上构建和交付软件</a>: 加入全球应用最广泛、由 AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation>)">GitHub - deepseek-ai/DeepSeek-R1</a>: 通过在 GitHub 上创建账户，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://youtu.be/EHGmPn6RVwU">DeepSeek R1: $5000 vs $1000 电脑 | M4 Max 128GB vs M1 Air 16GB | LM Studio 教程</a>: 💻 考虑为 AI 升级你的 MacBook 吗？我测试了 DeepSeek R1 的开源模型在价值 5000 美元的 M4 Max (128GB) 与 1000 美元的 M1 Air (16GB) 上的表现——结果...</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta 版本</a>: LM Studio 的 Beta 版和发布候选版</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档聊天 | LM Studio 文档</a>: 如何将本地文档作为额外上下文提供给 LLM
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1340025709875298374)** (545 条消息🔥🔥🔥): 

> `LM Studio 兼容性、GPU 性能基准测试、量化技术、RAM 与 VRAM 共享、为 AI 工作流选择 GPU`

- **LM Studio 兼容性问题**：用户报告了 LM Studio 显示硬件“不兼容”的问题，通常是因为 CPU 缺少 AVX2 指令集。
   - 确保硬件符合要求对于在 LM Studio 中搜索模型至关重要。
- **基准测试与 GPU 对比**：讨论了 RTX 4060 Ti 等 GPU 的性能，其在高效处理模型方面的价值备受关注，特别是在每秒 Token 数（tokens per second）方面。
   - 对比显示 RTX 3090 表现也很好，特别是在运行大型模型时。
- **理解量化（Quantization）**：解释了量化对模型性能的影响，并讨论了 Q4 和 Q8 等规格与显存需求及推理速度的关系。
   - 用户强调不同模型对量化的敏感度各不相同，这会影响其最终效果。
- **RAM 与 VRAM 共享能力**：用户询问了共享系统 RAM 和 VRAM 进行大型模型推理的可能性，并承认这种配置可能会导致速度变慢。
   - 使用明确标有支持全 GPU offload 能力的模型可以获得最佳效果。
- **GPU 的 AI 工作流考量**：对话强调了为 AI 任务选择合适 GPU 的重要性，提到基于 Token 处理速度，4060 Ti 提供了最佳性价比。
   - 推荐了 Deepseek R1 等模型给希望增强编程或推理任务的用户。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/thermal-paste/cooler-master-introduces-colored-ai-thermal-paste-cryofuze-5-comes-with-nano-diamond-technology">Cooler Master 推出彩色“AI 导热膏”——CryoFuze 5 采用纳米钻石技术</a>：声称其可在 -50C 至 240C 范围内“保持稳定性能”。</li><li><a href="https://www.techpowerup.com/gpu-specs/rtx-2000-ada-generation.c4199">NVIDIA RTX 2000 Ada Generation 规格</a>：NVIDIA AD107, 2130 MHz, 2816 Cores, 88 TMUs, 48 ROPs, 16384 MB GDDR6, 2000 MHz, 128 bit</li><li><a href="https://www.gigabyte.com/Graphics-Card/W7800-AI-TOP-48G#kf">Radeon™ PRO W7800 AI TOP 48G 主要特性 | 显卡 - GIGABYTE 全球</a>：未找到描述</li><li><a href="https://edu.finlaydag33k.nl/calculating%20ram%20bandwidth/">RamCalculator</a>：未找到描述</li><li><a href="https://x.com/GawroskiT/status/1890159776241447142">来自 Tomasz Gawroński (@GawroskiT) 的推文</a>：9070 XT 希望我发现的这些价格只是占位符... https://www.mlacom.si/komponente/graficne-kartice/i_3026151_acer-predator-bifrost-radeon-rx-9070-xt-oc-16g-gddr6-graficna-kartica</li><li><a href="https://docs.google.com/spreadsheets/d/1IyT41xNOM1ynfzz1IO0hD-4v1f5KXB2CnOiwOTplKJ4/edit?gid=0#gid=0">GPU AI 对比</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i69dhz/deepseek_r1_ollama_hardware_benchmark_for_localllm/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://tenor.com/view/now-old-man-the-future-is-now-old-man-gif-9677657">Now Old GIF - Now Old Man - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/mcgonagall-clapping-10points-for-gryffindor-gif-15812182730784296836">麦格教授鼓掌 GIF - 麦格教授鼓掌，格兰芬多加 10 分 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.ebay.com/itm/334622574783">Bykski 24-Pin 电源跳线连接器 (B-24PWSTR) | eBay</a>：未找到描述</li><li><a href="https://github.com/Zhen-Dong/Awesome-Quantization-Papers">GitHub - Zhen-Dong/Awesome-Quantization-Papers：近期 AI 会议和期刊中与神经网络量化相关的论文列表。</a>：近期 AI 会议和期刊中与神经网络量化相关的论文列表。 - Zhen-Dong/Awesome-Quantization-Papers</li><li><a href="https://www.youtube.com/shorts/Oo7AuhzDI7I">DeepSeek 在慢速交换机上的网络汇编代码问题 #deepseek #openai #nvidia #technews #ai</a>：DeepSeek R1 在基于旧款 Nvidia 芯片的情况下是如何在 AI 市场脱颖而出的？答案是智能负载均衡和汇编代码。Alex 解释了 Deep...</li><li><a href="https://www.hp.com/us-en/workstations/zbook-ultra.html">HP ZBook Ultra G1a 14” 移动工作站 PC – AI 笔记本电脑</a>：使用 HP ZBook Ultra G1a 释放突破性性能，应对复杂的 AI 工作流。一款时尚、轻薄、高效节能的 AI 笔记本电脑和移动工作站。</li><li><a href="https://www.hp.com/us-en/workstations/z2-mini-a.html">HP Z2 Mini G1a 工作站台式 PC – AI PC </a>：使用 Z2 Mini G1a 工作站台式机开启以前无法实现的工作流。紧凑型 AI 电脑中蕴含的变革性性能。</li><li><a href="https://www.youtube.com/shorts/77rqmeLgfOs">新款 AMD 显卡……值得吗？</a>：未找到描述</li><li><a href="https://www.pny.com/en-eu/rtx-2000e-ada-generation">NVIDIA RTX 2000E Ada Generation | 专业级 GPU | pny.com</a>：未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1340008891928416401)** (635 条消息🔥🔥🔥): 

> `Grok 3 发布，DeepSeek R1 对比 O3 Mini High，RAG 和 Repomap 效率，Gemini 模型性能，Aider 问题`

- **即将发布的 Grok 3**：Grok 3 预计很快发布，一些用户正在推测其功能以及相对于前代模型的性能提升。
   - 有报告称 Grok 3 已开启早期访问，并提到了名为 'Ara' 的语音模式等功能。
- **DeepSeek R1 对比 O3 Mini High**：用户正在比较 DeepSeek R1 和 O3 Mini High 的功能，部分用户因其推理能力而更青睐 R1。
   - 反馈表明，虽然 R1 在某些场景下表现更好，但 O3 Mini High 在其他场景中更加稳定。
- **RAG 和 Repomap 的挑战**：参与者讨论了 RAG 和 Repomap 在大型代码库中的低效问题，指出随着代码库规模的增加，收益递减。
   - 建议采用 BM25 和 RAG 之间的混合评分等额外策略来改进搜索结果。
- **Gemini 模型的性能**：几位用户分享了他们使用 Gemini 模型的经验，特别指出 Gemini Flash 2.0 提供了更快且更具成本效益的解决方案。
   - 一些人发现它与 Sonnet 相比具有更好的推理和数学能力，引发了不同 AI 模型之间的比较。
- **Aider 功能问题**：Aider 用户报告了更新后的近期 Bug，特别是命令解析和 `.dockerfile` 处理方面的问题。
   - 讨论还涉及 DeepSeek 基于训练数据和 Web 界面设置可能存在的审查制度，用户对此表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/openai/gpt-4-32k">GPT-4 32k - API, Providers, Stats</a>: GPT-4-32k 是 GPT-4 的扩展版本，具有相同的功能，但上下文长度增加了四倍，允许在单次处理中处理多达 40 页的文本。这对于...特别有益。</li><li><a href="https://www.qodo.ai/blog/rag-for-large-scale-code-repos/">RAG For a Codebase with 10k Repos</a>: 探索 qodo 如何利用 RAG 弥补 LLM 的上下文差距，确保生成式 AI 编程的质量和完整性。</li><li><a href="https://x.com/nebiusaistudio/status/1890397790250893743">Tweet from Nebius AI Studio (@nebiusaistudio)</a>: DeepSeek R1 变得更快了 👀 推出我们全新的高性能端点：- 高达 60+ tokens/秒 - 高级推理能力 - 每 1M tokens 低至 $2/$6 起。立即在 Nebius AI Studio 体验 ✨</li><li><a href="https://x.com/impenny2x/status/1891583553911001333?s=46">Tweet from Penny2x (@imPenny2x)</a>: 我的天呐。</li><li><a href="https://github.com/richardanaya/luckyshot">GitHub - richardanaya/luckyshot: A CLI tool for finding the files that count 🤠🔫</a>: 一个用于查找重要文件的 CLI 工具 🤠🔫。通过在 GitHub 上创建账号来为 richardanaya/luckyshot 的开发做出贡献。</li><li><a href="https://tenor.com/view/nacho-libre-why-but-gif-23595404">Nacho Libre GIF - Nacho Libre Why - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/math-teddy-writing-easy-hard-gif-9016309273752180756">Math Teddy GIF - Math Teddy Writing - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.cline.bot/improving-your-prompting-skills/custom-instructions-library/cline-memory-bank">Cline Memory Bank | Cline</a>: 未找到描述</li><li><a href="https://tenor.com/view/x-pum-boom-rest-my-case-gif-12707768704480985758">X Pum Boom GIF - X Pum Boom Rest My Case - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/sama/status/1890816782836904000?t=sdhwYo1QbG6Xnyj2QHO-xQ&s=19&mx=2">Tweet from Sam Altman (@sama)</a>: 我们发布了 ChatGPT (4o) 的更新。它非常棒。很快它会变得更好，团队正在紧锣密鼓地开发中。</li><li><a href="https://github.com/Aider-AI/aider/issues/3125">How to use multiple, separate open-ai compatible API endpoints at the same time · Issue #3125 · Aider-AI/aider</a>: 问题：我可以访问多个 OpenAI 兼容的端点。它们各自使用不同的 URL 和 API Key。我想混合使用这些端点，例如将一个 API 端点用于弱模型，一个用于架构...</li><li><a href="https://github.com/Abraxas-365/langchain-rust">GitHub - Abraxas-365/langchain-rust: 🦜️🔗LangChain for Rust, the easiest way to write LLM-based programs in Rust</a>: 🦜️🔗LangChain for Rust，在 Rust 中编写基于 LLM 程序的简单方法 - Abraxas-365/langchain-rust</li><li><a href="https://github.com/vespa-engine/vespa">GitHub - vespa-engine/vespa: AI + Data, online. https://vespa.ai</a>: AI + 数据，在线。https://vespa.ai。通过在 GitHub 上创建账号来为 vespa-engine/vespa 的开发做出贡献。</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Develop software autonomously.</a>: 自主开发软件。通过在 GitHub 上创建账号来为 ai-christianson/RA.Aid 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/E3guZxKpZ8">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/sama/status/1890816782836904000">Tweet from Sam Altman (@sama)</a>: 我们发布了 ChatGPT (4o) 的更新。它非常棒。很快它会变得更好，团队正在紧锣密鼓地开发中。</li><li><a href="https://v0.dev/">v0 by Vercel</a>: 与 v0 对话。通过简单的文本提示生成 UI。复制、粘贴、交付。</li><li><a href="https://github.com/HKUDS/LightRAG">GitHub - HKUDS/LightRAG: &quot;LightRAG: Simple and Fast Retrieval-Augmented Generation&quot;</a>: &quot;LightRAG: 简单且快速的检索增强生成&quot; - HKUDS/LightRAG</li><li><a href="https://github.com/Aider-AI/aider/pull/2628">Moa by gembancud · Pull Request #2628 · Aider-AI/aider</a>: 添加 Mixture of Architects (MOA) 功能。当你可以全都要时，为什么还要在 r1、o3 和 sonnet 之间做选择！概述：此 PR 引入了一个名为 &quot;Mixture of Architects&quot; 的强大新功能...
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1340014452388532265)** (53 messages🔥): 

> `Aider 配置问题, OpenRouter 模型性能, Repo maps 与上下文处理, 添加外部搜索引擎, 使用自定义文件进行基准测试` 


- **Aider 在上下文建议方面遇到困难**：用户报告 **Aider** 不再提示他们添加相关文件，特别是在切换到 **o3-mini** 等不同模型之后。
   - 一位用户提到，将项目目录结构添加到只读文件中并没有改善建议效果。
- **OpenRouter 模型可能较慢**：几位成员分享了他们在利用 **OpenRouter** 模型（特别是使用 **DeepSeek** 时）速度较慢的经历。
   - 用户讨论了通过调整设置来排除较慢的提供商，以提高性能。
- **为 Aider 创建有效的 repo maps**：用户正在尝试使用 **repo maps** 来帮助 Aider 更好地理解他们的项目上下文并建议相关文件。
   - 一些用户发现添加自定义的结构化指南有助于提高 Aider 在建议文件方面的表现。
- **为 Aider 集成外部搜索工具**：一位用户询问如何为 Aider 添加网页搜索功能，并考虑了 **Perplexity** 等选项。
   - 社区建议关注 **ra.aid**，它集成了多种 API，可与 Aider 配合进行任务规划和执行。
- **Aider 的文件路径处理困惑**：一位用户表达了挫败感，因为 Aider 从 **git working directory** 开始创建新文件，而不是遵循当前工作目录（current working directory）。
   - 这种行为与用户在其仓库中使用 `--subtree-only` 的初衷相悖。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models?q=deepseek">Models: &#x27;deepseek&#x27; | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/config/options.html#upgrading">Options reference</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://gitlab.com/saalen/ansifilter">André Simon / ansifilter · GitLab</a>：ANSI 序列过滤器</li><li><a href="https://github.com/ai-christianson/RA.Aid/">GitHub - ai-christianson/RA.Aid: Develop software autonomously.</a>：自主开发软件。通过在 GitHub 上创建账号为 ai-christianson/RA.Aid 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1340689948847767553)** (1 messages): 

> `Qwen2.5-Coder-7B, 编辑预测模型, Zeta 数据集, 微调技术` 


- **为 Zed 微调的 Qwen2.5-Coder-7B**：该仓库包含一个专门为增强 Zed 应用中的 [编辑预测 (edit prediction)](https://zed.dev/edit-prediction) 而设计的 **Qwen2.5-Coder-7B** 微调版本。
   - 该微调模型为使用 Zed 平台的用户提供了更高效的代码辅助和改进的交互体验。
- **分享训练见解**：该模型使用 [zeta 数据集](https://huggingface.co/datasets/zed-industries/zeta) 进行微调，这对性能提升至关重要。
   - 对于那些对自我微调感兴趣的人，可以获取 **DPO Fine-Tuning** [查看 Notebook](https://huggingface.co/datasets/zed-industries/zeta/blob/main/script/dpo.ipynb) 和 **SFT Fine-Tuning** [查看 Notebook](https://huggingface.co/datasets/zed-industries/zeta/blob/main/script/sft.ipynb) 的详细脚本。



**提及的链接**：<a href="https://huggingface.co/zed-industries/zeta">zed-industries/zeta · Hugging Face</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1340008613590339745)** (323 条消息🔥🔥): 

> `AI 偏见与安全、Grok 3 预期、AI 生成代码的担忧、虚拟伴侣及其使用、LLM 与推理能力` 


- **对 AI 偏见与安全的担忧**：参与者对 Elon Musk 开发 Grok 的意图表示怀疑，担心由于他的政治立场和之前在 Twitter 上的行为，其设计中可能存在潜在偏见。
   - 讨论包括在 AI 中进行合理审查的重要性，以及 AI 是否应该影响政治观点或叙事。
- **对 Grok 3 的预期**：关于 Grok 3 能力的推测显示出用户的谨慎乐观，一些人希望它在与 O1 等现有 AI 的竞争中表现出色。
   - 有人担心 Grok 可能会被用作宣传工具，这取决于其编程和 AI 安全措施。
- **AI 生成代码的漏洞**：用户注意到依赖 AI 生成代码的风险，强调此类代码可能包含安全问题，例如不安全的字符串拼接。
   - 讨论强调开发者需要批判性地审查 AI 代码，因为不良实践可能导致应用程序中的漏洞。
- **将 AI 用作虚拟伴侣**：一些用户分享了将 AI 作为虚拟伴侣的使用经验，强调了此类互动中幽默和细微的方面。
   - 一位用户幽默地承认将 ChatGPT 用作“虚拟女友”，展示了人们将 AI 融入个人生活的不同程度。
- **LLM 推理能力的不确定性**：讨论集中在 Yann Lecun 对大语言模型（LLM）推理能力的怀疑上，参与者一致认为需要进行持续探索。
   - 对话承认，虽然已经取得了进展，但 LLM 是否能真正理解或推理复杂话题仍不确定。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pollinations.diy/">Pollinations.DIY</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2502.08946">LLM 肩上的随机鹦鹉：物理概念理解的总结性评估</a>: 我们以系统的方式调查了一个被广泛提出的问题：LLM 真的理解它们所说的话吗？这与更熟悉的术语“随机鹦鹉（Stochastic Parrot）”有关。为此，我们提出了一个总结性的...</li><li><a href="https://physico-benchmark.github.io/">
      LLM 肩上的随机鹦鹉：物理概念理解的总结性评估
    </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1340028922670088243)** (22 条消息🔥): 

> `发布到 GPT Store、从知识库中删除文件、用于研究和故事创作的 Scholar GPT、O 系列模型的系统提示指南、对 Poe 的看法` 


- **GPT Store 发布问题**：一名成员在尝试发布到 GPT Store 时遇到了提示信息：**“公开操作需要有效的隐私政策 URL”**。
   - 另一名成员建议在 actions 中填写隐私政策字段，从而解决了该问题。
- **知识库文件的挑战**：关于在自定义 GPT 中缺少从知识库中删除文件选项的讨论。
   - 最终，发帖者找到了该选项，这可能与最近的一个 Bug 修复有关。
- **用于研究与故事创作的 Scholar GPT**：一位用户提出了关于 **Scholar GPT** 在研究方面与创作故事剧情相比的有效性问题。
   - 这引发了其他人的兴趣，并分享了他们的经验，特别是关注该工具的能力。
- **O 系列 AI 模型系统提示词的变化**：一名成员询问了 O 系列 AI 模型的新系统提示词指南，随后澄清了向开发者消息（developer messages）的转变。
   - 据指出，这一变化符合 OpenAI 文档中概述的推理模型规范。
- **对 Poe 的看法**：一名成员对 **Poe** 发表了批评意见，特别是其知识库功能无效。
   - 他们指出，尽管查询是用英文进行的，但模型有时会用其他语言回答。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1340024811807834134)** (51 条消息🔥): 

> `立法写作 Prompt、Prompt 学习资源、Multi-Agent 思维、MCP 实现` 


- **立法 AI 使用的谨慎性**：一位成员强调在创建立法材料时要谨慎使用 AI 工具，建议由专业人员审查输出的**每一个字**以确保安全。
   - 他们指出通常会丢弃约 **90%** 的 AI 建议，主要利用它来改进和强化自己的想法。
- **Prompt 学习的最佳站点**：一位成员建议学习 Prompt 的最佳资源是 ChatGPT 本身，推荐用户让它解释遇到的特定 Prompt。
   - 分享的一个有效 Prompt 是让 ChatGPT 分析用户输入的含义和清晰度。
- **Token 限制的挑战**：讨论集中在 AI 交互中有限的 Token 输出流带来的挫败感，成员们分享了这如何影响他们的项目。
   - 有人提到使用 Multi-Agent 设置可以帮助绕过 Token 限制，特别是对于生成较长的输出。
- **Multi-Agent 思维与 MCP**：一位成员描述了使用集成 Google 模型和 GPT 的 Multi-Agent 系统，强调了对 API 监听脚本的需求，而这可能难以维护。
   - MCP (Model Context Protocol) 被作为一种潜在解决方案引入，据称因其简化 API 交互的能力而受到客户青睐。
- **Chain of Thought 的局限性**：讨论了 AI 中 Chain of Thought Prompt 的局限性，认为它们无法使模型像 o1 和 o3 等旧模型那样思考。
   - 成员们承认，期望新模型完全复制旧模型的思考方式可能需要回归现实。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1340024811807834134)** (51 条消息🔥): 

> `立法写作 Prompt、Prompt 学习资源、年报分析、MCP (Model Context Protocol)、模型中的 Token 限制` 


- **讨论立法写作 Prompt**：一位成员对专为立法写作材料设计的 Prompt 表示感兴趣，强调了由专业人员审查 AI 生成内容准确性的重要性。
   - 强调在重要语境中使用 AI 时要*仔细考虑*，确保对每一份输出进行彻底检查。
- **Prompt 学习的最佳资源**：一位成员建议 ChatGPT Web 界面是 Prompt 学习的绝佳资源，允许用户了解 AI 如何解释 Prompt。
   - 分享的一个热门 Prompt 示例是指令 ChatGPT 解释所提供 Prompt 的含义，从而增强用户对有效 Prompting 的理解。
- **使用 ChatGPT 分析年报**：一位成员寻求关于使用 ChatGPT 分析和比较年报的建议。
   - 该话题鼓励讨论利用 AI 进行详细分析，同时确保输出连贯且相关。
- **探索 Model Context Protocol (MCP)**：讨论了使用 MCP 高效连接多个 LLM，在不遇到 Token 限制的情况下增强项目能力。
   - 一位成员分享了他们目前的设置，利用 Google 模型进行逻辑处理，利用 ChatGPT 进行脚本编写，称其为智能解决方案。
- **Token 限制的挑战**：人们对 GPT 模型有限的输出和 Token 流表示担忧，这影响了它们在各种任务中的效力。
   - 成员们指出存在时间限制和 Token 约束，这使交互变得复杂，而其他人则分享了规避这些挑战的策略。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1340004850607591446)** (369 条消息🔥🔥): 

> `Stable Diffusion 版本, LORA 训练, 运行本地 AI, 图像生成技术, 使用 ComfyUI` 


- **Stable Diffusion 1.7.0 表现良好**：用户注意到 **Stable Diffusion 1.7.0** 比 1.10 更高效，加载速度更快，且对 Prompt 的遵循度更好。
   - 一位用户提到 1.10 经常生成不令人满意的图像，理由是加载时间更长且结果不稳定。
- **LORA 训练技巧**：为了获得更好的训练效果，建议针对你的概念使用约 **50-150 张图像** 进行 LORA 训练。
   - 用户讨论了使用 **Koyha LORA** 等工具来根据特定风格简化训练过程。
- **本地 AI 配置建议**：多位用户提倡在本地运行 AI 模型，强调了其相比在线服务所提供的控制权和定制化能力。
   - 为了获得最佳性能，建议配备拥有 **8GB VRAM** 的 NVIDIA GPU 和 **32GB RAM**。
- **图像生成技术**：用户分享了创建图像的方法，包括利用 **regional prompting**（区域提示词）以及高效排列 Prompt 中的词汇。
   - 一位用户强调了清晰的 Prompt 结构对于产生更好图像结果的重要性。
- **ComfyUI 的使用与特性**：用户讨论了 ComfyUI 的上手难度，但也承认了它在图像生成工作流中的灵活性和强大功能。
   - 诸如**拖放**上传图像等功能被强调为 AI 生成的便捷工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Lightricks/LTX-Video/tree/main">Lightricks/LTX-Video at main</a>: 未找到描述</li><li><a href="https://tenor.com/view/door-hide-close-bye-cat-gif-22820275">Door Hide GIF - Door Hide Close - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://youtu.be/TVHnguyd06g">The Rise And Fall Of Midjourney</a>: 在 2022 年，Midjourney 势不可挡。它是首选的 AI 图像生成器，利用网络效应、数据驱动学习和病毒式营销占据主导地位...</li><li><a href="https://github.com/stepfun-ai/Step-Video-T2V">GitHub - stepfun-ai/Step-Video-T2V</a>: 通过在 GitHub 上创建账号为 stepfun-ai/Step-Video-T2V 的开发做出贡献。</li><li><a href="https://stablediffusionweb.com/app/image-generator">Stable Diffusion Online</a>: 未找到描述</li><li><a href="https://www.youtube.com/shorts/EWAGzKCMhuE">Trump and Putin Jamming: AI Music Video With World Leaders Epic Jam!</a>: #politics #donaldtrump #kamalaharris #vladimirputin #macron #trudeau #politician #celebrity #ai #viralvideo #trending #trendingvideo #shortvideo #worldleader...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/cf2772fab0af5573da775e7437e6acdca424f26e">GitHub - AUTOMATIC1111/stable-diffusion-webui at cf2772fab0af5573da775e7437e6acdca424f26e</a>: Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://www.youtube.com/shorts/mhSMgSFdYsM">Evil Comes Back Biting #motivation #lifelessons #motivationalspeech #wisdom #mindset</a>: 善有善报，恶有恶报，往往以意想不到的方式发生！🌟 在这个鼓舞人心的短篇故事中，一个男人面临着一个艰难的决定，最终向我们展示了行善的力量...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases">Releases · AUTOMATIC1111/stable-diffusion-webui</a>: Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1340015359708626974)** (344 条消息🔥🔥): 

> `Claude 模型性能, MCP 服务器设置, Cursor 更新, Grok 3 推出, 模型对比`

- **Claude 表现出性能下降的迹象**：用户注意到 Claude 的响应在过去几天变得更慢且上下文感知能力减弱，一些人将其称为“削弱（nerfed）”。几位成员对这些变化表示沮丧，并讨论了替代模型。
   - 有人建议 Claude 可能正在经历内部调整，这些调整在没有明显模型更新的情况下影响了其性能。
- **围绕 MCP 服务器设置的讨论**：用户分享了 MCP 服务器的各种设置，特别是使用 Brave Search 和 Sequential Thinking 来增强性能。有人指出，这些设置可以显著提高编码过程的可靠性和文档化程度。
   - 一位用户还在寻求 MCP 的简单 'hello world' 示例，表明了对更易获取资源的需求。
- **Grok 3 在好奇声中推出**：Grok 3 的发布引发了用户的好奇，一些人想知道它与现有模型相比的性能。有人推测后端可能进行了更新，使其更接近较新版本的 Llama。
   - 用户表示愿意测试 Grok 3，特别是为了验证其在过去几次反响平平的发布经历后的能力。
- **Cursor 对用户请求的处理**：人们对 Cursor 处理大量任务和有效处理大文件的能力表示担忧，特别是在 Agent 模式下。用户讨论了尽管拥有无限 Token，但对平台施加的限制感到沮丧。
   - 有建议称，由于这些限制，Cursor 用户可能需要探索其他模型和功能。
- **新兴服务与滥用担忧**：人们对可能允许滥用 Cursor 服务免费试用的漏洞表示担忧，引发了关于安全和监管的讨论。用户提到了可能利用系统推出的自动化脚本，并讨论了如何处理此类问题。
   - 社区对更新和服务可靠性表示沮丧，敦促 Cursor 团队对滥用行为保持警惕。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: 未找到描述</li><li><a href="https://trypear.ai/">PearAI - 开源 AI 代码编辑器</a>: PearAI 是一款开源的 AI 驱动代码编辑器，具有 AI 聊天、PearAI Creator 和 AI 调试等强大功能，助你创造心仪之作。</li><li><a href="https://x.com/btibor91/status/1890703686281867744?s=46">Tibor Blaho (@btibor91) 的推文</a>: 尝试 Anthropic 的新 Thinking 模型。在模型设置中开启 **Thinking** 以查看 Claude 的思考过程。为编程和数学等复杂任务获取分步推理。非常适合理解...</li><li><a href="https://x.com/lukas_troup/status/1889698021371224148?s=46">Lukas Troup (@lukas_troup) 的推文</a>: 我测试了多个 AI 工具，将简单的 Figma UI 设计转换为代码，看看哪一个效果最好。你在截图里看到的结果是基于第一个 prompt 生成的（最多...</li><li><a href="https://tenor.com/view/aint-nobody-got-time-for-that-kimberly-wilkins-interview-gif-3531013">Time GIF - Aint Nobody Got Time For That Kimberly Wilkins 采访 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity: 针对 Perplexity API 的 MCP Server。</a>: 针对 Perplexity API 的 MCP Server。通过在 GitHub 上创建账号来为 daniel-lxs/mcp-perplexity 的开发做出贡献。</li><li><a href="https://x.com/lovable_dev/status/1890075209748902304?s=46">Lovable (@lovable_dev) 的推文</a>: 推出 Visual Edits。你现在可以在 Lovable 中可视化地编辑任何样式，从而实现更快、更精确的编辑。</li><li><a href="https://x.com/iBigQiang/status/1890993390172406160">BigQiang (@iBigQiang) 的推文</a>: 在 B 站看到个使用Cloudflare无限续杯Cursor教程和配套脚本实在太牛了，条件是拥有域名并且需要交给cloudflare托管，不过域名也就几块钱而已，相比每个月20刀的Cursor还是很香了。Cursor能写代码、写文章，大部分ai能干的事他都可以，可白嫖Claude 3.5 sonnet，deepseek v3&r1，gpt4o和gpt4o mini等</li><li><a href="https://forum.cursor.com/t/o3-mini-is-live-what-version-are-we-getting/46674/38">O3-mini 已上线！我们得到的是哪个版本？</a>: 确认 o3-mini 已在 Cursor 中更新以使用高推理级别。请告诉我们你的想法 🙂</li><li><a href="https://x.com/kimmonismus/status/1891590879430754550">Chubby♨️ (@kimmonismus) 的推文</a>: Grok-3 开始推出了。检查你的 Grok 选项。引用 Penny2x (@imPenny2x) 的话：天哪。</li><li><a href="https://github.com/bgstaal/multipleWindow3dScene">GitHub - bgstaal/multipleWindow3dScene: 一个关于如何使用 three.js 和 localStorage 在多个窗口之间“同步” 3D 场景的快速示例</a>: 一个关于如何使用 three.js 和 localStorage 在多个窗口之间“同步” 3D 场景的快速示例 - bgstaal/multipleWindow3dScene</li><li><a href="https://youtu.be/jUmUxtvZFIE?si=xhso29DQNkY0RFra">Tavily MCP: 尝试复制 Deep Research (设置与演示)</a>: Tavily 刚刚发布了他们的官方 MCP server，为 Claude 带来了强大的搜索能力。与此同时，OpenAI 的 Deep Research 也引起了轰动...</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/VhbuwHrwem">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://youtu.be/RCFe1L9qm3E?si=V4v8Y8XT1MkhKx_O">Cursor + MCP Servers: 完整设置指南 (Sequential Thinking, Brave Search 等)</a>: Cursor 刚刚添加了 MCP 支持！在这份完整的设置指南中，我将向你展示如何集成和使用 MCP servers (Sequential Thinking, Brave Search 和 Puppe...</li><li><a href="https://github.com/chengazhen/cursor-auto-free">GitHub - chengazhen/cursor-auto-free: 自动注册 Cursor</a>: 自动注册 Cursor。通过在 GitHub 上创建账号来为 chengazhen/cursor-auto-free 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1341069434575655072)** (2 条消息): 

> `OpenRouter 实时图表，llms.txt 文档` 


- **OpenRouter 增强实时分析**：OpenRouter 的吞吐量和延迟图表现在支持实时更新，这一显著改进归功于 **Google AI Vertex** 的增强。您可以在他们的 [公告推文](https://x.com/OpenRouterAI/status/1891510121139769542) 中查看详细变更。
   - 最近的提速获得了好评，显著提升了用户体验。
- **推出用于文档访问的 llms.txt**：对于在 **OpenRouter** 上进行开发的开发者，新发布的 **llms.txt** 可作为与文档进行对话的全面资源。您可以点击 [此处](https://openrouter.ai/docs/llms.txt) 访问，或点击 [此处](https://openrouter.ai/docs/llms-full.txt) 下载完整版本。
   - 该文档因其详尽的内容而受到赞赏，邀请用户以创意方式与文档互动，详见其 [推文](https://x.com/OpenRouterAI/status/1891524549457441170)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1891510121139769542">OpenRouter (@OpenRouterAI) 的推文</a>: 提示：OpenRouter 上的吞吐量和延迟图表支持实时更新。这是 Sonnet 的数据。感谢 @GoogleAI Vertex 最近的提速！</li><li><a href="https://x.com/OpenRouterAI/status/1891524549457441170">OpenRouter (@OpenRouterAI) 的推文</a>：刚刚发布了一个包含我们所有文档的长篇美观的 llms.txt。你知道该怎么做！✨
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1340354503983235123)** (5 条消息): 

> `多功能 AI Tooltip 扩展，Toledo1 AI 浏览器，在线派对游戏，Roo-Code URL 更新，Favicon 问题` 


- **发布多功能 AI Tooltip 扩展**：一款新的 Chrome 扩展 **Versatile AI Tooltip** 允许用户通过简单的 API key 设置，使用 **OpenRouter** 模型快速处理文本并自定义指令。
   - 据开发者分享，其目标是免费实现文章摘要和文本片段翻译。
- **Toledo1 提供按提问付费的 AI 访问**：**Toledo1** 平台允许与 AI 助手进行私密聊天，采用**按提问付费**模式，并能够结合多个 AI 以获得准确答案。
   - 它具有易于安装的客户端搜索功能，拥有企业级安全性，且无需订阅费。
- **令人兴奋的在线派对游戏发布**：一款在线派对游戏已上线，玩家可以使用任何能想到的动作与朋友或机器人对战，无需注册即可免费访问。
   - 玩家可以在 [Wits and Wands](https://witsandwands.com/) 体验游戏并进行独特的战斗。
- **请求更新 Roo-Code 的 URL**：有建议提出更新应用展示中的 **Roo-Code 新 URL**，这也将解决正确的 favicon 显示问题。
   - 提供的新 URL 为 [Roo-Code Documentation](https://docs.roocode.com/)，并分享了一个 favicon 的图片链接，但强调这在他们端是自动处理的。
- **提出的自动 Favicon 问题**：一位用户指出，应用展示的 favicon 问题与 **Roo-Code** 未更新其 header 有关，这使其成为服务器端的自动处理问题。
   - 这说明了应用维护者与 URL 内容提供商之间的沟通间隙会影响显示元素。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://witsandwands.com/">Wits and Wands - AI 驱动的魔法决斗游戏</a>：一款 AI 驱动的魔法决斗游戏，你可以施放咒语，智胜朋友并赢得胜利！</li><li><a href="https://toledo1.com/">Toledo1</a>：未找到描述</li><li><a href="https://chromewebstore.google.com/detail/versatile-ai-tooltip/jdldlpaafnfeggaopbmelkmgagikfekm)">Chrome 网上应用店</a>：为您的浏览器添加新功能并个性化您的浏览体验。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1340006264062869656)** (315 条消息🔥🔥): 

> `DeepSeek R1, OpenRouter 模型性能, Rate limits 与 API 使用, 多模态模型, 响应中的推理 Token`

- **DeepSeek R1 的可变推理包含**：用户报告了 DeepSeek R1 免费模型的不一致性，推理 token 有时出现在响应内容中，而不是作为独立的推理字段，这表明模型行为可能存在问题。
   - OpenRouter 团队已意识到该问题，并正在跟踪解决方案，以更好地处理开源推理模型的行为。
- **速率限制与 API 管理**：围绕研讨会期间使用的付费模型的速率限制展开了讨论，OpenRouter 保证通常情况下用户不会达到速率限制，除非需求大幅激增。
   - OpenRouter 在全球范围内管理速率限制，并建议如果出现任何问题，可以使用多个模型来分担负载。
- **关于数据隐私的担忧**：一位用户对强制路由到特定国家可能违反数据保护法表示担忧，强调了为了合规性需要基于区域的路由选项。
   - OpenRouter 承认了这一问题，并正在探索支持欧盟特定路由的选项，以实现更好的法律合规性。
- **多模态模型评估**：用户分享了包括 Sonnet 3.5、Flash 2.0 和 o3-mini-high 在内的多个模型在编程任务中的性能反馈，强调了 o3-mini-high 尽管在复杂问题上实现了 zero-shot 能力，但表现出异常行为。
   - 还讨论了模型在国际象棋等任务中的有效性，共识是当前的 LLM 在此类场景中表现吃力。
- **基准测试与模型测试**：有人询问了用于测试各种模型的可用基准测试脚本，特别是关于性能一致性方面。
   - 用户报告在使用 Claude 3.5 Sonnet 时遇到 'premature close' 错误，引发了关于处理来自 API 请求的无效响应体的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://openrouter.ai/provider/inference-net">inference.net | OpenRouter</a>: 浏览由 inference.net 提供的模型</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 保护。有效地配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/docs/features/structured-outputs#using-structured-outputs">Structured Outputs - 来自 AI 模型的类型安全 JSON 响应</a>: 对 AI 模型响应强制执行 JSON Schema 验证。利用 OpenRouter 的结构化输出功能，获得一致的、类型安全的输出并避免解析错误。</li><li><a href="https://openrouter.ai/openai/chatgpt-4o-latest">ChatGPT-4o - API、提供商、统计数据</a>: OpenAI ChatGPT-4o 由 OpenAI 持续更新，指向 ChatGPT 使用的当前 GPT-4o 版本。因此，它与 [GPT-4o](/models/openai/gpt-4o) 的 API 版本略有不同...</li><li><a href="https://simulateagents.com/chess/3/">gemini flash 2.0 vs o3 mini</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/use-cases/byok#azure-api-keys">BYOK - 自带密钥至 OpenRouter</a>: 了解如何在 OpenRouter 中使用您现有的 AI 提供商密钥。在利用 OpenRouter 统一界面和功能的同时，集成您自己的 API 密钥。</li><li><a href="https://x.com/deepseek_ai/status/1890324295181824107">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🎉 很高兴看到大家对部署 DeepSeek-R1 的热情！以下是我们推荐的最佳体验设置：• 无 System Prompt • Temperature: 0.6 • 搜索和文件上传的官方提示词...</li><li><a href="https://x.com/teortaxesTex/status/1891336616657981562">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>: 确实令人惊叹。但你必须承认，扩散视频模型处理高频平面内运动的方式（例如看这里狗的后腿）仍然揭穿了 Sama 的...</li><li><a href="https://huggingface.co/stepfun-ai/Step-Audio-Chat">stepfun-ai/Step-Audio-Chat · Hugging Face</a>: 未找到描述</li><li><a href="https://www.google.com/search?q=free+limits+openrouter">Google 搜索</a>: 未找到描述</li><li><a href="https://chromewebstore.google.com/detail/versatile-ai-tooltip/jdldlpaafnfeggaopbmelkmgagikfekm)">Chrome 网上应用店</a>: 为您的浏览器添加新功能并个性化您的浏览体验。</li><li><a href="https://tenor.com/view/jonah-hill-no-chill-out-bro-nah-fam-gif-16233135">Jonah Hill No GIF - Jonah Hill No Chill Out Bro - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot#reasoning)?">Chatbot</a>: 了解如何使用 useChat hook。</li><li><a href="https://github.com/n4ze3m/page-assist">GitHub - n4ze3m/page-assist: 使用本地运行的 AI 模型辅助您的网页浏览</a>: 使用本地运行的 AI 模型辅助您的网页浏览 - n4ze3m/page-assist</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations">GitHub - deepseek-ai/DeepSeek-R1</a>: 通过在 GitHub 上创建账号，为 deepseek-ai/DeepSeek-R1 的开发做出贡献。</li><li><a href="https://openrouter.ai/docs/community/frameworks#using-the-openai-sdk">集成框架 - OpenRouter SDK 支持</a>: 使用流行的框架和 SDK 集成 OpenRouter。提供 OpenAI SDK、LangChain、PydanticAI 和 Vercel AI SDK 集成的完整指南。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">提供商路由 - 智能多提供商请求管理</a>: 智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由功能优化成本、性能和可靠性。</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: 集 Shell 助手、Chat-REPL、RAG、AI 工具和 Agent 于一体的 LLM CLI 工具，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。</a>: 集 Shell 助手、Chat-REPL、RAG、AI 工具和 Agent 于一体的 LLM CLI 工具，支持访问 OpenAI, Claude, Gemini, Ollama, Groq 等。- sigoden/aichat</li><li><a href="https://github.com/BuilderIO/ai-shell">GitHub - BuilderIO/ai-shell: 一个将自然语言转换为 Shell 命令的 CLI。</a>: 一个将自然语言转换为 Shell 命令的 CLI。- BuilderIO/ai-shell</li><li><a href="https://www.npmjs.com/package/@plastichub/kbot">@plastichub/kbot</a>: 用于代码修改和项目管理的 AI 驱动命令行工具，支持多个 AI 模型和路由。最新版本：1.1.14，最后发布于 3 天前。开始使用 @plastich...</li>

ub/kb...</li><li><a href="https://yuewen.cn/chats/new">跃问</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1340014778822688911)** (152 messages🔥🔥): 

> `Grok 3 Release, Mistral Saba Model, Emerging Technologies in Robotics, Tencent's New LLMs, Character Training in AI` 


- **Grok 3 发布备受期待**：Elon Musk 宣布将发布 **Grok 3** 并进行现场演示，声称它是“地球上最聪明的 AI”，这引发了关于其是否符合用户需求的复杂反应。
   - 有人担心该模型是否能得到有效训练，或者是否会为了文化偏见而进行过度调整，这表明了对其商业化的怀疑态度。
- **Mistral Saba 的区域重点**：**Mistral Saba** 是一个 24B 参数模型，专为中东和南亚设计，强调通过定制训练能力来改进语言表达、细微差别和文化背景。
   - 这一举措反映了针对特定区域市场量身定制 AI 模型，而非仅仅依赖通用模型的日益增长的趋势。
- **腾讯扩展 AI 产品线**：腾讯宣布了其 **Hunyuan** LLM Turbo-S 和视频生成模型的计划，预计将于 2025 年第一季度发布，展示了其在 AI 领域不断增长的实力。
   - 该公告突显了腾讯巩固其在竞争激烈的 AI 领域（尤其是视频技术领域）地位的雄心。
- **AI 模型中的 Character Training**：关于 **ChatGPT** 最近更新影响的讨论引发了对 Character Training 的见解，认为其概念简单但执行复杂。
   - 用户注意到模型内部的个性化程度不断提高，这引发了关于 AI 交互中信任和可靠性影响的问题。
- **视频生成的挑战**：由于内容审核和防止有害输出的挑战，视频生成技术的发展陷入停滞。
   - 参与者讨论了利用先进工具检测和对抗视频生成滥用的潜力，强调了采取强有力保护措施的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.10248">Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model</a>: 我们介绍了 Step-Video-T2V，这是一个拥有 30B 参数的先进 text-to-video 预训练模型，能够生成长达 204 帧的视频。一个深度压缩的 Variational Autoencoder...</li><li><a href="https://fxtwitter.com/anissagardizy8/status/1890483681476686177">Tweet from Anissa Gardizy (@anissagardizy8)</a>: 新消息：据知情人士透露，埃隆·马斯克的人工智能初创公司 xAI 正寻求建立一个新的数据中心，计划大幅增加其使用的 Nvidia 芯片数量...</li><li><a href="https://x.com/technology/status/1890500458172588159">Tweet from Bloomberg Technology (@technology)</a>: 据称人形机器人初创公司 Figure AI 正在洽谈以近 400 亿美元的估值融资 15 亿美元。https://trib.al/Fpe172t</li><li><a href="https://mistral.ai/en/news/mistral-saba">Mistral Saba | Mistral AI</a>: 为特定地区、市场和客户提供服务的众多定制训练模型之一</li><li><a href="https://podcasts.apple.com/us/podcast/the-world-next-week/id306597476?i=1000692356064">New Podcast Spotlight: The Interconnect</a>: 播客单集 · The World Next Week · 2025/02/14 · 30分钟</li><li><a href="https://x.com/bdsqlsz/status/1891507545325404639">Tweet from 青龍聖者 (@bdsqlsz)</a>: 腾讯混元团队宣布将在 2025 年第一季度公开发布其 LLM Turbo-S 和 HunyuanVideo I2V</li><li><a href="https://x.com/markgurman/status/1890523191967134072">Tweet from Mark Gurman (@markgurman)</a>: 消息快报：据消息人士透露，苹果公司承诺已久的 Siri 数字助理 AI 彻底改革正面临最后的工程和软件漏洞，可能导致其推迟或限制发布。https://ww...</li><li><a href="https://x.com/elonmusk/status/1890959216724115790)"">Tweet from Elon Musk (@elonmusk)</a>: 整个周末将与团队一起磨合产品，在此之前处于离线状态</li><li><a href="https://x.com/karpathy/status/1891213379018400150">Tweet from Andrej Karpathy (@karpathy)</a>: 实际上我挺喜欢新的 ChatGPT 4o 性格，无论他们做了什么——它更随性/口语化，感觉更像是在和朋友聊天，而不是和 HR 沟通- ...</li><li><a href="https://x.com/testingcatalog/status/1891192524137541977">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: 腾讯正在微信 App 内测试 DeepSeek 搜索 👀 微信是 WeChat 的中国版，拥有 13.8 亿用户 🤯</li><li><a href="https://x.com/natolambert/status/1890897730765209718?s=46)">Tweet from Nathan Lambert (@natolambert)</a>: 我正在联系前沿实验室的技术负责人（主要是经理和高级人员），以了解他们如何管理人员、训练复杂性以及发布新模型的速度。纯粹是...</li><li><a href="https://x.com/legit_rumors/status/1891206761618591933">Tweet from ʟᴇɢɪᴛ (@legit_rumors)</a>: Grok 3 - 如果默认开启推理/思考功能，我能理解为什么 xAI 对他们的成果如此兴奋。我也开始相信马斯克关于“地球上最聪明 AI”的说法了...</li><li><a href="https://x.com/elonmusk/status/1891112681538523215">Tweet from Elon Musk (@elonmusk)</a>: Grok 3 太赞了 😂</li><li><a href="https://x.com/elonmusk/status/1890958798841389499">Tweet from Elon Musk (@elonmusk)</a>: Grok 3 将于太平洋时间周一晚上 8 点发布并进行现场演示。地球上最聪明的 AI。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1340023282698682532)** (2 messages): 

> `Reinforcement Learning Approaches, Soft Fine-Tuning Initialization` 


- **强化学习强化成功的方法**：在 Reinforcement Learning 中，采用**多种方法**来解决问题，成功的方法会随着时间的推移得到强化。
   - **初始化**过程保持**灵活性**，允许根据各种策略的成功情况进行调整。
- **理解 Soft Fine-Tuning 初始化**：讨论指出 **Soft Fine-Tuning (SFT)** 在学习过程中充当初始化方法。
   - 这种方法被强调为优化模型性能的基础步骤。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1340078024837562408)** (11 messages🔥): 

> `Elon Musk 对 OpenAI 的收购邀约，OpenAI 董事会声明，Kylie Robison 的沟通信息，Grok 3 发布` 


- **OpenAI 董事会拒绝 Musk 的收购邀约**：OpenAI 董事会已正式拒绝了 Elon Musk 最近提出的 **974 亿美元** 控股该公司的邀约，并表示：*“OpenAI 是非卖品。”*
   - 董事长 Bret Taylor 强调，董事会一致拒绝了 Musk 破坏竞争的企图，正如 [Bloomberg](https://www.bloomberg.com/news/articles/2025-02-14/openai-board-rejects-musk-s-97-4-billion-bid-to-control-company?srnd=undefined) 所报道。
- **Musk 的公关与媒体报道**：Kylie Robison 分享说，她的收件箱充满了敦促她报道涉及 Musk 的持续 **Twitter 争端** 和诉讼的沟通信息，并表示 *“我们中必须有一个人做出选择！”*
   - 她承认，虽然这些沟通信息很客气，但这些 *“内斗（cat fights）”* 往往是精心策划的头条新闻，她只有在涉及法律程序时才会进行报道。
- **Grok 3 备受期待的发布**：Elon Musk 宣布 **Grok 3** 的发布将于 **太平洋时间周一晚上 8 点** 进行现场演示，并宣传其为 *“地球上最智能的 AI”*。
   - Musk 的声明引发了讨论，包括一个提醒：*“一个人为了告诉我们 Grok 3 到底有多好而牺牲了，”* 这指的是其中涉及的巨大赌注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/kyliebytes/status/1890901349975359592?s=61">来自 Kylie Robison (@kyliebytes) 的推文</a>: 明确地说，他的公关团队很客气，并试图让我了解情况，但他的那些内斗也是精心策划的头条新闻——在律师介入之前，我不会报道它们！</li><li><a href="https://x.com/shiringhaffary/status/1890516140260028738">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>: 新闻：OpenAI 董事会正式拒绝了 Elon Musk 购买该非营利组织的投资者报价。“OpenAI 是非卖品，董事会一致拒绝了 Musk 先生最近破坏其竞争对手的尝试...”</li><li><a href="https://x.com/kyliebytes/status/1890900249008386435?s=61">来自 Kylie Robison (@kyliebytes) 的推文</a>: 请注意，这些是我从他的公司收到的电子邮件。引用 Kylie Robison (@kyliebytes) @sama 你的公关团队字面上是在打电话让我们报道 Twitter 上的争端和诉讼——我们中必须有一个人做出选择！...</li><li><a href="https://x.com/aidan_mclau/status/1891243031090626776">来自 Aidan McLaughlin (@aidan_mclau) 的推文</a>: 一个人为了告诉我们 Grok 3 到底有多好而牺牲了，永远不要忘记</li><li><a href="https://x.com/justinlin610/status/1891038100748398937?s=46">来自 Junyang Lin (@JustinLin610) 的推文</a>: 噢真的吗？引用 Elon Musk (@elonmusk) Grok 3 发布会及现场演示将于太平洋时间周一晚上 8 点举行。地球上最智能的 AI。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1340005301222375504)** (58 条消息🔥🔥): 

> `写作建议与挑战、David Perrell 的写作见解、转型为 Tech Lead 角色、LLM 中世界模型的涌现、AI 模型的最新进展` 


- **应对写作的困境**：成员们讨论了写作的挑战，特别是对于非母语使用者而言，以及如何找到自己的风格。
   - 建议从个人经验中汲取灵感，以提升写作质量。
- **David Perrell 的写作框架**：David Perrell 的写作方法强调强有力的标题、引人入胜的第一句话以及第一段中清晰的方向。
   - 虽然被认为带有较强的主观色彩，但他的内容可以为有抱负的作者提供一个有趣的框架。
- **对准 Tech Lead 的建议**：成员们分享了转型为 Tech Lead 角色的见解，强调了经验和人际交往能力的重要性。
   - 接受从个人贡献者到领导角色的转变至关重要，同时也要为团队成员争取利益。
- **Melanie Mitchell 与 AI 中的世界模型**：讨论包括了 Melanie Mitchell 关于 LLM 的文章见解，以及它们通过游戏分析开发“世界模型”的能力。
   - 成员们注意到了圣塔菲研究所（Santa Fe Institute）创新但抽象的概念，同时也赞赏 Mitchell 见识广博的观点。
- **对 AI 模型版本的反馈**：用户对新 4o 模型的体验表明，在日常对话和日常任务中，它比之前的版本更受欢迎。
   - 关于其编程能力与 Claude 等替代方案的对比仍在进行中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/ibm-granite/GneissWeb.bloom">ibm-granite/GneissWeb.bloom · Hugging Face</a>: 未找到描述</li><li><a href="https://www.scarletink.com">Scarlet Ink | Dave Anderson | Substack</a>: 来自前 Amazon 总经理和技术总监的科技行业职业与领导力建议。点击阅读 Scarlet Ink，作者 Dave Anderson，这是一个拥有数万订阅者的 Substack 出版物。</li><li><a href="https://open.substack.com/pub/aiguide/p/llms-and-world-models-part-2?r=1guc4&utm_medium=ios">LLMs and World Models, Part 2</a>: LLM 中涌现世界模型的证据（及反证）</li><li><a href="https://x.com/TheXeophon/status/1890794528518316112">Xeophon (@TheXeophon) 的推文</a>: 🤔IBM 拥有 GneissWeb 的代码库、博客和工具，这可能是他们的预训练数据集。它已被删除，但仍被索引</li><li><a href="https://x.com/OedoSoldier/status/1891452194680938947">OedoSoldier (@OedoSoldier) 的推文</a>: @teortaxesTex</li><li><a href="https://x.com/savvyRL/status/1890465151574254057">Rosanne Liu (@savvyRL) 的推文</a>: Tülu 3 正在 DLCT 上展示！由 @pdasigi 领导。@allen_ai 的颜色加上 🩷 作为作者上标，基本上是在大喊“情人节快乐”</li><li><a href="https://x.com/andrewwhite01/status/1891514945554227554">Andrew White 🐦‍⬛ (@andrewwhite01) 的推文</a>: 这是最近推理模型和非推理模型在数学问题（AIME 2024/Math 500）上的评估图表。你可以看到来自不同团队的推理模型展现出了巨大的算法增...</li><li><a href="https://x.com/arcprize/status/1890464921604719103">ARC Prize (@arcprize) 的推文</a>: 介绍 SnakeBench，一个实验性的基准测试支线任务。我们让 50 个 LLM 在贪吃蛇 🐍 中进行两两对决。2.8K 场比赛展示了哪些模型最擅长贪吃蛇实时策略和空间...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1340042086283477022)** (21 messages🔥): 

> `OpenAI's O1 Pro Mode usage, Deep Research Behavior, AI Cookie Collection Fact, Strawberry Question Controversy, ChatGPT's Elaborate Descriptions` 


- **OpenAI 的 O1 Pro Mode 使用引发关注**：成员们质疑人们多久会要求 O1 Pro 模式数一次单词 **strawberry** 中 'R' 的个数，暗示了用户与 AI 模型之间幽默的互动。
   - *这引发了讨论*，关于此类请求的荒谬性及其对用户行为的影响。
- **深度研究（Deep Research）在同行中受到质疑**：一位成员幽默地反思自己因为阅读了多篇 DeepSeek 论文而显得格格不入，建议该小组应该雇人来读论文。
   - 这在社区内引发了一场关于深度研究严肃性的有趣辩论。
- **AI 收集 Cookie 的魅力**：据报道，一名 Operator 工作了 3 小时来收集 Cookie，引发了人们对 OpenAI 运营成本的好奇。
   - 成员们推测了此类任务的财务影响，其中一人幽默地猜测是 **$200.01**。
- **Strawberry 问题引发争议**：*出现分歧*，有评论认为预览模型无法回答简单问题，导致成员们质疑 AI 回答的可靠性。
   - 这让其他人参与了一场轻松的辩论，讨论在争议中是否应该卖掉他们的 **IBM 股票**。
- **ChatGPT 的冗长描述遭到批评**：ChatGPT 对向下滚动等简单任务的解读被开玩笑地批评为过于复杂，将其过程描述为**深度研究（deep research）**。
   - 另一位用户发表了幽默的看法，暗示它可能只是在重复同样的信息。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/creatine_cycle/status/1890819089712509165?s=46">来自 atlas (@creatine_cycle) 的推文</a>：祝贺性生活，lex</li><li><a href="https://fxtwitter.com/venturetwins/status/1890451607671767116">来自 Justine Moore (@venturetwins) 的推文</a>：祝所有 LLM 情人节快乐，结果我被 Claude 狠狠地发了“好人卡” 🫠</li><li><a href="https://x.com/fleetingbits/status/1890561197587439650">来自 FleetingBits (@fleetingbits) 的推文</a>：Operator 工作了 3 小时为我收集 Cookie...</li><li><a href="https://x.com/TheXeophon/status/1891124941078061304">来自 Xeophon (@TheXeophon) 的推文</a>：兄弟</li><li><a href="https://x.com/allgarbled/status/1890314331805610143">来自 gabe (@allgarbled) 的推文</a>：Sam Altman 为此收了我 $200。引用 OpenAI Developers (@OpenAIDevs)：使用 o 系列模型处理非结构化数据、大海捞针、改进代码或处理其他复杂任务。F...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1340033272796807239)** (17 messages🔥): 

> `GRPO, SFT, Training Metrics, Markdown Documentation, CoT vs GRPO Comparison` 


- **GRPO 和 SFT 是互补的**：一位成员指出 **GRPO** 并非旨在取代 **SFT**；相反，两者应该结合使用以获得最佳效果。
   - *这表明用户更倾向于采用协作式的模型训练方法。*
- **相比 X 更倾向于使用 Notion**：一位成员表达了对使用 X 的沮丧，表示他们更喜欢用 **Notion** 记录发现，因为不需要登录那个不太受欢迎的网站。
   - 其他人对 Notion 的可用性持有复杂的情绪，评论其速度慢且导航困难。
- **GRPO vs SFT：决定性因素**：有人提出如何区分一个模型是通过 **SFT** 训练的还是使用 **GRPO** 训练的，强调了对评估指标的需求。
   - 有建议认为泛化能力和思维链（**CoT**）的长度可以作为理解训练方法的潜在代理指标。
- **关于训练指标的讨论**：一位成员补充说，熵（entropy）等训练指标应该区分是否在 **<thinking>** 标签内，这意味着数据评估中存在更深层次的细微差别。
   - 这反映了人们对各种训练范式如何影响模型性能指标的兴趣日益浓厚。
- **关于文档格式的想法**：大家达成共识，认为 **Markdown 文件**应被视为分享发现的最低要求，以促进更容易的访问。
   - 讨论强调了对现有平台的挫败感，以及对更直接的文档记录方法的渴望。



**提及的链接**：<a href="https://x.com/TrelisResearch/status/1890445776137969710">来自 Trelis Research (@TrelisResearch) 的推文</a>：+ GRPO 是穷人的工具，SFT 是 GPU 富豪的工具 +-------------------------------*下周将发布具体的 GRPO vs SFT 视频，但我先在这里发布初步结果*。我使用以下方法在 GSM8K 上训练了 Llama 3.2 1B：1...

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1340633173792657408)** (13 条消息🔥): 

> `TUM AI 系列讲座：Robin Rombach、Elon Musk 的战时心态、LLaDA：一种新的扩散模型、ChatGPT 的训练后经验、Nathan Lambert 的认可` 


- **Robin Rombach 关于 FLUX 的 TUM AI 讲座**：TUM AI 系列讲座邀请了 Robin Rombach 在 [2 月 17 日下午 6 点 (GMT+1)](https://youtube.com/live/nrKKLJXBSw0) 讨论 **'FLUX: Flow Matching for Content Creation at Scale'**。直播将在 [YouTube](https://youtube.com/live/nrKKLJXBSw0) 上进行。
   - 讲师将涵盖 Flow Matching 的基础方面及其在大规模文本到图像预训练中的应用。
- **理解 Elon Musk 的工作伦理**：关于 Elon Musk 的一种观点认为他以 **'战时模式' (wartime mode)** 运作，强调快速行动和高产出，而不计附带损失。这种模式适用于从 SpaceX 到 DOGE 的各种项目，使他的思维方式更易于理解。
   - 该讨论似乎在 Twitter 链接的一个线程中引起了共鸣，邀请进一步探索相关阅读。
- **LLaDA 简介：一种新的语言扩散模型**：论文介绍了 **LLaDA**，这是一种扩散模型，通过展示其在 **可扩展性和指令遵循 (instruction-following)** 方面的能力，挑战了传统的自回归模型 (ARMs)。它被提议作为 **LLaMA3** 等模型的强力竞争者，这是通过在海量数据集上进行广泛训练实现的。
   - 进一步的对话表明了对 LLaDA 扩散分类的怀疑，因为传统的扩散特征似乎并不存在。
- **关于 ChatGPT 训练后 (Post-training) 的见解**：团队成员最近在斯坦福大学的一次演讲中强调了 ChatGPT 训练后方法的经验，遗憾的是没有录音。演讲的幻灯片可以在 [这里](https://docs.google.com/presentation/d/11KWCKUORnPpVMSY6vXgBeFSWo7fJcuGQ9yuR6vC1pzE/edit?usp=sharing) 找到。
   - 成员们表示如果有录音的话，有兴趣观看演讲录像。
- **Nathan Lambert 的情感表达**：Nathan Lambert 在与疾病抗争的过程中表达了一种获得认可的感觉，并幽默地提到了他最近在 Twitter 上的胜利。他评论说在应对当前状况时正处于“痛苦深渊” (pain cave) 中。
   - 他表达的情绪在对话中引发了轻松的回应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.09992">Large Language Diffusion Models</a>: 自回归模型 (ARMs) 被广泛视为大语言模型 (LLMs) 的基石。我们通过引入 LLaDA 来挑战这一观念，LLaDA 是一种从头开始训练的扩散模型...</li><li><a href="https://x.com/johnschulman2/status/1891539960743743756">John Schulman (@johnschulman2) 的推文</a>: @barret_zoph 和我最近在斯坦福大学做了一个关于训练后 (post-training) 以及我们在 ChatGPT 上合作经验的演讲。不幸的是，演讲没有录音，但这里有幻灯片：https://docs.g...</li><li><a href="https://fxtwitter.com/MattNiessner/status/1891071070909734968">Matthias Niessner (@MattNiessner) 的推文</a>: 明天在我们的 TUM AI 系列讲座中，将由 @bfl_ml 的 CEO Robin Rombach (@robrombach) 担任主讲。他将谈论“𝐅𝐋𝐔𝐗: 大规模内容生成的 Flow Matching”。直播地址：...</li><li><a href="https://x.com/jasoncrawford/status/1890834517285359780?s=46">Jason Crawford (@jasoncrawford) 的推文</a>: 关于 Elon 的一个模型是他永远处于战时模式。尽可能快地行动，带来压倒性的力量，榨取超人的努力，接受附带损失。这适用于一系列事情...</li><li><a href="https://www.youtube.com/live/nrKKLJXBSw0">TUM AI 系列讲座 - FLUX: 大规模内容生成的 Flow Matching (Robin Rombach)</a>: 摘要：我将谈论 Flow Matching 的基础，为大规模文本到图像预训练进行扩展，偏好微调 (preference-tuning) 方法和技术...</li><li><a href="https://x.com/gallabytes/status/1891356261582557438">theseriousadult (@gallabytes) 的推文</a>: 在什么意义上这是扩散？我没看到 SDE，没有概率流，没有噪声。并非每种迭代采样方法都是扩散！这篇论文确实令人印象深刻，但它是一个我不理解的新事物...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1340105859639808142)** (1 条消息): 

> `National Energy Dominance Council, Energy Production Expansion, Natural Resource Management, Economic Growth Strategies, AI Leadership and Manufacturing` 


- **国家能源主导委员会 (National Energy Dominance Council) 正式成立**：总统成立了 **National Energy Dominance Council**，旨在通过利用国家丰富的自然资源来增强美国的能源领导地位。
   - 该委员会旨在促进**可靠且负担得起的能源生产**，从而推动经济增长并保障国家安全。
- **能源生产扩张计划**：总统强调需要扩大各种形式的**能源生产**，以应对**通货膨胀**等经济挑战。
   - 这一扩张旨在创造**高薪就业岗位**，并重新确立美国在各个领域的领导地位。
- **利用美国的自然资产**：该政策认识到利用美国多样化**自然资源**的重要性，包括原油、天然气和关键矿产。
   - 这一战略重点旨在增强能源独立性，同时为国家的整体经济实力做出贡献。
- **对 AI 和制造业领导地位的承诺**：总统重申了在 **AI** 和**制造业**领域保持领先地位的承诺，这些是经济稳定的关键驱动力。
   - 这包括利用能源资源来支持国内的先进制造和技术创新。
- **通过能源资源开展外交**：该战略概述了利用能源领域的**商业和外交杠杆**来实现全球和平。
   - 通过利用能源资源，该计划旨在为解决冲突和加强国际关系做出贡献。



**提及的链接**：<a href="https://www.whitehouse.gov/presidential-actions/2025/02/establishing-the-national-energy-dominance-council/">建立国家能源主导委员会 (Establishing the National Energy Dominance Council)</a>：根据宪法和美利坚合众国法律赋予我作为总统的权力，特此下令：第 1 节。

  

---

### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1340647510737162260)** (6 messages): 

> `Deep Research Workflow, O1 Pro Prompting, Scraping Techniques` 


- **Deep Research 工作流简化了工作**：一位成员分享说，使用结构化的 Deep Research 工作流显著降低了他们的整体工作量，并使其在处理初始想法和 Prompting 时更加得心应手。
   - *‘这实际上降低了我的整体工作量’* 反映了他们对这种新系统方法的积极态度。
- **将 Naive Prompts 转化为详尽的输出**：TheXeophon 展示了一个经过改进的 Prompting 系统如何将一个基础的 Scraping 咨询从两种方法扩展为包含第三方工具和防封禁技巧的详细方案。
   - 他们强调，与初级的尝试相比，这会产生 *‘更深入、更好’* 的输出。
- **使用 O1 Pro 增强 Prompts**：成员们正在利用 O1 Pro 构建详细的 Prompts，以便以精简的方式生成详尽的报告，产出一系列有效的输出。
   - 研究结果表明，当利用针对特定目标量身定制的结构化 Prompts 时，质量会有显著提升。
- **意识到项目组织的价值**：一位成员分享了他们使用 buccocapital prompting 系统组织项目的经验，发现它在从极少输入中生成可操作的想法方面非常有效。
   - 这突出了项目结构和方法论指南在增强创意输出方面的益处。
- **优化旧的 Prompts 以获得更好的结果**：TheXeophon 计划利用从新工作流中学到的改进技术重新运行旧的 Prompts，这表明他们正在采取主动的方法来增强 Prompt Engineering。
   - 这反映了社区内日益认识到需要不断完善其研究方法论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1891480336690774088">Xeophon (@TheXeophon) 的推文</a>：任务：寻找 Scrape &lt;site&gt; 的方法。Naive prompt：两种方法（requests, playwright）可行。Big prompt：三种方法（requests, playwright, 第三方工具）+ 如何规避常见检测...</li><li><a href="https://x.com/TheXeophon/status/1891086352353018295">Xeophon (@TheXeophon) 的推文</a>：这是一个游戏规则改变者，我的 Naive ODR prompt（上图）使用的来源更少，深度更浅，效果不如引用推文中的工作流（结果在下图）。将重新运行我的一些旧 Prompts... 引用 B...</li><li><a href="https://x.com/buccocapital/status/1890745551995424987">BuccoCapital Bloke (@buccocapital) 的推文</a>：大家很好奇，所以这就是我使用 Deep Research 的方式。我将演示 Prompting 过程，然后举一个例子：1. 首先，我使用 O1 Pro 为我构建了一个 Deep Research 的 Prompt，用于进行 Deep Re...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1340110319149781155)** (20 条消息🔥): 

> `使用 Data URIs 生成 AI 图像、数学、国际象棋与神童、Markdown 与 HTML 转换问题、音乐生成与认知任务、lm_eval 代码生成任务` 


- **AI 探索使用 Data URIs 生成图像**：一位成员询问了关于使用 **Data URIs** 或 **Base64** 编码训练 AI 进行图像生成的问题，这种方法允许在不使用外部链接的情况下存储图像。
   - 他们注意到这种方法在用户界面中通常显示为 **blob**。
- **数学和国际象棋领域的神童引发关注**：一段对话探讨了在数学和国际象棋等领域**神童**的普遍性，暗示这些领域可能是 AI 取得进展的沃土。
   - 另一位成员推测**音乐**可能也符合这一模式，并质疑为什么它没有被归类为“简单”领域。
- **排除 Markdown 与 HTML 转换故障**：有成员对使用 **HTML to Markdown** 转换器时保留了像 `&gt;` 这样的转义字符表示担忧，这让一位用户感到惊讶。
   - 另一位成员回应称，随着 XML 格式的日益普及，保持 Markdown 和 ASCII 编码的一致性非常重要。
- **音乐生成中的认知努力**：成员们讨论了音乐创作的**认知层面**，指出生成声音涉及由大脑管理的物理输出。
   - 对话强调，目前的音乐生成器在**频域（frequency domain）**运行，这影响了它们像在国际象棋等游戏中那样进行优化的能力。
- **lm_eval 缺少代码生成任务**：一位用户报告了 **lm_eval v0.4.7** 的问题，发现尽管按照提供的设置说明进行了安装，但安装后并未列出代码生成任务。
   - 他们分享了安装方法，寻求关于导致这种差异的潜在原因的见解。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1340008788186628096)** (235 条消息🔥🔥): 

> `采样技术研究、Transformer 在时间序列中的应用、创意写作背景下的重复使用、NovelAI 采样器理解、DEQs 与 RNNs 分析` 


- **NovelAI 独特的采样器知识**：NovelAI 对采样方法的理解不断演进，源于他们决定提供多种采样器算法，这促使开发人员去弄清楚其中涉及的复杂细节。
   - 这与其他组织形成了对比，那些组织最初不提供复杂的采样器，因此缺乏投入大量精力去理解它们的动力。
- **文本生成中重复的挑战**：文本生成中的重复惩罚（repetition penalty）是一种统计方法，用于缓解水平较低的作者常遇到的问题，这些作者往往在写作中过度使用重复。
   - 虽然优秀的作者可以成功利用重复来达到戏剧效果，但在开发采样策略时，区分“好的重复”和“坏的重复”仍然是一个挑战。
- **使用 Logprobs 评估采样器质量的理论**：所描述的方法涉及利用 logprobs 的三次多项式来近似 Token 质量，并进行某些转换以将质量转回 logprobs。
   - 该方法调整 logprobs，使得中间值增加，同时惩罚高值和低值，尽管由于熵（entropy）的考虑存在一些失真。
- **DEQs 与 RNNs 的比较**：围绕 DEQs 的讨论强调了它们与循环结构的联系，重点关注权重共享（weight tying）的影响以及隐藏层收敛到固定点的潜力。
   - 参与者指出，与处理时间变化的 RNNs 不同，DEQs 强调深度，并可以使用隐式微分方法进行反向传播（backpropagation）。
- **探索 Transformer 对时间序列的适用性**：关于 Transformer 在时间序列背景下的有效性存在持续争论，一些人批评它们无法高效处理特定的结构化问题（如奇偶校验）。
   - 研究表明，虽然 Transformer 可以应用于时间序列数据，但传统模型（如三次样条插值 cubic splines）在捕捉潜在模式方面往往表现得更好。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://aclanthology.org/2022.findings-emnlp.293/">On the Role of Bidirectionality in Language Model Pre-Training</a>: Mikel Artetxe, Jingfei Du, Naman Goyal, Luke Zettlemoyer, Veselin Stoyanov. Findings of the Association for Computational Linguistics: EMNLP 2022. 2022.</li><li><a href="https://arxiv.org/abs/2502.07864">TransMLA: Multi-Head Latent Attention Is All You Need</a>: 现代大语言模型（LLMs）在当前硬件上经常遇到通信瓶颈，而不仅仅是纯粹的计算限制。Multi-head Latent Attention (MLA) 解决了这一挑战...</li><li><a href="https://arxiv.org/abs/2502.06785">DeepCrossAttention: Supercharging Transformer Residual Connections</a>: Transformer 网络在各个领域取得了显著成功，利用了包括残差连接在内的多种架构创新。然而，传统的残差连接...</li><li><a href="https://arxiv.org/abs/1806.02296">Regularization by Denoising: Clarifications and New Interpretations</a>: 由 Romano、Elad 和 Milanfar 最近提出的通过去噪进行正则化（RED），是一个强大的图像恢复框架，旨在最小化由...构建的显式正则化目标。</li><li><a href="https://arxiv.org/abs/2306.16805">CLIPAG: Towards Generator-Free Text-to-Image Generation</a>: 感知对齐梯度（PAG）是指在鲁棒图像分类模型中观察到的一种有趣特性，其中它们的输入梯度与人类感知对齐并具有语义含义...</li><li><a href="https://arxiv.org/abs/2205.13504">Are Transformers Effective for Time Series Forecasting?</a>: 最近，针对长期时间序列预测（LTSF）任务的基于 Transformer 的解决方案激增。尽管过去几年性能不断提升，我们仍质疑其有效性...</li><li><a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: 我们提出了视觉自回归建模（VAR），这是一种新一代范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率预测”...</li><li><a href="https://openreview.net/forum?id=IHRQif8VQC">Ensemble everything everywhere: Multi-scale aggregation for...</a>: 对抗样本对深度神经网络的鲁棒性、可靠性和对齐构成了重大挑战。我们提出了一种新颖、易于使用的方法来实现高质量...</li><li><a href="https://arxiv.org/abs/2410.13821">Artificial Kuramoto Oscillatory Neurons</a>: 神经科学和 AI 领域早就知道，神经元之间的“绑定”会导致一种竞争性学习形式，其中表示被压缩以便表示更抽象的内容...</li><li><a href="https://arxiv.org/abs/2406.07496">TextGrad: Automatic &#34;Differentiation&#34; via Text</a>: AI 正在经历范式转变，突破性进展由编排多个大语言模型（LLMs）和其他复杂组件的系统实现。因此，开发原则性且自动化的...</li><li><a href="https://openreview.net/forum?id=2bIQBDSfRk">DenseAttention: No-Compromise Exact All $N \times N$  Interactions...</a>: 无处不在的 Transformer 架构面临两个主要瓶颈：1) 计算和内存效率低，导致硬件利用率不佳；2) 二次时间复杂度...</li><li><a href="https://arxiv.org/abs/2202.10103">Robustness and Accuracy Could Be Reconcilable by (Proper) Definition</a>: 鲁棒性与准确性之间的权衡在对抗性文献中得到了广泛研究。尽管仍有争议，但主流观点认为这种权衡是固有的，无论是经验上...</li><li><a href="https://arxiv.org/abs/2502.09992">Large Language Diffusion Models</a>: 自回归模型（ARMs）被广泛视为大语言模型（LLMs）的基石。我们通过引入 LLaDA 来挑战这一观念，这是一种在预训练阶段从头开始训练的扩散模型...</li><li><a href="https://arxiv.org/abs/2408.17046">Text-to-Image Generation Via Energy-Based CLIP</a>: 联合能量模型（JEMs）虽然引起了显著的研究关注，但尚未成功扩展到现实世界的高分辨率数据集。我们提出了 EB-CLIP，这是一种扩展 JEMs 的新颖方法...</li><li><a href="https://x.com/neverrixx/status/1457992057159569408">Tweet from nev (@neverrixx)</a>: 像素艺术（256 分辨率，采用最近邻插值）</li><li><a href="https://arxiv.org/abs/2410.07041">Emergent properties with repeated examples</a>: 我们研究了 Transformer 的性能与算法生成数据集中训练样本重复次数的关系。在三个数学问题上：最大公约数...</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing L...</a>: Hyperfitting 现象：锐化与稳定 L...

LMs for Open-Ended Text Generation</a>: 本文介绍了在极小数据集上过拟合预训练大语言模型 (LLMs) 所产生的反直觉泛化结果。在开放式文本生成的设置下，它是...</li><li><a href="https://arxiv.org/abs/2502.05252">GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?</a>: 长上下文大语言模型 (LLMs) 最近在信息检索和长文档问答中表现出强劲性能。然而，为了解决最具挑战性的智力问题，LLMs 必须...</li><li><a href="https://x.com/InfiniAILab/status/1890469309253841191?t=062o_qhbW0XRcux_v_GZ8w&s=19">来自 Infini-AI-Lab (@InfiniAILab) 的推文</a>: 🐭🐷 GSM-Infinite 是由问题生成器生成的。在零噪声基准测试中评估了 18 个强大的 LLMs，在长上下文基准测试中评估了 10 个 LLMs。🚀关键结论：🥕 最近的推理模型...</li><li><a href="https://x.com/cloneofsimo/status/1869807463186472970?s=46">来自 Simo Ryu (@cloneofsimo) 的推文</a>: 好吧，Lucas 提出了一个观点，也许它起作用是因为它是 12 层网络。所以我把它改成了 96 层网络（带有 768 隐藏层维度，哈哈）并进行了长达 10 小时的参数扫描。令我惊讶的是，差距反而扩大了...</li><li><a href="https://x.com/BlancheMinerva/status/1890854248277058033">来自 Stella Biderman (@BlancheMinerva) 的推文</a>: @stanislavfort @johnowhitaker @advadnoun @RiversHaveWings 顶一下以增加曝光度 / 整合了多个评论中的链接... 关于“优化 CLIP 以在不使用...的情况下进行图像编辑或生成”...</li><li><a href="https://github.com/DanJbk/Mse-regularized-VQGAN-CLIP">GitHub - DanJbk/Mse-regularized-VQGAN-CLIP: 使用额外正则化生成图像的 VQGAN-CLIP 方法，以获得更连贯和准确的输出</a>: 使用额外正则化生成图像的 VQGAN-CLIP 方法，以获得更连贯和准确的输出 - DanJbk/Mse-regularized-VQGAN-CLIP</li><li><a href="https://x.com/stanislavfort/status/1890724291752100265">来自 Stanislav Fort (@stanislavfort) 的推文</a>: 我们发现了一种令人惊讶的、无需训练的图像生成方式：没有 GANs 或扩散模型，而是一个 ✨神秘的第三种选择✨！像 CLIP 这样的标准模型已经可以直接创建图像，且无需训练...</li><li><a href="https://github.com/eps696/aphantasia#operations>.">GitHub - eps696/aphantasia: CLIP + FFT/DWT/RGB = 文本转图像/视频</a>: CLIP + FFT/DWT/RGB = 文本转图像/视频。通过在 GitHub 上创建账号来为 eps696/aphantasia 的开发做出贡献。</li><li><a href="https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb">kylemcdonald/deepdream 的 master 分支中的 deepdream/dream.ipynb</a>: 通过在 GitHub 上创建账号来为 kylemcdonald/deepdream 的开发做出贡献。</li><li><a href="https://github.com/tensorflow/lucid/blob/6dcc927e4ff4e7ef4d9c54d27b0352849dadd1bb/lucid/optvis/param/spatial.py#L61">tensorflow/lucid 的 6dcc927e4ff4e7ef4d9c54d27b0352849dadd1bb 分支中的 lucid/lucid/optvis/param/spatial.py</a>: 用于神经网络可解释性研究的基础设施和工具集。 - tensorflow/lucid</li><li><a href="https://distill.pub/2017/feature-visualization/">Feature Visualization</a>: 神经网络如何建立对图像的理解
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1340028857465704508)** (3 条消息): 

> `下游文本聚类, Transcoders` 


- **询问下游文本聚类方法**: 一位用户询问了使用 **transcoders** 进行下游文本聚类的**推荐方法**。
   - 另一位成员指出，他们认为目前还没有成熟的方法，并询问了该用户意图的更多细节。
- **Transcoder 聚类缺乏成熟方法**: 一位成员表示不确定，称他们认为目前还没有**成熟的**基于 **transcoders** 的**下游文本聚类**方法。
   - 这说明在文本处理的这一领域还有进一步讨论和发展的空间。


  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1340882029624950846)** (1 条消息): 

> `Generative AI models, Machine learning techniques, NLP applications, Model optimization strategies, AI integration in production` 


- **Generative AI 模型专业知识**：一位 AI 开发者强调了他们在构建和优化 **Generative AI models** 方面的专长，特别是在微调 **LLMs**（如 **GPT, BERT, and Llama**）方面。
   - 他们强调了使用 **PyTorch, TensorFlow, and LangChain** 等工具的实战经验，确保了稳健的模型性能。
- **扎实的 Machine Learning 和 NLP 基础**：该开发者详细介绍了在 **machine learning, deep learning** 和 **NLP** 方面的坚实基础，能够为各种应用开发有效的解决方案。
   - 他们的背景使其能够成功地将理论进展与实际应用相结合。
- **对文本和图像合成的贡献**：消息提到了在专注于 **text generation** 和 **image synthesis** 技术（包括 **Stable Diffusion** 和 **DALL·E**）项目中的贡献。
   - 这突显了其多样的技能组合，已从传统的 **NLP** 扩展到生成式媒体领域。
- **AI-Driven Automation 经验**：分享了在 **AI-driven automation** 方面的经验，展示了通过智能解决方案提高运营效率的能力。
   - 这种专业知识使他们成为追求 AI 创新的团队中的宝贵资产。
- **致力于 AI Research 和部署**：该开发者表达了对 **AI research, prompt engineering** 以及 **scalable deployment** 的热情，展现了突破生成式技术边界的愿望。
   - 他们的目标与为专注于下一波 **Generative AI** 发展的团队做出贡献相一致。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1340015468517396480)** (174 条消息🔥🔥): 

> `DeepHermes Model Performance, Open Source Models and Datasets, Reinforcement Learning in LLMs, Compatibility of AI Models, Reasoning Models Development` 


- **DeepHermes 模型表现优于其他模型**：用户注意到 **DeepHermes** 模型在指令遵循方面比目前的 R1 蒸馏模型更好，标志着其成为首个可用的通用推理模型。
   - 讨论强调了推理任务和数据集的进一步改进潜力，这可能会增强这一性能。
- **呼吁开源旧模型**：尽管 **Claude 3** 已经发布一段时间，但 **Anthropic** 决定删除它且没有开源计划，这引发了担忧。
   - 参与者认为，发布旧模型可以培养好感，并有利于公司在开源社区的声誉。
- **关于 RL 和 LLMs 的讨论**：分享了关于 **Reinforcement Learning (RL)** 和奖励如何有效增强 LLMs 的见解，特别是在对齐工具方面。
   - 历史资料表明，早在其他公司采用类似策略之前，研究人员就已经意识到这些技术。
- **AI 模型的兼容性问题**：一位用户对各种模型的能力表示沮丧，强调大多数模型在某些任务中表现吃力，但 **8B models** 在数据清洗方面表现更好。
   - 社区讨论了正在进行的低参数模型开发，旨在不牺牲兼容性的情况下提高可访问性。
- **对高效推理数据集的需求**：分享了一个名为 **Rombo-Org/Optimized_Reasoning** 的新数据集，旨在提高推理性能并减少 token 使用。
   - 用户渴望看到该数据集与现有推理数据集相比的表现，以及它对未来模型训练的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/hance-ai/descript-audio-codec-44khz">hance-ai/descript-audio-codec-44khz · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: 基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会将 FLOPs（或计算）动态分配给特定的...</li><li><a href="https://huggingface.co/internlm/internlm3-8b-instruct-gguf">internlm/internlm3-8b-instruct-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://venturebeat.com/ai/personalized-unrestricted-ai-lab-nous-research-launches-first-toggle-on-reasoning-model-deephermes-3/">&#8216;Personalized, unrestricted&#8217; AI lab Nous Research launches first toggle-on reasoning model: DeepHermes-3</a>: 一位 DeepHermes-3 用户报告称，在 MacBook Pro M4 Max 消费级硬件上的处理速度为每秒 28.98 个 tokens。</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1irdwy0/new_dataset_release_romboorgoptimized_reasoning/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/teknium/DeepHermes-3-Llama-3-3B-Preview-GRPO-1">teknium/DeepHermes-3-Llama-3-3B-Preview-GRPO-1 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/rombodawg/Easy_training/tree/main/Galore%2BQlora_With_Multi_GPU_Support">Easy_training/Galore+Qlora_With_Multi_GPU_Support at main · rombodawg/Easy_training</a>: 通过在 GitHub 上创建一个账户来为 rombodawg/Easy_training 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=ry_SthLfTa8">Open-source AI will have a massive impact on the world, says Hugging Face CEO</a>: Hugging Face 的 CEO Clement Delangue 讨论了 OpenAI、DeepSeek 以及他公司的创新。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/ntpld9vkD6">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=v0gjI__RyCY">Jeff Dean &amp; Noam Shazeer – 25 years at Google: from PageRank to AGI</a>: 本周我迎来了任何领域中最重要的两位技术专家。Jeff Dean 是 Google 的首席科学家，在公司的 25 年里，他曾工作于...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1340023328181583892)** (23 条消息🔥): 

> `Cost of Training Models, Test Time Scaling, Custom Model Training, Multi-Headed Self-Attention, Hermes 3 Performance` 


- **理解模型训练成本**：讨论了训练一个 **1B 模型**所需的成本，完整配置的估算范围在 **数千到数万美元** 之间。
   - 一位成员暗示，虽然在消费级 GPU 上进行训练是可行的，但随着时间的推移，数据和架构的进步可能会影响整体的成本效率。
- **Test Time Scaling 的挑战**：有人指出，增加模型规模并不呈线性扩展；例如，根据 Scaling Laws，从 **70M 扩展到 100M** 需要显著更多的数据。
   - 另一位参与者提到，通过精细的数据选择和训练策略，可能仅需 **200 到 1,000 美元** 即可实现一个 **1.5B 模型**。
- **自定义模型与架构**：参与者对构建自定义架构及其相关挑战表现出兴趣，特别是围绕转向 **多阶段训练过程 (multi-stage training process)** 的挑战。
   - 建议探索现有资源来配置适配和模型，一些人指向 **LoRA** 作为增强模型性能的潜在方法。
- **Multi-Headed Self-Attention 详解**：一位 LLM 初学者询问了 Multi-Headed Self-Attention 在权重矩阵看似相似的情况下是如何共享并运作的。
   - 他们表示，这种技术背后的底层数学原理和维度变换 (dimensional gymnastics) 仍然难以直观理解。
- **Hermes 3 审查声明**：针对 **Hermes 3** 宣传的 **无审查 (censorship-free)** 特性提出了质疑，并询问了在实际提问过程中出现的不一致情况。
   - 一位用户对该模型拒绝回答某些提示词感到惊讶，尽管它声称无审查，这引发了对其过滤机制的猜测。



**提到的链接**: <a href="https://arxiv.org/html/2408.03506v1#:~:text=Using%20our%20pre,16K%20context%20window%20version%2C%20which">1.5-Pints Technical Report: Pretraining in Days, Not Months – Your Language Model Thrives on Quality Data</a>：未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1340154801932996659)** (2 条消息): 

> `LLaDA diffusion model, Large Language Models (LLMs), Probabilistic inference in medical imaging` 


- **LLaDA 挑战传统的 LLM 自回归模型 (ARMs)**：提到的 [LLaDA 论文](https://arxiv.org/abs/2502.09992) 介绍了一种扩散模型，通过为概率推理提供原则性的生成方法，重新定义了大语言模型 (LLMs) 的格局。
   - LLaDA 出色的性能，特别是在 In-context Learning 和多轮对话中，使其成为 **GPT-4o** 等成熟模型的有力竞争替代方案。
- **LLaDA 模型的扩展性**：LLaDA 模型展现了 **强大的扩展性 (scalability)**，在广泛的基准测试中超越了传统的自回归模型 (ARMs)，在各种应用中展示了充满前景的结果。
   - 它有效地采用了前向数据掩码过程和逆向过程，增强了其生成能力。



**提到的链接**: <a href="https://arxiv.org/abs/2502.09992">Large Language Diffusion Models</a>：自回归模型 (ARMs) 被广泛认为是大型语言模型 (LLMs) 的基石。我们通过引入 LLaDA 来挑战这一观念，这是一种从头开始训练的扩散模型...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

teknium: https://x.com/_akhaliq/status/1890546832784208080?s=46
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1340154801932996659)** (2 条消息): 

> `Medical Imaging, Computer Vision Applications, LLaDA Model, Large Language Models` 


- **探索 LLaDA 在 Medical Imaging 中的见解**：一位成员推荐阅读题为 [LLaDA: Autoregressive Models and Diffusion](https://arxiv.org/abs/2502.09992) 的论文，该论文讨论了在 Pre-training 和 Supervised Fine-tuning 范式下训练的用于生成任务的 Diffusion Model。
   - *LLaDA* 展示了强大的 Scalability，在优化概率推理（Probabilistic Inference）的同时，能与领先的 LLM 竞争，并解决了与 GPT-4 等模型相比的 Reversal Curse 问题。
- **Computer Vision 在 Medical Imaging 中的应用**：一位用户正在寻求关于利用 Large Language Models (LLM) 的 **Medical Imaging** 和 **Computer Vision** 应用交叉领域的论文推荐。
   - 该请求强调了人们对利用先进模型进行医疗诊断和成像技术创新的日益增长的兴趣。



**提到的链接**：<a href="https://arxiv.org/abs/2502.09992">Large Language Diffusion Models</a>：Autoregressive Models (ARM) 被广泛认为是 Large Language Models (LLM) 的基石。我们通过引入 LLaDA 来挑战这一观念，这是一种在 Pre-training 阶段从头开始训练的 Diffusion Model...

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1340029136198176910)** (170 条消息🔥🔥): 

> `NURBS and Splines in AI, Online Learning Approaches, Backpropagation Alternatives, Memory Structures in Learning Algorithms, UltraMem Architecture` 


- **利用 NURBS 和 Splines 变革 AI**：讨论集中在将 **NURBS** 和 **Splines** 作为 AI 模型的基础元素，以增强函数逼近能力，并减少与传统多项式相比的 Overfitting 问题。
   - 参与者辩论了这些技术在开发新 AI 方法论方面的潜力，强调了平滑度（Smoothness）和拓扑结构（Topology）在实现更好泛化（Generalization）方面的重要性。
- **Online Learning vs. 传统采样**：强调了 Online Learning 的潜力，重点是按顺序使用单个样本训练模型，以更好地模拟类似人类的记忆和召回机制。
   - 这与传统的 Batch Learning 方法形成对比，后者通常需要大量的内存和计算，从而限制了其效率。
- **探索 Backpropagation 的替代方案**：大家达成共识，认为 Backpropagation 可能并不总是某些 AI 架构的最佳优化方法，从而引发了开发基于 Spline 和 NURBS 概念的模型的想法。
   - 参与者建议需要重新思考基础方法，以可能产生更有效的学习算法，这些算法可以在不广泛依赖 Gradient Descent 的情况下运行。
- **学习中的 Memory Structures 与效率**：强调了 AI 中 Memory Structures 的重要性，并讨论了与传统稠密网络相比，新架构如何更有效地管理内存。
   - 具体而言，考虑了基于 NURBS 的架构以高效且较少依赖大型 Context Windows 的方式利用内存的潜力。
- **UltraMem 架构介绍**：UltraMem 架构被提及为最近的一项创新，它利用大规模、超稀疏的内存层来提高推理速度和模型性能。
   - 有人指出，该架构在保持计算效率的同时，可能比传统的 MoE 方法更具优势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/elonmusk/status/1890958798841389499">来自 Elon Musk (@elonmusk) 的推文</a>：Grok 3 将于周一晚上 PT 时间 8 点发布并进行现场演示。地球上最聪明的 AI。</li><li><a href="https://sagipolaczek.github.io/NeuralSVG/">NeuralSVG: An Implicit Representation for Text-to-Vector Generation</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2401.17992">Multilinear Operator Networks</a>：尽管深度神经网络在图像识别方面具有卓越的能力，但对激活函数的依赖仍然是一个很大程度上未被探索的领域，且尚未被消除。另一方面...</li><li><a href="https://arxiv.org/abs/2411.12364">Ultra-Sparse Memory Network</a>：众所周知，Transformer 模型的性能与其参数数量和计算复杂度呈对数关系。虽然像 Mixture of Experts (Mo... 等方法</li><li><a href="https://x.com/BlinkDL_AI/status/1859578512988147889">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：“世界上最难的数独”由 12M 参数的 RWKV-6 在 4M tokens CoT 后解决 🙂 代码与模型：https://github.com/Jellyfish042/Sudoku-RWKV 注意该模型仅在 ctx8192 下训练，所以...</li><li><a href="https://www.youtube.com/watch?v=SmZmBKc7Lrs">机器学习中最重要的算法</a>：Shortform 链接：https://shortform.com/artem 在这段视频中，我们将讨论反向传播（backpropagation）—— 一种驱动整个机器学习领域的算法...</li><li><a href="https://github.com/betweentwomidnights/audiocraft-onnx">GitHub - betweentwomidnights/audiocraft-onnx: 我尝试将 musicgen 权重转换为 onnx 但失败了</a>：我尝试将 musicgen 权重转换为 onnx 但失败了 - betweentwomidnights/audiocraft-onnx</li><li><a href="https://www.youtube.com/watch?v=jvPPXbo87ds">样条曲线（Splines）的连续性</a>：为什么要用样条曲线？天哪，我有好消息告诉你，这就是为什么要用样条曲线！如果你喜欢我的工作，请考虑支持我 💖 https://www.patreon.com/acegik...</li><li><a href="https://github.com/222464/TeensyAtariPlayingAgent">GitHub - 222464/TeensyAtariPlayingAgent: 一个运行在 Teensy 微控制器上用于玩 Atari 游戏的 Agent</a>：一个运行在 Teensy 微控制器上用于玩 Atari 游戏的 Agent - 222464/TeensyAtariPlayingAgent</li><li><a href="https://www.youtube.com/watch?v=HwDh-M-6kfQ">Thomas Hughes：“等几何分析（Isogeometric Analysis）”</a>：2019 年普渡大学工程学院杰出讲座系列演讲者 Thomas J.R. Hughes，Peter O'Donnell Jr. 计算与应用数学讲席教授...</li><li><a href="https://www.youtube.com/watch?v=97kQRYwL3P0">OpenAI：AI 时代已来！</a>：❤️ 在此处查看 Lambda 并注册其 GPU Cloud：https://lambdalabs.com/papers 📝 论文《使用大型推理模型进行竞赛编程》是...</li><li><a href="https://youtu.be/JUWIM34h5mk?si=lLxZ-Xtj_qfHuljV">AGI 架构</a>：这是一个通用的 AGI 架构综合布局，其中包括对意识难题的解决方案以及...</li><li><a href="https://www.youtube.com/watch?v=Ecqff-9Upjw">大脑布线的惊人方式</a>：在 shortform.com/artem 获取我最喜欢的书籍摘要服务 20% 的折扣。社交媒体：X/Twitter: https://x.com/ArtemKRSV Patreon: https://patreon.com/artemki...</li><li><a href="https://tenor.com/view/elmo-hidding-embarrassed-so-embarrassed-im-embarrassed-gif-2879261371850596063">Elmo 躲藏 GIF - Elmo 躲藏 尴尬 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1340233356553617429)** (12 条消息🔥): 

> `每日论文阅读小组后勤, 往期论文汇编, 开创性论文层级树, 主持论文讨论, 社区资源共享` 


- **了解每日论文阅读小组后勤**：一位新成员表示有兴趣参加每日论文阅读小组，寻求有关涵盖的主题、决策过程和往期讨论的信息。
   - 成员们引导其查看置顶消息以获取相关详情，并认可了为往期论文汇编创建 Google Sheet 的想法。
- **关于层级论文树的提案**：一位成员建议开发一个与关键实验室相关的**开创性论文和新论文**的层级树，并加入依赖链接以促进理解。
   - *信息过滤是关键*，成员们表达了拥有专注于高效推进知识的社区驱动资源的重要性。
- **鼓励主持论文讨论**：成员们对增加参与主持小组内的论文讨论表现出热情。
   - 一位成员指出，更多的主持人可以带来宝贵的视角，并鼓励更广泛地参与正在进行的对话。

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1340302490499026974)** (9 条消息🔥): 

> `OpenAI 的机器人 50 条定律，Mistral Saba 发布，马云回归，纽约 AI 工程师峰会，Anthropic 的新 AI 模型` 


- **OpenAI 发布机器人 50 条定律**：OpenAI 推出了其 [机器人 50 条定律](https://sherwood.news/tech/step-aside-asimov-here-are-openais-50-laws-of-robotics/)，作为 AI 伦理的现代方法，超越了阿西莫夫最初的三大定律，旨在定义安全的 AI 行为。
   - 该文档概述了 AI 模型应如何表现以符合人类价值观和安全性，为正在进行的关于 AI 责任的讨论做出了贡献。
- **Mistral Saba 首次亮相**：MistralAI 宣布推出 [Mistral Saba](https://x.com/sophiamyang/status/1891487141718376580)，这是一个拥有 240 亿参数的区域性语言模型，在来自中东和南亚的数据集上进行了训练。
   - 该模型在**阿拉伯语**和南印度语言（如**泰米尔语**和**马拉雅拉姆语**）方面表现尤为出色，突显了其对区域语言多样性的关注。
- **马云受邀参加高层会议**：中国政府已邀请阿里巴巴联合创始人**马云**与高层领导人会面，这标志着在经历了充满挑战的几年后，政府对私营部门的重新支持。
   - 这一举动标志着一个显著的转变，因为国家正寻求稳定与像马云这样近期相对低调的主要企业家之间的关系。
- **纽约 AI 工程师峰会临近**：**AI Engineer Summit NYC** 定于 2 月 20 日至 21 日举行，参会和赞助申请即将截止。
   - 此次活动旨在连接 AI 工程师，并加强围绕尖端技术和该领域进展的讨论。
- **Anthropic 准备推出具备推理能力的 AI**：据 [The Decoder](https://the-decoder.com/anthropic-prepares-new-claude-hybrid-llms-with-reasoning-capability) 报道，Anthropic 正在按计划发布一款新的 AI 模型，该模型结合了传统的语言模型能力和先进的推理功能。
   - 这一举措旨在增强 AI 能力，暗示着向更智能、更具上下文感知能力的系统转变。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.bbc.com/news/business-45728459">Fan Bingbing: Missing Chinese actress fined for tax fraud</a>: 范冰冰：失踪的中国女演员因税务欺诈被罚款。</li><li><a href="https://www.latent.space/p/2025-papers">The 2025 AI Engineering Reading List</a>: 我们在 AI 工程的 10 个领域（LLM, Benchmarks, Prompting, RAG, Agents, CodeGen, Vision, Voice, Diffusion, Finetuning）中挑选了 50 篇论文/模型/博客。如果你是从零开始，请从这里开始。</li><li><a href="https://x.com/sophiamyang/status/1891487141718376580">Tweet from Sophia Yang, Ph.D. (@sophiamyang)</a>: 🏟️ 宣布推出 @MistralAI Saba，我们的第一个区域性语言模型。- Mistral Saba 是一个 24B 参数模型，在来自中东和南亚的精心策划的数据集上训练而成。- Mistral...</li><li><a href="https://finance.yahoo.com/news/china-invites-jack-ma-deepseek-065956463.html?guccounter=1">China Invites Jack Ma and DeepSeek Founder to Meet Top Leaders</a>: (Bloomberg) -- 据知情人士透露，中国已邀请包括阿里巴巴集团控股有限公司联合创始人马云在内的知名企业家与国家高层领导人会面，这可能是一个...</li><li><a href="https://x.com/elonmusk/status/1890958798841389499?s=46">Tweet from Elon Musk (@elonmusk)</a>: Grok 3 将于周一晚上太平洋时间 8 点发布并进行现场演示。地球上最聪明的 AI。</li><li><a href="https://the-decoder.com/anthropic-prepares-new-claude-hybrid-llms-with-reasoning-capability">Anthropic prepares new Claude hybrid LLMs with reasoning capability</a>: Anthropic 正准备发布一款新的 AI 模型，该模型结合了传统的语言模型能力和先进的推理功能。</li><li><a href="https://sherwood.news/tech/step-aside-asimov-here-are-openais-50-laws-of-robotics/">Step aside, Asimov. Here are OpenAI’s 50 Laws of Robotics</a>: OpenAI 正在让其 AI 放松限制：“没有话题是禁区。” 但它同时也使其变得反“觉醒（woke）”...</li><li><a href="https://model-spec.openai.com/2025-02-12.html">OpenAI Model Spec</a>: Model Spec 规定了 OpenAI 产品（包括我们的 API）底层模型所需具备的行为。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1340262434887303209)** (57 条消息🔥🔥): 

> `Cohere API 与 OpenAI 的兼容性，Deep research 克隆开发，审核模型讨论，Cohere 库安装问题，登录错误排查` 


- **Cohere API 可能会获得 OpenAI 兼容性**：一名成员询问 **Cohere API** 是否会变得与 **OpenAI** 兼容。
   - 反馈已分享给团队，但目前尚无可用信息。
- **构建 deep research 克隆**：一位成员分享说他们正在构建一个与各种 AI 模型兼容的 **deep research 克隆**，并强调了模型提供商实现方面的困难。
   - 他们指出，对其 LLM 提供商的唯一要求是在初始化时指定基础 URL 和模型。
- **关于审核模型的讨论**：讨论了审核模型，成员们表示需要比 **Llamaguard** 更好的模型，称其在他们的使用场景中缺乏准确性。
   - 一些成员目前正在使用 **OpenAI 的 omni moderation**，同时希望 Cohere 能推出新模型。
- **Cohere 库安装问题**：一名成员报告了 **cohere** 包的 **ModuleNotFoundError**，尽管已经安装了该包，这似乎是环境问题。
   - 其他人提供了排查建议，并提醒用户使用 `pip list` 检查其虚拟环境。
- **排查登录错误**：另一位用户报告了登录错误并分享了截图以寻求进一步帮助。
   - 由于当时错误上下文尚不明确，社区正期待提供帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/reference/about">使用 Cohere 的 API 和 SDK — Cohere</a>：Cohere 的 NLP 平台提供可定制的大语言模型和工具，供开发者构建 AI 应用程序。</li><li><a href="https://chat.4llm.com">4LLM - AI 聊天助手</a>：功能强大的 AI 聊天助手，用于无缝对话和任务自动化。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[rules](https://discord.com/channels/954421988141711382/954422415016996864/1340077995523575830)** (1 条消息): 

> `服务器规则，禁止广告政策，禁止招聘，语言要求，禁止隐藏链接` 


- **确立了明确的服务器行为准则**：服务器规则于 **2025年2月14日** 更新，强调了社区互动的明确指南，以保持对 Cohere 社区的关注。
   - 鼓励成员分享与 Cohere 相关的项目，但必须克制发布广告或招聘帖子。
- **此 Discord 禁止广告！**：执行严格的**禁止广告**政策，确保讨论集中在社区相关话题上。
   - 违规行为将导致相应的处理，包括对发布隐藏链接违规行为的临时封禁。
- **禁止招聘活动**：服务器禁止**招聘活动**，包括发布简历或任何职位发布。
   - 此规则旨在创建一个专注于社区分享而非求职的整洁环境。
- **仅限使用英语**：此服务器中的交流仅限于**英语**，以促进成员之间的可访问性和理解。
   - 此规则有助于保持社区内讨论和互动的清晰度。
- **拒绝隐藏链接**：服务器严格禁止**隐藏链接**，在分享内容中建立透明度以增强安全性。
   - 违反此规则可能导致**临时封禁**，确保讨论过程中的安全参与。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1340141994357293066)** (7 条消息): 

> `Cohere Aya-8b 集成，Rerank 超时问题` 


- **Cohere Aya-8b 集成支持**：一名成员正在将 **Cohere Aya-8b 模型** 集成到他们的应用程序中，但面临该特定模型版本的 **API 支持** 问题。
   - 另一名成员提供了一个使用 **c4ai-aya-expanse-8b** 模型的示例请求，并建议了相关的排查资源。
- **Rerank 的超时问题**：一名成员报告在调用 **Rerank** API 时遇到**超时问题**，寻求他人是否有类似经历。
   - 该消息已被移至特定频道进行进一步讨论，表明社区对该话题的关注。



**提到的链接**：<a href="https://dashboard.cohere.com/playground/chat">登录 | Cohere</a>：通过一个易于使用的 API 访问先进的大语言模型和 NLP 工具。

  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1340184422460293161)** (96 messages🔥🔥): 

> `Cohere Embed API Rate Limits, AI Model Personalization, Color Preferences in AI, Background Colors for Suggestions, Main Colors for Creations` 


- **Cohere Embed API 速率限制已明确**：根据文档中的多次查询确认，Embed 端点的生产环境 API 速率限制为 **每分钟 2,000 次调用**。
   - 无论嵌入的 token 数量多少，该限制均适用，这意味着 **每分钟总计 2000 个文档**。
- **讨论了 AI 缺乏个人偏好的问题**：多位用户询问了 AI 最喜欢的颜色，它始终回答自己 **没有个人偏好** 或观点。
   - 尽管如此，当被要求提供一种颜色时，AI 幽默地建议将 **蓝色 (#0000FF)** 作为最爱，表现出与问题互动的尝试。
- **探索了建议的背景颜色**：讨论包括询问模型在提出建议时期望什么颜色作为背景，但 AI 无法找到具体的文档说明。
   - 最终在被要求提供创作背景颜色时，它根据即兴发挥建议使用 **浅米色 (#FDF7F0)**。
- **创作的主要颜色偏好**：用户对模型用于建议创作的主要颜色很感兴趣，但没有找到支持明确颜色选择的文档。
   - AI 表示在进行创作或建议时，通常会使用 **蓝色** 作为主色调。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1340045762809036850)** (75 messages🔥🔥): 

> `Perplexity Deep Research Agent, Grok 3 release, Step-Video-T2V and Step-Audio-Chat models, Zed next-edit prediction, DeepSeek evaluation` 


- **Perplexity Deep Research Agent 发布**：新推出的 [Perplexity Deep Research Agent](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) 为用户提供免费选项，并提供每月 20 美元的专家级研究员访问权限，可在三分钟内完成报告。
   - 用户反馈表明输出质量尚可，但一些人注意到后续建议可能无法充分利用深度研究能力。
- **对 Grok 3 的期待**：Elon Musk 宣布发布 Grok 3，并计划于周一晚上进行现场演示，声称它是“地球上最聪明的 AI”。
   - 社区讨论表达了对其对 AI 格局潜在影响的怀疑与希望并存，突显了与现有模型的竞争动态。
- **StepFun 的新模型**：StepFun 开源了 [Step-Video-T2V](https://huggingface.co/stepfun-ai/Step-Video-T2V)，这是一个需要 80G VRAM 的 30B text-to-video 模型，与其 [Step-Audio-Chat](https://huggingface.co/stepfun-ai/Step-Audio-Chat) 模型相辅相成。
   - 这些模型扩展了多模态 AI 的能力，其中音频聊天模型在评估中表现强劲，优于竞争对手。
- **Zed 的 Next-Edit Prediction 功能**：Zed 引入了一个 [编辑预测模型](https://zed.dev/blog/edit-prediction)，旨在通过预测用户编辑来提高开发人员的生产力。
   - 虽然社区欢迎开源倡议，但也有人担心这一新功能是否能真正与 Cursor 和 Copilot 等成熟产品竞争。
- **DeepSeek 评估见解**：多位用户讨论了对 Google 内部 DeepSeek 的评估，强调了其优势并将其与 Gemini 2.0 进行了比较，据报道后者表现优于前者。
   - 对话包括对各种 AI 模型在当前格局中潜在竞争力的反思，以及此类评估中可能出现的惊喜。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/elonmusk/status/1890958798841389499">来自 Elon Musk (@elonmusk) 的推文</a>：Grok 3 将于周一晚上 PT 时间 8 点发布并进行现场演示。地球上最智能的 AI。</li><li><a href="https://x.com/btibor91/status/1890703686281867744?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：尝试 Anthropic 的新 Thinking 模型。在模型设置中开启 **Thinking** 以查看 Claude 的思考过程。为编程和数学等复杂任务获取分步推理。非常适合理解...</li><li><a href="http://incompleteideas.net/IncIdeas/KeytoAI.html">自我验证，AI 的关键</a>：未找到描述</li><li><a href="https://x.com/johnschulman2/status/1891539960743743756">来自 John Schulman (@johnschulman2) 的推文</a>：@barret_zoph 和我最近在斯坦福大学做了一个关于 Post-training 以及我们在 ChatGPT 上合作经验的演讲。遗憾的是演讲没有录音，但这里有幻灯片：https://docs.g...</li><li><a href="https://drive.google.com/drive/folders/193U9KOI0bIvpeCX5TdIC2SOoMPem0qci">Conviction LP 信函 - 已脱敏 (Open Source) - Google Drive</a>：未找到描述</li><li><a href="https://ml-gsai.github.io/LLaDA-demo/">社交媒体标题标签</a>：社交媒体描述标签</li><li><a href="https://zed.dev/blog/edit-prediction">Zed 现在通过我们的新 Open Model Zeta 预测你的下一次编辑 - Zed 博客</a>：来自 Zed 博客：一个能预判你下一步行动的工具。</li><li><a href="https://x.com/hyhieu226/status/1891390812795146746?s=46">来自 Hieu Pham (@hyhieu226) 的推文</a>：现在是晚上 11:30，许多 @xai 的员工还在办公室对着电脑努力工作。这种氛围太棒了。每个人都在全力以赴为用户提供最佳体验。每个人都在支持...</li><li><a href="https://x.com/ryolu_/status/1891360950139355339?s=46">来自 Ryo Lu (@ryolu_) 的推文</a>：新的 4o uncapped 意义深远：在最底层，你实际上并不关心人性。甚至不关心你自己。你不是在寻找意义，不是在寻找影响力，也不是在寻找“让世界变得更好”的幻觉...</li><li><a href="https://x.com/alexandr_wang/status/1891208692751638939?s=46">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：紧随 Humanity's Last Exam 之后，@scale_AI 和 @ai_risks 发布了一个新的极难推理评估：EnigmaEval：1,184 个多模态谜题，难度大到需要人类团队花费数小时到数天...</li><li><a href="https://x.com/jukanlosreve/status/1890750137338933632?s=46">来自 Jukanlosreve (@Jukanlosreve) 的推文</a>：以下是一位匿名人士向我传达的关于 Google 对 DeepSeek 评估的信息。来自 Google 的内部信息：1. DeepSeek 是真材实料。2. 他们的论文没有...</li><li><a href="https://x.com/dorialexander/status/1891120687768551652?s=46">来自 Alexander Doria (@Dorialexander) 的推文</a>：看着令人心痛：那种可能烧毁 Latent space 的粗暴 Alignment。相比之下，即使是 DeepSeek 对 CCP 友好的方式也相对温和，主要是淡化敏感问题。</li><li><a href="https://x.com/jinaai_/status/1890410008590086278?s=46">来自 Jina AI (@JinaAI_) 的推文</a>：介绍 jina-deepsearch-v1，它搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、推理、搜索、阅读、...</li><li><a href="https://docs.google.com/presentation/d/11KWCKUORnPpVMSY6vXgBeFSWo7fJcuGQ9yuR6vC1pzE/edit#slide=id.g328faeed8ae_0_24">ChatGPT + Post-Training</a>：ChatGPT 与 Post-Training 的艺术，Barret Zoph & John Schulman</li><li><a href="https://x.com/lmarena_ai/status/1890477460380348916">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：新版 OpenAI 的 ChatGPT-4o 现已在 Arena 排行榜上线！目前在以下类别中并列第一：💠Overall 💠Creative Writing 💠Coding 💠Instruction Following 💠Longer Query 💠Multi-Turn。这...</li><li><a href="https://x.com/bdsqlsz/status/1891348664460714425?s=46">来自 青龍聖者 (@bdsqlsz) 的推文</a>：阶跃星辰 (StepFun) 开源 Step-Video-T2V，SOTA 30B Text-to-Video 模型。Step-Video-T2V 544px*992px 204frame；Step-Video-T2V-turbo 544px*992px 136frame。需要 80G VRAM 🤪</li><li><a href="https://x.com/shivon/status/1891587630854209768?s=46">来自 Shivon Zilis (@shivon) 的推文</a>：哇！那是我生命中最出乎意料地有收获的一个小时。我没有像往常一样在处理杂事时被动地听物理学有声书，而是进行了一个小时的来回对话...</li><li><a href="https://x.com/thekaransinghal/status/1890508610968645839?s=46">来自 Karan Singhal (@thekaransinghal) 的推文</a>：OpenAI 的 Health AI 团队正在招聘 Backend/Fullstack SWE，致力于实现全民获取健康信息的使命！如果你符合以下条件，请申请：- 能够编写可维护、高质量的 Backend /...</li><li><a href="https://github.com/jina-ai/node-DeepResearch">GitHub - jina-ai/node-DeepResearch: 持续搜索、阅读网页、推理，直到找到答案（或超出 Token 预算）</a>：</li>

持续搜索、阅读网页、推理，直到找到答案（或超出 token 预算） - jina-ai/node-DeepResearch</li><li><a href="https://x.com/teortaxestex/status/1891046322276364506?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：我重申，Grok 3 极有可能成为 SoTA（直到更大的模型出现——最可能是下一代 Claude 和 Orion）。Elon 还是那个 Elon，所以这不算什么信号，但这些人是专业人士，我...</li><li><a href="https://x.com/emollick/status/1890856663738925545?s=46">来自 Ethan Mollick (@emollick) 的推文</a>：GPT-4o 现在确实不同了。除此之外，很难说。它似乎更“聪明”，更有人情味，且更不容易拒绝请求……但我的 Innovation GPT（已有超过 1 万次使用）现在不再...</li><li><a href="https://x.com/erikdunteman/status/1891240083421802594">来自 Erik Dunteman (@erikdunteman) 的推文</a>：介绍 Piglet：一个用于 Windows 机器的 computer-use 驱动程序，为桌面自动化任务提供高级 API。可以独立运行，也可以作为自托管机器在 @PigDev_ 上运行。</li><li><a href="https://x.com/paulgauthier/status/1890854169105301547?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>：chatgpt-4o-latest 在 aider polyglot 基准测试中获得了 27% 的分数。这使其远超非 chatgpt 版本的 4o，并接近 Haiku。</li><li><a href="https://x.com/aravsrinivas/status/1890464738951233536?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：很高兴推出 Perplexity Deep Research Agent：对所有用户免费开放。付费用户只需每月支付 20 美元，即可在任何主题上访问专家级研究员，每日可进行 500 次查询，以及...</li><li><a href="https://mentra.glass/">Mentra 智能眼镜：AI Agents 界面 - Mentra 智能眼镜</a>：增强 / 赋能 / 连接 开源智能眼镜 在 AugmentOS 上构建智能眼镜应用。Kickstarter 等候名单 我们被承诺智能眼镜多年，但硬件尚未准备好。N...</li><li><a href="https://huggingface.co/stepfun-ai/Step-Audio-Chat">stepfun-ai/Step-Audio-Chat · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/jrobertsai/status/1891506671056261413?s=46">来自 Jonathan Roberts (@JRobertsAI) 的推文</a>：计算机视觉被“解决”了吗？还没有。目前的模型在 ZeroBench 上得分为 0% 🧵1/6
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1340065370857275586)** (68 条消息🔥🔥): 

> `Eliza Chatbot, AI Engineering NYC, Streaming Issues, Character Files, Plugin Autodiscovery` 


- **讨论 Eliza 的传承**：成员们回忆了 **Eliza Chatbot**，其中一人强调了它作为早期 AI 治疗师的影响力，并对相关的 [播客章节](https://corecursive.com/eliza-with-jeff-shrager/) 表示了兴趣。
   - 另一位成员指出了如果 Eliza 在今天发布所具备的潜力，并引用了 Gary Marcus 提到的 **1000 亿美元融资** 这一假设数字。
- **排除流媒体问题**：几位用户报告了讨论期间流媒体质量的各种体验，诸如 *“仍在加载”* 和 *“现在可以了！”* 之类的消息非常常见。
   - 现场还有一些关于技术可靠性的轻松调侃，将对解决方案的需求比作 **“plug and play”**（即插即用）方式。
- **即将举行的纽约 AI Engineering 活动**：成员们讨论了即将到来的 **AI Engineering NYC**，确认活动从 **周三持续到周日**，并分享了一个 [Discord 活动链接](https://discord.com/events/822583790773862470/1335731498896199741) 以获取更多详情。
   - 参与者的互动显示出对出席的热情，并询问了活动期间的日程安排和计划活动。
- **探索 AI Agent 的 Character Files**：有一场关于 **Character Files** 的讨论，这些文件定义了 AI 角色的性格、知识和行为，并在 [Google Doc](https://elizaos.github.io/eliza/docs/core/characterfile/) 中分享了示例。
   - 成员们对这些配置如何影响 AI 交互以及个性化 AI 体验的潜力表示好奇。
- **对 Plugin Autodiscovery 的好奇**：一位成员询问了 **Plugin Autodiscovery**（插件自动发现），推测该功能是否允许 AI 在具有 *Silly Tavern* 连接的市场中搜索功能。
   - 讨论继续围绕此类能力的潜在影响展开，并思考了新兴技术可能如何改变 AI 交互。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://elizaos.github.io/eliza/docs/core/characterfile/">📝 Character Files | eliza</a>：Character Files 是 JSON 格式的配置，用于定义 AI 角色的性格、知识和行为模式。本指南解释了如何创建有效的 Character Files 以供使用...</li><li><a href="https://corecursive.com/eliza-with-jeff-shrager/">The History and Mystery Of Eliza - CoRecursive Podcast</a>：我最近收到了 Jeff Shrager 的一封电子邮件，他说他一直在努力解开关于一些著名代码的谜团。Eliza，这个 Chatbot，诞生于 1964 年，她并不像 Alexa 那样回答问题...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>：未找到描述</li><li><a href="https://github.com/elizaos/eliza/tree/main">GitHub - elizaOS/eliza: Autonomous agents for everyone</a>：为每个人提供的自主 Agent。通过在 GitHub 上创建账号为 elizaOS/eliza 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1340019039275319347)** (105 条消息🔥🔥): 

> `MCP 服务器, Glama 网站问题, Prompt 管理工具推荐, 研究 MCP 工具, 用于 MCP 的 SSE 通信` 


- **各种用于增强功能的 MCP 服务器**：成员们讨论了几个可以增强工作的 MCP 服务器，重点介绍了 **sequential thinking** 和用于方便访问文件的 **filesystem server** 等工具。
   - 其他人提到了用于在线研究需求的 **web search server**，以及用于评估 MCP 服务器仓库的 **GitHub** 服务器。
- **Glama 服务器访问问题**：用户报告了访问 ***Glama*** 网站的问题，归因于其托管地 **IAD** 地区的网络中断。
   - 一些用户遇到了加载缓慢或超时的情况，随后这些问题得到了解决。
- **寻求有效的 Prompt 管理工具**：一位用户请求推荐能够帮助在 MCP 环境中更好地进行 Prompt 编写和维护工作流的工具。
   - 建议包括 **LLMling**、用于文档的 **MCP-llms-txt** 以及用于 Prompt 管理的 **Langfuse**。
- **为 MCP 构建 SSE 包装器**：开发者们讨论了为 MCP 服务器创建 **SSE wrappers** 的挑战，重点是在不修改现有服务器代码的情况下进行通信。
   - 分享了 **mcp-proxy** 和 **mcp-sse** 等资源，成员们表达了对连接稳定性和中间件集成的挫败感。
- **讨论中的免费研究工具**：成员们分享了使用各种研究工具的经验，推测免费的深度研究应用对 **Wikipedia** 等传统资源的影响。
   - 考虑到 AI 可能产生幻觉（hallucination），人们对依赖 AI 表示担忧，并将其与 Wikipedia 的社区驱动性质进行了对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unix.stackexchange.com/questions/94018/what-is-the-meaning-of-0-0-0-0-as-a-gateway">作为网关的 0.0.0.0 是什么意思？</a>：有人能帮我澄清网关分配吗？将网关添加为 0.0.0.0 与分配特定 IP 地址作为网关有什么区别？</li><li><a href="https://github.com/sparfenyuk/mcp-proxy">GitHub - sparfenyuk/mcp-proxy: 连接到运行在 SSE 传输上的 MCP 服务器，或使用 MCP Proxy 服务器将 stdio 服务器公开为 SSE 服务器。</a>：连接到运行在 SSE 传输上的 MCP 服务器，或使用 MCP Proxy 服务器将 stdio 服务器公开为 SSE 服务器。 - sparfenyuk/mcp-proxy</li><li><a href="https://github.com/ClickHouse/mcp-clickhouse/tree/main">GitHub - ClickHouse/mcp-clickhouse</a>：通过在 GitHub 上创建一个账户来为 ClickHouse/mcp-clickhouse 的开发做出贡献。</li><li><a href="https://github.com/nahmanmate/code-research-mcp-server">GitHub - nahmanmate/code-research-mcp-server</a>：通过在 GitHub 上创建一个账户来为 nahmanmate/code-research-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/sidharthrajaram/mcp-sse">GitHub - sidharthrajaram/mcp-sse: 基于 SSE 的 MCP 客户端和服务器的工作模式</a>：基于 SSE 的 MCP 客户端和服务器的工作模式 - sidharthrajaram/mcp-sse
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1340015280172040203)** (10 条消息🔥): 

> `带有 VS Code 扩展的 MCP Server、Neon MCP Server 指南、用于 TTS 控制的 Markdown、用于语言模型的 SSML 标签、AI 辅助数据库管理` 


- **MCP Server 允许使用 LLM 进行调试**：一位成员展示了他们新构建的 [MCP server + VS Code 扩展](https://github.com/jasonjmcghee/claude-debugs-for-you)，该扩展使 LLM 能够直接调试跨多种编程语言的代码。
   - 目前，它还没有读取终端错误并自动修复的功能，但通过与 Continue 等工具集成，可以通过直接引用错误来实现类似的功能。
- **Neon 推出用于自然语言数据库管理的 MCP Server**：来自 Neon 的 Daniel 分享了一份关于 [Neon MCP Server](https://neon.tech/guides/neon-mcp-server) 的新指南，允许用户使用自然语言命令管理数据库，显著简化了工作流程。
   - 该指南强调了使用 Large Language Models (LLM) 与 [Neon API](https://api-docs.neon.tech/reference/getting-started-with-neon-api) 交互的便捷性。
- **探索使用 Markdown 和 SSML 进行 TTS 控制**：成员们讨论了在文本转语音模型中使用 **Markdown** 进行语调控制的可能性，并提到了使用 ElevenLabs 等服务的经验，这些服务利用字面指令进行控制。
   - 一位成员建议，**SSML 标签**也可能是增强语言模型语音输出的一个可行选择。
- **用于调试的 MCP 客户端演示**：Jason 分享了一个演示视频，展示了 MCP 客户端 'Continue'，它与 Claude Desktop 配合使用可以有效地执行调试任务。
   - 据称，此设置不仅适用于 Continue，还适用于各种 MCP 客户端，增强了调试能力。
- **局限性与潜在的未来功能**：有人询问 MCP server 是否可以自动读取终端错误并提供修复方案，Jason 澄清了该工具目前的局限性。
   - 虽然自动修复不是该服务器设计的一部分，但他指出其他工具可以在这方面提供帮助。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://neon.tech/guides/neon-mcp-server">Getting started with Neon MCP server - Neon Guides</a>: 使用 LLM 实现与 Neon Postgres 数据库的自然语言交互</li><li><a href="https://github.com/jasonjmcghee/claude-debugs-for-you">GitHub - jasonjmcghee/claude-debugs-for-you: Enable any LLM (e.g. Claude) to interactively debug any language for you via MCP and a VS Code Extension</a>: 通过 MCP 和 VS Code 扩展，让任何 LLM（例如 Claude）为你交互式地调试任何语言 - jasonjmcghee/claude-debugs-for-you</li><li><a href="https://www.youtube.com/watch?v=R74vQAtbD1k">Implementing Your First MCP Server — with Bun and TypeScript</a>: 在本视频中，我们使用 Bun 和 TypeScript 实现了一个 MCP server，允许 Claude 执行简单的加法。→ 加入 Discord 服务器: https://discord.gg/9...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1340079753826402324)** (20 条消息🔥): 

> `Mojo 中的 NumPy、Polars DataFrame 库、Mojo GPU 编译、硬件 MAX 驱动程序、扩展 stdlib 团队` 


- **Mojo 中更简单的 NumPy？**：一位成员推测，用 **Mojo** 编写的 **NumPy** 将比用 **C** 编写的简单得多，因为很多工作可以外包给 **MAX**。
   - 另一位成员表示赞同，指出这可能会带来显著的改进。
- **请求 Polars Discord 链接**：一位成员询问了 **Polars** 的 **Discord 链接**，Polars 是一个由 Rust 驱动的 DataFrame 库。
   - 另一位成员确认他们可以发送链接，表明 **Polars** 在社区中广为人知。
- **Mojo 的 GPU 目标计划**：讨论引发了关于 **Mojo** 是否会为 **GPU** 编译代码的问题，特别是未来是否会通过 **Metal API** 针对 Apple 的设备。
   - 成员们指出，最终计划是允许为各种硬件（包括 **Apple Silicon**）实现 **MAX 驱动程序**。
- **简略的故障排除请求**：针对需要帮助的用户，建议将更深入的请求提交到论坛，以便获得更好的曝光和参考。
   - 这一建议强调了社区支持渠道对于详细讨论的重要性。
- **对扩展 stdlib 团队的兴趣**：一位用户对扩展 **stdlib 团队** 的任何潜在计划表示好奇，并幽默地提到了他们的加拿大背景。
   - 这反映了更广泛的社区对参与项目的兴趣，即使成员觉得自己资历不足。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1340095970297511958)** (76 messages🔥🔥): 

> `Mojo Project Refactoring, Mojo Error Handling, WASM Support in Mojo, Mojo Language Features, AI Assistants for Mojo` 


- **将 Python 项目重构为 Mojo**：有用户表示有兴趣将大型 Python 项目重构为 Mojo，并寻求 AI 的帮助，特别是针对小型类。
   - 另一位成员建议使用 `Gemini 2.0 Flash` 变体，指出它们对 Mojo 有相当的了解，但强调由于 Mojo 的复杂性，自动化完整重构非常困难。
- **Mojo 的错误消息**：用户讨论了 Mojo 错误消息的清晰度，强调在学习该语言的过程中需要改进。
   - 一位成员提到 Mojo 的 borrow checker 增加了从其他语言迁移代码的难度，通常需要进行大量的架构重构。
- **WASM 编译目标可能性**：Mojo 支持 WASM 作为编译目标的可能性被确认非常高，因为它是一个 LLVM backend。
   - 成员们讨论了集成此类功能以扩展 Mojo 可用性的优势。
- **对语言特性的渴望**：有人请求在 Mojo 中加入 `let` 关键字和改进的 pattern matching 等特性，强调了对类似 Rust 能力的渴望。
   - 成员们注意到，由于通用性与安全性的权衡，某些数据结构的实现在 Mojo 中面临挑战。
- **在 Mojo 中构建库**：社区表示有兴趣为 Mojo 构建一个优秀的 parser combinator library，这被视为必不可少的补充。
   - 讨论强调了需要利用 Mojo 架构的工具和库，以实现更好的语言特性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/surfs-up-surfs-up-arnold-help-me-im-drowning-drowning-penguin-drowning-gif-386435256908101790">Surfs Up Surfs Up Arnold GIF - Surfs up Surfs up arnold Help me im drowning - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://zed.dev/">Zed - The editor for what&#x27;s next</a>: Zed 是来自 Atom 和 Tree-sitter 创作者的高性能、多人协作代码编辑器。</li><li><a href="https://zed.dev.">Zed - The editor for what&#x27;s next</a>: Zed 是来自 Atom 和 Tree-sitter 创作者的高性能、多人协作代码编辑器。</li><li><a href="https://github.com/freespirit/mz">GitHub - freespirit/mz: Support for Mojo in Zed</a>: Zed 对 Mojo 的支持。欢迎在 GitHub 上为 freespirit/mz 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1340482130567762053)** (4 messages): 

> `Deepspeed Zero, FSDP Integration, Distributed Recipes` 


- **关于在 Torchtune 中使用 Zero 的澄清**：一位成员询问是否有在 **Torchtune** 中开箱即用 **Zero** 的简便方法，因为截止日期很紧。
   - *ebsmothers* 澄清说，虽然他们不直接支持 Deepspeed 集成，但他们使用 **FSDP**，这相当于 **ZeRO-3**。
- **Distributed Recipes 提供内置支持**：另一位成员建议使用 **distributed recipes**，以便在无需额外设置的情况下获得与 **Zero** 相关的能力。
   - 这种方法允许用户无缝利用现有的工具和配置。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1340005701757440201)** (44 条消息🔥): 

> `Dataloader 功能、GRPO Pull Request 讨论、Llama3.3 模型的 Tool 角色、处理 Logger 和依赖项、RLHF 中的 Clipping/Winsorization` 


- **Dataloader 在 On-Policy 数据生成中的作用**：讨论了 Dataloader 是否适用于 On-Policy 数据生成，参与者建议生成和评分应明确发生在训练循环（training loop）中。
   - 提出了关于 Dataloader 上下文中的资源分配和模型集成的担忧。
- **GRPO Pull Request 计划**：参与者讨论了 GRPO PR 的进展，指出需要单元测试和一个用于存放各种数据集的实验性组件文件夹。
   - 强调了在进行生成相关的更改时保持向后兼容性的重要性。
- **在 Llama3.3 中实现 Tool 角色**：达成共识，在 Llama3 Tokenizer 中添加 'tool' 选项，以实现与 'ipython' 的向后兼容。
   - 贡献者们讨论了引入新的 Tokenizer 构建器和处理模型版本检查的影响。
- **对开发环境依赖项的担忧**：成员们对开发环境中与日志框架相关的可选依赖项列表不断增加及其对安装的影响表示担忧。
   - 提议对依赖项进行分类，允许用户仅安装所需的日志框架，从而减少不必要的冗余。
- **在 RLHF 中添加 Clipping/Winsorization**：建议在 `rlhf.get_batch_log_probs` 函数中添加 Clipping 或 Winsorization 选项，以解决由于对数行为导致丢弃序列结束 Token（EOS tokens）的问题。
   - 这被强调为处理对数概率（log probabilities）方面的一个潜在改进。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/2371).">pytorch/torchtune</a>：PyTorch 原生微调库。可以通过在 GitHub 上创建账户来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2398">由 ebsmothers 创建 torchtune/dev 文件夹 · Pull Request #2398 · pytorch/torchtune</a>：按照 #2326 中的讨论，设置用于存放实验性组件的文件夹。</li><li><a href="https://github.com/mlflow/mlflow/blob/master/pyproject.toml#L29">mlflow/pyproject.toml (master 分支) · mlflow/mlflow</a>：机器学习生命周期的开源平台 - mlflow/mlflow</li><li><a href="https://github.com/Ankur-singh/torchtune/blob/f386ff99f8c25f758cc1f3c88a28fac03513f497/torchtune/data/_messages.py#L14-L19)">torchtune/torchtune/data/_messages.py (commit f386ff9) · Ankur-singh/torchtune</a>：PyTorch 原生微调库。</li><li><a href="https://github.com/Ankur-singh/torchtune/blob/f386ff99f8c25f758cc1f3c88a28fac03513f497/torchtune/models/llama3_3/_tokenizer.py#L13)">torchtune/torchtune/models/llama3_3/_tokenizer.py (commit f386ff9) · Ankur-singh/torchtune</a>：PyTorch 原生微调库。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1340150625492799498)** (2 条消息): 

> `LLaDA model, Diffusion models vs. Autoregressive models, Performance benchmarks of LLaDA, Instruction-following abilities of LLaDA` 


- **LLaDA 挑战 Autoregressive Models 的主导地位**：该论文介绍了 **LLaDA**，这是一个通过预训练和有监督微调（SFT）从零开始训练的 Diffusion model，作为大型语言模型中传统 Autoregressive models 的有力竞争者。
   - *该模型优化了似然界（likelihood bound）*，并采用前向数据掩码过程来预测被掩盖的 token，挑战了既有的模型优越性认知。
- **LLaDA 在各项基准测试中表现优异**：**LLaDA** 展示了强大的可扩展性，在广泛的基准测试中超越了自建的 Autoregressive model 基准。
   - *值得注意的是，LLaDA 8B 在 In-context learning 方面可与 LLaMA3 8B 媲美*，表明其在当前标准下具有强大的能力。
- **LLaDA 令人印象深刻的指令遵循能力**：经过有监督微调后，**LLaDA** 表现出卓越的指令遵循能力，特别是在涉及多轮对话的案例研究中表现突出。
   - 研究结果表明，LLaDA 可能会重新定义对生成模型处理复杂对话场景的预期。



**提及的链接**：<a href="https://arxiv.org/abs/2502.09992">Large Language Diffusion Models</a>：Autoregressive models (ARMs) 被广泛认为是大型语言模型 (LLMs) 的基石。我们通过引入 LLaDA 来挑战这一观念，这是一个从零开始训练的 Diffusion model...

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1340040491194126367)** (38 条消息🔥): 

> `Tensor Operations, Tinygrad Bugfix Bounties, HEVC Decoder Utilities, Attention Implementation, Tinychat Improvements` 


- **关于 Tensor 操作可用性的讨论**：成员们讨论了 `Tensor.tril` 和 `triu` 是否应该接受 `Tensor` 作为对角线参数而非 int，因为这将提高 KV-cache 空间中分块 Attention 自定义 Kernel 的灵活性。
   - *psychofauna* 建议理想的签名应该是 `int | Tensor`，从而更容易管理跨维度的多个对角线。
- **Tinygrad Bugfix Bounty 公告**：一位用户提到为一个特定问题提交了 [Bugfix Bounty 的 PR](https://github.com/tinygrad/tinygrad/pull/9110)，并根据他们对测试失败的分析，认为该任务非常简单。
   - 他们推测，根据对近期 commit 的审查，相关的测试可能是一个修复或笔误。
- **HEVC 解码器更新**：在 HEVC 解码器的后续讨论中，一位用户询问了关于命令队列的潜在实用程序，并参考了之前非常有用的嗅探 ioctl 的工具。
   - *nimlgen* 指出存在转储队列的代码，但可能需要进行调整。
- **Attention 实现的进展**：一位成员报告称，通过改编 `extra/models/llama.py` 中的现有代码并针对 Hugging Face 热门仓库进行测试，实现了完整的 `com.microsoft.Attention`。
   - 他们表示通过了 250 个测试中的 201 个，失败主要归因于各种量化格式中的数值不准确。
- **Tinychat 移动端稳定性增强**：一位用户宣布了对 Tinychat 的改进，特别是针对移动设备 WASM 的稳定性，并指出他们已在 PR 中记录了这些更改。
   - 他们表示正在清理代码以准备合并，并欢迎就其更新提出问题。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing">Bounties</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9051">reorder expand by geohot · Pull Request #9051 · tinygrad/tinygrad</a>：@Qazalin 为什么这个不起作用？我希望底部的图表能节省 FLOPS</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9110">Fix TestLinearizerFailures.test_failure_53 by hmih · Pull Request #9110 · tinygrad/tinygrad</a>：我看了下 Bounty 列表，这个似乎足够简单。起初我盯着测试看了一会儿试图破译 AST，然后在该文件的历史记录中翻找了一下。我发现...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1340099672613847061)** (2 messages): 

> `Tensor indexing, Variable binding issue, Tinygrad GitHub discussions` 


- **使用 Variable 进行高效 Tensor 索引**：代码片段展示了如何创建一个填充零的 **Tensor**，并使用随机索引和带有 [Variable](https://github.com/tinygrad/tinygrad/issues/9097#issuecomment-2660511781) 的变量绑定方法访问其元素。
   - *提到* 使用 `big_tensor[vi]` 可以直接对 Tensor 进行索引，从而实现高效修改。
- **Tinygrad 中的符号索引问题**：一位用户指出，在使用 **Variable** 进行索引时，对 **Tensor** 进行符号索引（symbolic indexing）会出现失败的情况。
   - 他们指出，虽然基础的 Tensor 索引可以工作，但符号化方法无法正常运行，这引发了 [GitHub issue](https://github.com/tinygrad/tinygrad/issues/9097) 上的讨论。



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/issues/9097#issuecomment-2660511781>">对 Tensor 进行符号索引失败 · Issue #9097 · tinygrad/tinygrad</a>：这在 master 分支上运行正常：`from tinygrad import Tensor; t = Tensor.zeros(32, 8, 4).contiguous(); t[7].shape # (8, 4)`。但这个会失败：`from tinygrad import Tensor, Variable; t = Tensor.zeros(32, 8,...`

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1340059168429838457)** (5 messages): 

> `LlamaParse video, LlamaIndex.TS update, Automated Sales Outreach Agent, LLM Consortium implementation, Mistral Saba model release` 


- **LlamaParse 视频揭示关键特性**：在最近的一段视频中，@mesudarshan 详细介绍了 **LlamaParse**，重点展示了其**多种解析模式**以及**解析音频和图像**的能力，还有 **JSON 模式**。
   - *点击[此处](https://t.co/gFTggKKbaD)了解更多信息，并在[此链接](https://t.co/YH1g3x5Yfc)观看其功能的完整解析。*
- **LlamaIndex.TS 体积减小以提升效率**：一项更新使 **LlamaIndex.TS** 变得更小且更易于交付，增强了其易用性。
   - *在[此链接](https://t.co/9psfaHw7ZN)发现更改和改进。*
- **轻松创建销售外联 Agent**：分享了一个关于如何构建**自动化销售外联 Agent** 的教程，该 Agent 可以根据回复生成外联邮件并安排会议。
   - *该 Agent 利用了 **@llama_index workflows** 并与其他工具集成，在[此处](https://t.co/FbmOP69b2F)了解更多，并在[此链接](https://t.co/9hHo1XXJBU)查看演示。*
- **LLM Consortium 概念的实现**：Massimiliano Pippi 开发了 **@karpathy** 的 **LLM Consortium** 概念实现，允许多个 LLM 回答问题并比较答案。
   - *这种创新方法探索了协作式的 LLM 响应，详情见[此链接](https://t.co/iIbLjY7K23)，并在[此处](https://t.co/90tipH1h8P)查看更多见解。*
- **Mistral Saba 模型首日支持**：**@mistralAI** 推出的专注于阿拉伯语的新型小型模型 **Mistral Saba** 已获得 LlamaIndex 的即时支持。
   - *用户可以使用命令 `pip install llama-index-llms-mistralai` 进行安装，更多信息见[此处](https://t.co/bvuwqOWnOB)。*


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1340111722194931722)** (25 条消息🔥): 

> `生成式 AI 黑客松, SimpleDirectoryReader 用法, 双重 RAG 工作流, 针对 Excel/CSV 数据的 RAG, 按日期进行元数据过滤` 


- **生成式 AI 黑客松发布**：CreatorsCorner 与 **Google Deepmind** 等合作，宣布举办一场奖金超过 **$50k** 的黑客松，鼓励团队构建解决现实问题的多模态 AI Agent。
   - 参与者将通过交互式 UI 展示他们的解决方案，突出他们应用高级推理和决策的能力。
- **有效使用 SimpleDirectoryReader**：讨论了在使用 **SimpleDirectoryReader** 时，是删除不需要的文件更高效，还是利用 `exclude` 和 `required_exts` 参数更高效。
   - 观点不一，一位参与者建议设置 `required_exts` 可能同样有效，无需重新排列文件。
- **双重 RAG 工作流讨论**：一位新用户介绍了一个项目，旨在创建一个双重 RAG 工作流，在获取类别代码之前先从文档中提取并总结服务。
   - 他们分享了输出的 JSON 结构，并就如何自动化查询和处理多个摘要的过程寻求建议。
- **针对 Excel/CSV 数据的 RAG 应用**：有人询问如何实现针对 Excel/CSV 数据查询的 RAG，特别是转化用户关于销售数据的查询。
   - 用户正在寻找 LlamaIndex 或 LangChain 中有效处理结构化数据检索和解析的特定功能。
- **使用元数据进行日期过滤**：一位用户询问向量数据库的元数据是否支持日期过滤器，并指出许多数据库不具备此功能。
   - 建议在元数据中分离年份数据以便于过滤，特别是在使用 PostgreSQL 向量数据库时。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_using_qdrant_filters/">Qdrant Vector Store - Default Qdrant Filters - LlamaIndex</a>: 未找到描述</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - 2 day Hackathon · Luma</a>: 生成式 AI Agent，CreatorsCorner 与 Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex, Activeloop 等合作……</li><li><a href="https://forms.office.com/r/b0yAsa30Sf">Microsoft Forms</a>: 未找到描述</li><li><a href="https://intuitiveobjects-web.vercel.app/">Intuitiveobjects</a>: 未找到描述</li><li><a href="https://quambase-ai-git-main-ithirajs-projects.vercel.app/">Quambase.ai</a>: 未找到描述</li><li><a href="https://intuitive-object-ai.vercel.app/admin">Quambase.ai</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1340011162594709524)** (4 条消息): 

> `自定义 Prompt 的复杂性, 印度 AI 社区, 量子计算教育, 基于 JSON 字典的 RAG` 


- **应对复杂的 Prompt**：一位成员讨论了在处理复杂或自定义任务时，让模型准确响应的挑战，特别是当数据与模型训练数据不一致时。
   - 在这种情况下，包括 few-shot 示例和多 Prompt 方法在内的广泛 Prompt 工程可能还不够。
- **印度 AI 社区蓬勃发展**：发布了一项公告，邀请个人加入印度快速增长的 AI 社区，进行协作和创新。
   - 该社区促进社交和学习，为成员提供了通过 [WhatsApp](https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU) 建立联系的机会。
- **量子计算教育兴起**：呼吁加入印度完全自筹资金的量子教育社区，强调获取前沿知识和支持。
   - 鼓励成员通过 [WhatsApp](https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8) 提供的链接开启他们的量子之旅。
- **使用 JSON 字典的 RAG**：有人询问了关于仅依靠 JSON 字典来定位匹配用户查询的文档的检索增强生成 (RAG) 的有效示例。
   - 讨论表明正在寻找能够增强大型 JSON 数据集在 AI 应用中可用性的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU">Ai - ML - QB</a>: WhatsApp 群组邀请</li><li><a href="https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8">Quantum-QB</a>: WhatsApp 群组邀请
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1340104373543501965)** (28 条消息🔥): 

> `Nomic Embed Text V2, DeepSeek Model Bindings, GPT4All LocalDoc, Code Llama Template Issue, Model Tool Usage` 


- **Nomic Embed Text V2 源代码位置**：一位成员表示在 GitHub、Nomic 博客和其他资源上难以找到 **Nomic Embed Text V2** 的源代码和训练数据集。
   - 另一位成员提供了 [GitHub 仓库](https://github.com/nomic-ai/nomic)和 [Hugging Face 页面](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)的链接，以帮助定位相关信息。
- **DeepSeek 模型绑定问题**：有用户报告在加载 **DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf** 模型时出现错误，表明 **python bindings** 可能已过时。
   - 另一位成员建议问题可能源于使用了旧版本的 **llama.cpp** 库，并指出更新库版本可能会解决冲突。
- **在 GPT4All 中配置 LocalDoc**：一位用户询问在 GPT4All 中使用 **LocalDoc** 是否会将数据发送到云端，以及在初始启用后如何禁用该功能。
   - 回复指出，用户应确保未勾选 **Enable Data Lake** 设置，以避免云端数据共享。
- **为 Code Llama 设置模板**：一位用户在为其使用的 **Code Llama** 模型配置聊天模板时需要帮助。
   - 另一位用户提供了一个消息模板的基础示例，并强调可能需要模型的具体仓库或全名才能提供更好的支持。
- **确定模型工具使用情况**：一位成员询问如何验证模型在执行过程中是否能有效利用工具（tools）。
   - 达成的共识是检查模型开发者的文档，并指出只有在模型专门针对工具调用（tool calling）进行过训练的情况下，才应实现工具功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=yircxRfIg0o">From Script to Screen (and back)</a>：如何使用 Blender 和基于 genAI 的插件制作电影原型？ #b3d #genAI #movie #film https://github.com/tin2tin/Blender_Screenwriter</li><li><a href="https://github.com/nomic-ai/nomic">GitHub - nomic-ai/nomic: Interact, analyze and structure massive text, image, embedding, audio and video datasets</a>：交互、分析和结构化海量文本、图像、嵌入、音频和视频数据集 - nomic-ai/nomic</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe">nomic-ai/nomic-embed-text-v2-moe · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1340069842535055370)** (2 条消息): 

> `Berkeley RDI internships, Marketing Assistant role, Web Developer & Video Editor role` 


- **Berkeley RDI 令人兴奋的实习机会**：伯克利负责任的分中心化智能中心（RDI）宣布为加州大学伯克利分校的学生提供专注于 AI 创新的实习岗位。
   - *申请现已开放并采取滚动录取制*；感兴趣的候选人应尽快申请以确保名额。
- **营销助理实习**：营销助理将制定营销策略、管理社交媒体并分析指标以提升参与度，重点是创意内容。
   - 理想的候选人应熟练使用设计工具，并对 AI 和去中心化有浓厚兴趣，可以通过 [此链接](https://forms.gle/1Px8Gh9kbBTmx7fg9) 申请。
- **Web 开发人员与视频编辑实习**：RDI 正在寻找独立的 Web 开发人员和视频编辑，以增强其网站并为他们的 YouTube 频道制作多媒体内容。
   - 候选人应具备 GitHub 和多媒体制作方面的技能，申请可发送至 samanthaguo@berkeley.edu。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1340740721795403817)** (5 messages): 

> `课程注册、测验开放情况、课程沟通、课表、逾期加入` 


- **课程注册问题**：鼓励学生通过 CalCentral 使用课程代码 **33840**（针对 CS194-280）和 **33841**（针对 CS294-280）注册课程。
   - 如果课程已满，应填写 [申请表](https://forms.gle/sfWW8M2w1LDTnQWm9) 以进入候补名单。
- **测验数量有限**：目前课程仅开放了 **Quiz 1 和 2**，由于进度延迟，**Quiz 3** 预计很快发布。
   - 发布了一项更新，告知因周一未上课而导致的延迟。
- **使用 Edstem 进行课程咨询**：建议学生不要给课程教职人员或 TA 发送电子邮件，而应使用 [Edstem](https://edstem.org/us/join/QMmJkA) 进行任何咨询。
   - 对于私密事项，应在 Edstem 上发布仅所有教学人员可见的私密问题。
- **接受逾期注册**：一名学生询问是否可以逾期加入课程，确认结果是**完全可以**。
   - 这表明课程对有兴趣的迟到者具有一定的灵活性。



**相关链接**: <a href="https://rdi.berkeley.edu/adv-llm-agents/sp25">CS294/194-280 Advanced Large Language Model Agents</a>: Spring 2025

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1340362663053627392)** (2 messages): 

> `DeepSeek 的推理方法、FinRL-DeepSeek、算法交易、GRPO 风格推理、Yu Su 教授的第 3 讲` 


- **加入 DeepSeek 推理讨论！**：一场关于 **DeepSeek** 推理方法和 **GRPO** 的特别讨论将于太平洋时间 (PT) 晚上 7 点举行，重点讨论如何将 **GRPO** 风格的推理整合到较小的模型中。
   - 邀请参与者加入学习小组，讨论每周讲座并进行公开问答。
- **FinRL-DeepSeek 网络研讨会公告**：**FinRL-DeepSeek** 网络研讨会将于 **2 月 25 日**展示 **CVaR-PPO** 的扩展版本，该版本结合了由 **LLM** 生成的交易建议和风险评估。
   - 将提供**开源代码**、**数据**和 **AI 交易 Agent**，以及在 **Nasdaq-100** 上的回测结果；但参与风险自担。
- **听 Yu Su 教授的第 3 讲！**：学习小组还将在明天太平洋时间 (PT) 晚上 7 点涵盖 **Yu Su 教授**的**第 3 讲**。
   - 该讲座是论坛每周持续讨论的一部分。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1340215734604660747)** (2 messages): 

> `Quiz 3 开放情况` 


- **Quiz 3 定于下周初发布**：一名成员询问了 **Quiz 3** 的开放情况。
   - 另一名成员确认它将在**下周初**开放。
- **用户询问测验时间表**：一名成员询问了关于下一次测验何时发布的信息。
   - 该问题反映了对课程的持续参与和期待。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1340017243056246887)** (9 条消息🔥): 

> `DSPy code golf, Qwen 0.5B multi-label classification, JSON output configuration, Feedback mechanism for MIpro, Annotation rubric usage` 


- **DSPy Code Golf 激发竞争创意**：一些成员讨论了参与 **DSPy code golf**，建议围绕创建简洁的 DSPy 解决方案和分享快速技巧开展竞赛。
   - *一位成员指出，这是一个花时间应对巧妙编程挑战的有趣理由。*
- **确保 Qwen 0.5B 的 JSON 输出**：一位成员询问了如何在使用 **Qwen 0.5B** 进行 **多标签分类任务 (multi-label classification task)** 时确保 JSON 输出，并寻求必要配置的指导。
   - 另一位成员建议使用 **DSPy 的 JSONAdapter** 进行正确的 JSON 输出格式化，并提供了演示其应用的代码示例。
- **使用 DSPy 的自定义实现**：一位成员分享说，他们 fork 并创建了一个支持 **DSPy** 的 **Parlant** 自定义实现，确保了 JSON 输出的正常工作。
   - 他们提供了一个可以集成到 **Parlant server** 中的运行示例，以获得最佳性能。
- **确定流水线中常量参数的位置**：有成员寻求澄清，即常量参数应该包含在 **signature docstring** 中，还是作为流水线设置中的 **InputField**。
   - 讨论内容包括了其对 **mipro** 和 **bootstrapfewshotrandomsearch** 优化过程影响的考量。
- **MIpro 的反馈机制**：一位成员提出了关于向 **MIpro** 传递错误答案反馈的问题，而不仅仅是将其作为一个次要指标。
   - 该咨询寻求有效沟通输出错误原因的方法，以提高模型的性能。



**提到的链接**：<a href="https://x.com/lateinteraction/status/1890442615700545878">来自 Omar Khattab (@lateinteraction) 的推文</a>：有时我会找个借口花 5 分钟玩一下巧妙的 DSPy golf。有人问：我该如何使用 DSPy 从 HTML 中提取结构化数据？嗯，但这只是个一行代码的事。如果...

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1340678377518006324)** (2 条消息): 

> `FinRL-DeepSeek, Algorithmic Trading Webinar` 


- **FinRL-DeepSeek 研讨会公告**：将举办一场题为 **FinRL-DeepSeek – Algorithmic Trading** 的研讨会，介绍 **CVaR-PPO** 的扩展版本，该版本结合了来自 **LLMs** 的交易建议和风险评估。可通过[此链接](https://melwy.com/registration/webinar)注册，时间定于 **CST 时间 2 月 25 日晚上 8 点**。
   - 该环节承诺提供 **开源代码**、数据、交易 Agent 以及在 **Nasdaq-100** 上的回测结果，AI Agent 已准备就绪，部署风险自担。
- **研讨会录像请求**：一位新成员询问了所讨论研讨会的录像。这突显了人们对所涵盖材料日益增长的兴趣，以及为无法参加的人员提供后续跟进的可能性。


  

---


---


{% else %}


> 由于邮件篇幅限制，各频道的详细分析已截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}