---
companies:
- mozilla
- llamaindex
- anthropic
- etched-ai
- sohu
- deepseek
- openai
date: '2024-06-27T01:37:35.020344Z'
description: '以下是该文本的中文翻译：


  **Mozilla** 在 AIE 世界博览会（AIE World''s Fair）上展示了 **llamafile** 的详细现场演示，并宣布推出用于向量搜索集成的
  **sqlite-vec**。**LlamaIndex** 推出了 **llama-agents**。**Anthropic** 为 **Claude** 引入了全新的
  UI 功能以及支持 200K 上下文窗口的 **Projects**（项目）功能。**Etched AI** 揭晓了一款专用推理芯片，声称其速度可达 **每秒
  50 万个 token**，尽管其基准测试数据遭到了质疑。**Sohu** 芯片可实现**每秒 15 条智能体轨迹**。**Tim Dettmers** 分享了
  GPU 推理的理论极限：在 8 个 B200 通过 NVLink 连接运行 70B Llama 模型时，速度约为 **每秒 30 万个 token**。**Deepseek
  Coder v2** 在编程和推理能力上超越了 **Gemini** 和 GPT-4 的多个变体。**PyTorch 纪录片** 已发布，但并未引起太多关注。'
id: 5193e590-b36c-4c34-a4f1-59f563eb578c
models:
- llama-3
- claude-3-opus
- gemini-1.5
- deepseek-coder-v2
- gpt-4
original_slug: ainews-mozillas-ai-second-act
people:
- justine-tunney
- stephen-hood
- tim-dettmers
- bindureddy
title: Mozilla 的 AI 第二幕
topics:
- vector-search
- inference-speed
- hardware-benchmarks
- context-windows
- open-source-models
- coding
- reasoning
- model-benchmarking
- gpu-inference
- agentic-ai
---

<!-- buttondown-editor-mode: plaintext -->**极速 CPU 推理就是你所需要的一切。**

> 2024/6/25-2024/6/26 AI 新闻。
我们为您检查了 7 个 subreddits、[384 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discords（416 个频道，3358 条消息）。
预计节省阅读时间（按 200wpm 计算）：**327 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Mozilla Firefox 市场份额](https://en.wikipedia.org/wiki/Usage_share_of_web_browsers#/media/File:StatCounter-browser-ww-yearly-2009-2023.png)的缓慢下降众所周知，在经历[多轮](https://arstechnica.com/information-technology/2020/08/firefox-maker-mozilla-lays-off-250-workers-says-covid-19-lowered-revenue/)[裁员](https://arstechnica.com/gadgets/2024/02/mozilla-lays-off-60-people-wants-to-build-ai-into-firefox/)后，其未来的故事非常不确定。然而，在今天的 AIE World's Fair 开幕主题演讲中，他们强势回归：

 
![image.png](https://assets.buttondown.email/images/e1440ead-35c7-4eed-a606-83f053b95424.png?w=960&fit=max)
 

Justine Tunney 亲自带来了非常详细的 [llamafile 现场演示](https://x.com/thedataroom/status/1806018145926455661)及技术讲解，[Stephen Hood 宣布](https://x.com/aiDotEngineer/status/1806072610683576368)了一个非常受欢迎的第二个项目 `sqlite-vec`，正如你所料，它为 sqlite 增加了向量搜索功能。

您可以在直播中观看完整演讲（从 53 分钟处开始）：

https://www.youtube.com/watch?v=5zE2sMka620&t=262s


LlamaIndex 也以发布备受瞩目的 [llama-agents](https://x.com/llama_index/status/1806116419995844947) 为当天画上句号。

 
![image.png](https://assets.buttondown.email/images/74816a0f-0cc4-4ca3-934d-ee691bbfa2f1.png?w=960&fit=max)
 



一些道歉：昨天我们漏掉了 [Etched 的重大发布](https://x.com/Etched/status/1805625693113663834)（[受到质疑](https://x.com/cHHillee/status/1805696613480022238?utm_source=ainews&utm_medium=email)），而 [Claude Projects](https://www.anthropic.com/news/projects) 引起了轰动。[PyTorch 纪录片](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s)发布后反应冷淡（奇怪？）。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**Anthropic Claude 更新**

- **新 UI 功能**：[@alexalbert__](https://twitter.com/alexalbert__/status/1805617407375065498) 注意到了 Claude UI 的新功能，包括用于收藏对话的**侧边栏 (sidebar)**、具有 200K context windows 用于文档和文件的**可共享项目 (shareable projects)**，以及用于定制回答的**自定义指令 (custom instructions)**。
- **Anthropic 发布 Projects**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1805616725733339199) 推出了 Projects，允许将对话组织成可共享的知识库，并为相关文档、代码和文件提供 200K context window。适用于 Claude Pro 和 Team 用户。

**硬件和性能基准测试**

- **Etched AI 专用推理芯片**：[@cHHillee](https://twitter.com/cHHillee/status/1805696613480022238) 分享了对 Etched 新推理芯片的看法，指出其在芯片效率和性能方面的营销主张可能存在**误导性**。基准测试声称达到 **500k tokens/sec**（针对多用户），并能用**一台 8x Sohu 服务器取代 160 块 H100**，但这些数据可能未针对关键细节进行标准化。基准测试方法论需要更多信息。
- **Sohu 芯片实现每秒 15 个 Agent 轨迹**：[@mathemagic1an](https://twitter.com/mathemagic1an/status/1805636415772147839) 强调，Sohu 上 500k tokens/sec 的速度意味着**每秒可处理 15 个完整的 30k token Agent 轨迹**，并强调基于这种算力假设进行开发的重要性，以避免被淘汰。
- **理论 GPU 推理极限**：[@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1805701944746590549) 分享了一个模型，估算在 70B Llama 上进行 8xB200 NVLink 8-bit 推理的**理论最大值约为 300k tokens/sec**（假设采用类似 OpenAI/Anthropic 的完美实现）。这表明 Etched 的基准测试似乎偏低。

**开源模型**

- **Deepseek Coder v2 击败 Gemini**：[@bindureddy](https://twitter.com/bindureddy/status/1805686571108384990) 声称一个开源模型在推理和代码方面击败了最新的 Gemini，更多关于开源进展的细节即将发布。一份[后续推文](https://twitter.com/bindureddy/status/1805747650823962795)提供了具体细节——Deepseek Coder v2 在编程和推理方面表现出色，在数学方面击败了 GPT-4 变体，并在真实生产用例中使开源模型位列第三，仅次于 Anthropic 和 OpenAI。
- **Sonnet 压倒 GPT-4**：[@bindureddy](https://twitter.com/bindureddy/status/1805661535597211832) 分享到，Anthropic 的 Sonnet 模型在各种工作负载的测试中继续压倒 GPT-4 变体，预示了即将推出的模型将令人印象深刻。

**生物 AI 突破**

- **ESM3 模拟进化以生成蛋白质**：[@ylecun](https://twitter.com/ylecun/status/1805581310548697360) 分享了 Evolutionary Scale AI 的消息，这是一家使用名为 ESM3 的 98B 参数 LLM 来“编程生物学”的初创公司。ESM3 模拟了 5 亿年的进化过程，生成了一种新型荧光蛋白。[博客文章](https://twitter.com/ylecun/status/1805634811773571496)包含更多细节。ESM3 由前 Meta AI 研究员开发。

**新兴 AI 趋势与观点**

- **数据丰度是 AI 进步的关键**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1805669049948327995) 强调，突破“数据墙”需要数据丰度方面的创新。AI 模型会压缩其训练数据，因此持续的进步将取决于新数据，而不仅仅是算法。
- **AGI 时代后人类智能的回报**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1805669260745916913) 预测，在 AGI 之后，人类天才的溢价将会增加而非减少，因为只有最聪明的人类才能理解 AGI 正在做什么。
- **多模态 AI 的术语**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1805690638647935016) 指出，将多模态 AI 称为“LLM”正变得奇怪，并征求替代术语的建议，因为模型正在向语言之外扩展。

**梗与幽默**

- [@Teknium1](https://twitter.com/Teknium1/status/1805718678526476655) 开玩笑说 OpenAI 在 GPT-4 语音模型更新中难以移除“waifu features”。
- [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1805602848920715519) 幽默地宣布 Noam Shazeer 因在 AI 女友方面的开创性工作而获得图灵奖。
- [@willdepue](https://twitter.com/willdepue/status/1805688616280293766) 调侃道，既然现在可以在聊天机器人中搜索历史对话，“AGI 已经解决了”。

---

# AI Reddit 摘要

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

AI 进展

- **AI 网站生成**：一个新的 AI 系统可以仅通过 URL 或描述输入生成完整的网页，展示了 AI 内容创建能力的进步。[视频演示](https://v.redd.it/y65tyl0r5r8d1)。
- **OpenAI Voice Mode 延迟**：OpenAI [宣布](https://x.com/openai/status/1805716393524183136?s=46)将高级 Voice Mode 的 alpha 版本发布推迟一个月，以改进安全性和用户体验。计划在秋季向所有 Plus 用户开放。
- **《奇点临近》新书发布**：Ray Kurzweil 发布了他 2005 年著作《奇点临近》（The Singularity is Near）的续作，引发了关于 AI 未来的[兴奋和讨论](https://i.redd.it/dfrwlkvoop8d1.jpeg)。
- **AI Agent 猜测**：OpenAI [收购](https://i.redd.it/swfpyso5jo8d1.png)了一家远程桌面控制初创公司，引发了关于将其与 ChatGPT 桌面版集成以实现 AI Agent 的猜测。
- **AI 生成广告**：Toys R Us 使用 SORA AI 系统[生成了宣传视频/广告](https://www.toysrus.com/pages/studios)，展示了 AI 在营销中的应用。

AI Research

- **新优化器性能超越 AdamW**：一篇[研究论文](https://arxiv.org/abs/2406.16793)介绍了 Adam-mini，这是一种新的优化器，其吞吐量比流行的 AdamW 高出 50%。
- **LLM 中消除矩阵乘法**：研究人员[展示了](https://arstechnica.com/information-technology/2024/06/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms/)消除矩阵乘法的 LLM，从而实现了更高效的模型，这对于在消费级硬件上运行大型模型具有重大意义。
- **用 AI 模拟进化**：EvolutionaryScale [发布了 ESM3](https://www.evolutionaryscale.ai/blog/esm3-release)，这是一种生成式语言模型，可以模拟 5 亿年的进化以生成新的功能性蛋白质。

AI Products & Services

- **Deepseek Coder V2 数学能力**：用户称赞了 Deepseek Coder V2 模型的[数学能力](https://www.reddit.com/r/LocalLLaMA/comments/1do72te/deepseek_coder_v2_is_so_good_for_math/)，这是一款来自中国的免费模型，表现优于 GPT-4 和 Claude。
- **AI 有声读物旁白**：一部 [AI 旁白的有声读物](https://i.redd.it/bc9hntyelr8d1.jpeg)广受好评，这意味着有声读物旁白现在已成为 AI 解决的问题。
- **新 AI 应用与功能**：宣布了几项新的 AI 应用和功能，包括 [Tcurtsni](https://www.reddit.com/gallery/1do5ykm)（一个“反向指令”聊天应用）、[Synthesia 2.0](https://youtu.be/gZaBwdru_bk?si=yHcsnnCJ5750xgPv)（一个合成媒体平台）以及 Claude 中的 [Projects](https://support.anthropic.com/en/articles/9517075-what-are-projects)（用于组织聊天和文档）。

AI Safety & Ethics

- **Rabbit 数据泄露**：一项安全披露揭示了 Rabbit 的[数据泄露](https://www.reddit.com/r/singularity/comments/1do6uxz/rabbit_data_breach_all_r1_responses_ever_given/)，其 R1 模型的所有回复都可以被下载，引发了对 AI 公司疏忽的担忧。 
- **幻觉担忧**：一篇[观点文章](https://www.reddit.com/r/singularity/comments/1do8aqf/hallucinations_could_lead_to_a_bigger_problem/)认为，“AI 幻觉”这一论点是危险的，因为它掩盖了快速进步的 AI 充斥就业市场的真实风险。

AI Hardware

- **AMD MI300X 基准测试**：发布并分析了 AMD 新型 MI300X AI 加速芯片的[基准测试](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/)。
- **Sohu AI 芯片声明**：一款新的 Sohu AI 芯片[发布](https://www.reddit.com/r/LocalLLaMA/comments/1dobzcs/meet_sohu_the_fastest_ai_chip_of_all_time/)，声称在 70B 模型上可达到 500K tokens/sec，8 颗芯片相当于 160 块 NVIDIA H100 GPU。
- **MI300X vs H100 对比**：一项[对比](https://i.redd.it/a8xl65u3ap8d1.jpeg)显示，在 LLaMA-2 70B 模型上，AMD 的 MI300X 比 NVIDIA 的 H100 慢约 5%，但价格便宜 46%，且显存是其 2.5 倍。

AI Art

- **A8R8 v0.7.0 发布**：新版本的 A8R8 Stable Diffusion UI [发布](https://github.com/ramyma/a8r8/releases/tag/v0.7.0)，集成了 ComfyUI 以支持区域提示（regional prompting）及其他更新。
- **ComfyUI 新功能**：一篇[详细文章](https://www.reddit.com/r/StableDiffusion/comments/1dohy20/quick_overview_of_some_newish_stuff_in_comfyui/)回顾了 ComfyUI Stable Diffusion 环境中的新功能，如 sampler、scheduler 和 CFG 实现。
- **Magnific AI 重光照工具**：Magnific AI 新重光照工具的结果与用户的日常工作流进行了[对比](https://www.reddit.com/r/StableDiffusion/comments/1do2nym/comparison_between_magnific_ais_new_relighting/)，发现其质量不足。 
- **SD 模型对比**：[对比](https://www.magicflow.ai/insights/read/sd-body-positions)了不同 Stable Diffusion 模型大小在生成指定身体姿势方面的表现，结果被指出“不佳”。

Other Notable News

- **Stability AI 领导层变动**：Stability AI [宣布](https://www.reddit.com/r/StableDiffusion/comments/1do9owa/stability_ai_announces_new_ceo_and_investors/)了新任 CEO、董事会成员、融资轮次，以及在扩展企业级工具的同时对开源的承诺。
- **AI 流量分析**：一篇 [帖子](https://www.reddit.com/r/singularity/comments/1dohcke/request_how_to_quantify_ai_traffic_between/) 提出了量化主要 AI 系统带宽使用情况的方法，估计 AI 仍仅占整个互联网流量的一小部分。
- **政客分享虚假的 ChatGPT 统计数据**：一份新闻报道称，一名加拿大政客分享了由 ChatGPT 生成的不准确统计数据，凸显了使用未经核实的 AI 输出的风险。
- **用于值班的开源 AI Agent**：Merlinn 是一款旨在协助值班工程师的开源 AI Slack 机器人，现已 [发布](https://www.reddit.com/r/singularity/comments/1dohfqo/created_an_opensource_ai_agent_that_helps_during/)。
- **活体皮肤机器人**：BBC [报道](https://www.bbc.com/news/articles/cedd3208veyo) 了一项关于用活体人类皮肤覆盖机器人的研究，使其更加逼真。
- **基因疗法进展**：一条 [推文](https://x.com/natrevdrugdisc/status/1805630241521435078?s=46) 讨论了基因疗法正从罕见病向常见病领域推进。
- **Google AI 活动**：有消息称 Google 将在 8 月的活动中展示新的 AI 技术和 Pixel 手机。
- **调低对 AI 发布日期的预期**：一篇 [帖子](https://www.reddit.com/r/singularity/comments/1dohqjt/reality_check_a_planned_release_window_of_an/) 建议对 AI 产品的发布日期持保留态度，因为研发过程存在不确定性。
- **AI 终结业余主义**：一篇 [评论文章](https://www.reddit.com/r/singularity/comments/1dohwlb/generative_ai_the_end_of_amateurism/) 认为，生成式 AI 将使每个人都能创作出专业水准的作品。

---

# AI Discord Recap

> 摘要的摘要的摘要

## Claude 3 Sonnet

**1. 🔥 LLM 进展与基准测试**

- Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在排行榜上名列前茅，根据 [ChatbotArena](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer) 的数据，其表现优于 GPT-4-Turbo 和 Claude 3 Opus。
- 新模型：用于编程的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)**，拥有 236B 参数的 **[DeepSeek-V2](https://x.com/deepseek_ai/status/1787478986731429933)**。
- 对某些基准测试持怀疑态度，呼吁可信来源设定现实的评估标准。

**2. 🤖 优化 LLM 推理与训练**

- **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 承诺将 GPU 上的通信开销降低 4 倍。
- **[vAttention](https://arxiv.org/abs/2405.04437)** 动态管理 KV-cache 内存以实现高效推理。
- **[QServe](https://arxiv.org/abs/2405.04532)** 使用 **W4A8KV4 量化** 来提升 GPU 上的云端服务性能。
- **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 探索并行 Token 解码以降低延迟。

**3. 🌐 开源 AI 框架与社区努力**  

- **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 支持多种格式的指令微调和预训练。
- **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 为一门关于构建 Agentic RAG 系统的课程提供支持。
- **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 声称是处理“枯燥数据任务”的最佳选择。
- **[Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 预告了 Mojo 的 Python 集成和 AI 扩展。

**4. 🖼 多模态 AI 与生成模型创新**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 用于提升聊天交互体验。 
- **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 优化了编程能力。
- **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 通过 WebGPU 为浏览器带来强大的聊天机器人。
- 结合 **Pixart Sigma + SDXL + PAG** 旨在实现 DALLE-3 级别的输出，并具备微调潜力。
- 开源的 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 用于图像重打光技术。

**5. 用于 Discord AI 媒体创作的 Stable Artisan**

- Stability AI 推出了 **Stable Artisan**，这是一款集成了 **Stable Diffusion 3**、**Stable Video Diffusion** 和 **Stable Image Core** 的 Discord 机器人，用于 [在 Discord 内生成媒体内容](https://bit.ly/4aiVy6C)。
- 引发了关于 SD3 开源状态以及 Artisan 作为付费 API 服务推出的讨论。

## Claude 3.5 Sonnet

1. **LLM 在性能和效率方面实现跨越**：

   - 像 [IBM 的 Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) 和 [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) 这样的新模型正在突破代码指令和数据任务的界限。Discord 频道中的各个社区正在讨论这些进展及其影响。

   - 诸如 [Adam-mini](https://github.com/zyushun/Adam-mini) 等优化技术正受到关注，有望在保持性能的同时，比 AdamW 减少 45-50% 的显存占用。这在 OpenAccess AI Collective 和 CUDA MODE 的 Discord 频道中引发了讨论。

   - 用于高效 KV-cache 内存管理的 [vAttention 系统](https://arxiv.org/abs/2405.04437) 正在作为 PagedAttention 的替代方案被探索，凸显了 AI 社区对推理优化的持续关注。

2. **开源 AI 在社区驱动工具的推动下蓬勃发展**：

   - [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 因其在 LLM 训练中对多样化数据集格式的支持而日益流行，在 OpenAccess AI Collective 和 HuggingFace 的 Discord 中都有讨论。

   - [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) 框架正在为构建代理式 RAG 系统的新课程提供支持，在 LlamaIndex 和通用 AI 开发社区中引起了热烈反响。

   - [Mojo](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo) 在 Python 集成和 AI 扩展方面的潜力是 Modular Discord 的热门话题，讨论集中在它对 AI 开发工作流的影响。

3. **多模态 AI 突破创意边界**：

   - Pixart Sigma、SDXL 和 PAG 的结合正在被探索以实现 DALLE-3 级别的输出，这在 Stability.ai 和通用 AI 社区中得到了讨论。

   - [Stable Artisan](https://bit.ly/4aiVy6C) 是来自 Stability AI 的一款新 Discord 机器人，它集成了 Stable Diffusion 3 和 Stable Video Diffusion 等模型，在多个 Discord 频道中引发了关于 AI 驱动媒体创作的对话。

   - 用于图像重光照的开源 [IC-Light 项目](https://github.com/lllyasviel/IC-Light) 正在计算机视觉圈引起关注，展示了图像处理技术中持续的创新。

4. **AI 硬件竞赛升温**：

   - AMD 的 [Radeon Instinct MI300X](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/) 正在挑战 Nvidia 在 GPU 计算市场的统治地位，尽管面临软件生态系统的挑战。这一直是 CUDA MODE 和硬件导向 Discord 频道的热门话题。

   - [Etched 的 Sohu AI 芯片](https://www.etched.com/) 的发布在 AI 硬件社区引发了辩论，讨论其在运行 Transformer 模型方面超越 GPU 的潜力，并声称可以取代多个 H100 GPU。

   - 关于专用 AI 芯片与通用 GPU 的讨论正在进行，各个 Discord 服务器的社区成员正在辩论 AI 硬件加速的未来方向。

## Claude 3 Opus

**1. LLM 性能与基准测试**：

- 关于各种 LLM 性能的讨论，例如 Meta 的 **Llama 3** 在 [ChatbotArena](https://lmsys.org/blog/2024-05-08-llama3/) 等排行榜上表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。
- IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 和 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 等新模型展示了在指令遵循和参数量方面的进展。
- 对某些基准测试可信度的担忧，以及对来自权威来源的现实 LLM 评估标准的需求。

**2. 硬件进展与优化技术**：

- 正在探索 **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 和 **[vAttention](https://arxiv.org/abs/2405.04437)** 等技术，以优化 GPU 内存使用并减少 LLM 训练和推理过程中的通信开销。
- 量化方面的进展，例如 **[QServe](https://arxiv.org/abs/2405.04532)** 引入了 **W4A8KV4 量化**，以提高云端 LLM 服务中的 GPU 性能。
- 关于 **Etched's Sohu** 等专用 AI 芯片潜力的讨论，以及在运行 Transformer 模型时与 GPU 性能的对比。

**3. 开源框架与社区努力**：

- **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 和 **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 等开源框架支持多样化的数据集格式，并助力开发 Agentic RAG 系统。
- 开源模型 **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 的发布，声称是处理“枯燥数据任务（unsexy data tasks）”的最佳 LLM。
- 社区致力于将 AI 能力集成到 Discord 等平台，例如来自 Stability AI 的 **Stable Artisan** 机器人，用于多媒体生成和编辑。

**4. 多模态 AI 与生成模型**：

- 专注于特定任务的新模型，例如用于提升对话交互的 **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 和用于编程能力的 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)**。
- 基于浏览器的 AI 聊天机器人进展，例如利用 WebGPU 进行强大交互的 **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型。
- 结合 **Pixart Sigma**、**SDXL** 和 **PAG** 等技术，以在生成模型中实现 **DALLE-3** 级别的输出。
- **[IC-Light](https://github.com/lllyasviel/IC-Light)** 等开源项目专注于图像重打光（image relighting）等特定任务。

## GPT4O (gpt-4o-2024-05-13)

1. **模型性能与基准测试**：
   - **[Llama3 70B 模型展现潜力](https://x.com/ClementDelangue/status/1805989925080219927)**：在 300 块 H100 GPU 上托管的新开源 LLM 排行榜显示 Qwen 72B 处于领先地位，尽管更大的模型并不总是等同于更好的性能。分析强调了训练与推理基准测试在范围上的差异。
   - **[解决小学算术问题](https://arxiv.org/abs/2405.00332)**：该研究引发了怀疑，指出大型 LLM 中的数据泄漏导致了误导性的高基准测试结果，尽管学习并不完整。文中呼吁进行可靠的评估。

2. **训练、优化与实现问题**：
   - **[推动更好的优化器](https://arxiv.org/abs/2406.16793)**：Adam-mini 优化器提供与 AdamW 相当的性能，但减少了 45-50% 的内存使用。该优化器通过减少每个参数的学习率数量来简化存储。
   - **[高上下文模型中的内存管理](https://github.com/zyushun/Adam-mini)**：在消费级 GPU 上加载大型模型（如 Llama3 70B 或 Hermes）的尝试受到显著的 OOM 错误的阻碍，引发了关于有效 GPU VRAM 利用率的讨论。

3. **AI 伦理与社区辩论**：
   - **[AI 数据使用的伦理](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main)**：**LAION** Discord 中的辩论强调了数据集中包含 NSFW 内容的争议性，在伦理担忧与无限制数据访问的动力之间寻求平衡。
   - **模型投毒担忧**：**LAION** 的讨论集中在伦理影响和潜在的模型投毒上，即在不广泛考虑长期影响的情况下，鼓励在训练和数据集使用中采用有争议的技术。

4. **专用 AI 硬件趋势**：
   - **[Etched 的 Sohu 芯片号称拥有 10 倍性能](https://x.com/bryan_johnson/status/1805629207374086490)**：Etched 的新型 Transformer ASIC 芯片声称性能显著优于 Nvidia GPU，并获得了可观的资金支持。然而，**CUDA MODE** 内部讨论了其实际适应性和缺乏灵活性的问题。
   - **[AMD 的 MI300X 挑战 Nvidia](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/)**：AMD 的 MI300X 寻求在 GPU 计算市场挑战 Nvidia 的地位，尽管在 Nvidia 的 CUDA 生态系统面前仍显落后。

5. **AI 应用集成**：
   - **[Hugging Face 上的自定义 GPT 应用蓬勃发展](https://github.com/teknium1/Prompt-Engineering-Toolkit)**：对基于自定义 GPT 的应用兴趣日益浓厚，针对日语语句解释等利基任务的需求依然强劲。社区的协作努力推动了资源和工具包的创建，以简化实现过程。
   - **[AI 辅助工具扩展学术影响力](https://gpasaver.com/)**：新的 **GPA Saver 平台** 利用 AI 进行学术辅助，表明 AI 在流线型教育工具中的集成度不断提高。关于改进 AI 驱动功能的社区讨论强调了其潜力和当前的局限。

---

# 第一部分：Discord 高层级摘要

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**通过快捷键快速访问**：适用于 macOS 的 ChatGPT 桌面应用现已发布，具有快速访问的 [Option + Space 快捷键](https://openai.com/chatgpt/mac/)，可与电子邮件和图像无缝集成。

**语音模式延迟**：备受期待的 ChatGPT 高级语音模式（Voice Mode）已推迟一个月，以确保 Alpha 测试前的质量；预计在秋季将推出更多功能，如情绪检测和非语言暗示。

**OpenAI 与 Anthropic 的重量级对决**：关于 GPTs Agent 在训练后无法学习，以及 Anthropic 的 Claude 凭借技术优势（如更大的 Token 上下文窗口和传闻中的 **MoE 架构**）领先于 ChatGPT 的讨论正趋于白热化。

**AI 定制热潮**：爱好者们正利用 **Hugging Face** 等资源创建自定义 GPT 应用，特别关注日语语句解释等利基任务，同时也对 OpenAI 当前模型更新和功能推出的局限性表示担忧。

**GPT-4 桌面应用与性能讨论**：用户注意到新的 macOS 桌面应用仅限于 Apple Silicon 芯片，并对 GPT-4 的性能评价褒贬不一，表达了对 Windows 应用支持和响应时间改进的渴望。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **显微镜下的 RAG**：一场围绕 **Retrieval-Augmented Generation (RAG)** 技术使用的讨论，强调了在使用 **Mamba** 等 SSM 时管理文档长度的考量，以及使用 **BM25** 进行面向关键词的检索。可以在[此处找到](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb)与 BM25 相关的 GitHub 资源。

- **交互式手势**：两个不同的语境都提到了一个基于 Python 的“**Hand Gesture Media Player Controller**”，并通过 [YouTube 演示](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM)进行了分享，表明了人们对应用计算机视觉控制界面的兴趣日益增长。

- **PAG 提升 'Diffusers' 库**：得益于社区贡献，**Perturbed Attention Guidance (PAG)** 已集成到 **`diffusers` 库**中，有望增强图像生成效果，正如 [HuggingFace 的核心公告](https://github.com/huggingface/diffusers/issues/8704)中所宣布的那样。

- **攻克特定语言的知识蒸馏**：关于知识蒸馏的咨询非常突出，一名成员提议为单一语言建立蒸馏多语言模型，另一名成员则推荐使用 SpeechBrain 来处理该任务。

- **聚焦 LLMs 和数据集质量**：除了 Microsoft 的 **Phi-3-Mini-128K-Instruct** 模型等进展外，社区还强调了数据集质量的重要性。同时，通过[此处](https://arxiv.org/abs/2404.18824)和[此处](https://arxiv.org/abs/2405.00332)引用的论文，探讨了与 LLMs 数据泄露相关的担忧。

- **对 AI 驱动工具的呼声**：从对无缝 **AI API development** 平台的需求（通过[反馈调查](https://forms.gle/yAfGjUtNTnf5mASK7)引用），到识别手写表格中数据的挑战，显然存在对能够简化任务并提高工作流程效率的 AI 驱动解决方案的需求。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 伦理成为焦点**：出现了关于 AI 训练伦理的对话，一名成员对积极鼓励模型 **poisoning** 表示担忧。另一名成员辩称 AIW+ 问题的解决方案是不正确的，提到它忽略了某些家庭关系，从而暗示了模糊性和伦理考量。

- **AI 音乐生成达到高潮**：讨论涉及使用 **RateYourMusic ID** 生成歌曲和歌词，一位个人确认了其成功并称结果“非常滑稽”。

- **关于 NSFW 内容的大辩论**：关于是否应将 NSFW 内容包含在数据集中的辩论激增，突显了道德担忧与反对过度谨慎的模型安全措施之间的对立。

- **GPU 对决与实用性**：成员们交流了对 **A6000s, 3090s** 和 **P40 GPUs** 权衡的见解，指出了在应用于 AI 训练时，VRAM、散热要求和模型效率方面的差异。

- **ASIC 芯片进入 Transformer 领域**：一个新兴话题是 **Etched's Sohu**，这是一种专门用于 Transformer 模型的芯片。其宣传的优势引发了关于其实用性和对各种 AI 模型适应性的讨论，与对其潜在缺乏灵活性的怀疑形成对比。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ICML 2024 备受关注的论文**：EleutherAI 的研究人员正为 ICML 2024 做准备，提交了[关于 classifier-free guidance 和开放基础模型影响的论文](https://arxiv.org/abs/2306.17806)。另一项研究深入探讨了[语言模型中的记忆现象 (memorization)](https://arxiv.org/abs/2406.17746)，检查了隐私和泛化等问题。

- **多模态奇迹与聚会盛况**：Huggingface 的排行榜已成为寻找顶尖多模态模型的便捷工具；与此同时，[ICML 维也纳见面会](https://icml.cc/Conferences/2024)吸引了一系列热情的计划。混合模型 Goldfinch 也参与了交流，它通过将 Llama 与 Finch B2 层合并以提升性能。

- **引发同行讨论的论文**：[#research](https://discord.com/channels/729741769192767510/747850033994662000/1255280862199550054) 频道的讨论围绕着从 Synquid 的对比评估到 Hopfield Networks 在 Transformer 中的应用等论文展开。成员们剖析了从多模态学习效率到泛化和 grokking 的实验方法等主题。

- **Hopfield 的回归**：成员们通过将神经网络中的 self-attention 纳入（异）关联记忆框架，提供了相关见解，并辅以连续现代 Hopfield Networks 及其作为单步 attention 实现的参考文献。

- **稀疏且智能**：Sparse Autoencoders (SAEs) 因其从过完备基中挖掘线性特征的能力而备受关注，正如 [LessWrong 文章](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)所宣传的那样。此外，值得一提的是一篇[关于多语言 LLM 安全性的论文](https://arxiv.org/abs/2406.16235)，展示了通过定向中毒优化 (DPO) 实现的跨语言去毒。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

### **AMD Radeon MI300X 对标 Nvidia**：
尽管 AMD 的软件生态系统 ROCm 仍落后于 Nvidia 的 CUDA，但新款 **AMD Radeon Instinct MI300X** 的定位是挑战 Nvidia 在 GPU 计算市场的统治地位，详见 [Chips and Cheese](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/) 的文章。

### **ASIC 芯片雄心**：
Etched 宣布的 **Transformer ASIC 芯片**旨在比 GPU 更高效地运行 AI 模型，并获得了包括 Bryan Johnson 支持的 1.2 亿美元 A 轮[融资](https://x.com/bryan_johnson/status/1805629207374086490)，引发了关于专用 AI 芯片未来角色的讨论。

### **优化调整与 Triton 查询**：
工程讨论围绕一个提议的 **Adam-mini 优化器**展开，该优化器可减少 45-50% 的内存占用，代码已在 [GitHub](https://github.com/zyushun/Adam-mini) 上发布。此外，社区正在寻求帮助，以便在 `python.triton.language.core` 中添加 `pow` 函数，如该 [Triton issue](https://github.com/triton-lang/triton/issues/4190) 所示。

### **PyTorch 发布纪录片庆祝**：
“PyTorch Documentary Virtual Premiere: Live Stream”的首映引起了关注，展示了 PyTorch 的演变及其社区。用户们反响热烈，并用 *山羊 (goat) 表情符号* 来表达兴奋之情，可在[此处](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s)观看。

### **Intel 寻求在 PyTorch 中集成 GPU 支持**：
Intel PyTorch 团队在 [GitHub 上发布了 RFC](https://github.com/pytorch/pytorch/issues/114723)，继续推进原生 PyTorch 对 **Intel GPU (XPU)** 的支持，标志着 Intel 致力于成为深度学习硬件领域的积极参与者。

### **AI 基础设施与实践讨论**：
社区对话涉及学习率缩放、参考 [AdamW 论文](https://mlfoundations.github.io/advancedml-sp23/assets/adam.pdf) 的更新裁剪 (update clipping) 见解、AMD 与 Nvidia 构建方案之间的基础设施选择，以及对 Sohu ASIC 芯片承诺的关注，这些都影响着大型 Transformer 模型的效能。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity API 带来的困扰**：工程师们讨论了 **Perplexity AI API** 间歇性出现的 **5xx 错误**，强调了通过状态页提高透明度的必要性。此外，还就 API 过滤器和未公开功能进行了辩论，一些用户探究了搜索域名过滤器和引用日期过滤器的存在。

**寻找更好的搜索**：**Perplexity Pro focus search** 因其局限性受到批评，而在与 **ChatGPT** 的对比中，用户注意到了 Perplexity 新的 Agent 式搜索能力，但也批评其在摘要中容易产生幻觉（hallucinate）。

**Claude 利用上下文优势**：社区热议 **Claude 3.5** 为 **Perplexity Pro** 用户提供的 32k token 上下文窗口，并确认了 Android 端支持。用户明显更倾向于 Claude Pro 提供的完整 200k token 窗口。

**与 Denis Yarats 洞察创新**：Perplexity AI 的 CTO 在一段 [YouTube 视频](https://www.youtube.com/watch?v=gvP-DxqatLQ)中剖析了 AI 的创新，讨论了它如何彻底改变搜索质量。在相关的对话中，研究人员展示了一种新方法，可能通过从语言模型计算中移除矩阵乘法（matrix multiplication）来改变游戏规则。

**分享空间的近期热点**：社区分享了大量的 Perplexity AI 搜索结果和页面，包括土卫六（Titan）缺失波浪的证据、中国的探月工程，以及一项关于重力如何影响感知的研究，鼓励他人在平台上探索这些精选搜索。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI World's Fair 观影会启动**：举办 **AI Engineer World’s Fair** 观影会的热情高涨，该活动在[此处](https://www.youtube.com/watch?v=5zE2sMka620)直播，重点展示前沿的主旨演讲和代码专题。

- **PyTorch 粉丝的预映之夜**：[PyTorch 纪录片虚拟首映式](https://www.youtube.com/live/EjgTv6aSeqk)备受期待，该片通过创始人及核心贡献者的评论，回顾了该项目的演变和影响。

- **ChatGPT 语音更新推迟**：由于语音功能的技术困难，ChatGPT 语音模式（Voice Mode）推迟发布，在 [Teknium 的推文](https://x.com/Teknium1/status/1805718678526476655/photo/1)发布后引起了轰动。

- **Bee Computer 展现智能活力**：AI 工程师活动的参与者对来自 **Bee Computer 的新型 AI 可穿戴技术**议论纷纷，该技术以其对个人数据的深度理解和主动任务列表而备受推崇。

- **神经视觉效果超出预期**：神经科学的一项突破引起了社区关注，即[从老鼠皮层活动中重建视觉体验](https://x.com/Neuro_Joel/status/1805221959191437356)，展示了神经成像技术的惊人进步。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 的技术故障与技巧**：工程师报告了 LM Studio (0.2.25) 的错误，包括加载模型时的 *Exit code: -1073740791*。对于 **Hermes 2 Theta Llama-3 70B**，使用 RTX 3060ti 的用户面临“显存不足”（Out of Memory）问题，并考虑使用 NousResearch 的 8b 等替代方案。在 Apple M 芯片上运行 **Llama 3 70B** 时，由于不同的量化类型（quant types）和设置，也出现了一些问题。

- **RAG 成为焦点**：进行了一场关于检索增强生成（RAG）的详细讨论，重点介绍了 NVIDIA 关于 RAG 利用外部数据增强信息生成准确性能力的[博客文章](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)。

- **诈骗警告与安全提示**：用户注意到指向冒充 Steam 的俄罗斯网站的诈骗链接，并已报告给管理员采取行动。社区对网络钓鱼攻击以及保护个人和项目数据的重要性保持警惕。

- **硬件对话升温**：提到了一个使用 **8x P40 GPU** 完成的配置，引发了关于服务器电源管理（涉及 **200 安培电路**）以及 LM Studio 在多 GPU 设置下显存（VRAM）报告准确性的进一步讨论。家庭服务器设置产生的噪音也被幽默地比作喷气发动机。

- **创新想法与 SDK 展示**：成员们分享了各种想法，从在科幻角色扮演游戏中使用 LLM 作为游戏主持人（game master），到解决 token 预测中大上下文窗口导致的性能不佳问题。这里有一份使用 SDK 构建 Discord 机器人的指南 [dev.to](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)，以及关于使用 Python 从 LM Studio 服务器提取数据的问题。

- **Open Interpreter 中的上传障碍**：用户对于无法直接将文档或图像上传到 Open Interpreter 终端感到沮丧，这限制了用户与 AI 模型交互及应用场景的扩展。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **使用 Mojo 数据类型绘制路径**：工程师们正在尝试使用 **Mojo 数据类型**进行直接绘图，而无需转换为 Numpy，并利用 [Mojo-SDL](https://github.com/msteele/mojo-sdl) 等库进行 SDL2 绑定。社区正在讨论 Mojo 图表库所需的功能，重点领域涵盖从高级接口到交互式图表，以及与 Arrow 等数据格式的集成。

- **用于多功能可视化的 Vega IR**：数据可视化对交互性的需求得到了强调，**Vega 规范**被提议作为中间表示（IR），以桥接 Web 和原生渲染。对话涉及了 **UW 的 Mosaic** 等库的独特方法，以及 D3、Altair 和 Plotly 等主流库。

- **WSL 作为 Windows 进入 Mojo 的门户**：已确认 Mojo 可通过 Windows Subsystem for Linux (WSL) 在 Windows 上运行，预计年底前将提供原生支持。Visual Studio Code 与 Linux 目录配合使用的易用性是一个亮点。

- **IQ 与智能之争升温**：社区就智能的本质展开了激烈辩论，**ARC 测试**因其以人为中心的模式识别任务而受到质疑。一些用户认为 AI 在 IQ 测试中表现出色并不代表真正的智能，而意识与记忆（recall）的概念引发了进一步的哲学讨论。

- **编译时特性与 Nightly 版本**：Mojo 编译器被曝出多个问题，从类型检查和布尔表达式处理中的 Bug，到编译时对 `List` 和 `Tensor` 的处理。各讨论串都鼓励用户报告问题，即使这些问题在 Nightly 版本中已得到解决。此外，还讨论了特定的 Commit、Nightly 版本更新以及引用不可变静态生命周期变量的建议，凝聚社区进行协作调试和改进。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LLM 排行榜的夸耀权受到质疑**：Clement Delangue 宣布推出[新的开源 LLM 排行榜](https://x.com/ClementDelangue/status/1805989925080219927)，并吹嘘使用 300 块 H100 GPU 重新运行 MMLU-pro 评估，这引发了关于此类算力必要性以及大模型有效性的讽刺与批评。
  
- **RabbitCode 的 API 安全出问题**：Rabbitude 发现其存在**硬编码 API 密钥**，包括 [ElevenLabs](https://elevenlabs.io) 等服务的密钥，导致 Azure 和 Google Maps 等服务面临风险，引发了对未经授权数据访问的担忧以及对滥用 ElevenLabs 额度的猜测。

- **ChatGPT 高级语音模式延迟**：[OpenAI](https://openai.com) 已将面向 Plus 订阅者的 ChatGPT 高级语音模式发布推迟到秋季，旨在增强内容检测和用户体验，该消息通过 [OpenAI 的 Twitter](https://x.com/openai/status/1805716393524183136?s=46) 发布。

- **关于 Imbue 突然成功的传闻**：Imbue 突然获得的 2 亿美元融资引起了成员的怀疑，大家探讨了该公司不明确的历史，并将其发展轨迹与 **Scale AI** 及其子公司在数据标注和远程 AI 项目博士招聘方面的策略进行了比较。

- **音乐行业的 AI 转型**：Udio 关于 AI 变革音乐行业潜力的[声明](https://x.com/udiomusic/status/1805694761891778783?s=46)与 [RIAA 的担忧](https://x.com/riaa/status/1805739691972452559?s=46)发生冲突，Udio 断言尽管行业存在阻力，AI 仍将成为音乐创作的必需品。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **挑战 Stability AI 进一步提升**：讨论指出，社区对 Stability AI 在 **Stable Diffusion 3 (SD3)** 上的做法日益担忧，强调需要无审查模型（uncensored models）和更新的许可证，以保持长期竞争力。社区要求提供除猎奇创作之外，更具实际应用价值的现实场景应用。

- **GPU 成本效益策略讨论**：GPU 租赁成本对比显示，与 **Runpod** 相比，使用 **Vast** 运行 3090 是更经济的选择，据称价格低至每小时 30 美分。

- **辩论：社区驱动 vs. 企业支持**：关于开源倡议与企业影响力之间平衡的辩论十分活跃，一些成员认为社区支持至关重要，而另一些成员则引用 Linux 在企业支持下取得的成功，认为这是一条可行的路径。

- **优化机器学习构建**：成员们正在分享针对高效 **Stable Diffusion** 配置的硬件建议，大家一致认为 Nvidia 4090 具有性能优势，并且为了节省成本，双 4090 可能比高 VRAM 的单 GPU 更具优势。

- **对 ICQ 的怀旧与 SDXL 的障碍**：老牌即时通讯服务 **ICQ** 的关闭引发了怀旧交流；同时，社区也报告了运行 **SDXL** 时面临的挑战，特别是由于 VRAM 不足导致 **"cuda out of memory"** 错误的成员，正在寻求命令行解决方案的建议。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **推出 Prompt Engineering Toolkit**：分享了一个开源的 [Prompt Engineering Toolkit](https://github.com/teknium1/Prompt-Engineering-Toolkit)，用于配合 **Sonnet 3.5** 使用，旨在帮助为 AI 应用创建更好的提示词。
  
- **模型性能引发质疑**：微软在 [Genstruct](https://huggingface.co/spaces/davanstrien/Genstruct-7B) 上展示的新原始文本数据增强模型引发了对其有效性的怀疑，展示结果似乎偏离了主题。

- **AI 芯片性能引发热议**：新型 "Sohu" AI 芯片引发了关于其高性能推理任务潜力的讨论，并链接到了 [Gergely Orosz 的帖子](https://x.com/GergelyOrosz/status/1805604272614088721)，该帖子暗示尽管硬件在进步，但 OpenAI 并不认为 AGI 即将到来。

- **Imbue AI 发布 70B 模型工具包**：Imbue AI 发布了一个 **70B 模型**工具包，资源包括 **11 个 NLP 基准测试**、一个**专注于代码的推理基准测试**以及一个**超参数优化器**，详见 [Imbue 的介绍页面](https://imbue.com/research/70b-intro/)。

- **拥抱搞怪的 AI**：一位用户发布了由 Anthropic 的 **Claude** 生成的梗图（meme）形式的内容，反映了 Claude 对复杂话题的解释，以及它对未经历过天气或存在危机的幽默看法。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **简化 AI 对话流**：工程师们强调了来自 `langchain_community.chat_models` 的 `.stream()` 方法，用于迭代 LangChain 响应；其他人讨论了集成 [Zep](https://www.getzep.com/) 以实现 AI 的长期记忆，并考虑在 LangChain 中直接使用 `BytesIO` 处理 PDF 而无需临时文件。

- **LangChain 中的可视化探索**：关于在 Streamlit 中实时可视化 Agent 思考过程的讨论涉及了使用 `StreamlitCallback`，但也指出了在不使用回调的情况下管理流式响应的空白。

- **排除不可见的故障**：有关于 LangSmith 在环境设置正确的情况下仍无法追踪执行的咨询，建议是检查追踪配额。

- **扩展容器化测试**：一位社区成员为 **testcontainers-python** 贡献了 Ollama 支持，方便了 LLM 端点测试，详见其 [GitHub issue](https://github.com/testcontainers/testcontainers-python/issues/617) 和 [pull request](https://github.com/testcontainers/testcontainers-python/pull/618)。

- **认知手艺与出版物**：分享了一篇关于在 LangChain 中结合工具调用进行少样本提示（few-shot prompting）的 [Medium 文章](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b38fe1)，以及一段探索 ARC AGI 挑战的 YouTube 视频，题为“[Claude 3.5 也挣扎？！百万美元挑战](https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap)”。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **寻求上下文清晰度的聊天机器人**：一位工程师询问了如何有效地直接从 [LlamaIndex 聊天机器人框架](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/)的聊天响应中检索上下文，并分享了实现细节及遇到的挑战。

- **即将进行的 Pull Request 审查**：一名成员分享了一个 GitHub PR 待审查，旨在为 **LlamaIndex 中的 Neo4J 数据库**添加查询过滤功能；另一名成员确认需要处理积压的任务。

- **消除多余通知**：讨论了如何抑制 Openailike 类中关于缺失机器学习库的不必要通知，并澄清此类消息并非错误。

- **使用 LLM 优化 SQL 查询**：用户间的对话强调了在使用 RAG SQL 层时，微调语言模型对于提高 SQL 查询精确度的好处，并建议使用高质量训练数据以获得更好性能。

- **平衡混合搜索**：解答了关于 **LlamaIndex** 中混合搜索实现的问题，重点在于调整 `alpha` 参数以平衡搜索结果中元数据和文本的相关性。

- **使用 LlamaIndex 增强 RAG**：分享了一篇文章，重点介绍了使用 LlamaIndex 和 DSPy 构建优化后的检索增强生成（RAG）系统的方法，为 AI 工程师提供了见解和实践步骤。

- **开源贡献奖励**：发起了一项针对开源项目 [Emerging-AI/ENOVA](https://github.com/Emerging-AI/ENOVA) 的反馈征集，该项目旨在增强 AI 部署、监控和自动扩展，参与者有机会获得 50 美元的礼品卡。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Claude-3.5-Sonnet 成为焦点**：最新的 **Anthropic 模型**正式命名为 `claude-3-5-sonnet-20240620`，结束了成员们对名称的困惑。

- **确认 MoonDream 的视觉限制**：虽然人们对 **OpenInterpreter (OI)** 适配基于 **MoonDream** 的视觉模型很感兴趣，但目前的对话确认其与 OI 不兼容。

- **多行输入异常与视觉命令错误**：在使用 `-ml` 进行多行输入以及执行 `interpreter --os --vision` 命令时出现了技术问题。一名用户验证了其 **API key** 但仍面临错误，另一名成员报告称因尝试直接将文件拖入终端而被**封禁**。

- **01：OI 的语音界面，并非到处有售**：**01** 作为 **OI** 的语音界面，无法在西班牙购买；爱好者们被引导至 GitHub 上的[开源开发套件](https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight)以寻求 DIY 替代方案。

- **构建你自己的 01**：从开源套件 DIY 组装 01 的教程将会增多，其中一个计划在 7 月发布，这暗示了社区致力于在商业销售限制之外确保更广泛的获取途径。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**对 Cohere 学者计划的好奇**：一名成员询问了今年**学者计划**的状态，但随后没有关于此话题的进一步信息或讨论。

**计费 Preamble Token 受到关注**：一位用户分享了一个涉及 API 调用 **preamble token** 的实验，提出了一个可能通过利用不计费的 preamble 使用来规避费用、降低成本的漏洞。

**使用 Rust 为 LLM 进行设计**：宣布发布 **Rig**，这是一个用于创建 LLM 驱动应用程序的 Rust 库，并邀请开发者参与有奖反馈计划以探索和审查该库。

**AI 使用中的伦理考量浮出水面**：针对 **SpicyChat AI**（一个提供 NSFW 机器人托管服务的平台）提出了担忧，认为其通过盈利性使用可能违反了 Cohere 的 **CC-BY-NA** 许可，并声称通过 **OpenRouter** 规避了这一限制。

**Hongyu Wang 关于 1Bit LLM 的学习活动**：宣布了一场由 **Hongyu Wang** 主持的题为《1Bit LLM 时代》的在线演讲，并邀请通过提供的 [Google Meet 链接](https://meet.google.com/yhv-tiir-ava)参加。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Adam 优化器瘦身**：工程师们讨论了一篇介绍 **Adam-mini** 的 [arXiv 论文](https://arxiv.org/abs/2406.16793)，强调其内存占用比 AdamW 减少了 45% 到 50%。它通过使用更少的学习率，并利用受 Transformer 的 Hessian 结构启发的参数块学习来实现这一目标。

- **训练陷阱与 CUDA 难题**：一位工程师寻求关于在训练期间实现输出文本掩码（类似于 `train_on_input`）的建议，而另一位工程师提出了 **CUDA** 错误问题，建议启用 `CUDA_LAUNCH_BLOCKING=1` 以识别模型训练期间的非法内存访问。

- **梯度累积——是友是敌？**：增加梯度累积的影响引发了热烈讨论；一些人认为这可以通过减少优化器的运行次数来缩短训练时间，而另一些人则担心这可能导致步长变慢并增加总训练时间。

- **余弦调度与 QDora 探索**：关于在 **Hugging Face 平台**上创建具有非零最小值的余弦学习率调度器的问题被提出，同时社区对在 **PEFT** 中启用 **QDora** 的拉取请求（Pull Request）表现出明显的兴奋。

- **叙事引擎与 Mistral 谜团**：展示了 [Storiagl](https://storiagl.web.app/)，一个使用自定义 LLM 构建故事的平台；而另一位工程师报告了 **Mistral7B** 的重复文本生成问题，尽管设置了高 Temperature 仍在寻求解决方案。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Prompting 在语言学习中更胜一筹**：包括 Eline Visser 在内的研究人员表明，在使用单本语法书学习 **Kalamang 语言**时，**对大语言模型 (LLM) 进行 Prompting** 的表现优于微调。这一“Prompting 胜出”的发现详见 [Jack Morris 的推文](https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ)，并在[学术论文](https://arxiv.org/abs/2309.16575)中进行了进一步阐述。

**在线观看 AI 工程师世界博览会**：**AI Engineer World's Fair 2024** 正在直播，重点关注主题演讲和 CodeGen 赛道，可通过 [YouTube](https://www.youtube.com/watch?v=5zE2sMka620) 观看；更多详情请见 [Twitter](https://twitter.com/aidotengineer)。

**Claude 竞赛征集创意**：2024 年 6 月的 **Build with Claude** 竞赛已宣布，邀请工程师展示他们在 Claude 方面的专业知识，详见[官方指南](https://docs.anthropic.com/en/build-with-claude-contest/overview)。

**额度问题协助**：有人针对额度表单问题提供协助，要求直接私信相关的电子邮件地址，以便*高效解决问题*。

**模型卸载（Offloading）技术辩论**：社区观察到，与 **FairScale 的 Fully Sharded Data Parallel (FSDP)** 相比，**DeepSpeed (DS)** 似乎拥有更有效的细粒度卸载策略。此外，寻求优化设置的成员正在考虑这些卸载策略在 **LLama 70B** 上的实用性。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla Builders 计划倒计时**：提醒成员在 **7 月 8 日**早期申请截止日期前提交 **Mozilla Builders Program** 的申请。如需支持和更多信息，请查看 [Mozilla Builders Program 页面](https://future.mozilla.org/builders/)。

- **通过 Firefox 和 llamafile 体验 90 年代怀旧风**：**Firefox** 集成了 llamafile 作为 HTTP 代理，允许用户在复古的 Web 体验中探索 LLM 权重；演示视频可在 [YouTube](https://youtu.be/YWQ5Kh9gNuo) 上观看。

- **创建你自己的聊天宇宙**：用户可以通过[此处](https://t.ly/y6jrZ)访问的共享 Notebook，将 llamafile 与 Haystack 和 Character Codex 融合，创建沉浸式聊天场景。

- **清理 Notebook 中的 CUDA 杂讯**：为了保持 Jupyter Notebook 的整洁，建议使用 [Haystack 的工具函数](https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py)来处理 CUDA 警告。

- **NVIDIA 股价坐上过山车**：在 AIEWF 的演讲之后，NVIDIA 的市值大幅下跌，引发了 [MarketWatch](https://www.marketwatch.com/story/nvidias-stock-is-set-to-gain-with-rivals-seen-to-be-in-perpetual-catch-up-mode-0552e514) 和 [Barrons](https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c) 等媒体对该公司财务表现催化剂的各种分析。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 探索 FPGA 加速**：有传闻称 **tinygrad** 正在利用 FPGA 作为后端，George Hotz 暗示可能会实现一种 **加速器设计（accelerator design）**。
- **Groq 校友推出面向高效 AI 的 Positron**：前 Groq 工程师推出了 [Positron](https://www.positron.ai/)，目标直指 AI 硬件市场。其产品如 Atlas Transformer Inference Server，声称比 DGX-H100 等竞争对手在单位成本下拥有 **10 倍的性能提升**。
- **FPGA 在定制化 AI 与 HDL 中的角色**：讨论集中在配备 DSP 模块和 HBM 的 FPGA 未来发展，这可能允许创建特定于模型的 HDL。不过也有人指出，Positron 的方法是通用的，并不绑定于特定的 FPGA 品牌。
- **纪录片致敬 PyTorch 对 AI 的影响**：社区分享了一部 [YouTube 纪录片](https://www.youtube.com/watch?v=rgP_LBtaUEc)，重点介绍了 PyTorch 的开发历程及其对 AI 研究和工具链的影响。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Angry.penguin 晋升版主**：用户 **angry.penguin** 被提升为版主（Moderator）以解决频道的垃圾信息（spam）问题。他以积极主动的态度自愿参与，并立即清理了现有的垃圾信息。Yoko Li 授权 angry.penguin 负责这些新职责和垃圾信息控制措施。
- **告别垃圾信息**：新任版主 **angry.penguin** 宣布成功实施了反垃圾信息措施，确保频道的讨论环境现在已针对破坏性的垃圾信息攻击加强了防御。成员们会发现后续的讨论环境更加整洁、专注。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **德语 Encoder 在 Hugging Face 上线**：AI 工程师可能会对新发布的 **German Semantic V3 和 V3b** 编码器感兴趣，可在 [Hugging Face](https://huggingface.co/aari1995/German_Semantic_V3) 上获取。V3 针对知识库应用，而 V3b 则强调高性能，具有 Matryoshka Embeddings 和 8k token 上下文能力等创新特性。
- **无 GGUF 格式下德语 Encoder 的微调步骤**：尽管有人询问，但 **German V3b encoder** 目前还没有 **GGUF** 格式；不过，对于有兴趣进行微调的用户，建议使用 UKPLab 的 sentence-transformers [微调脚本](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training)。
- **Encoder 适配 GGUF 的可能性**：在一番困惑之后，一名成员通过与 **Ollama** 对比进行了澄清，确认像 German V3 这样的编码器确实可以适配 **GGUF** 格式，这可能涉及使用双嵌入器（dual embedders）来增强性能。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **AI 领域新玩家加入**：OpenRouter 引入了 [01-ai/yi-large 模型](https://openrouter.ai/models/01-ai/yi-large)，这是一个专门用于知识搜索、数据分类、拟人化聊天机器人和客户服务的全新语言模型；该模型支持多语言能力。
- **参数显示故障已修复**：OpenRouter 模型页面上的“推荐参数”选项卡之前存在数据显示问题，目前已 **修复**，确保工程师现在能看到准确的配置选项。
- **AI 助力学术**：新推出的 [GPA Saver](https://gpasaver.com/) 利用 AI 提供学术辅助，包括聊天助手、快速测验解答器等工具；早期采用者使用代码 **BETA** 可获得折扣。
- **简化集成体验**：用户对 OpenRouter 简化 **AI 模型集成** 流程表示感谢，这对于 GPA Saver 平台的创建起到了关键作用。

---

**LLM Perf Enthusiasts AI Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**Datasette - LLM (@SimonW) Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**Torchtune Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 无新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255250846426595428)** (2 条消息): 

- **ChatGPT macOS 桌面端应用发布**：ChatGPT macOS 桌面端应用现已向所有用户开放。通过 [Option + Space 快捷键](https://openai.com/chatgpt/mac/)可以更快速地访问 ChatGPT，实现关于电子邮件、屏幕截图等内容的无缝对话。

- **高级语音模式（Advanced Voice Mode）延迟但即将到来**：原计划于 6 月底推出的高级语音模式为了确保质量已推迟一个月发布。该模式能够理解情感和非语言暗示，将首先在一小部分用户中进行 alpha 测试，随后在秋季推广至所有 Plus 用户，视频和屏幕共享功能的更新也将紧随其后。
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255251858323410985)** (388 条消息🔥🔥): 

- **GPTs Agents 与 OpenAI 的进展**：成员们对 GPTs Agents 在初始训练后无法学习新信息表示沮丧。他们指出，虽然 OpenAI 的模型在初始训练方面表现出色，但持续的改进受到了过度监管的阻碍。

- **Anthropic 的 Claude 受欢迎程度上升**：讨论强调了 **Anthropic 的 Claude 3.5 Sonnet** 如何获得青睐，有观点认为与 OpenAI 的模型相比，它在编程表现和更大的上下文窗口方面更具优势。一位用户推测其架构效率极高，可能采用了 **MoE 架构**。

- **模型性能对比**：用户讨论了 **Anthropic 的 Claude** 相比 **OpenAI 的 ChatGPT** 的优势，特别是在 token 上下文窗口和拒绝率方面。虽然有人认为 **Claude** 的审查更严格，但其他人注意到了 **Claude 的技术改进**，例如更大的 token 窗口和更快的响应速度。

- **开源与自定义模型**：用户对针对特定应用（如**日语语句解释**）定制的 GPTs 和合成数据集表现出兴趣。用户分享了 **Hugging Face** 数据集和 **LM Studio** 等本地推理工具，以便进行进一步的定制。

- **对 OpenAI 的批评与未来展望**：成员们对 **OpenAI 语音功能** 的延迟推出以及 **ChatGPT Plus 订阅** 有限的福利表示担忧。他们希望在上下文窗口和其他功能上有所突破，以追赶 **Google 的 Gemini** 和 **Anthropic 的 Claude** 等竞争对手。
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255251153948639294)** (21 条消息🔥): 

- **Windows 桌面端应用需求超过 Mac**：*"Windows 桌面端应用的使用率难道不会远高于 Mac 桌面端吗？"* 这一话题引发了讨论，另一位用户表示 *"是的，先给 Mac 发布简直是胡扯。"*
- **LaTeX 格式化与 GPT-4o 性能担忧**：一位成员解释说，指定 LaTeX 格式可以获得最佳结果。他们还指出了 GPT-4o 的性能问题，理由是它在逻辑和历史研究任务中表现失败。
- **Mac 桌面端应用仅限 Apple Silicon**：关于新 macOS 桌面端应用的讨论明确了它仅适用于 Apple Silicon（M1 或更高版本），目前没有支持 Intel Mac 的计划。
- **TTS 模型新语音咨询**：一位用户询问新语音是否会通过 TTS 模型提供，但未得到直接回复。
- **GPT-4o 响应缓慢令用户沮丧**：成员们询问并抱怨 GPT-4o 的速度缓慢，怀疑是否存在潜在问题。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255516869595627561)** (1 条消息): 

- **上下文对 AI 错误至关重要**：一位成员指出，理解 AI 的错误很大程度上取决于主题、**知识内容**和上下文。他们建议审查 AI 具体在犯什么错误会很有帮助。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255516869595627561)** (1 条消息): 

- **依赖于 AI 对上下文和知识的理解**：AI 回答的有效性很大程度上取决于主题和知识内容的上下文。一位成员指出：*"看到 AI 哪里弄错了会很有帮助。"*
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1255261278994305024)** (1 条消息):

- **Argilla 2.0 增强 AI 数据集创建**：[Argilla 2.0](https://x.com/argilla_io/status/1805250218184560772) 的发布引入了一个统一的反馈收集框架、全新的 Python SDK、灵活的数据标注 UI 以及更新的文档。这些功能旨在帮助 AI 构建者更高效地创建高质量数据集。
- **Microsoft 的 Florence 模型表现出色**：Microsoft 推出了 [Florence](https://x.com/osanseviero/status/1803324863492350208)，这是一款能够执行字幕生成（captioning）和 OCR 等多种任务的视觉模型。该模型采用 MIT 许可，尽管体积比大型模型小得多，但仍能提供高质量的表现。
- **Microsoft 的指令预训练（Instruction Pre-Training）**：Microsoft 的 [Instruction Pre-Training](https://x.com/osanseviero/status/1804136001465442530) 可以通过指令-响应对增强 LLM 预训练，使 Llama 3 8B 模型的性能可与 70B 模型相媲美。该方法已在 [Gradio space](https://huggingface.co/spaces/davanstrien/instruction-synthesizer) 中进行演示。
- **Marlin TGI 功能提升 GPTQ 模型性能**：下一次 Hugging Face TGI 更新将包含 [Marlin 功能](https://x.com/danieldekok/status/1804224598721830954)，支持针对 GPTQ 量化模型的快速 Marlin 矩阵乘法。这是在 Neural Magic 的 Marlin 内核帮助下实现的。
- **Ethics and Society 通讯强调数据质量**：[Ethics and Society 通讯](https://huggingface.co/blog/ethics-soc-6) 强调了数据质量的重要性。它展示了来自不同成员的协作成果，并提供了关于维持高质量数据标准的见解。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/argilla_io/status/1805250218184560772)">来自 Argilla (@argilla_io) 的推文</a>：📢 另一个重大公告：Argilla 2.0 rc 发布！这对 AI 构建者意味着什么？ 🤺 统一的反馈收集框架 🐍 用于处理数据集的新 Python SDK，包括一个新的 @huggingface da...</li><li><a href="https://x.com/osanseviero/status/1803324863492350208)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Microsoft 刚刚悄悄发布了 Florence 👀 视觉模型，可以处理许多视觉任务（字幕生成、检测、区域建议、OCR） 🤏 小型模型（200M 和 800M），质量堪比比其大 100 倍的模型...</li><li><a href="https://x.com/mervenoyann/status/1805265940134654424)">来自 merve (@mervenoyann) 的推文</a>：在任何任务上微调 Florence-2 🔥 今天我们发布了一个关于在 DocVQA 数据集上微调 Florence-2 的 notebook 和演示博客 @andi_marafioti @skalskip92 请继续阅读 ⇓</li><li><a href="https://x.com/reach_vb/status/1804615756568748537)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：在不到 120 秒内生成 GGUF 量化！⚡ &gt; 增加了对 imatrix 量化的支持 &gt; 支持大尺寸量化的 GGUF-split &gt; 自动上传到 hub &gt; 支持私有和组织仓库 U...</li><li><a href="https://x.com/osanseviero/status/1804136001465442530)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Microsoft 刚刚（再次！）悄悄发布了 Instruction Pre-Training！👀 通过生成指令来增强预训练数据集 🦙 一个性能堪比 70B 的 Llama 3 8B！🔥 通用+领域模型（m...</li><li><a href="https://x.com/vanstriendaniel/status/1804078257488495099)">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：Instruction pre-training 是一种增强 LLM 预训练的新方法，它使用来自指令合成器的指令-响应对，而不是原始数据。在 @gradio Space 中探索此方法...</li><li><a href="https://x.com/danieldekok/status/1804224598721830954)">来自 Daniël de Kok (@danieldekok) 的推文</a>：🐬 更多 Marlin 特性将进入下一个 @huggingface TGI 版本：支持在现有的 GPTQ 量化模型中使用快速的 Marlin 矩阵乘法内核。⚡ 这一特性的实现得益于...</li><li><a href="https://x.com/eustachelb/status/1805262952913858919)">来自 Eustache Le Bihan (@eustachelb) 的推文</a>：Distil-Whisper 迈向多语言！！🤗 法语蒸馏版 Whisper 来了！🇫🇷 准确度媲美 large-v3，速度比 tiny 还快。两全其美！🚀 详情请查看下方 ⬇️</li><li><a href="https://x.com/_philschmid/status/1805593591223398832)">来自 Philipp Schmid (@_philschmid) 的推文</a>：Embedding 模型对于成功的 RAG 应用至关重要，但它们通常是在通用知识上训练的！很高兴分享一份关于如何训练和部署开源 Embedding 模型的端到端指南...</li><li><a href="https://x.com/FrG_FM/status/1803703761119871122)">来自 F-G Fernandez (@FrG_FM) 的推文</a>：Xavier 和 @osanseviero 在 @linuxfoundation 的 #AIDev 上展示了 @huggingface 的机器人计划 🤗（包括由 @RemiCadene 领导的 LeRobot）。期待那一天...</li><li><a href="https://x.com/RisingSayak/status/1805521415543697582)">来自 Sayak Paul (@RisingSayak) 的推文</a>：你知道我们有一份专门的指南，介绍不同的 Prompting 机制以提高图像生成质量吗？🧨 带你了解简单的 Prompt Engineering、Prompt Weighting、Prompt Enhancement...</li><li><a href="https://x.com/evijitghosh/status/1805312283628761446)">来自 Avijit Ghosh (@evijitghosh) 的推文</a>：@huggingface 伦理与社会季度简报发布了！很高兴能与 @frimelle 以及伦理团队的常驻成员合作。本季度简报的主题是...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255239401722875904)** (245 条消息🔥🔥): 

- **VSCode 编程助手的困扰**：一位用户遇到了 Codiumate 在编程任务中途崩溃的问题，导致对 VSCode 编程助手感到沮丧。他们表示需要一种可靠的解决方案，能够检查文件并生成修复方案而不会失败。
- **用于测试和开发的 AI API 平台**：一位成员提议构建一个 AI 驱动的平台，以自动化测试和 API 代码生成，并分享了一份 [调查问卷](https://forms.gle/yAfGjUtNTnf5mASK7) 以收集反馈。他们正在寻求全栈开发人员和 prompt engineers 加入该项目。
- **Phi-3-Mini-128K-Instruct 模型亮点**：Phi-3-Mini-128K-Instruct 模型是 Microsoft 推出的一款轻量级且最先进的开源模型，已在 [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 上展示。它支持更长的 token 上下文，并经过先进的后训练过程以增强指令遵循和安全性。
- **Mozilla Builders 竞赛与协作**：成员们讨论了组队参加 Mozilla Builders 竞赛，该竞赛要求创建在本地运行的 AI 项目。为感兴趣的参与者分享了相关资源和指南。
- **优化 Stable Diffusion 推理**：用户讨论了加速 Stable Diffusion 推理的方法，建议包括使用 Accelerate 库和 [stable-fast](https://github.com/chengzeyi/stable-fast) 框架，以实现显著的性能提升。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap">Claude 3.5 也挣扎了？！百万美元挑战</a>：百万美元 ARC AGI 挑战。获取关于如何进行 AI 数据分析项目的免费 HubSpot 报告：https://clickhubspot.com/d30🔗 链接 - 在 twitter 上关注我...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>：未找到描述</li><li><a href="https://app.tweetscout.io/">未找到标题</a>：未找到描述</li><li><a href="https://future.mozilla.org/builders/">Mozilla Builders</a>：未找到描述</li><li><a href="https://tenor.com/view/hardest-choides-thanos-avengers-strongest-wills-gif-15279882">最艰难的选择灭霸 GIF - Hardest Choides Thanos Avengers - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/thom_wolf/status/1805244710258106369">来自 Thomas Wolf (@Thom_Wolf) 的推文</a>：一个 3.25B 参数的量化 Gemini 在即将推出的 Google Chrome 中本地运行，延迟低于 100ms，同时使用的 RAM 不到 2GB，这比我目前许多 Chrome 页面占用的内存还要少...</li><li><a href="https://tenor.com/view/beach-vacation-artem-gif-26266521">海滩度假 GIF - Beach Vacation Artem - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/chengzeyi/stable-fast">GitHub - chengzeyi/stable-fast: 针对 NVIDIA GPUs 上的 HuggingFace Diffusers 的最佳推理性能优化框架。</a>：针对 NVIDIA GPUs 上的 HuggingFace Diffusers 的最佳推理性能优化框架。 - chengzeyi/stable-fast</li><li><a href="https://app.tweetscout.io">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/thanos-memoji-gif-23490017">灭霸 Memoji GIF - Thanos Memoji - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://forms.gle/yAfGjUtNTnf5mASK7">端到端 AI API 平台 + 当前测试与开发的挑战</a>：描述：我们正在构建一个 AI 驱动的平台，帮助开发人员和其他人员测试 API、UI 以及任何使用 AI 自动化的内容。我们不限于测试，你也可以讨论开发...</li><li><a href="https://github.com/ToonCrafter/ToonCrafter">GitHub - ToonCrafter/ToonCrafter: 生成式卡通插值的研究论文</a>：生成式卡通插值的研究论文 - ToonCrafter/ToonCrafter</li><li><a href="https://huggingface.co/spaces/hpcai-tech/open-sora">Open Sora - hpcai-tech 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0">Open Sora Plan V1.0.0 - LanguageBind 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://t2v-turbo.github.io/">T2V-Turbo: 通过混合奖励反馈打破视频一致性模型的质量瓶颈
  </a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1255381029498130544)** (3 条消息): 

- **Kaggle 上的 Naive Bayes 算法**：一位用户分享了一个 [Kaggle 代码笔记本链接](https://www.kaggle.com/code/rauf111/naive-bayes-algorithm)，该笔记本探讨了 Naive Bayes 算法。该链接指向一个用于学习此机器学习算法的资源。

- **InfiniAttention 复现进展**：一位用户正在进行 **InfiniAttention** 论文的 95% 复现工作。他们提到需要修复梯度消失问题，并进行最后一次实验以完成工作。

**提及的链接**：<a href="https://www.kaggle.com/code/rauf111/naive-bayes-algorithm">Naive Bayes Algorithm</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255297047490461717)** (13 条消息🔥): 

- **使用 LlamaIndex 和 DSPy 优化 RAG 系统**：**Medium** 上的一篇文章深入探讨了如何使用 LlamaIndex 和 DSPy 构建优化的检索增强生成 (RAG) 系统。[构建优化的 RAG 系统](https://medium.com/ai-advances/building-optimized-retrieval-augmented-generation-rag-systems-with-llamaindex-and-dspy-cacaf7f7089f)
  
- **AI Canon：精选现代 AI 资源**：a16z 的一篇博客文章分享了一个被称为 “AI Canon” 的精选资源列表，对 AI 初学者和专家都非常有用。它包括基础论文、实践指南和技术资源。[AI Canon](https://a16z.com/ai-canon/)
  
- **手势媒体播放器控制器演示**：一个 YouTube 视频演示展示了一个基于 Python 的手势媒体播放器控制器项目。*“看看我一直在做的这个酷炫项目——一个使用 Python 的手势媒体播放器控制器！”* [手势媒体播放器控制器演示](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM)
  
- **Nature 关注蛋白质设计进展**：一篇 **Nature** 文章讨论了蛋白质设计的进展，指出了传统基于物理方法的挑战，并强调了 AlphaFold2 取得的突破。[蛋白质设计](https://www.nature.com/articles/s41586-024-07601-y)
  
- **在 Langchain 中结合 Tool Calling 使用 Few-Shot Prompting**：一篇文章讨论了在 Langchain 中将 few-shot prompting 与 tool calling 结合使用，以提高 AI 模型的性能。[Few-Shot Prompting](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1)
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/freddyaboulton/gradio-llamma-cpp">Gradio Llamma Cpp - freddyaboulton 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://www.deeplearning.ai/the-batch/coding-agents-are-evolving-from-novelties-to-widely-useful-tools/">Coding Agents 正在从新奇事物演变为广泛有用的工具</a>：在上周末的父亲节，我和女儿坐在一起，帮助她练习解决算术问题……</li><li><a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">Hand Gesture Media Player Controller Demo</a>：大家好！👋 看看我一直在做的这个酷炫项目——一个使用 Python 的手势媒体播放器控制器！🎮🖐️ 所以，我构建了一个基于 Python 的……</li><li><a href="https://github.com/papers-we-love/papers-we-love/blob/main/machine_learning/General-self-similarity--an-overview.pdf">papers-we-love/machine_learning/General-self-similarity--an-overview.pdf</a>：来自计算机科学界的论文，供阅读和讨论。</li><li><a href="https://a16z.com/ai-canon/">AI Canon | Andreessen Horowitz</a>：一份精选资源列表，我们依靠这些资源来更深入地了解现代 AI，包括生成式 AI、LLMs 和 Transformer 模型。</li><li><a href="https://drive.google.com/file/d/1DYL8jvuE49fN3bGVfFydesJdxKMaBBvh/view">Final EMW 2023 - Macro Keynote (06.28.23).pdf</a>：未找到描述</li><li><a href="https://t.me/RYIUNITY/197736">RYI UNITYDEFI 官方频道中的 Onlyone Dennis</a>：📣 大家系好安全带！本周一下午 1 点，参加我们的 X 竞赛并赚取 RYIU！- 0-300 粉丝：每条推文赚取 100 RYIU - 300-600 粉丝：每条推文赚取 200 RYIU...</li><li><a href="https://www.nature.com/articles/s41586-024-07601-y">Computational design of soluble and functional membrane protein analogues - Nature</a>：一种深度学习方法能够实现可溶性且功能性膜蛋白类似物的精确计算设计，扩展了可溶性蛋白质折叠空间并促进了新的……
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255236112973037629)** (66 条消息🔥🔥): 

- **探索 LLM 中的自定义字节编码**：在一次深入的技术讨论中，成员们探讨了为 LLM 使用自定义字节编码，以 UTF-32 预测序列。讨论内容包括浮点精度和鲁棒性的潜在问题，一位成员对其有效性表示怀疑，但对其结果保持好奇。

- **手势媒体播放器控制器演示**：一位成员分享了一个 [YouTube 视频](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM)，演示了使用 Python 开发的基于手势的媒体播放器控制器。

- **生物信息学工具与项目**：成员们分享了各种生物信息学工具和项目，包括用于脂质运动 PCA 及相关分析的工具 [PCALipids](https://github.com/membrane-systems/PCAlipids)，以及其他 GitHub 项目，如 [embedprepro-lib](https://github.com/Elma-dev/embedprepro-lib) 和 [PixUP-Upscale](https://github.com/U-C4N/PixUP-Upscale)。

- **发布新款文本分析 CLI 工具**：一位成员宣布发布了一款名为 [embedprepro](https://github.com/Elma-dev/embedprepro-lib) 的新文本分析命令行工具，旨在为研究人员和开发人员提供文本 Embedding 生成、聚类和可视化功能。

- **用于 RLHF 优化 LLM 的数据集**：一位成员在 Hugging Face 上发布了 Tasksource-DPO-pairs 数据集 [Tasksource](https://huggingface.co/datasets/tasksource)。该数据集专为通过人类反馈奖励学习 (RLHF) 优化 LLM 而定制，侧重于细粒度的语言推理任务。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://storiagl.web.app/">StorIA</a>: 未找到描述</li><li><a href="https://x.com/eggwens/status/1806016129875476886">Egg (@eggwens) 的推文</a>: 这是宠物通灵者的现场演示，附带了使用 React 制作的示例代码和样式：通过 Pet Psychic Scheduler，您可以：🔮 为您的宠物预订通灵阅读 ✨ 检查每日心情 f...</li><li><a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">手势媒体播放器控制器演示</a>: 大家好！👋 看看我一直在做的一个很酷的项目 - 使用 Python 的手势媒体播放器控制器！🎮🖐️ 所以，我构建了一个基于 Python 的...</li><li><a href="https://huggingface.co/spaces/KoboldAI/Koboldcpp-Tiefighter/blob/main/Dockerfile">Dockerfile · KoboldAI/Koboldcpp-Tiefighter 在 main 分支</a>: 未找到描述</li><li><a href="https://github.com/U-C4N/PixUP-Upscale/">GitHub - U-C4N/PixUP-Upscale</a>: 通过在 GitHub 上创建账号来为 U-C4N/PixUP-Upscale 的开发做出贡献。</li><li><a href="https://github.com/azmiord/project">GitHub - azmiord/project</a>: 通过在 GitHub 上创建账号来为 azmiord/project 的开发做出贡献。</li><li><a href="https://github.com/Elma-dev/embedprepro-lib">GitHub - Elma-dev/embedprepro-lib</a>: 通过在 GitHub 上创建账号来为 Elma-dev/embedprepro-lib 的开发做出贡献。</li><li><a href="https://github.com/membrane-systems/PCAlipids">GitHub - membrane-systems/PCAlipids: 用于脂质运动 PCA 及相关分析的脚本</a>: 用于脂质运动 PCA 及相关分析的脚本 - membrane-systems/PCAlipids</li><li><a href="https://github.com/bigsk1/voice-chat-ai">GitHub - bigsk1/voice-chat-ai: 🎙️ 与 AI 对话 - 使用 ollama 本地运行或使用 OpenAI - XTTS 或 OpenAI Speech 或 ElevenLabs</a>: 🎙️ 与 AI 对话 - 使用 ollama 本地运行或使用 OpenAI - XTTS 或 OpenAI Speech 或 ElevenLabs - bigsk1/voice-chat-ai</li><li><a href="https://huggingface.co/datasets/tasksource/tasksource_dpo_pairs">tasksource/tasksource_dpo_pairs · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/sileod/tasksource/blob/main/tasks.md">tasksource/tasks.md 在 main 分支 · sileod/tasksource</a>: 用于 NLP 极端多任务学习的数据集收集和预处理框架 - sileod/tasksource
</li>
</ul>

</div>

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255300766739206214)** (15 条消息🔥): 

- **感谢 Alex 推荐该工作**：一位成员对 Alex 的帖子表示感谢，该帖子引起了大家对一项此前无人推荐的特定工作的关注。 

- **关于 LLM 数据泄露的讨论**：Eleuther AI 提到了讨论 LLM 中基准测试数据集泄露的论文。他们分享了[一篇论文](https://arxiv.org/abs/2404.18824)的链接，该论文调查了这一现象，以及[另一篇论文](https://arxiv.org/abs/2405.00332)，该论文探讨了基准测试数据泄露的检测。

- **引入 Terminator 架构**：从 [Twitter 链接](https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w)分享了一种名为 "Terminator" 的新架构，并在 [GitHub 仓库](https://github.com/hyperevolnet/Terminator/blob/main/models/modules/hyperzzw.py)中提供了进一步的细节。该架构的显著特点是缺少残差连接（residuals）、点积注意力（dot product attention）和归一化（normalization）。

- **LLM 排行榜的饱和问题**：一位成员分享了一个指向 HuggingFace 博客的链接，讨论了 LLM 排行榜饱和的担忧，表明社区对该问题的关注。博客文章链接：[HuggingFace LLM Leaderboard Blog](https://huggingface.co/spaces/open-llm-leaderboard/blog)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/leopolisdream/status/1804627325583327358?s=46&t=BsqYoGA8vIHGcXwORlMk7w">来自 Alex Yanko 🇺🇦 (@LeopolisDream) 的推文</a>：欢迎新架构：Terminator。无残差，无点积注意力，无归一化... https://arxiv.org/pdf/2401.17948</li><li><a href="https://github.com/hyperevolnet/Terminator/blob/main/models/modules/hyperzzw.py">Terminator/models/modules/hyperzzw.py at main · hyperevolnet/Terminator</a>：通过在 GitHub 上创建账号来为 hyperevolnet/Terminator 的开发做出贡献。</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>：在预训练数据不断扩大的使用中，基准测试数据集泄露现象日益突出，而不透明的训练过程和通常未公开的包含情况加剧了这一现象...</li><li><a href="https://arxiv.org/abs/2405.00332">A Careful Examination of Large Language Model Performance on Grade School Arithmetic</a>：大语言模型 (LLMs) 在许多数学推理基准测试中取得了令人印象深刻的成功。然而，人们越来越担心其中一些性能实际上反映了数据集泄露...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1255397885210529902)** (1 条消息): 

- **Perturbed Attention Guidance 现已加入 Diffusers**：HuggingFace 宣布在其 `diffusers` 库中支持 **Perturbed Attention Guidance (PAG)**，在无需额外训练的情况下提升图像生成质量。[查看更新](https://github.com/huggingface/diffusers/issues/8704)，并向领导此次集成的贡献者致敬。

**提到的链接**：<a href="https://github.com/huggingface/diffusers/issues/8704">PAG 现已在核心 🤗 中支持 · Issue #8704 · huggingface/diffusers</a>：大家好！#7944 引入了对 Perturbed Attention Guidance (PAG) 的支持，它可以在无需训练的情况下增强图像生成质量。无 PAG 的生成图像 vs 有 PAG 的生成图像。查看详情...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255246614692368405)** (3 条消息): 

- **文件夹中 detection_util 的评估错误**：有人指出，如果 `detection_util` 位于 Space 的文件夹中，**`evaluate` 函数**在定位它时会出现问题。由于函数找不到所需文件，这会导致评估过程中出现问题。
- **手势媒体播放器控制器演示**：一位用户分享了一个 [YouTube 视频](https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM)，演示了用 Python 制作的“手势媒体播放器控制器”。他们鼓励其他人查看这个酷炫的项目。
- **开发手写表格数据流水线**：有人请求协助创建一个用于识别手写表格数据的流水线（pipeline）。他们提到尝试过 GPT-Vision，但未能达到预期。

**提到的链接**：<a href="https://youtu.be/MD8dZME-fBA?si=-1Zn6GeWSnknDWGM">Hand Gesture Media Player Controller Demo</a>：大家好！👋 看看我一直在做的这个酷炫项目——一个使用 Python 制作的手势媒体播放器控制器！🎮🖐️ 所以，我构建了一个基于 Python 的...

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1255236404712046733)** (5 messages): 

- **寻求多语言模型蒸馏的建议**：寻找针对单一语言进行多语言模型知识蒸馏（knowledge distillation）的建议。

- **使用 RAG 进行命名实体识别**：一位成员就使用检索增强生成（RAG）识别长文档中的命名实体寻求建议。考虑到使用 Mamba 等 SSM 来管理文档长度，另一位成员建议使用 BM25 进行面向关键词的搜索，并提供了一个 [GitHub 链接](https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb)以获取更多信息。

- **开发手写表格的处理流水线**：一位成员想要创建一个识别手写表格数据的流水线（pipeline），并发现 GPT-Vision 的表现未达预期。正在寻求更有效方法的建议。

- **寻求 LLM 知识编辑的经验**：有人提出了关于 LLM 知识编辑（knowledge editing）的实践经验及其在翻译等简单任务中部署的咨询。

**提到的链接**：<a href="https://github.com/aws-samples/semantic-search-with-amazon-opensearch/blob/main/Module%201%20-%20Difference%20between%20BM25%20similarity%20and%20Semantic%20similarity.ipynb">semantic-search-with-amazon-opensearch/Module 1 - Difference between BM25 similarity and Semantic similarity.ipynb at main · aws-samples/semantic-search-with-amazon-opensearch</a>：通过在 GitHub 上创建账号，为 aws-samples/semantic-search-with-amazon-opensearch 的开发做出贡献。

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1255403601770319944)** (3 messages): 

- **探索多语言模型的知识蒸馏**：一位成员询问了如何针对单一语言进行多语言模型的知识蒸馏（**knowledge distillation**）。另一位成员建议尝试 HuggingFace 上的 **SpeechBrain** 作为可能的解决方案。
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255237341052670063)** (327 条消息🔥🔥): 

- **在 RateYourMusic 上使用 AI 生成音乐**：成员们讨论了通过使用 RateYourMusic 网站的 ID 来生成任何音乐家的歌曲和歌词。一位成员尝试了这种方法并确认了其有效性，称其“非常滑稽”。
  
- **Open Model Initiative 争议**：关于 LAION 退出 Open Model Initiative 及其涉及含有问题内容的讨论非常激烈。一位成员推测 LAION 可能因为分享非合成数据集而被排除在外，但其他人认为这是一个自愿的决定。

- **合成数据与非合成数据的辩论**：几位成员辩论了在训练 AI 模型的数据集中包含 NSFW（不适宜工作场所）内容的问题。担忧包括道德和公关影响，一些人主张排除 NSFW 内容，而另一些人则对 SD3 等模型上严厉的安全措施持批评态度。

- **GPU 和工作站讨论**：成员们对比了用于 AI 训练的不同 GPU，包括 A6000s、3090s 和 P40s，讨论了在 VRAM、成本和性能之间的权衡。他们还谈到了实际方面，如系统散热、将模型放入单个 VRAM 与分片（sharding）的对比，以及特定模型的效率和与某些 GPU 的兼容性。

- **用于 Transformer 的 ASIC 芯片**：关于 Etched 的 Sohu 有一场有趣的讨论，这是一种专门用于 Transformer 模型的芯片，声称比 GPU 更快、更便宜。一些成员对其实用性表示怀疑，因为其明显的缺乏灵活性可能会限制其仅用于特定类型的 AI 模型。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07992">MambaOut: Do We Really Need Mamba for Vision?</a>：Mamba 是一种具有类似 RNN 的状态空间模型（SSM）Token Mixer 的架构，最近被引入以解决 Attention 机制的二次复杂度问题，并随后应用于视觉领域...</li><li><a href="https://www.etched.com/announcing-etched">Etched is Making the Biggest Bet in AI</a>：未找到描述</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Bryan Johnson /dd (@bryan_johnson) 的推文</a>：很高兴投资 @Etched 的 1.2 亿美元 A 轮融资。便宜 10 倍的 AI 模型将使我们解决衰老问题的速度提高 100 倍。引用 Etched (@Etched) —— 认识 Sohu，有史以来最快的 AI 芯片...</li><li><a href="https://tenor.com/view/theoffice-stevecarrell-michaelscott-no-godplease-gif-4593632">No GIF - Theoffice Stevecarrell Michaelscott - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/tenstorrent/tt-firmware">GitHub - tenstorrent/tt-firmware: Tenstorrent 固件仓库</a>：Tenstorrent 固件仓库。通过在 GitHub 上创建账户为 tenstorrent/tt-firmware 开发做出贡献。</li><li><a href="https://tenor.com/view/jim-halpert-the-office-confused-gif-25227530">Jim Halpert GIF - Jim Halpert The - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://shop.lambdalabs.com/gpu-workstations/vectorone/customize)">Lambda | 为 AI 打造的 GPU 计算</a>：为 AI 开发者构建的 GPU 云。提供按需和预留的云端 NVIDIA H100、NVIDIA H200 和 NVIDIA Blackwell GPU，用于 AI 训练和推理。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255240235139334155)** (7 条消息): 

- **关于模型投毒（Poisoning Models）的辩论**：一位成员对有人“积极鼓励模型投毒”表示担忧，这表明模型训练伦理方面存在争议。
- **AIW+ 问题更难但可解**：另一位成员澄清说，AIW+ 问题虽然比简单的 AIW 更复杂，但仍然是一个常识性问题且是可解的。他们建议查看论文的补充材料以获取解决方案。 
- **警惕人工评估**：建议不要进行人工评估，因为由于重复试验的结果不一致，人工评估可能会产生高度误导。建议使用系统的 Prompt 变体，并对每个 Prompt 变体进行至少 20 次试验。
- **对 AIW+ 解决方案的分歧**：一位成员对提供的 AIW+ 问题解决方案提出了异议，称其由于未考虑的亲属关系而是错误且模糊的。他们还评论说，模型与该解决方案的一致性并不能消除这种模糊性。

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1255332843534422038)** (2 messages): 

- **EleutherAI 在 ICML 2024**: EleutherAI 的成员们分享了他们在 ICML 2024 展示多篇论文的兴奋之情，涵盖了从 Classifier-Free Guidance 到开源基础模型的社会影响等一系列主题。为了让社区了解最新动态，他们提供了论文链接，例如 [Stay on topic with Classifier-Free Guidance](https://arxiv.org/abs/2306.17806) 和 [Neural Networks Learn Statistics of Increasing Complexity](https://arxiv.org/abs/2402.04362)。

- **理解 LM 中的记忆**: 一位成员强调了他们在深入理解语言模型（LM）记忆方面的研究工作，引入了一种分类法来区分背诵（recitation）、重构（reconstruction）和回忆（recollection）。他们分享了一份 [preprint](https://arxiv.org/abs/2406.17746) 和一个 [Twitter thread](https://x.com/nsaphra/status/1805964526405161457)，详细阐述了他们的发现及其对版权、隐私和泛化（generalization）的影响。

**提到的链接**: <a href="https://x.com/nsaphra/status/1805964526405161457)">Naomi Saphra (@nsaphra) 的推文</a>：人类不仅仅是“记忆”。我们会背诵学校里钻研过的诗歌。我们会根据更通用的知识重构代码片段。我们会回忆生活中的片段。为什么要以统一的方式对待 LM 中的记忆...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255269463402483853)** (98 messages🔥🔥): 

- **寻找最佳多模态模型**: 一位成员询问如何定位性能顶尖的多模态模型，特别是 Image+Text to Text 模型，并分享了一个 [Huggingface 链接](https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer) 作为参考。这为其他寻找类似资源的人提供了帮助。

- **ICML 社交线程启动**: 为在奥地利维也纳举行的 [ICML](https://icml.cc/Conferences/2024) 开设了一个社交线程，用于协调聚会和活动。成员们讨论了后勤工作并计划了聚会，表现出极高的参与热情。

- **Goldfinch 模型细节分享**: 分享了关于混合 Goldfinch 模型的信息，该模型具有改进的 Llama 风格 Transformer 层，并与 Finch B2 配对。成员们交换了 [链接并通过私信提供了更多细节](https://discord.com/channels/729741769192767510/1103039376184852622/1246105198963982348)，并讨论了技术细节。

- **记录 LLM 处理 OOD 输入的情况**: 讨论了一篇关于神经网络预测在面对分布外（OOD）输入时表现的论文，特别是 [这个 arxiv 链接](https://arxiv.org/abs/2310.00873)。这引发了关于 LLM 是否表现出类似行为以及对 Bayesian DL 影响的讨论。

- **视觉模型推荐请求**: 一位成员请求推荐能够对包含图像数据的 PDF 执行 RAG 的视觉模型。遗憾的是，对话中没有产生具体的模型推荐。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/leaderboards/LeaderboardsExplorer">LeaderboardExplorer - leaderboards 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/spongebob-eating-chewing-popcorn-gif-16655546">海绵宝宝吃东西 GIF - Spongebob Eating Chewing - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/cat-hi-hello-close-up-kitten-gif-16709314">猫咪打招呼 GIF - Cat Hi Hello - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/2310.00873">Deep Neural Networks Tend To Extrapolate Predictably</a>: 传统观点认为，当面对分布外 (OOD) 输入时，神经网络的预测往往是不可预测且过度自信的。我们的工作重新评估了神经网络的这一假设...</li><li><a href="https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009">担心害怕 GIF - Worried Scared Oh No - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/cat-cat-crazy-crazy-cat-insane-cat-going-insane-gif-5752628082217795406">疯狂猫咪 GIF - Cat Cat crazy Crazy cat - 发现并分享 GIF</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255280862199550054)** (114 条消息🔥🔥): 

- **Synquid 的对比评估**：成员们讨论了论文 [Synquid](https://arxiv.org/abs/2406.16450) 的优缺点，称赞其实验设计周详，但对缺失某些基准测试（如“无激活函数”）表达了复杂看法。一位成员指出，“在随机初始化时，它在复杂度度量上的得分也会更低”，强调了这一基准测试在分析中的重要性。

- **论文评议中的 NRS 框架**：讨论检查了一篇关于神经网络初始化和归纳偏置 (inductive biases) 论文中的假设检验。一位成员表示，*“初始化的复杂度与相似复杂度任务的下游性能相关”*，而其他人则批评了对现有工作的重新解读，特别是关于低损失解随机采样的立场。

- **多模态指标的实现与实验验证**：成员们分析了一篇关于 JEST 的论文，重点讨论了多模态对比学习中用于数据策展 (data curation) 的联合样本选择。他们讨论了论文中声称的显著效率提升，指出该方法以更少的迭代次数和计算需求超越了 state-of-the-art 模型。

- **同态加密与 LLM**：成员们简要探讨了在 LLM 中使用同态加密 (homomorphic encryption) 的推测性本质，正如 [Zama AI 博客文章](https://www.zama.ai/post/chatgpt-privacy-with-homomorphic-encryption) 中所讨论的那样。讨论对同态加密在实时应用中的实际进展持怀疑态度。

- **Transformer 中的泛化与 Grokking**：成员们争论了论文中是否混淆了 Grokking 和泛化 (generalization)，指出 *“Grokking 特指在长时间的训练平滞期后，评估性能突然发生的转变。”* 他们批评了论文的方法以及泛化研究的历史背景。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.17711">Data curation via joint example selection further accelerates multimodal learning</a>：数据策展是大规模预训练的重要组成部分。在这项工作中，我们证明了联合选择数据批次比独立选择样本对学习更有效...</li><li><a href="https://arxiv.org/abs/2406.16450">Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward Layers</a>：LLM 的 state-of-the-art 结果通常依赖于规模，这导致计算成本昂贵。这引发了一项旨在减少这些模型参数量和...的研究议程。</li><li><a href="https://arxiv.org/abs/2406.17224">Large Language Models are Interpretable Learners</a>：在构建以人为中心的分类和决策预测模型时，表达能力与可解释性之间的权衡仍然是一个核心挑战。虽然符号规则提供了可解释性...</li><li><a href="https://www.zama.ai/post/chatgpt-privacy-with-homomorphic-encryption">Making ChatGPT Encrypted End-to-end</a>：通过同态加密，您可以在不泄露个人数据的情况下使用 LLM。</li><li><a href="https://arxiv.org/abs/2406.16747">Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers</a>：在自回归 Transformer 中高效容纳长序列，特别是在扩展的上下文窗口内，由于二次计算复杂度和...面临着重大挑战。</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>：我们研究了 Transformer 是否能学会对参数化知识进行隐式推理，这是即使是最强大的语言模型也难以掌握的技能。重点关注两种代表性的推理类型...</li><li><a href="https://openreview.net/forum?id=siCt4xZn5Ve">What Happens after SGD Reaches Zero Loss? --A Mathematical Framework</a>：理解随机梯度下降 (SGD) 的隐式偏置是深度学习中的关键挑战之一，特别是对于过参数化模型，其损失函数的局部极小值...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1255327906629025832)** (15 messages🔥): 

- **Self-Attention 被确认为（异联想）记忆模型**：一位成员澄清了 Self-Attention 的功能是作为一种（异联想）记忆模型，并指出其与 Hopfield networks 等联想记忆框架的联系。他们引用了论文 [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) 来支持这一观点。
- **LeCun 对 Transformers 的看法**：讨论引用了 Yann LeCun 将 Transformers 描述为“联想记忆”的说法。这与 Self-Attention 机制具有记忆模型特征的观点相吻合。
- **Hopfield Networks 论文引发关注**：论文 [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) 引发了广泛讨论，提到了其作者以及关于现代连续 Hopfield networks (MCHNs) 与 Self-Attention 密切相关的观点。
- **对 "Is All You Need" 类论文的批评及例外**：一位成员表达了对标题为 "Is All You Need" 的论文的反感，但也承认某些论文，如 [Hopfield Networks is All You Need](https://arxiv.org/abs/2309.08632)，具有非凡的价值。该用户引用了其对 Grokking 的创新处理以及对该领域的整体贡献。
- **Hopfield layers 作为单步 Attention**：提供了关于 Hopfield layers 在神经网络中实际运作方式的说明，指出记忆发生在 Pre-training 期间，而检索发生在 Forward pass 过程中。每个操作都被定义为 Hopfield network 的单步执行，强调了在 Self-Attention 机制中的实际应用。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2008.02217">Hopfield Networks is All You Need</a>：我们引入了一种具有连续状态和相应更新规则的现代 Hopfield network。新的 Hopfield network 可以存储指数级（相对于联想空间的维度）多的模式...</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>：受近期展示了基于 Transformer 的小型语言模型在精心策划的数据上预训练前景的工作启发，我们通过投入大量资源策划一个...</li><li><a href="https://arxiv.org/abs/2202.04557">Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models</a>：文献中已经提出了大量的联想记忆神经网络模型。这些模型包括经典的 Hopfield networks (HNs)、稀疏分布式记忆 (SDMs) 以及更多...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1255240451045326870)** (4 条消息): 

- **SAEs 被识别为可恢复线性特征**：一位成员分享了一份研究报告，显示“SAEs 从过完备基（overcomplete basis）中恢复线性特征”，并强调使用“在隐藏层激活上带有 L1 惩罚的单层自编码器”可以识别出超越最小化损失的特征。他们链接到了 [LessWrong 帖子](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)，并感谢了其他研究人员的反馈。

- **对用于 SAE 测试的 toy models 感兴趣**：受到另一篇强调特征几何在叠加假设（superposition hypothesis）之外重要性的帖子启发，同一位成员表示有兴趣探索 toy models 来测试 SAEs。他们分享了关于该主题的另一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis)，其中讨论了神经网络特征向量中的结构信息。

- **对多语言和安全性工作的兴奋**：一位成员分享了一个关于多语言、安全性和机械可解释性（mechanistic interpretation）新工作的 [Twitter 链接](https://x.com/yong_zhengxin/status/1805616252490236235?s=46)，强调“仅使用英语进行 DPO 训练就可以使许多其他语言的 LLM 去毒”。他们还提供了相关的 [arXiv 研究论文](https://arxiv.org/abs/2406.16235)链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/yong_zhengxin/status/1805616252490236235?s=46">来自 Zheng-Xin Yong (Yong) (@yong_zhengxin) 的推文</a>：🔥多语言 + 安全性 + mech interp 的新工作！我们展示了仅使用英语进行 DPO 训练就可以使许多其他语言的 LLM 去毒。我们还对跨语言如何……给出了机械解释。</li><li><a href="https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis">SAE 特征几何在叠加假设之外 — LessWrong</a>：写于 Apollo Research • 摘要：基于叠加的神经网络激活空间解释是不完整的。特定位置……</li><li><a href="https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition.">[中期研究报告] 使用稀疏自编码器将特征从叠加中提取出来 — LessWrong</a>：我们感谢 Trenton Bricken、Eric Winsor、Noa Nabeshima 和 Sid Black 提供的有益评论。 …
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1255334051980771390)** (16 条消息🔥): 

- **AMD MI300X 挑战 Nvidia 的 GPU 霸主地位**：一篇关于 AMD Radeon Instinct MI300X 的帖子强调了其旨在挑战 Nvidia 在 GPU 计算市场领先地位的目标。虽然 AMD 的软件生态系统 ROCm 仍落后于 Nvidia 的 CUDA，但 MI300X 代表了其独立克服硬件差距的努力。[完整帖子](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/)。

- **Etched 推出 Transformer ASIC**：Etched 的[新型 Transformer ASIC 芯片](https://www.etched.com/)声称通过将 Transformer 架构直接刻入硅片，运行 AI 模型比 GPU 更快且更便宜。该芯片承诺可用于实时语音 Agent 等应用，并具备运行万亿参数模型的能力。

- **对 Etched 的 ASIC 声明持怀疑态度**：用户对 Etched ASIC 的实际优势表示怀疑，特别是仅刻入架构而不包含权重是否能实现承诺的性能提升。讨论强调了 AI 硬件领域的竞争和快速进步。

- **Etched 获得重大投资**：Bryan Johnson 宣布他很高兴投资 Etched 的 1.2 亿美元 A 轮融资，并引用了该公司的说法，即他们的 Sohu 芯片运行 AI 模型可以便宜 10 倍，并能用一台 8xSohu 服务器取代 160 块 Nvidia H100 GPU。[推文链接](https://x.com/bryan_johnson/status/1805629207374086490)。

- **关于 AI 芯片未来的辩论**：用户辩论了 ASIC 等专用 AI 芯片与 GPU 相比的未来角色，并提到了[行业方向](https://www.pixelstech.net/article/1719027344-The-Future-of-AI-Chips-Might-Not-Be-GPU)正朝着专用硬件加速器发展。讨论中强调了模型架构快速变化的潜力和 Tensor Core 的灵活性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/">测试 AMD 的巨型 MI300X</a>：编辑注 (2024/6/26)：我们重新阐述了致谢部分，以更清楚地说明我们在本文中没有得到来自 AMD 的直接支持。我们的测试是完全独立的，AMD 并没有……</li><li><a href="https://www.etched.com/">Etched | 全球首款 Transformer ASIC</a>：刻入硅片的 Transformer。通过将 Transformer 架构烧录进我们的芯片，我们正在为 Transformer 推理打造全球最强大的服务器。</li><li><a href="https://www.youtube.com/watch?v=zh6REnqwXe4">AI 芯片初创公司 Etched 旨在挑战 Nvidia</a>：AI 芯片初创公司 Etched 筹集了 1.2 亿美元，用于扩大其专用芯片的生产，并吹嘘该芯片将与 Nvidia 的产品竞争。Etched 首席执行官 Gavin Uber...</li><li><a href="https://x.com/bryan_johnson/status/1805629207374086490">Bryan Johnson /dd (@bryan_johnson) 的推文</a>：很高兴投资 @Etched 的 1.2 亿美元 A 轮融资。便宜 10 倍的 AI 模型将使我们解决衰老问题的速度提高 100 倍。引用 Etched (@Etched) 的话：认识一下 Sohu，史上最快的 AI 芯片。Wi...</li><li><a href="https://www.pixelstech.net/article/1719027344-The-Future-of-AI-Chips-Might-Not-Be-GPU">AI 芯片的未来可能不是 GPU | Pixelstech.net</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1255625274389954710)** (1 条消息): 

- **新用户寻求 Triton 问题帮助**：一位新成员介绍了自己，并分享了他们在 Triton 仓库中发起的一个 issue。他们正在寻找关于如何在 `python.triton.language.core` 中添加 `pow` 函数的指引，并提供了 [issue 链接](https://github.com/triton-lang/triton/issues/4190)。

**提到的链接**：<a href="https://github.com/triton-lang/triton/issues/4190">如何在 python.triton.language.core 中添加 pow 函数？· Issue #4190 · triton-lang/triton</a>：我尝试在 triton.jitted 函数中使用 pow 操作，如：output = x + y**3 ^ 然而得到了 AttributeError("'tensor' object has no attribute '__pow__'")。在文件 python/trit...

  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1255281600376209510)** (6 条消息): 

- **PyTorch 纪录片首映**：成员们分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s)，指向 "PyTorch Documentary Virtual Premiere: Live Stream"，该纪录片汇集了项目从早期到现在的关键人物。多位用户反复发布此链接以强调其重要性。
- **山羊表情符号造势**：一位成员用山羊表情符号 (*🐐*) 回应了 PyTorch 纪录片链接，象征着兴奋和期待。另一位成员也注意到了这一反应并进行了转发，以突出这种情绪。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=EjgTv6aSeqk&t=8...]">PyTorch Documentary Virtual Premiere: Live Stream</a>：加入我们，参加 PyTorch 纪录片的官方发布会！听取项目从早期到现在的关键人物的分享。</li><li><a href="https://www.youtube.com/watch?v=EjgTv6aSeqk&t=869s">PyTorch Documentary Virtual Premiere: Live Stream</a>：加入我们，参加 PyTorch 纪录片的官方发布会！听取项目从早期到现在的关键人物的分享。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1255414516959940608)** (1 条消息): 

- **Adam-mini 优化器减少内存占用**：Adam-mini 被提议作为一种优化器，在减少 45% 到 50% 内存使用的同时，提供与 AdamW 相当或更好的性能。[GitHub 仓库](https://github.com/zyushun/Adam-mini) 包含了代码和实现的详细信息。

**提到的链接**：<a href="https://github.com/zyushun/Adam-mini">GitHub - zyushun/Adam-mini: Code for the paper: Adam-mini: Use Fewer Learning Rates To Gain More</a>：论文代码：Adam-mini: Use Fewer Learning Rates To Gain More - zyushun/Adam-mini

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255335011754704979)** (38 messages🔥): 

- **PyTorch 中的线性代数原始 Kernel**：一位用户分享了 PyTorch 仓库中[原始 Kernel 的链接](https://github.com/pytorch/pytorch/blob/b7e7a4cb01de394af7686ab6feb216a8a5c716bb/aten/src/ATen/native/LinearAlgebra.cpp#L3476)。该 Kernel 位于代码的 native 线性代数部分。
- **PyTorch 中的 Subclass dtype 问题**：成员们讨论了 Tensor 子类无法反映其真实 `dtype` 的问题，这增加了兼容性和易用性的复杂性。Marksaroufim 鼓励在 PyTorch 上提交 issue，并建议研究内部改进。
- **开源贡献的价值**：Locknit3 质疑开源贡献是否有助于求职，引发了辩论。Gau.nernst 和 kashimoo 肯定了其价值，并提到了招聘人员注意到他们贡献的实例。
- **将 HQQ 与 TorchAO 集成**：成员们讨论了将 HQQ 与 TorchAO 的 `quantize()` API 集成的潜力，并链接到了 [HQQ 优化器](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L194-L243)。他们强调了该算法的简洁性，并建议将其作为 INT4 量化的新基准。
- **低比特融合 GEMV CUDA Kernels**：Mobicham 分享了他们一直在开发的低比特融合 GEMV CUDA Kernels，概述了其灵活性和当前的局限性。Gau.nernst 询问了对奇数位宽的支持，Mobicham 确认了其可行性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1jqC53MwiW9dSiPS-a6hng_yo0ywdc3nH#scrollTo=Aj9ii4darSRA">Google Colab</a>：未找到描述</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L194-L243">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/releases/tag/v0.3.0">Release v0.3.0 · pytorch/ao</a>：v0.3.0 亮点 我们很高兴地宣布 torchao 0.3 版本发布！此版本增加了对新 quantize API、MX 格式、FP6 dtype 和位打包、2:4 稀疏加速训练等的支持...</li><li><a href="https://github.com/pytorch/ao/blob/f172c474cbd56641bb34e73df5d61818a9d4e6e1/torchao/_models/llama/model.py#L122).">ao/torchao/_models/llama/model.py at f172c474cbd56641bb34e73df5d61818a9d4e6e1 · pytorch/ao</a>：创建并集成自定义数据类型、布局和 Kernel，推理速度提升高达 2 倍，显存占用减少 65%，并支持训练 - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch/blob/b7e7a4cb01de394af7686ab6feb216a8a5c716bb/aten/src/ATen/native/LinearAlgebra.cpp#L3476">pytorch/aten/src/ATen/native/LinearAlgebra.cpp at b7e7a4cb01de394af7686ab6feb216a8a5c716bb · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1255424670711287818)** (8 messages🔥): 

- **Axis 设置影响 HQQModelForCausalLM 性能**：一位用户报告了与 `meta-llama/Llama-2-7b-chat-hf` 和 `Mistral-7B-v0.1` 相关的 `HQQModelForCausalLM` 问题，特别是当设置 `axis=0` 时，会跳过使用 `torchao` 的 int4 kernel。另一位用户澄清说，`axis` 控制执行分组的轴，由于 Kernel 支持的原因，它会同时影响困惑度（perplexity）/lm-eval 分数以及推理速度。
- **推理质量问题与 HF 的 transformers 缓存实现有关**：提到了 [Hugging Face's Transformers](https://github.com/huggingface/transformers) 缓存实现的质量问题，认为这可能是模型评估问题的潜在根源。
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255237578945335449)** (146 messages🔥🔥): 

- **缩放 LR 和更新裁剪 (update clipping) 的争议**：成员们讨论了缩放**学习率 (LR)**和使用**更新裁剪 (update clipping)**。一位成员分享道，“*老实说，某种形式的 update clipping 对我来说听起来仍然是合理的，*”而另一位则指出，“*为了稳定化，我们分析了损失峰值 (loss spikes)*，”并指向了 [AdamW 的论文](https://mlfoundations.github.io/advancedml-sp23/assets/adam.pdf)和另一个 [arXiv 链接](https://arxiv.org/pdf/1905.11286)。
  
- **AMD 与 NVIDIA 系统构建**：一位用户报告称使用 *RDNA3 显卡* 构建了一台机器，并拥有大型机器的使用权限；而另一位提到正在使用 *A6000* 并计划构建一个 AMD 系统，同时考虑了像 *Azure 的 MI300X* 实例这样的潜在供应商。

- **Sohu ASIC 芯片可能彻底改变 Transformer**：一位成员强调了一条关于 [Sohu 新型 ASIC 芯片](https://x.com/Etched/status/1805625693113663834)的推文，该芯片声称运行 Llama 70B 时每秒可处理 500,000 个 token，有可能取代 160 块 H100 GPU。随后出现了关于其是否仅针对 Transformer 模型进行了专门优化以及对通用性影响的疑问。

- **FP8 集成引发褒贬不一的反应**：关于将 FP8 集成到现有系统中的讨论，在平衡简单性与更大幅度的改动之间，结论是“*可选，但如果 PR 状态足够好，还是会接受*”。此外还分析了避免全局 amax 历史记录并使用局部缩放 (local scaling) 的可行性。

- **在 Lambda 上进行有效的多节点训练**：一位成员分享了他们在 Lambda 上成功部署的 16 GPU 多节点训练设置，实现了近 2 倍的加速，并表示：“*设置它并不容易，但结果非常出色。*”

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Etched/status/1805625693113663834">来自 Etched (@Etched) 的推文</a>：遇见 Sohu，史上最快的 AI 芯片。运行 Llama 70B 每秒超过 500,000 个 token，Sohu 让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 块 H100。Soh...</li><li><a href="https://github.com/karpathy/llm.c/pull/636">karpathy 的滚动检查点 (rolling checkpoints) · Pull Request #636 · karpathy/llm.c</a>：检查点分为 MINOR 或 MAJOR，较小的检查点会在滚动窗口中被删除。这是一种优化，允许我们更频繁地保存状态，但总体上节省了磁盘空间。...</li><li><a href="https://github.com/karpathy/llm.c/pull/629/">rosslwheeler 的 CI Dataloader 测试和 ptx/sass 文件生成器 · Pull Request #629 · karpathy/llm.c</a>：新的 CI 测试文件 - 添加了 dataloader 测试和 ptx/sass 文件生成器。Cuda Makefile - 增加了从主 Makefile 构建的能力。增加了对 ptx 和 sass 输出文件的支持。layernorm_forwar...</li><li><a href="https://github.com/karpathy/llm.c/pull/629/files">rosslwheeler 的 CI Dataloader 测试和 ptx/sass 文件生成器 · Pull Request #629 · karpathy/llm.c</a>：新的 CI 测试文件 - 添加了 dataloader 测试和 ptx/sass 文件生成器。Cuda Makefile - 增加了从主 Makefile 构建的能力。增加了对 ptx 和 sass 输出文件的支持。layernorm_forwar...</li><li><a href="https://github.com/karpathy/llm.c/pull/635">ngc92 的设备端归约 (On-device reductions) · Pull Request #635 · karpathy/llm.c</a>：将损失计算移动到反向传播中，并确保我们可以进行更多的设备端归约，减少主机与设备之间的数据传输。还启用了一项微优化，即验证过程不计算 dlogit...</li><li><a href="https://github.com/warner-benjamin/optimi/blob/4542d04a3974bb3ac9baa97f4e417bda0432ad58/optimi/stableadamw.py#L28>).">warner-benjamin/optimi 中的 stableadamw.py</a>：快速、现代、内存高效且低精度的 PyTorch 优化器 - warner-benjamin/optimi
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1255352848498167808)** (2 messages): 

- **Intel PyTorch 团队致力于 XPU 支持**：Intel PyTorch 团队正积极致力于在原生 PyTorch 中启用 **XPU (Intel GPU)** 支持。他们在 GitHub 上分享了一个 [RFC](https://github.com/pytorch/pytorch/issues/114723)，以讨论上游化 (upstreaming) 过程，并利用 Intel 在 GPU 技术方面的进展。

**提到的链接**：<a href="https://github.com/pytorch/pytorch/issues/114723">[RFC] Intel GPU 上游化 · Issue #114723 · pytorch/pytorch</a>：TL;DR 本 RFC 文档旨在提议并讨论在 PyTorch 中集成 Intel GPU 支持的上游化。我们的重点是利用 Intel 在 GPU 技术方面的进展来增强 PyTorch 的性能...

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255240892625850368)** (153 条消息🔥🔥): 

- **Pro focus search 的限制与更新**：用户讨论了 Perplexity Pro focus search 的问题，指出非 Pro + Reddit 搜索运行正常。一位用户对 Standard 模式下的 Pro 搜索返回比以前更多的来源表示赞赏。

- **Claude 3.5 的能力与可用性**：成员们讨论了 **Claude 3.5 Sonnet** 在 Perplexity Pro 中的上下文窗口（context window）约为 32k tokens，而 Claude Pro 则提供完整的 200k tokens。Claude 3.5 在 Android 上的可用性也得到了确认。

- **API 过滤器与未公开功能**：有人提出了关于 API 引用日期过滤器和可能存在的未公开功能（如搜索域名过滤器）的问题。用户讨论了某些功能是在开发中，还是可以通过 workaround 方法使用。

- **关于 AI 搜索质量和新功能的辩论**：用户将 Perplexity 的回答质量与 ChatGPT 进行了对比，认可了 Perplexity 处理多步查询的新型 Agentic 搜索能力。一些人对 Perplexity 的摘要和来源处理表示不满，认为这经常导致回答中出现幻觉（hallucination）。

- **服务中断与 API 状态担忧**：用户报告了 Perplexity API 的 5xx 错误，并对缺乏检查服务运行时间（uptime）的状态页面表示沮丧。用户呼吁提供更好的透明度和基础的 API 管理功能。

**提到的链接**：<a href="https://entertainment.slashdot.org/story/24/06/26/001222/researchers-upend-ai-status-quo-by-eliminating-matrix-multiplication-in-llms">研究人员通过消除 LLM 中的矩阵乘法颠覆 AI 现状 - Slashdot</a>：来自加州大学圣克鲁斯分校（UC Santa Cruz）、加州大学戴维斯分校（UC Davis）、LuxiTech 和苏州大学的研究人员开发了一种新方法，通过消除矩阵乘法（matrix multiplication）来更高效地运行 AI 语言模型，这可能会减少...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255314661310726184)** (10 条消息🔥): 

- **CTO Denis Yarats 讨论 AI 创新**：观看 [YouTube 视频](https://www.youtube.com/watch?v=gvP-DxqatLQ)，了解 Perplexity AI 的 CTO Denis Yarats 讨论在创建高质量搜索体验中对 AI 的创新使用。Yarats 与 Lukas Biewald 在 Gradient Dissent 节目中深入探讨了这一话题。
- **土卫六（Titan）的波浪、中国的探月胜利以及大众汽车对 Rivian 的投资**：Perplexity AI 的 [YouTube 视频](https://www.youtube.com/embed/HSmt6qvwuS0) 探讨了引人入胜的话题，如土卫六失踪的波浪、中国的探月成就以及大众汽车对 Rivian 的投资。
- **Perplexity.ai 上的热门搜索**：探索 Perplexity AI 上的各种搜索，包括 [Intel CPU](https://www.perplexity.ai/search/IntelCPU-tz.a_Iv0TlultEIdNpQLPw)、[Perplexity 功能](https://www.perplexity.ai/search/how-does-perplexity-NVeFu4LMQ0K0RdU7PrFP1A)、[Hive 区块链](https://www.perplexity.ai/search/Hive-blockchain-HZ6MEvTqRf.HQpl2wMIurA) 以及 [5000 单元结果](https://www.perplexity.ai/search/5000-Jy_3Oq3dTpqG7l11w8PSNQ)。
- **Perplexity.ai 上的深度页面**：发现详细页面，例如[此链接](https://www.perplexity.ai/page/Overcoming-Trauma-and-9_3ox12FRFaMON3Zk8lezQ)中关于克服创伤的内容，以及[此处](https://www.perplexity.ai/page/Julian-Assange-Released-cLtbci_iSxW32Xve2NgKGA)关于 Julian Assange 获释的更新。
- **关于重力的奇闻**：通过此[链接](https://www.perplexity.ai/search/If-Gravity-affects-20bEiugFSnudRflOAGYlnA)查看关于重力如何影响感知及相关现象的深入搜索结果。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/HSmt6qvwuS0">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=gvP-DxqatLQ">利用 Perplexity AI CTO Denis Yarats 变革搜索</a>：在这一集 Gradient Dissent 中，Perplexity 的 CTO Denis Yarats 加入主持人 Lukas Biewald，讨论在创建高质量...中对 AI 的创新使用。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255366046337208432)** (5 条消息): 

- **关于 Perplexity AI 使用的困惑**：一名成员表示困惑，要求澄清另一名用户的意思。未提供额外的上下文或回答。
- **请求 Perplexity AI 搜索功能**：另一名成员想知道如何让 Perplexity AI 执行针对近期事件的搜索功能，例如获取有关新车的详细信息。
- **功能建议：llama-3-sonar-*-32k-online**：针对关于搜索功能的查询，另一名成员建议尝试名为 *"llama-3-sonar-*-32k-online"* 的功能。
- **关于支持引用和图像的 closed beta API 的咨询**：一名成员询问在申请访问权限后，是否有人获得了包含引用（citation）和图像功能的 closed beta API 访问权限。
- **Perplexity API 问题及状态查询**：一名成员报告在使用 Perplexity API 时收到 5xx 错误，并询问是否有状态页面可以检查 API 服务器何时恢复运行。

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255238244241641472)** (66 条消息🔥🔥): 

- **AI Engineer World’s Fair 观影会协调**：一名成员询问是否有人可以为 **AI Engineer World’s Fair** 举办观影会，该活动将在此处 [直播](https://www.youtube.com/watch?v=5zE2sMka620)。活动包括主题演讲和代码专场。

- **PyTorch 纪录片首映**：关于 [PyTorch Documentary Virtual Premiere](https://www.youtube.com/live/EjgTv6aSeqk) 的公告引发了关注。该片将邀请项目历史和现状发展中的关键人物。

- **ChatGPT Voice Mode 延迟讨论**：一名成员分享了 [Teknium 的推文](https://x.com/Teknium1/status/1805718678526476655/photo/1)，讨论 ChatGPT 延迟发布其高级 Voice Mode 的情况。更新中存在 **移除 waifu 功能** 的问题。

- **对 AIE 上的 AI 可穿戴设备感到兴奋**：参加 AI Engineer 活动嘉宾晚宴的与会者收到了来自 **Bee Computer 的 AI 可穿戴设备**。一位成员表示，“*它几乎了解关于我的所有最重要事实……并且有一份我的 TODO 列表*”，表明其功能令人印象深刻。

- **对从小鼠视觉皮层重建电影的着迷**：一名成员对 [从小鼠视觉皮层活动重建的电影](https://x.com/Neuro_Joel/status/1805221959191437356) 感到惊讶。他们形容这项神经科学成就令人叹为观止。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Neuro_Joel/status/1805221959191437356">Joel Bauer (@Neuro_Joel) 的推文</a>: 🚀 很高兴分享我博士后期间在 @ClopathLab 和 @SWC_Neuro 的 Troy Margrie 实验室的第一份预印本！我们从小鼠视觉皮层活动中重建了电影。📽️✨ #Neuroscience #...</li><li><a href="https://x.com/itsandrewgao/status/1805772589970649534?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">andrew gao (@itsandrewgao) 的推文</a>: 💋📚 唇语识别 AI 投入使用！我曾臭名昭著地发推说我看空语音。今天，我改变了主意。@SymphonicLabs 训练了 AI 来 **读你的唇语**，现在我可以完全使用语音界面了...</li><li><a href="https://x.com/altryne/status/1805840851626869196?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 天哪... 在 @aiDotEngineer 嘉宾晚宴上的每个人都收到了一款新的 AI 可穿戴设备 @bee__computer 作为礼物。我完成了新手引导（稍后会发布视频），戴上它后有点忘了这回事，然后工具...</li><li><a href="https://youtube.com/@aidotengineer?si=KfTkCwPDCRU7jY3t">AI Engineer</a>: 为 AI Engineers 提供的演讲、研讨会、活动和培训。</li><li><a href="https://www.youtube.com/live/EjgTv6aSeqk">PyTorch Documentary Virtual Premiere: Live Stream</a>: 加入我们的 PyTorch 纪录片官方发布会！听取项目中从早期到现在的关键人物的分享。</li><li><a href="https://youtu.be/ziGNnhNABqA?si=KcpiPiduDLHIpywA">David Luan: Why Nvidia Will Enter the Model Space &amp; Models Will Enter the Chip Space | E1169</a>: David Luan 是 Adept 的 CEO 兼联合创始人，该公司正在为知识工作者构建 AI agents。迄今为止，David 已为公司筹集了超过 4 亿美元...</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://x.com/TheXeophon/status/1805718926162280804">Xeophon (@TheXeophon) 的推文</a>: Anthropic 在一周内：- 发布新模型 - 发布 Artifacts - 在 Claude 中发布 Projects。OpenAI：- 发布了一个仅限 Mac 的应用 - 延迟语音发布数月。引用 OpenAI (@OpenAI)...</li><li><a href="https://x.com/MLStreetTalk/status/1805686042726445337">Machine Learning Street Talk (@MLStreetTalk) 的推文</a>: 我的梦想今天实现了——与 AI 界的元老 @SchmidhuberAI 进行了一场史诗般的拍摄。</li><li><a href="https://developer.bee.computer">Bee Developer Platform</a>: 未找到描述</li><li><a href="https://x.com/Teknium1/status/1805718678526476655/photo/1">Teknium (e/λ) (@Teknium1) 的推文</a>: 他们在 gpt4o 语音更新中移除 waifu 功能时遇到了麻烦 :l</li><li><a href="https://x.com/altryne/status/1805840851626869196?s=46&t=tMWvmS3OL3Ssg0b9l">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 天哪... 在 @aiDotEngineer 嘉宾晚宴上的每个人都收到了一款新的 AI 可穿戴设备 @bee__computer 作为礼物。我完成了新手引导（稍后会发布视频），戴上它后有点忘了这回事，然后工具...</li><li><a href="https://x.com/austintbyrd/status/1806017268796854753?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Austin Byrd (@AustinTByrd) 的推文</a>: Figma AI 在开始向所有人收费前将免费提供一年</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255563805493297153)** (100 条消息🔥🔥): 

- **SQLite 中的向量表现出色**：多位用户对该话题表示兴奋，其中一位提到 “Vectors in SQLite 🔥”，另一位则截取了相关幻灯片的屏幕截图。
- **向量数据库宣告死亡**：有人发表了 “vectordbs are dead” 的大胆言论，引发了参与者的热议。
- **幻灯片将不予公开**：当被问及演示文稿的幻灯片稍后是否会公开时，得到的回答是坚决的 “不”，这让一些与会者感到失望。
- **AI Engineer 会议的小插曲**：会议面临多个问题，包括延迟 10 分钟开始导致一场演讲被取消，以及 OpenAI 在不到 48 小时前通知退出。Swyxio 对音频问题表示沮丧，称：“这个音频问题快把我气疯了。”
- **YouTube 直播跟进**：Swyxio 建议想要补看错过内容的与会者观看 [YouTube 直播](https://www.youtube.com/watch?v=5zE2sMka620)，内容为 “AI Engineer World’s Fair 2024 — Keynotes & CodeGen Track”。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@aiDotEngineer">AI Engineer</a>: 为 AI Engineer 提供的演讲、研讨会、活动和培训。 </li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255243079569375426)** (109 条消息🔥🔥): 

- **LM Studio 加载模型错误**：一名成员报告在 LM Studio (0.2.25) 中尝试加载模型时出现重复错误 *(Exit code: -1073740791)*。建议其提供系统规格并尝试不同的模型或配置。
- **3060 的 OOM 问题及替代模型讨论**：尝试在 RTX 3060ti 上运行 Hermes 2 Theta Llama-3 70B 会导致 “显存溢出” (OOM) 问题。建议改用 NousResearch 的 8b 版本。
- **在 Apple M 芯片上运行大模型的困扰**：一位用户描述了在统一内存上运行 Llama 3 70B 时遇到的问题，其中一个模型运行正常，而另一个则导致输出乱码。大家公认不同的量化类型（quant types）和设置（如 Q3 或 Q4 KM）可能会影响性能。
- **RAG 解释与应用**：深入解释了检索增强生成 (RAG) 如何辅助生成详细信息。分享了 [NVIDIA 关于 RAG 的博客](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) 链接，以便更好地理解该概念。
- **用于文档分析和摘要的 AnythingLLM**：对于需要文档分析和摘要生成的用户，推荐使用 “AnythingLLM”，因为它易于处理各种文档类型并能与 LM Studio 集成。它被指出是一个免费且开源的解决方案。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/autotrain/en/llm_finetuning">LLM Finetuning</a>: 未找到描述</li><li><a href="https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/">What Is Retrieval-Augmented Generation aka RAG?</a>: 检索增强生成 (RAG) 是一种通过从外部源获取事实来提高生成式 AI 模型准确性和可靠性的技术。</li><li><a href="https://js.langchain.com/v0.1/docs/expression_language/cookbook/retrieval/">Retrieval augmented generation (RAG) | 🦜️🔗 Langchain</a>: 现在让我们来看看如何在 Prompt 和 LLM 中加入检索步骤，从而构成一个 “检索增强生成” 链：</li><li><a href="https://llm.extractum.io/">LLM Explorer: A Curated Large Language Model Directory. LLM List. 34902 Open-Source Language Models.</a>: 浏览 34902 个开源的大型和小型语言模型，这些模型被方便地分为各种类别和 LLM 列表，并附有基准测试和分析。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255295355122352321)** (7 条消息): 

- **Traveler 宇宙中的 GM 专家**：一位用户要求聊天机器人扮演设定在原始 Traveler 宇宙中的硬核科幻角色扮演游戏的 Game Master。他们强调需要包含随机事件的不利结果，以增强成功的价值，并让用户有机会对 NPC 的行动做出反应。

- **缺乏对 70b New Dawn 的讨论**：一位成员注意到关于 **70b New Dawn 模型** 的讨论明显缺失，称其“非常出色”。另一位用户建议这可能是因为大多数用户只运行 7b-13b 等较小模型，限制了大型模型的曝光度。

- **在学术级本地模型上的挣扎**：一位用户对本地模型（特别是 **L3-Daredevil-Obliterated、L3 SFR-Iterative 和 L3 Hermes**）在学术讨论和复杂指令方面的表现表示不满。他们询问 34B 以下（如果可能的话，最好是 20B）模型的推荐，强调他们相比 OpenAI 等选项更倾向于 FOSS 和注重隐私的模型。

- **Bartkowski 的 Q4KM DeepCoder V2 性能**：一位用户分享了使用 LM Studio 0.2.25 运行 **Bartkowski 的 Q4KM DeepCoder V2 230B**（8K context）的配置，并讨论了其令人印象深刻的性能。他们记录了该模型的 RAM 和 GPU 显存占用情况，达到了每秒 2.68-2.8 tokens，并讨论了在 32K 等更高 context 长度下遇到的挑战。
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1255358103369678873)** (2 条消息): 

- **Discord URL 诈骗警报**：一位用户警告称，分享的一个 URL 链接到了俄罗斯的一个诈骗网站。他们强调该 URL 是虚假的。
- **消息删除确认**：另一位用户表示看不到那条可疑消息，推测可能已被删除。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1255360235078549544)** (15 条消息🔥): 

- **A4000 vs 4060ti 推理对比**：一位成员询问是否有人有 A4000 的使用经验，并想了解其在单槽位推理性能上与 **4060ti 16G** 的对比。
- **8x P40 组装完成**：一位成员分享说他们已经完成了使用 **8x P40 GPU** 的组装，集成在一个二手服务器机柜中，运行 Garuda Linux 且 Nvidia 驱动开箱即用。
- **LM Studio 中的 VRAM 报告**：有人询问 LM Studio 是否能准确报告多 GPU 系统的 **VRAM 正确数值**，一位用户表示他们的配置正确报告了约 192GB 的 VRAM。
- **家庭实验室噪音问题**：一条幽默的评论指出了家庭实验室配置的缺点，将服务器启动时的噪音比作“喷气发动机”。
- **服务器电源管理**：讨论内容包括使用 **200 amp 电路** 和 4x1000w 电源，并指出功耗在 1KW 左右。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1255303988690030613)** (7 条消息): 

- **Context window 影响 token 预测**：一位成员幽默地解释说，“context window 越长，模型就需要越多的 scratchpad”来跟踪 token 预测，当空间耗尽时，它就会开始输出乱码。他们还开玩笑说袜子的颜色也会影响性能，强调这些评论纯属个人轶事。
- **报告潜在的诈骗链接**：一位用户提醒管理员，某人的账号可能被盗，并发布了指向“冒充 Steam 的俄罗斯诈骗网站”的链接。
- **请求管理员协助**：一位用户请求管理员干预，因为诈骗链接已在多个频道中存在了数小时，并感谢了做出响应的人员。
- **Mamba 架构支持咨询**：一位用户询问 LM Studio 何时开始支持 Mamba 架构。
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255598967190982757)** (1 条消息): 

- **Interpreter 拒绝文档和图像上传**：一位成员对无法在他们的 interpreter 中“直接将文档或图像移动到终端”感到沮丧。他们提到 interpreter 对这些操作“给了我禁令”且“不同意”。
  

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1255585869721505854)** (2 条消息): 

- **构建 Discord 机器人的 SDK 演练**：分享了一个针对有兴趣使用 SDK 创建 Discord 机器人的用户的实用指南。点击[此处](https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6)查看。

- **在 Python 中查询 Token 速度和生成时间**：一位成员询问如何使用 Python 从本地 **LM Studio server** 提取数据，特别是关注 Token 速度和生成首个 Token 所需的时间。

**提到的链接**：<a href="https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6">未找到标题</a>：未找到描述

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255416795654000770)** (49 条消息🔥): 

- **探索使用 Mojo 数据类型进行绘图**：一位用户询问是否有项目直接使用 **Mojo data types** 进行绘图，而不转换为 Numpy 和 Matplotlib。另一位用户分享了代码示例以及关于在 [GitHub](https://github.com/msteele/mojo-sdl) 上使用 **Mojo 与 Matplotlib** 以及用于 SDL2 绑定的 **Mojo-SDL 项目**的讨论链接。

- **社区关于图表库的决策**：一场关于社区对 **Mojo 图表库**需求的讨论展开了。对话涉及是否开发高层接口、支持交互式图表或专注于 Arrow 等数据输入格式的决策。

- **数据可视化中的交互性**：一位用户强调了**数据科学**可视化工具中交互性的重要性，建议像 **Vega specification** 这样的规范可以作为解决 Web 和原生渲染的 IR。一位 Vega 维护者透露了挑战以及像 **Mosaic** 所使用的替代查询引擎的潜力。

- **通过 WSL 在 Windows 上运行 Mojo**：一位用户确认 **Mojo 可以在 Windows 上使用 WSL 运行**，预计夏季结束前将提供原生支持。他们讨论了将 WSL 与 **VSCode** 链接的便利性，以及与使用 Linux 和目录相关的轻微学习曲线。

- **对绘图库的反思**：讨论了各种绘图库，如 **D3, Altair, Matplotlib, Seaborn** 和 **Plotly**，用户比较了它们的侧重点和目标受众。对话还涉及了 **UW 的 Mosaic 库**及其在数据可视化领域的创新方法。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/mojo-pi-approximating-pi-with-mojo-using-monte-carlo-methods">Modular: Mojo🔥 ❤️ Pi 🥧: 使用蒙特卡洛方法通过 Mojo 近似圆周率</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 ❤️ Pi 🥧: 使用蒙特卡洛方法通过 Mojo 近似圆周率</li><li><a href="https://github.com/msteele/mojo-sdl">GitHub - msteele/mojo-sdl: Mojo🔥 的最小 SDL2 绑定</a>：Mojo🔥 的最小 SDL2 绑定。通过在 GitHub 上创建账户为 msteele/mojo-sdl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1806070670293692594>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1255248595314278460)** (2 条消息): 

- **ARC 测试包含人类常见的模式识别**：一位成员将 ARC 测试描述为一个专注于“封闭区域、对称性、物体特征以及其他人类文化中常见事物”的目录。他们幽默地建议建立一个“狗狗 ARC 测试”，专注于与狗相关的特征，如粪便气味和吠叫声调。
- **AI 在智商测试中表现优异但缺乏真正的智能**：一位用户认为智商测试并不能衡量智能，这就是为什么 AI 可以在其中表现出色。他们认为 ARC 测试是“最 AI 的东西”，但批评其数据集非常贫乏。
- **质疑智能与意识的本质**：另一位成员思考了人类信息回忆与 LLM 等 AI 系统之间的区别。他们询问其他人是否区分智能与意识，并暗示回忆可能只是智能的一部分。
  

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255249270903275632)** (18 messages🔥): 

- **Mojo 讨论使用 MAX Graph API 进行 GPU 编程**：Brad Larson 提到了在 Mojo 中使用 MAX Graph API 针对 GPU 进行开发，该 API 允许用户构建和优化计算图。他解释道：*"自定义操作可以用 Mojo 编写，并暂存在该图中。"*

- **Mojo 24.4.0 中的类型检查 Bug**：Brooklyn_marc 报告了一个潜在的 Bug，即返回 `String` 的函数被错误地接受为 `Float`。Roboquant 和 carlcaulkett 确认较新的 nightly 版本会正确抛出错误，carlcaulkett 分享了具体的错误输出来说明该问题。

- **变量重新赋值和类型检查的奇特行为**：Darinsimmons 对报告的 Bug 进行了实验，并指出 nightly 版本在处理类型检查方面更加健壮。他评论了编译器内部赋值和类型检查的动态过程，怀疑这是否是一个操作顺序（order of operations）问题。

- **社区鼓励报告问题**：尽管 brooklyn_marc 有所顾虑，darinsimmons 和 soracc 仍鼓励报告潜在问题，即使这些问题在 nightly 构建中似乎已修复。*"如果你认为这是一个问题，请随时提出……他们一直鼓励在 GitHub 上提交问题。"*
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255244562297913487)** (64 messages🔥🔥): 

- **发现布尔表达式问题**：一位用户发现了 Mojo 在编译时处理某些布尔表达式的问题，并提到注释掉 `@parameter` 并移除 `not` 或使用 `var` 可以解决该问题。他们链接了一个可能与此问题相关的 [特定 commit](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5)。

- **Nightly 编译器更新发布**：Mojo 编译器的两个新 nightly 构建版本已发布。用户被告知更新到 `2024.6.2605` 和 `2024.6.2614`，并附带了 [raw diffs](https://github.com/modularml/mojo/compare/6961ce560d0457689f8667986b94a7ea02940cea...0ce792c1024386c17e8ced3d6cf4a70ec7113cc6) 和 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 的链接。

- **无符号整数转换 Bug**：关于 Mojo 中无符号整数转换像有符号整数一样溢出的 [问题](https://github.com/modularml/mojo/issues/3065) 正在讨论中。用户正在推测 `var` 和 `alias` 处理方式的行为及潜在 Bug。

- **List 和编译时求值 Bug**：有一份 [Bug 报告](https://github.com.modularml.mojo/issues/3126) 指出 `List` 在 Mojo 的编译时无法正常工作。此问题是另一个已报告的 `Tensor` 编译时问题的补充，后者导致连续运行期间结果不一致。

- **为引用使用静态生命周期**：引入了 `ImmutableStaticLifetime` 的概念，允许用户获取 `alias` 项的引用，这在以前由于生命周期问题是很困难的。这一增强功能类似于使用 `let`，并承诺能更好地管理静态项。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/compare/6961ce560d0457689f8667986b94a7ea02940cea...7c00fc9a5a3171531da871f7fc3925f960bd8d31">Comparing 6961ce560d0457689f8667986b94a7ea02940cea...7c00fc9a5a3171531da871f7fc3925f960bd8d31 · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/commit/6961ce560d0457689f8667986b94a7ea02940cea">[stdlib] Bump compiler version to 2024.6.2516 · modularml/mojo@6961ce5</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug 描述。在 Discord 进行了一些讨论后将其迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同情况下有不同的行为...</li><li><a href="https://github.com/modularml/mojo/issues/3098">[BUG] `Tensor` initialised from a list with wrong type shows weird behaviour · Issue #3098 · modularml/mojo</a>: Bug 描述。具体来说，使用 `List[UIn8]` 初始化的 `Tensor[DType.int8]` 无法正确计算其元素总数。我认为这又是与隐式转换有关...</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` doesn&#39;t work at compile time. · Issue #3126 · modularml/mojo</a>: Bug 描述。如题。至少 `List.__getitem__` 不起作用。复现步骤：`fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # prints 0`。系统信息：Mojo 2024.6.2614 (366c690a) o...</li><li><a href="https://github.com/modular">Modular Inc</a>: Modular 是一个集成的、可组合的工具套件，旨在简化您的 AI 基础设施，让您的团队能够更快地开发、部署和创新。 - Modular Inc</li><li><a href="https://github.com/modularml/mojo/issues/1405">[Modular CLI]: modular install mojo should support version pinning · Issue #1405 · modularml/mojo</a>: 问题描述。我无法弄清楚如何安装特定版本的 Mojo。这对于库维护者和 CI/CD 系统来说非常有用，甚至是必不可少的。复现步骤：`$ modular inst...`</li><li><a href="https://github.com/modularml/mojo/issues/3065#issuecomment-2173567566**">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug 描述。在 Discord 进行了一些讨论后将其迁移至此。似乎转换为无符号整数实际上只是转换为有符号整数，但在不同情况下有不同的行为...</li><li><a href="https://github.com/modularml/mojo/pull/2847">[stdlib] List __getitem__ returns auto-dereferenced ref by mikowals · Pull Request #2847 · modularml/mojo</a>: 通过此更改，`List.__getitem__()` 在返回值时不再进行拷贝。我还添加了一个测试，证明使用语法糖 `my_list[0].value = 1` 设置单个字段不再产生额外的...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255585372436561984)** (14 messages🔥): 

- **Clement Delangue 宣布使用 300 块 H100 的新 LLM 排行榜**：Clement Delangue 宣布了一个[新的开源 LLM 排行榜](https://x.com/ClementDelangue/status/1805989925080219927)，其中使用了 300 块 H100 GPU 重新运行了 MMLU-pro 等评估。关键结论包括 Qwen 72B 占据主导地位、评估标准过时、关注主要评估而非其他，以及意识到更大的模型并不总是更聪明。

- **对 Delangue 使用 300 块 H100 的反应**：Nathan Lambert 幽默地对这一努力表示不屑，称“300 块 H100 实在太少了”，并认为企业 CEO 对此吹嘘很“尴尬（cringe）”。

- **社区以幽默回应**：xeophon. 等成员开玩笑说他们的大学也需要 300 块 H100，并为此准备发一条 LinkedIn 动态；而 sumo43 则讽刺地将“消耗（burned）” 300 块 H100 解释为献给 Jensen 的祭品。

- **尽管存在质疑，排行榜仍获得支持**：尽管对这一发布方式有所批评，Nathan Lambert 还是对新的排行榜表示了支持，称其“很不错（nice）”。

- **企业虚张声势遭到批评**：社区普遍批评了这次发布中表现出的虚张声势，xeophon. 提到这如何利用了“弱者迷因/博弈（underdog meme/game）”，但也承认这并不奏效。

**提到的链接**：<a href="https://x.com/ClementDelangue/status/1805989925080219927">来自 clem 🤗 (@ClementDelangue) 的推文</a>：非常激动地宣布全新的开源 LLM 排行榜。我们消耗了 300 块 H100 来为所有主流开源 LLM 重新运行 MMLU-pro 等新评估！一些发现：- Qwen 72B 是王者，中国开源模型...

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1255268298577154152)** (28 messages🔥): 

- **RabbitCode API 密钥漏洞曝光**：Rabbitude 团队在 2024 年 5 月 16 日发现了 Rabbit 代码库中存在关键的**硬编码 API 密钥**，危及了 [ElevenLabs](https://elevenlabs.io)、[Azure](https://azure.com)、[Yelp](https://yelp.com) 和 [Google Maps](https://maps.google.com) 等服务。这些密钥可能允许未经授权的更改和对敏感数据的访问。
- **ElevenLabs 积分可能被利用**：讨论围绕被泄露密钥中的 ElevenLabs 积分是否可以被使用展开，一名成员评论道“这只是 VC 的钱”，所以不是真钱。
- **OpenAI 推迟高级 Voice Mode**：根据 [OpenAI](https://x.com/openai/status/1805716393524183136?s=46) 的消息，**ChatGPT 的高级 Voice Mode** 面向所有 Plus 订阅用户的推出已推迟至秋季。此次推迟是为了改进内容检测和用户体验。
- **HF 排行榜变更影响 7B 模型 merge**：[Open LLM Leaderboard 的变更](https://x.com/sebastianb929/status/1805996999499514233?s=46)导致 7B 模型 merge 的评分下降最为显著，引发了社区的反应。
- **Udio 讨论 AI 在音乐中的变革作用**：在[一份详细声明](https://x.com/udiomusic/status/1805694761891778783?s=46)中，Udio 强调了 AI 赋能艺术家的潜力，尽管 [RIAA](https://x.com/riaa/status/1805739691972452559?s=46) 对此表示担忧。他们预测 AI 将成为音乐创作和行业增长不可或缺的一部分。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/riaa/status/1805739691972452559?s=46">来自 RIAA (@RIAA) 的推文</a>: @RIAA 对 @udiomusic 的回应 ⬇️</li><li><a href="https://rabbitu.de/articles/security-disclosure-1">rabbit 数据泄露：所有曾给出的 r1 响应均可下载 - rabbitude</a>: rabbit inc 已经知道我们掌握了他们的 elevenlabs (tts) api 密钥一个月了，但他们没有采取任何行动来轮换 api 密钥。</li><li><a href="https://x.com/sebastianb929/status/1805996999499514233?s=46">来自 SebastianBoo (@SebastianB929) 的推文</a>: 谁能想到…… 7B 模型 merge 掉分最严重 😅</li><li><a href="https://x.com/techmeme/status/1805951837713088643">来自 Techmeme (@Techmeme) 的推文</a>: 消息人士：YouTube 正在与索尼、华纳和环球洽谈，为其模仿流行歌手的 AI 音乐生成工具获取歌曲授权（金融时报） https://on.ft.com/3L0zaEQ 📫 订阅...</li><li><a href="https://x.com/nathanbenaich/status/1805360586420895770">来自 Nathan Benaich (@nathanbenaich) 的推文</a>: 这两起音乐生成式 AI 诉讼值得一读，内容劲爆且极具启发性</li><li><a href="https://x.com/udiomusic/status/1805694761891778783?s=46">来自 udio (@udiomusic) 的推文</a>: 今天，我们想分享一些关于 AI 和音乐未来的想法。在过去的两年里，AI 已成为跨多种媒介（从文本到图像再到电影）进行创意表达的强大工具...</li><li><a href="https://x.com/openai/status/1805716393524183136?s=46">来自 OpenAI (@OpenAI) 的推文</a>: 我们正在分享关于我们在春季更新中演示的高级 Voice Mode 的最新进展，我们对此依然感到非常兴奋：我们原计划开始向一小部分 Ch... 用户推出 Alpha 测试。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255269708316541008)** (69 条消息🔥🔥): 

- **对 Imbue 2 亿美元融资的质疑**：成员们讨论了对 **Imbue** 最近成功的怀疑，质疑其缺乏过往记录和突然的融资。一位成员提到在他们那里的糟糕面试经历，而其他人则指出他们现在似乎走上了更好的轨道。
  
- **围绕 CARBs 和 Scale 的热度**：人们对 CARBs 的发布感到兴奋，一位成员提到之前曾尝试实现它。聊天随后转向讨论 **Scale AI** 的策略，包括 Scale 使用 **remotasks.com** 等子公司进行数据标注工作，以潜在地将品牌感知与客户隔离开来。
  
- **Scale AI 雇佣博士进行远程工作**：讨论包括 **Scale AI** 为项目协作空运承包商，其中一些人正在为 **Alphabet** 的 **Bard** 等公司的 AI 项目工作。一位成员指出，出于各种战略原因使用子公司，使其做法与 **surgehq.ai** 和 **Dynata** 等竞争对手保持一致。
  
- **对 Gemma V2 模型的兴奋**：成员们对 **Gemma V2** 模型表现出热情，讨论了其含糊的命名惯例，并赞赏其立即开放的态度。一位成员强调了一篇间接揭露内部细节的分享文章，产生了大量的注册。
  
- **AI2 的 AI 研究实验室动态**：有一个关于教经理 System Prompts 的有趣轶事，将其高水平的专业能力与基础知识进行了对比。讨论转向内部办公室动态，并批评他们目前的模型输出因为缺乏 System Prompts 而表现平平。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/minjunesh/status/1805691167037919536">来自 minjune (@minjunesh) 的推文</a>: 你在开玩笑吗 Claude？这是 1984 级别的信息封锁</li><li><a href="https://fxtwitter.com/iamkrishnasoham/status/1805937316164198560?s=46">来自 krishna soham (@iamkrishnasoham) 的推文</a>: im-also-a-late-june-chatbot  抄送: @apples_jimmy @jeremyphoward</li><li><a href="https://x.com/decompwlj/status/1805961700522291222?s=46">来自 Rémi Eismann (@decompwlj) 的推文</a>: @TheXeophon 这似乎是一个 Gemma V2 模型: https://reddit.com/r/LocalLLaMA/comments/1dovvbd/gemma_v2_in_the_lmsys_arena/</li><li><a href="https://outlier.ai/">Outlier</a>: 用您的专业知识完善下一代 AI。 
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255242721212502077)** (11 条消息🔥): 

- **诱人的推文引发辩论**：关于是否发布某些推文的对话，有人评论道：“我应该发这条推文吗，哈哈”。另一位成员表示希望避免“直接攻击或嘲讽他人”，并表示更安全的选择是将其发布在他们的“秘密 Discord 好友”群组中。
- **额外库存请求**：一位成员幽默地提到他们正在市场上寻找稀缺物品，说：“如果他买了一个用来穿，另一个用来收藏，请告诉他我也在市场上 😏🫴💳”。
  


---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255239693776453742)** (121 messages🔥🔥): 

- **对 Stability AI 开源模型的担忧**：成员们表示担心，除非 Stability AI 在更新其许可证的同时修复并解除 SD3 的审查，否则从长远来看，再多的新投资者也救不了这家公司。一位成员补充道：*“整天坐着制作色情二次元角色和 deep fakes 对生成式 AI 社区没有任何实际好处……”*，这表明了对现实世界实用性的需求。

- **GPU 成本对比**：成员们讨论了在 Runpod 和 Vast 等平台上的 GPU 租赁成本，强调目前在 Vast 上运行 3090 更便宜。一位成员指出：*“在 runpod 上运行每小时真的只要 30 美分。”*

- **开源与企业利益之争**：聊天内容在倡导开源哲学与企业利益之间摇摆。一位成员认为：*“你需要社区来推动开源哲学，”* 而另一位成员则反驳说，Linux 的成功主要归功于企业和公司的支持。

- **新机配置与硬件建议**：一位用户寻求构建合适 SD 配置的建议，讨论倾向于选择 Nvidia 4090，因为它具有性能优势。有人建议：*“买 2 个 4090 可能比买一个 48G 显存的 GPU 更划算，”* 这是一个具有成本效益的选择。

- **ICQ 关闭引起关注**：随着曾经流行的即时通讯服务 ICQ 关闭，成员们开始怀旧。*“噢，ICQ 今天关闭了…… R-I-P，”* 一位成员评论道，引发了一场怀旧讨论。

- **运行 SDXL 的问题**：用户报告了在硬件上运行 SDXL 的困难，提到了 *“cuda out of memory”* 错误，特别是在显存（VRAM）有限的机器上。用户们寻求关于合适的命令行参数和优化的建议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/the-office-no-angry-steve-carell-michael-scott-gif-5606969">The Office No GIF - The Office No Angry - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://opendata.blender.org/">Blender - Open Data</a>：Blender Open Data 是一个收集、展示和查询硬件及软件性能测试结果的平台，由公众提供数据。
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1255578706479550690)** (5 messages): 

- **宠物通灵 App 演示引发关注**：一位成员发布的 [meme](https://x.com/eggwens/status/1806016129875476886) 无意中展示了一个 **Pet Psychic Scheduler app** 的实时演示，其功能包括预约宠物的通灵阅读和查看每日情绪预测。另一位成员幽默地询问该应用是否真实存在，提到他们的狗需要看星座运势。

**提到的链接**：<a href="https://x.com/eggwens/status/1806016129875476886">Egg (@eggwens) 的推文</a>：这是宠物通灵的实时演示，附带用 React 编写的示例代码和样式：通过 Pet Psychic Scheduler，你可以：🔮 为你的宠物预约通灵阅读 ✨ 查看每日情绪预测...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255394755248918670)** (2 messages): 

- **Imbue AI 发布强大的 70B 模型工具包**：Imbue AI 宣布他们训练了一个针对**推理和编码**优化的 **70B 模型**，仅用 1/7 的数据就达到了 LLAMA 3 70B 的性能。他们正在发布[一个工具包](https://imbue.com/research/70b-intro/)，其中包括 **11 个 NLP 基准测试**、一个**以代码为中心的推理基准测试**，以及一个用于扩展实验的**超参数优化器**。 
- **社区对 Imbue 基础设施深度解析的反应**：关于 Imbue AI 旨在用于高容量训练的全套基础设施脚本的实用性和受众，引发了讨论。一位用户指出，虽然承认其有用性，但这些详细信息可能只对一小部分利基市场有用。

**提到的链接**：<a href="https://x.com/imbue_ai/status/1805629542914211951?s=46">Imbue (@imbue_ai) 的推文</a>：今年早些时候，我们训练了一个针对推理和编码优化的 70B 模型。尽管训练数据减少了 7 倍，该模型仍能与 LLAMA 3 70B 媲美。今天，我们将发布一个工具包来帮助其他开发者...

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255252246514372768)** (87 messages🔥🔥): 

- **新的 Prompt Engineering 工具包发布**：一位用户分享道：“我利用周末时间通过 Sonnet 3.5 开源了一个个人小项目，一个 Prompt Engineering 工具包，”并附带了他们的 [GitHub 项目](https://github.com/teknium1/Prompt-Engineering-Toolkit)链接。

- **对 Microsoft 模型感到失望**：一位用户对 Microsoft 新的原始文本数据增强模型表示不满，并分享了 [Genstruct](https://huggingface.co/spaces/davanstrien/Genstruct-7B) 的演示链接，该模型产生了与提供的上下文无关的混乱结果。

- **关于专用 AI 硬件的推测**：成员们讨论了各种高性能 AI 芯片，如 “Sohu” 等，辩论了它们的实际性能和推理潜力，并引用了 [Gergely Orosz 关于 OpenAI 对 AGI 内部预期](https://x.com/GergelyOrosz/status/1805604272614088721)的帖子。

- **新的本地项目公告**：另一位用户兴奋地分享了一个涉及使用本地 LLM 进行角色模拟的创意项目，参考了 [NousResearch 的 CharacterCodex](https://huggingface.co/datasets/NousResearch/CharacterCodex) 以及 Haystack 等工具。

- **关于模型重复和采样问题的讨论**：用户讨论了为什么经过指令微调（Instruction-tuned）的 LLM 可能会重复内容，将其归因于“缺乏重复惩罚或错误的采样设置”，资深成员确认了这些潜在问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Etched/status/1805625693113663834">Etched (@Etched) 的推文</a>：认识 Sohu，史上最快的 AI 芯片。运行 Llama 70B 时每秒处理超过 500,000 个 token，Sohu 让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 台 H100。Soh...</li><li><a href="https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about">关于</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/GergelyOrosz/status/1805604272614088721">Gergely Orosz (@GergelyOrosz) 的推文</a>：@ByrneHobart 关于 OpenAI 甚至都不太可能相信 “AGI” 即将到来的迷人观察，基于其不断变化的股权结构。基本上，OpenAI 预计会有更多员工...</li><li><a href="https://huggingface.org/spaces/davanstrien/Genstruct-7B">Genstruct 7B - Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>：通过在 GitHub 上创建账号来为 teknium1/Prompt-Engineering-Toolkit 的开发做出贡献。</li><li><a href="https://huggingface.co/posts/anakin87/427727576111455">Hugging Face 上的 @anakin87：“🌌 使用本地 LLM 创造冒险。如果 🤔... 荷马·辛普森遇到了……”</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

namayra: 我！
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1255238857339830282)** (69 messages🔥🔥): 

- **使用 .stream() 方法流式传输 LangChain 响应**：在从 `langchain_community.chat_models` 导入 `LLM` 并安装 `ollama` 后，建议使用 `.stream("query")`。该方法允许迭代 token 并逐行打印。

- **使用 Zep 的长期记忆看起来很有前景**：用户正在讨论使用 [Zep](https://www.getzep.com/) 作为 AI 长期记忆的潜力，它可以利用过去对话中的相关事实填充 Prompt。

- **在 LangChain 中为 PDF 使用 BytesIO**：一位用户正在寻求一种直接从 `BytesIO` 对象加载 PDF 文档而无需创建临时文件的方法。目前的权宜之计涉及创建临时文件，这被认为效率低下。

- **Streamlit 结合 `AgentExecutor` 和流式响应**：提供了使用 `StreamlitCallbackHandler` 在 Streamlit 应用中实时可视化 Agent 的思考和行动的说明。用户正在寻求在此设置中不使用回调处理器（Callback Handlers）来处理流式响应的方法。

- **LangSmith 追踪问题排查**：一位用户询问尽管设置了所有必需的环境变量，但 LangSmith 不再追踪其项目的问题。建议是检查追踪配额是否已耗尽。

<div class="linksMentioned">

<strong>提及的链接</strong>：

</div>

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/callbacks/streamlit/#scenario-1-using-an-agent-with-tools>).">Streamlit | 🦜️🔗 LangChain</a>: Streamlit 是一种构建和共享数据应用的更快速方式。</li><li><a href="https://www.getzep.com/">Zep - AI 助手的长期记忆</a>: 召回、理解并解析聊天对话，以提供个性化体验。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/callbacks/argilla/#scenario-3-using-an-agent-with-tools>).">Argilla | 🦜️🔗 LangChain</a>: Argilla 是一个面向 LLM 的开源数据整理平台。</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/providers/pebblo/pebblo_retrieval_qa/">使用 PebbloRetrievalQA 实现身份感知的 RAG | 🦜️🔗 LangChain</a>: PebbloRetrievalQA 是一个具有身份和语义强制执行功能的检索链，用于问答任务。</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/agents/#adding-in-memory>)">构建 Agent | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/agent_executor/#adding-in-memory>).">使用 AgentExecutor (旧版) 构建 Agent | 🦜️🔗 LangChain</a>: 本节将介绍如何使用旧版的 LangChain AgentExecutor 进行构建。这些工具适合入门，但到了一定阶段，你可能需要它们无法提供的灵活性和控制力...</li><li><a href="https://github.com/SuperDuperDB/superduperdb">GitHub - SuperDuperDB/superduperdb: 🔮 SuperDuperDB: 将 AI 引入你的数据库！直接利用现有的数据基础设施构建、部署和管理任何 AI 应用，无需移动数据。包括流式推理、可扩展的模型训练和向量搜索。</a>: 🔮 SuperDuperDB：将 AI 引入你的数据库！直接利用现有的数据基础设施构建、部署和管理任何 AI 应用，无需移动数据。包括流式推理、可扩展...</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/#streaming>)">构建聊天机器人 | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/functions/#next-steps>),">如何运行自定义函数 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#non-streaming-components>).">如何实现流式传输 | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://ai.stackexchange.com/questions/40753/how-to-generate-original-training-videos-based-on-existing-videoset">如何基于现有视频集生成原创训练视频？</a>: 我是一名正在快速学习 AI 技术的软件工程师，但对该领域仍然非常陌生。一位同事拥有大量的训练视频，垂直领域是轮椅座椅...</li><li><a href="https://github.com/langchain-ai/langchain/issues/16980>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#streaming-tokens>)">构建 Agent | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langchain/issues/12441>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/7747>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/19944>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建具备上下文感知能力的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/debugging/#set_debug-and-set_verbose>)">如何调试你的 LLM 应用 | 🦜️🔗 LangChain</a>: 与构建任何类型的软件一样，在使用 LLM 构建应用时，你总会在某些时刻需要进行调试。模型调用可能会失败，或者模型输出格式错误，或者会出现一些嵌套的模型调用...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1255465297880551464)** (1 条消息): 

- **在 GA4 中追踪来自 langserve 后端的执行链引起了兴趣**: 一位成员询问了如何使用 langserve 后端在 GA4 中追踪执行链。他们考虑过*对 Langsmith 进行子类化*，并明确表示只需要捕获第一次 invoke 或 stream，而无需跟踪任何后续步骤。
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255492691605716993)** (2 条消息): 

- **Testcontainers-Python 新增 Ollama 支持**：一位成员分享了他们对 **testcontainers-python** 的新贡献，增加了一个 **Ollama 模块** 的支持，以便在 Python 中通过 Ollama 测试 LLM 端点。你可以查看他们的 [issue](https://github.com/testcontainers/testcontainers-python/issues/617) 和 [pull request](https://github.com/testcontainers/testcontainers-python/pull/618) 了解更多详情并提供反馈。

- **关于 Tool Calling 的 Few-Shot Prompting Medium 文章**：一位成员分享了一篇 [Medium 文章](https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1)，讨论了 LangChain 中结合 **tool calling** 的 **few-shot prompting**。文章提供了实现该方法的见解和方案。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/testcontainers/testcontainers-python/issues/617">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>: 添加对 OllamaContainer 的支持，以简化通过 Ollama 运行和测试 LLM 的过程。你想要什么样的新容器？我希望请求支持一个新容器：OllamaCo...</li><li><a href="https://github.com/testcontainers/testcontainers-python/pull/618">feat(core): Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>: 添加了一个新的 OllamaContainer 类，包含几个处理 Ollama 容器的方法。_check_and_add_gpu_capabilities 方法检查主机是否有 GPU，并向其添加必要的 capabilities...</li><li><a href="https://medium.com/ai-advances/few-shot-prompting-with-tool-calling-in-langchain-19db16b39fe1">Few-Shot Prompting with Tool Calling in Langchain</a>: Ankush k Singal
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1255531227516244051)** (1 条消息): 

- **ARC AGI 挑战视频分享**：分享了一段名为“[Claude 3.5 struggle too?! The $Million dollar challenge](https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap)”的 YouTube 视频。该视频提供了如何使用 Agent 完成 ARC AGI 挑战的教程，并包含一份关于 AI 数据分析项目的免费 HubSpot 报告链接。

**提及的链接**: <a href="https://youtu.be/kZap-tDA1i4?si=HyF9dnOm9VRY3_Ap">Claude 3.5 struggle too?! The $Million dollar challenge</a>: 百万美元 ARC AGI 挑战。获取如何进行 AI 数据分析项目的免费 HubSpot 报告：https://clickhubspot.com/d30🔗 链接- 在 twitter 上关注我...

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255298940090187776)** (37 条消息🔥): 

- **LlamaIndex 聊天机器人开发问题**：一位用户在构建 [使用 LlamaIndex 的聊天机器人](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/) 时，询问如何直接从聊天响应中获取上下文，而不是从单个查询结果中获取。他们提供了具体的实现细节和面临的挑战。

- **GitHub PR 审查讨论**：一位成员就合并一个为 [LlamaIndex 中的 Neo4J 数据库](https://github.com/run-llama/llama_index/pull/14362) 添加查询过滤功能的 PR 寻求建议。另一位成员表示目前积压了较多审查任务，但会尽快处理。

- **ML 库通知问题**：一位用户询问在使用 `Openailike` 类时如何移除关于缺失 ML 库的通知。回复澄清了这并非错误，并指出了该消息的具体来源。

- **针对 SQL 查询微调 LLM**：用户讨论了在使用 RAG SQL 层时，通过微调 LLM 来提高查询精度的潜力。建议在优质数据上进行微调可能会获得更好的性能。

- **使用 LlamaIndex 进行混合搜索 (Hybrid search)**：有人咨询如何通过平衡元数据和文本块的影响来实现混合搜索，对此，一份详细的回复概述了使用 `alpha` 参数来配置搜索权重的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/Emerging-AI/ENOVA">GitHub - Emerging-AI/ENOVA: A deployment, monitoring and autoscaling service towards serverless LLM serving.</a>：一个面向 serverless LLM 服务的部署、监控和自动扩缩容服务。- Emerging-AI/ENOVA</li><li><a href="https://github.com/run-llama/llama_index/pull/14362">Add MetadataFilters to neo4j_property_graph by theoneamendez · Pull Request #14362 · run-llama/llama_index</a>：描述：请包含更改摘要以及修复了哪个问题。还请包含相关的动机和背景。列出此更改所需的任何依赖项。摘要...</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/">How to Build a Chatbot - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/indices/keyword/#llama_index.core.indices.SimpleKeywordTableIndex>)">Keyword - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/json.py#L51">llama_index/llama-index-core/llama_index/core/readers/json.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/readers/file/base.py#L69">llama_index/llama-index-core/llama_index/core/readers/file/base.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/?h=table+parser#define-expanded-query-pipeline">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/#setup-simple-retry-agent-pipeline-for-text-to-sql">Building an Agent around a Query Pipeline - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1255296974819950632)** (2 条消息): 

- **使用 LlamaIndex 和 DSPy 优化 RAG 系统**：一位成员分享了一篇关于利用 LlamaIndex 和 DSPy 构建优化的检索增强生成 (RAG) 系统的 [Medium 文章](https://medium.com/ai-advances/building-optimized-retrieval-augmented-generation-rag-systems-with-llamaindex-and-dspy-cacaf7f7089f)。该文章详细介绍了实现稳健 RAG 实施的实际步骤和见解。

- **开源项目寻求反馈并提供奖励**：另一位成员介绍了一个 [GitHub 上的开源项目](https://github.com/Emerging-AI/ENOVA)，旨在增强 serverless LLM 服务的 AI 部署、监控和自动扩缩容服务。他们正在寻求反馈，并提供 50 美元的礼品卡以换取在线访谈来分享见解。

**提到的链接**：<a href="https://github.com/Emerging-AI/ENOVA">GitHub - Emerging-AI/ENOVA: A deployment, monitoring and autoscaling service towards serverless LLM serving.</a>：一个面向 serverless LLM 服务的部署、监控和自动扩缩容服务。- Emerging-AI/ENOVA

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1255247980051824650)** (9 messages🔥): 

- **模型混淆已解决：是 Claude-3.5-Sonnet**：关于 **Anthropic 新模型的名称** 存在一些混淆。现已明确为 `claude-3-5-sonnet-20240620`。

- **讨论基于 MoonDream 的本地视觉模型**：成员们讨论了 **OI** 是否拥有基于 **MoonDream** 的本地视觉模型，但指出目前该模型还无法在 OI 中使用。

- **多行输入问题**：一名成员在使用 `-ml` 选项通过 `'''` 进行多行输入时遇到了问题。

- **视觉错误担忧**：另一名成员在尝试使用 `interpreter --os --vision` 识别屏幕内容时遇到错误，尽管他们已经验证了其 **API key**。

- **解释器中的文件拖放限制**：有人提出了关于直接将文档或图像移入终端的限制问题，一名成员表示在尝试这样做时遭到了 **ban**。
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1255273144059428894)** (17 messages🔥): 

- **01 是 OpenInterpreter 的语音界面**：一位用户确认 01 是 OpenInterpreter (**OI**) 的语音界面，解决了另一位成员关于 01 与 OI 之间关系的困惑。
- **01 未在西班牙发售**：一位成员表示有兴趣在西班牙购买 01，但被告知该产品仅在美国销售。他们被引导至一个 GitHub 仓库，以便使用 [开源开发套件 (open-source dev kit)](https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight) 自行构建。
- **网上已有 DIY 01 教程**：另一位用户确认 YouTube 上已有基于开源套件构建 01 的教程，并计划在 7 月创建一个新教程。
- **设置语音功能的挑战**：成员们讨论了使用 01 设置语音功能的困难和细节，包括在 Macbook Pro M1 上集成西班牙语的 TTS 和 STT，以及将语音发送到 ngrok websocket。

**提到的链接**：<a href="https://github.com/OpenInterpreter/01/tree/main/hardware%2Flight">01/hardware/light at main · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255267975372734475)** (16 messages🔥): 

- **咨询学者计划 (Scholars Program)**：一位成员最近询问 **学者计划** 今年是否运行。目前没有关于此话题的进一步讨论。

- **Preamble Tokens 计费讨论**：一位成员详细介绍了一个关于 API 调用中 **preamble tokens** 计费的实验，强调“*preamble 是计费的*”。他们提到了一种情况，即如果 16k token 的 preamble 不计费，理论上可以免费使用。

- **1Bit LLMs 时代讲座活动**：宣布由 **Hongyu Wang** 主讲的题为 *The Era of 1Bit LLMs* 的讲座。参与者受邀通过 [Google Meet 链接](https://meet.google.com/yhv-tiir-ava) 加入讲座。

- **Websim.ai 网页模拟**：成员们喜欢尝试 [Websim.ai](https://websim.ai/c/0NUkL2gMKZefC1AoZ)，它可以根据 URL 模拟预测的网页。它使用 Anthropic 的 Claude 来创建 artifacts 并模拟个人口袋互联网。

- **举报 Command-R 的商业滥用**：一位成员对一家名为 **SpicyChat AI** 的 NSFW 机器人托管服务使用 Command-R 营利表示担忧。他们强调该服务的所有者声称使用 **OpenRouter** 就可以无视 Cohere 的 **CC-BY-NA** 许可证。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://websim.ai/c/0NUkL2gMKZefC1AoZ]">...</a>：未找到描述</li><li><a href="https://rhea.run)">无标题</a>：未找到描述</li><li><a href="https://websim.ai/c/0NUkL2gMKZefC1AoZ">Run From Dinosaurs - Crash Bandicoot Style</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1255618654142202156)** (2 messages): 

- **宣布 Rig Rust 库发布**：一位成员分享了 **Rig** 的发布更新，这是一个用于构建 LLM 驱动应用程序的 Rust 库。他们正在运行一个激励性反馈计划，开发者通过构建用例并提供关于该库的反馈来获得奖励。
- **反馈计划符合频道主题**：另一位成员确认在该频道发布关于反馈计划的内容是合适的。他们幽默地提到，该库显然应该支持 **Cohere 的模型**。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255268720448635040)** (11 条消息🔥): 

- **Adam-mini 旨在优化内存使用**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/abs/2406.16793)，提出了一种名为 Adam-mini 的优化器。该优化器在性能上与 AdamW 相当甚至更好，同时内存占用减少了 45% 到 50%。该优化器通过减少 Adam 中的学习率数量来降低内存消耗，其灵感来自 Transformer 的 Hessian 结构，对参数块使用单一学习率。
- **询问训练期间屏蔽输出文本的方法**：一位成员询问是否有办法屏蔽某些“输出”文本，类似于 `train_on_input`，并建议增加一个类似 `do_not_train_on_output_marked_as_masked` 的功能。
- **关于梯度累积对训练时间影响的讨论**：多位成员讨论了增加梯度累积（GA）次数（例如累积 100 次）是否会影响每个 epoch 的总训练时间。一位成员认为这可能会更快，因为优化器运行次数减少，可能减少参数吸收的噪声；而另一位成员则认为高 GA 会降低每步（per step）的性能。
- **训练过程中的 CUDA 错误问题**：一位成员分享了一个与 CUDA 非法内存访问相关的错误，并建议了调试步骤，如使用 `CUDA_LAUNCH_BLOCKING=1` 或在编译时使用 `TORCH_USE_CUDA_DSA` 以启用设备端断言（device-side assertions）。

**提到的链接**：<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，其性能与 AdamW 相当或更好，但内存占用减少了 45% 到 50%。Adam-mini 通过减少学习率的数量来降低内存...

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255240413720215695)** (4 条消息): 

- **在 HF 上创建带有自定义最小学习率的余弦学习率调度器**：一位成员询问如何在 Hugging Face 上轻松创建一个最小学习率大于零的余弦学习率调度器（cosine learning rate scheduler）。这指向了对 HF 库进行实际实现微调的可能性。

- **PEFT 中启用 QDora**：Caseus 提到一个在 PEFT 中启用 QDora 的 pull request，并承诺会跟进。这引起了另一位愿意投入大量精力使其工作的成员的兴趣。

- **Mistral7B 重复生成问题**：一位用户报告称，他们经过全量指令微调的 Mistral7B 模型即使在高温度（temperature）设置下也会重复生成句子或段落。他们指出其数据集并不包含此类重复内容，并寻求原因和解决方案。

---

### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1255484851180671027)** (1 条消息): 

- **Storiagl：利用 LLM 免费创作故事**：一个令人兴奋的新平台 [Storiagl](https://storiagl.web.app/)，允许用户利用自定义 LLM 进行角色演绎来创作和游玩故事。它提供了高级设置来构建复杂且详尽的叙事。

**提到的链接**：<a href="https://storiagl.web.app/">StorIA</a>：未找到描述内容。

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255328754041880628)** (6 条消息): 

- **仅凭一本书教 LLM 学习 Kalamang 语**：荷兰博士生 Eline Visser 撰写了《Kalamang 语语法》，这是该语言唯一的文本。研究人员利用这本书来观察语言模型是否可以通过各种类型的 Fine-tuning 和 Prompting 学习 Kalamang 语。有趣的是，在这项实验中，*"Prompting 胜出（且优势巨大）"*，尽管人类在这一任务上的表现仍然优于 LLM。[详情](https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ)；[摘要](https://arxiv.org/abs/2309.16575)。

- **AI Engineer World’s Fair 2024 正在直播**：专注于 Keynotes & CodeGen 赛道的“AI Engineer World’s Fair 2024”目前正在 YouTube 上进行直播。[点击观看](https://www.youtube.com/watch?v=5zE2sMka620)；更多详情可通过该活动的 [Twitter 描述](https://twitter.com/aidotengineer)获取。

- **2024 年 6 月 Build with Claude 竞赛**：2024 年 6 月的“Build with Claude”竞赛已公布，为参与者提供了展示其利用 Claude 开发能力的机会。更多详情请参阅[官方竞赛概览](https://docs.anthropic.com/en/build-with-claude-contest/overview)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/jxmnop/status/1805756434824806499?s=46&t=lR4AowAEET_5VqponFnfGQ">来自 jack morris (@jxmnop) 的推文</a>：最近读到了我读过最有趣的 LLM 论文之一，故事大概是这样的：荷兰博士生/研究员 Eline Visser 在印度尼西亚的一个偏远岛屿生活了几年...</li><li><a href="https://arxiv.org/abs/2309.16575">从一本语法书学习翻译新语言的基准测试</a>：大型语言模型（LLM）可以通过 In-context learning 或轻量级 Fine-tuning 实现令人惊叹的壮举。人们自然会好奇这些模型对全新任务的适应能力如何，但该如何...</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen 赛道</a>：https://twitter.com/aidotengineer</li><li><a href="https://docs.anthropic.com/en/build-with-claude-contest/overview">Anthropic 2024 年 6 月 Build with Claude 竞赛</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1255290715492323431)** (1 条消息): 

- **针对额度问题请求邮件验证**：一名成员针对额度表单问题提供了帮助，建议另一名成员私信（DM）他们在表单中使用的电子邮件地址。*"随时私信我，我可以查看一下 —— 请告诉我你在额度表单中使用了哪个邮箱。"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1255496809317400617)** (3 条消息): 

- **在 Offloading 方面 DS 优于 FSDP**：*"这可能是因为 DS 相比 FSDP 具有更细粒度的 Offloading，前提是确实发生了 Offloading"*。
- **缺乏 DS 和 FSDP 的经验**：一位成员提到，*"我还没有使用过它们"*。
- **探索 LLama 70B 设置**：一位成员分享说他们想尝试 LLama 70B，但承认需要进一步了解相关设置。
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1255588743599882260)** (1 条消息): 

- **Builders Program 早期截止日期提醒**：发布了关于 Builders Program 的提醒，敦促成员在 **7 月 8 日** 早期申请截止日期前提交申请。更多信息和申请方式请见[此处](https://future.mozilla.org/builders/)。

- **提供问题咨询与支持**：对于任何与 Builders Program 相关的问题，成员可以通过此 [Discord 频道](https://discord.com/channels/1089876418936180786/1245083732319408195)获得支持。
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1255492503646244945)** (9 messages🔥): 

- **Firefox 将 llamafile 集成打造为怀旧的 Web 冒险**：通过将 llamafile 用作 HTTP proxy，**Firefox** 可以探索 LLM 权重中的知识，创造出一种让人联想起 90 年代的 Web 体验。查看演示此集成的 [YouTube 视频](https://youtu.be/YWQ5Kh9gNuo)。

- **使用 llamafile 和 Character Codex 进行聊天冒险**：一位成员分享了关于使用 llamafile、Haystack 和 Character Codex *从零开始创建聊天冒险* 的详细 notebook。访问 [此处](https://t.ly/y6jrZ) 的 notebook，尝试荷马·辛普森（Homer Simpson）遇见蜘蛛侠（Spider-Man）等场景。

- **Jupyter notebook 中的 CUDA 警告**：有一场关于在类 Jupyter 环境中处理 CUDA 警告以保持 notebook 整洁的讨论。建议的解决方案涉及使用 [来自 Haystack 的工具](https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py) 来检测程序是否在类似环境中运行。

- **AI 新闻背景下的 NVIDIA 股价波动**：一条推文强调了在 AIEWF 演讲后 NVIDIA 市值大幅缩水，[MarketWatch](https://www.marketwatch.com/story/nvidias-stock-is-set-to-gain-with-rivals-seen-to-be-in-perpetual-catch-up-mode-0552e514) 和 [Barrons](https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c) 的分析对影响股价的因素给出了截然不同的讨论。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/aiDotEngineer/status/1806012306046046232?t=BudJTKP1KpdJcSNGzihIYg&s=19">来自 AI Engineer (@aiDotEngineer) 的推文</a>：突发新闻：在 @JustineTunney 和 @stlhood 在 AIEWF 演讲后，$NVDA 市值蒸发了 560 亿美元。引用 swyx 👉 ai.engineer (@swyx)：MOZILLA 回归了，太不可思议了，MOZILLA AI 可能正是他们的...</li><li><a href="https://youtu.be/YWQ5Kh9gNuo">llm 浏览器演示</a>：未找到描述</li><li><a href="https://github.com/deepset-ai/haystack/blob/main/haystack/utils/jupyter.py">haystack/haystack/utils/jupyter.py (GitHub)</a>：🔍 用于构建可定制、生产级 LLM 应用的 LLM 编排框架。将组件（模型、向量数据库、文件转换器）连接到可以与你交互的 pipeline 或 Agent...</li><li><a href="https://t.ly/y6jrZ">haystack-cookbook/notebooks/charactercodex_llamafile.ipynb (GitHub)</a>：👩🏻‍🍳 示例 notebook 集合。通过在 GitHub 上创建账号为 deepset-ai/haystack-cookbook 的开发做出贡献。</li><li><a href="https://www.barrons.com/amp/articles/nvidia-shareholder-meeting-stock-price-today-6d01b66c">股东大会结束，Nvidia 股价仍在下跌</a>：在简短的年度会议上，Nvidia 股东批准了所有 12 名推荐的公司董事会提名人。
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1255308575144677426)** (7 messages): 

- **讨论 tinygrad 的 FPGA backend**：成员们讨论了 **tinygrad** 拥有 FPGA backend 的可能性。George Hotz 建议，设计一个运行在 FPGA 上的 **accelerator** 然后以此为目标可能更实际。

- **Positron 旨在实现 Transformer 推理**：一些 **Groq 工程师** 离职创办了 [Positron](https://www.positron.ai/)，旨在提供 Atlas Transformer 推理服务器和 Redstone 开发者工作站等硬件解决方案，声称每美元性能比 DGX-H100 高出 **10 倍**。

- **FPGA 专业化与 HDL**：成员们提到了配备 DSP blocks 和 HBM 的新型 FPGA 平台，它们可以通过生成特定的 HDL 来实现模型专业化，尽管 **trsohmers** 澄清说 Positron 并没有使用 Xilinx/AMD 的 FPGA，且其设计对所有 Transformer 模型都是通用的。

- **分享 PyTorch 纪录片**：分享了一个名为《官方 PyTorch 纪录片：助力 AI 革命》的 [YouTube 纪录片](https://www.youtube.com/watch?v=rgP_LBtaUEc) 链接，深入介绍了 PyTorch 的起源及其对 AI 领域的影响。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.positron.ai/">Positron | 性能最强的 Transformer 推理系统</a>：Positron 制造专用硬件以加速多模态 AI。</li><li><a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">官方 PyTorch 纪录片：助力 AI 革命</a>：这部影片揭示了 PyTorch 创立的真实叙事，将其存在归功于一群推动技术创新的无名英雄...
</li>
</ul>

</div>
  

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1255604696694128782)** (4 messages): 

- **Angry.penguin 担任版主**：成员 angry.penguin 主动提出担任版主，以防止未来的垃圾信息问题，并表示 *"如果你愿意让我当版主，我可以进行设置，让这种情况以后不再发生！"* Yoko Li 迅速接受了这一提议。
- **Yoko Li 和 angry.penguin 处理垃圾信息**：在晋升为版主后，angry.penguin 报告称他们已经解决了垃圾信息问题，提到 *"以后应该不会再有垃圾信息了 😄"* 以及 *"同时也清理了现有的垃圾信息"*。
  

---



### **DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1255457292179214436)** (4 messages): 

- **新德语编码器发布**：一位成员宣布在 [Hugging Face](https://huggingface.co/aari1995/German_Semantic_V3) 上发布了 **German Semantic V3** 和 **V3b**。V3 侧重于知识密集型任务，而 V3b 则通过 Matryoshka Embeddings 和 8k 上下文长度等特性专注于性能。

- **关于 GGUF 的咨询**：另一位成员询问是否存在新编码器的 **GGUF** 格式，以及如何进一步微调 (finetune) **V3b**。回复指出目前没有可用的 GGUF，并建议使用 [UKPLab 的 GitHub 仓库](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training) 中的 sentence-transformers 脚本进行微调。

- **GGUF 格式的可行性**：该成员澄清说，编码器是可以使用 GGUF 格式的，并引用 **Ollama** 作为以该格式使用两个 embedder 的例子。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/UKPLab/sentence-transformers/tree/master/examples/training">sentence-transformers/examples/training at master · UKPLab/sentence-transformers</a>：使用 BERT 的多语言句子和图像嵌入 - UKPLab/sentence-transformers</li><li><a href="https://huggingface.co/aari1995/German_Semantic_V3">aari1995/German_Semantic_V3 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/aari1995/German_Semantic_V3b">aari1995/German_Semantic_V3b · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255238674099077120)** (2 messages): 

- **OpenRouter 推出新模型**：查看 OpenRouter, LLC 刚刚宣布的 2023 - 2024 年度新模型 [01-ai/yi-large](https://openrouter.ai/models/01-ai/yi-large)。这是他们产品线中的最新成员。

- **推荐参数标签页问题已修复**：之前模型页面的“推荐参数 (Recommended Parameters)”标签页显示的数据有误。该问题已得到解决，现在该标签页显示正确信息。

**提到的链接**：<a href="https://openrouter.ai/models/01-ai/yi-large)">Yi Large by 01-ai</a>：Yi Large 模型由 01.AI 设计，考虑了以下用例：知识搜索、数据分类、类人聊天机器人和客户服务。它以其多语言专业能力脱颖而出...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1255275992071667743)** (1 messages): 

- **GPA Saver 网站上线**：一位成员分享了他们的新网站 [GPA Saver](https://gpasaver.com/)，该网站集成了 AI 用于学术辅助。他们感谢 OpenRouter 让 **LLM 集成** 变得无缝且简单。

- **AI 驱动的学术工具**：该网站提供多种学术工具，包括助手聊天、快速测验解答器、PDF 摘要生成器、交互式白板和闪卡生成器。前 100 名用户可使用特别折扣码 **BETA**，享受约 6.3 折优惠。

**提到的链接**：<a href="https://gpasaver.com/">GPA Saver</a>：利用 AI 的力量辅助你的学习。

  

---



---



---



---



---



---



{% else %}


> 邮件中已截断完整的频道细分内容。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}