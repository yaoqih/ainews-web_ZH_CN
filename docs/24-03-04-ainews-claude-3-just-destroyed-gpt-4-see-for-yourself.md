---
companies:
- anthropic
- amazon
- google
- claude-ai
date: '2024-03-04T23:59:02.180354Z'
description: '**Anthropic** 推出的 **Claude 3** 包含三种型号：Haiku（小型，尚未发布）、Sonnet（中型，claude.ai、AWS
  和 GCP 上的默认型号）以及 Opus（大型，适用于 Claude Pro 用户）。Opus 在 GPQA 等关键基准测试中表现优于 **GPT-4**，令基准测试的作者们印象深刻。所有型号均支持**多模态**，具备先进的视觉能力，包括将一段
  2 小时的视频转化为一篇博文。Claude 3 提供了更好的对齐性、更少的拒绝回答情况，并将上下文长度扩展至 **100 万个 token**，且拥有近乎完美的召回率。Haiku
  以速度和成本效益著称，处理内容密集的学术论文只需不到三秒钟。这些模型擅长遵循复杂指令并生成 JSON 等结构化输出。安全性的改进降低了拒绝率，尽管仍有一些专家对此持有批评意见。Claude
  3 采用合成数据进行训练，并在金融、医学和哲学等特定领域评估中表现强劲。'
id: bd591bdf-2e17-43a4-942e-2384e35a4b5a
models:
- claude-3
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku
- gpt-4
original_slug: ainews-claude-3-just-destroyed-gpt-4-see-for
people:
- mmitchell
- connor-leahy
title: Claude 3 刚刚完爆了 GPT-4（不信你看）。
topics:
- multimodality
- vision
- long-context
- model-alignment
- model-evaluation
- synthetic-data
- structured-output
- instruction-following
- model-speed
- cost-efficiency
- benchmarking
- safety
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月2日至3月4日的 AI 新闻。我们为您查看了 [**356** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **22** 个 Discord 服务端（**352** 个频道，**9688** 条消息）。预计节省阅读时间（按 200wpm 计算）：**984 分钟**。

[Claude 3 发布了](https://news.ycombinator.com/item?id=39590666)！相比之下，周末的其他消息都显得无足轻重，这对于工作日的新闻简报作者来说真是太棒了。

 
![image.png](https://assets.buttondown.email/images/e5df127c-db27-4d63-b1ca-efffc631fcf4.png?w=960&fit=max)
 

**TLDR**: 

- Claude 现在提供 3 种尺寸——最小的两个（Haiku - 尚未发布，Sonnet - 默认版本，可在 claude.ai、AWS 和 GCP 上使用）速度很快（比 Claude 2 快 2 倍）、价格便宜（[成本仅为 GPT4T 的一半](https://x.com/mattshumer_/status/1764738098389225759?s=20)）且表现出色；而最大的一个（Opus，在 Claude Pro 上提供，但速度较慢且价格更高）似乎在**每一个重要的 Benchmark 上都击败了 GPT4**。有时，例如在 GPQA 中，表现要好得多，[给 GPQA Benchmark 的作者留下了深刻印象](https://x.com/idavidrein/status/1764675668175094169?s=20)。
- 它们都是多模态的，特别是 Vision 能力，并令人信服地将一段 [2 小时的 Karpathy 视频转换成了博客文章](https://x.com/mlpowered/status/1764718705991442622?s=20)。
- 更好的 Alignment——更少的错误拒绝，以及在难题上更高的准确率。
- 200k Token 上下文，最高可扩展至 1m Token，具有类似 Gemini 1.5 的完美召回率（Recall）；
  - 值得注意的是，在进行常规的 Needle in Haystack 测试时，它[察觉到自己正在接受测试](https://twitter.com/alexalbert__/status/1764722513014329620)。[Safetyists 对此反应强烈](https://x.com/NPCollapse/status/1764740710731837516?s=20)。

以下是我们的完整笔记：

- Haiku (小型, $0.25/mtok - “即将推出”), Sonnet (中型, $3/mtok - 驱动 claude.ai, 已在 Amazon Bedrock 和 Google Vertex 上线), Opus (大型 $15/mtok - 驱动 Claude Pro)
- **速度**：Haiku 是其智能类别中市场上速度最快、性价比最高的模型。**它可以在不到三秒的时间内阅读 arXiv 上一篇包含图表、信息和数据密集的论文（约 10k tokens）**。发布后，我们预计将进一步提升性能。[Sonnet 的速度是 Opus 和 Claude 2/2.1 的 2 倍](https://x.com/AnthropicAI/status/1764653835568726215?s=20)
- **视觉**：Claude 3 模型具有与其他领先模型不相上下的**复杂视觉能力**。它们可以处理各种视觉格式，包括照片、图表、图形和技术图解。
- [Opus 可以将一段 2 小时的视频转化为一篇博客文章](https://x.com/mlpowered/status/1764718705991442622?s=20)
- **长上下文和近乎完美的召回率**：Claude 3 Opus 不仅实现了近乎完美的召回率（准确率超过 99%），在某些情况下，它甚至通过识别出“针”（needle）句子似乎是由人类人工插入到原始文本中的，从而指出了评估本身的局限性。
- **更易于使用**：Claude 3 模型在遵循复杂的、多步骤指令方面表现更好。它们特别擅长遵循品牌语调和回复指南，并开发用户可以信赖的面向客户的体验。此外，Claude 3 模型在生成 JSON 等流行结构化输出格式方面表现更好，这使得在自然语言分类和情感分析等用例中指导 Claude 变得更加简单。
- **安全性**
- 更低的拒绝率 —— 这对于消除 Anthropic 的“安全主义者”形象，以及应对 2 月份 Gemini 出现的各种话题性问题非常有帮助。
- “Opus 不仅找到了那根‘针’，它还意识到插入的‘针’在‘大海捞针’中显得如此格格不入，以至于这一定是我们要测试其注意力能力而构建的人工测试。” [来自 Anthropic 提示词工程师](https://twitter.com/alexalbert__/status/1764722513014329620)
- 受到 [MMitchell](https://x.com/mmitchell_ai/status/1764739357112713267?s=20) 和 [Connor Leahy](https://x.com/NPCollapse/status/1764740710731837516?s=20) 的批评。
- **评估 (Evals)**
- [选择突出金融、医学、哲学领域的评估，而不是 MMLU/HumanEval，这是一个明智的选择](https://twitter.com/DrJimFan/status/1764719012678897738)
- [在 GPQA 上达到 59.5%](https://x.com/idavidrein/status/1764675668175094169?s=20)，远好于全才博士和 GPT4 —— GPQA 作者对此印象深刻。[论文]([arxiv.org/abs/2311.12022](https://t.co/hb4u4xXzkw))。
- **与 GPT4 的对比**
- 在[编写 Discord 机器人代码](https://twitter.com/Teknium1/status/1764746084436607010)方面击败了 GPT4。
- [在简单的晾衣服逻辑题上失败了，但 GPT4 没有](https://x.com/abacaj/status/1764698421749756317?s=20)。
- **其他评论**
- [200k 上下文，可扩展至 1m tokens](https://x.com/mattshumer_/status/1764657732727066914?s=20)
- [Haiku 在评估中接近 GPT4，但成本仅为 GPT3.5T 的一半](https://x.com/mattshumer_/status/1764738098389225759?s=20)
- [在合成数据上训练](https://x.com/Justin_Halford_/status/1764677260555034844?s=20)
- [在代码上的低损耗（lower loss）是正常的/平淡无奇的](https://twitter.com/kipperrii/status/1764673822987538622)

另外，Noah 针对下方看到的相同 Twitter 抓取数据，对 Claude 3 (Sonnet) 和 GPT4 进行了两次运行对比。我们认为 Claude 3 的总结能力要好得多。


---

**目录**

[TOC] 


---

# 第 X 部分：AI Twitter

> 比较 [Claude 3](https://github.com/smol-ai/SocialPipelines/blob/summary-prod/data_ingestion/scripts/constants/summary_27_anthropic.md) vs [GPT4T](https://github.com/smol-ai/SocialPipelines/blob/summary-prod/data_ingestion/scripts/constants/summary_27_anthropic.md)


**AI 进展与能力**

- Sam Altman 表示“[这一切以前都发生过，这一切还会再次发生](https://twitter.com/sama/status/1764178151889084558)”，并且“[飓风转得越来越快，但风眼却保持着完美的平静](https://twitter.com/sama/status/1764179620486930941)”，这可能是在暗指 AI 的飞速进展。
- Yann LeCun 认为，Google 的 Gemini 1.5 Pro 令人印象深刻，具有“[客观上比 Apple Vision Pro 更清晰的光学效果和更高对比度的图像](https://twitter.com/ylecun/status/1764005546456199345)”。然而，John Carmack [指出](https://twitter.com/ID_AA_Carmack/status/1764016979101286756)有许多变量使得这种比较并非定论。
- François Chollet [认为](https://twitter.com/fchollet/status/1763629637689893228)他 2023 年对 LLM 能力的看法高估了它们的潜力和实用性。他概述了 LLM 可以实现的[四个泛化层级](https://twitter.com/fchollet/status/1763692655408779455)，其中通用智能是合成新程序以解决从未见过的任务的能力。
- Google 的 [Gemma](https://twitter.com/AravSrinivas/status/1764063557841469758) 能够在旧金山的野外环境中针对现实任务进行 zero-shot 部署，无需任何强化学习，仅通过对模拟和 YouTube 数据的 next-token prediction 即可实现。

**AI 投资与商业**

- 软银在 2019 年以 36 亿美元卖掉了所有 Nvidia 股份，这些股份在[今天价值 930 亿美元](https://twitter.com/nearcyan/status/1763692669698748478)。投资 AI 曾是软银愿景基金（Vision Fund）的主要目标之一。
- Nvidia 的早期阶段涉及在竞争对手拥有优势的情况下[坚持不懈地改进](https://twitter.com/ID_AA_Carmack/status/1763949929611796919)。他们的差异化优势在于更认真地对待软件，构建了 CUDA 生态系统。
- Google 面临着来自 [OpenAI 和 Perplexity](https://twitter.com/AravSrinivas/status/1763666653748105404) 等公司的挑战，它们表明许多“搜索”任务通过对话式 AI 能得到更好的服务，就像 25 年前 Google 凭借 PageRank 和链接颠覆行业一样。
- Alexandr Wang 认为，[Compute 和数据是未来的货币](https://twitter.com/alexandr_wang/status/1764071674767720823)。

**AI 安全与监管**

- Elon Musk 的诉讼披露了一位投资者在会见 Demis Hassabis 后的言论：“[他能为人类做的最好的事情就是当场击毙 Hassabis 先生](https://twitter.com/AISafetyMemes/status/1763793546535223468)”。
- 印度正在[监管启动 AI 模型的能力](https://twitter.com/levelsio/status/1764422501243703684)，一些人认为这是在关键时刻的自我削弱，类似于中国排挤其科技巨头。
- Vinod Khosla 呼吁禁止开源 AI 平台，Yann LeCun [认为](https://twitter.com/ylecun/status/1764083890119942533)这将导致我们输掉他认为我们正处于其中的“战争”。

**迷因与幽默**

- “[谢天谢地我没去学计算机科学](https://twitter.com/Nexuist/status/1763651659886969329)，”纽约的一位盯着 Excel 的初级分析师说道。“谢天谢地我没去金融行业，”旧金山的一位同样盯着电子表格的 ML 科学家说道。
- Geoff Hinton 被发现在 Google 参与 Gemini 的工作，引发了人们猜测他正准备[从 Sundar Pichai 手中夺回 CEO 职位](https://twitter.com/levelsio/status/1764100109325791561)，以拯救他建立的公司。
- “[Trump 的内部 LLM 似乎遭受了过度剪枝（pruning）。他还剩下多少参数？他的 context window 现在有多短？](https://twitter.com/ylecun/status/1764133615590346994)”

**其他对 AI 工程师有价值的推文**

- “[当你发现那些你没有显式设计、曾经很复杂的事情变得异常简单时，你就知道你新系统的核心抽象（abstractions）找对了](https://twitter.com/gdb/status/1764005795799400796)”
- 一份关于[概率编程（probabilistic programming）](https://twitter.com/svpino/status/1763914648359748066)和分析的指南。


---

# 第 0 部分：摘要的摘要的摘要

> 现在这也由 Claude 3 驱动，其效果[远优于 OpenAI 的输出](https://chat.openai.com/share/b6e0a4c6-ee07-4a45-9215-b4a9408b7493)。

<div class="contents"><p class="whitespace-pre-wrap">明白了，这是以 bullet point markdown 格式呈现的摘要：</p>
<h2><strong>AI 模型性能与对比</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">Anthropic 发布 <strong><a href="https://x.com/anthropicai/status/1764653830468428150?s=46">Claude 3</a></strong> 引发了多个 Discord 服务器上的广泛讨论，以及与 GPT-4 的基准测试对比，用户声称其在数学和编程任务上表现更优。<strong><a href="https://x.com/idavidrein/status/1764675668175094169?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Claude 3 在 GPQA 上的准确率达到约 60%</a></strong> 成为关注焦点。</li>
<li class="whitespace-normal" index="1">关于 <strong>Mistral Large</strong> 模型与 GPT-4 在编程任务上的性能对比引发了争论，尽管有官方基准测试，一些人仍声称其具有优越性。</li>
<li class="whitespace-normal" index="2">拥有 11M 参数的 <strong><a href="https://huggingface.co/HaileyStorm/MambaMate-Micro">Mamba LM 国际象棋模型</a></strong> 展示了令人期待的结果，作为白棋对阵 Stockfish level 0 时达到了 37.7% 的胜率。</li>
</ul>
<h2><strong>AI 工程与部署挑战</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">广泛的讨论围绕部署像 <strong>Mistral</strong> 这样的大语言模型（LLMs）的困难展开，特别关注 <strong>VRAM 需求</strong>、<strong>量化策略（quantization strategies）</strong>，以及针对双 NVIDIA 3090 GPU 等配置的最佳设置。</li>
<li class="whitespace-normal" index="1"><strong>CUDA</strong> 和 GPU 优化是反复出现的话题，相关的资源如 <strong><a href="https://docs.nvidia.com/cuda/cublasdx/index.html">NVIDIA 的 cuBLASDx 文档</a></strong> 和关于 <strong><a href="https://github.com/cuda-mode/lectures/tree/main/lecture8">CUDA 性能陷阱的讲座</a></strong> 被广泛分享。</li>
<li class="whitespace-normal" index="2"><strong><a href="https://arxiv.org/abs/2401.17948">Terminator 架构</a></strong> 被引入，提议用一种新颖的全上下文交互方法取代残差学习（residual learning）。</li>
</ul>
<h2><strong>AI 伦理、隐私与监管</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">在某个 AI 模型的回答包含可识别的个人详情后，人们对 <strong>从个人资料中抓取数据</strong> 的潜在风险表示担忧，引发了关于伦理和法律的讨论。</li>
<li class="whitespace-normal" index="1">根据 <strong><a href="https://x.com/martin_casado/status/1764408870804623753?s=46&amp;t=90xQ8sGy63D2OtiaoGJuww">Martin Casado 的推文</a></strong>，<strong>印度的 AI 部署法规</strong> 要求政府批准，这引发了对可能扼杀创新的警报。</li>
<li class="whitespace-normal" index="2">Open Source Initiative 正在制定 <strong>开源 AI 定义</strong> 的新草案，不断演进的草案可在此处 <strong><a href="https://opensource.org/deepdive/drafts">查看</a></strong>。</li>
</ul>
<h2><strong>前沿 AI 研究与技术</strong></h2>
<ul class="list-disc pl-8 space-y-2" depth="0">
<li class="whitespace-normal" index="0">讨论了 <strong>Aristotelian Rescoring</strong> 这一可能解决复杂 AI 挑战的概念，相关作品如 <strong>STORIUM</strong>、<strong>FairytaleQA</strong> 和 <strong>TellMeWhy</strong> 已在 <strong><a href="https://github.com/StonyBrookNLP/tellmewhy">GitHub</a></strong> 和 <strong><a href="https://huggingface.co/datasets/StonyBrookNLP/tellmewhy">Hugging Face</a></strong> 上发布。</li>
<li class="whitespace-normal" index="1">作为 #Terminator 网络的一部分，引入了新颖的 <strong>HyperZ⋅Z⋅W Operator</strong>，融合了经典与现代技术，完整研究 <strong><a href="https://arxiv.org/pdf/2401.17948.pdf">可在此处获取</a></strong>。</li>
<li class="whitespace-normal" index="2"><strong>RAPTOR</strong> 是 LlamaIndex 引入的一种用于检索增强生成（RAG）的新技术，旨在改进更高层级的上下文检索，正如其在 <strong><a href="https://twitter.com/llama_index/status/1763972097628684607">Twitter</a></strong> 上所宣布的那样。</li>
</ul></div>

---

# 第 1 部分：高层级 Discord 摘要

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **AI 敏感度上升**：Claude 3 AI 的最新版本对潜在攻击性内容和版权问题的敏感度有所提高，引发了关于安全性或过度谨慎的讨论。提到 Claude 3 时关联到了 [Google 支持的 Anthropic 推出的迄今为止最强大的聊天机器人](https://www.cnbc.com/2024/03/04/google-backed-anthropic-debuts-claude-3-its-most-powerful-chatbot-yet.html)。
  
- **CUDA 难题**：社区对 NVIDIA 新的许可条款表示担忧，该条款限制在非 NVIDIA 硬件上使用 CUDA，特别影响了翻译层（translation layers）。讨论围绕最近的更新展开，即 [Nvidia 禁止使用翻译层](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers)。

- **游戏开发陷入僵局**：对于 AI 在不久的将来在游戏开发中的作用，怀疑态度占据主导，因为目前的 AI 局限性可能无法通过单纯增加算力的暴力破解（brute-forcing）方式轻松超越。

- **微调挫败**：报告了一个关于微调的问题，特别是 **OpenOrca Mistral 7b 模型**在 `gguf` 转换后输出错误。该问题在多个频道中被提及，表明了对该问题及其潜在解决方案的广泛关注，建议包括检查量化前的性能以及考虑对离群值使用 imatrix。

- **象棋模型表现出色**：一个参数量为 11M 的小型 Mamba LM 象棋模型训练成功，在作为白方对抗 Stockfish level 0 时表现更好，胜率为 37.7%。模型可在 [HaileyStorm/MambaMate-Micro · Hugging Face](https://huggingface.co/HaileyStorm/MambaMate-Micro) 获取。

- **具备代码能力的 AI 达到新高度**：用户 @ajibawa_2023 展示了其微调模型，特别是 **[OpenHermes-2.5-Code-290k-13B](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B)**，该模型展示了精湛的代码编写能力，并具有应用于博客编写和故事生成等多种任务的潜力。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **GPT 停机期间 AI 社区寻找替代方案**：在最近的一次服务停机期间，用户讨论了 GPT 的替代方案，提到了 *Bing Chat, Hugginface, Deft GPT, LeChat, together.xyz* 和 *Stable Chat*。Anthropic 的 **Claude 3** 被认为是一个令人印象深刻的替代方案，一位用户提到正在尝试免费的 Sonnet 模型，而其他人则在讨论 Claude 3 和 OpenAI 产品等 AI 模型的能力和成本考量。

- **自定义 Agent 与最优代码生成**：关于自定义 Agent 是否可以将 **CSV 文件**集成到其知识库中的问题引发了关于文件类型的技术讨论。用户 @yuyu1337 探索了寻找用于代码生成的最佳 GPT 模型，引发了关于实现最佳时间/空间复杂度以及使用伪代码建议的对话。

- **Vision 与幽默 API 难倒工程师**：参与者尝试在 Prompt 中应用幽默，但在 ChatGPT 和 GPT 3.5 API 之间的效果参差不齐。Discord 社区还沉浸在一个“猫头鹰与棕榈树”的脑筋急转弯中，尝试使用多种 Prompt 策略通过 **GPT-V** 解决该谜题，但由于模型在解释测量数据方面的局限性而遇到了障碍。

- **社区对使用限制的自嘲与感叹**：在关于 AI 局限性和使用限制的玩笑中，用户交流了 Prompt Engineering 技巧，结果各异。有人担心服务器的自动审核会影响讨论和分享高级 Prompt 的能力，呼吁 OpenAI 重新考虑 Prompt 限制，以便进行更有效的知识共享。

- **AI 爱好者提供技巧并寻求训练建议**：新手和资深用户都在询问并提供关于 Prompt Engineering 的建议，讨论了模板结构化的重要性，以及在遵守 OpenAI 政策的前提下利用 AI 进行内容创作任务。讨论强调了在不断发展的 AI 工程和使用领域中，社区和知识交流的重要性。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **订阅混乱引发不满**：@vivekrgopal 反馈称，尽管在试用期间尝试取消，但仍被收取了 **Perplexity** 年度订阅费用，目前正通过私信寻求退款。

- **AI 集成热度攀升**：@bioforever 和 @sebastyan5218 等用户正热切期待 **Claude 3** 和 **Gemini Ultra** 等新语言模型集成到 **Perplexity** 中，这表明了对前沿 AI 功能的强劲需求。

- **AI 回答的基准测试困惑**：@dailyfocus_daily 通过对比 **GPT-4**、**Claude 3** 等模型对“将贴标球分类到盒子中”这一基准问题的回答，深入探讨了不同 AI 模型在解决问题时的一致性差异。

- **IAM 见解与 AI 基础知识**：@riverborse 和 @krishna08011 分享了 **Perplexity** [链接](https://www.perplexity.ai/search/What-is-iam-o2tdFsxGRraeKVWSzo.fIg)，重点介绍身份访问管理（IAM）的见解以及 AI 基础知识，这对希望加深对关键概念理解的技术专业人士非常有用。

- **API 讨论：担忧与期待并存**：用户讨论了 **Perplexity API** 的限制，包括时效性查询问题和缺失的 YouTube 摘要功能；他们还期待诸如引用访问等新功能。关于 Temperature 设置的讨论重新审视了其如何影响语言输出的自然度和可靠性，@icelavaman 分享了一个协助 API 使用的链接：[Perplexity API Info](https://discord.com/channels/1047197230748151888/1118264005207793674/1213229028870324305)。



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord 摘要

**Hermes 2.5 占据领先地位**：社区讨论显示，**Hermes 2.5** 在各项基准测试中意外地超越了 **Hermes 2**，特别提到了 MMLU 基准测试的表现——这对于考虑升级或新部署的用户来说是一个重要参考点。

**Mistral 部署与配置见解**：寻求 **Mistral** 部署最佳配置的工程师们获得了宝贵建议，讨论涵盖了双 NVIDIA 3090 设置的最佳实践、fp16 精度所需的 VRAM（约 90GB）以及量化（quantization）策略。好奇的目光也投向了 "thebloke" 的 Discord 以获取更多社区支持。

**基准测试与个人体验产生共鸣**：大量帖子围绕性能基准测试和不同模型的个人体验展开。特别引人注目的是，据报道 **Mistral Large** 在编程任务上优于 GPT-4，这挑战了官方测试结果，并表明了对特定用户场景基准测试的需求。

**围绕模型局限性的讨论**：技术对话集中在 **Mistral** 和 **Mixtral** 等模型的固有局限性上，具体讨论了 **Mistral-7B-Instruct-v0.2** 的 32k token 限制，以及可能导致性能下降的滑动窗口（sliding window）功能问题。

**微调与使用细节探索**：用户分享了成功利用模型进行情感分析和科学推理等特定任务的见解。然而，对 **Mixtral** 训练实现方案的担忧以及对极简指南的需求，表明社区渴望更清晰的文档。

**新兴 AI 工具与竞争格局**：爱好者和从业者都将注意力转向了新兴的 AI 工具，包括 Kubernetes AI 工具和 Anthropic 发布的 **Claude-3**，引发了关于竞争产品以及 AI 模型开源权重（open weights）重要性的讨论。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **Phi-2 的 Token 限制遭遇瓶颈**：用户讨论了 **Phi-2** 模型在 Token 扩展方面的限制，有建议指出在超过其配置的 2,048 个 Token 限制后，它的表现可能表现得像默认的 Transformer。建议在修改 Phi-2 设置时保持谨慎，以避免性能不稳定。Phi-2 的配置文件链接见[此处](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19)。

- **工程师的 Mac 配置建议**：社区成员交流了大量关于配置新 Mac 的建议，提到了 Homebrew、用于温度监控的 TG Pro 以及用于备份的 Time Machine 等工具。推荐了一个关于 Python/ML 的 Mac 配置 YouTube 教程，链接见[此处](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s)。

- **AI 模型缩放（Scaling）辩论愈演愈烈**：关于扩大 AI 模型规模的益处展开了激烈辩论。一些用户认为，在超过 500B-1T 参数后，效率提升更有可能来自训练技术而非单纯的规模，并引用了批评缩放方法的文章。争论涉及训练 100T 参数模型的实用性以及小模型的潜力，一方表示怀疑，另一方则认为像 Redpajama v2 这样充足的数据阈值仍能推动缩放效益。成本效益和近期 AI 模型的对比也是关注的话题。

- **Claude 3 引起关注**：在综合讨论中，**Claude 3** 因其相对于 **GPT-4** 的潜在性能而备受关注。人们对支持 function calling 模型的推理平台表现出兴趣，并交流了关于 B2B 软件销售策略的建议。此外，还讨论了构建知识图谱（knowledge graphs）的方法，并期待新模型能增强结构化数据提取。

- **针对 LLM 的多样化查询得到解答**：问题涉及 LLM 的 PPO 脚本可用性、模型推理的最佳平台、ChatML 中的 1-shot 训练以及针对客户交互的 AI 微调（fine-tuning）。分享了一个针对可能发生的模型操纵的警告，并附带了一篇 Business Insider 的文章作为背景参考，链接见[此处](https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12)。

- **Project Obsidian 中对 Moondream 的赞誉**：**Moondream**，一个微型视觉语言模型（vision language model），在初步测试中的表现获得了赞誉，为有兴趣探索的用户提供了 GitHub 链接，见[此处](https://github.com/vikhyat/moondream)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

- **开源 AI 接近里程碑**：Open Source Initiative (OSI) 正在制定一份新的*开源 AI 定义*草案，并保持每月更新的节奏，目标是在 2024 年 10 月发布 1.0 版本。相关讨论正在其公共论坛中持续进行。不断演进的草案可以在[这里](https://opensource.org/deepdive/drafts)查看。

- **EFF 对 DMCA 的法律立场**：Electronic Frontier Foundation (EFF) 发起了一项名为 _Green v. Department of Justice_ 的法律挑战，反对 DMCA 的反规避条款，声称这些条款阻碍了对合法购买的受版权保护内容的访问。该案件的详情记录在[这里](https://www.eff.org/cases/green-v-us-department-justice)。

- **AI 中的 Quantization 受到关注**：围绕神经网络中的 Quantization 展开了辩论，特别是关于权重（weights）和激活（activations）的讨论。研究人员讨论了诸如“bitlinear paper”以及激活函数的 Quantization 等论文，并触及了认知不确定性（epistemic uncertainty）的概念。

- **安全警报：通过 GitHub 恶意软件入侵代码**：GitHub 上的一项恶意软件活动通过克隆合法仓库来分发恶意软件。Apiiro 提供的详细威胁分析可在此处查看：[这里](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/)。

- **挑战生物学中的预测建模**：一位用户声称，由于生物系统的复杂性，预测建模无法有效创建具有经济可行性的生物分子，这与工程学中使用的更具预测性的物理模型形成了鲜明对比。

- **利用 Counterfactuals 变革 AI**：一种名为 **CounterCurate** 的新方法结合了 **GPT-4V** 和 **DALLE-3**，带来了视觉语言（visio-linguistic）方面的改进。CounterCurate 使用反事实（counterfactual）图像-字幕对来提升在 Benchmark 上的表现。解释该方法的论文可以在[这里](https://countercurate.github.io/)找到。

- **LLM 被过度炒作了？功能性 Benchmark 如是说**：一场讨论源于一个 [Twitter 线程](https://x.com/_saurabh/status/1763626711407816930?s=20)，该线程质疑了被过度报道的 LLM 推理能力，并引用了表明存在显著推理差距的功能性 Benchmark，详见[这里](http://arxiv.org/abs/2402.19450)，以及配套的 [GitHub 仓库](https://github.com/ConsequentAI/fneval)。

- **Terminator 架构可能取代 Residual Learning**：**Terminator** 网络架构凭借其全新的全上下文交互（full context interaction）方法，可能会取代 Residual Learning。一篇 [arXiv 论文](https://arxiv.org/abs/2401.17948)讨论了其潜力。社区成员暗示了未来的应用和代码发布。

- **AzureML 与 `lm-eval-harness` 的集成**：AzureML 用户讨论了关于设置 `lm-eval-harness` 的问题和解决方案。讨论内容包括依赖项、CUDA 检测、多 GPU 使用以及跨节点的编排，相关见解可以在[这里](https://docs.ray.io/en/latest/serve/index.html)和[这里](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate)找到。

- **Mamba vs Transformer**：对比了 Mamba 和 Transformer 模型在 PARITY 任务上的学习和泛化能力。用户表达了对 LSTM、Mamba 性能以及模型学习 PARITY 机制的看法，并分享了一个用于训练 Mamba 的 [GitHub 脚本](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py)。

- **推进数据集开发**：分享了一个包含 **The Pile** 数据集开发脚本的 GitHub 仓库，这对于从事语言模型训练的人员特别有用。该仓库及其 README 可以在[这里](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts)访问。

- **Figma 与 Imageio 在创意动画中的结合**：提到了一种创新的工作流，通过将 Figma 中创建的 SVG 帧利用 imageio 处理成 GIF 来实现动画效果。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **当模型表现异常时切换 Bot**：用户在 LM Studio 中使用 **Codellama Python 7B** 模型时遇到了问题，`@heyitsyorkie` 建议切换到 [Hugging Face 上的 Magicoder-S-DS-6.7B-GGUF 模型](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF) 以解决“损坏的量化 (broken quant)”问题。关于模型支持的讨论（如 **LoRAs** 和 **QLoRA**）表明这些功能目前尚不可用，且用户无法直接向 LM Studio 上传 PDF 文件。
  
- **数据隐私警钟敲响**：在一次意外的模型响应中包含了可识别的个人详情后，引发了对个人资料潜在数据抓取的担忧，并导致了关于在训练 AI 时此类行为的伦理和法律问题的讨论。

- **VRAM：硬件极客们的热门话题**：多个线程涉及了对充足 VRAM 的必要性，建议 GPU 至少具备 24 GB 以高效运行 LLM。讨论中提到了诸如 [Debian on Apple M1](https://wiki.debian.org/InstallingDebianOn/Apple/M1) 等资源，强调了 Apple 的统一内存架构以及在 Linux 环境下使用 Apple M1 Macs 进行 AI 工作的局限性和潜在挑战。

- **即将发布的 Beta 版本引发热议**：`@heyitsyorkie` 表示即将发布的 LM Studio Beta 版本将包含 **Starcoder2-15b** 的集成。这一讨论得到了一个[为 llama.cpp 添加该支持](https://github.com/ggerganov/llama.cpp/pull/5795)的 GitHub Pull Request 的支持。

- **Autogen 的试错历程**：用户在 Autogen 集成过程中遇到了问题，例如 **401 错误** 以及 LM Studio 中模型加载缓慢。故障排除建议包括重新安装，并参考 [StackOverflow](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10/41489151#41489151) 上的说明，在使用 Docker 指南时针对 Windows 系统路径进行调整。

- **使用分布式 LLM 进行 AI 工程**：有人提出了关于开发自定义 AI Agent 以及在各种硬件配置上运行不同 LLM 的查询，提到了包括 **3090 GPU**、**Jetson Orion** 和 **6800XT GPU** 在内的特定硬件。然而，关于这些主题没有提供进一步的上下文或详细讨论。

- **简短交流**：一位用户确认了在 Arch Linux 上使用 *yay* 存在相关软件包，另一位用户在没有额外上下文的情况下询问了某项功能的 Linux 支持情况。

- **AI 讨论中需要更高的清晰度**：评论指出，关于 **JavaScript** 与 **crew ai** 的兼容性讨论缺乏上下文和清晰度，此外还提到了 *Visual Studio Code*，这需要进一步的信息。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **模型训练的“饥饿游戏”**：工程师们调侃 AI 模型在训练期间的“饥饿感”，吞噬了 90GB 的显存。部署时最好检查一下 **Gradio** 组件，因为像 `ImageEditor` 这样过时的版本可能会成为你的噩梦。

- **AI 学习阶梯**：从新手到专家，成员们都渴望攀登 AI 学习曲线，分享了关于 **CUDA**、**Gradio** 中的 **SASS** 以及 **PPO** 理论的资源——没有任何安全绳保护。

- **聊天会议通话**：像北卡罗来纳州阿什维尔的会议这样的 AI 社区活动是 **GenAI** 爱好者的线下聚会场所。与此同时，在 TTS 和编写读书会问题等任务上也出现了合作——谁说 AI 没有爱好？

- **Discord 聚焦 Diffusers**：`diffusers` 调度器命名问题让大家在更新后反复检查他们的类，直到一个 [Pull Request 修复](https://github.com/huggingface/diffusers/pull/7192) 被合并。讨论中展示了 Inpainting（局部重绘）的案例，而 **LoRA** 适配器的实现建议则像万圣节糖果一样随处可见。

- **前卫机器人与数据侦探**：创意工程师在 [Poe 平台](https://poe.com) 上发布了 DeepSeek Coder 33b 和 V0.4 Proteus 等机器人。其他人分享了在蛋白质异常检测方面的突破，以及对 AI 与音乐采样交集的思考，暗示着 AI 可能会成为你下一场派对的 DJ。

- **Diffusers 中调度器混淆问题的解决**：`diffusers` 中调度器类名错误的 GitHub Issue 已通过 [Pull Request](https://github.com/huggingface/diffusers/pull/7192) 解决，为需要正确工具且不希望被混淆的 AI 工程师提高了准确性。

- **NLP 模型部署之争**：在部署 NLP 模型时，Flask 与 Triton 的比较并非同类竞争——选择你的战场。如果你在追求效率，**Adam 优化器**在某些圈子里仍然稳坐头把交椅，但也要留意竞争对手。

- **搭建通往计算机视觉的桥梁**：人们正在探索土木图纸的地理参考 PDF 与 GIS CAD 之间的联系，而好奇者则在思考小型 Visual Language Models 在客户入职等任务中的潜力。AI 与视觉协同作用的前景正在不断扩展，就在可见光谱之外。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **HyperZ⋅Z⋅W 算子撼动根基**：`@alex_cool6` 介绍了 #Terminator 网络，它融合了经典与现代技术，并利用了新颖的 **HyperZ⋅Z⋅W Operator**，完整研究可[在此查看](https://arxiv.org/pdf/2401.17948.pdf)。

- **Claude 3 引发关注**：围绕 **Claude 3 模型** 的讨论正在升温，其性能基准测试引起了社区的轰动。一个 [Reddit 帖子](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/) 展示了社区对其能力的调查。

- **Claude 3 表现优于 GPT-4**：`@segmentationfault8268` 发现 **Claude 3** 在动态响应和理解方面超越了 GPT-4，可能会从现有的 ChatGPT Plus 订阅用户中吸引走用户。

- **Claude 3 在 CUDA Kernel 方面仍存挑战**：尽管有所进步，但正如 `@twoabove` 所指出的，Claude 3 在处理 **PyTorch CUDA kernels** 等非标准任务方面似乎没有改进。

- **Sonnet 进入 VLM 赛场**：对话引发了对 **Sonnet** 的兴趣，它被确定为一种 Visual Language Model (VLM)，并将其性能与 **GPT4v** 和 **CogVLM** 等巨头进行了对比。

- **寻求 DPO 调整方面的帮助**：`@huunguyen` 发出了合作呼吁，以优化 **Dynamic Programming Optimizer (DPO)**。欢迎感兴趣的合作者通过私信联系。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

**VRAM 交换空间探讨**：讨论集中在将 **Linux VRAM** 用作交换空间，相比传统的磁盘分页具有潜在的速度优势，尽管也提到了可能存在的资源冲突。分享了 [GitHub 上的 vramfs](https://github.com/Overv/vramfs) 和 [ArchLinux 文档](https://wiki.archlinux.org/title/Swap_on_video_RAM) 等资源。

**快速验证与聊天记录检索**：用户寻求关于访问前一天实时聊天讨论的帮助，并询问了 lightning.ai 上的 Gmail 验证时间，强调了快速解决时间和访问录播课程的便利性。

**CUDA 难题与 Triton 调整**：工程师们分享了对 CUDA 编程难点的见解，探讨了 **Triton 与 NVCC 的关系** 以及 Hopper 架构中的 **异步矩阵乘法**。重点介绍了 [unsloth 仓库](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53) 和 [Triton GitHub 页面](https://github.com/openai/triton) 等资源。

**GPU 驱动的数据库**：在 GPU 上运行数据库的想法引起了关注，提到了 [cuDF 库](https://github.com/rapidsai/cudf) 并引用了一篇 [关于 GPU 数据库的 ZDNet 文章](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/)。

**Mistral 的计算思考**：围绕 **Mistral 的计算能力** 展开了辩论，质疑 1.5k H100 GPU 对于大规模模型训练是否足够，并讨论了异步操作。相关链接包括 [NVIDIA 的 cuBLASDx 文档](https://docs.nvidia.com/cuda/cublasdx/index.html) 和 [Arthur Mensch 的推文](https://x.com/arthurmensch/status/1762818733016322168)。

**PyTorch 开发者播客发布新剧集**：讨论 [AoTInductor](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor) 的播客剧集被分享，呼应了社区对该系列的热情。

**Ring Attention 引起关注**：Ring Attention 和 Striped Attention 是热门话题，引用了 YK Discord 上的讨论和 [Together.ai 的博客文章](https://www.together.ai/blog/flash-decoding-for-long-context-inference)。[ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 和 [flash-attention](https://github.com/Dao-AILab/flash-attention) 等各种代码库提供了实现见解。

**CUDA-MODE 课程上线**：宣布了关于 **CUDA 性能陷阱的第 8 课**，承诺提供最大化 Occupancy 和最小化问题的技巧，即将为热情的学习者开讲。

**职业基石**：Lamini AI 和 Quadrature 发布了招聘 HPC 和 GPU 优化工程师的职位，重点介绍了参与激动人心项目的机会，例如在 AMD GPU 上优化 LLM 以及全球金融市场中的 AI 工作负载。详情请见 [Lamini AI 招聘](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced) 和 [Quadrature 招聘](https://quadrature.ai/)。

**第 8 课在 YouTube 上重新发布**：在之前的录制出现技术问题后，题为 *CUDA Performance Checklist* 的第 8 课已重新录制并分享，同时附带了相应的 [代码示例](https://github.com/cuda-mode/lectures/tree/main/lecture8) 和 [幻灯片](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **RAPTOR 提升 RAG 检索**：LlamaIndex 推出了 **RAPTOR**，这是一种用于 *Retrieval-Augmented Generation* (RAG) 的新技术，可改进更高层级上下文的检索。该技术旨在更好地处理复杂问题，已通过 [Twitter](https://twitter.com/llama_index/status/1763972097628684607) 发布。

- **GAI 进入城市规划**：LlamaIndex 展示了 RAG 的实际应用，包括一个 **GAI 驱动的 ADU 规划器**，旨在优化附属居住单元 (accessory dwelling units) 的建造过程 [Tweet](https://twitter.com/llama_index/status/1764020728427667605)。

- **MongoDB 与 RAG 结合**：根据 [Twitter 更新](https://twitter.com/llama_index/status/1764078471276642469)，LlamaIndex 推出了由 @AlakeRichmond 开发的新参考架构，利用 @MongoDB Atlas 进行高效的数据索引，这对于构建复杂的 RAG 系统至关重要。

- **语义策略强化 RAG**：语义分块 (Semantic chunking) 因其通过对语义相似的数据进行分组来提升 RAG 的检索和合成能力的潜力而受到关注。这一方法由 Florian June 分享并被 LlamaIndex 采纳 [Twitter post](https://twitter.com/llama_index/status/1764335221141631471)。

- **Claude 3 的胜利三部曲**：Claude 3 已发布多个变体，包括 Claude Opus。根据 LlamaIndex 的说法，其性能已超越 GPT-4，并宣布立即支持该模型 [公告](https://twitter.com/llama_index/status/1764731195286577247)。

- **在 LlamaIndex 中利用 LongContext**：**LlamaIndex** 与 **LongContext** 的集成显示出增强 RAG 的前景，特别是随着 Google 最近发布的具有 1M 上下文窗口的 Gemini 1.5 Pro，该功能可能会被整合 [Medium 文章](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738)。

- **社区角落火热**：**LlamaIndex Discord 社区**被称赞比其他社区更有组织且更具支持性，特别是在 API 文档结构见解以及关于建立涉及混合向量和关键词搜索的复杂搜索系统的实用指南方面 [Y Combinator news](https://news.ycombinator.com/item?id=37764489), [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/)，以及上述列出的多种其他资源。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 总结

- **Claude 3.0 登陆 OpenRouter**：备受期待的 **Claude 3** AI 已经发布。`@alexatallah` 提到，OpenRouter 上提供了一个独家的实验性自我审查版本。

- **LLM 安全游戏引发关注**：`@leumon` 在服务器上发起了一个游戏，尝试欺骗 GPT3.5 泄露密钥，强调了谨慎处理 AI 输出和保护敏感数据的重要性。玩家还可以自由体验各种 AI 模型，如 **Claude-v1, Gemini Pro, Mixtral, Dolphin** 和 **Yi**。

- **Claude 3 vs GPT-4 的反应与测试**：关于 Claude 3（包括 **Claude 3 Opus**）与 GPT-4 之间的对比讨论和反应正在进行中。用户 `@arsoban` 在测试中注意到 Claude 3 Opus 具有更强的文本理解能力，而其他用户则对其定价表示担忧。

- **AI 之间的性能辩论升温**：不同 Claude 3 变体的能力引发了辩论。分享的观察结果包括 Sonnet 有时表现优于 Opus，并计划测试 Claude 3 在游戏应用中的英译代码能力。

- **社区发现 AI 性能退化**：`@capitaindave` 指出 **Gemini Ultra** 的推理能力似乎随着时间的推移而下降，引发了关于模型发布后性能可能退化的讨论。

**提到的链接**：
- OpenRouter 上的 Claude 3.0：[OpenRouter](https://openrouter.ai/playground?models=anthropic/claude-instant-1.2)
- Discord 上的 LLM 加密挑战：[Discord - 与好友和社区聊天的新方式](https://discord.gg/YWX8Eft6R8)
- Claude 性能测试结果图片：[codebyars.dev](https://share.codebyars.dev/u/jGY25U.png)

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 摘要

- **OpenAI 凭借 Browsing 功能开启新篇章**：OpenAI 推出了 **Browsing 功能**，因其与 Gemini/Perplexity 等现有工具的相似性而引发关注。该公告通过 [OpenAI 的推文](https://twitter.com/wangzjeff/status/1764572262743851339)发布。

- **Claude 3 的惊艳亮相**：根据用户 `@res6969` 的说法，新的 **Claude 3** 模型系列因在数学和代码任务中可能超越 GPT-4 而引起轰动。关于其成本效益的讨论以及对 **Haiku 模型** 的期待，凸显了用户在平衡价格与性能方面的兴趣。

- **Claude 3 的运行优势**：`@res6969` 提到的实验表明，**Claude 3 的延迟**表现优于其他模型，首个 Token 响应时间约为 4 秒，展示了其在用户体验方面的运行效率。

- **探索高性价比的 Embedding 解决方案**：为了在生产环境中实现每秒 100 次推理的目标，`@iyevenko` 探索了最具成本效益的 Embedding 模型。用户 `@yikesawjeez` 的建议包括 [Qdrant](https://qdrant.tech/) 和 [Weaviate](https://www.semi.technology/developers/weaviate/current/)。

- **权衡 OpenAI Embedding 的成本效益**：尽管最初存在质量方面的顾虑，`@iyevenko` 正在考虑将 OpenAI 的 Embedding 解决方案用于云基础设施，鉴于其 Embedding 的改进，这些方案看起来非常有性价比。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 摘要

- **Anthropic 发布 Claude 3，好评如潮**：[AnthropicAI 宣布](https://x.com/anthropicai/status/1764653830468428150?s=46)推出 **Claude 3**，这是其最新的 AI 模型系列，包括 Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku，挑战了 AI 性能基准。`@sid221134224` 和 `@canadagoose1` 等用户表达了他们的兴奋，指出了 Claude 3 优于 GPT-4 的强项，以及由于不依赖专有数据集而具备的潜力。

- **Claude 3 引发误导信息和争议**：**Claude 3** 的发布催生了一些问题推文的传播，导致 `@natolambert` 直接干预，指责误导性帖子“愚蠢”。`@natolambert` 还幽默地拒绝了使用备用账号来打击误导信息的想法，因为这太费精力。

- **RL 创新与讨论**：一篇关于 RL 基础模型的论文受到关注，讨论了基于奖励函数 Embedding 的策略，以实现自适应泛化（[Sergey Levine 的推文](https://twitter.com/svlevine/status/1764116636142076070)）。同时，社区探讨了 Cohere PPO 论文中的观点，即 LLM 可能不需要策略梯度优化 (PPO) 的修正，这引发了其他研究小组进行验证的兴趣。

- **从大银幕到 AI 梦想**：`@natolambert` 正在寻找 **视频编辑** 合作伙伴来制作预告片，灵感可能来自电影《她》(Her)，强调 AI 主题。此外，`@natolambert` 预告了即将发布的内容，并提到可能与 Hugging Face 的 CTO 合作，并链接到了关于 **开源 AI** 益处的讨论（[Logan.GPT 的推文](https://x.com/officiallogank/status/1764435268021502226?s=46)）。

- **AI 社区拥抱 Julia**：在讨论中，`@xeophon.` 关注了 **Julia 编程语言** 在 AI 开发中的优势，并为感兴趣的人提供了 [JuliaLang](https://julialang.org/) 的链接。对话表明工程社区对 Julia 的参与度正在提高。



---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **破译 Tokenizer 机制**：`@lhc1921` 分享了一个 [YouTube 教程](https://www.youtube.com/watch?v=zduSFxRajkE)，深入探讨了如何为大语言模型 (LLMs) 构建 **tokenizer**，并强调了其在将字符串转换为 token 过程中的重要性。

- **Galaxy AI 提供免费 API 访问**：`@white_d3vil` 介绍了 **Galaxy AI**，该平台为包括 **GPT-4**、**GPT-4-1106-PREVIEW** 和 **GPT-3.5-turbo-1106** 在内的高端 AI 模型提供免费 API 服务。

- **可扩展 LLM Web 应用的技术栈建议**：关于构建可扩展的 **LLM Web 应用**，社区提出了多种建议，从使用 Python 3.11 配合 FastAPI 和 Langchain，到在 Vercel 上利用 Next.js 配合 Langserve.js。用户询问了 Langchain 的生产就绪性以及商业用途的定制化，并表示在生产环境中更倾向于使用自定义代码。

- **警惕潜在的垃圾链接**：提醒用户注意 `@teitei40` 在多个频道分享的可疑链接，该链接声称提供 50 美元的 Steam 礼品卡，但其合法性令人担忧，可能存在网络钓鱼风险。

- **创新项目与教育资源**：社区展示了多项作品，包括 **Devscribe AI** 的 YouTube 视频聊天工具、使用生成式 AI 进行资产负债管理的指南，以及用于现代 Web 开发的 **Next.js 14+ 启动模板**。此外，还重点讨论了增强 Langchain 的 **retrieval-augmented generation** (RAG) 以及费曼技巧 (Feynman Technique) 在学习中的功效。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **深度集成 Stack Overflow 知识**：Gemini for Google Cloud 将通过 [OverflowAPI](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/) 集成 Stack Overflow 的知识库，旨在直接在云控制台中提升 AI 辅助的精准度。
  
- **布林寄希望于 AGI 突破**：Google 联合创始人 Sergey Brin 建议像 Gemini 这样的项目可能会引领 Google 的人工智能走向 AGI，这一观点在一条流传的关于其见解的 [推文](https://twitter.com/marvinvonhagen/status/1764036713889116661) 中引发了讨论。

- **完善数字现实**：[LayerDiffusion](https://github.com/layerdiffusion/sd-forge-layerdiffusion) 为 AI 创意开辟了新视野，提供的工具可以将物体无缝插入照片并带有真实的反射效果，这对 Stable Diffusion 爱好者来说是一个极具前景的项目。

- **Claude 3 引起轰动**：Anthropic 宣布其 Claude 3 模型家族，因其先进的元数据感知能力以及对当前 AI 模型的影响在 AI 社区引发热议，并分享了重要的基准测试数据，例如 [Claude 3 在 GPQA 上的准确率约为 60%](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **印度 AI 监管瓶颈**：Martin Casado 关于印度 AI 部署法规的 [推文](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww) 引发了对政府审批可能扼杀创新的担忧，在技术社区中激起了关于监管与进步之间平衡的辩论。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **已解决：Hugging Face Commit 混乱**：`@giftedgummybee` 报告称，通过识别 commit 版本混淆，解决了 Hugging Face `KTO` 问题。这主要涉及 Hugging Face transformers 库，与 **Axolotl** 的部署相关。

- **Axolotl 坚持使用 Hugging Face**：`@nanobitz` 澄清说，目前没有将 **Axolotl** 移植到 Tinygrad 的计划，理由是其对 Hugging Face transformers 库的依赖，并提醒用户将配置问题发布在适当的帮助频道中。

- **Axolotl 考虑集成 Optuna CLI**：`@casper_ai` 建议集成一个使用 Optuna 进行超参数优化的 CLI 工具，并引用了一个 [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356) 作为参考。

- **深度学习 GPU 难题与修复**：出现了一些与 GPU 相关的问题，包括 `python` 与 `python3` 的冲突以及 deepspeed 最终保存时的故障；然而，`@rtyax` 表示在 deepspeed 0.13.4 的最终保存功能上没有遇到问题。

- **Mixtral vs. Mistral：模型偏好对决**：`@dctanner` 发起了一场关于使用 **Mixtral** 还是 **Mistral Large** 进行合成数据生成的讨论，`@le_mess` 表示相比 **Mixtral** 更倾向于个人模型，并指出不同用例下的性能表现存在细微差别。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- **Aristotelian AI 模型登场**: [@crispstrobe](https://discord.com/channels/1178995845727785010/1178996063537991752/1213540286941364254) 讨论了 "Aristotelian Rescoring" 的潜力，这一概念可能解决复杂的 AI 挑战，并重点介绍了 STORIUM、FairytaleQA 和 TellMeWhy 等相关工作，资源可在 [GitHub](https://github.com/StonyBrookNLP/tellmewhy) 和 [Hugging Face](https://huggingface.co/datasets/StonyBrookNLP/tellmewhy) 获取。
- **德语语义取得跨越式进展**: `@sten6633` 通过使用特定领域文本和 Telekom 的释义数据集微调 Deepset 的 `gbertlarge`，并将其转换为 Sentence Transformer，改进了德语语义相似度计算。
- **渴望 AI 生产实践经验分享**: `@dsquared70` 邀请从事 Generative AI 生产环境工作的个人在即将于北卡罗来纳州阿什维尔举行的会议上发言，申请截止日期为 [4月30日](https://www.aiinproduction.com/cfp)。
- **精细对齐德语数据**: `@johannhartmann` 指出了数据集中的一个翻译错误，并在修复其使用 `./fasteval` 进行评估的 bug 后，成功将修正后的数据集集成到 [FastEval](https://github.com/mayflower/FastEval) 中。
- **Brezn 的双语突破**: [@thomasrenkert](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) 赞扬了 **Brezn-7b** 在德语方面的表现，该模型由 model merging 驱动并与 3 个 DPO 数据集对齐；而 [@johannhartmann](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) 建议默认使用 ChatML 以提高 Brezn 的基准测试分数。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **Stable Diffusion XL Lightning 表现出色**: **Stable Diffusion XL Lightning** 的能力给用户留下了深刻印象，详见分享的演示链接：[fastsdxl.ai](https://fastsdxl.ai/)。
- **Claude 3 交互现已简化**: **SimonW** 发布了一个针对 **Claude 3** 模型的新插件，仓库地址位于 [GitHub](https://github.com/simonw/llm-claude-3)。
- **朝鲜蓟命名富有创意**: 一位用户通过为朝鲜蓟建议诙谐的名字（如 "Choke-a-tastic" 和 "Arti-party"）为讨论增添了幽默感。
- **Mistral 模型价格引起关注**: **Mistral large** 模型的数据提取性能赢得了赞誉，但其高于预期的成本也受到了关注。
- **插件开发速度赢得掌声**: 用于与 Claude 3 模型交互的新插件的开发速度迅速获得了社区的称赞。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- **新的合作机会**: `@wasooli` 表现出在 **Alignment Lab AI** 社区内合作的浓厚兴趣，`@taodoggy` 表示愿意通过私信进一步讨论。
- **GenAI 会议征稿**: `@dsquared70` 宣布了一个 **GenAI in production** 会议，鼓励在 **4月30日** 前提交申请。更多信息和申请详情见 [AI in Production](https://www.aiinproduction.com/cfp)。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **征集 AI 集成专家**: 诚邀正积极将 **GenAI** 集成到生产系统中的开发者在阿什维尔的会议上分享见解。有意者可在 4 月 30 日前通过 [AI in Production Call for Presentations](https://www.aiinproduction.com/cfp) 提交论文。
- **幽默的一天开始**: 尽管将 "yolks" 误拼为 "yokks"，但一句 "good morning yokks" 的问候还是为讨论带来了一个幽默的开始。

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

- **黑客松层级说明**: `@needforspeed4` 询问 **Agape hackathon** 是否与管理该 Discord 服务器的 **AI Engineer Foundation** 有关，以及不同黑客松使用独立 Discord 的情况。`@hackgoofer` 澄清说 **AI Engineer Foundation Hackathons** 在此 Discord 内举行，但指出 **Agape hackathon** 是独立运行的。

---

# 第二部分：按频道划分的详细摘要和链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1213408815492374580) (994 条消息🔥🔥🔥):

- **Claude 3 AI，是安全还是过度谨慎？**：Claude 3 是一款新的 AI 模型，用户展示了它对潜在冒犯性内容或版权问题的高度敏感性。
- **关于 AI 在游戏开发中作用的反思**：一些用户预测 AI 将参与未来视频游戏的渲染和创作；然而，netrve 对通过原始算力暴力突破当前 AI 局限性的能力表示怀疑。
- **备受争议的 NVIDIA 授权策略**：NVIDIA 试图通过授权条款限制在非 NVIDIA 硬件上使用 CUDA，这引发了关于其合法性以及对开发者影响的讨论，特别是针对翻译层（translation layers）方面。
- **基准测试与 OpenAI 的未来**：在讨论 Phind 70b 等模型的同时，用户也质疑了基准测试的可靠性，以及在对 GPT-5 的期待中，持续发布的 AI 模型的意义。
- **GPU 技术深度探讨**：Netrve 讨论了游戏渲染的复杂性和进展，包括 Epic 在 Unreal Engine 5 中的 Nanite 系统，而其他人则对 NVIDIA 的限制性举措表示遗憾。

**提到的链接**：

- [未找到标题](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-closes-execution-speed-and-the-code-generation-quality-gap-with-gpt-4-turbo/?amp=)：未找到描述
- [未找到标题](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-clos)：未找到描述
- [AI 公开信 - SVA](https://openletter.svangel.com/)：为更美好的未来构建 AI
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1764653833970659560?s=20)：随着此次发布，用户可以根据其使用场景选择智能、速度和成本的理想组合。Opus 是我们最智能的模型，实现了接近人类的理解能力。我...
- [Nvidia 禁止在 CUDA 软件中使用翻译层 —— 此前该禁令仅列在在线 EULA 中，现在已包含在安装文件中 [已更新]](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers)：翻译层成为针对目标。
- [Fr Lie GIF - Fr Lie - 发现并分享 GIF](https://tenor.com/view/fr-lie-gif-9063740662564569899)：点击查看 GIF
- [谷歌支持的 Anthropic 推出其迄今为止最强大的聊天机器人，生成式 AI 竞争升温](https://www.cnbc.com/2024/03/04/google-backed-anthropic-debuts-claude-3-its-most-powerful-chatbot-yet.html)：Anthropic 周一推出了 Claude 3，这是一个聊天机器人及 AI 模型系列，称其为迄今为止速度最快、功能最强大的模型。
- [谷歌正迅速成为其“好友”Nvidia 的强大对手 —— 为其超级计算机提供动力的 TPU v5p AI 芯片比以往任何时候都更快，拥有更多的内存和带宽，甚至击败了强大的 H100](https://www.techradar.com/pro/google-is-rapidly-turning-into-a-formidable-opponent-to-bff-nvidia-the-tpu-v5p-ai-chip-powering-its-hypercomputer-is-faster-and-has-more-memory-and-bandwidth-than-ever-before-beating-even-the-mighty-h100)：谷歌最新的 AI 芯片在训练 LLM 方面的速度比其前代产品快 2.8 倍，并已集成到 AI Hypercomputing 架构中。
- [GPU 评论、分析和购买指南 | Tom's Hardware](https://www.tomshardware.com/pc-components/gpus)：未找到描述
- [Lone (Hippie)](https://huggingface.co/Lone)：未找到描述
- [Turbo (Chen)](https://huggingface.co/Turbo)：未找到描述
- [pip wheel - pip 文档 v24.0](https://pip.pypa.io/en/stable/cli/pip_wheel/#cmdoption-only-binary>)：未找到描述
- [GitHub: 从这里开始构建](https://github.com)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...
- [修复默认值 + 纠正 Mixtral 配置文档中的错误，由 kalomaze 提交 · Pull Request #29436 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29436)：此 PR 做了什么？max_position_embeddings 的默认值被错误地设置为 4096 * 32。这已被修正为 32768。Mixtral 不使用 Sliding Window Attention，它被设置为...
- [添加 Q4 缓存模式 · turboderp/exllamav2@bafe539](https://github.com/turboderp/exllamav2/commit/bafe53972840cb0d0673bbd85a3afdeab360a9ab#diff-71431ec327109cd8333884920a1573a325ed1eea3dea804d6bc652f91a4a91f8)：未找到描述
- [[Mixtral] 修复 loss 中的 attention masking，由 DesmonDay 提交 · Pull Request #29363 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29363)：此 PR 做了什么？我认为 load_balancing_loss 中可能存在一些不太正确的地方。在提交之前，此 PR 修复了一个拼写错误或改进了文档（你可以忽略其他检查，如果这...
- [GitHub - e-p-armstrong/augmentoolkit: 将算力和书籍转换为指令微调数据集](https://github.com/e-p-armstrong/augmentoolkit)：将算力和书籍转换为指令微调数据集 - e-p-armstrong/augmentoolkit

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1213410060785684571) (379 条消息🔥🔥): 

- **理解 Llama.cpp 的局限性**：`@pri9278` 指出，虽然 SD (Sparse Diffusion) 和 Lookup decoding 已在 llama.cpp 中实现，但它们尚未集成到 server API 中，这限制了该模型服务端实现的能力。
- **模型性能与硬编码**：`@superking__` 讨论了硬编码模型的复杂性，指出了在使用 Transformer 时存在的困难，以及在使用严格格式进行模型提示（prompting）时的可能性。
- **关于角色扮演与故事生成的讨论**：聊天成员（包括 `@gamingdaveuk`、`@netrve`、`@lisamacintosh` 和 `@concedo`）就使用 AI 模型进行角色扮演和故事生成展开了深入讨论，探讨了诸如用于优化的 context caching、前端/用户界面特性，以及聊天机器人在角色扮演场景中的具体用例。
- **分享微调模型的经验**：`@c.gato` 分享了测试 Thespis-CurtainCall Mixtral 模型的经验，评价了它在处理复杂任务（如玩井字游戏和基于 greentext 故事生成提示词）时的表现。
- **参与 AutoGPT 和 DSPY 的讨论**：`@sunija` 询问了 AutoGPT 的现状及其在角色扮演中的应用，引发了 `@wolfsauge` 和 `@maldevide` 的回复，讨论了如 DSPY 等替代方法，用于优化提示词生成和自动评估响应变化。

**提到的链接**：

- [Constructive](https://xkcd.com/810/)：未找到描述
- [Chub](https://chub.ai/characters/illuminaryidiot/vixens-of-the-orient-express-freeplay-a39e7fe1>)：查找、分享、修改、转换和版本控制用于对话式大语言模型 (LLMs) 的角色及其他数据。曾用名/别名：Character Hub, CharacterHub, CharHub, CharaHub, Char Hub。
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述
- [cgato/Thespis-CurtainCall-8x7b-v0.3 · Hugging Face](https://huggingface.co/cgato/Thespis-CurtainCall-8x7b-v0.3)：未找到描述
- [Thanos Memoji GIF - Thanos Memoji - Discover &amp; Share GIFs](https://tenor.com/view/thanos-memoji-gif-23490017)：点击查看 GIF
- [ZeroBin.net](https://zerobin.net/?946f95701988d7a9#qkUcZ1pb/9O5nK4Zal)：未找到描述
- [ZeroBin.net](https://zerobin.net/?946f95701988d7a9#qkUcZ1pb/9O5nK4ZalZTRztZwuZnU3hwBu9cK3hgLVo=)：未找到描述
- [Mihawk Zoro GIF - Mihawk Zoro One piece - Discover &amp; Share GIFs](https://tenor.com/view/mihawk-zoro-one-piece-gif-15330479296855641524)：点击查看 GIF
- [Worldsgreatestswordsmen Onepiece GIF - Worldsgreatestswordsmen Onepiece Mihawk - Discover &amp; Share GIFs](https://tenor.com/view/worldsgreatestswordsmen-onepiece-mihawk-anime-one-gif-25849503)：点击查看 GIF
- [Cat Cat Meme GIF - Cat Cat meme Funny cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-cat-meme-funny-cat-cat-eating-cat-eating-chips-gif-10455465908695706650)：点击查看 GIF
- [Rapeface Smile GIF - Rapeface Smile Transform - Discover &amp; Share GIFs](https://tenor.com/view/rapeface-smile-transform-gif-12599812)：点击查看 GIF
- [GGUF quantizations overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)：GGUF 量化概述。GitHub Gist：即时分享代码、笔记和代码片段。
- [Family guys - Sony Sexbox](https://youtu.be/7ciVKIm7bcg?si=2Tu1DYbRtuRhkgXT)：选自《盖酷家庭》(Family Guy) 第 3 季第 16 集。

  

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1213458845616177172) (39 messages🔥): 

- **微调困扰**：`@coldedkiller` 在微调 **OpenOrca Mistral 7b 模型**时遇到问题；在转换为 'gguf' 格式后，模型无法针对其原始数据和微调数据给出正确的输出。
- **训练模型中的余弦相似度阈值**：`@gt9393` 询问了模型合适的余弦相似度阈值（cosine similarity cutoff），`@dirtytigerx` 回复称这取决于多种因素，无法提供固定的硬性阈值。
- **特殊 Token 的使用与模型训练**：`@gt9393` 讨论了关于在数据集中包含序列开始和结束 Token（start and end of sequence tokens）的不确定性。`@dirtytigerx` 建议保留这些 Token，但在 Prompt 编码后再进行追加。
- **国际象棋模型训练成果**：`@.haileystorm` 分享了他们成功训练一个 11M 参数的 **Mamba LM 国际象棋模型**的经验，并提供了相关资源和训练代码的链接，表示该模型作为白方表现更好。该模型的训练过程与更大参数的模型进行了对比，并展示了对阵 Stockfish 0 级时 **37.7% 的胜率**。
- **寻求中小规模 LLM 的微调指导**：用户 `@coldedkiller` 和 `@zelrik` 寻求微调语言模型的建议，被引导至 Jon Durbin 的资源和 **UnslothAI** 的指南。讨论涵盖了格式、特殊 Token 和硬件要求，`@maldevide` 提供了关于预处理书籍文本、硬件容量以及参数高效微调（PEFT）工具的见解。

**提到的链接**：

- [HaileyStorm/MambaMate-Micro · Hugging Face](https://huggingface.co/HaileyStorm/MambaMate-Micro)：暂无描述
- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel)：A bagel, with everything. 通过创建 GitHub 账号参与 jondurbin/bagel 的开发。
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth?tab=readme-ov-file#-finetune-for-free)：速度提升 5 倍，显存占用减少 60% 的 QLoRA 微调。通过创建 GitHub 账号参与 unslothai/unsloth 的开发。

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1213458904906727496) (1 messages): 

- **OpenOrca 微调苦恼**：`@coldedkiller` 在使用**微调后的 OpenOrca Mistral 7b 模型**时遇到问题。转换为 `gguf` 格式后，模型在原始数据集和微调数据集上均无法产生正常的输出。
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1213459165041655818) (11 messages🔥): 

- **OpenOrca 微调苦恼**：用户 `@coldedkiller` 面临微调后的 OpenOrca Mistral 7b 模型在转换为 gguf 格式后无法输出预期答案的问题。`@spottyluck` 建议在量化前检查模型性能，如果存在离群值（outliers）问题，可以考虑使用 imatrix。
  
- **GPTQ 退出聚光灯**：`@yeghro` 询问 GPTQ 是否不再是关注焦点，因为 TheBloke 已经停止发布相关内容；`@_._pandora_._` 暗示有传言称 *TheBloke 失踪了*，导致近期没有新发布。

- **模型测试困境**：`@gamingdaveuk` 正在寻找可以在 6GB VRAM 笔记本电脑上加载的最小模型，用于 API 调用测试。他们提到在 Reddit 上找到的答案建议使用 *Mistral instruct v0.2*，而 `@dirtytigerx` 则主张使用任何大小在 4GB 左右的 gguf 量化模型。

- **Coldedkiller 的模型故障**：在后续讨论中，`@coldedkiller` 详细说明了他们的微调模型在格式转换后无法从训练好的问答数据集中提供答案的问题。他们观察到模型在被查询时给出了无关的响应。

- **Ajibawa_2023 展示增强版模型**：用户 `@ajibawa_2023` 分享了他们微调模型的链接，这些模型具有增强的编程能力。其中一个模型 **[OpenHermes-2.5-Code-290k-13B](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B)** 整合了他们的数据集，在编程排名中表现良好，并能处理包括博客编写和故事生成在内的各种任务。

**提到的链接**：

- [ajibawa-2023/OpenHermes-2.5-Code-290k-13B · Hugging Face](https://huggingface.co/ajibawa-2023/OpenHermes-2.5-Code-290k-13B)：暂无描述
- [ajibawa-2023/Code-290k-6.7B-Instruct · Hugging Face](https://huggingface.co/ajibawa-2023/Code-290k-6.7B-Instruct)：暂无描述

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1213492101631840336) (128 条消息🔥🔥): 

- **GPT 宕机期间讨论替代方案**：用户 `@whodidthatt12` 对 GPT 宕机表示沮丧，并询问了替代的 AI 写作助手。建议包括 *Bing Chat, Hugginface, Deft GPT, LeChat, together.xyz* 和 *Stable Chat*。 

- **Claude 3 AI 印象**：`@glamrat` 提到测试了 Anthropic 的 Claude 3，认为其表现令人印象深刻，尤其是免费的 Sonnet 模型。多位用户正在讨论他们的体验和期望，从使用 Claude 3 进行数学辅导 (`@reynupj`) 到可能取消 GPT Plus 订阅转而使用 Claude (`@treks1766`)。

- **对 AI 竞争的热情**：`@treks1766` 和 `@lolrepeatlol` 等用户对 Claude 3 和 GPT-4 等 AI 服务之间的竞争表示兴奋，期待这能为消费者带来利益并推动 AI 领域的进步。

- **关于 AI 模型能力的辩论**：一些用户就报道中 Claude 3 优于 OpenAI 模型的情况展开了争论 (`@darthcourt.`, `@hanah_34414`, `@cosmosraven`)，评论从怀疑 (`@drinkoblog.weebly.com`) 到对 OpenAI 下一个重大版本的期待不等。

- **成本考量与可用性**：用户对使用 Claude 3 API 的成本 (`@dezuzel`) 以及不同模型在各地区的可用性表示担忧。人们也在期待 Perplexity AI Pro 等现有服务将如何集成 Claude 3 等新模型 (`@hugovranic`)。

**提到的链接**：

- [Anthropic 称其最新的 AI 机器人可以击败 Gemini 和 ChatGPT](https://www.theverge.com/2024/3/4/24090087/anthropic-claude-3-opus-ai-chatbot-multimodal)：Claude 3 带着重大改进登场。
- [OpenAI 状态](https://status.openai.com/)：未找到描述

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1213485605443346462) (38 条消息🔥): 

- **GPT 宕机期间寻求替代方案**：由于 GPT 宕机，用户 `@whodidthatt12` 正在为写作任务寻找替代的 AI 选项。
- **用于 AI 知识库的自定义 CSV**：`@.bren_._` 询问自定义 Agent 是否可以使用 CSV 文件作为其知识库的一部分，并在确认其是否为有效文件类型时遇到了技术困难。
- **自定义 Agent 中的文件类型与技术支持**：`@.bren_._` 分享了一个关于访问系统根目录的错误消息，而 `@darthgustav.` 建议在纯文本文件中使用行分隔值作为更成功的方案。
- **寻找最理想的代码生成 GPT**：`@yuyu1337` 正在寻找能够生成具有最优时间/空间复杂度代码的 GPT 模型，`@eskcanta` 和 `@beanz_and_rice` 等其他用户也参与了关于实现最优性及提供创意伪代码的讨论。
- **GPT Store 发布路径明确**：`@bluenail65` 询问在商店中列出 GPT 是否必须拥有网站，`@solbus` 澄清了发布选项，包括使用账单名称或通过链接私下分享。
  

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1213497548589899818) (506 条消息🔥🔥🔥): 

- **API 中的幽默感困境**：`@dantekavala` 遇到了一个差异问题，即在 ChatGPT 中进行测试时，提示幽默写作风格的效果很好，但在使用 GPT 3.5 API 时，同样的方法却失败了；API 的输出保持一致，不受所请求风格的影响。他们尝试了各种风格，并在 Developers Corner 寻求指导。
  
- **猫头鹰与棕榈树谜题依然存在**：包括 `@madame_architect`、`@aminelg` 和 `@eskcanta` 在内的许多参与者都对“猫头鹰与棕榈树”脑筋急转弯进行了热烈的探索。虽然他们都尝试了各种 Prompting 策略，试图使用 GPT-V 准确解决该谜题，但都没有取得持续的成功。

- **Prompt Engineering 策略讨论**：用户 `@madame_architect` 建议使用多种 Prompting 策略，如 System 2 Thinking 论文中的“深呼吸（take a deep breath）”技巧以及 EmotionPrompt (tm) 的要点来解决问题。然而，`@eskcanta` 指出，核心问题可能在于 Vision 模型的训练，而不在于 Prompting 方法本身。

- **Vision 模型的局限性**：尽管测试了各种 Prompt 以及关于 Vision 模型对图像测量理解的理论，但 `@eskcanta` 和 `@darthgustav.` 等用户强调，模型无法持续正确解释测量结果，这可能源于需要额外的训练，而不是 Prompting 的不足。

- **关于个人创作的反馈**：新人 `@dollengo` 询问了如何为教育目的创建和训练 AI，并打算发布，但重点是保持在 OpenAI 的对话和分享政策范围内。用户 `@eskcanta` 和 `@aminelg` 提供了尊重平台服务条款和 AI 模型 Prompt 编写实践的建议。

**相关链接**：

- [Terms of use](https://openai.com/policies/terms-of-use)：未找到描述
- [DALL·E 3](https://openai.com/dall-e-3)：DALL·E 3 比我们之前的系统能理解显著更多的细微差别和细节，让你能够轻松地将你的想法转化为异常准确的图像。

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1213497548589899818) (506 条消息🔥🔥🔥): 

- **谜题 Prompt Engineering 传奇继续**：用户 `@aminelg`、`@eskcanta`、`@darthgustav.` 和 `@madame_architect` 继续努力为涉及猫头鹰和树的 AI Vision 谜题构建完美的进阶 Prompt。尽管采取了各种策略，GPT-V 在准确解释图像方面仍然存在问题，引发了关于模型局限性和潜在重新训练需求的讨论。

- **模型行为的起伏**：在多次尝试使用细微差别的 Prompt（如 `@madame_architect` 取得了一次成功的 Prompt）后，GPT-V 始终误解图像右侧 200 个单位的测量值，经常将其与树的全高混淆，这成为模型能力中一个明显的弱点。

- **有趣的竞争升温**：讨论变得幽默起来，`@aminelg` 和 `@spikyd` 互相开玩笑说达到了使用限制，并调侃要生成能超越 AI 目前对复杂图像理解的 Prompt，偶尔出现的正确响应被视为“GPT V 得 10 分”的时刻。

- **分享知识的代价**：`@darthgustav.` 对 Discord 服务器的自动审核表示沮丧，这限制了他讨论某些细节和分享 Prompt 的能力，引发了要求 OpenAI 修改系统 Prompt 限制的呼声，以实现更透明、更有利于 Prompt Engineering 的讨论。

- **新人查询与技巧交流**：`@snkrbots`、`@chenzhen0048` 和 `@dollengo` 等新参与者寻求关于 Prompt Engineering 和 AI 训练的建议，得到了资深贡献者的回应。交流的想法包括通过模板结构改进 Prompt、请求 GPT 协助优化，以及 AI 辅助内容创作任务的潜力。

**相关链接**：

- [Terms of use](https://openai.com/policies/terms-of-use)：未找到描述
- [DALL·E 3](https://openai.com/dall-e-3)：DALL·E 3 比我们之前的系统能理解显著更多的细微差别和细节，让你能够轻松地将你的想法转化为异常准确的图像。

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1213430364501049414) (618 条消息🔥🔥🔥): 

- **潜在的 Perplexity 订阅问题**：用户 `@vivekrgopal` 对在试用期内尝试取消订阅后仍被收取年费表示沮丧。他们请求通过私信协助退款。
- **用户渴望新的 AI 集成**：`@bioforever` 和 `@sebastyan5218` 等用户期待 Perplexity 集成新的语言模型，如 Claude 3 和 Gemini Ultra，凸显了社区对最新 AI 进展的渴望。
- **关于 Perplexity AI 有效性的讨论**：用户 `@names8619` 为 Perplexity Pro 的表现喝彩，认为它在进行无标题党干扰的研究方面优于 YouTube；而其他用户则提到 OpenAI 的 GPT-3 结果存在挑战，在某些主题上需要切换到 Mistral 等模型。
- **AI 模型可用性的不确定性**：用户 `@gooddawg10` 和 `@fluxkraken` 讨论了 Perplexity 中某些 AI 模型（Gemini Ultra, Claude 3）的可用性，对于哪些模型对用户开放存在一些困惑。
- **AI 模型及其回答的对比**：用户 `@dailyfocus_daily` 分享了一个关于将贴标球分类到盒子里的基准测试问题，并对比了包括 GPT-4、Claude 3 在内的不同 AI 模型给出的各种答案，说明了它们在解决问题能力上的不一致性。

**提到的链接**：

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): 未找到描述
- [GitHub Next | GPT-4 with Calc](https://githubnext.com/projects/gpt4-with-calc/): GitHub Next 项目：探索使用计算生成来提高 GPT-4 的数值推理能力。
- [Tweet from Ananay (@ananayarora)](https://fxtwitter.com/ananayarora/status/1762439921825120578): 刚刚将 Perplexity 移植到了 Apple Watch！🚀@perplexity_ai
- [The One Billion Row Challenge in Go: from 1m45s to 4s in nine solutions](https://benhoyt.com/writings/go-1brc/): 未找到描述
- [Oliver Twist GIF - Oliver Twist - Discover &amp; Share GIFs](https://tenor.com/view/oliver-twist-gif-26543489): 点击查看 GIF
- [David Leonhardt book talk: Ours Was the Shining Future, The Story of the American Dream](https://www.youtube.com/watch?v=ovkwsvbGq1I): 加入 Jeff Colgan 教授与《纽约时报》资深作家 David Leonhardt 的对话，讨论 David 的新书，该书探讨了过去一个世纪...
- [SmartGPT: Major Benchmark Broken - 89.0% on MMLU + Exam&#39;s Many Errors](https://youtu.be/hVade_8H8mE): 使用 SmartGPT 系统的 GPT-4 是否以多种方式打破了重大基准测试 MMLU？89.0% 是一个非官方记录，但我们是否迫切需要一个新的、更...
- [Perplexity.ai Turns Tables on Google, Upends SEO Credos](https://spectrum.ieee.org/perplexity-ai): AI 搜索领导者将 Meta 构建的智能与初创公司的拼搏热情相结合
- [PerplexityBot](https://docs.perplexity.ai/docs/perplexitybot): 我们致力于每天改进服务。为了提供最佳的搜索体验，我们需要收集数据。我们使用网络爬虫从互联网收集信息，并为我们的搜索引擎建立索引...
- [GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.](https://github.com/danielmiessler/fabric): fabric 是一个利用 AI 增强人类能力的开源框架。它提供了一个模块化框架，通过众包的 AI Prompt 集来解决特定问题，可在任何地方使用。

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1213417151206264852) (20 条消息🔥): 

- **探索身份访问管理**：用户 `@riverborse` 分享了一个[链接](https://www.perplexity.ai/search/What-is-iam-o2tdFsxGRraeKVWSzo.fIg)，深入探讨了身份访问管理 (IAM) 的含义。
- **理解 Perplexity v2**：`@scarey022` 提供了一个[链接](https://www.perplexity.ai/search/What-is-perplexity-v2XimT_gTp6evpknTUvSUg)，以进一步了解语言模型中 Perplexity（困惑度）的概念。
- **寻找最优解决方案**：用户 `@dtyler10` 发布了一个[链接](https://www.perplexity.ai/search/Create-an-optimal-Nrj9EJpnQ0KSI7vHs8Siaw)，引导至关于创建最优设置、环境或结果的讨论。
- **提供技术见解**：`@imigueldiaz` 分享的一个[链接](https://www.perplexity.ai/search/A-technical-explanation-LEqjHwN4Qa613cx57Dg9AQ?s=m)重点介绍了技术解释。
- **探索 AI 基础知识**：`@krishna08011` 和 `@elpacon64` 分享了链接（[链接1](https://www.perplexity.ai/search/What-are-AI-bQw5YfH6RdOkhyPdrw7XnA)，[链接2](https://www.perplexity.ai/search/What-are-AI-BQsp1PCvS5WWDM0tmj3w0g)），讨论了什么是 AI 及其各个方面。
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1213428411372933151) (27 条消息🔥): 

- **对随机数生成器伦理的困惑**：用户 `@moistcornflake` 对 **codellama** 在被要求创建随机数生成器时提供伦理警告表示好笑和困惑。机器人的回复建议优先考虑促进积极价值观和伦理考量的内容。

- **注意到时效性查询的性能问题**：`@brknclock1215` 观察到整体质量有所提高，但报告称在时效性查询中仍然失败，并回忆说过去在处理此类任务时表现更好。

- **YouTube 摘要功能缺失**：`@rexx.0569` 强调缺少总结 YouTube 视频的功能，这似乎曾是 **Perplexity** 的原生功能。他们注意到该功能在不同设备上都无法访问。

- **关于 Perplexity API 使用的咨询**：`@marvin_luck` 寻求帮助，想知道如何通过 **Perplexity API** 实现与 Web 请求相同的效果。对此，`@icelavaman` 分享了一个 Discord 链接，推测包含相关信息：[Perplexity API 信息](https://discord.com/channels/1047197230748151888/1118264005207793674/1213229028870324305)。

- **用户期待引用功能权限**：`@_samrat` 和 `@brknclock1215` 正在等待获取 API 中的引用 (citations) 访问权限，`@icelavaman` 提到这个过程可能需要 1-2 周或更长时间。`@brknclock1215` 随后确认看到响应质量有所提高，并热切期待引用的加入。

- **Temperature 设置讨论**：`@brknclock1215`、`@thedigitalcat` 和 `@heathenist` 讨论了 AI 模型中的 Temperature 设置如何影响语言输出的自然度和可靠性。他们认为较低的 Temperature 设置并不总是能保证更可靠的输出，并触及了自然语言和 Self-attention 机制的复杂性。

**提到的链接**：

[Perplexity 博客](https://blog.perplexity.ai/faq/what-is-collections)：浏览 Perplexity 的博客，获取文章、公告、产品更新以及优化体验的技巧。保持关注并充分利用 Perplexity。

  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1213481272891146270) (213 messages🔥🔥): 

- **Discord 链接失效**：用户 `@v01338` 和 `_._pandora_._` 提到 Mistral AI 官网上的 Discord 和 LinkedIn 链接都失效了。`_._pandora_._` 通过检查 HTML 源码确认了这一点。
- **关于模型锁定场景的讨论**：`@justandi` 询问在企业背景下从一个模型迁移到另一个模型是否会锁定在特定的实现上。`@mrdragonfox` 插话表示，各平台的推理 API 非常相似，暗示可以无缝迁移。
- **对模型基准测试透明度的担忧**：`@i_am_dom` 对特定 Mistral 模型基准测试缺乏已发布评分表示担忧，认为透明度至关重要，尤其是对于基准测试所有者而言。
- **关于 Mixtral 推理的 Ollama 和 VLLM 讨论**：`@distro1546` 询问如何使用 A100 服务器让 Mixtral 实现亚秒级推理时间，`@mrdragonfox` 建议考虑使用 exllamav2 或 vLLM 部署（6bpw），而不是使用无法充分利用 GPU 能力的 llama.cpp。
- **关于 Mixtral 上下文窗口的澄清**：`@_._pandora_._` 和 `@i_am_dom` 讨论了关于 Mistral 和 Mixtral 上下文大小及滑动窗口（sliding window）功能的困惑。提到了 Reddit 的更新以及 Hugging Face 文档的不准确，强调 HF 需要更新其文档。

**提到的链接**：

- [vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/)：vLLM 可以使用我们提供的 Docker 镜像部署，或者直接从 Python 包部署。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18k0fek/psa_you_can_and_may_want_to_disable_mixtrals/)：未找到描述
- [Mixtral Tiny GPTQ By TheBlokeAI: Benchmarks and Detailed Analysis. Insights on Mixtral Tiny GPTQ.](https://llm.extractum.io/model/TheBlokeAI%2FMixtral-tiny-GPTQ,2VHCHigcDcquIs0aVBv3Ea)：LLM 卡片：90.1m LLM，显存：0.2GB，上下文：128K，已量化。
- [You Have GIF - You Have No - Discover &amp; Share GIFs](https://tenor.com/view/you-have-no-idea-gif-27149353)：点击查看 GIF
- [TheBloke/Mixtral-8x7B-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF)：未找到描述
- [Mixtral](https://huggingface.co/docs/transformers/en/model_doc/mixtral)：未找到描述

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1213412767990685727) (79 messages🔥🔥): 

- **Mistral Large 在编程方面的惊喜**：`@claidler` 报告称，在编程任务中 **Mistral Large** 的表现优于 GPT-4，尽管官方测试显示 GPT-4 更胜一筹。他们观察到 Mistral Large 在 GPT-4 反复失败的地方提供了正确的解决方案，这引发了对测试准确性或在某些场景下适用性的质疑。

- **个人基准测试最重要**：`@tom_lrd` 建议，对模型的**个人经验**应被视为最佳基准，并建议针对特定用例在不同模型上尝试相同的输入以查看其表现。

- **Mistral Next 的速度受到质疑**：`@nezha___` 询问 **Mistral Next** 是否比 Mistral Large 更小，注意到其响应速度更快，并好奇其速度是否源于它是一个 Mixture of Experts (MoE) 模型。

- **上下文大小限制澄清**：`@fauji2464`、`@mrdragonfox` 和 `._pandora_._` 之间的对话讨论了使用 **Mistral-7B-Instruct-v0.2** 时超出模型最大长度的警告。澄清了模型将忽略超过 **32k token** 限制的内容，这会导致性能问题。

- **LLM 上下文窗口解释**：`._pandora_._` 解释说，像 Mistral 和 Mixtral 这样的 Large Language Models (LLMs) 具有“狭窄的视野”，在每个推理周期中只能考虑最多 **32k tokens** 的当前上下文。如果输入超过此限制，多出的内容将被忽略，但模型仍会基于最后 32k tokens 生成输出。

**提到的链接**：

[LLM Visualization](https://bbycroft.net/llm)：未找到描述

  

---

### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1213921912879579179) (17 messages🔥): 

- **寻求在双 3090 上部署 Mistral**：用户 `@generalenthu` 询问了在配备 2x NVIDIA 3090 GPU 的系统上设置 **Mistral** 的最佳方法，目标是实现最小化量化，并就速度与使用 GPU vs RAM 之间的权衡寻求建议。
- **fp16 的 VRAM 需求**：`@mrdragonfox` 告知，使用 fp16 精度运行该模型大约需要 **90GB 的 VRAM**。
- **使用 Exllama 运行模型**：`@mrdragonfox` 提到，在 48GB VRAM 的配置下，可以使用 **exllama** 配置以约 **5-6 bits per word (bpw)** 顺利运行 Mistral。
- **如何开始设置及使用量化模型**：`@mrdragonfox` 建议 `@generalenthu` 从“常规 oobabooga”作为默认设置开始，访问 *N inferences*，并使用 **Hugging Face** 上由 **lonestriker** 和 **turboderp** 提供的量化模型。
- **额外资源与社区支持**：`@mrdragonfox` 建议 `@generalenthu` 加入 “thebloke” 的 Discord，以获得来自协助本地模型部署社区的进一步支持，并指出这可以作为当前社区针对该特定用例的补充。
  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1214156902918000702) (1 messages): 

- **请求极简 Mistral 训练指南**：用户 `@casper_ai` 提到社区在 **Mixtral model** 获得最佳结果方面面临挑战。他们引用了之前的对话，指出 Huggingface trainer 中存在实现差异，并请求提供一个 Mixtral 训练的极简参考实现。
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1213512787087859772) (1 messages): 

- **Smaug-Mixtral 表现优于 Mixtral-8x7b**：`@bdambrosio` 提到，在 8bit exl2 量化测试中，**Smaug-Mixtral** 超过了 **mixtral-8x7b-instruct-v0.1**，特别是在 **长上下文科学推理和中等长度报告撰写** 的应用中。虽然未提供确切的性能指标，但结果可能因用例而异。
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1213782556814741535) (3 messages): 

- **用于离线 LLM Agent 的协作 AI**：用户 `@yoan8095` 分享了他们在离线运行的 **Mistral 7b LLM Agents** 方面的工作，将其与神经符号系统结合以实现更好的规划。该项目在 [GitHub 上的 HybridAGI](https://github.com/SynaLinks/HybridAGI) 开源，允许通过 *基于图的 Prompt 编程 (Graph-based Prompt Programming)* 来编写 AI 行为。
- **功能丰富的 Discord 机器人发布**：`@jakobdylanc` 推广了他们的 Discord 机器人，该机器人能够与 100 多个 **LLMs** 交互，提供 *协作式 Prompting*、*视觉支持* 和 *流式响应* 等功能，且代码量在 200 行以内。项目详情见 [GitHub](https://github.com/jakobdylanc/discord-llm-chatbot)。
- **Mistral-Large 的格式缺陷**：`@fergusfettes` 报告称，虽然 **Mistral-large** 生成的结果不错，但在格式化以及在 *completion mode* 和 *chat mode* 之间切换时表现挣扎。他们分享了一个视频，展示了不同 LLM 的 *loomed* 集成如何工作：[Multiloom Demo: Fieldshifting Nightshade](https://youtu.be/xiQDGxqEals)。

**提到的链接**：

- [Multiloom Demo: Fieldshifting Nightshade](https://youtu.be/xiQDGxqEals)：演示了一个用于将 LLM 输出集成到一份连贯文档中的 loom，通过将一篇计算机科学研究论文“领域迁移 (fieldshifting)”到社会学领域。结果可见...
- [GitHub - jakobdylanc/discord-llm-chatbot: Supports 100+ LLMs • Collaborative prompting • Vision support • Streamed responses • 200 lines of code 🔥](https://github.com/jakobdylanc/discord-llm-chatbot)：支持 100+ LLMs • 协作式 Prompting • 视觉支持 • 流式响应 • 200 行代码 🔥 - jakobdylanc/discord-llm-chatbot
- [GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected](https://github.com/SynaLinks/HybridAGI)：可编程的神经符号 AGI，允许你使用基于图的 Prompt 编程来编写其行为：适用于希望 AI 按预期运行的人。 - SynaLinks/HybridAGI

  

---

### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1214232800719409182) (13 messages🔥): 

- **Kubernetes AI 工具化变得简单**：`@alextreebeard` 分享了他们的开源包，旨在简化在 Kubernetes 上设置 AI 工具的过程，并邀请用户提供反馈。该工具可以在 [GitHub - treebeardtech/terraform-helm-kubeflow](https://github.com/treebeardtech/terraform-helm-kubeflow) 找到。
- **Claude-3 的到来**：`@benjoyo.` 链接了 Anthropic AI 关于其新模型家族 [Claude-3](https://www.anthropic.com/news/claude-3-family) 的公告，并暗示询问何时会发布可与之媲美的 "mistral-huge"。
- **模型训练需要时间**：在回答有关 Mistral 如何应对新竞争的查询时，`@mrdragonfox` 解释说，大型模型的训练需要相当长的时间，大型版本最近才刚刚推出。
- **竞争升温**：在初步测试后，`@benjoyo.` 观察到 Anthropic 的新模型“能力极强且极具可控性/遵循性”，同时继续倡导 Open Weights（开放权重）在差异化方面的价值。
- **讨论新 AI 模型定价**：`@nunodonato` 反思了新模型的高昂成本，而 `@mrdragonfox` 提供了 Opus 模型使用的具体定价，输入成本为每百万 Token (MTok) 15 美元，输出为每百万 Token (MTok) 75 美元。

**提到的链接**：

[GitHub - treebeardtech/terraform-helm-kubeflow: Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐](https://github.com/treebeardtech/terraform-helm-kubeflow)：Kubeflow Terraform 模块 - 在 Kubernetes 中运行 Jupyter 🪐 - treebeardtech/terraform-helm-kubeflow

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1213409058300891136) (82 messages🔥🔥): 

- **NodeJS 中的 Function Calling**：`@jetset2000` 正在寻找在 NodeJS 中使用 Mistral 进行 Function Calling 的文档。`@sophiamyang` 提供了一个有用的回复，其中包含 Mistral AI 的 JS 客户端仓库中的一个[示例](https://github.com/mistralai/client-js/blob/main/examples/function_calling.js)。
- **Mistral Medium 模型超时问题**：`@patrice_33841` 报告了向 mistral-medium-latest 模型发起请求时出现超时。其他用户似乎没有遇到 Medium 模型的问题，`@mrdragonfox` 提供了支持联系信息，建议在技术支持频道发帖或直接发送电子邮件给支持团队。
- **对 Prompt 文档的困惑**：`@benjoyo` 对 Mistral 文档中 User 和 System 消息与实际 Prompt 之间的一致性表示困惑，`@sophiamyang` 承认了这一点并承诺很快会进行澄清。
- **需要澄清响应格式**：`@gbourdin` 遇到了新的 JSON 响应格式问题，引发了关于正确 Prompt 设置的讨论，`@proffessorblue` 根据文档说明进行了澄清，解决了 `@gbourdin` 的问题。
- **探索情感分析的功效**：`@krangbae` 分享了使用不同 Mistral 模型进行情感分析的经验，指出 8x7b 似乎比 Small 模型更有效。

**提到的链接**：

- [Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/)：按需付费
- [Model Selection | Mistral AI Large Language Models](https://docs.mistral.ai/guides/model-selection/)：Mistral AI 提供五个 API 端点，包含五个领先的 Large Language Models：
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#json-mode)：我们提供 Python 和 Javascript 的客户端代码。
- [Function Calling | Mistral AI Large Language Models](https://docs.mistral.ai/guides/function-calling/)：Function Calling 允许 Mistral 模型连接到外部工具。通过将 Mistral 模型与用户定义的函数或 API 等外部工具集成，用户可以轻松构建满足需求的应用程序...
- [client-js/examples/function_calling.js at main · mistralai/client-js](https://github.com/mistralai/client-js/blob/main/examples/function_calling.js)：Mistral AI 平台的 JS 客户端库。通过在 GitHub 上创建账户为 mistralai/client-js 的开发做出贡献。
- [Function Calling | Mistral AI Large Language Models](https://docs.mistral.ai/guides/function-calling/.)：Function Calling 允许 Mistral 模型连接到外部工具。通过将 Mistral 模型与用户定义的函数或 API 等外部工具集成，用户可以轻松构建满足需求的应用程序...
- [Mistral AI API | Mistral AI Large Language Models](https://docs.mistral.ai/api/)：Chat Completion 和 Embeddings API

  

---

### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1213408985500352573) (126 messages🔥🔥): 

- **Mistral Large 风格获赞**：`@foxalabs_32486` 赞赏了 Mistral Large 更加自然且不那么死板的写作风格，同时保持了与 GPT-4 类似的深度。
- **用户界面小问题**：`@steelpotato1` 报告了一个用户界面问题，即在生成过程中 Prompt 和回复会跳动位置，导致用户体验混乱。
- **速率限制困扰与解决方法**：`@shanman6991` 和 `@tom_lrd` 等用户在使用 Chat API 时遇到了速率限制（Rate Limit），引发了关于使用限制的讨论，并建议联系支持部门进行调整。
- **幻觉与误导信息担忧**：`@godefv` 指出 Le Chat 有时会提供错误信息或基于幻觉（Hallucination）而非实际知识生成内容，例如声称某个不存在的博士论文细节。
- **API 使用难题**：`@sim3239` 困扰于 API 和 Le Chat 回复之间的差异，询问 Le Chat 使用了哪些参数，以便在他们自己的 Python 应用程序中复制其完整的回复。

**相关链接**：

- [LLM Tokenizer](https://www.danieldemmel.me/tokenizer.html)：无描述
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/)：我们提供 Python 和 Javascript 的客户端代码。

---

### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1213499316518260777) (13 messages🔥): 

- **Mistral 模型数学错误**：`@propheticus_05547` 发现 **Mistral Instruct 7B v0.2 Q4_K_M** 在使用 Vulkan 加速的 [Jan](https://github.com/janhq/jan) 中运行时，错误地将 `10+3-9+33` 计算为 22 而非正确答案 37，质疑该模型的算术能力。
- **本地运行模型的学习曲线**：针对 `@_._pandora_._` 对 LLM 数学能力较弱的解释，`@propheticus_05547` 注意到当 Prompt 限制在知识和语言类问题时表现有所改善，并分享了另一个版本 **Q5_K_M** 的成功经验，该版本可以处理简单的数学。
- **Mistral 模型抵制 System Prompts**：`@jakobdylanc` 报告称，当被提示为一个名为 Jakobson 的友好 Discord 聊天机器人时，**Mistral Large** 比 **Mistral Medium** 模型更抗拒遵循其 System Prompt。
- **API 暴露方面与 GPT-4 的差异**：`@benjoyo` 观察到 API 上的 **Mistral Large** 往往比 GPT-4 更容易透露其功能特性，而 GPT-4 通常不会向用户暴露此类技术细节。
- **并非所有 Mistral 行为都是可预测的**：针对 **Mistral Large** 中观察到的行为，`@mrdragonfox` 警告不要假设机器人的回复总是有意义的，暗示有些可能纯粹是幻觉（Hallucination）。

---

### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1213984969504854047) (5 messages): 

- **Phi-2 Token 限制困惑**：`@faldore` 质疑了使用超过 2k Token 的 **Phi-2** 的可能性。他们指向了 [Hugging Face 的 Phi-2 模型摘要](https://huggingface.co/microsoft/phi-2)，其中显示了 2k Token 的限制。
- **Phi-2 配置的直接链接**：在后续中，`@faldore` 提供了 Phi-2 配置文件的一个 [直接链接](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19)，显示了 `"max_position_embeddings": 2048` 的设置。
- **关于 Phi-2 Token 扩展的解释**：`@vatsadev` 回应了关于扩展 Phi-2 Token 的问题，指出它的行为会像默认的 Transformer 一样，暗示在超出配置限制后会表现出标准的 Transformer 行为。
- **扩展 Phi-2 能力的警告**：在另一条消息中，`@vatsadev` 警告说，偏离 Phi-2 的配置设置可能会导致模型阻塞或性能不稳定。

**相关链接**：

- [config.json · microsoft/phi-2 at main](https://huggingface.co/microsoft/phi-2/blob/main/config.json#L19)：无描述
- [microsoft/phi-2 · Hugging Face](https://huggingface.co/microsoft/phi-2)：无描述

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1213548816205353001) (31 条消息🔥): 

- **Mac 设置建议大放送**：`@gabriel_syme` 正在寻求 Mac 应用程序建议。许多用户如 `@max_paperclips` 和 `@deki04` 推荐了必备工具，如 Homebrew、适用于 ARM 架构 Mac 的 Parallels、使用 TG Pro 进行温度监控，以及使用 Time Machine 进行备份，`@deki04` 还分享了一位 YouTuber 提供的有用的 [Python/ML Mac 设置技巧](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s)。
- **Better Touch Tool 及更多**：`@denovich` 推荐使用 Better Touch Tool 进行手势控制，通过 Time Machine 设置 Samba 共享，并为需要在 Mac 上使用 Windows 的用户推荐在 Parallels 下运行 Windows 11 ARM。
- **实用 App 与 Homebrew 助阵**：`@eas2535` 强调了 Homebrew 的实用性，分享了[该工具的链接](https://brew.sh)，并列举了 Maccy、Hyperkey、Shortcuts 等实用程序，以提升 Mac 使用效率。
- **人类遗传多样性达到顶峰**：`@teknium` 分享了 `@richardfuisz` 的一条推文，该推文声称人类基因组中每种可能的突变现在都至少存在于 50 个人身上。这引发了 `.ben.com` 对相关科学论文的请求，因为他无法访问该 Twitter 线程。
- **对 tridao 的热情**：`@hexani` 提到 "tridao is so goated"（tridao 太神了），`@teknium` 以猫咪表情符号回应，表示赞同。

**提到的链接**：

- [no title found](https://brew.sh)：未找到描述
- [来自 Richard Fuisz (@richardfuisz) 的推文](https://x.com/richardfuisz/status/1763591765620121990?s=46)：每一个可能存在的突变都确实存在。这在过去约 200 年里才成为现实。大多数有益的变体只是还没有时间变得无处不在。但现在，世界上至少有 50 个人...
- [TG Pro](https://www.tunabellysoftware.com/tgpro/#)：使用 TG Pro 最大化您的 Mac 性能。风扇控制和广泛温度监控的终极解决方案：CPU、GPU、SSD 等。
- [为软件开发设置新的 MacBook](https://www.youtube.com/watch?v=mmkDyV59nRo&t=1368s)：在这里，我将介绍如何为软件开发设置新的 MacBook，这是我通常为自己的任务设置的方式。▶️ 设置新的 M2 Mac Mini - https...

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1213440419803963473) (49 条消息🔥): 

- **规模化与效率的热烈辩论**：`@ldj` 认为 **compute optimality**（计算最优性）在 500B-1T 参数量级之后就会失效，效率的提升来自于 MoE 和训练技术而非单纯的规模。他们引用了 **Sam Altman** 的观点，暗示规模化时代已经结束，未来的收益将来自架构创新，详见[这篇文章](https://www.analyticsvidhya.com/blog/2023/04/the-end-of-the-giant-ai-models-era-openai-ceo-warns-scaling-era-is-over/)和一篇支持性的 **Medium** 文章[这里](https://medium.datadriveninvestor.com/dear-sam-altman-there-was-never-an-era-of-making-models-bigger-288c5f2b743c )。
  
- **对 100T 模型持怀疑态度**：`@intervitens` 和 `@.ben.com` 对训练 100T 参数模型的可行性和实用性表示怀疑，质疑硬件能力和数据可用性。`@euclaise` 提出了反驳，认为存在足够的数据资源，如 Redpajama v2。

- **小型模型的潜力**：`@ldj` 进一步强调大模型并不一定更好，指出 GPT-4 在约 200B 激活参数下的表现可能优于超过 500B 激活参数的模型。`@teknium` 表示不同意，认为如果结合充足的训练数据，**parameter scaling**（参数规模化）仍可能是有益的。

- **AI 规模化的成本担忧**：`@ldj` 提出了关于模型规模化成本效益的实际担忧，暗示增加参数数量可能会导致训练和推理成本高得令人望而却步。

- **近期 AI 模型对比参考**：`@mautonomy` 分享了一个 **Reddit 帖子**链接，其中包含 17 个新模型的对比，总计 64 个排名，频道内其他人未对此发表评论。

**提到的链接**：

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：高效服务大语言模型（LLMs）需要将许多请求批处理在一起，以降低每个请求的成本。然而，存储 Attention keys 和 values 以避免重复计算的 KV Cache...
- [- Fuck You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/)：通过拦截 API 调用，快速理解那些难以捉摸的 LLM 框架。
- [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516)：激活稀疏性是指激活输出中存在大量贡献较弱的元素。作为使用 ReLU 激活函数的模型的一种普遍属性，它一直被...
- [Technological Approach to Mind Everywhere | Michael Levin](https://www.youtube.com/watch?v=JC4FOzAuHF4)：摘自“进化、基础认知和再生医学”，由 Michael Levin 在 SEMF 2023 跨学科夏令营中提供 (http...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/)：未找到描述
- [Dear Sam Altman- There was never an era of making models bigger](https://medium.datadriveninvestor.com/dear-sam-altman-there-was-never-an-era-of-making-models-bigger-288c5f2b743c)：LLMs 从来没有像网上的大师们让你相信的那样具有革命性或改变游戏规则。
- [The End of the Giant AI Models Era: OpenAI CEO Warns Scaling Era Is Over](https://www.analyticsvidhya.com/blog/2023/04/the-end-of-the-giant-ai-models-era-openai-ceo-warns-scaling-era-is-over/)：了解 OpenAI CEO Sam Altman 对 ChatGPT 等 AI 模型未来进展的看法，以及 GPU 的获取为何仍然至关重要。

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1213447897216057364) (328 条消息🔥🔥):

- **AI for Music**：用户 `@audaciousd` 期待新的音乐生成式 AI，特别关注一家名为 stabilities 的公司即将发布的内容。他们询问了其他人对该话题的了解。
- **Claude 3 引发热议**：关于 Claude 3 发布的讨论：`@fibleep` 引用了一份公告，`@4biddden` 和 `@mautonomy` 等用户推测其与 GPT-4 的性能对比。
- **GPT-4 vs. Claude 3 观点**：包括 `@teknium` 在内的几位用户通过 [Twitter 投票](https://x.com/teknium1/status/1764732905660830024?s=46) 分享并寻求反馈，探讨 Claude 3 Opus 是否真的比 GPT-4 更好。
- **B2B 销售策略分享**：用户 `@mihai4256` 征求销售 B2B 软件产品的建议，促使 `@hexani` 分享了针对小型企业的经验以及相关的挑战和策略。Hexani 强调了直接接触的必要性以及对产品可行性的高标准要求。
- **知识图谱构建资源探索**：用户 `@mihai4256` 和 `@everyoneisgross` 讨论了创建知识图谱的模型和方法，`@max_paperclips` 建议使用 Hermes 进行 JSON 结构化三元组提取。他们分享道，一个具有改进的结构化数据提取能力的新模型即将发布。

**提到的链接**：

- [AI in Production - AI 策略与战术](https://www.aiinproduction.com/cfp)：未找到描述
- [来自 roon (@tszzl) 的推文](https://x.com/tszzl/status/1493816776731205643?s=46)：是的，这是正确的看法。GPT4 即将到来。2 万亿，宝贝 ↘️ 引用 bayes (@bayeslord)：我觉得 OpenAI 员工突然开始讨论对齐 (alignment) 和有意识的语言模型是一个暗示...
- [来自 Teknium (e/λ) (@Teknium1) 的推文](https://x.com/teknium1/status/1764737777667891291?s=46)：Claude 3 Opus 比 GPT4 更好吗？新的投票，因为上一个太模糊了，而且我没有设置查看结果选项 ██████ 是 (18.8%) ███ 否 (10.6%) ██████████████████████ 查看结果 (70.6%)...
- [来自 db (@dxyba) 的推文](https://x.com/dxyba/status/1763756934262321574?s=20))：职位被取消，角色终止。我现在有 5 个月的时间找工作——普通州立大学毕业，无工作经验，没刷过 LeetCode，没做过项目。是时候拼命工作，实现人生最大的逆转了...
- [wandb/gemma-2b-zephyr-dpo · Hugging Face](https://huggingface.co/wandb/gemma-2b-zephyr-dpo)：未找到描述
- [来自 Yam Peleg (@Yampeleg) 的推文](https://fxtwitter.com/Yampeleg/status/1763532342482612625?s=20)：由 Mistral-7B 在 GPT-4 Turbo 生成的合成数据集上合并训练而成。这是持续预训练，但在样本中包含一些指令（但它们未对齐到...）
- [来自 virat (@virattt) 的推文](https://x.com/virattt/status/1764363199049072743?s=46)：我被 RAGAS 震撼了。只需 10 行代码，我就创建了一个关于 Airbnb 最新年报 (10-K) 的问答数据集。该数据集包含 3 部分：• 问题 • 上下文 • 基准答案 (ground truth)...
- [来自 John Nay (@johnjnay) 的推文](https://x.com/johnjnay/status/1764470331568238618?s=46)：LLM 预测能力媲美人类准确度。在为期 3 个月的预测竞赛中，12 个 LLM 组成的群体 vs 925 位人类预测者。LLM 群体在统计学上等同于人类群体...
- [来自 Tsarathustra (@tsarnick) 的推文](https://x.com/tsarnick/status/1763756693610184811?s=46)：Nick Chater：AI 语言模型无法创造新知识，因为它们只是在反映我们已经知道的内容。
- [GPT4All 文档](https://docs.gpt4all.io/gpt4all_python_embedding.html")：未找到描述
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1764653830468428150?s=20)：今天，我们发布 Claude 3，我们的下一代 AI 模型。这三个最先进的模型——Claude 3 Opus, Claude 3 Sonnet, 和 Claude 3 Haiku——在推理等方面树立了新的行业标杆...
- [来自 Teknium (e/λ) (@Teknium1) 的推文](https://x.com/teknium1/status/1764732905660830024?s=46)：所以它真的比 GPT 4 更好吗？ ████████████████ 是 (52%) ███████████████ 否 (48%) 960 票 · 剩余 21 小时
- [Google Colaboratory](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_tacred.ipynb)：未找到描述
- [studio-ousia/luke-large-finetuned-tacred · Hugging Face](https://huggingface.co/studio-ousia/luke-large-finetuned-tacred)：未找到描述
- [Reddit - 深入讨论](https://www.reddit.com/r/LocalLLaMA/comments/16lnvv1/is_it_legal_to_use_gpt4_output_to_finetune_llama2/)：未找到描述
- [来自 Bin Lin (@LinBin46984) 的推文](https://x.com/LinBin46984/status/1763476690385424554?s=20)：👏👏👏 我们很高兴启动一个名为 Open-Sora 的计划，旨在复现 OpenAI 的（“CloseAI”🤪）Sora。该项目目前支持🎉🎉🎉：(1) 🚀 可变宽高比 (2) ✈️ 可变...

- [来自 Blaze (Balázs Galambosi) (@gblazex) 的推文](https://x.com/gblazex/status/1764664048522600690?s=20)：Claude 3 Opus (output) 非常昂贵。它确实具有扎实的推理分数，所以我们将看看它是否值得这些额外成本。但 GPT-4 Turbo 仍然是性价比最高的高端方案...
- [euclaise (Jade)](http://hf.co/euclaise)：未找到描述
- [GitHub - AnswerDotAI/fsdp_qlora: 使用 QLoRA + FSDP 训练 LLM](https://github.com/AnswerDotAI/fsdp_qlora)：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账户来为 AnswerDotAI/fsdp_qlora 的开发做出贡献。
- [llama : 添加 T5 (encoder-decoder) 支持 · Issue #5763 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5763)：仍然不熟悉细节，但在 llama.cpp 中支持这种架构似乎很有用。首先，需要决定 API 并查看需要哪些更改。请参阅讨论...
- [laion/OIG · Hugging Face 数据集](https://huggingface.co/datasets/laion/OIG)：未找到描述
- [google-coral](https://github.com/google-coral)：coral.ai 的开源项目。google-coral 有 37 个可用的代码库。在 GitHub 上关注他们的代码。
- [使用 Direct Preference Optimization 对齐 LLM](https://youtu.be/QXVCqtAZAn4?t=1512)：在本次研讨会中，来自 Hugging Face 的 Lewis Tunstall 和 Edward Beeching 将讨论一种强大的对齐技术，称为 Direct Preference Optimisation (DPO)...
- [GitHub - parthsarthi03/raptor: RAPTOR 的官方实现：Recursive Abstractive Processing for Tree-Organized Retrieval](https://github.com/parthsarthi03/raptor)：RAPTOR 的官方实现：Recursive Abstractive Processing for Tree-Organized Retrieval - parthsarthi03/raptor
- [LUKE](https://huggingface.co/docs/transformers/model_doc/luke)：未找到描述

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1213437854038884352) (32 条消息🔥): 

- **PPO 脚本查询**：`@xela_akwa` 正在寻找用于 LLM 的 PPO 的 PyTorch 或 PyTorch Lightning 脚本，发现 hf trl (HuggingFace Transformers Reinforcement Learning) 功能有限。随后展开了讨论并提出了潜在的替代方案，包括 `@.mahouko` 提供的一个相关工作的 GitHub 仓库。目前尚未提供 PPO 的最终解决方案。
- **函数调用模型服务器对决**：`@giulio123456` 询问关于最快函数调用模型的最佳推理平台。`@sundar_99385` 和 `@dustinwcarr` 的回复建议 Anyscale 和 Deepinfra 是支持 Mistral/Mixtral 且性能显著的平台，但未提供直接的延迟对比。
- **ChatML 中的 1-Shot 格式**：`@cognitivetech` 询问关于使用 ChatML 进行 1-shot 训练的正确模板，重点是 system-user 交互；`@teknium` 确认正确格式不包含 'name=' 约定，并支持更简单的模板。
- **LLaMa 架构澄清**：`@qtnx` 询问 LLaMa 1 和 1.5 架构中 patch embedding 转换的具体细节，收到了 `@teknium` 的简短确认但未给出具体细节，随后 `@qnguyen3` 进行了追问。
- **AI 辅助聊天注意事项**：`@betim01` 讨论了针对客户交互微调 AI 模型的策略，考虑了 Nous Hermes 和 RAG。`@teknium` 警告了潜在的负面影响，并举了 ChatGPT 在经销商场景中被戏弄的例子，建议采用更可靠的 RAG 方法，并列出了潜在的推理平台选项。
- **语言模型的下一步**：`@pier1337` 推测了语言模型的未来，提到了 Sutskever 关于对象驱动 AI 的观点以及在模拟环境中的潜在应用，但在提供的聊天记录中没有对这一预测的直接回应。

**提到的链接**：

- [来自 Chris Bakke (@ChrisJBakke) 的推文](https://x.com/chrisjbakke/status/1736533308849443121?s=46)：我刚刚花 1 美元买了一辆 2024 款雪佛兰 Tahoe。
- [一家汽车经销商在其网站上添加了 AI 聊天机器人。然后场面失控了。](https://www.businessinsider.com/car-dealership-chevrolet-chatbot-chatgpt-pranks-chevy-2023-12)：恶作剧者发现他们可以利用当地雪佛兰经销商网站上由 ChatGPT 驱动的机器人做更多事情，而不仅仅是谈论汽车。

  

---

### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1214171242362306580) (2 条消息): 

- **探索 Moondream**: 用户 `@ee.dd` 分享了他们对 **Moondream** 的正面使用体验，在经过一些测试后强调了其速度和效果。他们提供了 GitHub 链接：[Moondream - tiny vision language model](https://github.com/vikhyat/moondream)。

**提到的链接**:

[GitHub - vikhyat/moondream: tiny vision language model](https://github.com/vikhyat/moondream): tiny vision language model。通过在 GitHub 上创建账号，为 vikhyat/moondream 的开发做出贡献。

  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1213547327097274369) (197 messages🔥🔥): 

- **开源 AI 对齐**：Open Source Initiative (OSI) 将每月发布一份开源 AI 定义的新草案，目标是在 2024 年 10 月底发布 1.0 版本，并在其公共论坛进行讨论，[草案文档可供审阅](https://opensource.org/deepdive/drafts)。

- **针对 DMCA 的法律诉讼**：EFF 提起了一项诉讼 *Green v. Department of Justice*，挑战 DMCA 的反规避和反贩运条款，认为其限制了对已购买版权材料的访问。[案件详情](https://www.eff.org/cases/green-v-us-department-justice)。

- **神经网络量化辩论**：关于神经网络权重和激活量化的实用性及其影响展开了讨论。用户对 [bitlinear 论文](https://arxiv.org/pdf/2310.11453.pdf) 以及量化激活函数的概念进行了辩论，并引用了认知不确定性 (epistemic uncertainty) 等概念。

- **GitHub 恶意软件传播活动**：GitHub 上的一项恶意软件分发活动导致合法仓库被克隆、注入恶意软件并推广受损代码。[Apiiro 的安全分析](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/) 详细解释了这一威胁。

- **预测建模局限性的讨论**：用户 `@rallio.` 断言，目前无法通过预测建模从头开始 (de novo) 创建具有经济可行性的生物分子，并认为生物系统的复杂性使其具有不可预测性，这与用于工程的物理模型不同。

**提到的链接**：

- [Green v. U.S. Department of Justice](https://www.eff.org/cases/green-v-us-department-justice)：Green v. Department of Justice 是 EFF 的一项诉讼，基于第一修正案挑战《数字千年版权法》(DMCA) 反规避和反贩运条款的合宪性...
- [GitHub struggles to keep up with automated malicious forks](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/)：被克隆后遭到破坏，恶意仓库的 fork 速度超过了其被删除的速度。
- [Turing jump - Wikipedia](https://en.wikipedia.org/wiki/Turing_jump)：未找到描述。
- [Scientists aghast at bizarre AI rat with huge genitals in peer-reviewed article](https://arstechnica.com/science/2024/02/scientists-aghast-at-bizarre-ai-rat-with-huge-genitals-in-peer-reviewed-article/)：目前尚不清楚如此糟糕的图像是如何通过同行评审的。
- [jax/docs/multi_process.md at main · google/jax](https://github.com/google/jax/blob/main/docs/multi_process.md)：Python+NumPy 程序的可组合变换：微分、向量化、JIT 到 GPU/TPU 等 - google/jax
- [Ergonomic way to extract a single iteration output from a scan · google/jax · Discussion #20054](https://github.com/google/jax/discussions/20054)：提取网络中单个隐藏层的激活是很常见的，但在对参数使用 scan 时会变得很麻烦。这是一个玩具示例：import jax import jax.numpy as jn...
- [GitHub - davisyoshida/gemma-flax: Implementation of Gemma in Jax/Flax](https://github.com/davisyoshida/gemma-flax)：Gemma 在 Jax/Flax 中的实现。通过在 GitHub 上创建账号为 davisyoshida/gemma-flax 的开发做出贡献。
- [Attention entire world!!!](https://professorhooh.substack.com/p/attention-entire-world-f1f)：我向你发起挑战：The Game
- [google/jax · Discussions](https://github.com/google/jax/discussions)：探索 google jax 的 GitHub Discussions 论坛。讨论代码、提问并与开发者社区协作。
- [All-atom Molecular Dynamics Simulation of the Bacterial Cytoplasm](https://www.youtube.com/watch?v=5JcFgj2gHx8)：细菌细胞质的全原子分子动力学模拟：生物分子在拥挤的细胞环境中如何表现一直是生命科学中的重要问题。理化学研究所 (RIKEN) 和密歇根州立大学的研究人员...
- [Drafts of the Open Source AI Definition](https://opensource.org/deepdive/drafts)：开源 AI 定义的草案。我们正在发布已发布的草案文档。查看下方的各个草案，了解如何留下评论的说明。

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1213413741757202432) (115 messages🔥🔥): 

- **反事实样本提升 AI 的视觉-语言推理能力**：`@digthatdata` 分享了一种名为 **CounterCurate** 的新方法，详见 [研究论文](https://countercurate.github.io/)，该方法提高了多模态模型中的视觉-语言组合推理能力。CounterCurate 利用 **GPT-4V** 和 **DALLE-3** 创建反事实的图像-标题对，在 SugarCrepe 等基准测试中实现了更高的性能。

- **Functional Benchmarks 挑战 LLMs**：`@.the_alt_man` 指向了 `@_saurabh` 的一条 [Twitter 线程](https://x.com/_saurabh/status/1763626711407816930?s=20)，暗示 **超过 50% 的 LLMs 报告的推理能力可能并非真正的推理**。该线程讨论了一篇介绍 functional benchmarks 的论文，揭示了 SOTA 模型中显著的推理差距，并附有相关的 [arXiv 草案](http://arxiv.org/abs/2402.19450) 和 [GitHub 仓库](https://github.com/ConsequentAI/fneval)。

- **SQuADv2 中不可回答问题的 Contrastive Learning**：`@paganpegasus` 询问了在 SQuADv2 的不可回答问题中，为 Contrastive Learning 创建负样本的最佳方法，建议使用 Spacy 提取名词短语（noun chunks）作为潜在的负样本。`@fern.bear` 建议使用由模型标记的、SQuADv2 特有的、具有最大置信度的答案集作为另一种方法。

- **关于 RLHF 对模型能力影响的担忧**：`@.the_alt_man` 和 `@canadagoose1` 的讨论围绕 Reinforcement Learning from Human Feedback (RLHF) 对模型能力的影响展开，怀疑 RLHF 可能由于实现不佳而导致性能下降。

- **Terminator 架构：AI 的潜在游戏规则改变者？**：`@fredholm` 强调了 [arXiv 论文](https://arxiv.org/abs/2401.17948) 中描述的 **Terminator** 网络，该论文提出了一种新架构，可能用大型隐式核（large implicit kernels）取代残差学习（residual learning），以实现全上下文交互。在随后的对话中，`@harvie_zhang_32234` 和 `@alex_cool6` 确认了 Terminator 的独特方法，后者表示计划将其应用于图像生成并在未来发布代码。

**提到的链接**：

- [Saurabh Srivastava (@_saurabh) 的推文](https://x.com/_saurabh/status/1763626711407816930?s=20): 超过 50% 被报道的 LLM 推理能力可能并非真正的推理。我们该如何评估在整个互联网数据上训练的模型？也就是说，对于一个已经见过几乎所有内容的东西，我们可以提出哪些新颖的问题？
- [Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1764653830468428150?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): 今天，我们发布了 Claude 3，我们的下一代 AI 模型。这三款尖端模型——Claude 3 Opus, Claude 3 Sonnet 和 Claude 3 Haiku——在推理等各项指标上树立了新的行业基准...
- [AtP*: 一种将 LLM 行为定位到组件的高效且可扩展的方法](https://arxiv.org/abs/2403.00745): Activation Patching 是一种直接计算行为对模型组件因果归因的方法。然而，详尽地应用它需要进行扫描，其成本随组件数量线性增加...
- [KIVI: 一种用于 KV Cache 的免微调非对称 2bit 量化](https://arxiv.org/abs/2402.02750): 高效地提供大语言模型 (LLM) 服务需要将多个请求打包在一起以降低单个请求的成本。然而，存储 Attention 的 Key 和 Value 以避免重复计算的 KV Cache...
- [领域特定张量语言](https://arxiv.org/abs/2312.02664): 在多个数学领域中使用的张量符号非常有用，但在函数式编程社区中并未广泛使用。从实用角度来看，（嵌入式）领域特定语言...
- [超越语言模型：Byte 模型是数字世界模拟器](https://arxiv.org/abs/2402.19155): 传统的深度学习往往忽略了 Byte（字节），这是数字世界的基本单位，所有形式的信息和操作都以二进制格式进行编码和处理。受...成功的启发
- [CounterCurate](https://countercurate.github.io/): 未找到描述
- [在扩展 Transformer 时，稀疏性就足够了](https://arxiv.org/abs/2111.12763): 大型 Transformer 模型在许多任务上取得了令人印象深刻的结果，但训练甚至微调的成本都很高，且解码速度非常慢，以至于它们的使用和研究变得遥不可及。我们解决了这个...
- [Mega: 配备移动平均的门控注意力机制](https://arxiv.org/abs/2209.10655): Transformer 注意力机制的设计选择，包括弱归纳偏置和二次计算复杂度，限制了其在长序列建模中的应用。在本文中...
- [HyperZ$\cdot$Z$\cdot$W 算子连接快慢网络以实现全上下文交互](https://arxiv.org/abs/2401.17948): Self-attention 机制利用大型隐式权重矩阵，通过基于点积的激活函数（仅含极少可训练参数）进行编程，从而实现长序列建模。在本文中...
- [如何扩展你的 EMA](https://arxiv.org/abs/2307.13813): 在不同的 Batch Size 之间保持训练动态是实际机器学习中的一个重要工具，因为它允许在 Batch Size 和实际耗时（wall-clock time）之间进行权衡。这种权衡通常通过...
- [Maracas Jimcarrey GIF - Maracas Jimcarrey Jim - 发现并分享 GIF](https://tenor.com/view/maracas-jimcarrey-jim-celebrate-dance-gif-5055069): 点击查看 GIF
- [随着 Batch Size 增加如何扩展超参数](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/): 未找到描述
- [APAR: LLM 可以进行自动并行自回归解码](https://arxiv.org/abs/2401.06761): 大语言模型 (LLM) 的大规模采用需要高效的部署策略。然而，作为大多数 LLM 生成文本基础的自回归解码过程，给...带来了挑战。
- [使用深度表示检测异常蛋白质](https://doi.org/10.1093/nargab/lqae021): 摘要。生物医学领域的许多进展可以归功于对异常蛋白质和基因的识别。这些蛋白质的许多独特特性是通过...发现的。

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1214275379494395945) (1 条消息): 

- **在动画中创造性地使用 Figma**: 用户 `@kyo_takano` 描述了他们制作动画的过程：他们在 **Figma 中制作了一个 SVG 模板**，通过操作它来组合不同的帧，然后使用 **imageio** 将这些帧混合成 **GIF 动画**。
  

---

### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1213428209148760064) (50 messages🔥): 

- **Mamba vs Transformers 在学习 PARITY 上的表现**：`@dashiell_s` 报告称，一个双层 Mamba 模型可以学习长度达 128 的序列的 PARITY（奇偶校验），但在更长序列上的泛化效果不佳。测试显示 Mamba 的表现远优于类似配置的 Transformer，后者在处理长度超过 64 的序列时表现挣扎。
  
- **对结合律递归架构效率的怀疑**：`@norabelrose` 对基于结合律递归关系（associative recurrence relations）的架构能否高效学习 PARITY 表示怀疑，并建议通过实验来对比 LSTM 和 Mamba 的性能。

- **机器学习文献中对敏感度（Sensitivity）的可能误解**：`@stellaathena` 指出，一篇讨论“敏感度”的论文实际上是指平均敏感度（average sensitivity）而非最大敏感度（maximum sensitivity），这可能意味着不同的理论含义。

- **在 PARITY 上训练 Mamba**：`@dashiell_s` 分享了他们在 PARITY 问题上对 Mamba 进行的实验，结果和代码可在 GitHub 上获取 ([train_mamba.py](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py))。

- **辩论学习 PARITY 的机制**：围绕模型学习的是查找表（lookup table）还是实际的 PARITY 计算，展开了多项讨论（`@norabelrose` 和 `@dashiell_s`）。此外，大家也对更深的 Transformer 是否能找到更复杂的解决方案感到好奇。

**提到的链接**：

- [jax.lax.associative_scan &#8212; JAX  documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html)：未找到描述
- [Prefix sum - Wikipedia](https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithms)：未找到描述
- [automatic-circuits/train_mamba.py at main · dashstander/automatic-circuits](https://github.com/dashstander/automatic-circuits/blob/main/train_mamba.py)：通过在 GitHub 上创建一个账号来为 dashstander/automatic-circuits 的开发做出贡献。

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1213591513821610024) (71 条消息🔥🔥): 

- **AzureML 上的 `lm-eval-harness` 问题**：`@synthetic_johnny` 在 AzureML 计算集群上设置 `lm-eval-harness` 时遇到了问题，包括依赖关系和 CUDA 设备检测问题。随后展开了关于寻找正确环境构建的讨论，`@hailey_schoelkopf` 针对该工具的使用细节提供了指导，包括多 GPU 使用以及模型与 AzureML 的兼容性详情。

- **多机并行挑战**：`@hailey_schoelkopf` 澄清了 `lm-eval-harness` 不支持多机并行，这正是导致 `@synthetic_johnny` 出现问题的原因。`@rand0mm` 建议了一个变通方案，分享了 [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)，它可以帮助在不同节点上编排 `lm-eval-harness` 的执行。

- **在单节点上处理大型模型**：`@hailey_schoelkopf` 向 `@synthetic_johnny` 建议，在评估类似 GPT-J-6B 的 LLM 时，可以使用 `model_args parallelize=True` 和 `dtype=bfloat16` 将模型分散到单个节点内的多个 GPU 上，并从 batch size 为 1 开始以避免显存溢出（out-of-memory）错误。讨论还涉及了在 AzureML 上使用时，模型并行（model-parallel）优于数据并行（data-parallel）配置的重要性。

- **关于 LAMBADA 训练数据使用的困惑**：`@smerkyg` 询问了关于正确使用 LAMBADA 数据集训练 LLM 的问题。`@hailey_schoelkopf` 澄清说，最好不要在 LAMBADA 训练集上进行微调（finetune），因为该基准测试现在的目的是评估通用语言建模能力。

- **寻求 Python 环境下多 GPU 运行 HELLASWAG 的示例**：`@antonvls` 询问了在 Python 中使用多 GPU 运行 HELLASWAG 评估的示例或成功案例。`@stellaathena` 引导他们查看该库的自动化多 GPU 处理功能，并提供了一个 [GitHub 链接](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate) 以获取进一步指导。

**提到的链接**：

- [Ray Serve: Scalable and Programmable Serving — Ray 2.9.3](https://docs.ray.io/en/latest/serve/index.html)：未找到描述
- [lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#multi-gpu-evaluation-with-hugging-face-accelerate)：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/) (1 条消息): 

besiktas：还没真正看到过相关内容，我也一直在思考/实验这一点。
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1213722504741396480) (2 条消息): 

- **The Pile 的处理脚本**：用户 `@catboy_slim_` 分享了一个 [GitHub 链接](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts)，其中的 README 文件可能会有帮助。该仓库包含与 **The Pile**（一个用于训练语言模型的大规模数据集）开发相关的脚本。
- **验证数据路径查询**：用户 `@pietrolesci` 询问了 wandb 日志中 run `v2 1.4B deduped_1dhzgs7f` 提到的**验证数据文件**。他们试图了解该文件是否是从去重后的 Pile 中随机抽样的。

**提到的链接**：

[the-pile/processing_scripts at master · EleutherAI/the-pile](https://github.com/EleutherAI/The-Pile/tree/master/processing_scripts)：通过在 GitHub 上创建一个账户来为 EleutherAI/the-pile 的开发做出贡献。

  

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1213428130828521502) (155 条消息🔥🔥): 

- **LM Studio 中的模型故障排除**：`@helloxan.` 在 LM Studio 中使用 Codellama Python 7B 模型时遇到了问题，并寻求如何让机器人做出响应的帮助。`@heyitsyorkie` 提供了协助，建议使用来自 Hugging Face 的另一个模型（[Magicoder-S-DS-6.7B-GGUF](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF)），并就解决“损坏的量化（broken quant）”问题提供了指导。
- **关于模型支持和功能的疑问**：`@ciphersson`、`@justmarky` 和 `@archi_95` 等用户询问了在 LM Studio 中加载特定模型（如 LoRAs、QLoRA 和 starCoder 2）以及上传 PDF 文件的问题。`@heyitsyorkie` 澄清说，目前尚不支持 QLoRA 和 starCoder 2 等功能，且无法直接上传 PDF。
- **讨论 LM Studio 的技术困难**：`@sourguava`、`@shadowdoggie` 和 `@boting_0215` 等几位用户遇到了技术问题，包括模型加载时间过长以及遇到未提供说明的错误。
- **模型预设与参数探讨**：用户正在寻求并分享有关获取更多预设的信息（`@techfren` 分享了一个 [YouTube 视频资源](https://youtu.be/LUiVbOeLeas?si=y96mVuTitAFjX_Zq)），了解模型量化参数（`@unkown101`），以及更改代码生成的随机性设置的效果（`@drawless111`）。
- **LLM 的 GPU 需求与性能**：包括 `@ethanboyle`、`@broski_1337` 和 `@ocn` 在内的多位用户探讨了所需的硬件规格，如 GPU offloading 以及高效利用模型对高性能 GPU 的必要性。`@heyitsyorkie` 建议，运行大型语言模型时，为了保证速度和效率，至少需要 24GB 显存（VRAM）的 GPU。

**提到的链接**：

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/careers)：查找、下载并实验本地 LLM。
- [no title found](https://www.marktechpost.com/2024/03/03/meet-phind-70b-an-artificial-intelligence-ai-model-that-closes-execution-speed-and-the-code-generation-quality-gap-with-gpt-4-turbo/)：未找到标题（关于 Phind 70B AI 模型的文章，该模型缩小了与 GPT-4 Turbo 在执行速度和代码生成质量上的差距）。
- [itsdotscience/Magicoder-S-DS-6.7B-GGUF · Hugging Face](https://huggingface.co/itsdotscience/Magicoder-S-DS-6.7B-GGUF)：未找到描述。
- [ItsD (D)](https://huggingface.co/itsd)：未找到描述。
- [AI solves huge problem holding back fusion power](https://flip.it/Go0HL0)：普林斯顿大学的研究人员训练了一个 AI，用于预测并防止核聚变反应中的常见问题。
- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family)：今天，我们发布了 Claude 3 模型家族，它在广泛的认知任务中树立了新的行业基准。该家族包括三个按能力递增排序的最先进模型...
- [LM Studio Models not behaving? Try this!](https://youtu.be/LUiVbOeLeas?si=y96mVuTitAFjX_Zq)：免费预设仓库：https://github.com/aj47/lm-studio-presets ➤ Twitter - https://twitter.com/techfrenaj ➤ Twitch - https://www.twitch.tv/techfren...
- [afrideva/TinyLlama-con-creative-writing-v0.2-GGUF · Hugging Face](https://huggingface.co/afrideva/TinyLlama-con-creative-writing-v0.2-GGUF)：未找到描述。
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui)：大语言模型的 Gradio Web UI。支持 Transformer, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1213430946599018556) (49 条消息🔥): 

- **对模型泄露个人数据的担忧**：用户 `@tobitege` 分享了从模型中获得的一个意外且无关的回复，引发了对数据隐私的担忧。`@tay2win` 推测这可能是数据抓取（data scraping）的情况，例如来自 LinkedIn 或 GitHub，他们认为这应该是违法的。
- **对 Hugging Face 模型来源的质疑**：`@tobitege` 在发现一个真实人物与模型无关回复中给出的名字匹配后表示不安。这引发了关于训练数据来源的讨论，`@tay2win` 希望电子邮件和聊天记录不会被用于训练 AI。
- **解决对模型过拟合（Overfitting）的误解**：当 `@tay2win` 建议过拟合可能是原因时，`@aswarp` 澄清了 AI 模型“反刍（regurgitating）”数据的概念。`@aswarp` 指出这是一个已知问题，当模型重复训练数据的片段时就会发生。
- **关于在 LM Studio 中使用 Grok 的困惑**：在关于将 Grok 与 LM Studio 集成的对话中，`@pandora_box_open` 提供了 Groq.com 的链接，但对意图的澄清导致了来自 `@wildcat_aurora` 和 `@jedd1` 的不同反应和修正。
- **寻找适合 VRAM 和上下文大小（Context Size）的模型**：`@jason_2065` 与 `@heyitsyorkie` 和 `@vinni_spx` 交流了关于适合编程的各种模型、它们的 VRAM 占用、上下文长度以及在 Hugging Face 上使用过滤器的需求。`@jason_2065` 还询问了“混合专家（mixture of experts）”模型，提到 Laser Dolphin Mixtral 在速度和 VRAM 适配方面表现良好。

**提到的链接**：

- [GroqChat](https://groq.com/)：未找到描述
- [混合专家模型详解](https://huggingface.co/blog/moe)：未找到描述
- [TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF · Hugging Face](https://huggingface.co/TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF)：未找到描述

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1213560308120158229) (5 条消息): 

- **寻求非 API CURL 指导**：`@newoperator` 询问如何在不使用 OpenAI completion API 的情况下直接向 LM Studio 模型发送 curl 请求，并指出缺乏相关文档。`@fabguy` 回复说 CURL 可以直接与 LM Studio 服务器交互，无需额外的 API。
- **LM Studio 中的错误代码难题**：`@instamailing` 发布了一个 LM Studio 的问题，特征是退出错误代码 (-1073740791) 以及随附的显示 RAM 和 VRAM 详情的 JSON 数据，但没有明确的原因。
- **引导至正确的支持频道**：`@heyitsyorkie` 将 `@instamailing` 引导至处理其所遇问题的适当支持频道，并建议提供比错误消息更多的信息以获得更好的帮助。
  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1213542955915419808) (114 messages🔥🔥): 

- **16GB VRAM 与关于 Mac Pro 的辩论**：`@ethanboyle` 指出消费级显卡的 VRAM 上限似乎是 16GB。在讨论关于搭载 Apple Silicon 的 MacBook Pro 的建议时，用户讨论了内存不可升级以及在 Mac 上安装 Linux 的担忧，包括 [Debian on Apple M1](https://wiki.debian.org/InstallingDebianOn/Apple/M1) 和 [Debian ARM port](https://www.debian.org/ports/arm/) 中强调的潜在问题。
- **Apple 的 Unified Memory Architecture 成为焦点**：用户们辩论了 Apple M 系列芯片 Unified Memory 的可升级性和性能方面的问题 (`@heyitsyorkie`, `@nink1`, `@wyrath`)。这种缺乏用户可升级内存的架构，与未来潜在的 AMD APU 和 CAMM 内存模块进行了对比。
- **在集成 GPU 上运行 LM Studio 的潜在挑战**：`@ayyouboss` 在一台拥有 16GB RAM 的 Ryzen 平台上使用集成 VEGA GPU 运行 LLM 时遇到问题，尽管 LM Studio 是在 CPU 上运行的。`@rewire` 建议可能是 VRAM 限制在起作用，在经过反复的故障排除后，他们建议尝试使用 Windows 而不是 Linux，因为可能存在驱动程序问题。
- **评估 Apple Silicon Mac 在 Linux 和 AI 方面的用途**：`@ethanboyle` 和其他人讨论了使用搭载 Apple Silicon 的 Mac 进行 AI 工作的可行性，同时考虑到了安装 Linux 的挑战和不可升级的 Unified Memory。社区分享了一些知识和外部链接，例如 [Tart virtualization for Apple Silicon](https://tart.run)，`@wolfspyre` 称其为在 Mac 容器中运行 Linux 的强大且免费的工具。
- **昂贵的 Groq 芯片**：在硬件性能和成本对比中，`@nink1` 和 `@wyrath` 讨论了 Groq 芯片架构如何需要将大量芯片集群化以实现高性能，导致其与 Nvidia 解决方案相比存在巨大的成本差距。投资一个运行大型模型的 Groq 集群可能需要耗资数百万美元。

**提到的链接**：

- [Debian -- ARM Ports ](https://www.debian.org/ports/arm/)：未找到描述
- [Mac computers with Apple silicon - Apple Support](https://support.apple.com/en-us/116943)：从 2020 年底推出的某些机型开始，Apple 开始在 Mac 电脑中从 Intel 处理器过渡到 Apple Silicon。
- [Apple Macbook Pro M1 Max 16&quot; 2021 10-Core CPU 32 GPU 1TB SSD 32GB Ram Gray 194252546833 | eBay](https://www.ebay.com/itm/225868883217)：未找到描述
- [InstallingDebianOn/Apple/M1 - Debian Wiki](https://wiki.debian.org/InstallingDebianOn/Apple/M1)：未找到描述

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1213428496811163678) (2 messages): 

- **对 StarCoder2-15b 的期待**：用户 `@.bambalejo` 询问何时可以在 LM Studio 中尝试 StarCoder2-15b，并引用了一个为 llama.cpp 添加该模型支持的 GitHub Pull Request：[https://github.com/ggerganov/llama.cpp/pull/5795](https://github.com/ggerganov/llama.cpp/pull/5795)。
- **LM Studio 等待更新以集成 StarCoder2**：`@heyitsyorkie` 回复称，一旦 LM Studio 更新到支持该模型的 llama.cpp 版本，StarCoder2-15b 可能会在下一个 Beta 版本中集成到 LM Studio 中。

**提到的链接**：

- [Request: add support for Cerebras GPT models just released · ggerganov/llama.cpp · Discussion #579](https://github.com/ggerganov/llama.cpp/pull/579)：公告在此：https://www.cerebras.net/press-release/cerebras-systems-releases-seven-new-gpt-models-trained-on-cs-2-wafer-scale-systems 模型在此处可用：https://huggingfac...
- [Add support for StarCoder2 by pacman100 · Pull Request #5795 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795)：此 PR 的作用是什么？添加了对最近发布的 StarCoder 2 模型的支持。

  

---

### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1214041526532050984) (4 messages): 

- **Autogen 集成问题**：用户 `@sourguava` 在测试模型时遇到了连接错误，特别是提示“API 密钥不正确”的 **401 错误**。他们提到了在 API 密钥方面的困扰，并提供了在 [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) 查找正确密钥的链接。
- **重新安装 Autogen 可能会有帮助**：针对 `@sourguava` 的问题，`@thebest6337` 建议**重新安装 autogen**，作为解决连接问题的可能方案。
- **LM Studio 可能存在延迟**：`@sourguava` 提到 LM Studio 一直存在模型加载非常缓慢的问题，暗示该平台可能存在性能问题。
- **Docker 卷挂载错误**：`@remliv` 尝试遵循 AutoGen 的 Docker 安装指南，但在使用 `docker run` 命令时遇到了提示本地卷名包含**无效字符**的错误。
- **Docker 在 Windows 上的路径挑战**：`@remliv` 在 StackOverflow 上找到了卷挂载错误的可能解决方案，建议在 Windows 系统上将 `$(pwd)` 替换为 `%cd%`，但这导致了另一个错误，提示文件未找到而无法打开。

**提到的链接**：

- [Docker | AutoGen](https://microsoft.github.io/autogen/docs/installation/Docker/#option-1-install-and-run-autogen-in-docker)：Docker 是现代软件开发中不可或缺的工具，为 AutoGen 的设置提供了极具吸引力的解决方案。Docker 允许你创建一致、便携且隔离的环境...
- [在 Windows 10 的 Docker 中将当前目录挂载为卷](https://stackoverflow.com/questions/41485217/mount-current-directory-as-a-volume-in-docker-on-windows-10/41489151#41489151)：描述：我正在 Windows 10 上通过 Hyper-V 使用 Docker 1.12.5 版本，并希望在当前路径下将容器可执行文件作为命令使用。我构建了一个运行良好的 Docker 镜像，但是...

  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 

triffed.: <@1211375065191682131> 它确实存在，我在 Arch 上，刚用 yay 获取了它。
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/) (1 messages): 

.tntflo: 我们也能在 Linux 上使用这个吗？
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1213737128849186848) (5 messages): 

- **JavaScript 兼容性查询**：`noneofya_business` 询问 **crew ai** 是否支持 JavaScript，但未提供更多细节或收到回复。
- **提到了 Visual Studio Code**：`tobitege` 提到了 *Visual Studio Code (VSC)*，推测是针对之前某个查询的回复，但上下文不明确。`wolfspyre` 呼应了对 **Visual Studio Code** 的提及，并强调需要进一步明确。
- **寻求对困惑话题的澄清**：`wolfspyre` 评论说，理解这些话题可能相当令人困惑，强调需要进一步的解释或指导。
- **探索构建个性化 AI Agent**：`ccarroz` 询问了在预设示例之外构建自定义 AI Agent（定义角色和任务）的经验，以及在各种本地设备上利用不同 LLM 的可行性。他们分享了一个宏伟的计划，即在包括 **3090 GPU**、**Jetson Orion** 和 **6800XT GPU** 在内的不同硬件上运行多种 LLM。
  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1213408961244561418) (121 条消息🔥🔥): 

- **本地模型训练的内存困扰**：`@chunkchampion` 开玩笑说，考虑到本地模型训练消耗了 90 GB 内存，是否还值得推荐。
- **Gradio 版本问题困扰 Space 部署者**：包括 `@ilovesass`、`@cubietom` 和 `@vipitis` 在内的几位用户讨论了部署 Space 时遇到的问题，建议检查过时的 Gradio 版本，并寻找如 `ImageEditor` 等更新后的组件。
- **寻求提升 AI 学习曲线的指导**：`@gschwepp_84093` 询问如何从入门级 AI 项目过渡到更复杂的项目。用户 `@dailafing` 也表达了对同一主题获取建议的渴望，希望有经验的成员能提供见解。
- **针对特定用例寻找模型**：`@pazanchick` 和 `@apaz` 等用户分别就适用于 TTS 和生成读书会问题等任务的模型寻求建议。
- **在 AI 社区分享和寻求机会**：`@dsquared70` 宣传了一个在北卡罗来纳州阿什维尔举行的会议，面向在生产环境中使用 GenAI 的开发者；而 `@jan_skaryna` 正在寻找高级 AI/ML 开发者。

**提到的链接**：

- [AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp)：未找到描述
- [Creausdemo - a Hugging Face Space by niggathug](https://huggingface.co/spaces/niggathug/creausdemo)：未找到描述
- [LGM - a Hugging Face Space by ashawkey](https://huggingface.co/spaces/ashawkey/LGM)：未找到描述
- [GitHub struggles to keep up with automated malicious forks](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/)：GitHub 难以应对自动化的恶意 Fork。被克隆后遭破坏，恶意仓库的 Fork 速度超过了删除速度。
- [Fbi Fbiopenup GIF - Fbi Fbiopenup Carlwhitman - Discover &amp; Share GIFs](https://tenor.com/view/fbi-fbiopenup-carlwhitman-gif-19586039)：点击查看 GIF
- [Marching cubes - Wikipedia](https://en.wikipedia.org/wiki/Marching_cubes)：未找到描述
- [Gradio ImageEditor Docs](https://www.gradio.app/docs/imageeditor)：未找到描述
- [Find the best open source model for your project with Sambaverse](https://sambaverse.sambanova.net/)：未找到描述
- [SambaLingo Chat Space - a Hugging Face Space by sambanovasystems](https://huggingface.co/spaces/sambanovasystems/SambaLingo-chat-space)：未找到描述
- [SambaLingo  - a sambanovasystems Collection](https://huggingface.co/collections/sambanovasystems/sambalingo-65e25770f2037c85ad35ca77)：未找到描述
- [Open-source LLM Ecosystem at Hugging Face](https://youtu.be/e9gNEAlsOvU)：如何寻找、压缩、适配和部署开源大语言模型？这里有一个关于 @huggingface 🤗 所有工具的 10 分钟演示，重点介绍了 transforme...

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1213471352464351263) (5 条消息): 

- **发现 Helix 这一“新手友好型”编辑器**：`@ai_noob` 分享了初次使用 **Helix 编辑器** 的体验，并指出可以通过 **`helix --tutor`** 命令获取详尽的教程。
- **关注 CUDA MODE YouTube 系列**：`@iakhil` 正利用周末时间探索 **CUDA MODE YouTube** 系列视频，以进行深入理解。
- **使用 SASS 为 Gradio 设置样式**：`@targetdummy5623` 正在进行一个项目，通过使用 **SASS** 而非 Python 来实现样式，从而替换 **Gradio** 中的默认主题。
- **HuggingMod 维持讨论节奏**：HuggingMod 提醒一位用户（`<@500991911650394143>`）降低发帖频率，以保持讨论质量。🤗
- **学习 PPO 理论**：`@0enzi` 提到正在深入研究 **Proximal Policy Optimization (PPO)** 背后的理论，暗示其正在加深对强化学习的理解。
  

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1213555785213149204) (7 条消息): 

- **探索 Bigcode 的 In-The-Stack**：用户 `tonic_1` 分享了 [In-The-Stack 的链接](https://huggingface.co/spaces/bigcode/in-the-stack)，这是 Bigcode 在 HuggingFace 上创建的一个 Space，并询问了其他人的使用体验。
- **大脑，而非计算机**：`markplusai` 发布了一篇来自《卫报》的文章，讨论了人类大脑的复杂性，并链接了关于操纵小鼠记忆的研究，强调我们正处于理解大脑这一重大科学旅程的关键时期。这是那篇[发人深省的文章](https://www.theguardian.com/science/2020/feb/27/why-your-brain-is-not-a-computer-neuroscience-neural-networks-consciousness)。
- **拥有超级赛亚人力量的 LLaMA**：`pacozaa` 发现了一篇内容丰富的 Medium 文章，介绍了如何对 LLaMA2 使用 few-shot prompts，并在 Claude 的协助下提升其性能。文章还讨论了使用大语言模型 (LLMs) 辅助创建 macOS agents 的内容，详情请参阅[此处](https://medium.com/@sarinsuriyakoon/creating-macos-agent-part-2-applied-a-few-shot-prompts-to-llama2-33eea86ac366)。
- **使用 Pika 进行唇形同步**：`jacob_f97` 分享了一个名为 "Introducing Lip Sync on Pika" 的 YouTube 视频，展示了一项新功能，允许在该平台的视频中实现嘴唇动作与语音的同步。点击[此处](https://youtube.com/watch?v=oVEOwMkm0SM&feature=shared)观看该功能演示。
- **阿里云的 AI 创新**：`littlehorse` 发布了关于阿里云推出通义千问 2.0 及一系列行业特定模型的消息，以满足日益增长的生成式 AI 需求。更多信息请阅读 [阿里云博客](https://www.alibabacloud.com/blog/alibaba-cloud-launches-tongyi-qianwen-2-0-and-industry-specific-models-to-support-customers-reap-benefits-of-generative-ai_600526)。

**提到的链接**：

- [Am I in The Stack? - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/in-the-stack)：未找到描述
- [Klarna AI assistant handles two-thirds of customer service chats in its first month](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)：纽约，纽约州 —— 2024 年 2 月 27 日 —— Klarna 今日宣布了其由 OpenAI 驱动的 AI 助手。目前已在全球上线 1 个月，数据说明了一切。
- [Why your brain is not a computer](https://www.theguardian.com/science/2020/feb/27/why-your-brain-is-not-a-computer-neuroscience-neural-networks-consciousness)：深度阅读：几十年来，这一直是神经科学中的主导隐喻。但这个想法是否一直误导着我们？
- [Creating MacOS-Agent Part 2: Applied A Few Shot Prompts to LLAMA2](https://medium.com/@sarinsuriyakoon/creating-macos-agent-part-2-applied-a-few-shot-prompts-to-llama2-33eea86ac366)：在 Claude 的帮助下，使用 few-shot prompts 来改进并保证 LLaMA2 7B 的表现。
- [Introducing Lip Sync on Pika](https://youtube.com/watch?v=oVEOwMkm0SM&feature=shared)：没有声音很难讲好一个故事。这就是我们推出 Lip Sync 的原因。现在，当你使用 Pika 创建或上传视频时，你可以让你的角色...
- [Large Language Models for Code Generation](https://blog.fabrichq.ai/large-language-models-for-code-generation-f95f93fe7de4)：从头开始编写无错误的代码是一项耗时且容易出错的任务。然而，四十多年来，开发者们一直……
- [Alibaba Cloud Launches Tongyi Qianwen 2.0 and Industry-specific Models to Support Customers Reap Benefits of Generative AI](https://www.alibabacloud.com/blog/alibaba-cloud-launches-tongyi-qianwen-2-0-and-industry-specific-models-to-support-customers-reap-benefits-of-generative-ai_600526)：推出了新的 AI 模型构建平台和一系列创新云产品，以满足客户和开发者激增的需求。

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1213523045113925702) (8 条消息🔥): 

- **介绍四款奇特的机器人**：`@samakakreacher` 在 Poe 上发布了一系列专用机器人：[DeepSeek Coder 33b](https://poe.com/DeepseekCoder33B)、Mistral 0.2 32k、Proetus 0.4 和 Shap-E，每款机器人都具备从代码辅助到 3D 建模的不同能力。一张 [介绍图片](https://smeyersmrovkill--image-retrieval-get-image.modal.run/?prompt_hash=e8b76fa1278f72bcf5404e6e18685228b58823c356d97b1b80a3bcd9c00a4ba7&image_extension=png) 展示了这一新机器人家族的多样化功能。
- **蛋白质异常检测突破**：`@grimsqueaker` 重点介绍了他们在 NAR Genomics and Bioinformatics 上发表的论文 "**Detecting Anomalous Proteins Using Deep Representations**"，该研究结合了蛋白质语言模型（Protein Language Models）和异常检测。可以通过 [高层级的 Twitter 线程](https://twitter.com/danofer/status/1763962202472484991) 和 [完整论文链接](https://doi.org/10.1093/nargab/lqae021) 获取研究内容。
- **音乐中的采样与 AI**：在 'kevin makes the weirdest dataset' 第 17 集中，`@bigdookie` 反思了围绕 AI 和传统音乐采样的版权争论，并在 [YouTube 视频](https://youtu.be/-Gzh7WtLp0I) 中通过 musicgen 延续和 Ableton 阐述了其观点。
- **AI 变革性模型**：`@andysingal` 分享了他们的模型 [lora_gemma](https://huggingface.co/Andyrasika/lora_gemma)，该模型使用 unsloth 的 TRL 库开发，承诺更快的训练速度，并通过 Hugging Face 上的示例和 Notebook 进行了展示。
- **支持 AI 的 Kubernetes 模块**：`@alextreebeard` 创建了一个 Terraform 模块，用于将 Kubernetes 集群转换为 AI 环境，通过 GitOps 引入了 Jupyter 和 Kubeflow，并考虑集成容器化 GPU。该模块可在 [GitHub](https://github.com/treebeardtech/terraform-helm-kubeflow) 上获取。

**提到的链接**：

- [Andyrasika/lora_gemma · Hugging Face](https://huggingface.co/Andyrasika/lora_gemma)：未找到描述
- [BEE-spoke-data/mega-encoder-small-16k-v1 · Hugging Face](https://huggingface.co/BEE-spoke-data/mega-encoder-small-16k-v1)：未找到描述
- [ableton speedrun - acoustic downtempo lo-fi dnb?? - captain&#39;s chair s1 ep. 17](https://youtu.be/-Gzh7WtLp0I)：一如既往，感谢 musicgen Discord 成员的参与，这确实是一次以最奇特方式进行的协作。感谢 @...
- [GitHub - treebeardtech/terraform-helm-kubeflow: Kubeflow Terraform Modules - run Jupyter in Kubernetes 🪐](https://github.com/treebeardtech/terraform-helm-kubeflow)：Kubeflow Terraform 模块 - 在 Kubernetes 中运行 Jupyter 🪐 - treebeardtech/terraform-helm-kubeflow
- [Detecting anomalous proteins using deep representations](https://doi.org/10.1093/nargab/lqae021)：摘要。生物医学的许多进展可归功于对异常蛋白质和基因的识别。这些蛋白质的许多独特特性是通过...发现的。
- [DeepseekCoder33B - Poe](https://poe.com/DeepseekCoder33B.)：Deepseek Coder 33B 是来自 Deepseek AI 的先进代码模型。如有问题或建议，请联系 sam@samuellmeyers.com。README ========== Deepseek Coder 具有极高的代码性能，其他...
- [Mistralv0-2-32k - Poe](https://poe.com/Mistralv0-2-32k.)：Mistral Instruct - v0.2 - 32k，采用 Mistral AI 已知的最先进技术设计，具有超长上下文窗口，该机器人可以与最优秀的通用机器人相媲美。我个人...
- [Proteus0-4 - Poe](https://poe.com/Proteus0-4.)：Proteus V0.4 是 Proteus 模型的第 4 个版本。它基于 OpenDall-E，可以生成高质量图像。尽情享受吧！
- [ShapEAlpha - Poe](https://poe.com/ShapEAlpha.)：为游戏开发制作 3D 模型，使用 OpenAI 的 Shap-E 模型架构和 Modal 进行无服务器/GPU 托管。感谢大家。爱你们！ <3

---

### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1213414831441448980) (67 条消息🔥🔥): 

- **协作共情**：`@tonic_1` 为没能让 `@582573083500478464` 的工作更轻松而道歉，他原本打算为一些幻灯片提交 PR，但发现 `@582573083500478464` 已经将其制作得很完美了。
- **AI 压缩与合并的进展**：`@nrs9044` 发起了一场讨论，探讨压缩技术的改进如何通过更有效地识别重要权重来增强合并算法。他们还推测了 1.58bit 架构的成功对这两个领域当前算法可迁移性的影响。
- **读书会活动日历**：`@chad_in_the_house` 回答了关于参加读书会的问题，建议目前先查看公告/活动板块，并提到计划创建一个 Google Calendar 以发布更新。
- **寻求关于 Diffusion 和 Consistency Models 的解答**：`@riteshrm` 正在寻找理解 Diffusion 和 Consistency Models 背后数学原理的资源。`@chad_in_the_house` 建议查看解释 Diffusion 模型的博客文章，并提到了 Hugging Face 关于该主题的课程，并在此提供了链接 [here](https://github.com/huggingface/diffusion-models-class)。
- **周末 vs. 周五读书会**：`@shafi8433` 发起讨论，提议在周末而不是周五举行读书会，引发了关于排期偏好和时区的反复讨论；`@lunarflu` 建议在欧洲中部时间 (CET) 的周末举行。

**相关链接**:

- [GitHub - huggingface/diffusion-models-class: Materials for the Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class): Hugging Face Diffusion Models 课程材料 - huggingface/diffusion-models-class
- [GitHub - hyperevolnet/Terminator](https://github.com/hyperevolnet/Terminator): 通过在 GitHub 上创建账号来为 hyperevolnet/Terminator 的开发做出贡献。

  

---


### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1213762457278091294) (1 条消息): 

- **DreamBooth 引入 EDM 节奏**：`@sayakpaul` 分享了 SDXL LoRA DreamBooth 脚本现在包含 **EDM 风格训练支持**。该更新还引入了对近期 **Playground model** 的兼容性，增强了该脚本的功能。查看 Pull Request 了解详情：[Support EDM-style training in DreamBooth LoRA SDXL script](https://github.com/huggingface/diffusers/pull/7126)。

**相关链接**:

[Support EDM-style training in DreamBooth LoRA SDXL script by sayakpaul · Pull Request #7126 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7126): 命令示例：CUDA_VISIBLE_DEVICES=1 accelerate launch train_dreambooth_lora_sdxl.py \   --pretrained_model_name_or_path=&quot;playgroundai/playground-v2.5-1024px-aesthetic&quot;  \   --instance_da...

  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1213447653371940905) (21 messages🔥): 

- **Diffusers 中的 Scheduler 混淆**：`_vargol` 遇到一个问题，在更新后 `print(pipe.scheduler.config._class_name)` 显示了错误的 Scheduler 类。为此提交了一个 GitHub issue ([#7183](https://github.com/huggingface/diffusers/issues/7183))，他们建议通过打印 `pipe.scheduler` 和 `pipe.scheduler._class_name` 来获取正确值的临时修复方案。

- **Diffusers 继承中的 Bug 已修复**：在上述问题被指出后，一个新的 pull request ([#7192](https://github.com/huggingface/diffusers/pull/7192)) 已被合并，以修正 Diffusers 中的 'from_config' bug。`@_vargol` 建议了如何使用 pip 直接从 pull request 安装补丁。

- **使用 Diffusers 进行 Inpainting**：`@sayakpaul` 链接了 [Diffusers 的 Inpainting 文档](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)，这促使 `@tony_assi` 询问关于使用图像提示（image prompt）而非文本进行图生图（image-to-image）Inpainting 的方法。

- **IP-Adapter 图像提示指南**：作为回应，`_homoludens` 分享了 [IP-Adapter 指南](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)的链接，该工具允许在 Inpainting 任务中使用图像提示。

- **如何在 Diffusers 中处理 LoRA 权重**：`@crapthings` 询问了如何将 LoRA 权重集成到 Diffusers 中，`@sayakpaul` 对此进行了回答，并指导如何使用 `set_adapters()` 来管理包括 LoRA 在内的多个适配器（adapters）以实现图像效果。

- **处理 HuggingFace Hub 上的 NSFW 内容**：当 `@pseudoterminalx` 指出 HuggingFace Hub 上可能存在 NSFW 模型时，`@lunarflu` 指示最佳流程是将模型标记为 'NFAA'，或者在必要时开启举报（report），并提供了一个链接 ([pony-diffusion-v2 讨论](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7)) 来解决该问题。

**提到的链接**：

- [AstraliteHeart/pony-diffusion-v2 · 请求为模型添加 NFAA (nsfw) 标签](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7)：未找到描述
- [AstraliteHeart (Astralite Heart)](https://huggingface.co/AstraliteHeart)：未找到描述
- [为推理加载 LoRA](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters)：未找到描述
- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#:~:text=from%20diffusers%20import-,AutoPipelineForInpainting,-from%20diffusers.utils)：未找到描述
- [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)：未找到描述
- [加载适配器](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora)：未找到描述
- [如果更改了 scheduler，scheduler.config._class_name 会显示错误的类名 · Issue #7183 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7183)：描述该 bug：print(pipe.scheduler._class_name) 和 print(pipe.scheduler.config._class_name) 在使用 from_config 类方法更改 scheduler 后都会打印错误的类名...
- [由 yiyixuxu 修复 `from_config` 中的 bug · Pull Request #7192 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7192)：修复 #7183，此脚本现在可以按预期运行：from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, LCMScheduler, AutoencoderKL import torch model_id = &quot;stabilityai/stable-diffusio...

---

### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1213541658646421615) (7 messages): 

- **对 fireche 项目的好奇**：用户 `@fireche` 表示虽然无法提供直接帮助，但对另一位成员的工作表现出兴趣，随后 `@dillonkyle` 提到了他们关于将土木工程图纸的地理参考 PDF 转换为 GIS CAD 的构想。
- **请求 xformers 安装协助**：`@sai_nm` 寻求有关 xformers 安装的帮助，但未提供进一步的上下文或细节。
- **#Terminator 网络介绍**：`@alex_cool6` 分享了他们最近在 #Terminator 网络上的工作，该网络集成了多项关键技术，并重新审视了 20 世纪 90 年代的概念（如 Slow-Fast 网络），并附带了题为 "HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction" 的论文，可在 [arXiv.org](https://arxiv.org/pdf/2401.17948.pdf) 获取。
- **探索用于客户入职的小型 VLM**：`@n278jm` 询问了最适合集成到客户入职流程中以进行图像细节提取的小型视觉语言模型 (VLM)，并提到他们已在 Vision Arena 空间进行了实验。
- **寻求 VLM 实验反馈**：继续对话，`@n278jm` 希望在不牺牲有效性的情况下，针对这个快速发展的领域中如何优化小型模型的输入获取外部见解；`@johko990` 对此表示不确定，但承认这值得探索。
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1213423733637906493) (15 messages🔥): 

- **医学百科 AI 需要咨询**：`@dracula14.` 使用 **Llama 2** 和 **ChromaDB** 创建了一个百科 AI，现在正在寻求关于如何从包含 Embeddings 的 **sqlite 文件**中进行查询的建议。
- **Adam 优化器捍卫其地位**：`@nrs9044` 询问 **Adam 优化器** 是否仍被视为 SOTA。`@lavi_39761` 回复肯定了其在常规用途中的功效，并提供了一个[进一步阅读的链接](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy)。
- **模型部署大对决：Flask vs Triton**：`@frosty04212` 询问部署 NLP 模型的最佳方法，`@vipitis` 澄清说 **Flask** 是一个 Web 框架，而 **Triton** 是一个机器学习编译器，暗示它们承担不同的功能。
- **按需定制 LLM**：`@onedumbdude` 对使用 LLM 执行运行脚本和进行 API 调用等任务充满热情。`@vipitis` 提到了一种称为 **Function-calling** 的技术，它可以实现与此类模型的交互。
- **推理时间之争**：`@anna017150` 发现 **mistral-7b-instruct-v02** 在相同输入下的推理时间比 **bloomz-7b1** 更长，并寻求改进建议。

**提到的链接**：

- [dalle-mini](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--Vm)：Weights & Biases，机器学习开发者工具。
- [Evaluation of Distributed Shampoo](https://wandb.ai/dalle-mini/dalle-mini/reports/Evaluation-of-Distributed-Shampoo--VmlldzoxNDIyNTUy)：优化器对比：Distributed Shampoo、Adam 和 Adafactor。由 Boris Dayma 使用 Weights & Biases 制作。

  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1213447653371940905) (21 条消息🔥): 

- **重复的 Scheduler 名称具有误导性**：`@_vargol` 发现了一个 `diffusers` 的 Bug，即在更新 Scheduler 后，名称显示不正确，显示为 **EulerDiscreteScheduler** 而非 **LCMScheduler**。该问题已在 [GitHub #7183](https://github.com/huggingface/diffusers/issues/7183) 中提出，临时解决方案是使用显式的 `print` 语句来确认正确的 Scheduler 类。

- **Scheduler 命名错误的 Bug 修复**：`@sayakpaul` 分享了一个由 yiyixuxu 提交的 [GitHub pull request #7192](https://github.com/huggingface/diffusers/pull/7192)，旨在修复 `diffusers` 中的 Scheduler 类命名 Bug。该 Pull Request 包含了针对该问题的修复代码。

- **Diffusers 图像局部重绘（Inpainting）教程**：`@sayakpaul` 引用了一份关于使用 Hugging Face 🤗 Diffusers 进行图像局部重绘的指南，该指南依赖 Mask 来定义编辑区域。`@tony_assi` 询问了关于图生图（Image-to-Image）局部重绘的问题，`_homoludens` 提供了关于 [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting) 的额外资源作为指导。

- **直接从 Pull Request 安装**：针对 `@luihis` 的提问，`_vargol` 建议了一种直接从 GitHub Pull Request 安装更新的方法，使用命令 `pip install -U git+https://github.com/huggingface/diffusers@refs/pull/7192/head`。这种方法允许在 PyPi 正式发布之前升级到最新版本。

- **关于设置 LoRA 权重的困惑**：`@crapthings` 询问了如何在 Diffusers 中实现特定的 LoRA 权重，`@sayakpaul` 提供了使用 [PEFT 指南](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters)中的 `set_adapters()` 的解决方案，允许组合和管理 Adapter，以利用 LoRA 生成独特的图像效果。

- **处理 Hugging Face 上的 NSFW 生成模型**：`@pseudoterminalx` 指出了 NSFW 生成模型的存在，促使 `@lunarflu` 建议提交 PR 以添加 NFAA 标签并在必要时进行举报。关于该问题的讨论在 [AstraliteHeart 的 v2 讨论帖 #7](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7) 中继续进行。

**提到的链接**：

- [AstraliteHeart (Astralite Heart)](https://huggingface.co/AstraliteHeart)：未找到描述
- [AstraliteHeart/pony-diffusion-v2 · 请求为模型添加 NFAA (nsfw) 标签](https://huggingface.co/AstraliteHeart/pony-diffusion-v2/discussions/7)：未找到描述
- [为推理加载 LoRA](https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference#combine-multiple-adapters)：未找到描述
- [加载 Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters#lora)：未找到描述
- [IP-Adapter](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter?tasks=Inpainting#:~:text=from%20diffusers%20import-,AutoPipelineForInpainting,-from%20diffusers.utils)：未找到描述
- [Inpainting](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)：未找到描述
- [如果更改了 scheduler，scheduler.config._class_name 会显示错误的类名 · Issue #7183 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7183)：描述 Bug：print(pipe.scheduler._class_name) 和 print(pipe.scheduler.config._class_name) 在使用 from_config 类方法更改 scheduler 后都会打印错误的类名...
- [由 yiyixuxu 修复 from_config 中的 Bug · Pull Request #7192 · huggingface/diffusers](https://github.com/huggingface/diffusers/pull/7192)：修复 #7183，此脚本现在可以按预期运行：from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, LCMScheduler, AutoencoderKL import torch model_id = "stabilityai/stable-diffusio...

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1213461939896721459) (238 条消息🔥🔥): 

- **关于模型性能和训练技术的讨论**：成员们分享了对模型训练重要性的见解，`@thejonasbrothers` 强调了 Pony 等模型面临的挑战以及 NLP 理解的局限性。同时，`@pseudoterminalx` 对某些训练方法表示怀疑，并认为对于某些模型来说，大规模 compute scale 并不是核心问题。对话触及了 finetuning 诸如 Stable Diffusion 2.1 等模型的特殊性，探索了 bias-only training 和 low-rank methods 等技术。辩论对比了不同的 finetuning 过程及其对图像连贯性的影响，并引用了相关的学术论文。
  
- **关于 AI 生成音乐和人声质量的讨论**：频道参与者讨论了 Suno 等模型的人声合成质量，`@pseudoterminalx` 抱怨其产生的声音具有金属感且千篇一律。其他人讨论了 Mistral 和 MusicLM 在特定应用中的潜力，同时对初创公司的开源实践表示担忧，并渴望改进音乐生成模型。焦点转向利用设计巧妙、能适应现场演奏（live play）的伴奏轨道 (`@top_walk_town`)，以及对 YouTube 哼唱转 MIDI 等创新的期待 (`@metal63`)。

- **探索 AI 生成艺术及数据集问题**：对话触及了当前模型在处理 AI 生成艺术的透明度和辨识度方面的局限与挑战 (`@pseudoterminalx`, `@chad_in_the_house`, `@metal63`)。`@pseudoterminalx` 分享了历史数据管理的困扰，讲述了 2009 年某大学的一个案例，涉及一个备份周期为 11 天的邮件服务器，该服务器深受陈旧政策和缺乏停机规划之苦。随后展开了关于 Pony diffusion 等模型美学输出的对比讨论，包括涉及不同作品角色的 prompt (`@thejonasbrothers`)。

- **AI 研究与分享的技术和伦理挑战**：聊天强调了理解技术细节（如 tokenization）的重要性 (`@pseudoterminalx`)，以及围绕数据集处理的道德问题 (`@.undeleted`)。大家普遍对 Twitter 作为 AI 对话媒介的局限性感到沮丧。

- **个人动力与 AI 价值的融合**：用户 `@metal63` 表达了他们在 Pony 等 AI 模型中发现的“救命价值”，引发了围绕主观价值、效用以及在 AI 社区推广此类模型的讨论。对话还涵盖了获取和使用 AI 技术对个人福祉的更广泛影响。

**提到的链接**：

- [来自 Suhail (@Suhail) 的推文](https://fxtwitter.com/Suhail/status/1764395365510660157)：如果你有兴趣复现 MagViT2（或超越其实现/训练性能），请联系我。我为你准备了算力。
- [BRIA 2.3 - briaai 的 Hugging Face Space](https://huggingface.co/spaces/briaai/BRIA-2.3)：未找到描述
- [GitHub 难以应对自动化的恶意 Fork](https://www.theregister.com/2024/03/01/github_automated_fork_campaign/)：先克隆再破坏，恶意仓库的 Fork 速度超过了删除速度
- [使用 Latent Transparency 的透明图像层扩散](https://arxiv.org/abs/2402.17113)：我们提出了 LayerDiffusion，一种使大规模预训练的 latent diffusion 模型能够生成透明图像的方法。该方法允许生成单个透明图像或多个……
- [Doubt Press X GIF - Doubt Press X La Noire - 发现并分享 GIF](https://tenor.com/bsYm1.gif)：点击查看 GIF
- [Dick Experts GIF - Silicon Valley - 发现并分享 GIF](https://tenor.com/view/silicon-valley-gif-5518488)：点击查看 GIF
- [musiclm_large_small_context - Google Drive](https://drive.google.com/drive/u/0/folders/1347glwEc-6XWulfU7NGrFrYTvTnjeVJE)：未找到描述
- [GitHub - zhvng/open-musiclm: Google Research 发布的文本转音乐模型 MusicLM 的实现，并进行了一些修改。](https://github.com/zhvng/open-musiclm)：Google Research 发布的文本转音乐模型 MusicLM 的实现，并进行了一些修改。- zhvng/open-musiclm

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1214134815998214254) (11 条消息🔥): 

- **Terminator Network 的降临**：`@alex_cool6` 宣布了他们最近关于 **#Terminator** 网络的工作，该网络结合了 ResNet 和 Self-Attention 等以往技术，以及 20 世纪 90 年代的 slow-fast networks 概念。他们分享了一篇[研究论文](https://arxiv.org/pdf/2401.17948.pdf)，详细介绍了用于全上下文交互的 **HyperZ⋅Z⋅W Operator**。
- **Claude 3 模型热议**：`@vrus0188` 报告称收到了大量关于 **Claude 3 Model** 的提及，并提供了一个 [Reddit 链接](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/)，讨论与该模型性能和 singularity（奇点）相关的 benchmarks。
- **Claude 3 与 GPT-4 的对比**：`@segmentationfault8268` 测试了 **Claude 3** 模型，发现它在不“偷懒”和理解力方面优于 GPT-4，如果这一点持续得到证实，他们可能会取消 ChatGPT Plus 订阅。
- **PyTorch CUDA Kernels 的挑战**：`@twoabove` 评论称 Claude 3 在处理非通用任务方面缺乏改进，特别提到 **PyTorch CUDA kernels** 是该模型仍然表现出懒惰（laziness）的一个领域。
- **Sonnet 是一个 VLM**：`@jh0482` 加入对话并指出 **Sonnet** 被 bedrock 分类为视觉语言模型（VLM），这引发了人们对其与 **GPT4v** 和 **CogVLM** 相比表现如何的好奇。

**提及的链接**：

[Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1b6dn1m/claude_3_benchmarks/)：未找到描述

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1214024145361436732) (1 条消息): 

- **DPO 优化协作邀请**：用户 `@huunguyen` 正在考虑对 **DPO** (Dynamic Programming Optimizer) 进行微小改进，并寻求帮助。他们已请感兴趣的人员通过私信联系。
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1213620016172371978) (21 条消息🔥): 

- **寻找直播聊天记录**：用户 `@le_tech` 询问前一天直播聊天讨论的位置。`@marksaroufim` 回复了操作说明：导航到 "**reading group stage**" 并点击 Discord 应用右上角的聊天按钮。
- **快速验证**：用户 `@umerha` 询问使用 Gmail 在 lightning.ai 上进行验证需要多长时间，结果发现验证过程迅速完成。
- **弗罗茨瓦夫的串烧惊喜**：在一次轻松的交流中，`@andreaskoepf` 分享了弗罗茨瓦夫著名的烧烤酒吧 "CUDA NA KIJU" 的详情和[链接](https://visitwroclaw.eu/de/ort/cuda-na-kiju-bar-grill)，并澄清这与用户 `@umerha` 提到的明斯特（Münster）无关。
- **GenAI 集成见解征集**：`@dsquared70` 宣布在北卡罗来纳州阿什维尔将举行一场专注于 **生产环境中的 GenAI** 的会议，邀请开发者通过其[网站](https://www.aiinproduction.com/cfp)提交论文和演讲稿。
- **需要重新录制**：`@marksaroufim` 表示打算将之前会议的录音上传到频道，但后来发现录音已损坏。该用户计划当晚重新录制，正如 `_t_vi_` 所建议的，这表明普遍需要备份录音。

**提及的链接**：

- [AI in Production - AI 策略与战术](https://www.aiinproduction.com/cfp)：未找到描述
- [Nvidia 禁止使用 CUDA 软件的转换层 —— 此前该禁令仅列在在线 EULA 中，现在已包含在安装文件中 [已更新]](https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers)：转换层（Translators）成为众矢之的。
- [Cuda na Kiju Bar & Grill](https://visitwroclaw.eu/de/ort/cuda-na-kiju-bar-grill)：我们希望为您带来世界各地不同风味的串烧（Schaschliks）。以一种美妙的形式呈现。

  

---

### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1213453273114349598) (11 条消息🔥): 

- **关于 Python `pass` 关键字的讨论**：`@iron_bound` 询问了在 Python 函数中使用 `pass` 的情况，并链接到了 [unsloth 仓库](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53)。`@apaz` 指出 `pass` 是一个空操作（no-op），通常是习惯使用大括号语言的人为了可读性而添加的，而 `@andreaskoepf` 建议它可以作为“块结束”的标记。
  
- **Python 'pass' 的字节码确认**：针对 `@iron_bound` 对使用 `pass` 进行基准测试的兴趣，`@apaz` 建议使用 `import dis; dis.dis(fn)` 检查字节码是否存在差异。
 
- **Triton 与 CUDA 性能查询**：`@piotr.mazurek` 询问了使用 Triton 编写的 Kernel 与 CUDA 编写的 Kernel 之间的性能差异，以及编译后的 PTX 输出是否有区别。`@andreaskoepf` 澄清了编译过程，并将 Triton 比作 NVCC，并提供了 [GitHub 链接](https://github.com/openai/triton/blob/bfb8e413b075583228c961bdbb65a98dc54d0868/third_party/nvidia/backend/compiler.py#L236) 作为支持。

- **Triton 社区见面会视频分享**：`@andreaskoepf` 分享了一个名为 "Triton Feb community meetup 20240220" 的 [YouTube 视频](https://youtu.be/JDQCdj18Snc?si=yQ10-vOm3ziCe9AO)，展示了 Triton 二月份的社区见面会。

**提到的链接**：

- [triton/third_party/nvidia/backend/compiler.py at bfb8e413b075583228c961bdbb65a98dc54d0868 · openai/triton](https://github.com/openai/triton/blob/bfb8e413b075583228c961bdbb65a98dc54d0868/third_party/nvidia/backend/compiler.py#L236)：Triton 语言和编译器的开发仓库 - openai/triton
- [Triton Feb community meetup 20240220](https://youtu.be/JDQCdj18Snc?si=yQ10-vOm3ziCe9AO)：Triton 二月社区见面会
- [unsloth/unsloth/kernels/rms_layernorm.py at dbba69b085b9d6049b57b48b882af7e9f29df5b2 · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/dbba69b085b9d6049b57b48b882af7e9f29df5b2/unsloth/kernels/rms_layernorm.py#L53)：提速 5 倍、显存占用减少 60% 的 QLoRA 微调。欢迎在 GitHub 上为 unslothai/unsloth 的开发做出贡献。

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1213429661434912780) (113 条消息🔥🔥): 

- **探索将 VRAM 作为交换空间（Swap Space）**：`@nat.42` 发现了将 **Linux VRAM 用作交换文件** 的资源，并提供了 [GitHub 上的 vramfs](https://github.com/Overv/vramfs) 和 [ArchLinux 文档](https://wiki.archlinux.org/title/Swap_on_video_RAM) 的链接。他们认为 VRAM 可能比磁盘分页更快，但也意识到 **对 VRAM 的需求可能会使其作为交换空间的使用变得复杂**。

- **GPU 加速数据库**：`@iron_bound` 开玩笑地提议 **在 GPU 上运行数据库**，这引发了关于现有用于 GPU DataFrame 操作的 [cuDF 库](https://github.com/rapidsai/cudf) 的讨论。`@vim410` 确认了 **GPU 加速数据库** 的巨大潜力，并提到了一些实现这一目标的努力，包括 `@jeremyhoward` 强调的一篇过去的 [ZDNet 文章](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/)。

- **CUDA 编程挑战与解决方案**：包括 `@zippika` 和 `@morousg` 在内的成员讨论了 CUDA 编程的复杂性、**缓存利用率** 以及不同 GPU（如 **NVIDIA A100** 和 **4090** 型号）的 **性能**。`@vim410` 建议研究 **CUTLASS 栈中的 CUTE** 以解决编程复杂性，并提出可以将反馈直接转达给 CUTLASS 开发人员。

- **Hopper's 在异步操作方面的特性**：`@zippika` 研究了 **Hopper 架构的异步矩阵乘法** 及其对性能的影响，指出虽然 Hopper 拥有异步 matmuls，但 4090 仅支持异步加载（loads）和存储（stores），这可能会影响操作的优化方式。

- **关于 Mistral 运营规模的辩论**：围绕 **Mistral 的计算资源** 展开了讨论，`@andreaskoepf` 等人谈到了据报道的 1.5k 张 H100 GPU，并对其与行业巨头相比在训练大规模模型方面的充足性表示怀疑。讨论中包含了社交媒体帖子的链接以及学术论文的引用，以提供关于 Mistral 策略和能力的背景信息。

**提到的链接**：

- [Legion Overview](https://legion.stanford.edu/overview/index.html): Legion 并行编程系统的首页
- [Mistral 7B](https://arxiv.org/abs/2310.06825): 我们推出了 Mistral 7B v0.1，这是一个拥有 70 亿参数的语言模型，旨在提供卓越的性能和效率。Mistral 7B 在所有评估基准测试中均优于 Llama 2 13B 和 Llama 1 3...
- [NVIDIA cuBLASDx &mdash; cuBLASDx 0.1.0 documentation](https://docs.nvidia.com/cuda/cublasdx/index.html): 未找到描述
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762818733016322168): 针对我们最新公告的一些“创意性解读”，澄清几点：- 我们仍然致力于引领开源权重模型！请保持一点耐心，1.5k H100s ...
- [RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`](https://discuss.pytorch.org/t/runtimeerror-cuda-error-cublas-status-not-initialized-when-calling-cublascreate-handle/170409): Error:   Variable._execution_engine.run_backward(  # 调用 C++ 引擎运行 backward pass 时报错 RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`  I h...
- [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl): 未找到描述
- [GitHub - rapidsai/cudf: cuDF - GPU DataFrame Library](https://github.com/rapidsai/cudf): cuDF - GPU DataFrame 库。通过在 GitHub 上创建账号来为 rapidsai/cudf 的开发做出贡献。
- [GitHub - Bruce-Lee-LY/cuda_hgemm: Several optimization methods of half-precision general matrix multiplication (HGEMM) using tensor core with WMMA API and MMA PTX instruction.](https://github.com/Bruce-Lee-LY/cuda_hgemm): 使用 Tensor Core 配合 WMMA API 和 MMA PTX 指令的几种半精度通用矩阵乘法 (HGEMM) 优化方法。 - GitHub - Bruce-Lee-LY/cuda_hgemm: Several optimizati...
- [Buy NVIDIA DGX Station A100 - AI Workstation | Microway](https://www.microway.com/preconfiguredsystems/nvidia-dgx-station-a1): NVIDIA 的 DGX Station A100 在您的办公桌前即可提供“开箱即用”的 AI Datacenter 能力。为您的训练、推理、HPC 或数据科学工作负载实现更快的迭代和创新。
- [GPU databases are coming of age](https://www.zdnet.com/article/gpu-databases-are-coming-of-age): GPU 正在驱动新一代数据库。它们有何特别之处，能否大放异彩？
- [GPU databases are coming of age](https://www.zdnet.com/article/gpu-databases-are-coming-of-age/): GPU 正在驱动新一代数据库。它们有何特别之处，能否大放异彩？
- [Overv - Overview](https://github.com/Overv): 一位对从晶体管到 Vulkan、Kubernetes 再到前端开发的整个技术栈都充满好奇的软件开发者。 - Overv
- [GitHub - Overv/vramfs: VRAM based file system for Linux](https://github.com/Overv/vramfs): 基于 VRAM 的 Linux 文件系统。通过在 GitHub 上创建账号来为 Overv/vramfs 的开发做出贡献。
- [Swap on video RAM - ArchWiki](https://wiki.archlinux.org/title/Swap_on_video_RAM): 未找到描述
- [Buy NVIDIA DGX Station A100 - AI Workstation | Microway](https://www.microway.com/preconfiguredsystems/nvidia-dgx-station-a100/): NVIDIA 的 DGX Station A100 在您的办公桌前即可提供“开箱即用”的 AI Datacenter 能力。为您的训练、推理、HPC 或数据科学工作负载实现更快的迭代和创新。

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1213945343163633695) (3 条消息): 

- **PyTorch 开发播客新集提醒**: `@andreaskoepf` 分享了 PyTorch Developer Podcast 的新集链接，讨论了 [AoTInductor](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor)。
- **直方图 CUDA Kernel 故障排除**: `@srns27` 正在寻求关于他们编写的 CUDA Kernel 的帮助；其预期功能是创建并行直方图，但他们在 `gpuAtomicAdd` 上遇到了结果不一致的问题。他们质疑为什么 `atomicAdd` 在其 Kernel 代码中无法正常工作。
- **播客热忱分享**: `@ericauld` 表达了对新 PyTorch Developer Podcast 剧集的喜爱，并赞赏其简洁的形式。

**提到的链接**:

[未找到标题](https://pytorch-dev-podcast.simplecast.com/episodes/aotinductor): 未找到描述

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1213565082383155220) (1 条消息): 

- **收听 CUDA 陷阱讲座**: `@andreaskoepf` 提醒 `@everyone`，**CUDA-MODE 第 8 课：CUDA performance gotchas** 即将开始，承诺提供关于最大化 Occupancy、合并内存访问（coalescing memory accesses）以及最小化控制分歧（control divergence）的技巧，并包含现场演示。讲座定于 <t:1709409600:t>。
  

---

### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1213771961264644146) (5 条消息): 

- **Asianometry 讨论 SRAM 缩减问题**：用户 `@iron_bound` 分享了来自 Asianometry 的一段名为 [“Can SRAM Keep Shrinking?”](https://www.youtube.com/watch?v=2G4_RZo41Zw) 的 YouTube 视频，以及视频描述中的几个相关链接，包括时事通讯和 Patreon。
- **赞赏 Asianometry 的深度内容**：`@apaz` 称赞了 Asianometry 频道，并在**关注该内容约一年后**进行了推荐。
- **分享 CUDA 编程资源**：`@ttuurrkkii.` 发布了一个 [GitHub 仓库链接](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main)，作为 **CUDA 并行编程和 GPU** 初学者的有用资源。
- **构建 GPT 的视频教程**：`@iron_bound` 的另一个贡献是一个 [YouTube 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY)，解释了如何构建 GPT 模型，遵循了 OpenAI 研究中的重要论文和技术。

**提到的链接**：

- [Let&#39;s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)：我们从零开始，通过代码构建了一个 Generatively Pretrained Transformer (GPT)，遵循论文 "Attention is All You Need" 以及 OpenAI 的 GPT-2 / GPT-3。我们讨论了连接到...
- [Can SRAM Keep Shrinking?](https://www.youtube.com/watch?v=2G4_RZo41Zw)：链接：- Asianometry 时事通讯：https://www.asianometry.com - Patreon：https://www.patreon.com/Asianometry - Threads：https://www.threads.net/@asianometry-...
- [GitHub - CisMine/Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C/tree/main)：通过在 GitHub 上创建一个账户，为 CisMine/Parallel-Computing-Cuda-C 的开发做出贡献。

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1213585987037302794) (4 条消息): 

- **加入 Lamini AI 民主化生成式 AI 的使命**：`@muhtasham` 分享了 Lamini AI 的一个机会，该公司正在寻找 HPC 工程师在 AMD GPU 上优化 LLM，并提到公司对多样性和平等就业的承诺。了解更多关于该职位的信息，涉及使用 MPI、ROCe、UCX 和 OpenAI Triton，请访问 [Lamini AI Careers](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced) 的职位发布。

- **Quadrature 招聘 GPU 优化工程师**：`@d2y.dx2` 强调了 Quadrature 在伦敦或纽约的一个职位空缺，寻找专门从事 GPU 上 AI 工作负载优化的工程师。在 [Quadrature Careers](https://quadrature.ai/) 探索该职位的详情，你可以成为这家研究驱动型公司的一员，并对全球金融市场产生影响。

**提到的链接**：

- [no title found](https://news.ycombinator.com/threads?id=quadrature_ai)：未找到描述
- [Lamini AI - High Performance Computing (Triton + MPI) Engineer](https://jobs.lever.co/laminiai/af688bf8-6c6e-42b5-87aa-0ee9afccdced)：我们团队中的 HPC 工程师负责以下一项或多项工作：开发和优化用于在 AMD GPU 上运行 LLM 的高性能集合通信和 kernel 库，使用的技术包括 MPI...
- [Quadrature](https://quadrature.ai/)：我们正在构建终极自动化交易业务...
- [Quadrature](https://quadrature.ai)：我们正在构建终极自动化交易业务...

  

---

### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1213796429676548166) (11 messages🔥): 

- **Google Colab 中的 CUDA 问题**：用户 `@ttuurrkkii.` 表达了尽管参考了教程，但在 Google Colab 中运行 CUDA 仍遇到困难。`@andreaskoepf` 回复询问是否选择了 Nvidia GPU (A100 或 V100)，并建议使用 `!nvidia-smi` 命令进行检查。
  
- **Lightning AI 来救场？**：在帮助 `@ttuurrkkii.` 时，`@andreaskoepf` 建议尝试使用 [Lightning AI studios](https://lightning.ai/studios)，将其作为解决 Google Colab 上 CUDA 问题的潜在方案。

- **在 Kaggle 上设置 CUDA**：用户 `._bob_` 提到需要在 Kaggle 上设置 CUDA 以便在多 GPU (multi-GPU) 环境下工作。发布的后续消息中没有更多细节或回复。

- **学习 CUDA 和 Triton 是否需要 C 或 C++？**：`@pyro99x` 询问在学习 Triton 和 CUDA 时是否有必要掌握 C 或 C++ 等低级语言。`@briggers` 澄清说，虽然 CUDA 需要此类知识，但 Triton 并不强制要求，尽管了解底层概念会有所帮助。

- **使用 Triton 实现性能最大化**：在讨论的后续中，`@briggers` 建议如果有人已经掌握了 Torch/System/nsys 级别的性能优化，那么 Triton 将是提升性能的值得尝试的下一步。

- **在 CUDA-Mode 中从 Python 调用 C**：针对 `@pyro99x` 关于以 Python 友好方式使用 Triton 和 CUDA 的疑问，`@jeremyhoward` 提到他的 CUDA-Mode 视频演示了如何从 Python 自动生成大部分 C 代码。

- **如何安装 Cutlass 包**：`@umerha` 询问如何安装和包含 CUTLASS C++ 软件包，寻找类似于 `pip install` 的方法。`@andreaskoepf` 确认用户需要克隆 CUTLASS 的 GitHub 仓库，并将 include 目录包含在项目的路径中，因为 CUTLASS 是一个 header-only 的模板库。

**提到的链接**：

- [Lightning Studios - 社区构建的可复现 AI 环境](https://lightning.ai/studios)：用于训练和部署模型、启动端点等的可复现环境。可复制到你的云端，并在你的数据上运行。
- [GitHub - NVIDIA/cutlass: 用于线性代数子程序的 CUDA 模板](https://github.com/NVIDIA/cutlass?tab=readme-ov-file#building-cutlass)：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1214267551010259064) (5 messages): 

- **第 8 课重录版上线 YouTube**：`@marksaroufim` 在 YouTube 上分享了题为 *CUDA Performance Checklist* 的[讲座](https://www.youtube.com/watch?v=SGhfUhlowB4)，包括[代码示例](https://github.com/cuda-mode/lectures/tree/main/lecture8)和[幻灯片](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit)。
- **感谢重录**：`@andreaskoepf` 和 `@ericauld` 对 `@marksaroufim` 为重录第 8 课所投入的时间和精力表示感谢。
- **重录耗时**：`@marksaroufim` 提到，令人惊讶的是重录这节课仍然花费了 1.5 小时，不过这带来了更清晰的演示效果。
- **社区感谢**：`@iron_bound` 也对 `@marksaroufim` 的专注努力表示感谢，并配以庆祝表情：🎉。

**提到的链接**：

[Lecture 8: CUDA Performance Checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)：代码 https://github.com/cuda-mode/lectures/tree/main/lecture8 幻灯片 https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1213598592481099807) (53 messages🔥): 

_

- **Ring Attention 备受关注**：`@andreaskoepf` 强调了在 YK Discord 上关于 **Ring Attention** 和 **Striped Attention** 的讨论，并引用了 @ykilcher 分享的链接。可以通过[此链接](https://x.com/ykilcher/status/1764005196999295282?s=20)关注讨论，并加入 [Yannic Kilcher Discord server](https://ykilcher.com/discord)。
- **探索 LLM 的 Flash Decoding**：`@andreaskoepf` 表示有兴趣尝试 **Flash Decoding**，这是一种提高大语言模型（LLM）推理效率的方法，并指向 [Together.ai 博客文章](https://www.together.ai/blog/flash-decoding-for-long-context-inference)以获取更多信息。
- **深入研究 Flash-Decoding 和 Ring Attention 的实现**：`@iron_bound` 和 `@andreaskoepf` 深入探讨了 **Flash-Decoding** 的细节，讨论了 `log-sum-exp` 等步骤、代码中的引用，并与 `softmax_lse` 等解决方案进行了比较，这些内容可以在 GitHub 仓库 [ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) 和 [flash-attention](https://github.com/Dao-AILab/flash-attention) 中找到。
- **澄清 Flash-Decoding 细节**：`@apaz`、`@nshepperd` 和 `@andreaskoepf` 的讨论详细阐述了 **Flash Attention** 的工作原理及其在分块注意力（blockwise attention）操作中返回的 LogSumExp (lse) 值，引用了代码并对其实现提供了说明，详见[此处](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/ring_flash_attn/utils.py#L19-L21)。
- **协作开发与即兴会议**：`@andreaskoepf` 表示已准备好进行初步的 **Ring-Llama** 测试，并说明因家庭事务会晚些到达，而 `@ericauld` 和 `@iron_bound` 等用户则协调参与语音聊天进行协作，并分享了他们的进展见解。

**Links mentioned**:

- [torch.cuda.empty_cache &mdash; PyTorch 2.2 文档](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html)：未找到描述
- [来自 Yannic Kilcher 🇸🇨 (@ykilcher) 的推文](https://x.com/ykilcher/status/1764005196999295282?s=20)：关于 Ring attention 和 striped attention 的论文讨论正在进行中！https://ykilcher.com/discord
- [laion_idle_cap/docker/sampling.py at main · andreaskoepf/laion_idle_cap](https://github.com/andreaskoepf/laion_idle_cap/blob/main/docker/sampling.py)：通过在 GitHub 上创建账号，为 andreaskoepf/laion_idle_cap 的开发做出贡献。
- [ring-attention/ring-llama at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/tree/main/ring-llama)：ring-attention 实验。通过在 GitHub 上创建账号，为 cuda-mode/ring-attention 的开发做出贡献。
- [flash_attn_jax/src/flash_attn_jax/flash_sharding.py at bc9a01dd7c642730b0b66182cc497633f16f1a29 · nshepperd/flash_attn_jax](https://github.com/nshepperd/flash_attn_jax/blob/bc9a01dd7c642730b0b66182cc497633f16f1a29/src/flash_attn_jax/flash_sharding.py#L137)：Flash Attention v2 的 JAX 绑定。通过在 GitHub 上创建账号，为 nshepperd/flash_attn_jax 的开发做出贡献。
- [xformers/xformers/ops/fmha/__init__.py at fe0526babcd2114e70d9f4d9b10c729628461170 · facebookresearch/xformers](https://github.com/facebookresearch/xformers/blob/fe0526babcd2114e70d9f4d9b10c729628461170/xformers/ops/fmha/__init__.py#L121)：可定制且优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers
- [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/live/gMOAud7hZg4?si=CC9qfwE53qVUY8Qw&t=120,)：Stanford MLSys 研讨会“基础模型限量系列”第 67 集！演讲者：Tri Dao。摘要：Transformers 在长序列上运行缓慢且耗费内存...
- [FlashDecoding++: Faster Large Language Model Inference on GPUs](https://arxiv.org/abs/2311.01282)：随着 Large Language Model (LLM) 在各个领域变得日益重要。然而，在加速 LLM 推理方面仍存在以下未解决的挑战：(1) 同步的部分 sof...
- [Ring Attention &amp; Friends](https://docs.google.com/presentation/d/1cGSFV3rRqhhkLnBwLFtn1GoA8-zh_LYYNVkUFG2hfO0/edit#slide=id.g2bec6cdaf41_0_93)：Ring Attention 及其相关技术：Gemini 1.5 如何扩展到 1000 万个 Token 的上下文，3 月 2 日 - Yannic 的 Discord
- [Flash-Decoding for long-context inference](https://www.together.ai/blog/flash-decoding-for-long-context-inference)：未找到描述
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4)：你好！感谢你在 PyTorch 中实现 Ring attention 的工作！我刚刚尝试实现一个 ring_flash_attn_qkvpacked_func（对应 Flash Attention 中的 flash_attn_qkvpacked_func...）
- [GitHub - cuda-mode/ring-attention: ring-attention experiments](https://github.com/cuda-mode/ring-attention)：ring-attention 实验。通过在 GitHub 上创建账号，为 cuda-mode/ring-attention 的开发做出贡献。
- [ring-flash-attention/test/test_ring_flash_attn_func.py at 78959746e8ce88394ded9263b417ec4708f3cc45 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/test/test_ring_flash_attn_func.py#L97)：结合 Flash Attention 的 Ring attention 实现 - zhuzilin/ring-flash-attention
- [flash-attention/flash_attn/flash_attn_interface.py at 184b992dcb2a0890adaa19eb9b541c3e4f9d2a08 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/184b992dcb2a0890adaa19eb9b541c3e4f9d2a08/flash_attn/flash_attn_interface.py#L482C4-L482C4)：快速且内存高效的精确 Attention。通过在 GitHub 上创建账号，为 Dao-AILab/flash-attention 的开发做出贡献。
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention)：结合 Flash Attention 的 Ring attention 实现 - zhuzilin/ring-flash-attention
- [ring-flash-attention/ring_flash_attn/utils.py at 78959746e8ce88394ded9263b417ec4708f3cc45 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/blob/78959746e8ce88394ded9263b417ec4708f3cc45/ring_flash_attn/utils.py#L19-L21)：结合 Flash Attention 的 Ring attention 实现 - zhuzilin/ring-flash-attention

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1213531875898036295) (5 messages): 

- **Introducing RAPTOR for Advanced RAG**: LlamaIndex 引入了 **RAPTOR**，这是一种用于 *Retrieval-Augmented Generation* (RAG) 的新型树状结构技术，旨在解决原生 top-k RAG 在检索高层级上下文细节方面的局限性。正如这篇 [Tweet](https://twitter.com/llama_index/status/1763972097628684607) 中提到的，它有望更好地处理文档中特定事实的问题。

- **Showcasing RAG in Real-world Applications**: LlamaIndex 的一场新网络研讨会展示了在实际应用中使用 RAG 的项目，包括一个创新的 **GAI 驱动的 ADU 规划器**，旨在简化增加附属居住单元（accessory dwelling units）的过程，详见其最新的 [Tweet](https://twitter.com/llama_index/status/1764020728427667605)。

- **Build RAG with LlamaIndex + MongoDB**: @AlakeRichmond 使用 @MongoDB Atlas 开发了一个用于数据索引的参考架构，LlamaIndex 强调了其对正确数据准备的高度重视。对于那些想要使用 MongoDB 构建 RAG 系统的人来说，这份指南至关重要，详见分享的 [Twitter 帖子](https://twitter.com/llama_index/status/1764078471276642469)。

- **Semantic Chunking for Enhanced RAG**: Florian June 关于 Semantic Chunking 的文章被 LlamaIndex 推荐为一份全面的指南，通过将语义相似的信息进行分组，有望为 RAG 提供更好的检索和合成效果。在他们的 [Tweet](https://twitter.com/llama_index/status/1764335221141631471) 中了解更多关于此方法的信息。

- **Claude 3 Released with Day 0 Support from LlamaIndex**: @Llama_Index 宣布 Claude 3 发布，包含三个版本，其中包括声称性能超越 GPT-4 的 Claude Opus。LlamaIndex 已准备好集成这一新模型，正如他们在热情的 [公告](https://twitter.com/llama_index/status/1764731195286577247) 中所声明的那样。

**Links mentioned**:

[ADU Planner](https://t.co/teMjG0e9Zh): 通过我们的 GAI 驱动的 ADU 规划器彻底改变 ADU 建造过程，这是一种全新的解决方案，只需点击一下即可提供轻松的设计、本地合规性检查和快速的供应商连接。

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1213465825588023316) (178 条消息🔥🔥): 

- **使用 ollama 尝试本地 ReacAgents**：`@impactframes.` 分享了在使本地 ReacAgents 与 ollama 协同工作时遇到的困难，而 `@whitefang_jr` 建议验证 LLM 是否已使用 ollama 设置进行部署和托管。对话围绕可能的部署问题和配置设置展开，`@cheesyfishes` 强调结构化输出对于开源模型来说可能具有挑战性。
- **ICLR 2024 论文 Prompt 痛点**：`@antelope6345` 需要一种查询 ICLR 2024 论文的方法，并在使用提供的某些代码示例时面临挑战，而 `@cheesyfishes` 建议使用 vector index 和 sub question query engine 或 document summary index 以获得更高效的结果。
- **混合向量和关键词搜索提示**：`@valu_` 询问了如何在一组问题中搜索相似性。`@cheesyfishes` 就设置结合向量和关键词搜索的 hybrid search 提供了建议，并引导至相关资源，包括使用 Qdrant、Weaviate 或自定义 BM25 实现的设置。
- **API 文档结构建议**：用户 `@tusharganguli` 对 API 参考文档的结构提出了担忧。`@cheesyfishes` 承认 API 参考文档一直被忽视，但提到即将进行重大升级。
- **Llama Index Discord 获赞**：`@.tarpus` 对 OpenAI API 最近的变化表示沮丧，这些变化需要更新他们的代码。他们评论说，Llama Index Discord 社区比其他社区更有组织且更有帮助。

**提到的链接**：

- [未找到标题](https://news.ycombinator.com/item?id=37764489)：未找到描述
- [LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/)：未找到描述
- [impactframes/mistral_alpha_xs](https://ollama.com/impactframes/mistral_alpha_xs)：基于泄露的 Mistral Medium 70B。该模型是一个 2bit imatrix 量化版本，可以在来自 Knut 的 https://huggingface.co/KnutJaegersberg/awesome-2bit-gguf HF 仓库集合的消费级硬件上流畅运行...
- [带有查询引擎 (RAG) 工具的 ReAct Agent - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine.html)：未找到描述
- [OpenAI - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/openai.html#openai)：未找到描述
- [Ollama - Llama 2 7B - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html?)：未找到描述
- [RAG CLI - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/rag_cli.html)：未找到描述
- [Observability | LlamaIndex.TS](https://ts.llamaindex.ai/observability/)：LlamaIndex 提供一键式 observability 🔭，允许你在生产环境中构建规范的 LLM 应用程序。
- [Llama API - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/llm/llama_api.html)：未找到描述
- [Observability - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html)：未找到描述
- [在高级模块中访问/自定义 Prompt - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin.html)：未找到描述
- [如何在 5 分钟内改进任何 Prompt（聊天 UI 和代码）](https://towardsdatascience.com/how-to-improve-any-prompt-in-less-than-5-minutes-chat-ui-and-code-8a819e2fa2ba)：将半成品句子转化为专家级 Prompt
- [修复：mean_agg 返回不可序列化的 numpy float64，由 TonyBotongChu 提交 · Pull Request #11458 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11458/files)：描述：mean_agg 函数返回的是 numpy float64 列表，而不是预期的 python float 列表。这在查询基于 http 的向量数据库（如 Chromadb）时会导致错误...
- [Vector Stores - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html#vector-store-options-feature-support)：未找到描述
- [Qdrant Hybrid Search - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_hybrid.html#qdrant-hybrid-search)：未找到描述
- [Weaviate Vector Store - Hybrid Search - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndexDemo-Hybrid.html)：未找到描述
- [Reciprocal Rerank Fusion Retriever - LlamaIndex 🦙 v0.10.15](https://docs.llamaindex.ai/en/stable/examples/retrievers/reciprocal_rerank_fusion.html#reciprocal-rerank-fusion-retriever)：未找到描述

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1214256910199685190) (1 条消息): 

- **LlamaIndex 与 LongContext 的集成**：`@andysingal` 分享了一个链接，讨论通过 **LlamaIndex** 与 **LongContext** 的集成来**赋能长上下文 RAG**。文章重点介绍了谷歌发布的具有 1M 上下文窗口的 Gemini 1.5 Pro 及其潜在的集成方案，详见[此处](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738)。

**提到的链接**：

[Empowering Long Context RAG: The Integration of LlamaIndex with LongContext](https://medium.com/ai-advances/empowering-long-context-rag-the-integration-of-llamaindex-with-longcontext-6cf014d4d738)：Ankush k Singal

  

---



### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1214271206610571345) (1 条消息): 

- **Claude 3.0 今日发布**：`@alexatallah` 宣布 **Claude 3** 正在 OpenRouter 上发布，其中包括一个实验性的自我审查（self-moderated）版本。社区的期待终于随着这次最新更新得到了满足。
  

---


### OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1213640674000511046) (1 条消息): 

- **Leumon 发起的 LLM 挑战**：`@leumon` 建立了一个服务器，用于进行一款有趣且具有教育意义的游戏，尝试诱导 GPT3.5 泄露密钥。该游戏强调了谨慎对待 AI 输出的重要性，并确保在处理机密信息时有额外的安全措施。这个概念最初由 `@h43z` 提出，并由 `@leumon` 通过新的 Prompt 进行了改进。
- **与多种 AI 免费对话**：除了挑战赛，`@leumon` 的服务器还允许用户使用 OpenRouter API 免费与各种 AI 模型聊天，如 **Claude-v1、Gemini Pro、Mixtral、Dolphin** 和 **Yi**。这为探索不同 LLM 的能力和响应提供了独特的机会。

**提到的链接**：

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/YWX8Eft6R8)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1213522011071189052) (96 条消息🔥🔥): 

- **Claude-3 的访问与讨论**：`@justjumper_` 在 Claude 3 发布后不久表达了对其访问权限的渴望。`@louisgv` 确认所有 Claude 3 版本都在添加中，并特别提到“实验性”版本也将上线；而 `@arsoban` 分享称，在他们的测试中，Claude3 Opus 展现出比 GPT-4 更强的文本理解能力。

- **OpenAI 与 Claude 的定价对比**：成员 `@oti5` 和 `@voidlunaa` 讨论了 Anthropic 的 Claude 3 相比 GPT-4 似乎偏高的定价，特别是对从 Claude-3-Sonnet 到 Claude-3-Opus 的价格跳跃感到困惑。

- **Claude 的性能与可用性**：讨论了 Claude 3 各变体的性能，`@arsoban` 在某些测试中建议 Sonnet 的表现优于 Opus，并提议在语音聊天中分享他们的见解。`@alexatallah` 向 `@billbear` 保证 Claude 3 正在路上，“实验性”版本也将提供。

- **测试 Claude 的能力**：用户 `@arsoban` 和 `@you.wish` 计划进行 Claude 3 的英语到代码转换测试，特别是在游戏开发背景下，尽管 `@arsoban` 尚未安装用于实际实施的游戏引擎。

- **模型性能随时间下降**：`@capitaindave` 观察到 Gemini Ultra 的推理能力与其发布时相比可能有所下降，AI 表现出一种比实际内容更强的连贯性假象。

**提到的链接**：

- [OpenRouter](https://openrouter.ai/playground?models=anthropic/claude-instant-1.2)：LLM 和其他 AI 模型的路由服务。
- [codebyars.dev](https://share.codebyars.dev/u/jGY25U.png)：未找到描述。

  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1214131362290864178) (1 条消息): 

- **来自 OpenAI 的重大新闻**：用户 `@jeffreyw128` 兴奋地宣布 **OpenAI 发布了类似于 Gemini/Perplexity 的浏览功能**。这是[包含该公告的推文](https://twitter.com/wangzjeff/status/1764572262743851339)。
  

---

### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1214213745199554571) (71 条消息🔥🔥): 

- **Claude 3 挑战 GPT-4**：`#claude` 频道的爱好者们对新的 **Claude 3** 模型系列充满期待。`@res6969` 声称其表现优于 GPT-4，特别是在**数学和代码任务**方面。
- **讨论性价比**：虽然 `@pantsforbirds` 和 `@emfastic` 等用户在纠结 **Claude 3** 与 GPT-4 相比的成本，但根据 `@res6969` 的说法，价格可能会在未来几个月内更新，尽管存在价格担忧，许多人仍保持浓厚兴趣。
- **合成数据生成**：用户 `@edencoder` 提出 Claude 3 的优势可能在于 **synthetic data generation**（合成数据生成），考虑到该模型提供了显著更好的生产速率限制 (production rate limits)，较高的成本是合理的。
- **对 Haiku 模型的期待**：`@potrock` 和 `@pantsforbirds` 的讨论表达了对尚未发布的 **Haiku 模型**的兴趣，该模型以其极具竞争力的定价和在 **human eval** 中的潜力令人印象深刻。
- **运营效率查询**：`@res6969` 分享了非科学的团队实验，强调了 **Claude 3 的 latency**（延迟）表现，首个 token 响应时间约为 4 秒，完整响应时间以秒计，展示了用户体验到的实际运营效率。

**提到的链接**：

- [介绍下一代 Claude](https://www.anthropic.com/news/claude-3-family)：今天，我们宣布推出 Claude 3 模型系列，它在广泛的认知任务中树立了新的行业标杆。该系列包括三个按能力递增排序的最先进模型...
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1764653833970659560?s=20)：通过此次发布，用户可以根据其用例选择智能、速度和成本的理想组合。Opus，我们最智能的模型，实现了接近人类的理解能力。I...
- [模型与 API 提供商分析 | Artificial Analysis](https://artificialanalysis.ai/)：AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、性能和速度（吞吐量和延迟）等关键指标的独立基准测试。

  

---


### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1213986178064453694) (10 条消息🔥): 

- **寻找高性价比的 Embedding 推理**：用户 `@iyevenko` 询问了运行 embedding 模型最划算的选项，目标是在生产环境中达到每秒约 100 次推理。
- **向量数据库推荐汇总**：`@iyevenko` 还对向量数据库推荐表现出兴趣，`@yikesawjeez` 推荐了 Qdrant（速度快）和 Weaviate（支持混合查询），还为熟悉 PostgreSQL 的用户提到了 pgvector。
- **云端 vs 裸机 (Bare Metal) 的性价比**：`@yikesawjeez` 区分了云基础设施与裸机上的高性价比解决方案，暗示不同的环境可能会影响决策。
- **OpenAI 的 Embedding 模型被认为很便宜**：`@iyevenko` 经过计算后认定 OpenAI 的解决方案相当便宜，并考虑将其用于云基础设施方案。
- **评估 OpenAI 改进后的 Embeddings**：`@iyevenko` 对过去 embedding 的质量表示担忧，但愿意重新评估，尤其是 `@yikesawjeez` 建议最新发布的版本值得一试。
  

---

### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1214198220817694824) (43 messages🔥): 

- **Philpax 深入探讨 RLHF 和 AI 争议**：`@philpax` 分享了一段 [YouTube 视频访谈](https://www.youtube.com/watch?v=olpJrXgHc4M)，嘉宾是来自 Synth Labs 和 Eleuther AI 的 Louis Castricato，讨论了 RLHF、Gemini 争议、DPO 以及 Carper AI。
- **Anthropic 发布新 AI 模型**：`@xeophon.` 转发了 [AnthropicAI 的公告](https://x.com/anthropicai/status/1764653830468428150?s=46)，宣布推出其下一代 AI 模型 **Claude 3** 系列，包括 Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku，声称这些模型在 AI 性能方面树立了新基准。
- **Claude 3 模型规格揭晓**：`@xeophon.` 提到了 Claude 3 的模型规格，包括支持图像输入，发布时即拥有 200K 上下文窗口（最高可扩展至 1M），并号称比 GPT-4 具有更高的效率提升。
- **Claude 3 模型 API 上线**：`@xeophon.` 分享称 AnthropicAI 的 Claude 3 Opus 和 Sonnet 模型现已通过 API 提供，Haiku 也将很快发布；同时指出欧盟地区现在无需 VPN 即可访问基础版 Claude。
- **对 Claude 性能的反应**：`@sid221134224` 和 `@canadagoose1` 等多位用户对 **Claude 3** 表示惊叹，认为其表现优于 GPT-4，并讨论了那些无法访问私有数据集的 AI 模型的潜力。

**提到的链接**：

- [Dimitris Papailiopoulos (@DimitrisPapail) 的推文](https://fxtwitter.com/DimitrisPapail/status/1764659274821595209)：AnthropicAI 的 Claude 3 Sonnet（中端模型）搞定了我的“字母限制”问题：“仅使用以 [某个字母] 开头的单词来描述 [某物]”。太牛了。
- [Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1764653830468428150?s=46)：今天，我们发布了下一代 AI 模型 Claude 3。这三款尖端模型——Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku——在推理等领域树立了新的行业基准……
- [访谈 Synth Labs 和 Eleuther AI 的 Louis Castricato，探讨 RLHF、Gemini 争议、DPO、Carper AI](https://www.youtube.com/watch?v=olpJrXgHc4M)：很高兴为大家带来另一场访谈！这次是深入探讨我的擅长领域——关于 RLHF 的一切。Louis Castricato 可能是其中的隐藏明星……

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1214297685507710976) (6 messages): 

- **Claude 3 引发存疑推文**：`@natolambert` 指出 **Claude 3** 的发布导致一些有问题的推文出现，并将其总结为“伙计们，因为 Claude 3，关于 Q* 的推文又要冒出来了。”
- **对用户回应感到沮丧**：`@natolambert` 对 Claude 3 发布后的讨论质量表示沮丧，形容情况“非常糟糕”。
- **直接应对误导信息**：针对与 Claude 3 相关的误导性推文，`@natolambert` 提到他采取了直接回复“你太蠢了”的方式。
- **对马甲账号的预期**：`xeophon.` 幽默地将 `@natolambert` 的直接回复误解为可能是使用小号（alt）进行的，并建议这是一种讽刺性的互动策略。
- **不用小号，纯粹是懒**：`@natolambert` 澄清了不使用小号的决定，承认自己不想麻烦，说“懒得用小号”且“启动能量太高”。
  

---

### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1213698383462801459) (24 messages🔥): 

- **AI 的电影化解读**：`@natolambert` 表达了对电影 *Her* 的热情，并考虑为一个模仿该电影主题的虚构 OpenAI 项目制作**模拟预告片 (mock trailer)**。
- **寻找视频编辑伙伴**：`@natolambert` 正在寻找具有**视频编辑**技能的人合作预告片项目，可能与之前提到的受 *Her* 启发的想法有关。
- **内容预告与 Hugging Face 热议**：`@natolambert` 暗示本周将有一些**有趣的内容**，并透露 Hugging Face 的 CTO Julien 可能会加入 Discord，成为播客的新付费支持者。
- **参与开源 AI 讨论**：`@xeophon.` 关注了 **@OfficialLoganK** 的一条推文，引发了 `@natolambert` 和 `@mike.lambert` 对 OpenAI 在**开源 AI** 上的立场及其影响的一系列反思。
- **学习与讨论 Julia 语言**：在 `@natolambert` 询问 **JuliaLang** 后，`@sid221134224` 提供了关于 Julia 编程语言的详细概述及资源链接 (https://julialang.org/)。

**提到的链接**：

- [The Julia Programming Language](https://julialang.org/)：未找到描述
- [Logan.GPT (@OfficialLoganK) 的推文](https://x.com/officiallogank/status/1764435268021502226?s=46)：开源 AI 对开发者、企业和人类来说是净收益。

  

---


### Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/) (1 messages): 

natolambert: TBT（往日回顾），那是梗图 (meme) 最多的一天。
  

---


### Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1213679451565727764) (5 messages): 

- **强化学习基础模型**：`@sid221134224` 分享了一篇关于 RL 基础模型新论文的 [Twitter 链接](https://twitter.com/svlevine/status/1764116636142076070)，该模型训练了一个以奖励函数嵌入为条件的策略，使其在测试时能够泛化到新的奖励函数。
- **Nat 的下一个采访目标**：`@natolambert` 表示有兴趣采访 Sergey，可能与讨论的 RL 基础模型有关。
- **Cohere 的 PPO 论文讨论**：`@vj256` 询问是否有额外数据或复现研究支持 Cohere 论文的论点，即由于 LLMs 的稳定性，不需要对 PPO 进行修正。
- **寻求独立验证**：在询问其他团队的复现情况后，`@vj256` 表现出对独立验证 Cohere 论文结论的持续兴趣。
- **关于 LLMs 中 PPO 修正的见解**：`@natolambert` 提到 `<@304671004599255043>` 几个月前就掌握了关于 LLMs 中不需要 PPO 修正的相关知识，这一话题在最近发布的访谈中有所涵盖。
  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1213502882758791238) (56 条消息🔥🔥): 

- **探索 SharePoint 数据摄取**：`@rajib2189` 提到成功从 SharePoint 上的 PDF 文件夹加载数据，并分享了一个演示如何使用 Langchain 从 SharePoint 提取文档内容的 YouTube 视频。更多详情可以在 Langchain 文档中关于 [Microsoft SharePoint 集成](https://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepoint)的部分找到。
  
- **Langchain 实现咨询**：`@tawsif2781` 尝试直接将字典传递给 Langchain 中的 `RunnablePassthrough`，旨在避免 "stuff" 键并为其用例维护特定的字典结构。他们正在寻求修改链以实现此目的的建议。

- **为可扩展的 LLM Web 应用选择合适的技术栈**：针对 `@thebeast3326` 的询问，`@sharrajesh` 建议了一个包含 Python 3.11、FastAPI、Langchain 等的技术栈，而 `@lhc1921` 则推荐了托管在 Vercel 上的 Next.js 结合 Langchain.js。
  
- **关于 Langchain 生产就绪性的讨论**：`@buzzoo123` 和 `@mintier` 讨论了对 Langchain 稳定性和商业用途定制化的担忧，承认其在高层理解和业余项目方面的优势，但选择为生产目的编写自定义代码。

- **关于 Anthropic Claude 3 模型的问题**：`@dclarktandem` 询问了如何通过 Langchain 使用新的 Claude 3 模型。在经过一番困惑后，`@.bagatur` 澄清了应使用的正确包和模型字符串（`"claude-3-opus-20240229"`），并提供了相关的代码片段以及 Langchain 文档中 [Anthropic 集成](https://python.langchain.com/docs/integrations/chat/anthropic)的链接。

**提到的链接**：

- [ChatAnthropic | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/anthropic)：此 Notebook 介绍了如何开始使用 Anthropic 聊天模型。
- [RAG | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/cookbook/retrieval#with-memory-and-returning-source-documents)：让我们看看如何在 Prompt 和 LLM 中添加检索步骤，这增加了...
- [How to extract document content from Sharepoint using Langchain](https://youtu.be/2-vjzsjVmik)：我按照下面的 Langchain 链接进行了此演示 https://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepoint 遵循的步骤_______...
- [Microsoft SharePoint | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/document_loaders/microsoft_sharepoint)：Microsoft SharePoint 是一个...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1213580016873373746) (5 条消息): 

- **寻求 LLM Web 应用技术栈建议**：用户 `@thebeast3326` 询问了构建可扩展 LLM（大语言模型）Web 应用的合适**技术栈**，但频道内未提供任何建议或后续讨论。
- **探索使用 Langserve 创建 .docx 文件**：`@yoangab` 询问 **Langserve** 是否能够返回由 runnable 创建的 `.docx` 文件；然而，关于该功能是否存在或如何实现的细节尚未讨论。
- **Langserve 的缓存难题**：`@kandiesky` 遇到了 **Langserve** 请求不使用其 LLM 缓存的问题，尽管他们遵循了 Langchain 缓存（`set_llm_cache`）文档，并提到 **In Memory Cache** 也不起作用；该线程尚未提供解决方案或回复。
- **垃圾信息警报**：`@teitei40` 发布了一条看似**垃圾链接**的消息，承诺提供 50 美元的 Steam 礼品，并附带一段包含各种随机词汇和链接（[https://u.to/BkNtIA](https://u.to/BkNtIA)）的无意义文本；用户应保持警惕，因为这看起来无关且可能具有恶意。

**提到的链接**：

[21 YEARS TOGETHER Get a $50 gift card!](https://u.to/BkNtIA)：Steam 是玩游戏、讨论和创作游戏的终极目的地。

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1214307152420675624) (2 条消息): 

- **可疑 Steam 礼品链接警报**：用户 `@teitei40` 分享了一个据称提供 **50 美元 Steam 礼品**的链接（[steamcommunity.com/gift/7584903](https://u.to/BkNtIA)）并艾特了所有人（`@everyone`）。鉴于消息和链接的性质，用户应保持警惕。
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1213909611183214622) (9 条消息🔥): 

- **通过 Devscribe AI 与 YouTube 视频聊天**：`@deadmanabir` 与 `@Faisal` 介绍了 **Devscribe AI**，这是一个 GEN AI 项目，可以直接与 YouTube 视频对话，无需观看完整内容即可获取摘要和核心概念。他们强调了预生成摘要、视频管理和上下文视频聊天等功能，并提供了 [视频演示](https://youtu.be/HfhXaXkeeWs) 和 [项目链接](https://dev-scribe-ai-7fj7.vercel.app/)，同时请求在 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7170090830647451648/) 和 [Twitter](https://twitter.com/ItsDutta99/status/1764326912732839940) 上提供反馈和分享。

- **生成式 AI 增强资产负债管理**：`@solo78` 在 Medium 上分享了一篇文章，讨论了生成式 AI 在革新人寿保险业资产负债管理中的作用，详细说明了潜在收益，并附上了 [文章链接](https://medium.com/@bsouleymane78/revolutionizing-asset-liability-management-in-life-insurance-the-role-of-generative-ai-85857c854609)。

- **高效学习的费曼技巧 (Feynman Technique)**：`@shving90` 分享了来自 `@OranAITech` 的 [Twitter 线程](https://x.com/OranAITech/status/1764282509766877465?s=20)，介绍了结合他们最新工作流采用的费曼技巧，旨在帮助用户清晰地表达对概念的理解。

- **推出 Galaxy AI 免费 API 服务**：`@white_d3vil` 宣布启动 **Galaxy AI**，为包括 **GPT-4**、**GPT-4-1106-PREVIEW** 和 **GPT-3.5-turbo-1106** 在内的优质 AI 模型提供免费 API 服务。邀请用户尝试并将其集成到自己的项目中，但未提供链接。

- **发布 Next.js 14+ Starter 模板**：`@anayatk` 发布了一个包含多种现代开发工具的 Next.js 14+ starter 模板，并分享了 GitHub [模板链接](https://github.com/anayatkhan1/Nextjs-template)。

- **关于使用 LangChain 构建实时 RAG 的博客**：`@hkdulay` 分享了一篇详细介绍使用 LangChain 构建实时检索增强生成 (RAG) 的文章，旨在通过引用来源来提高大语言模型 (LLM) 的回答准确性，并提供了 [博客链接](https://hubertdulay.substack.com/p/easy-introduction-to-real-time-rag)。

- **探索 RAG 系列中的高级索引**：`@tailwind8960` 讨论了检索增强生成中索引的复杂性，并分享了关于如何避免回答中出现不准确或幻觉 (hallucinations) 的见解，附带 [对话链接](https://div.beehiiv.com/p/advanced-rag-series-indexing)。 

- **关于 Steam 礼品卡的重复消息**：`@teitei40` 两次发布了关于 50 美元 Steam 礼品的消息，并提供了兑换 [链接](https://u.to/BkNtIA)，但未提供更多背景信息。

**提到的链接**：

- [Revolutionizing Asset & Liability Management in Life Insurance: The Role of Generative AI](https://medium.com/@bsouleymane78/revolutionizing-asset-liability-management-in-life-insurance-the-role-of-generative-ai-85857c854609)：探索 AI 和生成式 AI 为改善人寿保险公司资产负债管理 (ALM) 任务和流程提供的各种机会。
- [Easy Introduction to Real-Time RAG](https://hubertdulay.substack.com/p/easy-introduction-to-real-time-rag)：使用 Apache Pinot 向量索引。
- [Advanced RAG series: Indexing](https://div.beehiiv.com/p/advanced-rag-series-indexing)：如何优化 Embedding 以实现准确检索。
- [Tweet from Adi Oran (@OranAITech)](https://x.com/OranAITech/status/1764282509766877465?s=20)：通过 OranScribe 的最新工作流探索费曼技巧！🚀 拥抱革命性的学习方法：你的费曼式见解是什么？揭示并表达你的核心理解...
- [Galaxy AI - Swagger UI](https://galaxyapi.onrender.com)：未找到描述。
- [Devscribe AI : Your personal video summariser.](https://youtu.be/HfhXaXkeeWs)：DevscribeAI 是一个可以用来与 YouTube 视频聊天的工具。让生活更轻松，无需观看整个视频，只需获取摘要，你还可以提出问题...
- [DevscribeAI](https://dev-scribe-ai-7fj7.vercel.app/)：未找到描述。
- [GitHub - DeadmanAbir/DevScribe-AI: A platform that lets you create pre-generated summaries and key concepts by only giving the YouTube video URL link.](https://github.com/DeadmanAbir/DevScribe-AI)：一个只需提供 YouTube 视频 URL 链接即可创建预生成摘要和核心概念的平台。- DeadmanAbir/DevScribe-AI

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1213481155500970035) (3 条消息): 

- **解码 Tokenizer**：`@lhc1921` 分享了一个名为 "Let's build the GPT Tokenizer" 的 [YouTube 视频](https://www.youtube.com/watch?v=zduSFxRajkE)，深入探讨了 Tokenizer 的创建，这对于在 **Large Language Models (LLMs)** 中进行字符串与 Token 之间的转换至关重要。
- **可疑的 Steam 礼品链接**：用户 `@teitei40` 发布了一个链接，声称提供 **$50 Steam 礼品卡**，但该 URL ([https://u.to/BkNtIA](https://u.to/BkNtIA)) 看起来很可疑，后面跟着看似随机的文本，引发了对其合法性的担忧。

**提到的链接**：

- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)：Tokenizer 是 Large Language Models (LLMs) 中一个必要且普遍存在的组件，它在字符串和 Token（文本块）之间进行转换。Tokenizer...
- [21 YEARS TOGETHER Get a $50 gift card!](https://u.to/BkNtIA)：Steam 是玩游戏、讨论和创作游戏的终极目的地。

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1213411222385332226) (51 条消息🔥): 

- **Google 利用 Stack Overflow 强化 AI**：用户 `@mjng93` 分享了一篇 [TechCrunch 文章](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/)，宣布了 Stack Overflow 的新 OverflowAPI，Google 将利用它来增强 Google Cloud 的 Gemini。该合作伙伴关系旨在将经过验证的 Stack Overflow 答案直接集成到 Google Cloud 控制台中。

- **Sergey Brin 聚焦 Google 的 Gemini**：用户 `@swyxio` 分享了一条 [推文](https://twitter.com/marvinvonhagen/status/1764036713889116661)，其中 Sergey Brin 讨论了 Google 的人工智能可能通过 Gemini 等计划实现 AGI，引发了广泛关注。

- **Photoshop 中创新的 AI 反射效果**：`@swyxio` 通过分享一个 [LayerDiffusion GitHub 仓库](https://github.com/layerdiffusion/sd-forge-layerdiffusion) 展示了 Stable Diffusion 的创意潜力，该仓库允许用户将物品 Photoshop 到场景中，并带有逼真的反射效果。

- **Claude 3 模型发布引发轰动**：用户们讨论了 Anthropic 的 Claude 3 模型系列的发布；`@jreddy` 分享了该公告，而 `@guardiang` 和 `@thenoahhein` 等用户讨论了其影响和性能，并与现有模型进行了对比，包括正面交锋的总结以及对 Claude 3 中增强的元数据感知能力的观察（[来源推文](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)）。

- **对印度 AI 部署监管的担忧**：用户 `@swyxio` 转发了 Martin Casado 的一条 [推文](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww)，表达了对印度要求在部署 AI 模型前需获得政府批准的担忧，引发了关于潜在政府监管和创新影响的辩论。

**提到的链接**：

- [AI in Production - AI 策略与战术。](https://www.aiinproduction.com/cfp): 未找到描述
- [速率限制](https://docs.anthropic.com/claude/reference/rate-limits): 为了防止滥用并管理 API 的容量，我们对组织使用 Claude API 的额度实施了限制。我们有两种类型的限制：使用限制设定了每月最大...
- [Google 将 Stack Overflow 的知识库引入 Google Cloud 的 Gemini | TechCrunch](https://techcrunch.com/2024/02/29/google-brings-stack-overflows-knowledge-base-to-gemini/): 开发者问答网站 Stack Overflow 今天推出了一项新计划，将通过一个新的 API 为 AI 公司提供其知识库的访问权限，这非常贴切地...
- [来自 david rein (@idavidrein) 的推文](https://x.com/idavidrein/status/1764675668175094169?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Claude 3 在 GPQA 上获得了约 60% 的准确率。我很难形容这些问题有多难——即使是能够访问互联网的博士（与问题领域不同）也只能达到 34%。博士们...
- [介绍下一代 Claude](https://www.anthropic.com/news/claude-3-family): 今天，我们发布了 Claude 3 模型家族，它在广泛的认知任务中树立了新的行业标杆。该家族包括三个按能力递增排序的最先进模型...
- [来自 Swizec Teller (@Swizec) 的推文](https://x.com/swizec/status/1764103976264650840): 我可能找到了买新电脑的借口
- [Suno](https://app.suno.ai/): 未找到描述
- [来自 martin_casado (@martin_casado) 的推文](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJ): 天哪。真是场闹剧。部署模型竟然需要政府批准。这是像 Vinod 这种言论的必然结果。这是反创新的。这是反公众的。我们都...
- [PredictiveChat：一种通过预测最小化对话式 AI 延迟的新方法 – Yohei Nakajima](https://yoheinakajima.com/predictivechat-a-novel-approach-to-minimizing-latency-in-conversational-ai-through-anticipation/): 未找到描述
- [来自 Sankeerth Rao Karingula, Ph.D. (@sankeerth1729) 的推文](https://x.com/sankeerth1729/status/1764240593528705417?s=46&t=90xQ8sGy63D2OtiaoGJuww): 听 Sergey 谈论 AGI、Gemini 以及 Google 的其他举措，并诚实地回答大家的这么多问题，真是太受启发了！
- [来自 Bill Peebles (@billpeeb) 的推文](https://x.com/billpeeb/status/1764074070688088341?s=46&t=90xQ8sGy63D2OtiaoGJuww): “一个外星人自然地融入纽约市，偏执惊悚片风格，35mm 胶片” 由 Sora 生成的视频。
- [来自 Sully (@SullyOmarr) 的推文](https://x.com/sullyomarr/status/1764684780460036144?s=46&t=90xQ8sGy63D2OtiaoGJuww): Anthropic 刚刚杀死了所有小模型吗？如果我没理解错的话，Haiku 的基准测试几乎和 GPT4 一样好，但价格仅为每百万 token 0.25 美元。它绝对完胜 3.5 + OSS...
- [来自 Alex (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1764722513014329620?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 关于我们对 Claude 3 Opus 进行内部测试的一个有趣故事。当我们运行“大海捞针”（needle-in-the-haystack）评估时，它做了一些我从未在 LLM 中见过的事情。背景是，这项测试旨在评估模型的...
- [Twitter 周末总结](https://gist.github.com/nheingit/9abca8536693817eedd614d9571f3b07): Twitter 周末总结。GitHub Gist：即时分享代码、笔记和片段。
- [来自 Anthropic (@AnthropicAI) 的推文](https://x.com/anthropicai/status/1764653830468428150?s=46&t=90xQ8sGy63D2OtiaoGJuww): 今天，我们发布了我们的下一代 AI 模型 Claude 3。这三个最先进的模型——Claude 3 Opus、Claude 3 Sonnet 和 Claude 3 Haiku——在推理等领域树立了新的行业标杆...
- [来自 martin_casado (@martin_casado) 的推文](https://x.com/martin_casado/status/1764408870804623753?s=46&t=90xQ8sGy63D2OtiaoGJuww): 天哪。真是场闹剧。部署模型竟然需要政府批准。这是像 Vinod 这种言论的必然结果。这是反创新的。这是反公众的。我们都...
- [Geoffrey Hinton 教授 - “数字智能会取代生物智能吗？” Romanes 讲座](https://www.youtube.com/watch?v=N1TEjTeQeg0): 被誉为“AI 教父”的 Geoffrey Hinton 教授，CC, FRS, FRSC，于 2 月 19 日星期一在 Sheldonian 剧院发表了牛津大学年度 Romanes 讲座...
- [GitHub - layerdiffusion/sd-forge-layerdiffusion: [开发中] 用于 WebUI 的分层扩散 (通过 Forge)](https://github.com/layerdiffusion/sd-forge-layerdiffusion): [开发中] 用于 WebUI 的分层扩散 (通过 Forge)。通过在 GitHub 上创建账号，为 layerdiffusion/sd-forge-layerdiffusion 的开发做出贡献。

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1213451867422466048) (22 条消息🔥): 

- **寻找 Gemma 见解**：用户 `@drewskidang_82747` 询问了关于 Gemma 的成功案例，但未提供进一步的讨论或细节。
- **Reddit 上的装机喜剧**：`@yamashi` 分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1b4lru9/rate_my_jank_finally_maxed_out_my_available_pcie/) 的链接，展示了一个插满 PCIe 插槽的幽默配置，随后用一句简单的 "i am wheezing"（笑到抽搐）表达了乐不可支。
- **Nvidia Nemo Megatron 工具**：`@le_mess` 发布了 Nvidia NeMo-Megatron-Launcher 的链接并询问是否有人有使用经验，同时附带了 [GitHub URL](https://github.com/NVIDIA/NeMo-Megatron-Launcher)。
- **模型合并技术与工具**：`@yamashi` 询问了如何从较小的模型创建 Mixture of Experts (MoE) 模型，`@dreamgen` 建议查看 GitHub 上的 [mergekit](https://github.com/arcee-ai/mergekit)，该工具用于合并预训练语言模型。
- **关于 LoRA 和 DoRA 的讨论**：`@stoicbatman` 发起了关于 LoRA 与 DORA 比较的讨论，`@nruaif` 和 `@dreamgen` 加入讨论了具体实现并分享了额外研究，包括一篇介绍新型微调方法的 DoRA 论文的 [arXiv 链接](https://arxiv.org/abs/2402.09353)。

**提到的链接**：

- [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)：在这篇论文中，我们展示了 Hu 等人 (2021) 最初引入的 Low Rank Adaptation (LoRA) 在具有大宽度（嵌入维度）的模型中会导致次优的微调效果。这是由于...
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)：在广泛使用的参数高效微调 (PEFT) 方法中，LoRA 及其变体因避免了额外的推理成本而广受欢迎。然而，仍然经常存在...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b4lru9/rate_my_jank_finally_maxed_out_my_available_pcie/)：未找到描述
- [GitHub - NVIDIA/NeMo-Megatron-Launcher: NeMo Megatron launcher and tools](https://github.com/NVIDIA/NeMo-Megatron-Launcher)：NeMo Megatron 启动器和工具。通过在 GitHub 上创建账号来为 NVIDIA/NeMo-Megatron-Launcher 的开发做出贡献。
- [GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.](https://github.com/arcee-ai/mergekit)：用于合并预训练大语言模型的工具。- arcee-ai/mergekit

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1213475689454632990) (12 条消息🔥): 

- **Hugging Face 混乱已解决**：`@giftedgummybee` 提到 Hugging Face `KTO` 的问题已解决，此前发现是使用的 git commit 版本搞混了。
- **Axolotl 移植到 Tinygrad 的可能性较小**：针对 `@realmrfakename` 的询问，`@nanobitz` 确认目前没有将 Axolotl 移植到 Tinygrad 的计划，因为该项目依赖于 Hugging Face transformers 库。
- **Padding Token 难题**：`@realmrfakename` 询问了如何从配置中为模型添加 padding token，并分享了一个关于 tokenizer 中缺少 padding token 的 `ValueError`。
- **频道礼仪提醒**：`@nanobitz` 建议 `@realmrfakename` 将配置和错误相关的提问留在另一个更合适的帮助频道中。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1213443275562819644) (6 条消息): 

- **Optuna CLI 功能建议**：用户 `@casper_ai` 强调了在 axolotl 中需要一个用于 Optuna 超参数优化的 CLI 工具，并引用了 [GitHub issue #1356](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356)。
- **Python 版本引发 GPU 问题**：`@dreamgen` 提到一位用户发现 `python` 与 `python3` 的冲突导致无法使用 GPU，但未提及解决者的 ID。
- **Axolotl 缺失 Tokenizer 文件**：`@dreamgen` 报告了一个严重问题，即 Axolotl 未保存 `tokenizer.json`，但未提供更多细节或解决方案。
- **Deepseed 配置困扰**：在 `@dreamgen` 提到 `python` vs `python3` 问题后，用户 `@c.gato` 解决了 Axolotl 配置中由 Deepseed 引起的 GPU 问题，但未透露具体解决办法。
- **报告 Deepspeed 保存故障**：`@nanobitz` 提出了最近 DeepSpeed 最终保存时出现的问题，导致他们必须回滚到上一个 Checkpoint，并证实其他人也观察到了这一故障。相比之下，`@rtyax` 确认 DeepSpeed ZeRO 3 的最终保存在两天前对他们来说是正常的，使用的是 DeepSpeed 0.13.4。

**提到的链接**：

[Hyperparameter optimization CLI · Issue #1356 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1356)：⚠️ 请检查此功能请求之前是否已被提出。我在 Discussions 的 Ideas 中搜索过，没有发现类似的功能请求。我也搜索了之前的 Issues...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1213883622357213225) (4 条消息): 

- **Mixtral 与 Mistral Large 之谜**：`@dctanner` 询问了 **Mixtral** 和 **Mistral Large** 在合成数据生成方面的性能差异，并思考了后者的潜在性价比。
- **个人模型胜过 Mixtral**：`@le_mess` 指出他们只简短测试了 **Mixtral**，发现它在他们的用途中表现“一般”，更倾向于使用自己的模型。
  

---



### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1213540286941364254) (2 条消息): 

- **关于 "Aristotelian Rescoring" 的见解**：`@crispstrobe` 建议探索 "[Aristotelian Rescoring](https://arxiv.org/pdf/2009.09870.pdf)" 方法，该方法可能适用于复杂的挑战。此外还提到了相关工作，如 STORIUM、FairytaleQA 和 TellMeWhy，并附上了 TellMeWhy 数据集在 [GitHub](https://github.com/StonyBrookNLP/tellmewhy) 和 [Hugging Face](https://huggingface.co/datasets/StonyBrookNLP/tellmewhy) 上的链接。

- **寻求 DPO 优化的合作者**：`@huunguyen` 正在考虑对 DPO 进行微小的优化，并寻求测试方面的帮助。欢迎任何有兴趣合作的人参与。

**提到的链接**：

[GitHub - StonyBrookNLP/tellmewhy: Website for release of TellMeWhy dataset for why question answering](https://github.com/StonyBrookNLP/tellmewhy)：TellMeWhy 数据集发布的网站，用于“为什么”类问答 - StonyBrookNLP/tellmewhy

  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1213448595986972692) (8 条消息🔥): 

- **德语语义相似度提升**：用户 `@sten6633` 通过使用德语特定领域文本微调来自 deepset 的 `gbertlarge`，将其转换为 Sentence Transformer，并进一步使用 Telekom 的释义数据集进行微调，成功增强了语义相似度计算。每一步都带来了显著的改进。

- **"AI in Production" 会议征集讲者**：`@dsquared70` 邀请正在将生成式 AI 集成到生产环境中的开发者在北卡罗来纳州阿什维尔的会议上发表演讲。潜在讲者可以在 [4 月 30 日](https://www.aiinproduction.com/cfp)前申请参加 7 月 18 日和 19 日举行的活动。
  
- **Claude-3 的德语表现尚不明确**：`@bjoernp` 询问 Anthropic 的 Claude-3 在德语方面的表现，并分享了相关链接；而用户 `@devnull0` 提到访问受限以及德语手机号码的问题。

- **Claude AI 在欧盟的访问问题**：`@bjoernp` 通过分享[位置限制链接](https://www.anthropic.com/claude-ai-locations)提醒 Claude AI 在欧盟不可用，尽管 `@devnull0` 提到在 12 月曾使用 tardigrada.io 进行访问。

- **德语手机号成功注册 Claude AI**：与 `@devnull0` 的经历相反，用户 `@sten6633` 表示使用德语手机号注册没有问题。

**提到的链接**：

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp)：未找到描述

  

---

### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1213455217165733928) (3 messages): 

- **发现数据集翻译异常**：`@johannhartmann` 指出德语数据集中存在一个翻译问题，类别名称 "Stem" 被错误地翻译成了 "Stamm"。
- **集成到 FastEval 的工作**：`@johannhartmann` 宣布他们正在将一个数据集集成到 [FastEval](https://github.com/mayflower/FastEval) 中，这是一个用于对聊天语言模型进行真实评估的工具。
- **技术问题已解决**：在遇到可能由线程切换到 asyncio 导致的 VLLM 错误后，`@johannhartmann` 成功解决了问题，并使用命令 `./fasteval -b mt-bench-vago -t chatml -m malteos/hermeo-7b` 成功运行了 FastEval。

**提及的链接**：

[GitHub - mayflower/FastEval: Fast &amp; more realistic evaluation of chat language models. Includes leaderboard.](https://github.com/mayflower/FastEval)：快速且更真实的聊天语言模型评估工具。包含排行榜。 - mayflower/FastEval

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1213464731348762634) (18 messages🔥): 

- **Brezn 的出色表现与未来可能性**：`@thomasrenkert` 认可了 **Brezn-7b** 的成功，而 `@johannhartmann` 透露 Brezn 在德语方面表现优异，归功于合并了与 3 个 DPO 数据集对齐的优质模型，这使得回答更加可靠。Johannhartmann 正在考虑在 Brezn 中默认使用 ChatML，以获得更好的 Benchmark 分数。
  
- **语言模型的合并与 Laser 策略**：`@devnull0` 询问了在对模型进行 lasering 之前的合并过程，促使 `@johannhartmann` 讨论了他在一种被称为 "shotgun training" 的实验性方法中使用 **DARE TIES** 和 lasered 模型的情况。

- **数据集对齐的翻译技术**：`@crispstrobe` 链接了一篇讨论 Prompt 格式对模型输出影响的 Reddit 帖子，并提到了数据集策划的重要性。`@johannhartmann` 使用 AzureML 进行成本效益高且高质量的数据集翻译，并指出了 Mayflower GmbH 对 Hugging Face 上德语 LLM 和数据集的贡献。
  
- **Brezn 作为基座模型的潜力**：`@thomasrenkert` 测试了 **Brezn** 并对其性能表示惊讶，假设将其与 DiscoLM_German_8x7b_v2 结合作为基座模型可能会产生更好的结果。 

- **关于德国仇恨言论数据集相关性的辩论**：`@_chromix_` 和 `@sten6633` 讨论了来自 Zenodo 的德国仇恨言论数据集的优缺点，指出该数据集可能更多地反映了报纸审核的偏见，并且需要进行清洗以避免训练出过度敏感的模型。

**提及的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm)：未找到描述
- [mayflowergmbh (Mayflower GmbH)](https://huggingface.co/mayflowergmbh)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/)：未找到描述
- [ifioravanti (@ivanfioravanti) 的推文](https://x.com/ivanfioravanti/status/1759705134680940828?s=20)：@c_stroebele @ollama @maximelabonne 这是 mergekit 的最大问题。似乎 ChatML 的效果稍好一些。我仍在测试中。也许合并后的一些微调可以帮助推动模型...
- [FreedomIntelligence (FreedomAI)](https://huggingface.co/FreedomIntelligence)：未找到描述
- [RP-Mod &amp; RP-Crowd: Moderator- and Crowd-Annotated German News Comment Datasets](https://zenodo.org/records/5291339)：辱骂和仇恨正渗透到社交媒体和许多新闻媒体公司的评论区。这些平台提供商投入了大量精力来审核用户生成的内容，以防止...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (1 messages): 

dbreunig：这个 Stable Diffusion XL Lightning 的演示让我大受震撼：https://fastsdxl.ai/

### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1213841930316816514) (4 messages): 

- **Artichoke Amusement**: 用户 `@bdexter` 提供了一份极具创意的朝鲜蓟名称列表，包括 **"Choke-a-tastic," "Arti-party,"** 和 **"Leafy Delight"** 等俏皮的绰号。

- **Mistral's High Price Performance**: `@derekpwillis` 测试了新的 **Mistral large model**，并评论其在从文本提取数据方面表现稳健，尽管成本比预期的要高一些。

- **Introducing Claude 3 Plugin**: `@simonw` 宣布了一个用于与 Claude 3 系列模型交互的新插件，并分享了其 GitHub 仓库链接 ([GitHub - simonw/llm-claude-3](https://github.com/simonw/llm-claude-3))。

- **Quick Praise for Plugin Development**: 针对新插件，`@0xgrrr` 迅速称赞了 `@simonw` 对该工具的快速开发。

**Links mentioned**:

[GitHub - simonw/llm-claude-3: LLM plugin for interacting with the Claude 3 family of models](https://github.com/simonw/llm-claude-3): 用于与 Claude 3 系列模型交互的 LLM 插件 - simonw/llm-claude-3

  

---



### Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1214202836079083591) (2 messages): 

- **Invitation to Collaborate Accepted**: 用户 `@wasooli` 表达了对合作项目的兴趣，并询问是否可以进行私信。`@taodoggy` 给予了积极回应，欢迎私信交流。
  

---


### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1213905078289961000) (1 messages): 

- **Calling All AI Enthusiasts**: `@dsquared70` 正在北卡罗来纳州阿什维尔组织一场专注于 **GenAI in production** 的会议，并已开启征稿（call for papers）。欢迎感兴趣的开发者和演讲者在 4 月 30 日前申请，更多详情请见 [AI in Production](https://www.aiinproduction.com/cfp)。 🏔️ 🍻

**Links mentioned**:

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): 未找到描述

  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1213905737177374720) (3 messages): 

- **AI in Production Conference Call**: `@dsquared70` 邀请将 GenAI 集成到生产环境中的开发者在阿什维尔的会议上发表演讲。详情和征稿信息可在 [AI in Production Call for Presentations](https://www.aiinproduction.com/cfp) 找到，提交截止日期为 4 月 30 日。

- **A Bright "Yolks" Morning**: `@oleegg` 用一句俏皮的 "good morning yokks" 向频道打招呼，随后将其纠正为 "yolks"。

**Links mentioned**:

[AI in Production - AI strategy and tactics.](https://www.aiinproduction.com/cfp): 未找到描述

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1213932721420636190) (3 messages): 

- **Hackathon Confusion Cleared Up**: 用户 `@needforspeed4` 询问在 **Agape** 举办的黑客松是否与管理此 Discord 服务器的 **AI Engineer Foundation** 有关。他们还询问是否每个黑客松都会使用不同的 Discord。
- **Distinct Hackathon Entities**: `@hackgoofer` 澄清说，**The AI Engineer Foundation Hackathons** 确实是在此 Discord 中举办的，但是 Agape 黑客松并不隶属于 **AI Engineer Foundation**。