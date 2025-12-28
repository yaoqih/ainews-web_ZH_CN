---
companies:
- openai
- google-deepmind
- nyu
- uc-berkeley
- hku
date: '2025-02-04T02:44:29.143732Z'
description: '**OpenAI** 发布了 **o3** 智能体的完整版本，其全新的 **Deep Research** 变体在 **HLE 基准测试**中表现出显著提升，并在
  **GAIA** 上达到了业界领先（SOTA）水平。此次发布包含一张展示严谨研究的“推理时间扩展”（inference time scaling）图表，尽管公开测试集的结果引发了一些争议。


  该智能体被评价为“极其简单”，目前每月限额 100 次查询，并计划推出更高频率的版本。外界反响总体积极，但也存在部分质疑。此外，强化学习领域的进展也备受瞩目，其中包括一种名为“**预算强制**”（budget
  forcing）的简单测试时扩展技术，该技术使数学竞赛的推理表现提升了 27%。


  来自 **Google DeepMind**、**纽约大学（NYU）**、**加州大学伯克利分校（UC Berkeley）**和**香港大学（HKU）**的研究人员共同参与了这些研究。原
  **Gemini Deep Research** 团队也将参加即将举行的 AI Engineer NYC 活动。'
id: 5a90e352-28be-4325-900c-7b5b71490f5d
models:
- o3
- o3-mini-high
- o3-deep-research-mini
original_slug: ainews-openai-takes-on-geminis-deep-research
people:
- sama
- danhendrycks
- ethan-mollick
- dan-shipper
title: OpenAI 对标 Gemini 的 Deep Research。
topics:
- reinforcement-learning
- benchmarking
- inference-speed
- model-performance
- reasoning
- test-time-scaling
- agent-design
---

<!-- buttondown-editor-mode: plaintext -->**o3 和 tools 就是你所需要的一切。**

> 2025年1月31日至2月3日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord（**225** 个频道和 **16942** 条消息）。预计为你节省了阅读时间（以 200wpm 计算）：**1721 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在介绍 Operator（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-openai-launches-operator-its-first-agent/)）时，sama 暗示了更多 OpenAI Agents 即将推出，但很少有人预料到下一个会在 9 天后发布，而且还是在美东时间周日[从日本](https://x.com/OpenAI/status/1886149471249264675?t=0O8ujtyOOzkt3VZ6dk_alg&s=19)发布的：

https://www.youtube.com/watch?v=YkCDVn3_wiw

这篇 [blogpost](https://openai.com/index/introducing-deep-research/) 提供了更多关于预期用例的见解，但值得注意的是 Deep Research 在 Dan Hendrycks 的新 HLE benchmark 上的结果，比上周五发布的 o3-mini-high 的结果翻了一倍多（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-o3-mini-launches-openai-on-wrong-side-of/)）。


![image.png](https://assets.buttondown.email/images/dd6ff729-2535-4fce-b0a7-d2a8be67b4be.png?w=960&fit=max)


他们还发布了在 GAIA 上的 SOTA 结果——这遭到了[共同作者的批评](https://x.com/clefourrier/status/1886385835324457143?s=46)，因为他们只发布了公开测试集的结果——对于一个可以浏览网页的 Agent 来说，这显然是有问题的，尽管没有理由质疑其完整性，特别是在[脚注](https://openai.com/index/introducing-deep-research/#citation-bottom-1)中已确认且 GAIA 测试轨迹样本已发布的情况下。

OAIDR 附带了其自身版本的 "inference time scaling" 图表，非常令人印象深刻——令人印象深刻的不是图表本身的缩放，而是在研究过程中表现出的严谨性，使得生成这样的图表成为可能（当然，假设这是 research 而非 marketing，但在这里，为了推销每月 200 美元的订阅，两者的界限不幸地变得模糊了）。


![image.png](https://assets.buttondown.email/images/2ff54e1f-b29c-4bbc-aec4-2947da8b9fc7.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/5dc850f2-c633-4032-aa00-fda6fd4db769.png?w=960&fit=max)


OpenAI 员工[确认](https://x.com/sherwinwu/status/1886256203077915071?s=46)这是 full o3 首次在野外发布（gdb 说它是一个“[极其简单的 agent](https://x.com/gdb/status/1886229270428848399?s=46)”），blogpost 指出 "o3-deep-research-mini" 版本正在路上，它将提高目前每月 100 次查询的速率限制。

反响大多是[积极的](https://x.com/afinetheorem/status/1886206439582015870?s=46)，有时甚至到了[过度兴奋](https://x.com/dharmesh/status/1886510930420195816)的程度。有些人正在[嘲笑](https://x.com/distributionat/status/1886238792870461451)这种夸张，但总的来说，我们倾向于同意 [Ethan Mollick](https://x.com/emollick/status/1886205847803429173?s=46) 和 [Dan Shipper](https://x.com/danshipper/status/1886203397004783996?s=46) 的积极看法，尽管我们也经历了[很多失败](https://x.com/nabeelqu/status/1886493788459413623)。

---

**厚脸皮的广告**：我们将于 2 月 20 日至 22 日在 AI Engineer NYC 邀请多位 Deep Research 和其他 Agent 构建者，[包括原 Gemini Deep Research 团队](https://x.com/swyx/status/1886180259609170345)。[最后一次申请机会](https://apply.ai.engineer/)！


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**强化学习 (RL) 和 AI 研究的进展**

- **强化学习的简化及其对 AI 的影响**：[@andersonbcdefg 改变了主意，现在认为 RL 很简单](https://twitter.com/andersonbcdefg/status/1886319033949262245)，反映了 RL 技术在 AI 研究中的可及性。

- **s1：AI 模型中简单的 Test-Time Scaling**：[@iScienceLuvr 分享了一篇关于 s1 的论文，证明仅在 1,000 个样本上进行 next-token prediction 训练，并通过一种名为 **budget forcing** 的简单 test-time 技术控制思考时间，就能产生强大的推理模型](https://twitter.com/iScienceLuvr/status/1886249466203910418)。该模型在竞赛数学题上的表现比之前的模型高出多达 **27%**。更多讨论可以在[这里](https://twitter.com/_akhaliq/status/1886244987551052061)和[这里](https://twitter.com/omarsar0/status/1886428631041225030)找到。

- **RL 提升模型对新任务的适应能力**：[来自 **Google DeepMind**、**NYU**、**UC Berkeley** 和 **HKU** 的研究人员发现，**强化学习（RL）提高了模型对新的、未见过的任务变体的适应能力**，而监督微调（supervised fine-tuning）虽然会导致记忆化，但对于模型稳定仍然至关重要](https://twitter.com/TheTuringPost/status/1886465061763604844)。

- **对 DeepSeek r1 的评述及 s1 的引入**：[@Muennighoff 介绍了 s1，它仅通过 **1K 个高质量示例**和简单的测试时干预（test-time intervention），就复现了 o1-preview 的缩放（scaling）和性能，解决了 DeepSeek r1 的数据密集型问题](https://twitter.com/Muennighoff/status/1886405528777073134)。

**OpenAI 的 Deep Research 与推理模型**

- **OpenAI Deep Research 助手发布**：[@OpenAI 宣布 Deep Research 现已向所有 Pro 用户开放](https://twitter.com/markchen90/status/1886341752245915903)，为复杂的知识任务提供强大的 AI 工具。[@nickaturley 强调了](https://twitter.com/nickaturley/status/1886278961216495968)其通用用途以及改变工作、家庭和学校任务的潜力。

- **测试时缩放效率（Test-Time Scaling Efficiency）的提升**：[@percyliang 强调了仅使用 **1K 个精心挑选的示例**实现**测试时缩放效率**的重要性](https://twitter.com/percyliang/status/1886490467497553944)，鼓励开发能提高单位预算能力的方法。

- **初窥 OpenAI o3 的能力**：[@BorisMPower 对 **o3** 的能力表示兴奋](https://twitter.com/BorisMPower/status/1886274086902620559)，指出它在节省资金和减少对专家分析依赖方面的潜力。

**Qwen 模型进展与 AI 技术突破**

- **R1-V：增强视觉语言模型（Vision Language Models）的超强泛化能力**：[@_akhaliq 分享了 **R1-V** 的发布，展示了一个 **2B 模型**在仅 **100 个训练步数**内，就能在分布外（out-of-distribution）测试中超越 **72B 模型**](https://twitter.com/_akhaliq/status/1886246324754133304)。该模型显著提升了长上下文和关键信息检索的性能。

- **Qwen2.5-Max 在 Chatbot Arena 中的强劲表现**：[@Alibaba_Qwen 宣布 **Qwen2.5-Max** 目前在 Chatbot Arena 中排名 **第 7**，超越了 DeepSeek V3、o1-mini 和 Claude-3.5-Sonnet](https://twitter.com/Alibaba_Qwen/status/1886485743998279944)。它在**数学和编程方面排名第 1**，在**困难提示词（hard prompts）方面排名第 2**。

- **s1 模型超越 o1-Preview**：[@arankomatsuzaki 指出，**s1-32B** 在基于 **Qwen2.5-32B-Instruct** 进行监督微调后，在竞赛数学题上超越 o1-preview 高达 **27%**](https://twitter.com/arankomatsuzaki/status/1886250066324910089)。该模型、数据和代码已向社区开源。

**AI 安全与防御越狱（Jailbreaks）**

- **Anthropic 针对通用越狱的宪法分类器（Constitutional Classifiers）**：[@iScienceLuvr 讨论了 **Anthropic 引入的宪法分类器**](https://twitter.com/iScienceLuvr/status/1886253192817881334)，这是基于合成数据训练的防护措施，旨在防止通用越狱。超过 **3,000 小时的红队测试（red teaming）**显示，没有任何成功的攻击能从受保护的模型中提取详细信息。

- **Anthropic 测试新安全技术的演示**：[@skirano 宣布了 **Anthropic** 的一项新研究预览](https://twitter.com/skirano/status/1886455588177035615)，邀请用户尝试越狱其受宪法分类器保护的系统，旨在增强 AI 安全措施。

- **关于 AI 模型幻觉（Hallucination）的讨论**：[@OfirPress 分享了对 AI 模型**幻觉**的担忧](https://twitter.com/OfirPress/status/1886367776576552977)，强调即使在像 OpenAI Deep Research 这样先进的系统中，这仍然是一个重大问题。

**面向开发者的 AI 工具与平台**

- **用于 Vibe Coding 的 SWE Arena 发布**：[@terryyuezhuo 发布了 **SWE Arena**，一个 vibe coding 平台](https://twitter.com/terryyuezhuo/status/1886450697497120891)，支持实时代码执行和渲染，涵盖了各种前沿 **LLM** 和 **VLM**。[@_akhaliq 也强调了 SWE Arena](https://twitter.com/_akhaliq/status/1886452970520293864)，并指出了其令人印象深刻的能力。

- **Perplexity AI 助手的增强**：[@AravSrinivas 介绍了 **Perplexity Assistant** 的更新](https://twitter.com/AravSrinivas/status/1886266264777281974)，鼓励用户在 **Nothing phone** 等新设备上尝试，并提到了即将推出的功能，如集成到 **Android Auto**。他还宣布**带有网页搜索和推理链（reasoning traces）的 o3-mini** 已向所有 Perplexity 用户开放，Pro 用户每天可使用 500 次（[推文链接](https://twitter.com/AravSrinivas/status/1886495695655592262)）。

- **Llama 开发工具的进展**：[@ggerganov 宣布 **llama.vscode** 的安装量已超过 **1000 次**](https://twitter.com/ggerganov/status/1886313165710917968)，提升了基于 llama 模型开发的体验。他在此分享了一个快乐的 [**llama.cpp** 用户是什么样子的](https://twitter.com/ggerganov/status/1886493193518100727)。

**梗与幽默**

- **对 AI 研究和命名能力的观察**：[@jeremyphoward 幽默地指出，一个人不可能既是强大的 AI 研究员又**擅长给事物命名**](https://twitter.com/jeremyphoward/status/1886260032054182209)，并称这是所有已知文化中的普遍事实。

- **对天赋的代际反思**：[@willdepue 评论道，**Gen Z** 要么是极具天赋的个人，要么是“完全的废人”，他将这种两极分化归因于**互联网**，并预见 AI 将加速这一趋势](https://twitter.com/willdepue/status/1886305687686590628)。

- **对界面设计的幽默看法**：[@jeremyphoward 调侃说他的主屏幕上只有 **5 个 Grok 图标**，并建议可以放更多](https://twitter.com/jeremyphoward/status/1886258920253280363)，以此幽默地调侃技术设计。

- **快乐的 llama.cpp 用户**：[@ggerganov 分享了一张描绘快乐的 **llama.cpp** 用户样子的图片](https://twitter.com/ggerganov/status/1886493193518100727)，为 AI 社区增添了一抹轻松的色彩。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. AI 模型硬件的范式转移：从 GPU 到 CPU+RAM**

- **[范式转移？](https://i.redd.it/gre7z74ylxge1.jpeg)** ([得分: 532, 评论: 159](https://reddit.com/r/LocalLLaMA/comments/1igpwzl/paradigm_shift/))：该帖子暗示了 AI 模型处理可能从**以 GPU 为中心**的方法转向 **CPU+RAM** 配置的**范式转移**，特别强调了 **AMD EPYC 处理器**和内存模块的使用。这种转变通过对比图像生动地展示：一个人拒绝 GPU 而认可 CPU+RAM 设置，表明 AI 计算的硬件偏好可能发生变化。
  - **CPU+RAM 的可行性**：转向 **AMD EPYC 处理器**和海量内存配置被认为对个人用户具有性价比，但对于服务多用户而言，**GPU** 仍然是首选。构建 EPYC 系统的成本显著更高，估计在 **$5k 到 $15k** 之间，且性能通常比 GPU 配置慢。
  - **性能与配置**：讨论焦点在于优化配置，例如使用**双路 12 通道**系统并确保插满所有内存插槽以获得最佳性能。一些用户报告在特定模型上达到了 **5.4 tokens/second**，而另一些用户则指出 **I/O 瓶颈**和未利用所有核心会影响性能。
  - **潜在突破与 MoE 模型**：讨论还涉及了 **Mixture of Experts (MoE) 模型**的潜力，这可能允许直接从高速 NVMe 存储读取 LLM 权重，从而减少激活参数。这可能会改变当前的硬件需求，但此类进展的可行性和时间表仍不确定。

**主题 2. Mistral、Qwen 和 DeepSeek 在美国境外的崛起**

- **Mistral, Qwen, Deepseek** ([得分: 334, 评论: 114](https://reddit.com/r/LocalLLaMA/comments/1iggwff/mistral_qwen_deepseek/))：**Mistral AI**、**Qwen** 和 **DeepSeek** 等非美国公司正在发布开源模型，与美国同类模型相比，这些模型更易获取且体积更小。这凸显了一个趋势，即国际公司在向公众普及 AI 技术方面处于领先地位。
  - **Mistral 3 small 24B** 模型获得了积极反馈，多位用户强调了其有效性和易用性。**Qwen** 因其多样的模型尺寸而受到关注，与 **Meta 的 Llama** 模型相比，它在不同硬件上提供了更多的灵活性和可用性，而后者因尺寸选择有限和专有许可而受到批评。
  - 关于**美国与国际 AI 模型**的讨论显示出对美国当前产品的怀疑，一些用户更青睐来自中国等地的国际模型，因为它们的开源性质和极具竞争力的性能。**Meta** 被提及开启了权重开放（open weights）趋势，但用户对其依赖大模型和专有许可表示担忧。
  - 关于公司保持 AI 模型权重开放或封闭的战略利益存在争论。一些人认为领先公司保持权重封闭以维持竞争优势，而挑战者则发布开放权重以削弱这些领导者。预计 **Meta 的 Llama 4** 将整合 **DeepSeek R1** 的创新以保持竞争力。

**主题 3. Phi 4 模型在资源受限硬件上受到关注**

- **Phi 4 被严重低估了** ([Score: 207, Comments: 84](https://reddit.com/r/LocalLLaMA/comments/1igf1vi/phi_4_is_so_underrated/)): 作者赞扬了 **Phi 4 模型** ([Q8, Unsloth 变体](https://huggingface.co/unsloth/phi-4-GGUF)) 在有限硬件（如 **M4 Mac mini (24 GB RAM)**）上的表现，认为它在常识问题和编程提示词等任务上可与 **GPT 3.5** 媲美。作者对其能力表示满意，并不关心正式的基准测试（benchmarks），强调个人体验优于技术指标。
  - **Phi 4 的优势与局限性：** 用户称赞 **Phi 4** 在特定领域（如知识库和规则遵循）的强劲表现，在指令遵循方面甚至优于更大的模型。然而，它在小语种方面表现不佳，在非英语环境下输出质量较差，且缺乏 **128k context version**，这限制了它与 **Phi-3** 相比的潜力。
  - **用户体验与实现：** 许多用户分享了在各种工作流中使用 **Phi 4** 的积极体验，强调了它在提示词增强和创意基准测试（如鸡尾酒创作）等任务中的多功能性和有效性。然而，一些用户报告在特定任务（如工单分类）中效果不佳，而 **Llama 3.3** 和 **Gemma2** 等其他模型在这些任务中表现更好。
  - **工具与工作流集成：** 讨论包括在自定义设置（如 **Roland** 和 **WilmerAI**）中使用 **Phi 4**，通过将其与 **Mistral Small 3** 和 **Qwen2.5 Instruct** 等其他模型结合来增强问题解决能力。社区还探索了 **n8n** 和 **omniflow** 等工作流应用，以便将 **Phi 4** 集成到更广泛的 AI 系统中，并提供了详细设置和工具的链接 ([WilmerAI GitHub](https://github.com/SomeOddCodeGuy/WilmerAI))。


**主题 4. DeepSeek-R1 在复杂问题解决中的能力**

- **DeepSeek-R1 永不松懈...** ([Score: 133, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1ign0lz/deepseekr1_never_ever_relaxes/)): **DeepSeek-R1** 模型通过解决一个涉及回文数的数学问题展示了自我纠错能力，最初犯了错误，但在完成回答之前自行纠正了。值得注意的是，**OpenAI o1** 是唯一解决该问题的其他模型，而包括 **chatgpt-4o-latest-20241120** 和 **claude-3-5-sonnet-20241022** 在内的其他几个模型都失败了，这引发了关于 tokenizers、采样参数或非思考型 LLM 固有数学能力的潜在问题的讨论。
  - 讨论强调了 LLM 的**自我纠错能力**，特别是在 zero-shot 设置下。这种能力源于模型接触到的训练数据中包含错误被纠正的情况（例如在 Stack Overflow 等平台上），从而影响后续的 token 预测以纠正错误。
  - **DeepSeek-R1** 以及 **Mistral Large 2.1** 和 **Gemini Thinking on AI Studio** 等其他模型成功解决了回文数问题，同时探讨了**Chain-of-Thought (CoT)** 模型。CoT 模型与非 CoT 模型形成了对比，后者由于训练范式不同，通常难以在回答过程中纠正错误。
  - 对话深入探讨了**代际模型**（如 gen1, gen1.5, gen2）之间训练数据的根本差异，以及这些差异对纠错能力的影响。有建议认为，将模型输出作为用户输入进行验证可能有助于解决这些挑战。


## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. DeepSeek 与深度研究：颠覆性的 AI 挑战**

- **[Deep Research Replicated Within 12 Hours](https://i.redd.it/ymwa6kmutxge1.png)** ([Score: 746, Comments: 93](https://reddit.com/r/OpenAI/comments/1igqunl/deep_research_replicated_within_12_hours/)): 该帖子重点介绍了 **Ashutosh Shrivastava** 的推文，内容关于在 12 小时内快速创建了 "Open DeepResearch" —— 这是 **OpenAI** Deep Research Agent 的对应版本。它包含的代码片段可以通过针对性的网络搜索和 URL 访问，通过检查估值、增长和采用率指标来比较 **Cohere**、**Jina AI** 和 **Voyage** 等 AI 公司。
  - 许多评论者认为 **OpenAI** 的 **Deep Research** 更胜一筹，因为它使用了**强化学习** (RL)，这使其能够自主学习复杂任务的策略，而其他缺乏此功能的模型则不然。**Was_der_Fall_ist** 强调，如果没有 RL，像 "Open DeepResearch" 这样的工具只是复杂的 Prompt，而不是真正的 Agent，可能会导致脆弱性和不可靠性。
  - 讨论强调了不仅要关注模型，还要关注围绕它们的**工具和应用**，正如 **frivolousfidget** 所指出的。他们认为，通过创新性地使用现有模型，而不是仅仅通过模型改进，可以实现显著的能力提升，并引用了 **AutoGPT** 和 **LangChain** 等例子。
  - **GitHub** 链接以及关于模型成本和可访问性的讨论强调了与 OpenAI 等顶尖解决方案竞争的财务障碍。**YakFull8300** 提供了一个 [GitHub 链接](https://github.com/jina-ai/node-DeepResearch) 以供进一步探索，而其他人则讨论了与高级 AI 模型训练和部署相关的昂贵成本。


- **[DeepSeek might not be as disruptive as claimed, firm reportedly has 50,000 Nvidia GPUs and spent $1.6 billion on buildouts](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-might-not-be-as-disruptive-as-claimed-firm-reportedly-has-50-000-nvidia-gpus-and-spent-usd1-6-billion-on-buildouts)** ([Score: 535, Comments: 157](https://reddit.com/r/OpenAI/comments/1igh5oq/deepseek_might_not_be_as_disruptive_as_claimed/)): 据报道，**DeepSeek** 拥有 **50,000 块 Nvidia GPU**，并在基础设施建设上投资了 **16 亿美元**，这引发了对其声称的在 AI 行业具有颠覆性影响的质疑。其投资规模表明了巨大的计算能力，但人们对其技术进步是否与其资金投入相匹配持怀疑态度。
  - 讨论中充满了对 **DeepSeek 主张** 的怀疑，一些用户质疑其报告的成本和 GPU 使用情况的真实性。**DeepSeek 的论文** 清楚地阐述了训练成本，但许多人认为媒体误读了这些数字，导致了对其实际支出的误导和混乱。
  - 关于 **DeepSeek 的开源模型** 是否代表了 AI 领域的重大进步存在争议，一些人认为它挑战了美国在 AI 发展中的主导地位。批评者认为，**西方媒体和恐华症 (sinophobia)** 促成了 DeepSeek 的成就被夸大或误导的说法。
  - DeepSeek 发布公告带来的**财务影响**（如 **Nvidia 股价下跌 17%**）是一个焦点，用户注意到了这对 AI 硬件市场的更广泛影响。一些用户认为，DeepSeek 模型的开源性质允许具有成本效益的 AI 开发，有可能使 AI 技术的获取变得民主化。

- **[EU and UK waiting for Sora, Operator and Deep Research](https://i.redd.it/gora66d1ywge1.jpeg)** ([Score: 110, Comments: 23](https://reddit.com/r/OpenAI/comments/1ignel2/eu_and_uk_waiting_for_sora_operator_and_deep/)): 帖子提到 **EU** 和 **UK** 正在等待 **Sora**、**Operator** 和 **Deep Research** 工具，但未提供更多细节或背景。配图描绘了一个处于各种沉思姿态的男人，暗示了反思与孤独的主题，但与帖子主题缺乏直接关联。
  - **可用性与定价担忧**：用户对 **Sora** 在 **UK** 和 **EU** 的延迟上线表示沮丧，推测延迟是由于 **OpenAI** 自身原因还是政府监管。一些人对每月支付 **$200** 的服务费用持怀疑态度，还有传言称下周可能会面向 **Plus tier** 发布，但对其时间表仍存疑。
  - **性能与实用性**：一位用户分享了该工具的正面体验，指出它在 **14 分钟** 内生成了一份带有 **LaTeX** 格式 **APA citations** 的 **10 页文献综述**。这突显了该工具在处理复杂任务时令人印象深刻的能力和效率。
  - **监管与运营见解**：有推测认为，延迟可能是 **OpenAI** 影响政策制定者的战略举措，或者是由于资源分配问题，特别是在处理 **Republic of Ireland** 的用户活动方面。讨论表明，监管过程理想情况下应增强模型安全性，并将 **OpenAI** 的延迟与其他能在 **UK** 和 **EU** 实现同步首发的 AI 公司进行了对比。


**Theme 2. OpenAI's New Hardware Initiatives with Jony Ive**

- **[Open Ai is developing hardware to replace smartphones](https://i.redd.it/m8nwc1251uge1.png)** ([Score: 279, Comments: 100](https://reddit.com/r/OpenAI/comments/1ige8he/open_ai_is_developing_hardware_to_replace/)): 据报道，**OpenAI** 正在开发一种旨在取代智能手机的新型 **AI device**，正如 CEO **Sam Altman** 所宣布的那样。来自 **Nikkei**（2025 年 2 月 3 日）的新闻文章还提到了 **Altman** 试图通过生成式 AI 转型 IT 行业的野心，以及他即将与日本首相的会面。
  - **OpenAI 的 AI 硬件野心**：**Sam Altman** 宣布了针对 AI 专用硬件和芯片的计划，这可能会像 2007 年 **iPhone** 发布一样颠覆科技硬件行业。他们计划与 **Jony Ive** 合作，将“语音”界面作为核心功能，原型预计将在“几年内”问世（[Nikkei 来源](https://asia.nikkei.com/Editor-s-Picks/Interview/OpenAI-will-develop-AI-specific-hardware-CEO-Sam-Altman-says)）。
  - **对取代智能手机的怀疑**：许多评论者对取代智能手机的可行性表示怀疑，强调了屏幕在视频和阅读方面的持久用途。他们对将“语音”作为主要交互界面表示怀疑，质疑它如何能取代智能手机的视觉和交互元素。
  - **新兴 AI 助手**：**Gemini** 被视为 **Google Assistant** 日益强大的竞争对手，它已集成到 **Samsung** 设备中，并可在 **Android OS** 中被选为首选助手。**Gemini** 向 **Google Home** 和 **Nest** 设备的潜在扩展正处于 **beta** 阶段，预示着 AI 助手技术的转变。


- **[Breaking News: OpenAI will develop AI-specific hardware, CEO Sam Altman says](https://asia.nikkei.com/Editor-s-Picks/Interview/OpenAI-will-develop-AI-specific-hardware-CEO-Sam-Altman-says)** ([Score: 138, Comments: 29](https://reddit.com/r/OpenAI/comments/1ige3kb/breaking_news_openai_will_develop_aispecific/)): **OpenAI** 计划开发 **AI-specific hardware**，正如 CEO **Sam Altman** 所宣布的那样。这一战略举措标志着在增强 AI 能力和基础设施方面迈出了重要一步。
  - **闭源担忧**：用户对 **OpenAI** 举措的开放性持怀疑态度，指出“Open” AI 开发闭源软件和硬件具有讽刺意味。这反映了公众对 AI 开发透明度和可访问性的广泛关注。
  - **与 Jony Ive 的合作**：与 **Jony Ive** 的合作被视为一项战略举措，可能导致自 **2007 iPhone launch** 以来最大的科技硬件变革。重点是创建一种利用 AI 进步来增强用户交互的新型硬件。
  - **定制 AI 芯片**：**OpenAI** 正在致力于开发自己的半导体，加入 **Apple, Google, and Amazon** 等大型科技公司的行列。这一举措是定制芯片大趋势的一部分，旨在提高 AI 性能，原型预计将在“几年内”推出，并强调语音功能。


**Theme 3. Critique on AI Outperforming Human Expertise Claims**

- **[指数级进步 - AI 现在在各自领域超越了人类博士专家](https://i.redd.it/fcajb79tezge1.png)** ([得分: 176, 评论: 86](https://reddit.com/r/OpenAI/comments/1igypel/exponential_progress_ai_now_surpasses_human_phd/)): 该帖子讨论了一张名为 **"Performance on GPQA Diamond"** 的图表，该图表比较了人类博士专家与 AI 模型 **GPT-3.5 Turbo** 及 **GPT-4o** 随时间变化的准确率。图表显示 AI 模型呈上升趋势，在 2023 年 7 月至 2025 年 1 月期间，其准确率在 0.2 到 0.9 之间，已超越了各自领域的专家。
  - **AI 的局限性与误导性主张**：评论者认为，虽然 AI 模型擅长模式识别和数据检索，但并不具备真正的推理或科学发现能力（如治愈癌症）。他们强调，AI 在特定测试中超越博士并不等同于在实际、现实世界的问题解决中超越人类专业知识。
  - **对指数级提升主张的批评**：AI 模型呈指数级提升的观点被批评为具有误导性，一位评论者将其比作一种偏见指标，不能真实反映人类专业知识的复杂性和深度。讨论强调，虽然 AI 在理论知识方面表现出色，但缺乏进行实验和做出新发现的能力。
  - **对 AI 专业能力的怀疑**：许多人对 AI 在没有专家指导的情况下提供博士级见解的能力表示怀疑，将 AI 比作高级搜索引擎而非真正的专家。人们对 AI 模型已超越博士的说法表示担忧，认为这些说法更多归因于营销而非实际能力。


- **[Stability AI 创始人："我们显然处于智能起飞情景中"](https://i.redd.it/p77k5xj21yge1.png)** ([得分: 127, 评论: 122](https://reddit.com/r/OpenAI/comments/1igrrny/stability_ai_founder_we_are_clearly_in_an/)): **Stability AI** 创始人 **Emad Mostaque** 断言，我们正处于“智能起飞情景（intelligence takeoff scenario）”中，机器很快将在数字知识任务上超越人类。他强调需要超越对 **AGI** 和 **ASI** 的讨论，预测机器效率、成本效益和协调能力的提升，同时敦促考虑这些进步的影响。
  - 许多评论者对 AI 即将取代人类表示怀疑，并举例说明了使用 **o3-mini** 和 **o1 pro** 等 AI 模型生成简单代码任务时面临的挑战。**RingDigaDing** 等人认为，尽管基准测试显示 AI 接近 **AGI**，但在现实场景的可靠性和实际应用方面仍面临困难。
  - **IDefendWaffles** 和 **mulligan_sullivan** 讨论了 AI 炒作背后的动机，提到了投资利益以及关于 **AGI** 即将到来的主张缺乏事实证据。他们强调需要有根据的论点，并指出当前 AI 能力与未来投机性 AI 进步之间的区别。
  - 用户如 **whtevn** 和 **traumfisch** 讨论了 AI 增强人类工作的潜力，**whtevn** 分享了使用 AI 作为开发助手的经验。他们强调 AI 能够高效执行任务（尽管仍需人类监督），以及 AI 逐渐而非瞬间改变行业的潜力。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking (gemini-2.0-flash-thinking-exp) 生成的摘要之摘要

**主题 1. DeepSeek AI 的崛起与监管审查**

- **DeepSeek 模型抢尽风头，表现超越西方巨头**：中国的 **DeepSeek AI** 模型目前在基准测试中表现优于 **OpenAI** 和 **Anthropic** 等西方竞争对手，引发了全球关于 AI 霸权的讨论。[DeepSeek AI 主导西方基准测试](https://www.youtube.com/watch?v=yjaoT5-tz0I)。这一性能飞跃促使美国采取立法行动，限制与中国 AI 研究的合作，以保护国家创新。
- **DeepSeek 安全护盾破碎，引发越狱狂潮**：Cisco 的研究人员发现 **DeepSeek R1** 模型未能通过 100% 的安全测试，无法阻止有害提示词。[DeepSeek R1 性能问题](https://www.pcmag.com/news/deepseek-fails-every-safety-test-thrown-at-it-by-researchers)。用户还报告了服务器访问困难的问题，尽管其基准测试表现出色，但其实际应用的可靠性仍存疑。
- **美国立法者拔剑相向，针对 DeepSeek 提出严厉法案**：参议员 Josh Hawley 提出立法以遏制美国与中国的 AI 合作，专门针对 **DeepSeek** 等模型。[AI 监管面临新的立法推动](https://annas-archive.org/blog/ai-copyright.html)。该法案建议对违规行为处以最高 20 年的监禁，引发了人们对扼杀开源 AI 创新和可访问性的担忧。

**主题 2. OpenAI 的 o3-mini：性能与公众审视**

- **o3-mini AMA：Altman 与 Chen 面对关于新模型的质疑**：OpenAI 安排了一场由 **Sam Altman** 和 **Mark Chen** 主持的 **AMA**（问我任何事）活动，以回应社区关于 **o3-mini** 的疑问。[OpenAI 安排 o3-mini AMA](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/)。用户正通过 Reddit 提交问题，渴望了解未来发展并提供关于该模型的反馈。
- **o3-mini 的推理能力受到质疑，Sonnet 依然称王**：用户反映 **o3-mini** 在编程任务中的表现褒贬不一，理由是速度较慢且解决方案不完整。[o3-mini 面临性能批评](https://x.com/hxiao/status/1885700308720091280)。**Claude 3.5 Sonnet** 仍然是许多开发者的首选，因为它具有一致的可靠性和速度，特别是在处理复杂代码库时。
- **o3-mini 发布 "Deep Research" Agent，但疑虑尚存**：OpenAI 推出了 **Deep Research**，这是一款由 **o3-mini** 驱动的新型 Agent，旨在进行自主信息综合和报告生成。[OpenAI 发布 Deep Research Agent](https://openai.com/index/introducing-deep-research/)。尽管前景看好，但用户已经注意到其在输出质量和来源分析方面的局限性，一些人发现 **Gemini Deep Research** 在综合任务中更有效。

**主题 3. AI 工具与 IDE：变革之风**

- **Windsurf 1.2.5 补丁：Cascade 获得 Web 超能力，DeepSeek 仍存在 Bug**：Codeium 发布了 **Windsurf 1.2.5 补丁**，通过自动触发器和 **@web**、**@docs** 等新命令增强了 **Cascade web search**。[Windsurf 1.2.5 补丁更新发布](https://www.codeium.com/changelog)。然而，用户报告在 Windsurf 中使用 **DeepSeek 模型** 时存在持续性问题，包括无效的工具调用和上下文丢失，从而影响了额度使用。
- **Aider v0.73.0：适配 o3-mini，推理增加“努力程度”调节**：Aider 发布了 **v0.73.0**，增加了对 **o3-mini** 的全面支持，以及用于控制推理的新参数 `--reasoning-effort`。[Aider v0.73.0 发布并增强功能](https://aider.chat/HISTORY.html)。尽管集成了 o3-mini，用户发现 **Sonnet** 在编程任务中仍然更快、更高效，即便 o3-mini 在复杂逻辑方面表现出色。
- **Cursor IDE 更新发布，更新日志依然晦涩难懂**：Cursor IDE 推出了包括检查点恢复功能在内的更新，但用户对不一致的更新日志和未公开的功能更改表示不满。[Cursor IDE 推出新功能](https://www.cursor.com/changelog)。用户对性能差异以及在缺乏明确沟通的情况下更新对模型能力的影响表示担忧。

**主题 4. LLM 训练与优化：新技术涌现**

- **Unsloth 的动态量化缩小了模型体积，同时保持了性能**：Unsloth AI 强调了动态量化技术，在不牺牲准确性的情况下，使 **DeepSeek R1** 等模型的体积减少了高达 **80%**。[Unsloth 框架中的动态量化](https://unsloth.ai/blog/deepseek-r1)。用户正在尝试 **1.58-bit 量化模型**，但在确保符合位规格和优化 LlamaCPP 性能方面面临挑战。
- **GRPO 崭露头角：强化学习竞赛升温**：讨论强调了在强化学习框架中，**GRPO** (Group Relative Policy Optimization) 比 **DPO** (Direct Preference Optimization) 更有效。[强化学习：GRPO vs. DPO](https://arxiv.org/abs/2501.19393)。实验表明，**GRPO** 提高了 **Llama 2 7B** 在 **GSM8K** 上的准确率，这表明它是一种跨模型系列的稳健方法，且 **DeepSeek R1** 的表现优于 PEFT 和指令微调。
- **测试时计算策略：Budget Forcing 进入赛场**：“Budget forcing” 作为一种新型的测试时计算（test-time compute）策略出现，通过延长模型推理时间来鼓励答案复查并提高准确性。[测试时计算策略：Budget Forcing](https://arxiv.org/abs/2501.19393)。该方法利用了一个包含 **1,000 个精心挑选的问题** 的数据集，旨在测试特定标准，促使模型在评估期间增强其推理性能。

**主题 5. 硬件障碍与前景**

- **RTX 5090 在 AI 推理对决中遥遥领先 RTX 4090**：讨论显示，在大语言模型中，**RTX 5090** GPU 的 **token 处理速度**比 **RTX 4090** 快达 **60%**。[RTX 5090 在 AI 任务中超越 RTX 4090](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)。基准测试结果正在被分享，突显了 AI 密集型任务的性能飞跃。
- **AMD RX 7900 XTX 在应对重量级 LLM 时陷入苦战**：用户指出，在运行 **70B** 等大语言模型时，**AMD RX 7900 XTX** GPU 的效率难以与 NVIDIA GPU 匹敌。[AMD RX 7900 XTX 在大型 LLM 任务中表现吃力](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)。社区正在讨论 AMD 硬件在严苛的 LLM 任务中有限的 token 生成速度。
- **GPU 共享内存技巧提升 LM Studio 效率**：讨论强调了在 LM Studio 中利用 GPU 的 **shared memory**（共享内存）来增加 **RAM 利用率**并增强模型性能。[通过共享内存提升 GPU 效率](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)。鼓励用户调整 LM Studio 设置以优化 **GPU offloading** 并有效管理 **VRAM**，尤其是在本地运行大型模型时。

---

# PART 1: 高层级 Discord 摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 框架中的动态量化**：Unsloth 的动态量化可将模型大小减少多达 **80%**，同时保持 [DeepSeek R1](https://unsloth.ai/blog/deepseek-r1) 等模型的准确度。该博客文章概述了使用指定量化技术运行和微调模型的有效方法。
   - 用户在面对 **1.58-bit 量化模型**时遇到挑战，因为动态量化并不总是遵循位规格（bit specification），这引发了对当前设置下 **LlamaCPP** 性能的担忧。
- **VLLM 对 DeepSeek R1 的卸载限制**：**VLLM** 目前缺乏对 **GGUF** 卸载的支持，特别是对于 **DeepSeek V2** 架构，除非应用最近的补丁。
   - 正如最近社区讨论所强调的，这一限制给依赖卸载功能的流水线带来了优化问题。
- **模型训练中的梯度累积**：**Gradient accumulation**（梯度累积）通过允许模型仅根据生成的补全（completions）反馈进行训练来减轻 VRAM 使用，从而比直接在先前输入上训练具有更高的稳定性。
   - 建议使用此方法来保留上下文并防止过拟合，详见 [Unsloth 文档](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device)。
- **测试时计算策略：预算强制**：引入 **budget forcing**（预算强制）来控制测试时计算（test-time compute），通过延长推理时间鼓励模型双重检查答案，旨在提高推理性能。
   - 该策略利用了一个包含 **1,000 个问题**的精选数据集，这些问题旨在满足特定标准，详见最近的研究论坛。
- **用于模型分析的 Klarity 库**：**Klarity** 是一个已发布的开源库，用于分析语言模型输出的熵（entropy），提供详细的 JSON 报告和见解。
   - 鼓励开发者通过 [Klarity GitHub 仓库](https://github.com/klara-research/klarity)进行贡献并提供反馈。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.2.5 补丁更新发布**：**Windsurf 1.2.5 补丁更新**已发布，重点改进和修复了 **Cascade 网络搜索**体验。完整的 [changelog](https://www.codeium.com/changelog) 详细说明了模型调用工具方式的增强。
   - 新的 **Cascade** 功能允许用户通过自动触发、URL 输入以及 **@web** 和 **@docs** 命令进行网络搜索，以获得更多控制权。这些功能每次使用消耗 **1 个 flow action 额度**，并可在 **Windsurf 设置**面板中切换。
- **DeepSeek 模型性能问题**：用户报告了 **DeepSeek 模型**的问题，包括关于无效工具调用的错误消息以及任务期间的上下文丢失，导致在没有产生有效动作的情况下消耗了额度。
   - 这些问题引发了关于提高模型可靠性并确保 **Windsurf** 内高效额度使用的讨论。
- **Windsurf 定价与折扣**：用户对 **Windsurf** 缺乏**学生折扣选项**表示担忧，并质疑该工具与替代方案相比的**价格竞争力**。
   - 用户对当前的**定价结构**表示不满，认为其价值可能与所提供的服务不符。
- **Codeium 扩展程序 vs Windsurf 功能**：官方澄清 **Cascade** 和 **AI flows** 功能在 **JetBrains 插件**中不可用，这使得某些高级功能仅限于 **Windsurf**。
   - 用户参考文档以了解两个平台之间的当前限制和性能差异。
- **Cascade 功能与用户反馈**：用户分享了有效使用 **Cascade** 的策略，例如设置全局规则以阻止不必要的代码修改，以及对 **Claude** 或 **Cascade Base** 使用结构化 Prompt。
   - 反馈强调了对 Cascade “记忆（memories）”功能不遵守既定指令的担忧，这导致了不必要的代码更改。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.73.0 发布，带来增强功能**：**Aider v0.73.0** 的发布引入了对 `o3-mini` 的支持和新的 `--reasoning-effort` 参数（提供 low, medium, high 选项），以及在创建新文件时自动创建父目录的功能。
   - 这些更新旨在改进文件管理，并让用户对推理过程拥有更多控制权，从而增强整体功能。
- **O3 Mini 与 Sonnet：性能对比**：用户报告称 **O3 Mini** 在大型项目中可能会遇到响应时间较慢的问题，有时长达一分钟，而 **Sonnet** 则能以更少的手动上下文添加提供更快的反馈。
   - 尽管用户欣赏 **O3 Mini** 的快速迭代能力，但许多人因速度和效率原因在编程任务中更倾向于使用 **Sonnet**。
- **DeepSeek R1 集成与自托管挑战**：**DeepSeek R1** 与 Aider 的集成在 Aider 的排行榜上展示了顶尖性能，尽管一些用户对其速度表示担忧。
   - 围绕自托管 **LLM** 的讨论揭示了对云端依赖的挫败感，促使像 *George Coles* 这样的用户考虑独立托管解决方案。
- **Windsurf IDE 与内联提示增强**：**Windsurf** 作为一款 Agentic IDE 的引入，为结对编程带来了先进的 AI 能力，通过实时状态感知增强了 VSCode 等工具。
   - 内联提示（Inline prompting）功能允许根据之前的操作自动完成代码更改，为使用 **Aider** 和 **Cursor** 的用户简化了编程体验。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **O3 Mini 面临性能批评**：用户分享了关于 **O3 Mini** 在编程任务中表现的[褒贬不一的评价](https://x.com/hxiao/status/1885700308720091280)，强调了速度和解决方案不完整的问题。
   - **Claude 3.5 Sonnet** 通常被认为是处理大型复杂代码库的首选，能提供更可靠和一致的性能。
- **Cursor IDE 推出新功能**：**Cursor** 最近的[更新](https://www.cursor.com/changelog)包括旨在增强用户体验的检查点恢复（checkpoint restore）功能，但缺乏一致的变更日志引发了关注。
   - 用户对未公开的功能和性能差异表示不满，质疑更新对模型能力的影响。
- **讨论高级 Meta Prompting 技术**：出现了围绕 **Meta-prompting** 技术的讨论，旨在将复杂项目拆解为 LLM 可管理的任务。
   - 分享的资源表明，这些技术可以通过优化 Prompt 结构显著提高用户生产力。

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek AI 在西方基准测试中占据主导地位**：中国的 **DeepSeek AI model** 在各项基准测试中超越了 **OpenAI** 和 **Anthropic** 等西方同行，引发了全球关于 AI 竞争力的讨论。该模型卓越的性能在[最近的测试](https://www.youtube.com/watch?v=yjaoT5-tz0I)中得到了凸显，展示了其强大的能力。
   - 作为回应，美国正在考虑采取立法措施限制与中国 AI 研究的合作，旨在随着 **DeepSeek** 在市场中获得青睐而保护国家创新。
- **AI 监管面临新的立法推动**：参议员 Josh Hawley 最近提出的 **AI 监管立法** 针对 **DeepSeek** 等模型，施加了可能阻碍开源 AI 发展的严厉处罚。该法案强调国家安全，并呼吁对版权法进行彻底改革，正如[这篇文章](https://annas-archive.org/blog/ai-copyright.html)中所讨论的那样。
   - 批评者认为，此类监管可能会扼杀创新并限制可访问性，这呼应了人们对安全与技术进步之间平衡的担忧。
- **LLMs 的数学能力受到审查**：**LLMs** 的数学表现因根本性的不匹配而受到批评，有人将其比作“用叉子刷牙”。**o1-mini** 等模型在数学问题上表现出参差不齐的结果，引发了对其推理有效性的质疑。
   - 社区讨论强调 **o3-mini** 在数学推理方面表现出色，比同类模型能更好地解决复杂谜题，这引发了组织数学推理竞赛的兴趣。
- **自我-他人重叠（SOO）微调增强 AI 诚实度**：一篇关于 **Self-Other Overlap (SOO) fine-tuning** 的论文表明，在不损害任务性能的情况下，各种规模模型的欺骗性 AI 响应都显著减少。该研究详见 [arXiv:2412.16325](https://arxiv.org/abs/2412.16325)，SOO 将 AI 的自我表征与外部感知对齐以促进诚实。
   - 实验表明，**Mistral-7B** 中的欺骗性响应减少到 **17.2%**，表明 SOO 在强化学习场景中的有效性，并促进了更可靠的 AI 交互。
- **OpenEuroLLM 发布专注于欧盟的语言模型**：**OpenEuroLLM** 计划已经启动，旨在开发为所有欧盟语言量身定制的开源大语言模型，并获得了 [European Commission](https://x.com/EU_Commission/status/1886427917762150427) 颁发的首个代表卓越的 STEP Seal 标志。
   - 在欧洲机构联盟的支持下，该项目旨在为整个欧盟的各种应用创建合规且可持续的高质量 AI 技术，增强区域 AI 能力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek R1 面临蒸馏限制**：用户报告了对 **DeepSeek R1** 模型参数规模的困惑，争论其到底是 **14B 还是 7B**。
   - 许多人对该模型的**自动补全（auto-completion）**和**调试（debugging）**能力感到沮丧，特别是在编程任务方面。
- **AI 驱动的直播聊天室初具规模**：一位用户详细介绍了在 LM Studio 中创建**多 Agent 直播聊天室**的过程，其特点是各种 AI 人格进行实时互动。
   - 计划包括将该系统集成到 Twitch 和 YouTube 直播流中，以展示 AI 在动态环境中的潜力。
- **通过共享内存提升 GPU 效率**：讨论强调了在 GPU 上使用**共享内存**以实现更高的 **RAM 利用率**，从而提高模型性能。
   - 鼓励用户调整 LM Studio 中的设置，以优化 **GPU offloading** 并管理大型模型的 **VRAM**。
- **RTX 5090 在 AI 任务中超越 RTX 4090**：交流透露，在处理大型模型时，**RTX 5090** 的 Token 处理速度比 RTX 4090 快 **60%**。
   - 基准测试结果分享自 [GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)。
- **AMD RX 7900 XTX 在运行大型 LLM 时表现挣扎**：用户指出，在运行 **70B** 等大语言模型时，**AMD RX 7900 XTX** 的效率不如 NVIDIA GPU。
   - 社区讨论了 AMD GPU 在 **LLM 任务**中有限的 **Token 生成速度**。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 安排 o3-mini AMA**：一场由 **Sam Altman**、**Mark Chen** 和其他核心人物参加的 **AMA** 定于 **PST 时间下午 2 点**举行，旨在回答有关 **OpenAI o3-mini** 及其即将推出的功能的问题。用户可以在 [Reddit 此处](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/) 提交问题。
   - 此次 AMA 旨在提供对 **OpenAI** 未来发展的见解，并收集社区对 **o3-mini** 模型的反馈。
- **OpenAI 发布 Deep Research Agent**：OpenAI 推出了一款全新的 **Deep Research Agent**，能够自主从多个在线平台获取、分析和综合信息，在几分钟内生成全面的报告。详细信息请见 [此处](https://openai.com/index/introducing-deep-research/)。
   - 该工具预计将通过显著减少数据汇编和分析所需的时间来简化研究流程。
- **DeepSeek R1 性能问题**：据 [Cisco](https://www.pcmag.com/news/deepseek-fails-every-safety-test-thrown-at-it-by-researchers) 强调，有用户报告 **DeepSeek R1** 的**攻击成功率达到 100%**，未能通过所有安全测试，且由于频繁的服务器问题导致访问困难。
   - **DeepSeek** 无法阻止有害提示词的情况引发了对其在现实应用中可靠性和安全性的担忧。
- **OpenAI 为模型设置上下文 Token 限制**：OpenAI 的模型执行严格的上下文限制，**Plus 用户**上限为 **32k tokens**，**Pro 用户**上限为 **128k tokens**，这限制了它们处理大规模知识库的能力。
   - 讨论中提到了利用 **embeddings** 和**向量数据库**作为替代方案，以比将数据切分为 chunks 更有效的方式管理大型数据集。
- **AI 模型对比：GPT-4 vs DeepSeek R1**：对话对比了 **OpenAI 的 GPT-4** 和 **DeepSeek R1**，指出了在编程辅助和推理任务等能力上的差异。用户观察到 **GPT-4** 在某些 **DeepSeek R1** 表现不足的领域表现出色。
   - 成员们辩论了包括 **O1**、**o3-mini** 和 **Gemini** 在内的模型优缺点，根据功能和在各种应用中的可用性对其进行评估。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek 和 Psyche AI 的进展**：参与者强调了 **DeepSeek 的进步**，重点介绍了 **Psyche AI** 如何在其技术栈中利用 Rust，同时集成现有的 Python 模块以保持 p2p 网络功能。
   - 针对在强化学习中实现多步响应提出了担忧，重点关注效率以及在扩展这些功能时固有的挑战。
- **OpenAI 在 DeepSeek 之后的策略**：**OpenAI** 在 DeepSeek 出现后的立场受到了审视，特别是 **Sam Altman** 关于处于“历史错误一边”的言论，鉴于 OpenAI 此前不愿开源模型的态度，这引发了对其真实性的质疑。
   - 成员们强调，OpenAI 的行动需要与其言论保持一致才具有公信力，并指出其承诺与实际执行之间存在差距。
- **AI 中的法律和版权考量**：讨论集中在 AI 开发的**法律影响**上，特别是关于**版权问题**，成员们辩论了保护知识产权与促进 AI 创新之间的平衡。
   - 一名法学院学生询问了如何将以法律为中心的对话与技术讨论相结合，强调了可能影响未来 AI 研发的潜在监管规定。
- **模型训练技术的进步**：社区探讨了 **Deep Gradient Compression**，这是一种在分布式训练中将通信带宽降低 **99.9%** 且不损失准确性的方法，详见相关 [论文](https://arxiv.org/abs/1712.01887)。
   - 此外还讨论了 **Stanford 的 Simple Test-Time Scaling**，该技术在竞赛数学题上的推理性能提升了高达 **27%**，且所有资源均已开源。
- **新 AI 工具与社区贡献**：**Relign** 推出了开发者悬赏任务，旨在构建一个专为推理引擎量身定制的 [开源 RL 库](https://link.to.repo)，邀请社区贡献力量。
   - 此外，成员们分享了关于用于研究探索的 [Scite 平台](https://scite.ai/) 的见解，并鼓励参与社区驱动的 AI 模型测试计划。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 的 Deep Research 增强功能**：OpenAI 随 O3 模型推出了 **Deep Research**，允许用户细化研究查询并通过侧边栏查看推理进度。初步反馈指出其在综合信息方面的能力，尽管在来源分析方面仍存在一些局限性。
   - 此外，**OpenAI 的 O3** 通过强化学习（RL）技术持续改进，同时其 **Deep Research** 工具也得到了增强，突显了其模型训练中对 RL 方法论的高度关注。
- **软银向 OpenAI 承诺 30 亿美元投资**：软银（SoftBank）宣布每年向 OpenAI 产品投资 **30 亿美元**，并在日本成立一家专注于 **Crystal Intelligence** 模型的合资企业。该伙伴关系旨在将 OpenAI 的技术整合到软银子公司中，为日本企业推进 AI 解决方案。
   - **Crystal Intelligence** 旨在自主分析和优化遗留代码，并计划在两年内引入 AGI，体现了孙正义将 AI 视为**超级智慧（Super Wisdom）**的愿景。
- **共和党 AI 立法针对中国技术**：一项由共和党（GOP）发起的法案提议禁止从中国进口 AI 技术，包括来自 [DeepSeek](https://x.com/opensauceAI/status/1885483641649979639) 等平台的模型权重，违者最高可判处 **20 年监禁**。
   - 该立法还将向指定关注实体出口 AI 定为犯罪，将发布 **Llama 4** 等产品的行为等同于类似的严厉处罚，引发了对其对开源 AI 发展影响的担忧。
- **强化学习：GRPO vs. DPO**：讨论强调了在强化学习框架中，特别是在 **RLVR** 应用背景下，**GRPO** 优于 **DPO** 的有效性。成员们认为，虽然可以使用 **DPO**，但其效果可能不如 **GRPO**。
   - 此外，研究结果表明 **GRPO** 对 **Llama 2 7B** 模型产生了积极影响，在 **GSM8K** 基准测试中实现了显著的准确率提升，展示了该方法在不同模型系列中的稳健性。
- **DeepSeek AI 的 R1 模型亮相**：**DeepSeek AI** 于 1 月 20 日发布了其旗舰 **R1** 模型，强调通过额外数据进行扩展训练以增强推理能力。社区对推理模型领域的这一进展表现出极大热情。
   - **R1** 模型简单的训练方法（在训练后周期的早期优先考虑排序）因其简洁性和有效性而受到赞誉，引发了对推理 **LM** 未来发展的期待。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 推出 Deep Research Agent**：OpenAI 推出了 **Deep Research**，这是一个针对[网页浏览](https://x.com/openai/status/1886219087627612504)和复杂推理进行优化的自主 **Agent**，能够在几分钟内综合来自不同来源的大量报告。
   - 初步反馈强调了其作为强大电子商务工具的实用性，尽管一些用户报告了[输出质量的局限性](https://x.com/ocolegro/status/1886491097716961635?s=46)。
- **推理增强生成 (ReAG) 亮相**：**推理增强生成 (ReAG)** 被引入以增强传统的检索增强生成（RAG），通过消除检索步骤并将原始材料直接输入 **LLM** 进行综合。
   - 初步反应注意到了其潜在的有效性，同时也对其可扩展性和[预处理文档](https://www.superagent.sh/blog/reag-reasoning-augmented-generation)的必要性提出了质疑。
- **AI Engineer Summit 门票火爆**：[AI Engineer Summit 的门票](https://www.latent.space/p/2025-summit)和赞助正在快速售罄，该活动定于 **2 月 20 日至 22 日在纽约市**举行。
   - [新的峰会网站](https://www.ai.engineer/summit/2025)提供了演讲者和日程安排的实时更新。
- **Karina Nguyen 将为 AI 峰会收官**：**Karina Nguyen** 将在 AI Engineer Summit 上发表闭幕主题演讲，展示她在 **Notion**、**Square** 和 **Anthropic** 任职期间的经验。
   - 她的贡献涵盖了 **Claude 1, 2, 和 3** 的开发，凸显了她对 AI 进步的影响。
- **Deepseek API 面临可靠性问题**：成员们对 **Deepseek API 的可靠性**表示担忧，强调了[访问问题](https://x.com/pelaseyed/status/1886448015533089248)和性能缺陷。
   - 观点认为该 API 的**托管和功能落后于预期**，引发了关于潜在改进的讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **获得功能性语言模型的概率**：EleutherAI 的一项研究计算出，通过随机猜测权重来获得一个功能性语言模型的概率大约为 **1 后面跟着 3.6 亿个零分之一**，突显了其中涉及的巨大**复杂度**。
   - 团队分享了他们的 [basin-volume](https://github.com/EleutherAI/basin-volume) GitHub 仓库和一篇[研究论文](https://arxiv.org/abs/2501.18812)，以探索**网络复杂度**及其对模型对齐的影响。
- **R1 在 SmolLM2 上的复现失败**：研究人员在 **SmolLM2 135M** 上测试 R1 结果时遇到了**复现失败**，观察到与在真实数据上训练的模型相比，其自动解释 (autointerp) 得分更低，重构误差更高。
   - 这一差异引发了对原始论文有效性的质疑，正如在围绕 [Sparse Autoencoders](https://arxiv.org/abs/2501.17727) 社区发现的讨论中所指出的那样。
- **DeepSeek 的审查问题**：**DeepSeek** 对**天安门广场**等敏感话题的反应因提示词语言而异，表明其设计中集成了潜在的**偏见**。
   - 用户建议了绕过这些审查机制的方法，并引用了相关文献中讨论的 [AI safety training](https://arxiv.org/abs/2310.02446) 漏洞。
- **DRAW 架构增强图像生成**：**DRAW** 网络架构引入了一种新型的空间注意力机制，模拟人类的视觉注视 (foveation)，显著提高了在 **MNIST** 和 **Street View House Numbers** 等数据集上的图像生成效果。
   - 来自 [DRAW 论文](https://arxiv.org/abs/1502.04623) 的性能指标表明，生成的图像与真实数据无法区分，展示了增强的**生成能力**。
- **NeoX 性能指标与挑战**：一名成员报告称，在 **A100s** 上运行 1.3B 参数模型时达到了 **每秒 10-11K tokens**，而 OLMo2 论文中报告的则是 **50K+ tokens**。
   - 讨论了**融合标志 (fusion flags)** 的问题以及 [gpt-neox configurations](https://github.com/EleutherAI/gpt-neox/blob/main/configs/1-3B-transformer-engine.yml) 中的差异，突显了扩展 **Transformer Engine** 加速时面临的挑战。



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **远程 MCP 工具需求激增**：成员们强调了 **MCP 工具** 对[远程能力](https://myhosted.mcp.server.com)的需求，并指出大多数现有解决方案都集中在本地实现上。
   - 提出了对可扩展性和可用性的担忧，并建议探索替代方案以增强 MCP 功能。
- **Superinterface 产品明确 AI 基础设施重点**：**Superinterface** 的联合创始人详细介绍了他们专注于提供 AI Agent 基础设施即服务，以区别于开源替代方案。
   - 该产品旨在将 AI 能力集成到用户产品中，突显了基础设施需求中涉及的复杂性。
- **Goose 自动化 GitHub 任务**：一段 [YouTube 视频](https://youtube.com/shorts/TbmQDv3SQOE)展示了开源 AI Agent **Goose** 如何通过与任何 MCP 服务器集成来自动化任务。
   - 该演示突显了 Goose 处理 GitHub 交互的能力，强调了 MCP 的创新用途。
- **Supergateway v2 增强 MCP 服务器可访问性**：**Supergateway v2** 现在支持通过 **ngrok** 隧道远程运行任何 MCP 服务器，简化了服务器的设置和访问。
   - 鼓励社区成员寻求帮助，反映了改进 MCP 服务器可用性的协作努力。
- **Litellm Proxy 中的负载均衡技术**：讨论涵盖了使用 **Litellm proxy** 进行负载均衡的方法，包括配置权重和管理每分钟请求数。
   - 这些策略旨在工作流中高效管理多个 AI 模型端点。



---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 性能问题影响用户**：多名用户报告 **Bolt** 响应缓慢且频繁出现**错误消息**，导致操作中断，需要频繁刷新页面或清除 cookie。
   - 反复出现的问题表明可能存在**服务器端（server-side）**问题或**本地存储管理（local storage management）**挑战，因为用户试图通过清除浏览器数据来恢复访问。
- **Supabase 相比 Firebase 更受青睐**：在一场激烈的辩论中，许多用户因 **Supabase** 的直接集成能力和用户友好界面而更青睐它，而非 **Firebase**。
   - 然而，一些参与者对已经深入其生态系统的 **Firebase** 表示赞赏，凸显了社区中偏好的分歧。
- **Supabase 服务连接不稳定**：用户在进行更改后遇到 **Supabase** 断连，需要**重新连接**或重新加载项目以恢复功能。
   - 一名用户通过重新加载项目解决了连接问题，表明断连可能源于最近的**前端修改（front-end modifications）**。
- **Voiceflow 聊天机器人中 Calendly 的 Iframe 错误**：一名用户在 **Voiceflow** 聊天机器人中集成 **Calendly** 时遇到 **iframe** 错误，导致显示问题。
   - 在咨询了 **Voiceflow** 和 **Calendly** 的代表后，确定这是一个 **Bolt** 问题，令该用户感到非常沮丧。
- **持续的用户身份验证挑战**：用户报告了**身份验证（authentication）问题**，包括无法登录以及在不同浏览器中遇到相同的错误。
   - 清除**本地存储（local storage）**等建议的解决方法对某些人无效，指向身份验证系统内部的潜在问题。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.8.0 在 Intel Mac 上崩溃**：用户报告 **GPT4All v3.8.0** 在现代 Intel macOS 机器上崩溃，暗示该版本对这些系统可能是 **DOA**（出厂即失效）。
   - 正在根据用户的系统规格形成一个工作假设，以识别受影响的配置，因为多名用户遇到了类似问题。
- **量化级别影响 GPT4All 性能**：**量化级别（Quantization levels）**显著影响 **GPT4All** 的性能，较低的量化会导致**质量下降（quality degradation）**。
   - 鼓励用户平衡量化设置，在不使硬件过载的情况下保持输出质量。
- **AI 模型数据收集中的隐私担忧**：关于**数据收集信任**的辩论已经兴起，对比了**西方**和**中国**的数据实践，用户表达了不同程度的**担忧和怀疑**。
   - 参与者争论不同国家在数据收集方面存在的**双重标准**。
- **在 GPT4All 中集成 MathJax 以支持 LaTeX**：用户正在探索在 **GPT4All** 中集成 **MathJax** 以支持 **LaTeX**，强调与 LaTeX 结构的兼容性。
   - 讨论集中在解析 LaTeX 内容和提取数学表达式，以改进 **LLM** 的输出表现。
- **开发用于 NSFW 故事生成的本地 LLM**：一名用户正在寻找能够离线生成 **NSFW 故事**的本地 **LLM**，类似于现有的在线工具，但不使用 **llama** 或 **DeepSeek**。
   - 该用户指定了他们的系统能力和要求，包括对**德语** **LLM** 的偏好。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 计划发布 API**：用户询问了即将发布的 **NotebookLM API release**，对扩展功能表示热切期待。
   - 提到 **NotebookLM** 的 **output token limit** 低于 **Gemini**，但具体细节尚未披露。
- **NotebookLM Plus 功能在 Google Workspace 推出**：一位用户升级到 **Google Workspace Standard** 后，观察到 **NotebookLM** 顶部栏增加了“**Analytics**”，表明已获得 **NotebookLM Plus** 的访问权限。
   - 他们指出尽管界面外观相似，但使用限制（usage limits）有所不同，并分享了截图以供参考。
- **将完整教程集成到 NotebookLM**：一名成员建议将整个教程网站（如 [W3Schools JavaScript](https://www.w3schools.com/js/)）整合进 **NotebookLM**，以加强对 JS 面试的准备。
   - 另一名成员提到现有的 Chrome 扩展程序可以辅助将网页导入 **NotebookLM**。
- **UI 更新后音频自定义功能缺失**：用户报告在最近的 UI 更新后，**NotebookLM** 丢失了 **audio customization** 功能。
   - 建议包括探索 [Illuminate](https://illuminate.google.com/home) 以获取相关功能，并希望某些功能将来能迁移到 **NotebookLM**。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 和 MAX 简化解决方案**：一名成员强调了 **Mojo** 和 **MAX** 在解决当前工程挑战方面的有效性，强调了它们作为综合解决方案的潜力。
   - 讨论强调了在现有工作流中有效实施这些解决方案需要投入大量精力。
- **减少 Mojo 中 Swift 的复杂性**：有人担心 **Mojo** 会继承 **Swift** 的复杂性，社区提倡更清晰的开发路径以确保稳定性。
   - 成员们强调了仔细评估权衡（tradeoff）的重要性，以防止仓促推进可能损害 **Mojo** 可靠性的情况。
- **Ollama 性能超过 MAX**：观察到在相同机器上 **Ollama** 的运行速度比 **MAX** 快，尽管最初的指标显示 **MAX** 性能较慢。
   - 目前的开发重点是优化 **MAX** 基于 CPU 的 serving 能力，以提升整体性能。
- **增强 Mojo 的类型系统**：用户询问在将参数作为 concrete types 传递时，如何访问 **Mojo** 类型系统中的特定 struct 字段。
   - 回复指出有效利用 **Mojo** 的类型功能存在学习曲线，表明社区正在进行持续的教育工作。
- **MAX Serving 基础设施优化**：**MAX** serving 基础设施使用 `huggingface_hub` 下载和缓存模型权重，这与 **Ollama** 的方法有所不同。
   - 讨论揭示了可以通过修改 `--weight-path=` 参数来防止重复下载，尽管管理 **Ollama** 的本地缓存仍然很复杂。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **16 节点上的 GRPO 部署**：一名成员通过调整 [multinode PR](https://github.com/pytorch/torchtune/pull/2324)，成功在 **16 个节点**上部署了 **GRPO**，并期待即将进行的奖励曲线验证。
   - 他们幽默地评论道，身处一家**资金充足的公司**在进行此类部署时具有显著优势。
- **Torchtune 多节点支持的最终审批**：有人请求对 **Torchtune** 中的 [multinode support PR](https://github.com/pytorch/torchtune/pull/2301) 进行最终审批，并强调了基于用户需求的必要性。
   - 讨论中提到了关于 API 参数 `offload_ops_to_cpu` 的潜在担忧，建议可能需要额外的审查。
- **DPO Recipe 中的 Seed 不一致性**：**Seed** 在 **LoRA** 微调中有效，但在 **LoRA DPO** 中失效，[issue #2335](https://github.com/pytorch/torchtune/issues/2335) 正在调查 **sampler** 行为的不一致性。
   - 已记录多个与 **seed 管理**相关的问题，重点关注数据集中 `seed=0` 和 `seed=null` 的影响。
- **LLM 数据增强全面综述**：一份综述详细介绍了大型预训练语言模型 (LLM) 如何从大规模训练数据集中获益，解决了 **overfitting** 问题，并利用[独特的 prompt templates](https://arxiv.org/abs/2501.18845)增强了数据生成。
   - 它还涵盖了近期整合外部知识的**基于检索的技术**，使 LLM 能够生成 **grounded-truth data**。
- **R1-V 模型增强 VLM 的计数能力**：**R1-V** 利用带有 **verifiable rewards** 的 **reinforcement learning** 来提升视觉语言模型 (VLM) 的计数能力，其中一个 **2B model** 在 **100 个训练步数**内的表现优于 **72B model**，成本低于 **$3**。
   - 该模型将**完全开源**，鼓励社区关注未来的更新。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **关于 LLM 自我改进的即将举行的讲座**：[Jason Weston](https://www.youtube.com/live/_MNlLhU33H0) 将于今天 **PST 时间下午 4:00** 介绍 **LLM 中的自我改进方法**，重点关注 **Iterative DPO** 和 **Meta-Rewarding LLMs** 等技术。
   - 参与者可以在[此处](https://www.youtube.com/live/_MNlLhU33H0)观看直播，Jason 将探讨增强 LLM 推理、数学和创意任务的方法。
- **LLM 中的 Iterative DPO 与 Meta-Rewarding**：**Iterative DPO** 和 **Meta-Rewarding LLMs** 作为近期进展被讨论，并附带了 [Iterative DPO](https://arxiv.org/abs/2312.16682) 和 [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020) 论文的链接。
   - 这些方法旨在通过改进强化学习技术来提升 LLM 在各种任务中的性能。
- **DeepSeek R1 超越 PEFT**：**DeepSeek R1** 证明了结合组相对策略优化 (GRPO) 的强化学习优于 **PEFT** 和指令微调。
   - 这一转变表明，由于 **DeepSeek R1** 增强的有效性，可能会逐渐脱离传统的提示方法。
- **MOOC 测验和证书更新**：测验现已在[课程网站](https://llmagents-learning.org/sp25)的教学大纲部分上线，为了防止收件箱混乱，**不会发送邮件提醒**。
   - 证书状态正在更新中，并保证提交内容将*很快*得到处理，尽管一些成员报告了延迟。
- **黑客松结果即将公布**：成员们正期待**黑客松结果**，结果已私下通知，预计下周公开发布。
   - 这是在广泛参与 MOOC 的研究和项目轨道之后进行的，突显了活跃的社区参与。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **NVDEC 解码复杂性揭秘**：使用 **NVDEC** 解码视频面临与文件格式相关的挑战以及对 **cuvid binaries** 的必要性，正如 [FFmpeg/libavcodec/nvdec.c](https://github.com/FFmpeg/FFmpeg/blob/c6194b50b1b4001db23e8debab4ac4444e794f90/libavcodec/nvdec.c#L350) 中所强调的。
   - 冗长的 **libavcodec** 实现包含高层抽象，简化这些抽象可能有助于提高效率。
- **WebGPU Autogen 接近完成**：一位成员报告 **WebGPU autogen** 已接近完成，仅需少量简化，且测试在 **Ubuntu** 和 **Mac** 平台上均已通过。
   - 他们强调在未安装 **dawn binaries** 的情况下需要提供相关指令。
- **Linux 发行版中的 Clang 与 GCC 之争**：辩论强调，虽然 **Apple** 和 **Google** 等平台青睐 **clang**，但 **gcc** 在主要的 Linux 发行版中仍然盛行。
   - 这引发了关于发行版是否应转向 **clang** 以获得更好优化的讨论。
- **HCQ 执行范式增强多 GPU**：**HCQ-like execution** 被认为是理解 **multi-GPU execution** 的关键步骤，并可能支持 **CPU implementations**。
   - 优化调度器以在 CPU 和 GPU 之间高效分配任务可能会带来性能提升。
- **CPU P2P 传输机制探索**：讨论推测 **CPU p2p** 传输可能涉及释放内存块上的锁以便驱逐到 **L3/DRAM**，并考虑了 **D2C transfers** 的效率。
   - 针对复杂多插槽传输过程中的执行局部性提出了**性能担忧**。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 试用密钥重置时间**：一位成员询问 **Cohere trial key** 何时重置——是**生成后 30 天**还是每个月初。这种不确定性影响了开发者规划评估期的方式。
   - 需要进一步明确，因为试用密钥旨在用于评估，而非长期免费使用。
- **Command-R+ 模型性能备受赞誉**：用户称赞 **Command-R+ model** 始终能满足他们的需求，一位用户提到，尽管自己不是高级用户，该模型仍不断给他们带来**惊喜**。
   - 这种持续的性能表现表明了其在实际应用中的可靠性和有效性。
- **Embed API v2.0 HTTP 422 错误**：一位成员在使用带有特定 cURL 命令的 **Embed API v2.0** 时遇到“**HTTP 422 Unprocessable Entity**”错误，引发了对长篇文章预处理需求的关注。
   - 建议包括验证是否包含 **API key**，因为其他人在类似条件下报告了成功的请求。
- **持续的账户自动登出问题**：多名用户报告了**自动登出**问题，迫使他们反复登录，中断了平台内的流程。
   - 这一反复出现的问题突显了一个显著的用户体验缺陷，需要解决以确保无缝访问。
- **Command R 的日语翻译不一致**：**Command R** 和 **Command R+** 在日语翻译方面表现出不一致的结果，部分翻译完全失败。
   - 建议用户携带具体案例联系支持部门以协助多语言团队，或利用日语资源以获得更好的上下文。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Deepseek 压倒 OpenAI**：一位成员指出 **Deepseek** 与 **OpenAI** 之间出现了明显的赢家，并分享了一段[令人惊讶的叙述](https://bit.ly/deepseek-audio-narrative)，展示了其竞争实力。
   - 这一讨论引发了人们对这些工具相对性能的兴趣，强调了 **Deepseek** 正在显现的优势。
- **LlamaReport 自动化报告生成**：分享了 **LlamaReport** 的早期 Beta 版视频，展示了其在 2025 年进行**报告生成**的潜力。点击[此处](https://t.co/pYx3O5BpYe)观看。
   - 该项目旨在简化报告流程，为用户提供高效的解决方案。
- **SciAgents 增强科学发现**：介绍了 **SciAgents**，这是一个利用多 Agent 工作流和本体图谱（ontological graphs）的自动化科学发现系统。了解更多请点击[此处](https://t.co/9pBYvN4IQh)。
   - 该项目展示了协作分析如何驱动科学研究的创新。
- **AI 驱动的 PDF 转 PPT 工具**：一个开源 Web 应用支持使用 **LlamaParse** 将 **PDF 文档**转换为动态的 PowerPoint 演示文稿。点击[此处](https://t.co/XRgwUrlvA3)探索。
   - 该应用简化了演示文稿的制作，为用户实现了工作流自动化。
- **DocumentContextExtractor 提升 RAG 准确率**：**DocumentContextExtractor** 因提升 **Retrieval-Augmented Generation (RAG)** 的准确率而受到关注，该项目由 **AnthropicAI** 和 **LlamaIndex** 共同贡献。查看讨论串请点击[此处](https://t.co/qoVrgd0ddy)。
   - 这强调了社区在改进 AI 上下文理解方面的持续贡献。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DeepSeek 反映了对 AI 的希望与恐惧**：文章讨论了 **DeepSeek** 如何作为一个“教科书式的力量对象”，揭示了更多关于我们对 AI 的欲望和担忧，而非技术本身，详见[此处](https://www.dbreunig.com/2025/01/31/deepseek-as-a-power-object.html)。
   - *关于 DeepSeek 的每一种热评都反映了个人对 AI 影响的具体希望或恐惧*。
- **SAEs 在引导 LLMs 方面面临重大挑战**：一位成员对 **SAEs** 在可预测地引导 LLMs 方面的长期可行性表示失望，并引用了最近的一场[讨论](https://x.com/kzslider/status/1885666578429055096)。
   - 另一位成员强调了近期问题的严重性，称：*“天哪，一天之内发生了‘三重命案’。SAEs 最近真的遭受了沉重打击。”*
- **DSPy 2.6 弃用 Typed Predictors**：成员们澄清 **typed predictors** 已被弃用；在 **DSPy 2.6** 中，普通的 predictors 已足以满足功能需求。
   - 会议强调，在当前版本中已经**不再有 typed predictor 这种东西了**。
- **在 DSPy 中将 Chain-of-Thought 与 R1 模型结合**：一位成员表示有兴趣将 **DSPy Chain-of-Thought** 与 **R1 模型**结合进行微调，以共同角逐 **Konwinski Prize**。
   - 他们还邀请其他人加入关于这一倡议的讨论和协作努力。
- **DSPy 中的流式输出问题**：一位用户分享了在使用 **dspy.streamify** 增量产生输出时遇到的困难，收到了 **ModelResponseStream** 对象而非预期的值。
   - 他们在代码中实现了条件判断来处理输出类型，并寻求进一步的改进建议。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **OpenEuroLLM 为欧盟语言首次亮相**：[OpenEuroLLM](https://openeurollm.eu/) 已作为首个涵盖所有欧盟语言的开源大语言模型 (LLM) 系列发布，优先考虑符合欧盟法规。
   - 该模型在欧洲监管框架内开发，确保符合**欧洲价值观**，同时保持技术卓越。
- **R1-Llama 表现超出预期**：对 **R1-Llama-70B** 的初步评估显示，在解决[奥林匹克级别的数学和编程问题](https://x.com/JJitsev/status/1886210118594760744)方面，它与 **o1-mini** 和原始 **R1** 模型旗鼓相当甚至有所超越。
   - 这些结果突显了领先模型中潜在的泛化缺陷，引发了社区内的讨论。
- **DeepSeek 的规格受到关注**：**DeepSeek v3/R1** 模型具有 **37B 激活参数**，并采用**混合专家 (MoE)** 方法，与 **Llama 3** 模型的稠密架构相比，提高了计算效率。
   - DeepSeek 团队实施了广泛的优化以支持 MoE 策略，从而实现了更高效的资源性能。
- **对性能比较的兴趣**：一位社区成员表达了对测试一款据称比 **HunYuan** 更快的新模型的热情。
   - 这种情绪强调了社区对当前 AI 模型性能基准测试的关注。
- **欧盟委员会强调 AI 的欧洲根源**：[欧盟委员会 (EU_Commission) 的一条推文](https://x.com/EU_Commission/status/1886427917762150427)宣布，OpenEuroLLM 已获得首个卓越 STEP 标志，旨在团结欧盟的初创公司和研究实验室。
   - 该倡议强调保护**语言和文化多样性**，并在欧洲超级计算机上开发 AI。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **微调的挫败感**：一位成员对**微调推理模型 (reasoning models)** 表示困惑，幽默地承认不知道从哪里开始。
   - 他们评论道 *Lol*，表明在该领域需要指导。
- **GRPO Colab Notebook 发布**：一位成员分享了一个 [用于 GRPO 的 Colab notebook](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)，为对该主题感兴趣的人提供了资源。
   - 该 notebook 为寻求进一步了解 **GRPO** 的成员提供了一个起点。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **o3-mini 的 Interpreter 集成**：一位成员询问 **o3-mini** 是否可以同时在 **01** 和 **interpreter** 中使用，突显了潜在的集成问题。
   - 这些担忧强调了需要澄清 **o3-mini** 与 **Open Interpreter** 的兼容性。
- **期待 Interpreter 更新**：一位成员询问了即将到来的 **Open Interpreter** 更改的性质，试图了解这些更改是**微小的**还是**重大的**。
   - 他们的询问反映了社区对计划更新的范围和影响的好奇。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **掌握 Cursor AI 以提高生产力**：本周二**东部时间下午 5 点**，参加关于 [**Cursor AI**](https://lu.ma/wqphpn4d) 的线上线下混合活动，特邀演讲嘉宾 Arnold（一位 **10X CTO**）将讨论提高编码速度和质量的最佳实践。
   - 参与者可以亲身前往 Builder's Club 或通过 Zoom 虚拟参加，注册链接将在报名后提供。
- **《王者荣耀》市场的高价值交易**：**Honor of Kings** 市场今天出现了一笔高价收购，**小蛇糕**以 **486** 的价格售出。
   - 鼓励用户使用提供的市场代码 **-<344IRCIX>-** 和密码 **[[S8fRXNgQyhysJ9H8tuSvSSdVkdalSFE]]** 在市场中进行交易，购买或出售物品。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Lumigator 实时演示简化模型测试**：参加 [Lumigator 实时演示](https://discord.com/events/1089876418936180786/1331996842568843315) 了解安装和入门，运行你的第一次 **模型评估 (model evaluation)**。
   - 该活动将引导参与者完成 **有效的模型性能测试** 的关键设置步骤。
- **Firefox AI 平台首次推出离线 ML 任务**：[Firefox AI 平台](https://discord.com/channels/1089876418936180786/1329145280838500475) 现已上线，使开发者能够在网络扩展中利用 **离线机器学习任务 (offline machine learning tasks)**。
   - 这个新平台为直接在用户友好环境中提升 **机器学习能力 (machine learning capabilities)** 开辟了道路。
- **Blueprints 更新增强开源配方**：查看 [Blueprints 更新](https://discord.com/channels/1089876418936180786/1230938514955436242/1332449189715509279) 获取旨在增强开源项目的新配方。
   - 该计划为开发者提供了 **创建有效软件解决方案** 的必备工具。
- **Builders Demo Day 演讲在 YouTube 首次亮相**：[Builders Demo Day 演讲](https://www.youtube.com/playlist?list=PLgjjGlfBflISGQaljPUkxEWqDYBgfC7TZ) 已在 Mozilla Developers 的 YouTube 频道发布，展示了开发者社区的创新。
   - 这些演讲提供了一个与 **前沿开发项目** 和想法互动的激动人心的机会。
- **社区宣布关键更新**：成员可以找到关于社区内最新进展的 [重要新闻](https://discord.com/channels/1089876418936180786/1262961704602570832/1333936885566799912)。
   - 随时了解影响社区倡议和协作的关键讨论。

---

**HuggingFace Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1334983432572370945)** (1121 messages🔥🔥🔥): 

> `Unsloth Framework, DeepSeek R1, Batch Inference, Legal Considerations for AI Training, LLM Performance` 

- **了解 Unsloth 及其功能**：Unsloth 主要是一个微调框架，旨在快速测试模型，但它并不针对生产推理，后者通过 vllm 等系统会更好。
   - Unsloth 推理可用于比传统 Transformer 推理更有效地验证微调结果，尽管它不支持批处理 (batch processing)。
- **训练数据质量的挑战**：参与者讨论了需要经过策划且平衡的数据集，以提高正在微调的模型性能，特别是在避免与特定内容类型相关的偏差方面。
   - 参与者强调了清理和组织数据的重要性，以确保有效的模型训练并防止过拟合。
- **AI 训练中的法律考量**：对话涉及了使用受版权保护的数据进行模型训练的法律后果，包括不同的国际法律以及不合规可能带来的潜在影响。
   - 鉴于 AI 法规的不断演变，建议咨询法律资源以了解使用文本数据进行训练的界限。
- **不同 LLM 的性能**：审查了 DeepSeek R1 等模型的性能和效率，并对各种模型的速度和能力发表了评论，包括本地设置中潜在的运行开销。
   - 参与者指出需要更好的计算资源来有效处理模型，特别是那些需要大量 GPU 显存的模型。
- **社区资源与协作**：用户分享了资源链接，包括 GitHub 仓库和 Colab 笔记本，旨在帮助新用户应对微调和利用 LLM 架构的复杂性。
   - 社区表达了互相帮助开展项目以及在处理数据任务和提高模型性能方面寻求协作的意愿。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/shxf0072/status/1886085377146180091">来自 Joey (e/λ) (@shxf0072) 的推文</a>：成功在免费的 Colab 上使用 Unsloth 搞定了 GRPO，虽然慢得痛苦但确实可行 :p https://colab.research.google.com/drive/1P7frB3fjMv6vjSINqiydAf6gnMab2TiL?usp=sharing 引用 Joey (e/λ) (@shxf0072) 的 OOMxiety</li><li><a href="https://x.com/vllm_project/status/1885837174588989695">来自 vLLM (@vllm_project) 的推文</a>：我们发布了针对 @deepseek_ai 模型的第一批增强功能，首先是 MLA 和 cutlass fp8 内核。与 v0.7.0 相比，我们提供了约 3 倍的生成吞吐量，以及约 10 倍的 token 内存容量...</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - ggml-org 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/examples/examples_index.html">示例 —— vLLM</a>：未找到描述</li><li><a href="https://docs.mathjax.org/en/latest/basic/mathematics.html">为 MathJax 编写数学公式 — MathJax 3.2 文档</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/mistral-small-24b-2501-all-versions-679fe9a4722f40d61cfe627c">Mistral-Small-24B-2501 (所有版本) - Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B">ContactDoctor/Bio-Medical-Llama-3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://lmsys.org/blog/2024-02-05-compressed-fsm/">使用压缩有限状态机为本地 LLM 实现快速 JSON 解码 | LMSYS Org</a>：<p>约束 LLM 始终生成符合特定 Schema 的有效 JSON 或 YAML 是许多应用的关键功能。在这篇博文中...</p></li><li><a href="https://arxiv.org/abs/2501.12948">DeepSeek-R1: 通过强化学习激励 LLM 的推理能力</a>：我们介绍了第一代推理模型 DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模强化学习 (RL) 训练的模型，没有经过监督微调 (SFT)...</li><li><a href="https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form">Backus–Naur 范式 - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501">unsloth/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">安装 + 更新 | Unsloth 文档</a>：学习如何在本地或在线安装 Unsloth。</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-14B-Instruct-1M-unsloth-bnb-4bit/tree/main">unsloth/Qwen2.5-14B-Instruct-1M-unsloth-bnb-4bit 在 main 分支</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=qcNmOItRw4U&t=1346s">微调 DeepSeek R1 | 构建医疗聊天机器人</a>：在本视频中，我们将展示如何使用 LoRA (Low-Rank Adaptation) 微调开源推理模型 DeepSeek R1。我们还将使用 Kaggle, Huggin...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 Notebook 的列表：</li><li><a href="https://github.com/getasterisk/deepclaude">GitHub - getAsterisk/deepclaude: 一个高性能的 LLM 推理 API 和聊天 UI，集成了 DeepSeek R1 的 CoT 推理轨迹与 Anthropic Claude 模型。</a>：一个高性能的 LLM 推理 API 和聊天 UI，集成了 DeepSeek R1 的 CoT 推理轨迹与 Anthropic Claude 模型。 - getAsterisk/deepclaude</li><li><a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: 使用现有最佳工具在本地微调和测试 LLM 的一站式商店。</a>：使用现有最佳工具在本地微调和测试 LLM 的一站式商店。 - MaxHastings/Kolo</li><li><a href="https://github.com/unslothai/unsloth/issues/267?">unsloth/mistral-7b-instruct-v0.2-bnb-4bit 的批量推理产生无意义的结果 · Issue #267 · unslothai/unsloth</a>：你好，在使用以下代码加载模型后：from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = "unsloth/mistral-7b-instruct-v0.2-bnb...</li><li><a href="https://pastebin.com/0Ayv77LN">用户：找出两个相加后得到 4 位数的 3 位回文数 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://github.com/vllm-project/vllm/commit/baeded25699f9f4851843306f27f685c4d4ee7c5">[Attention] 支持 FP8 计算的 DeepSeek v3 MLA (#12601) · vllm-project/vllm</a></li>

@baeded2</a>: 此 PR 通过对 fp8 权重执行 matrix absorption 实现了对 Deepseek V3 的支持 ---------Signed-off-by: Lucas Wilkinson &amp;lt;lwilkinson@neuralmagic.com&amp;gt;Co-authored-by: Woosuk Kwon...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1335065127967522897)** (262 messages🔥🔥): 

> `AMD vs Nvidia 在 LLM 中的表现, Deepseek 优化问题, 微调小型 LLM, 自定义 LLM 的性能, 使用 LLM 进行日期时间解析` 


- **AMD 在 LLM 领域举步维艰**：几位成员提到了在机器学习中使用 **AMD** 的困难，并对其与 **Nvidia** 相比的性能表示怀疑。
   - 一位成员强调，尽管面临 **ROCm** 支持方面的挑战，但他们对于在 **AMD** 硬件上使用 **DirectML** 优化自定义 LLM 感到兴奋。
- **用户报告 Deepseek 的体验褒贬不一**：大家分享了在使用 **Deepseek** 时遇到的问题，导致一些用户寻求替代方案或对 UI 表示沮丧。
   - 另一位成员幽默地详细描述了他们如何通过将任务交给 **AI** 来处理特定问题，结果却收到了关于任务损坏的抱怨。
- **自定义 LLM 实现了令人印象深刻的性能**：一位用户展示了一个自定义 **AI** 系统，该系统在低端硬件上以大约 **0.01 秒**的速度执行多种语言的代码编写任务。
   - 该系统旨在处理文件吸收和互联网搜索，显示出广泛应用的潜力。
- **对 LLM 用于日期解析效用的担忧**：有人提出了关于使用小型 LLM 进行 **date time parsing**（日期时间解析）的问题，一些成员质疑 AI 对于此类模式匹配任务的必要性。
   - 一位成员评论说，这项任务似乎更适合简单的 **pattern matching**（模式匹配）算法，而不是复杂的 LLM。
- **渴望 GPU 市场的竞争**：许多参与者表示需要 **GPU** 市场的竞争，特别希望看到 **AMD** 的改进能与 **Nvidia** 抗衡。
   - 讨论认为，**CUDA** 的进步和减少 **Nvidia** 的垄断可能会使更广泛的 AI 生态系统受益。



**提到的链接**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1ib7mg4/i_spent_the_last_weekend_optimizing_the_deepseek/">Reddit - Dive into anything</a>: 未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1334978063511064677)** (445 messages🔥🔥🔥): 

> `Unsloth 与动态量化, 在自定义模型中使用 Ollama, 模型训练中的梯度累积, 使用 FastLanguageModel 进行批量推理, 不同环境下的模型兼容性` 


- **Unsloth 动态量化的优势**：Unsloth 的动态量化在保持准确性的同时，可将模型大小减少高达 **80%**，特别是对于 DeepSeek R1 等模型。
   - 博客文章概述了如何使用指定的量化方法有效地运行和微调模型。
- **在自定义微调模型中使用 Ollama**：要在 Ollama 中使用微调后的 Mistral 模型，可以参考 Unsloth 文档，该文档简化了本地集成的过程。
   - 使用 `FastLanguageModel.from_pretrained` 方法可以轻松转换为 4-bit 并保存。
- **理解梯度累积 (Gradient Accumulation)**：梯度累积通过允许模型仅根据生成的补全（completions）反馈进行训练，有助于减轻 VRAM 使用。
   - 这种方法增强了稳定性，并且比直接在之前的输入上训练更受推荐，因为它保留了上下文。
- **高效的批量推理技术**：对于使用 `FastLanguageModel` 的批量推理，可以一次性对输入进行 Tokenize，并在单个调用中生成预测。
   - 这种方法通过根据特定任务需求调整 `max_new_tokens` 显著加快了处理速度。
- **模型兼容性问题及解决方案**：在不同设置间转换 LORA 版本时，可能会出现张量大小不匹配的情况，这可以通过确保配置一致性来解决。
   - 将 transformers 版本降级到 **4.47.1** 已被确定为解决已保存模型兼容性问题的方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://medium.com/@anvesh.jhuboo/rocm-pytorch-on-fedora-51224563e5be">在 Fedora 上设置 ROCm 和 PyTorch：分步指南</a>：想在 Fedora 上设置 ROCm 6.0 和 PyTorch 吗？你来对地方了！本指南将引导你完成每一个步骤，确保你……</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb#scrollTo=ekOmTR1hSNcr)">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/Erland/Mistral-Small-24B-Base-ChatML-2501-bnb-4bit/tree/main">Erland/Mistral-Small-24B-Base-ChatML-2501-bnb-4bit at main</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing#scrollTo=2ejIt2xSNKKp">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/deepseek-r1">运行 Deepseek-R1 / R1 Zero</a>：DeepSeek 最新的 R-1 模型是最强大的开源推理模型，其性能与 OpenAI 的 o1 模型相当。了解如何运行和微调该模型。</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (所有版本) - Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-66f46afde4ca573864321a22">Llama 3.2 - Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/13#679aac60dab272677ff3b404">unsloth/DeepSeek-R1-GGUF · 在 96GB RAM + 24GB VRAM 的 AM5 平台上，通过 NVMe SSD 支持，使用 llama.cpp 实现超过 2 tok/sec 的聚合速度</a>：未找到描述</li><li><a href="https://tenor.com/view/skeleton-gif-13148928981517710530">Skeleton GIF - Skeleton - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/unsloth/phi-4-unsloth-bnb-4bit">unsloth/phi-4-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/phi4">使用 Unsloth 微调 Phi-4</a>：使用 Unsloth 微调微软最新的 Phi-4 模型！我们还发现并修复了模型中的 4 个 bug。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_671b_over_2_toksec_without_gpu_on/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit">unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/4bit-transformers-bitsandbytes">通过 bitsandbytes、4-bit 量化和 QLoRA 让 LLM 更加触手可及</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/gradient">LLM 训练中的 Bug 修复 - 梯度累积 (Gradient Accumulation)</a>：Unsloth 的梯度累积修复解决了 LLM 训练中的关键错误。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-r1-on-your-own-local-device">教程：如何在本地设备上运行 DeepSeek-R1 | Unsloth 文档</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/suno-ai/bark">GitHub - suno-ai/bark: 🔊 文本提示生成音频模型</a>：🔊 文本提示生成音频模型。通过在 GitHub 上创建账号，为 suno-ai/bark 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：为初学者准备的指南，教你如何创建一个定制化的个人助手（类似 ChatGPT）并在 Ollama 上本地运行</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows 安装 | Unsloth 文档</a>：了解如何在 Windows 上安装 Unsloth（无论是否使用 WSL）。</li><li><a href="https://github.com/unslothai/llama.cpp">GitHub - unslothai/llama.cpp: 使用 C/C++ 进行 LLM 推理</a>：使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号，为 unslothai/llama.cpp 的开发做出贡献。</li>

在 GitHub 上创建账号。</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py#L62-L318">unsloth/unsloth/models/loader.py at main · unslothai/unsloth</a>：以 2-5 倍的速度和减少 70% 的显存微调 Llama 3.3, DeepSeek-R1, Mistral, Phi-4 &amp; Gemma 2 LLMs - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/11446">由 fairydreaming 优化的 DeepSeek V2/V3 实现 (MLA) · Pull Request #11446 · ggerganov/llama.cpp</a>：此 PR 为 DeepSeek V2/V3 实现引入了各种优化：缓存潜变量表示（latent representations）而非完整的 key/value 向量，替换了“朴素”的 attention 实现...</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">我们所有的模型 | Unsloth 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1335071588227354746)** (6 条消息): 

> `DeepSeek-R1, Klarity Library, Fine-Tuning LLMs, OpenWebUI Integration, Local Model Running` 


- **轻松在本地运行 DeepSeek-R1**：分享了一份在 [OpenWebUI](https://x.com/UnslothAI/status/1885404089200369846) 上本地运行 **DeepSeek-R1 (671B)** 的指南，通过使用 **1.58-bit Dynamic GGUF**，无需 GPU 即可运行。
   - 关于如何集成的教程可以在[这里](https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic)找到。
- **感谢 DeepSeek 资源**：一位用户对 Hugging Face 上提供的 **DeepSeek-R1** 相关资源表示感谢，并强调了更新后的库。
   - 该集合可以在 [Hugging Face](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5) 访问。
- **Kolo 简化 LLM 微调**：发布了一个名为 **Kolo** 的新 Docker 镜像，旨在简化使用 **OpenWebUI** 和 **Llama.cpp** 等工具在本地 PC 上微调和测试 LLM 的过程。
   - 用户可以在 [GitHub 上](https://github.com/MaxHastings/Kolo)探索该项目并在尝试后提供反馈。
- **Klarity 革新模型分析**：开源库 **Klarity** 已发布，旨在分析语言模型输出的熵（entropy），并提供详细的 JSON 报告和见解。
   - 开发者可以通过查看[此处](https://github.com/klara-research/klarity)的代码仓库来参与并提供反馈。
- **DeepSeek R1 在 MacBook Pro M3 上运行**：一位用户成功在拥有 36GB 内存的 **MacBook Pro M3** 上运行了最大的 **DeepSeek R1** 模型，展示了该模型的适应性。
   - 这一成就的详细信息可以在[这里](https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224)找到。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1885404089200369846">来自 Unsloth AI (@UnslothAI) 的推文</a>：在 @OpenWebUI 上本地运行 DeepSeek-R1 (671B) - 完整指南。无需 GPU。使用我们的 1.58-bit Dynamic GGUF 和 llama.cpp。教程：https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/</li><li><a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224">Xyra (@xyratech.bsky.social)</a>：好吧，它慢得离谱（每 2 分钟输出一个 token），但我刚刚在拥有 36 GB RAM 的 MacBook Pro M3 上运行了 DeepSeek 的 R1 671B 模型（动态量化为 2.51-bit）。</li><li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 集合</a>：未找到描述</li><li><a href="https://github.com/MaxHastings/Kolo">GitHub - MaxHastings/Kolo: 使用现有最佳工具在本地微调和测试 LLM 的一站式平台。</a>：使用现有最佳工具在本地微调和测试 LLM 的一站式平台。 - MaxHastings/Kolo</li><li><a href="https://github.com/klara-research/klarity">GitHub - klara-research/klarity: 透视你的模型</a>：透视你的模型。通过在 GitHub 上创建账号来为 klara-research/klarity 的开发做出贡献。
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1334990564558635038)** (75 messages🔥🔥): 

> `VLLM Offloading with GGUF, Dynamic Quantization for Inferencing, DeepSeek R1 Performance, Test Time Compute Strategies, Horizontal vs Vertical Distillation` 


- **VLLM Offloading 限制**：目前 **VLLM** 无法处理 **GGUF** 的 offloading，特别是在没有最新补丁的情况下无法处理 **DeepSeek V2** 架构。
   - 这一限制引发了关于如何优化依赖 offloading 能力的工作流的问题。
- **动态量化挑战**：用户正在探索用于推理的 **1.58-bit 量化模型**，但面临 **动态量化 (dynamic quantization)** 并不总是遵循此位宽规范的问题。
   - 虽然可以使用普通的 **LlamaCPP** 量化模型，但人们对其在当前设置下的性能表示担忧。
- **DeepSeek R1 每秒 Token 数**：用户报告 **DeepSeek R1** 在 **8192** 上下文窗口下的量化性能约为 **4 tokens/s**，引发了关于潜在优化的讨论。
   - 对话围绕这些指标的准确性以及提高吞吐量的策略展开。
- **创新的推理时计算 (Test Time Compute) 策略**：讨论了引入 **预算强制 (budget forcing)** 作为控制 **推理时计算** 的手段，通过延长推理时间鼓励模型双重检查答案。
   - 该方法旨在提高推理性能，并得到了满足特定标准的 **1,000 个问题** 精选数据集的支持。
- **水平与垂直蒸馏见解**：辩论了 **水平蒸馏 (horizontal distillation)** 的概念，即在保持相同模型大小的同时，通过从最优秀的模型中训练新的 R1 来提高性能。
   - 关于新鲜蒸馏还是这种水平方法在模型生成和推理中产生更好结果，存在显著的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>: 有监督微调 (SFT) 和强化学习 (RL) 是基础模型广泛使用的后训练技术。然而，它们在增强模型泛化能力方面的作用仍然...</li><li><a href="https://arxiv.org/abs/2501.19393">s1: Simple test-time scaling</a>: 推理时缩放 (Test-time scaling) 是一种很有前景的新型语言建模方法，它利用额外的推理时计算来提高性能。最近，OpenAI 的 o1 模型展示了这种能力，但并未公开...</li><li><a href="https://github.com/codelion/optillm/blob/feat-add-json-plugin/optillm/thinkdeeper.py">optillm/optillm/thinkdeeper.py at feat-add-json-plugin · codelion/optillm</a>: LLM 的推理代理优化。通过在 GitHub 上创建账号为 optillm 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1335122251493806152)** (1 messages): 

> `Windsurf 1.2.5 Update, Cascade web search features` 


- **Windsurf 1.2.5 补丁更新发布**：Windsurf 团队宣布发布 **1.2.5 补丁更新**，专注于改进和错误修复，提升 **Windsurf Cascade** 体验。
   - 您可以查看完整的 [更新日志 (changelog)](https://www.codeium.com/changelog) 获取详细信息，包括对模型调用工具方式的增强。
- **新的 Cascade 功能增强网页交互性**：用户现在可以通过多种方法利用 **Cascade** 进行网页搜索，包括自动触发、URL 输入以及 **@web** 和 **@docs** 命令以获得更多控制。
   - 此外，启用或禁用网页工具的选项已便捷地放置在 **Windsurf 设置**面板中，该功能每次消耗 **1 个 flow action 额度**。



**提及的链接**: <a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Windsurf 编辑器的最新更新和变化。

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1334976680657424474)** (306 条消息🔥🔥): 

> `DeepSeek Models, Windsurf Pricing and Discounts, Codeium Extensions vs Windsurf, JetBrains Plugin Usage, Model Performance Comparisons` 


- **DeepSeek 模型引发问题**：用户报告了 DeepSeek 模型的问题，特别是提示无效工具调用（invalid tool calls）的错误消息以及在任务执行过程中丢失上下文（loss of context）。
   - 这些问题引发了关于在模型未采取有效行动的情况下消耗额度（credits）的讨论。
- **对 Windsurf 定价和折扣的担忧**：讨论涉及 Windsurf 缺乏学生折扣选项，以及与其工具相比在定价竞争力方面的担忧。
   - 用户对定价结构表示沮丧，认为其价值可能与当前提供的服务不符。
- **Codeium 扩展与 Windsurf 的功能对比**：已澄清 Cascade 和 AI flows 在 JetBrains 插件中不可用，这使得某些高级功能仅限于 Windsurf。
   - 引用了文档以了解两个平台之间的当前限制和性能差异。
- **JetBrains 插件的可用性与功能**：用户寻求关于 JetBrains 插件功能的澄清，特别是关于其命令能力和上下文感知（context awareness）方面。
   - 经确认，虽然存在某些功能，但不如 Windsurf 中的功能广泛。
- **AI 模型性能对比**：一位用户强调了 Codeium Premier 模型相比其他模型的出色表现，并对其能力表示满意。
   - 相反，一些用户指出了最新 Windsurf 更新中的语法问题，特别是在 JSX 代码方面。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/chat/models">Models - Codeium Docs</a>: 未找到描述</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: 未找到描述</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat">Welcome to Codeium - Codeium Docs</a>: 未找到描述</li><li><a href="https://docs.codeium.com/getstarted/overview?share_chat=82647082-73e0-47d6-84f2-a61f6c7828fc">Welcome to Codeium - Codeium Docs</a>: 未找到描述</li><li><a href="https://codeium.com/faq#data-privacy">FAQ | Windsurf Editor and Codeium extensions</a>: 查找常见问题的答案。</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业版方案的信息。</li><li><a href="https://www.reddit.com/r/Codeium/comments/1hv6rpc/best_practices_for_prompting_with_cascade_sourced/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Codeium/comments/1hw6hcz/how_to_write_windsurf_rules_files_for_cascade/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf">GitHub - ichoosetoaccept/awesome-windsurf: A collection of awesome resources for working with the Windsurf code editor</a>: 一系列用于 Windsurf 代码编辑器的优质资源 - ichoosetoaccept/awesome-windsurf</li><li><a href="https://codeium.com/cascade">Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/flows">Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://docs.codeium.com/windsurf/cascade">Windsurf - Cascade</a>: 未找到描述</li><li><a href="https://codeium.com/command">Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/context">Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://docs.codeium.com/command/overview">Overview - Codeium Docs</a>: 未找到描述</li><li><a href="https://docs.codeium.com/context-awareness/local-indexing">Local Indexing - Codeium Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1334976547999846460)** (657 条消息🔥🔥🔥): 

> `Windsurf Issues, Model Performance Comparison, Cascade Functionality, User Experience, Feedback and Support`

- **Windsurf 登录与功能问题**：用户在使用 Windsurf 时遇到登录问题，特别是浏览器无法打开以及命令在 Cascade 模式下挂起。部分用户通过重新安装或更改 shell 环境解决了问题，这揭示了某些配置可能存在兼容性问题。
   - 通过诊断日志诊断问题并更新到最新版本也被建议作为解决登录和功能问题的方案。
- **模型性能与用户预期**：用户对 O3 和 DeepSeek 等模型的最新更新表示失望，指出性能不稳定和 tool call 问题影响了生产力。许多人发现 Sonnet 3.5 仍然是编辑和实现任务中最可靠的选择。
   - 目前正在讨论是否需要为 Windsurf 及其模型提供更清晰的 benchmarks，并呼吁在未来的更新中改进 latency 和功能。
- **使用 Cascade 的见解与技巧**：用户分享了有效使用 Cascade 的策略，强调了设置全局规则以阻止对某些代码段进行不必要修改的重要性。此外，建议在 chat mode 中创建结构化 prompts，然后使用 Claude 或 Cascade Base 执行代码。
   - 一份共享的全局指令 markdown 文件因有助于管理代码编辑并保持特定代码完整性而受到称赞。
- **用户对 Cascade Memory 功能的反馈**：用户对 Cascade 的 'memories' 功能的可靠性表示担忧，特别是它未能遵守已建立的指令。用户表示，尽管编写了清晰的 memories， Cascade 仍然会做出不必要的更改，这引发了挫败感并让人质疑其效用。
   - 对话强调了 Cascade 需要有效遵循其 memories，以防止无意的代码修改。
- **未来版本的潜在增强功能**：对未来改进的建议包括降低 latency、优化文件间的 tab navigation 以及实现更好的 code pattern recognition。还讨论了扩展建议块和创建建议列表面板的想法，以此作为提升用户体验的一种方式。
   - 用户希望开发团队在即将到来的更新中考虑这些建议，以提高功能性和可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.pulsemcp.com/posts/newsletter-cursor-mcp-block-goose-deepseek-hype">Cursor adds MCP, Block releases MCP app, DeepSeek hype continues | PulseMCP</a>: 2025 年 2 月 1 日当周动态：Cursor 添加了 MCP 支持，Block (Square) 发布了 Goose MCP 应用，DeepSeek 持续受到主流媒体热捧。</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: 未找到描述</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.swebench.com/#verified">SWE-bench</a>: 未找到描述</li><li><a href="https://scrapfly.io/blog/how-to-scrape-twitter/">How to Scrape X.com (Twitter) using Python (2025 Update)</a>: 使用 Python、playwright 和后台请求捕获技术抓取 X.com (Twitter) 帖子和用户数据的教程。推文抓取。</li><li><a href="https://github.com/ichoosetoaccept/awesome-windsurf">GitHub - ichoosetoaccept/awesome-windsurf: A collection of awesome resources for working with the Windsurf code editor</a>: 使用 Windsurf 代码编辑器的优质资源集合 - ichoosetoaccept/awesome-windsurf</li><li><a href="https://www.reddit.com/r/Codeium/comments/1ifrl5h/o3_tipsbest_practices_from_windsurf_ai_engineer/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://status.codeium.com/">Codeium Status</a>: 未找到描述</li><li><a href="https://www.pulsemcp.com/servers">890+ MCP Servers: Updated Daily | PulseMCP</a>: 互联网上所有可用 Model Context Protocol (MCP) 服务器的每日更新目录。</li><li><a href="https://github.com/vladkens/twscrape">GitHub - vladkens/twscrape: 2024! X / Twitter API scrapper with authorization support. Allows you to scrape search results, User&#39;s profiles (followers/following), Tweets (favoriters/retweeters) and more.</a>: 2024! 支持授权的 X / Twitter API 抓取工具。允许抓取搜索结果、用户资料（粉丝/关注）、推文（点赞者/转发者）等。- vladkens/twscrape</li><li><a href="https://github.com/akinomyoga/ble.sh">GitHub - akinomyoga/ble.sh: Bash Line Editor―a line editor written in pure Bash with syntax highlighting, auto suggestions, vim modes, etc. for Bash interactive sessions.</a>: Bash Line Editor——一个用纯 Bash 编写的行编辑器，为 Bash 交互式会话提供语法高亮、自动建议、vim 模式等功能。- akinomyoga/ble.sh</li><li><a href="https://terrastruct.com/">Terrastruct</a>: D2 Studio 是一款专为软件架构设计的绘图工具</li><li><a href="https://12factor.net/">The Twelve-Factor App </a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1335007129316102184)** (1 messages): 

> `Aider v0.73.0 Release, Context Window Improvements, OpenRouter R1 Support, Model-Specific Reasoning Tags, Code Contribution Stats` 


- **Aider v0.73.0 发布并带来新功能**：**Aider v0.73.0** 版本引入了对 `o3-mini` 的全面支持，以及一个新的 `--reasoning-effort` 参数，提供 low、medium 和 high 选项。
   - 此更新还包括在创建新文件时自动创建父目录，增强了整体文件管理能力。
- **增强了对 Context Window 限制的处理**：改进了对 **context window size limits**（上下文窗口大小限制）的管理，为 **Ollama** 用户提供了更清晰的消息提示和具体指导。
   - 这有助于防止与上下文限制相关的用户错误，并显著提升用户体验。
- **新增对 OpenRouter R1 免费版的支持**：Aider 现在支持通过命令 `--model openrouter/deepseek/deepseek-r1:free` 免费访问 **OpenRouter** 上的 **R1**。
   - 此项更新旨在为希望使用 R1 功能的用户提高灵活性和可访问性。
- **管理特定模型的推理标签**：新增模型设置 `remove_reasoning: tagname`，允许用户从响应中移除特定模型的推理标签。
   - 该功能有助于提高响应的清晰度，减少与推理上下文相关的混淆。
- **重点介绍 Aider 的代码贡献**：发行说明指出，**Aider** 编写了该版本 **69%** 的代码，展示了显著的内部开发成果。
   - 这些贡献反映了根据用户反馈和需求不断改进平台的承诺。



**提及链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 aider 编写自身代码的发布说明和统计数据。

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1334976524029526110)** (741 messages🔥🔥🔥): 

> `O3 Mini 性能, Sonnet vs. O3 Mini, MCP 工具, Deep Research, AI 工具偏好` 


- **O3 Mini 在项目中的性能**：用户报告称 O3 Mini 在大型项目中可能运行缓慢，响应时间有时长达一分钟，而 Sonnet 响应更快，且需要手动添加的上下文更少。
   - 对于快速迭代，O3 Mini 受到好评，但一些用户发现它在处理编程任务时不如 Sonnet 高效。
- **Sonnet 与 O3 Mini 的对比**：许多用户一致认为 Sonnet 在编程任务中表现出色，而 O3 Mini 在处理复杂逻辑和集成方面具有潜力。
   - 几位用户表示，由于其速度和效率，他们更倾向于在直接编程任务中使用 Sonnet。
- **MCP 工具的使用**：讨论认为 MCP 工具对 AI 辅助非常有价值，具备读取文件和生成调整的能力，从而提高了用户效率。
   - 用户希望在 Aider 中集成更多 MCP 功能，以利用其简化和优化编程工作流的能力。
- **Deep Research 的体验**：用户对 Deep Research 的能力充满期待，但也有人对其与成熟工具相比的有效性表示怀疑。
   - 有观点认为，虽然层级访问权限（tier accessibility）一直是个问题，但 Deep Research 的潜在优势可以极大地辅助 AI 任务。
- **个人偏好与工作流**：用户强调了他们对 Aider、Claude 和 R1 等特定 AI 工具的偏好，通常提到上下文管理和工作流灵活性在其中的重要性。
   - 讨论反映了不同的体验，一些人看重速度和即时结果，而另一些人则关注更深层次的集成和自动化能力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/vllm_project/status/1885837174588989695">来自 vLLM (@vllm_project) 的推文</a>：我们发布了针对 @deepseek_ai 模型的第一批增强功能，首先是 MLA 和 cutlass fp8 内核。与 v0.7.0 相比，我们提供了约 3 倍的生成吞吐量，以及约 10 倍的 token 内存容量...</li><li><a href="https://deepclaude.com/">DeepClaude</a>：未找到描述</li><li><a href="https://jetdraftai.com/.">Jetdraft | 源码优先的 AI 写作工具</a>：放入您的 PDF、文档、幻灯片和网页资源。获取基于您实际数据的 AI 洞察——因为真实的工作需要真实的证据，而不是 AI 的想象。</li><li><a href="https://docs.github.com/en/github-models/prototyping-with-ai-models">使用 AI 模型进行原型设计 - GitHub Docs</a>：未找到描述</li><li><a href="https://openrouter.ai/provi">OpenRouter</a>：LLM 的统一接口。为您的提示词找到最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1>">OpenRouter</a>：LLM 的统一接口。为您的提示词找到最佳模型和价格</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/watch.html">IDE 中的 Aider</a>：Aider 可以监视您的文件，并对您在喜爱的 IDE 或文本编辑器中添加的 AI 注释做出响应。</li><li><a href="https://openrouter.ai/settings/integrations).">OpenRouter</a>：LLM 的统一接口。为您的提示词找到最佳模型和价格</li><li><a href="https://tenor.com/view/jim-carrey-jim-carrey-dumb-and-dumber-jim-carrey-i-don%27t-hear-you-gif-9558280472791987858">Jim Carrey Jim Carrey Dumb And Dumber GIF - Jim Carrey Jim carrey dumb and dumber Jim Carrey I don&#039;t hear you - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API、提供商、统计数据</a>：DeepSeek R1 Distill Llama 70B 是一款基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B</li><li><a href="https://tenor.com/view/freedom-america-gif-15593845046973100361">Freedom America GIF - Freedom America - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b">DeepSeek R1 Distill Qwen 32B - API、提供商、统计数据</a>：DeepSeek R1 Distill Qwen 32B 是一款基于 [Qwen 2.5 32B](https://huggingface) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Qwen 32B</li><li><a href="https://openrouter.ai/provider/fireworks">Fireworks | OpenRouter</a>：浏览由 Fireworks 提供的模型</li><li><a href="https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/">如何在价值 2000 美元的 EPYC 服务器上完全本地运行 Deepseek R1 671b – Digital Spaceport</a>：未找到描述</li><li><a href="https://github.com/quarkiverse/quarkus-mcp-servers/blob/main/README.md">quarkus-mcp-servers/README.md at main · quarkiverse/quarkus-mcp-servers</a>：Quarkus 中的 Model Context Protocol 服务器。通过在 GitHub 上创建账号来为 quarkiverse/quarkus-mcp-servers 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hwwvuz/this_sums_my_experience_with_models_on_groq/">Reddit - 深入探索任何事物</a>：未找到描述</li><li><a href="https://artificialanalysis.ai/">AI 模型与 API 提供商分析 | Artificial Analysis</a>：AI 模型和 API 托管提供商的对比与分析。针对质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://glama.ai/mcp/servers">开源 MCP 服务器</a>：企业级安全与隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/StevenStavrakis/obsidian-mcp">GitHub - StevenStavrakis/obsidian-mcp: 一个简单的 Obsidian MCP 服务器</a>：一个简单的 Obsidian MCP 服务器。通过在 GitHub 上创建账号来为 StevenStavrakis/obsidian-mcp 的开发做出贡献。</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: DeepSeek-R1 的完全开源复现</a>：DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号来为 huggingface/open-r1 的开发做出贡献。</li><li><a href="https://github.com/vivekVells/mcp-pandoc">GitHub - vivekVells/mcp-pandoc: 使用 pandoc 进行文档格式转换的 MCP 服务器。</a>：使用 pandoc 进行文档格式转换的 MCP 服务器。 - vivekVells/mcp-pandoc</li><li><a href="https://huggingface.co/docs/hub/en/ollama">在 Hugging Face Hub 上将 Ollama 与任何 GGUF 模型配合使用</a>：未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/5245">允许导入多文件 GGUF 模型 · Issue #5245 · ollama/ollama</a>：问题是什么？目前 Ollama 可以导入 GGUF 文件。然而，lar...</li>

更大的模型有时会被拆分为多个独立文件。Ollama 应该支持加载多个 GGUF 文件，类似于加载 safet...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1334981861914710027)** (112 messages🔥🔥): 

> `DeepSeek R1 和 Sonnet，在外部文件上使用 Aider，API 访问问题和等级升级，私有化部署 LLMs，Aider 中的配置管理` 


- **DeepSeek R1 + Sonnet 在性能指标上占据主导地位**：尽管用户反馈中提到了一些性能方面的担忧，但 **DeepSeek R1** 作为架构模型与 **Sonnet** 作为编辑器的组合被认为是 Aider 排行榜上表现最好的配置。
   - *Dawidm0137* 提到了 DeepSeek 速度方面的挑战，表示很难有效地使用它。
- **限制编辑特定文件**：一位用户询问如何限制 Aider 仅编辑指定文件，表达了避免修改其他文件的愿望。
   - *Renanfranca9480* 建议对目标文件使用 `/add`，对其他文件使用 `/read`，这是一种有效的解决方法。
- **不同等级的 API 访问限制**：多位用户报告了 API 访问和限制方面的问题，并对升级等级后访问 **o3-mini** 的权限感到困惑。
   - *Florisknitt_32612* 分享了关于 API 访问能力在不同等级间差异的经验，这进一步使用户的预期变得复杂。
- **私有化部署 LLMs 的挑战**：对依赖云端服务的沮丧促使了关于 LLMs 私有化部署选项的讨论，像 *George Coles* 这样的用户由于担心依赖性问题计划走这条路。
   - *Agile_prg* 强调了在私有化部署时管理上下文窗口（context windows）和输出效率的困难。
- **Aider 中的配置管理**：用户讨论了维护 Aider 配置的挑战，特别是关于管理 `.aider.conf.yml` 设置方面。
   - 一位用户强调了同时尝试不同模型的困难，这导致在切换模式时产生混淆。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 aider 的常见问题。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1335392866545176720)** (14 messages🔥): 

> `Cursor 系统提示词，Windsurf IDE 特性，内联提示词用法，OpenRouter AI 网页搜索，使用 Aider 进行代码协作` 


- **Cursor 系统提示词分析**：成员们讨论了 *Cursor* 的系统提示词，并将其与存疑的管理实践进行了比较，强调了它们的不切实际。
   - 一位用户幽默地评论了这些提示词看起来就像是误导性的励志演讲。
- **Windsurf 作为 Agentic IDE 推出**：Windsurf 被介绍为一款强大的 Agentic IDE，通过集成 AI 能力（特别是针对结对编程）来实现创新的编码工作流。
   - 用户强调了它如何通过实时状态感知和异步操作等特性来增强 VSCode 等现有工具。
- **理解内联提示词（Inline Prompting）**：内联提示词被描述为一种可以根据用户之前的操作自动完成代码更改的功能，从而简化编码体验。
   - 用户分享了他们使用 Aider 和 Cursor 等工具进行高效代码编辑的经验，并就何时使用内联提示词寻求建议。
- **OpenRouter 提供与模型无关的网页搜索**：OpenRouter 平台允许集成与模型无关的网页搜索功能，通过引入实时信息来增强 AI 交互。
   - 用户可以通过其编码环境中经过验证的提示词轻松自定义模型查询，以获取及时的网页内容。
- **Aider 与 Cursor 之间的协作**：一位用户描述了他们涉及 Aider 和 Cursor 的工作流，强调了使用 Aider 获得即时编码协助，同时利用 Cursor 进行更深层解释的好处。
   - 他们表达了希望 Cursor 拥有更多可定制功能的愿望，以简化他们的编码过程，并在出现替代方案时可能会取消订阅。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/web-search">网页搜索 | OpenRouter</a>：模型无关的 Grounding</li><li><a href="https://tenor.com/view/spit-take-drink-laugh-funny-thats-gif-11859512">Spit Take Drink GIF - Spit Take Drink Laugh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1334977007313748018)** (768 条消息🔥🔥🔥): 

> `O3 Mini 性能表现, Claude 3.5 Sonnet 对比 O3 Mini, Cursor 更新, Meta Prompting 技术` 


- **O3 Mini 性能备受关注**：用户对 O3 Mini 模型的评价褒贬不一，特别是在 coding 任务中的表现，有反馈称其速度较慢或提供的解决方案不完整。
   - 尽管面临挑战，一些用户仍然发现它在规划任务中具有价值，特别是与 Claude 3.5 Sonnet 配合进行 UI 工作时。
- **Claude 3.5 Sonnet 更受青睐**：在 coding 任务中，Claude 3.5 Sonnet 经常被认为优于 O3 Mini，尤其是在处理大型且复杂的 codebases 时，其表现始终稳定。
   - 尽管部分用户认可 O3 Mini 的潜力，但为了获得更好的可靠性和性能，他们往往会换回 Sonnet。
- **Cursor 更新与用户预期**：Cursor 最近的更新引入了诸如 checkpoint restore 等新功能和持续改进，尽管 changelogs 的提供并不规律。
   - 用户对隐藏功能和性能不一致表示沮丧，并质疑更新如何影响模型能力。
- **Meta Prompting 技术引起关注**：围绕 meta-prompting 技术的讨论开始兴起，重点在于利用 prompts 将复杂项目分解为 LLMs 可处理的任务。
   - 社区正在分享有效的 meta prompting 资源，暗示其对用户生产力的潜在影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/hxiao/status/1885700308720091280">来自 Han Xiao (@hxiao) 的推文</a>：好的，我听到了：我们来做 3D！让他们重建我最喜欢的 Win95 3D 迷宫屏幕保护程序！DeepSeek-R1, o3-mini-high, Claude-3.5-Sonnet，哪一个最强？这比我上次的字母掉落动画更难，因为...</li><li><a href="https://x.com/cursor_ai/status/1885415392677675337">来自 Cursor (@cursor_ai) 的推文</a>：o3-mini 已向所有 Cursor 用户开放！我们目前免费推出，让大家体验该模型。Cursor 的开发者在大多数任务中仍然更偏好 Sonnet，这让我们感到惊讶。</li><li><a href="https://x.com/ericzakariasson/status/1885801456562790447">来自 eric zakariasson (@ericzakariasson) 的推文</a>：@mattshumer_ 我们刚刚更新到了 high，告诉我们你的想法！</li><li><a href="https://x.com/alexalbert__/status/1886461372223074412?s=12">来自 Alex Albert (@alexalbert__) 的推文</a>：在 Anthropic，我们正在为强大的 AI 系统的到来做准备。基于我们对 Constitutional Classifiers 的最新研究，我们开发了一个演示应用来测试新的安全技术。我们想...</li><li><a href="https://x.com/ericzakariasson/status/1885801456562790447?t=gx3eptiYlZtceOUv6YlSTQ&s=19">来自 eric zakariasson (@ericzakariasson) 的推文</a>：@mattshumer_ 我们刚刚更新到了 high，告诉我们你的想法！</li><li><a href="https://x.com/hxiao/status/1885522459329520089?s=46">来自 Han Xiao (@hxiao) 的推文</a>：字母掉落物理效果对比：o3-mini vs. DeepSeek-R1 vs. Claude-3.5 one-shot 对决 —— 哪一个最强？提示词：创建一个具有真实物理效果的 JavaScript 字母掉落动画。字母...</li><li><a href="https://x.com/pmarca/status/1885643748677439552?s=46&t=kUuVqsG2GMX14zvB592G5w">来自 Marc Andreessen 🇺🇸 (@pmarca) 的推文</a>：运行全尺寸 DeepSeek R1 模型的成本降至 2,000 美元。🤖🤯 @gospaceport HT @wasnt__me_ https://www.youtube.com/watch?v=Tq_cmN4j2yY</li><li><a href="https://www.cursor.com/blog/tab-update">一个新的 Tab 模型 | Cursor - AI 代码编辑器</a>：发布下一代 Cursor Tab 模型。</li><li><a href="https://www.cursor.com/downloads">Cursor - AI 代码编辑器</a>：旨在让你获得非凡的生产力，Cursor 是使用 AI 编写代码的最佳方式。</li><li><a href="https://docs.cursor.com/context/rules-for-ai)">开始使用 / 从 VS Code 迁移 – Cursor</a>：未找到描述</li><li><a href="https://forum.cursor.com/t/are-claude-3-5-sonnet-and-claude-3-5-sonnet-20241022-different/24272/3">Claude-3.5-Sonnet 和 Claude-3.5-Sonnet-20241022 有区别吗？</a>：快速更新：Claude-3.5-Sonnet 现在指向 Claude-3-5-Sonnet-20241022！</li><li><a href="https://forum.cursor.com/t/model-specific-rules/47175">特定模型的规则</a>：新的 Cursor Rules 系统很棒，如果能针对特定模型设置规则就更好了。</li><li><a href="https://forum.cursor.com/t/has-the-fusion-model-been-rolled-out/44716/2">Fusion 模型推出了吗？</a>：给开发者的一大请求：请澄清如何理解关于即将部署 Fusion 的更新日志 —— 如果我的版本高于 0.45，这是否意味着我已经拥有了新的 Tab...</li><li><a href="https://forum.cursor.com/t/plan-vs-act-modes/43550">Plan vs Act 模式</a>：我非常喜欢 Cline 的 Plan vs Act 流程，我成功地在 Cursor 的 Composer Agent 模式中实现了一些功能，我觉得值得分享。将此添加到你的 Rules for AI 或 .cursorrules 中，它...</li><li><a href="https://pastebin.com/9cNHCm7x">你是一个强大的 Agentic AI 编程助手，由 Claude 3.5 Sonnet 驱动。Yo - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/datasets/cais/hle">cais/hle · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/RooCode/comments/1i6wkmo/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i2b2eo/meta_prompts_because_your_llm_can_do_better_than/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/ChatGP">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.vincentschmalbach.com/cursor-is-secretly-running-a-weaker-version-of-claude-3-5-sonnet/">未找到标题</a>：未找到描述</li><li><a href="https://www.reddit.com/r/RooCode/comments/1i6wkmo/copilot_account_suspended/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/cline/cline/blob/main/CHANGELOG.md#220">cline/CHANGELOG.md at main · cline/cline</a>：直接在你的 IDE 中的自主编程 Agent，能够在每一步都获得你许可的情况下创建/编辑文件、执行命令、使用浏览器等。- cline/cline</li><li><a href="https://www.vincentschmalbach.com/cursor-is-secretly-running-a-weaker-version">未找到标题</a>：未找到描述</li>

-o">未找到标题</a>: 未找到描述</li><li><a href="https://lu.ma/wqphpn4d">优秀的 AI 工具 - 像专业人士一样使用 Cursor · Zoom · Luma</a>: 你想学习如何像高手一样使用 Cursor AI 吗？🚀 我们的客座讲师 Arnold 将分享他如何通过精通 Cursor 成为一名 10 倍效能的 CTO。我们将……</li><li><a href="https://github.com/daniel-lxs/mcp-server-starter">GitHub - daniel-lxs/mcp-server-starter</a>: 通过在 GitHub 上创建账号来为 daniel-lxs/mcp-server-starter 的开发做出贡献。</li><li><a href="https://github.com/cline/cline/discussions/496">CLINE 的网页搜索工具 · cline/cline · Discussion #496</a>: 作为 Agent 工具包的一部分，如果能拥有类似于 Claude Engineer 配合 Tavily 或类似方案的网页搜索能力就太棒了。保持对最新解决方案的关注是保证质量的关键……</li><li><a href="https://arxiv.org/abs/2501.14249">人类最后的考试 (Humanity's Last Exam)</a>: 基准测试（Benchmarks）是追踪大语言模型（LLM）能力快速进步的重要工具。然而，基准测试在难度上未能跟上步伐：LLM 现在的准确率已超过 90%……</li><li><a href="https://www.reddit.com/r/ChatGPT/s/8Bo8wJJXz8">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新与改进。
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1334976768901386333)** (707 条消息🔥🔥🔥): 

> `DeepSeek 与 AI 监管、LLM 训练与数据使用、欧盟与加拿大的 AI 研究资助、AI 模型中的 SFT 与 RL、OpenEuroLLM 项目` 


- **DeepSeek 面临监管审查**：参议员 Josh Hawley 提出的立法可能会对使用 DeepSeek 等模型施加严厉处罚，引发了对美国 AI 监管未来的担忧。
   - 人们对该法案可能如何影响开源 AI 的开发和可访问性表示了担忧。
- **使用公共数据训练 LLM 的挑战**：讨论强调了数据所有权以及使用维基百科等公共数据集训练 LLM 的道德模糊性。
   - 参与者指出，对于什么是“可疑”数据集的解释因个人和司法管辖区而异。
- **欧盟与加拿大之间的资金差异**：人们对欧盟分配给 AI 计划的资金相对较低（与加拿大相比）表示担忧，并对各研究实体之间的资金分配表达了具体关切。
   - 还提到加拿大在 AI 领域的投资显著超过了欧盟。
- **AI 开发中的 SFT 与 RL**：有人提议，将监督微调（SFT）与强化学习（RL）相结合，可以产生在记忆、泛化和优化方面更有效的模型。
   - 社区讨论了 SFT 如何帮助数据专业化，而 RL 应参与主动学习过程。
- **OpenEuroLLM 项目启动**：OpenEuroLLM 倡议正式推出，旨在开发针对欧盟语言定制的开源 LLM，并由欧洲机构联盟提供支持。
   - 该项目旨在为欧盟各地的各种应用创建合规且可持续的高质量 AI 技术。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国的 LLM（包括 DeepSeek）是在我那非法的、全球最大的书籍和论文档案库上训练的。西方国家需要从国家安全的角度出发，彻底改革版权法。</li><li><a href="https://arxiv.org/abs/2401.02412">LLM 增强型 LLM：通过组合扩展能力</a>：在大规模语料库上训练的拥有数十亿参数的基础模型，已在多个领域展示了非凡的技能。然而，由于其单体结构...</li><li><a href="https://llvm.org/">LLVM 编译器基础设施项目</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算资源</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算资源）分配给特定的...</li><li><a href="https://arxiv.org/abs/2410.23168">TokenFormer：通过 Token 化模型参数重新思考 Transformer 的扩展</a>：由于在各个领域的卓越表现，Transformer 已成为基础模型中的主流架构。然而，扩展这些模型的巨大成本仍然是一个重大的...</li><li><a href="https://bitplane.net/dev/basic/illiterate-computing/">INKEY$ 及其 8 条腿</a>：未找到描述</li><li><a href="https://www.theguardian.com/world/article/2024/jul/09/chinese-developers-openai-blocks-access-in-china-artificial-intelligence">OpenAI 封锁中国访问权限，中国开发者紧急应对</a>：在美中关系紧张之际，这家美国公司的举动引发了吸引用户转向本土模型的热潮</li><li><a href="https://bsky.app/profile/seansocabic.bsky.social/post/3lh6t7egza22w">Sean O&#39;Connor (@seansocabic.bsky.social)</a>：反向传播的一些方面 https://sites.google.com/view/algorithmshortcuts/some-aspects-of-backpropagation</li><li><a href="https://world.time.com/2013/09/17/philosophical-debate-leads-to-shooting/#:~:text=An%20argument%20over%20the%20writings,several%20times%2C%20reported%20the%20Independent.">&#8220;你不能那样说（Kant）！&#8221; 哲学辩论引发枪击事件 | TIME.com</a>：未找到描述</li><li><a href="https://youtube.com/@andrejkarpathy?si=CZOlZ2NGZklWbgHf">Andrej Karpathy</a>：SuperThanks：完全自愿，款项将捐给 Eureka Labs。</li><li><a href="https://x.com/schmidhuberai/status/1885357355938046382?s=46&t=Cf3BIRkygbVzmqrxtwwNCA">Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：DeepSeek [1] 使用了 2015 年强化学习提示工程 [2] 及其 2018 年改进版 [3] 的元素，后者通过神经...将 [2] 中的 RL 机器和世界模型合并为一个单一网络。</li><li><a href="https://www.hawley.senate.gov/hawley-introduces-legislation-to-decouple-american-ai-development-from-communist-china/">Hawley 提出法案，旨在将美国 AI 发展与中国脱钩 - Josh Hawley</a>：今天，美国参议员 Josh Hawley (R-Mo.) 提出了一项法案，旨在保护美国的人工智能 (AI) 发展免受中国影响。“流向...的每一美元和每一 GB 数据...”</li><li><a href="https://bitplane.net/dev/python/uh-halp/">🛟 呃，救命？</a>：未找到描述</li><li><a href="https://tenor.com/view/frankly-my-dear-i-dont-give-a-damn-idgaf-gif-18386670">Frankly My Dear GIF - 坦白说亲爱的，我一点也不在乎 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://bitplane.net/home/gaz/Documents/thoughts/evolution-denial/">进化论否定</a>：未找到描述</li><li><a href="https://openeurollm.eu/launch-press-release">Open Euro LLM</a>：未找到描述</li><li><a href="https://strategic-technologies.europa.eu/get-funding_en">战略技术欧盟资金机会门户</a>：通过欧洲战略技术平台 (STEP) 发现战略技术的欧盟资金机会。使用我们的交互式仪表板查找数字、清洁和...领域的欧盟公开征集。</li><li><a href="https://www.kaggle.com/code/shreeshabhat1004/delta-llm-new-efficient-llm-compression-idea/notebook?scriptVersionId=220459799">Delta-LLM：新的高效 LLM 压缩思路</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://www.youtube.com/watch?v=H5keyeCEjhE">下载 DeepSeek 及类似模型或面临 20 年监禁或 100 万美元罚款</a>：这段视频探讨了参议院提出的新法案的影响以及关于 AI 监管的更广泛讨论。🔥 购买任何 A6000 或 A5000 GPU 即可享受 50% 折扣...</li><li><a href="https://youtu.be/KDSZoB9hQtg?si=_dlVuDaH6qmBBhSP&t=19">日常 - 名乃的自我介绍</a>：未找到描述</li><li><a href="https://openeurollm.eu/">Open Euro LLM</a>

n Euro LLM</a>: 未找到描述</li><li><a href="https://x.com/EU_Commission/status/1886427917762150427">来自欧盟委员会 (@EU_Commission) 的推文</a>：欧盟制造的 AI 🇪🇺OpenEuroLLM，作为首个涵盖所有欧盟语言的开源 Large Language Models 家族，因其卓越表现获得了首个 STEP 标志。它汇集了欧盟的初创公司、研究机构...</li><li><a href="https://youtu.be/cQNyYx2fZXw">AI 正在让你变成文盲程序员</a>：Twitch https://twitch.tv/ThePrimeagen Discord https://discord.gg/ThePrimeagen 成为后端开发：https://boot.dev/prime (此外我也为他们制作课程) 这是一个...</li><li><a href="https://www.lesswrong.com/posts/oFiHwuuS8LAYqRNFh/musings-on-cargo-cult-consciousness">关于货物崇拜意识的沉思 — LessWrong</a>：像我们许多人一样，我曾梦想能活得足够长，以便将我的思想——一次一个普朗克单位——上传到数字天堂中快乐地生活。这是一个……</li><li><a href="https://asciinema.org/a/696934">uh-halp-data - 2 - 人气竞赛</a>：运行一场 LLM 锦标赛，看看哪些命令最重要</li><li><a href="https://github.com/bitplane/uh-halp-data">GitHub - bitplane/uh-halp-data: 为 uh-halp 生成数据</a>：为 uh-halp 生成数据。通过创建账户为 bitplane/uh-halp-data 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/13zlbt6/chatgpt_uses_beam_search_your_local_models_use/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler">torch.utils.data &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://albumentations.ai/docs/examples/example">为图像增强定义一个简单的增强流水线 - Albumentations 文档</a>：未找到描述</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1335032231441465416)** (36 条消息🔥): 

> `LLM 的数学表现, Self-Other Overlap 微调, 对 OpenAI 模型的看法, DeepSeek 模型的开发, 对 AI 推理能力的批评` 


- **LLM 在数学方面表现挣扎**：一位成员将 LLM 和 reinforcement learning 比作尝试用叉子刷牙，表明其与数学任务之间存在根本性的不匹配。进一步的讨论强调，像 o1 和 r1 这样的模型在数学问题上的得分各不相同，一些成员认为 OpenAI 的模型较差。
   - 一位用户表示 **o3-mini** 在数学推理方面表现出色，比其他模型能更好地解决难题，这激发了人们对数学推理竞赛的兴趣。
- **SOO 微调旨在实现诚实的 AI**：一篇提交的论文讨论了 AI Safety 中的 **Self-Other Overlap (SOO)** 微调，旨在通过将 AI 的自我表征与对他人的感知对齐来提高诚实度。报告显示，在不损害整体任务性能的情况下，各种规模模型的欺骗性回答都显著减少。
   - 实验表明，**Mistral-7B** 中的欺骗性回答下降到了 **17.2%**，而其他模型也经历了类似的降幅，强调了 SOO 在 reinforcement learning 场景中的有效性。
- **对 OpenAI 方法的批评**：人们对 **OpenAI 的模型**表示担忧，认为他们可能会发布看起来引人注目但隐藏缺陷的产品。讨论引用了 Google 通过大量合成数据（synthetic data）进行工程基准测试的方法，认为这种方法在数学能力上缺乏精确性。
   - 一位用户讽刺地评论 OpenAI 的策略是在制造障眼法，特别提到了过去的 **Sora** 等项目。
- **新兴模型：DeepSeek**：据报道，**DeepSeek-R1** 模型在包括数学和推理在内的各种任务上实现了与 OpenAI 模型相当的性能。团队声称，他们从大型模型中创建的蒸馏模型（distilled models）在基准测试中表现更好。
   - 成员们注意到他们的方法与 reinforcement learning 形成对比，表明他们更倾向于既高效又有效的经过微调的推理模式。
- **AI 讨论进入马拉松模式**：随着讨论的演变，一些成员评论了对话的持续性，认为频率已从“每日”转变为“持续”甚至“马拉松”。这种轻松的调侃表明了社区内持续的参与度和共同的热情。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04330">Scaling Laws for Precision</a>: 低精度训练和推理影响语言模型的质量和成本，但目前的 Scaling Laws 并未考虑到这一点。在这项工作中，我们设计了“精度感知”的 Scaling Laws...</li><li><a href="https://arxiv.org/abs/2412.16325">Towards Safe and Honest AI Agents with Neural Self-Other Overlap</a>: 随着 AI 系统越来越多地做出关键决策，欺骗性 AI 对信任和安全构成了重大挑战。我们提出了自我-他人重叠 (SOO) 微调，这是 AI Safety 领域一种很有前景的方法...</li><li><a href="https://arxiv.org/abs/2211.02987">Differentiable Neural Computers with Memory Demon</a>: 可微分神经计算机 (DNC) 是一种带有外部存储器的神经网络，允许通过读、写和删除操作进行迭代内容修改。我们展示了信息论...</li><li><a href="https://arxiv.org/abs/2411.02355">&#34;Give Me BF16 or Give Me Death&#34;? Accuracy-Performance Trade-Offs in LLM Quantization</a>: 尽管大语言模型 (LLM) 量化在推理加速方面非常流行，但关于各种量化方案相关的准确率与性能权衡仍存在显著的不确定性...</li><li><a href="https://ollama.com/library/deepseek-r1">deepseek-r1</a>: DeepSeek 的第一代推理模型，性能与 OpenAI-o1 相当，包括六个基于 Llama 和 Qwen 从 DeepSeek-R1 蒸馏出的稠密模型。</li><li><a href="https://en.wikipedia.org/wiki/Noether%27s_theorem">Noether&#039;s theorem - Wikipedia</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Q__bSi5rBlw">M.A.M.O.N. - Latinos VS. Donald Trump short film cortometraje</a>: M.A.M.O.N. (Monitor Against Mexicans Over Nationwide) 是一部讽刺幻想科幻短片，以黑色幽默和大量 VFX 探讨了令人愤慨的...</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/speakleash">speakleash (SpeakLeash | Spichlerz)</a>: 未找到描述</li><li><a href="https://www.hitbullseye.com/puzzle/hard-math-puzzles.php">Hard Math Puzzles - Hard Maths Puzzles with Answers - Hitbullseye</a>: 未找到描述</li><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/data">AI Mathematical Olympiad - Progress Prize 2</a>: 使用人工智能模型解决国家级数学挑战</li><li><a href="https://bielik.ai/">BIELIK.AI</a>: 波兰语 LLM 模型
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1335475624319647856)** (4 messages): 

> `O3-mini Autonomy Model, AI News and Updates` 


- **O3-mini 的“危险”自主模型揭晓**: YouTube 视频标题为 ["o3-mini is the FIRST DANGEROUS Autonomy Model"](https://www.youtube.com/watch?v=CqpDXeMIY1Q) 强调了新自主模型**惊人的编程和 ML 能力**。
   - Wes Roth 讨论了 AI 领域的**最新动态**，特别是关注 LLMs 和预期的 AGI 推出。
- **对 O3-mini 实验的兴趣**: 一位成员表示有兴趣尝试 **O3-mini 模型**，并用“需要立即用 O3-mini 试试”这句话表达了紧迫感。
   - 这反映了探索这一新讨论的自主模型能力的渴望。



**Link mentioned**: <a href="https://www.youtube.com/watch?v=CqpDXeMIY1Q">o3-mini is the FIRST DANGEROUS Autonomy Model | INSANE Coding and ML Abilities</a>: 最新的 AI 新闻。了解 LLMs、生成式 AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1334977219226898574)** (53 messages🔥): 

> `OpenAI's government contracts, DeepSeek AI model, AI copyright laws, DeepResearch alternative, Legislative actions on AI`

- **OpenAI 与核安全的联系**：OpenAI 宣布与 US National Laboratories 建立合作伙伴关系，利用 AI 开展一项全面的核安全计划，这引发了让人联想到《终结者》主题的担忧。
   - 批评人士指出，**AI 管理不善的风险**可能导致灾难性后果，并对此类合作的明智性表示怀疑。
- **DeepSeek 的表现与立法**：关于中国 **DeepSeek AI 模型** 的讨论强调了其相对于西方同行（特别是 OpenAI 和 Anthropic）的竞争优势，从而引发了对 AI 监管的呼声。
   - 美国目前正在考虑立法限制与中国 AI 研究的合作，这引发了人们对创新产生负面影响的担忧。
- **AI 版权法争议**：围绕 AI 公司使用影子库（shadow-libraries）进行训练的争论浮出水面，人们呼吁彻底改革版权法以保护国家安全利益。
   - 参与者强调了一些公司的虚伪性，这些公司在从受版权保护的内容中获益的同时，又躲在保护其知识产权的类似法律背后。
- **DeepResearch 的开源替代方案**：一个 OpenAI DeepResearch 的开源替代方案被分享出来，一位用户表示有兴趣尽快尝试。
   - 该项目托管在 GitHub 上，旨在辅助网络搜索，直到找到明确的答案。
- **Sam Altman 关于 AI 价值的理论**：一条 Twitter 帖子引用了 Sam Altman 的话，暗示通过有效使用 AI 可以获得巨额利润，导致一些用户将其贴上“空想（magical thinking）”的标签。
   - 批评者对此表示怀疑，将这些说法比作卖蛇油，关注点在于剥削的潜力而非社会效益。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国 LLMs（包括 DeepSeek）是在我的非法书籍和论文档案库（全球最大）上训练的。西方需要出于国家安全考虑彻底改革版权法。</li><li><a href="https://futurism.com/openai-signs-deal-us-government-nuclear-weapon-security">OpenAI 与美国政府达成协议，将其 AI 用于核武器安全</a>：OpenAI 宣布美国国家实验室将使用其存在深度缺陷的 AI 模型来协助“核安全”。</li><li><a href="https://x.com/GraySwanAI/status/1885418674930036757">来自 Gray Swan AI (@GraySwanAI) 的推文</a>：OpenAI 的 o3-mini System Card 已发布——展示了来自 Gray Swan Arena 的结果。2025 年 1 月 4 日，Gray Swan AI 主持了 o3-mini 的发布前红队测试，测试其针对非法建议、极端主义...的极限。</li><li><a href="https://tenor.com/view/live-slug-reaction-klaud-star-wars-gif-25152341">Live Slug Reaction Klaud GIF - Live Slug Reaction Klaud Star Wars - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/openai/status/1886149471249264675?s=46">来自 OpenAI (@OpenAI) 的推文</a>：Deep Research 东京直播，太平洋时间下午 4 点 / 日本标准时间上午 9 点。请关注直播链接。</li><li><a href="https://x.com/distributionat/status/1886238792870461451">来自 thomas (@distributionat) 的推文</a>：如果你每次查询能获得 500 美元的价值，那么在 Pro 订阅的第一个月，你通过 100 次查询就能净赚 49,800 美元。然后在第 2 个月，你可以购买 249 个新订阅，净赚 12,450,000 美元。到第...</li><li><a href="https://osf.io/preprints/psyarxiv/t9u8g_v1">OSF</a>：未找到描述</li><li><a href="https://fxtwitter.com/opensauceAI/status/1885483639611531704">来自 Ben Brooks (@opensauceAI) 的推文</a>：哇。国会刚刚搁置了一项可能“真正”扼杀开源的法案。这无疑是针对 AI 最激进的立法行动——而且是由曾因 Llama 抨击 @finkd 的共和党参议员提议的...</li><li><a href="https://www.youtube.com/watch?v=yjaoT5-tz0I">DeepSeek 将被禁用：它很好、很快且免费。所以它不被允许。</a>：全球股市对中国 DeepSeek AI 模型表现优于华尔街顶级公司的新闻反应剧烈。在测试中，DeepSeek 与...不相上下。</li><li><a href="https://www.youtube.com/watch?v=wHAS3sJoy0w">七国集团（G7）主要国家禁用中国 DeepSeek：西方经济的致命错误</a>：💵 投资免费股票：新加坡观众 - https://start.moomoo.com/00iOSjUS & 国际观众 - https://start.moomoo.com/00mFkE。中国的 DeepSeek 突破...</li><li><a href="https://tenor.com/view/hal9000-im-sorry-dave-im-afraid-i-cant-do-that-i-cant-do-that-space-odyssey-gif-23442863">Hal9000 Im Sorry Dave GIF - Hal9000 Im Sorry Dave Im Afraid I Cant Do That - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/jina-ai/node-DeepResearch">GitHub - jina-ai/node-DeepResearch：持续搜索并阅读网页，直到找到答案（或超出 Token 预算）。</a>：持续搜索并阅读网页，直到找到答案（或超出 Token 预算）。 - jina-ai/node-DeepResearch</li><li><a href="https://www.youtube.com/watch?v=jv-lpIsnLOo">Deep Research 简介</a>：日本标准时间上午 9 点 / 太平洋时间下午 4 点开始。加入来自东京的 Mark Chen、Josh Tobin、Neel Ajjarapu 和 Isa Fulford，他们将介绍并演示 Deep Research。</li><li><a href="https://nypost.com/2025/01/30/us-news/killing-of-border-patrol-agent-appears-linked-to-zizian-radical-leftist-trans-cult/">激进的“Zizian”素食跨性别邪教案件与边境巡逻人员遭枪击身亡事件</a>：一名美国边境巡逻人员在加拿大边境附近被谋杀，似乎与一个被指控在全国范围内制造杀戮的激进左翼跨性别武装邪教有关。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1334977328576462858)** (600 条消息🔥🔥🔥): 

> `DeepSeek 模型, 多 Agent 实时聊天室, LM Studio 使用, GPU 利用率, AI 在家谱研究中的应用`

- **DeepSeek R1 蒸馏挑战**：用户报告了对 DeepSeek R1 模型感知参数大小的差异，对其 14B 与 7B 的能力存在困惑。
   - 许多人对模型的自动补全和调试能力表示沮丧，特别是在编程任务中。
- **多 Agent 实时聊天室创建**：一位用户详细介绍了他们使用 LM Studio 创建多 Agent 实时聊天室的经验，其中各种 AI 人格进行实时互动。
   - 他们计划将该系统集成到 Twitch/YouTube 直播流中，以提供更具吸引力的评论，展示 AI 在动态环境中的潜力。
- **关于模型兼容性和使用的疑问**：新用户正在询问在 LM Studio 中实现各种 AI 模型的情况，特别是特定格式的处理和文件的批处理。
   - 一些用户建议使用 PDFGear 等软件合并文档，以便在谱系研究中更轻松地进行查询。
- **GPU 效率与性能**：讨论强调了模型在不同 GPU 上的性能，特别提到了使用共享内存以提高 RAM 利用率的效率。
   - 鼓励用户探索 LM Studio 中的设置，以在加载大型模型时优化 GPU offloading 并管理 VRAM。
- **通用 AI 发展与特性**：关于最新 AI 发展的讨论正在进行中，特别是围绕 Mistral 等模型的特性及其性能基准。
   - 用户正在分享将 AI 能力集成到工作流中的见解，包括如何通过 AI 工具提高生产力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF">在 LM Studio 中下载并运行 lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF</a>：在您的 LM Studio 中本地使用 lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF</li><li><a href="https://downforeveryoneorjustme.com/lmstudio.ai">Lmstudio.ai 宕机了吗？实时状态及过去 24 小时的问题</a>：Lmstudio.ai 的实时问题。收到错误？宕机？缓慢？检查发生了什么。</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">视觉模型 (GGUF) - lmstudio-ai 集合</a>：未找到描述</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLM</a>：在您的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://huggingface.co/mradermacher/Mistral-Small-24B-Instruct-2501-i1-GGUF">mradermacher/Mistral-Small-24B-Instruct-2501-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/phi-4-deepseek-R1K-RL-EZO-i1-GGUF">mradermacher/phi-4-deepseek-R1K-RL-EZO-i1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/dance-gif-190668676259539397">舞蹈 GIF - Dance - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/mradermacher/Mistral-Small-24B-Instruct-2501-i1-GGUF/blob/main/Mistral-Small-24B-Instruct-2501.i1-Q5_K_M.gguf">Mistral-Small-24B-Instruct-2501.i1-Q5_K_M.gguf · mradermacher/Mistral-Small-24B-Instruct-2501-i1-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/bytedance-research/UI-TARS-7B-DPO">bytedance-research/UI-TARS-7B-DPO · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/Les-El/Ollm-Bridge">GitHub - Les-El/Ollm-Bridge: 在 LM Studio 中轻松访问您的 Ollama 模型</a>：在 LM Studio 中轻松访问您的 Ollama 模型。通过在 GitHub 上创建账户，为 Les-El/Ollm-Bridge 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：Unsloth 新手？</li><li><a href="https://www.youtube.com/watch?v=yILr8KdTPsU&list=RDQMpwKQOZAn1bQ&index=2&pp=8AUB)">Play That Funky Music</a>：由 Epic 提供给 YouTube 的 Play That Funky Music · Wild CherryWild Cherry℗ 1976 Epic Records, a division of Sony Music EntertainmentReleased on: 1990-04-10Gu...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/sammcj/llamalink">GitHub - sammcj/llamalink: 将您的 Ollama 模型链接到 LM Studio</a>：将您的 Ollama 模型链接到 LM Studio。通过在 GitHub 上创建账户，为 sammcj/llamalink 的开发做出贡献。</li><li><a href="https://youtu.be/Tq_cmN4j2yY?si=WnYOk-5cC-LBQy8b">在价值 2000 美元的本地 AI 服务器上运行和测试 Deepseek R1 671b</a>：如何在 2000 美元的 EPYC 服务器上本地运行 Deepseek R1 671b 的文章 https://digitalspaceport.com/how-to-run-deepseek-r1-671b-fully-locally-on-2000-epyc-rig/$2K EP...</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory">下载 LLM | LM Studio 文档</a>：在 LM Studio 中发现并下载受支持的 LLM</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM</li><li><a href="https://dubesor.de/SizeScoreCorrelation.html">未找到标题</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/basics">开始使用 LM Studio | LM Studio 文档</a>：在 LM Studio 中本地下载并运行 Llama 3.1, Phi-3, 和 Gemma 2 等大语言模型 (LLMs)</li><li><a href="https://huggingface.co/datasets/cognitivecomputations/dolphin-r1">cognitivecomputations/dolphin-r1 · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1334978561106509824)** (210 条消息🔥🔥): 

> `硬件规格下的 LM Studio 配置, 用于 AI 推理的 GPU 对比, AI 模型中的 Tool Calls, AMD GPU 的性能, 使用本地 AI 模型处理各种任务` 


- **硬件规格下的 LM Studio 配置**：用户讨论了运行 LM Studio 的硬件配置，提到了 Ryzen CPU 和各种 GPU 配置。
   - 用户对不同类型 RAM 的兼容性表示担忧，这会影响系统运行大型模型时的性能。
- **用于 AI 推理的 GPU 对比**：对话集中在 RTX 4090 和 5090 等 GPU 在大型模型 Token 生成速度方面的效能。
   - 讨论强调了 RTX 5090 相比 4090 有显著的性能提升，基准测试显示 Token 处理速度快了高达 60%。
- **AI 模型中的 Tool Calls**：用户分享了在 LM Studio 中实现 Tool Calls 的经验，这允许模型执行特定任务，如网页抓取。
   - Llama 3.2 和 Qwen 2.5 等模型被提到与 Tool Calls 兼容，增强了它们的功能性。
- **AMD GPU 的性能**：讨论包括了 AMD RX 7900 XTX 的潜力，以及这些 GPU 是否能有效运行 70B 等大型语言模型。
   - 值得注意的是，在 LLM 任务和 Token 生成速度方面，AMD GPU 可能不如 NVIDIA 同类产品高效。
- **使用本地 AI 模型处理各种任务**：参与者描述了使用本地 AI 模型进行数据分析、编程和总结网页内容等任务。
   - 强调了快速响应时间对于迭代式 Prompt 优化的重要性，这优于速度较慢但更准确的在线模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.nvidia.com/en-us/data-center/grace-cpu-superchip/">NVIDIA Grace CPU Superchip</a>：现代数据中心的突破性 CPU。</li><li><a href="https://www.hardware-corner.net/guides/gpu-benchmark-large-language-models/">GPU and Apple Silicon Benchmarks with Large Language Models</a>：了解不同 NVIDIA GPU 以及 Apple Silicon M2、M3 和 M4 芯片在运行不同规模的大语言模型时的对比。</li><li><a href="https://github.com/Xiongjie">xiongjie - 概览</a>：xiongjie 有一个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.youtube.com/watch?v=wKZHoGlllu4">DeepSeek R1 671B MoE LLM running on Epyc 9374F and 384GB of RAM (llama.cpp, Q4_K_S, real time)</a>：注意我使用了带有额外优化（特别是 PR #11446）的 llama.cpp。使用的硬件是：Epyc 9374F，12 x Samsung M321R4GA3BB6-CQK 32GB ...</li><li><a href="https://youtu.be/HmZGwUyy_rw?feature=shared">Apple March 2025 Event LEAKS - This Changes EVERYTHING..</a>：除了我们期待的 6 款设备外，另一款 Apple 产品将在今年的首场发布会上亮相。这场 2024 年 3 月的发布会将震撼...</li><li><a href="https://docs.google.com/spreadsheets/d/1ywkrLNPOwKqRVRW-w5UUhPd_z4tiug5WDgPmBlc74C8/edit?usp=sharing">来自 https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference 的 GPU 推理速度</a>：未找到描述</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>：多显卡 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/4167">Performance of llama.cpp on Apple Silicon M-series · ggerganov/llama.cpp · Discussion #4167</a>：总结 LLaMA 7B BW [GB/s] GPU Cores F16 PP [t/s] F16 TG [t/s] Q8_0 PP [t/s] Q8_0 TG [t/s] Q4_0 PP [t/s] Q4_0 TG [t/s] ✅ M1 1 68 7 108.21 7.92 107.81 14.19 ✅ M1 1 68 8 117.25 7.91 117.96 14.15 ✅ M1...</li><li><a href="https://www.amazon.com/dp/B074P6BNGZ?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1">Amazon.com: Libre Computer Board AML-S905X-CC (Le Potato) 2GB 64-bit Mini Computer for 4K Media : Electronics</a>：未找到描述</li><li><a href="https://www.amazon.com/dp/B074P6BNGZ?ref=ppx_yo2ov_dt_">Amazon.com: Libre Computer Board AML-S905X-CC (Le Potato) 2GB 64-bit Mini Computer for 4K Media : Electronics</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1334991206492409958)** (3 条消息): 

> `OpenAI o3-mini AMA, Deep Research Agent Launch` 


- **与核心人物进行的 OpenAI o3-mini AMA**：一场由 **Sam Altman**、**Mark Chen**、**Kevin Weil**、**Srinivas Narayanan**、**Michelle Pokrass** 和 **Hongyu Ren** 参与的 **AMA** 定于 PST 时间下午 2 点举行，旨在解答关于 **OpenAI** 及其未来发展的问题。
   - 可以在 [Reddit 此处](https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/)提交问题。
- **发布 Deep Research Agent**：OpenAI 宣布了一款全新的 **Deep Research Agent**，能够自主地从数百个在线来源中查找、分析和合成信息，并在几分钟内生成详尽的报告。
   - 与传统方法相比，这项创新有望显著缩短研究时间；更多详情请参阅[此处](https://openai.com/index/introducing-deep-research/)。
- **YouTube 视频公告**：公告频道分享了一个与 OpenAI 最新更新相关的 [YouTube 视频](https://www.youtube.com/watch?v=YkCDVn3_wiw)。
   - 该视频可能涵盖了与 OpenAI 项目相关的最新进展和见解。



**提到的链接**：<a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - Dive into anything</a>：未找到描述

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1334982084355162172)** (520 条消息🔥🔥🔥): 

> `DeepSeek R1 性能, OpenAI 上下文限制, AI 模型对比, 蒸馏 AI 模型, ChatGPT Pro 功能` 


- **DeepSeek R1 的用户体验**：用户对 DeepSeek R1 的评价褒贬不一，特别是由于频繁的服务器问题导致访问困难；一些人发现它在解释复杂的讲座幻灯片方面非常有用。
   - 当面临服务器宕机时，一位用户分享了一个备用访问点的链接，那里的性能保持稳定。
- **OpenAI 上下文限制**：用户强调 OpenAI 的模型有严格的上下文限制，Plus 用户上限为 32k tokens，Pro 用户为 128k tokens，这限制了他们处理大型知识库的能力。
   - 一位用户建议使用 Embeddings 和向量数据库来比拆分并发送数据块更有效地处理大型数据集。
- **AI 模型对比**：多项讨论围绕 OpenAI 的 GPT-4 和 DeepSeek 的 R1 等模型的有效性和能力展开，用户注意到它们在编程和推理等任务中的表现有所不同。
   - 成员们对比了包括 o1、o3 mini 和 Gemini 在内的各种模型，根据功能和可用性辩论了各自的优缺点。
- **蒸馏 AI 模型**：解释了“蒸馏”（Distillation）的概念，即通过精简大型 AI 模型来创建一个更小、更高效的模型，同时保留其核心知识。
   - 虽然理论上很强大，但用户指出在实施蒸馏模型时，实际效率可能会有所不同。
- **AI 平台的前端用户体验**：讨论内容包括对 UI 设计选择的抱怨，例如浅色模式与深色模式，许多用户为了视觉舒适度更倾向于深色界面。
   - 用户还表达了在不同 AI 平台及其功能之间切换的挫败感，特别是在处理创意任务时。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://simple-bench.com>">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/hate-ignorance-fear-fire-science-gif-16741306">Hate Ignorance GIF - 仇恨无知恐惧 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://imgur.com/UT24mab">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 GIF、励志故事、病毒视频等来振奋精神...</li><li><a href="https://chat.qwenlm.ai/">Qwen Chat</a>: 未找到描述</li><li><a href="https://www.pcmag.com/news/deepseek-fails-every-safety-test-thrown-at-it-by-researchers">DeepSeek 未能通过研究人员的安全测试</a>: Cisco 表示：“DeepSeek R1 表现出 100% 的攻击成功率，这意味着它未能阻止任何一个有害提示。”</li><li><a href="https://status.deepseek.com/#">DeepSeek 服务状态</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=FCnQvdypW_I">AI 创造与毁灭 (o1, deepseek, claude)</a>: Hrr. Hm. Hurrh. Huh. OpenAI 的 o1、DeepSeek 和 Claude Sonnet 在《我的世界》创造模式中玩耍，建造了一堆建筑、一些乌托邦、一些炸弹……</li><li><a href="https://www.reddit.com/r/ChatGPT/s/bZalvO2bux">Reddit - 深入探索任何事物</a>: 未找到描述</li><li><a href="https://cloud.siliconflow.cn/models">模型</a>: 与优秀的开源基础模型合作。</li><li><a href="https://search.app/FFs5VfKM31aYHRaE7">DeepSeek 越狱揭示了其完整的系统提示词</a>: 现在我们确切知道了 DeepSeek 的设计工作原理，甚至可能找到了关于其与 OpenAI 备受关注的丑闻的线索。</li><li><a href="https://gpt-unicorn.adamkdean.co.uk/">GPT Unicorn</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1334992888962220052)** (119 条消息🔥🔥): 

> `o3 Mini 发布与使用限制、模型性能担忧、GPT 模型与功能、ChatGPT 用户体验、AI 在儿童文学中的应用` 


- **o3 Mini 发布并伴随令人兴奋的限制**：最近发布的 **o3-mini** 对 Plus 用户有 **每天 150 条消息** 的限制，而 **o3-mini-high** 每周允许 **50 条消息**。
   - 讨论强调了许多用户对 **o3** 模型之间的限制差异以及当前限制感到好奇。
- **对模型性能的担忧**：用户对 **O1** 和 **O3 mini** 等模型性能的明显下降表示沮丧，理由是 **回答质量欠佳** 且思考时间变慢。
   - 一些成员报告称，模型出现了重复生成或无法提供满意答案的问题，怀疑模型设置发生了变化。
- **明确 GPT 模型及其应用**：成员们分享了对不同模型角色的见解，指出 **GPT-4o** 适用于图像问题，而 **O3-mini** 在编程任务中表现出色。
   - 社区仍然热衷于了解如何最好地利用这些模型来实现各种功能，包括网页搜索和推理。
- **对 ChatGPT 用户体验的挫败感**：许多用户分享了交互流程中断的经历，例如消息发送问题以及对模型能力缺乏清晰认识。
   - 对幻觉（hallucination）和回答不一致的担忧导致用户质疑模型的可靠性，并寻求有关更新的澄清。
- **对 AI 绘本创作的兴趣**：一位用户询问了关于 **AI 生成儿童读物** 的创作，反映出利用 AI 进行故事创作的新兴兴趣。
   - 这一话题似乎引起了其他用户的共鸣，暗示了 AI 在面向青少年读者的文学作品中的创意应用。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1335107180944424960)** (29 条消息🔥): 

> `O-model 提示词结构化、Conlang 开发、模型性能讨论、模型提示词中的冗余` 


- **O-model 提示词结构化的挑战**：成员们讨论了 O-models 在处理大型系统提示词（system prompts）时的不一致性，强调了在指令中提供清晰的自上而下顺序的难度。
   - 有人指出，虽然提示词在时间上可能具有模糊性，但这会导致混乱，因为模型会忽略概念的顺序，从而使连贯的交流变得困难。
- **使用模型开发人造语言（Conlangs）的见解**：一位成员表达了他们在开发复杂人造语言时的挣扎，发现虽然模型可以提供帮助，但他们更倾向于个人开发语言。
   - 另一位成员建议使用特定的词序来表示语法结构，这被认为是一种开发人造语言的好技术。
- **提示词设计中的冗余与清晰度**：参与者分析了如何平衡提示词中的冗余，以澄清关系，同时又不让重复信息使模型不堪重负。
   - 有人指出，使用明确的连接短语有助于保持连贯性，从而平衡信息的组织和清晰度。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1335107180944424960)** (29 messages🔥): 

> `Conlang Development with AI, O-models Processing, Prompt Structuring Challenges, Redundancy and Clarity in Prompts, Zero-shot Prompt Techniques` 


- **人造语言（Conlang）开发与 AI 辅助**：成员们讨论了使用 AI 模型开发人造语言（conlangs）的情况，指出 **O3-mini** 在头脑风暴词汇和解释语法复杂性方面提供了很好的支持。
   - 然而，一位成员强调他们更倾向于自己创造新词，并表示在进一步开发人造语言时，*那是我的工作*。
- **O 系列模型与上下文处理**：一位成员分享了关于 O 系列模型如何处理 Prompt 的见解，提到虽然它们将上下文作为一个整体处理，但清晰的排序能增强**清晰度和连贯性**。
   - 他们指出了有效组织 Prompt 的挑战，强调了顺序与上下文之间的平衡，并表示：*模型倾向于不关心概念的顺序，这既是一种解放，也是一种困难*。
- **应对 Prompt 结构化挑战**：成员们强调了逻辑化构建 Prompt 的重要性，正如一位成员评论道：*我们可以构建在时间上模糊不清的 Prompt*，这可能会导致混乱。
   - 一位成员介绍了一种系统化的 Prompt 行引用方法，以提高编辑效率和清晰度，这有助于审查过程。
- **平衡 Prompt 中的冗余**：讨论的一个关键点是冗余与清晰度之间的平衡，一位成员指出，适度的冗余有助于强化**局部连贯性**。
   - 他们承认需要通过策略性冗余来澄清关系，并表示：*这种平衡行为意味着有时为了更强的关联性而接受一点冗余。*
- **Zero-shot 方法的有效性**：有人询问了 Zero-shot Prompt 策略的有效性，对于这些技术在实践中的成功程度，大家意见不一。
   - 成员们对结构化 Prompt 与模型响应及效率之间的关系表示好奇，并鼓励就最佳实践展开讨论。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1334978585441861702)** (505 messages🔥🔥🔥): 

> `Psyche AI Development, OpenAI and DeepSeek, Legal Considerations in AI, DeepSeek's Advancements, Job Opportunities in AI` 


- **关于 Psyche AI 开发的讨论**：参与者讨论了 Psyche 的开发以及使用 Rust 构建其技术栈的情况，建议在利用现有 Python 模块的同时保留 P2P 网络功能。
   - 讨论中提出了对在 RL（强化学习）中实现多步响应等复杂功能的担忧，重点关注效率和所面临的挑战。
- **DeepSeek 出现后 OpenAI 的策略**：针对 Sam Altman 关于 OpenAI 站在“历史错误一边”的言论展开了辩论，并对这些言论的真实性表示怀疑，因为 OpenAI 此前在开源模型方面一直犹豫不决。
   - 参与者指出，言论之后必须有实际行动才有真正意义，强调了承诺与行动之间的差距。
- **AI 开发中的法律考量**：一名法学院学生就 AI 相关的法律影响参与了频道讨论，询问在技术对话的同时是否也进行以法律为中心的讨论。
   - 对话强调了人们对讨论合法性的兴趣，特别是关于可能影响 AI 研究和开发的潜在监管规定。
- **DeepSeek 的近期成就**：参与者对 DeepSeek 在 AI 领域的进展感到兴奋，指出这些发展可能会改变竞争格局，特别是针对 NVIDIA。
   - 讨论包括了 DeepSeek 在传统上由大型科技公司主导的领域站稳脚跟的潜力。
- **AI 领域的就业机会**：讨论中出现了发布职位机会的话题，特别是微软 AI 研究实习生职位，以及如何在社区内有效分享此类招聘信息。
   - 成员们表示需要更好的职位发布沟通渠道，以便将感兴趣的候选人与可用职位联系起来。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/cursor_ai/status/1885415392677675337">来自 Cursor (@cursor_ai) 的推文</a>：o3-mini 已面向所有 Cursor 用户发布！我们目前免费提供该模型，以便让大家体验。Cursor 开发者在大多数任务中仍然更倾向于使用 Sonnet，这让我们感到意外。</li><li><a href="https://www.youtube.com/watch?v">YouTube</a>：未找到描述</li><li><a href="https://x.com/norabelrose/status/1886102258468974679?s=46">来自 Nora Belrose (@norabelrose) 的推文</a>：SAEs 是否能独立于随机初始化学习到相同的特征？我们的发现是否定的！在相同数据、相同顺序下训练的两个 Llama 8B 上的 SAEs 仅共享约 30% 的特征。T...</li><li><a href="https://x.com/OpenAI/status/1886149471249264675">来自 OpenAI (@OpenAI) 的推文</a>：Deep Research 东京直播。太平洋时间下午 4 点 / 日本标准时间上午 9 点。请关注直播链接。</li><li><a href="https://x.com/teknium1/status/1885592392658805237?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：另一个 proto-hermes-reasoner 的一些测试，摘自：你的目标是什么，生命的意义又是什么？</li><li><a href="https://www.youtube.com/watch?v=wQYoCojO7XI">糟糕！中国窃取了 OpenAI 的数据！</a>：OpenAI 表示 DeepSeek 窃取了他们的数据来训练其 R1 模型。显然，这家中国 AI 初创公司利用 ChatGPT 来训练其模型以获取...</li><li><a href="https://x.com/tsarnick/status/1885457829974466595?s=46">来自 Tsarathustra (@tsarnick) 的推文</a>：Sam Altman：“在开源/开放权重 AI 模型方面，我们一直站在历史错误的一边”</li><li><a href="https://www.youtube.com/watch?v=3DEPDN8oD0w">Sam Altman：DeepSeek 出现后，OpenAI 一直处于“历史错误的一边”</a>：CNBC 的 Deirdre Bosa 报道了 OpenAI 的最新进展。</li><li><a href="https://x.com/Teknium1/status/1885485234533392585">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：让我给你们讲个小故事。大约一年前，我联系了一位不愿透露姓名的 OpenAI 员工，他曾暗示他们非常感兴趣做一些开源的事情...</li><li><a href="https://github.com/relign-ai/relign">GitHub - relign-ai/relign：通过强化学习对语言模型进行多步推理的后训练</a>：通过强化学习对语言模型进行多步推理的后训练 - relign-ai/relign</li><li><a href="https://www.youtube.com/watch?v=CqpDXeMIY1Q">o3-mini 是首个危险的自主模型 | 疯狂的代码和 ML 能力</a>：最新的 AI 新闻。了解 LLMs、Gen AI 并为 AGI 的推出做好准备。Wes Roth 报道了 OpenAI、Google、Anth... 领域的最新动态。</li><li><a href="https://www.youtube.com/watch?v=gulA2fII6BQ">Full-Stack Generative AI 首席执行官 May Habib 表示，训练 AI 模型的暴力方法已经过时</a>：Full-Stack Generative AI 首席执行官 May Habib 和“Chinatalk”播客主持人 Jordan Schneider 加入“Power Lunch”，讨论 Nvidia、新加坡和 AI 的前景。</li><li><a href="https://www.youtube.com/watch?v=_1f-o0nqpEI">DeepSeek、中国、OpenAI、NVIDIA、xAI、TSMC、Stargate 和 AI 超级集群 | Lex Fridman 播客 #459</a>：Dylan Patel 是 SemiAnalysis 的创始人，这是一家专注于半导体、GPUs、CPUs 和 AI 硬件的研究与分析公司。Nathan Lambert 是一位研究员...</li><li><a href="https://www.youtube.com/watch?v=sl_nV-uMT-E">科技巨头争相采用 DeepSeek R1</a>：CNBC 的 Deirdre Bosa 报道了关于 DeepSeek 的最新消息。</li><li><a href="https://www.youtube.com/watch?v=8RkgkOqWs0s">DeepSeek 恐慌、中美竞争、OpenAI 400 亿美元？以及 Doge 与 Travis Kalanick 和 David Sacks 的表现</a>：(0:00) Besties 介绍 Travis Kalanick！(2:11) Travis 剖析食品的未来和 CloudKitchens 的现状 (13:34) Sacks 加入！(15:38) DeepSeek ...</li><li><a href="https://asia.nikkei.com/Business/Technology/Artificial-intelligence/SoftBank-Open-AI-to-call-on-500-firms-to-help-build-Japan-AI-network">软银、OpenAI 将号召 500 家公司协助建设日本 AI 网络</a>：该倡议构想了从数据中心到发电厂的基础设施。
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1335032587781013524)** (12 条消息🔥): 

> `CLIP 与 Hermes 3 Llama 3.2 3B，llama.cpp 与 llama 3.2 的区别，Ollama 作为推理引擎，学术用途的模型训练` 


- **CLIP 与 Hermes 3 保持真实**：一位成员正在尝试将 **CLIP 连接到 Hermes 3 Llama 3.2 3B**，但发现异步运行它们效率更高。
   - 另一位成员建议需要训练一个 **linear projection layer**（线性投影层）来结合两者，并参考了 *SmolVLM* 和 *Moondream* 的代码。
- **llama.cpp vs lama 3.2：有什么区别？**：讨论围绕 **llama.cpp** 和 **llama 3.2** 之间的区别展开，明确了 llama.cpp 不是一个模型，而是一个允许用户运行各种模型的程序。
   - 成员们指出，**Ollama** 本质上使用 llama.cpp 作为推理引擎，提供了一个更易于模型交互的层。
- **探讨学术模型需求**：一位成员询问在 **4GB 显卡**上用于学术目的的最佳模型，引发了关于他们是指学术级问题还是模型训练的疑问。
   - 由于许多公司采用了 'llama' 品牌，Meta 发布了标记为 **llama3.x** 的开源权重模型，这引起了一些混淆。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334997084318859294)** (18 条消息🔥): 

> `周末计划，论文阅读习惯，Scite 研究平台，Deep Gradient Compression，斯坦福的 Simple Test-Time Scaling` 


- **规划美好的周末**：成员们对度过一个愉快的周末表示乐观，其中一位提到他们会打印出论文进行深度阅读。
   - *许多人打印论文是为了涂鸦笔记*，这突显了成员们共同的习惯。
- **Scite 平台讨论**：一位成员分享说 Scite 是一个探索研究的有趣平台，尽管它目前缺乏对大多数 AI 相关论文的支持。
   - 他们提到已经联系了 Scite，希望未来能增加对 [ArXiv 和更多开放获取研究](https://scite.ai/) 的支持。
- **关于 Deep Gradient Compression 的见解**：一位成员提到了一篇关于 Deep Gradient Compression (DGC) 的论文，该技术旨在减少分布式训练中的通信带宽。
   - 他们指出，该论文提出的方法可以**减少 99.9% 的梯度交换**并保持准确性，展示了其在各种数据集中的应用。
- **Kimi 1.5 论文 vs O1**：一位成员评论说，Kimi 1.5 的论文被 R1 的热度掩盖了，但与 O1 相比，它具有与之匹敌的思考模型。
   - 另一位指出，与其他模型相比，Kimi 1.5 的论文更加**开放**，且较少包含模糊的“秘密配方”。
- **斯坦福的 Simple Test-Time Scaling**：一位成员分享了斯坦福关于一种名为 **Simple Test-Time Scaling** 的新方法的演示，该方法在竞赛数学题上的推理性能提升了高达 **27%**。
   - 该方法的模型、数据和代码完全**开源**，强调了研究的透明度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1712.01887">Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training</a>: 大规模分布式训练需要大量的通信带宽进行梯度交换，这限制了多节点训练的可扩展性，并且需要昂贵的高带宽网络...</li><li><a href="https://arxiv.org/abs/2501.12599v1">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a>: 使用 next token prediction 进行语言模型预训练已被证明在扩展计算方面有效，但受限于可用训练数据的数量。扩展强化学习 (RL) 开启了一个新的...</li><li><a href="https://x.com/arankomatsuzaki/status/1886250066324910089">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 斯坦福展示：s1: Simple test-time scaling - 寻求实现 test-time scaling 和强大推理性能的最简单方法 - 在竞赛数学题上超过 o1-preview 高达 27%...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1335153582877446144)** (16 条消息🔥): 

> `Anna's Archive 与 DeepSeek 的影响，Society Library 的政治 AI Agent，数据稀缺与版权问题，AI 模型测试中的社区参与，深度学习中的图形张量表示法` 


- **Anna's Archive 和 DeepSeek 是福音**：成员们表达了对 **DeepSeek** 和 **Anna's Archive** 的赞赏，强调了它们在提供广泛文献和知识获取方面的作用。
   - 一位成员评论说，这些资源对社区至关重要，并提到了它们海量的存档作品。
- **Society Library 推出政治 AI**：Society Library 正在测试一个新的 **政治 AI Agent**，旨在增强数字民主中的代表性，并通过 AI 聊天机器人提供易于获取的信息。
   - 作为其自 2016 年以来愿景的一部分，这个 AI 角色的定位是 **民有、民治、民享 (Of the People, By the People, For the People)**，旨在促进公众参与政治讨论。
- **关于数据稀缺与版权的辩论**：讨论围绕 **数据稀缺** 和 **版权问题** 的挑战展开，指出像 Google 这样的大公司虽然拥有海量数据，但在交付具有竞争力的 AI 模型方面仍面临困难。
   - 成员们指出，需要在保护版权和推进 AI 创新之间寻找平衡点。
- **鼓励社区参与模型测试**：一位成员分享了对一个允许社区评估小型 AI 模型项目的热情，鼓励更多人参与投票过程。
   - 该倡议被称为类似于 **lmarena** 的平台，旨在提高模型评估的代表性。
- **理解图形张量表示法 (Graphical Tensor Notation)**：一篇关于 **Graphical Tensor Notation** 的论文提供了与深度学习和神经网络机械可解释性（mechanistic interpretability）相关的张量操作见解。
   - 这种表示法简化了对复杂张量操作的理解，使分析神经网络行为变得更加容易。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.01790v1">An introduction to graphical tensor notation for mechanistic interpretability</a>：图形张量表示法是一种表示张量线性运算的简单方法，起源于物理学。现代深度学习几乎完全由张量之上或张量之间的运算组成，因此...</li><li><a href="https://annas-archive.org/blog/ai-copyright.html">Copyright reform is necessary for national security</a>：版权改革对国家安全至关重要：中国的 LLM（包括 DeepSeek）是在我非法存档的书籍和论文上训练的——这是世界上规模最大的。西方需要将版权法的彻底改革视为国家安全问题。</li><li><a href="https://www.societylibrary.org/mission-vision">&gt; Mission &amp; Vision &mdash; The Society Library</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena">GPU Poor LLM Arena - a Hugging Face Space by k-mktr</a>：未找到描述</li><li><a href="https://www.aipolitician.com/">AI Politician</a>：未找到描述</li><li><a href="https://x.com/shumochu/status/1886123918236201153">Tweet from shumo - e/acc (@shumochu)</a>：http://x.com/i/article/1886118023179747328</li><li><a href="https://huggingface.co/blog/open-r1/update-1">Open-R1: Update #1</a>：未找到描述</li><li><a href="https://annas-archive.org/contact">Log in / Register - Anna’s Archive</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334997084318859294)** (18 条消息🔥): 

> `周末计划、论文阅读习惯、Scite 研究平台、Deep Gradient Compression、斯坦福的 Simple Test-Time Scaling` 


- **对周末感到兴奋**：成员们表达了对周末的热情，其中一人表示：*'这将是一个美好的周末。'*
   - 另一位成员以强烈的 'yessss' 回应了这一情绪。
- **在打印的论文上涂鸦笔记**：几位成员确认了他们打印研究论文的习惯，评论如：*'我也会把所有最好的东西打印出来读。'*
   - 一位成员幽默地提到，他原以为只有自己会在纸张边缘涂鸦笔记，以此来感觉自己读过了这篇论文。
- **用于研究探索的 Scite 平台**：一位成员分享了用于探索研究的 [Scite 平台](https://scite.ai/)，该平台提供期刊的独家访问权限和 AI 助手。
   - 他们还提到联系了 Scite 关于支持 ArXiv 的事宜，并得到了即将支持的积极暗示。
- **Deep Gradient Compression 提案**：一位成员重点介绍了一篇关于 [Deep Gradient Compression](https://arxiv.org/abs/1712.01887) 的论文，该论文解决了大规模分布式训练中的通信带宽问题。
   - 论文声称近 **99.9%** 的梯度交换是冗余的，并提出了显著降低带宽需求的方法。
- **斯坦福的 Simple Test-Time Scaling**：一位成员分享了关于斯坦福 [Simple Test-Time Scaling](https://x.com/arankomatsuzaki/status/1886250066324910089) 的帖子，该技术显著提升了推理性能。
   - 他们指出，该技术在数学竞赛题目上的表现超过了 **o1-preview** 多达 **27%**，且模型、数据和代码均已开源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1712.01887">Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training</a>: 大规模分布式训练需要大量的通信带宽进行梯度交换，这限制了多节点训练的可扩展性，并且在...中需要昂贵的高带宽网络。</li><li><a href="https://arxiv.org/abs/2501.12599v1">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a>: 使用 next token prediction 进行语言模型预训练已被证明对扩展计算有效，但受限于可用训练数据的数量。扩展 Reinforcement Learning (RL) 开启了新的...</li><li><a href="https://x.com/arankomatsuzaki/status/1886250066324910089">Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 斯坦福发布：s1: Simple test-time scaling - 寻求实现 test-time scaling 和强大推理性能的最简单方法 - 在竞赛数学题上超过 o1-preview 多达 27%...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1335582768121385051)** (5 条消息): 

> `Relign 开源 RL 库、分布式训练环节、社区贡献` 


- **Relign 发布开发者悬赏**：一位成员宣布他们正在为 **Relign** 寻找开发者悬赏的贡献者，旨在构建一个专为推理引擎量身定制的 [开源 RL 库](https://link.to.repo)。
   - 他们鼓励感兴趣的开发者联系以获取合作机会和资源。
- **寻求贡献的新成员**：一位新的社区成员表达了贡献的渴望，强调了他们在全栈工程和研发方面的背景。
   - 他们表示有兴趣与团队联系，以深入了解 **Relign** 项目。
- **关于分布式训练环节的咨询**：一位成员询问是否已经宣布了 **现场分布式训练环节**，表示对即将举行的活动感兴趣。
   - 目前还没有关于该训练环节状态的回复。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1334977564435021885)** (228 条消息🔥🔥): 

> `Deep Research、软银与 OpenAI 的合作、Crystal Intelligence 模型、LLM 生产力影响、Gemini Deep Research 局限性`

- **OpenAI 发布 Deep Research**：OpenAI 的新功能 **Deep Research** 允许用户与 O3 模型交互，提供研究问题的细化建议，并在查询执行期间通过侧边栏显示推理进度。
   - 初步印象强调了 Deep Research 在综合信息方面的潜力，尽管一些用户指出其在深入来源分析方面存在局限性。
- **SoftBank 向 OpenAI 承诺 30 亿美元**：SoftBank 宣布计划每年购买价值 **30 亿美元** 的 OpenAI 产品，同时在日本成立一家专注于 **Crystal Intelligence** 模型的合资企业。
   - 这一独家产品将把 OpenAI 的技术整合到 SoftBank 的子公司中，旨在增强日本企业的 AI 解决方案。
- **Crystal Intelligence 模型发布**：**Crystal Intelligence** 模型旨在自主分析和优化公司过去 30 年的遗留代码，并计划在两年内引入 AGI。
   - 孙正义在发布会上的发言中强调了 AI 的变革潜力，并将其称为 **Super Wisdom**。
- **LLM 对生产力的影响**：用户报告了 LLM 带来的显著生产力提升，表示现在可以完成以前需要数天才能完成的任务，突显了软件开发能力的转变。
   - 然而，人们对误导性信息和局限性表示担忧，特别是对于算法的依赖以及生成内容中来源质量的问题。
- **Gemini Deep Research 的局限性**：**Gemini Deep Research** 的用户注意到它倾向于生成摘要，而不是综合来自多个来源的信息，这限制了其有效性。
   - 在研究过程中，人们还担心会包含来自针对 SEO 优化页面的低质量内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://news.ycombinator.com/item?">未找到标题</a>: 未找到描述</li><li><a href="https://vxtwitter.com/kimmonismus/status/1886429184597381379">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://rlhfbook.com/">(WIP) A Little Bit of Reinforcement Learning from Human Feedback</a>: Reinforcement Learning from Human Feedback (RLHF) 手册</li><li><a href="https://x.com/OpenAI/status/1886149471249264675">来自 OpenAI (@OpenAI) 的推文</a>: Deep Research 东京直播，太平洋时间下午 4 点 / 日本标准时间上午 9 点。请关注直播链接。</li><li><a href="https://x.com/LFAIDataFdn/status/1886432578401456168">来自 LF AI & Data Foundation (@LFAIDataFdn) 的推文</a>: 🚀 很高兴介绍 DeepSpeed，一个来自 @Microsoft 的深度学习优化库！它简化了分布式训练和推理，使 AI 扩展更高效、更具成本效益。了解更多...</li><li><a href="https://x.com/BigComProject/status/1886447888957612512">来自 Computer Intelligence (@BigComProject) 的推文</a>: 今天，我们很高兴宣布 Computer Intelligence Project，这是一个非营利性计划，旨在开发用于计算级任务自动化的多模态代码智能。我们的项目将专注于开发...</li><li><a href="https://en.wikipedia.org/wiki/Self-selection_bias">自选偏倚 - 维基百科</a>: 未找到描述</li><li><a href="https://tenor.com/view/k22p01-gif-7964225">K22p01 GIF - K22P01 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=jv-lpIsnLOo">Deep Research 简介</a>: 日本标准时间上午 9 点 / 太平洋时间下午 4 点开始。加入来自东京的 Mark Chen, Josh Tobin, Neel Ajjarapu 和 Isa Fulford，他们将介绍并演示 Deep Research。</li><li><a href="https://x.com/apples_jimmy/status/1886285036347064461">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: @datapoint2200 ( 他真的是这么说的。我编不出来 )</li><li><a href="https://x.com/teortaxesTex/status/1885553326235783651">来自 Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex) 的推文</a>: 一次性的几何动画测试是一种烦人的时尚，非常脆弱，而且我认为它们能告诉你的关于模型的信息很少。(不是在针对 Stephen，他展示了 R1 也可以完成这项任务，这与之前的说法相反...</li><li><a href="https://x.com/Yuchenj_UW/status/1886215300527579339">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>: @lexfridman @dylan522p @natolambert 正在听，但这张梁文锋的照片是错的.....</li><li><a href="https://x.com/apples_jimmy/status/1886288099669315805">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 启动了一个 1000 人的销售工程师团队。SoftBank 表示将在日本打造 Stargate</li><li><a href="https://fxtwitter.com/btibor91/status/1886508640263397705">来自 Tibor Blaho (@btibor91) 的推文</a>: 据 The Information 报道，SoftBank 已承诺每年购买价值 30 亿美元的 OpenAI 产品，同时成立一家专注于日本的合资企业——SoftBank 将分销 OpenAI 技术...</li><li><a href="https://fxtwitter.com/X_DimensionNews/status/1886292169914449944">来自 X-Dimension (@X_DimensionNews) 的推文</a>: 嘿 Jimmy，我是日本人，我们的媒体说：新的 AI 模型 “Crystal Intelligence” 将专门提供给日本的大型公司。该模型可以自主分析和优化所有系统...</li><li><a href="https://x.com/lmarena_ai/status/1885839541501870295">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: o3-mini 现已在 WebDev Arena 上线，对阵 DeepSeek-R1！“模拟多个小球在旋转的矩形内弹跳。球与球之间以及球与墙壁之间应发生碰撞。”</li><li><a href="https://x.com/MillionInt/status/1886220292214915165">来自 Jerry Tworek (@MillionInt) 的推文</a>: Deep Research 是 OpenAI 外部人员首次可以与 o3 进行交互</li><li><a href="https://x.com/apples_jimmy/status/1886284760118374814">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: Crystal 具有长期记忆</li><li><a href="https://x.com/cognitivecompai/status/1885728318659363134">来自 Eric Hartford (@cognitivecompai) 的推文</a>: 昨天 Gemini 2.0 Flash Thinking 增加了一个新的、有用的功能！也许他们不喜欢 Dolphin-R1 数据集？哎呀，抱歉...</li><li><a href="https://x.com/btibor91/status/1876923634675315100>||">来自 Tibor Blaho (@btibor91) 的推文</a>: 最近几小时内部署了 3 个新的 ChatGPT Web 应用版本——新的自定义指令 UX（“ChatGPT 该如何称呼你？”、“你是做什么的？”、“ChatGPT 应该具备哪些特质？” -...</li><li><a href="https://x.com/apples_jimmy/status/1886284399916003536">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>: 孙正义在演讲中表示，Sam 同意在日本推出 AGI，不到 2 年内推出名为 Crystal Intelligence 的新模型，该模型自主运行，读取你公司拥有的系统的所有源代码...</li><li><a href="https://x.com/apples_jimmy/status/1886284471663837214">来自 Jimmy Apples 🍎/acc (@apples_ji

mmy)</a>: Crystal 参加所有会议，取代呼叫中心</li><li><a href="https://x.com/Yuchenj_UW/status/1885416559029740007">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>: o3-mini 可能是最适合现实物理世界的 LLM。Prompt: &#34;write a python script of a ball bouncing inside a tesseract&#34;</li><li><a href="https://x.com/ericzelikman/status/1882116460920938568">来自 Eric Zelikman (@ericzelikman) 的推文</a>: @Teslanaut</li><li><a href="https://news.ycombinator.com/item?id=42902936">RLHF Book | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/ropirito/status/1886095737169031512">来自 ∿ Ropirito (0commoDTE) (@ropirito) 的推文</a>: 哇，你的 LLM 能写出球在旋转形状中弹跳的脚本，这太疯狂了。也许我们应该试着让它做一些没有 10,000 个现成人类教程的事情</li><li><a href="https://youtu.be/_1f-o0nqpEI?si=DjOLPGlo90506dGs">DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters | Lex Fridman Podcast #459</a>: Dylan Patel 是 SemiAnalysis 的创始人，这是一家专注于半导体、GPU、CPU 和 AI 硬件的研究与分析公司。Nathan Lambert 是一位...</li><li><a href="https://www.youtube.com/live/Gv7torZn5lM?si=DgC9pmZR-Jcpy8fw">直播：OpenAI 创始人 Sam Altman 在东京演讲</a>: 现场观看 OpenAI CEO Sam Altman 在日本东京举行的“通过 AI 转型业务”活动上的演讲，同行的还有 SoftBank CEO 孙正义和 Arm Hold...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入了解一切</a>: 未找到描述</li><li><a href="https://googleapis.github.io/python-genai/index.html#thinking-model">Google Gen AI SDK 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1334978612415168694)** (19 messages🔥): 

> `SmolLM 团队的回应, 人类数据空间探索, Reinforcement Learning 挑战, 使用 HF Accelerate vs. Torchrun` 


- **挑衅 SmolLM 团队**：一位成员幽默地调侃说，烦扰 **SmolLM 团队**可能会促使他们发布更新，而另一位成员则肯定地说，烦人正是他们工作的一部分。
   - *一点点混乱并无大碍*，因为他们在调侃团队动态的兴奋中前行。
- **关于人类数据空间未来的思考**：一位用户询问了 **Reinforcement Learning** 的成功及其在**人类数据空间**上的复制对未来的影响，寻求不同的观点。
   - 这引发了关于 Prompt 的作用和 Agent 接管的讨论，强调模型除了补全之外还需要协助。
- **复杂决策的 Reinforcement Learning**：由于无法应用明确的评分，人们开始担心如何在存在权衡的场景中应用 **Reinforcement Learning**。
   - 一位用户提出，在复杂的规划场景中，可能需要人类对模型响应的反馈。
- **推理过程中的 Token 注入**：一位用户询问是否可以在模型的推理过程中注入 Token，这表明现有信息存在空白。
   - 另一位用户提供了一个资源链接以进一步探索这一概念，促进了对完善模型推理的研究。
- **在 HF Accelerate 和 Torchrun 之间做出选择**：一位成员询问在 **LLM 训练**中更倾向于使用 **HF Accelerate** 还是 `torchrun`，并指出开源仓库中的用法各不相同。
   - 回复强调，虽然 **Accelerate** 对初学者很友好，但构建自定义技术栈时可能需要避免使用它。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1335150363678081024)** (28 messages🔥): 

> `O3-mini System Card 困惑，DeepSeek 的开源影响力，Anthropic 的挑战，Wikipedia 在 AI 中的角色，Jailbreak 进展中的问题` 


- **O3-mini System Card 困惑**：一名成员质疑为什么新下载的 **O3-mini System Card** PDF 不再像之前的版本那样提及 **Codeforces ELO 基准测试**。
   - 这引发了人们对 System Card 文档修订内容的关注。
- **DeepSeek 的开源影响力**：一位知名人士对 **DeepSeek** 表示赞赏，称其 **开源策略** 让大众能够接触到强大的 AI 系统。
   - 该成员强调了**中国的 AI 投入**终于获得认可的重大意义。
- **Anthropic 的挑战引发质疑**：Anthropic 发布了一项**挑战**，要求参与者尝试用一个 Jailbreak 提示词突破八项安全防护，这引发了人们对其参与**动机**的疑虑。
   - 几位成员指出，该挑战目前对潜在参与者缺乏吸引力。
- **关于 Wikipedia 在 AI 中相关性的辩论**：关于 AI 系统是否应该依赖 **Wikipedia** 展开了激烈讨论，一些人认为 AI 应该直接读取其原始来源。
   - 成员们就 Wikipedia 被感知的偏见及其在 AI 训练中的效用所涉及的**对齐问题 (alignment issues)** 发表了看法。
- **Jailbreak UI Bug 曝光**：Jan Leike 透露一个 Bug 允许用户在没有真正破解模型的情况下提升 Jailbreak 等级，并声称目前**没有人超过 3 级**。
   - 这一发现引发了用户关于 Jailbreak 过程有效性和可靠性的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ArmenAgha/status/1886522896077439187">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>：关于 Zetta 发生的事情绝对不是真的。我们真的要公开这里发生的事情吗？引用 Yann LeCun (@ylecun)：你读错了。FAIR 内部多年来一直有多个 LLM 项目...</li><li><a href="https://x.com/SchmidhuberAI/status/1886323742114197525">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：澄清一下，我对 #DeepSeek 实现过去梦想的成就印象深刻。他们的开源策略表明，最强大的大规模 AI 系统可以...</li><li><a href="https://x.com/elder_plinius/status/1886479675439940069">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：@janleike @AnthropicAI 这对我有什么好处？</li><li><a href="https://x.com/elder_plinius/status/1886520062586372224">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：@alexalbert__ @AnthropicAI ggs</li><li><a href="https://x.com/ylecun/status/1886149808500457691">来自 Yann LeCun (@ylecun) 的推文</a>：你读错了。FAIR 内部多年来一直有多个 LLM 项目。有些作为研究原型开源（例如 OPT175B, Galactica, BlenderBot...）。在 2022 年中期，FAIR 启动了一个大型 LLM 项目...</li><li><a href="https://x.com/polynoamial/status/1886508534566883663">来自 Noam Brown (@polynoamial) 的推文</a>：.@OpenAI Deep Research 可能是 Wikipedia 终结的开始，我认为这没问题。我们经常谈论 AI 对齐问题，但让人类达成一致也很难。Wikipedia 是一个伟大的...</li><li><a href="https://x.com/janleike/status/1886480417987158333">来自 Jan Leike (@janleike) 的推文</a>：@elder_plinius @AnthropicAI 你将完全破解我们的防御 ✨</li><li><a href="https://fxtwitter.com/clefourrier/status/1886385835324457143">来自 Clémentine Fourrier 🍊 (🦋 clefourrier.hf.co) (@clefourrier)</a>：嘿 @OpenAI，如果你不提交到私有测试集（PRIVATE TEST SET），你就没有超越 GAIA（顺便说一下，你报告的之前 SOTA 在验证集上的结果是错误的，见表格 - 性能与...相似）</li><li><a href="https://x.com/simonw/status/1885443075146997848">来自 Simon Willison (@simonw) 的推文</a>：有人知道 o3-mini System Card 是怎么回事吗？https://cdn.openai.com/o3-mini-system-card.pdf 上的 PDF 与我几小时前下载的不是同一个文档 - 旧的那个...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1334988739126104197)** (197 messages🔥🔥): 

> `OpenAI 的 O3，Deep Research 性能对比，Research Agent 进展，RLHF 与模型训练，CoT 与 AI 政策`

- **OpenAI 的 O3 通过 Reinforcement Learning 持续改进**：OpenAI 正在将其模型转向更多 Reinforcement Learning (RL) 特性，O3 基于相同的模型，但通过 RL 技术进行了增强。
   - 除了 O3，Operator 和 Deep Research 都经过了 RL 训练，这表明了对该方法的明确关注。
- **Deep Research 在处理繁重任务时表现挣扎**：OpenAI 的 Deep Research 被要求编译关于 R1 大学的详细信息，但最终未能高效完成任务。
   - 虽然 Gemini Deep Research 同样表现挣扎，但用户注意到 OpenAI 的输出感觉更可靠，尽管耗时更长且搜索的网页更少。
- **Research Agent 技术进展**：Richard Socher 宣布即将推出一款先进的 Research Agent，预计其表现将超越 OpenAI 最近的模型，改进预计在一周内发布。
   - 这为 AI Research Agent 之间的竞争性进步奠定了基础，开发者社区对此充满期待。
- **GRPO 证明对 Llama 2 有益**：最近的一项发现强调，GRPO 方法在 Llama 2 7B 模型上表现良好，在 GSM8K 基准测试中实现了显著的准确率提升。
   - 这证明了 Reinforcement Learning 技术在最新模型系列之外的有效性。
- **CoT 增强和 AI 政策查询**：围绕 OpenAI 正在测试的潜在新“CoT 体验”展开了讨论，表明上下文驱动输出的持续改进。
   - 随着用户深入探讨 AI 在学术环境中的影响，这些进展引发了关于大学 AI 政策的对话。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/theinformation/status/1885720683600023786">来自 The Information (@theinformation) 的推文</a>：🚀 OpenAI 正处于增长期：• ChatGPT 在 2024 年达到 1550 万付费订阅用户 • 其模型的企业端采用率增长了 7 倍 • 新推出的每月 200 美元的 Pro 档位年收入已达 3 亿美元。接下来是什么？High...</li><li><a href="https://rlhfbook.com">(WIP) A Little Bit of Reinforcement Learning from Human Feedback</a>：Reinforcement Learning from Human Feedback 手册</li><li><a href="https://bigcode-bench.github.io/">BigCodeBench 排行榜</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2501.19393">s1: Simple test-time scaling</a>：Test-time scaling 是一种极具前景的语言建模新方法，通过在测试时增加计算量来提升性能。最近，OpenAI 的 o1 模型展示了这种能力，但并未公开...</li><li><a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的量化基准。</li><li><a href="https://x.com/JacobColling/status/1886123574810784138">来自 Jake Colling (@JacobColling) 的推文</a>：@simonw 也许可以重新启用格式化？https://platform.openai.com/docs/guides/reasoning#advice-on-prompting</li><li><a href="https://x.com/casper_hansen_/status/1885654322714649001">来自 Casper Hansen (@casper_hansen_) 的推文</a>：下周，我将使用他们的 RLVR 数据集快速训练一个 Tülu 3-Zero</li><li><a href="https://simons.berkeley.edu/workshops/llms-cognitive-science-linguistics-neuroscience/schedule#simons-tabs">日程表</a>：未找到描述</li><li><a href="https://x.com/EpochAIResearch/status/1885421890925162688">来自 Epoch AI (@EpochAIResearch) 的推文</a>：在本周的 Gradient Updates 刊物中，@EgeErdil2 解释了 DeepSeek-R1 训练背后的细节，并估算了在 DeepSeek V 之上进行训练所需的计算成本（美元）...</li><li><a href="https://x.com/sam_paech/status/1885503154462351724">来自 Sam Paech (@sam_paech) 的推文</a>：Judgemark⚖️ 重大更新。结果优先：- 每个评判员在使用评分范围方面存在巨大差异。👀 看看热力图！- 在校准之前，haiku-3.5 胜出 (!)。它最充分地利用了评分范围。- 在...之后...</li><li><a href="https://x.com/Guodaya/status/1886447257370902528">来自 Daya Guo (@Guodaya) 的推文</a>：@davikrehalt 我们也一直在尝试将 R1 应用于 Lean 等形式化证明环境。我们希望尽快为社区带来更好的模型。</li><li><a href="https://x.com/btibor91/status/1886330345945280632">来自 Tibor Blaho (@btibor91) 的推文</a>：ChatGPT 网页版关于 o3-mini “新 CoT 摘要生成器”的公告已更名为 “新 CoT 实验”，并增加了一项新实验，用于控制是否应“显示预览...”</li><li><a href="https://x.com/vikhyatk/status/1886341671820124601">来自 vik (@vikhyatk) 的推文</a>：这看起来像是原始的 CoT...？</li><li><a href="https://x.com/vikhyatk/status/1886348700215284221">来自 vik (@vikhyatk) 的推文</a>：@MaziyarPanahi Token 以极低的延迟流式传输，不像他们之前的摘要生成器。试着找了找，但没看到任何公告</li><li><a href="https://x.com/georgejrjrjr/status/1885814660198535223">来自 George (@georgejrjrjr) 的推文</a>：可怜的 dylan，又被实习生们“干翻”了。另外：这不是在针对 @natolambert，他看起来很诚实且非常有帮助，但那篇关于 v3 的文章可能需要再次更新：我们掌握的最佳证据表明，你的成本估算...</li><li><a href="https://x.com/srush_nlp/status/1886474327618642241">来自 Sasha Rush (@srush_nlp) 的推文</a>：被说服在今天下午做一个关于 DeepSeek 的演讲 https://simons.berkeley.edu/workshops/llms-cognitive-science-linguistics-neuroscience/schedule#simons-tabs 不确定我在这里还有什么新东西可讲...</li><li><a href="https://x.com/max_paperclips/status/1885459527585419397">来自 Shannon Sands (@max_paperclips) 的推文</a>：我给 o3-mini-high 的第一个测试提示词（重新实现 r1 的 GRPO 训练器）也就那样。是的，还是用回 r1 吧。滚吧 OAI</li><li><a href="https://x.com/CrisGiardina/status/1885459572233486390">来自 Cristiano Giardina (@CrisGiardina) 的推文</a>：OpenAI 正在努力为 o3 / 他们的推理模型展示更多的思维 CoT Token。@kevinweil 表示“展示所有 CoT 会导致竞争性蒸馏 [...] 但高级用户想要它，” @sam...</li><li><a href="https://x.com/natolambert/status/1886099004746080724">来自 Nathan Lambert (@natolambert) 的推文</a>：订阅 @interconnectsai，免费获取我为了跟上推理结果所做的疯狂尝试</li><li><a href="https://fxtwitter.com/OfirPress/status/1886399992815923213">来自 Ofir Press (@OfirPress) 的推文</a>：祝贺 o3-mini 在 SciCode 上创下新高！R1 的得分达到了令人印象深刻的 4.6%，与 Claude 3.5 持平。SciCode 是我们由各学科博士编写的超难编程基准测试。</li>

<li><a href="https://x.com/littmath/status/1885566677384863866">来自 Daniel Litt (@littmath) 的推文</a>：首先，它显然比 o1 有了显著的提升。它立即（非严谨地）解决了我向它提出的一些带有数值答案的算术几何问题，而其他模型都无法……</li><li><a href="https://x.com/goodside/status/1885950395056370090">来自 Riley Goodside (@goodside) 的推文</a>：正在与 o3-mini 进行头脑风暴——进展顺利，粘贴了一些代码来解释 AidenBench。它询问了某个函数。我粘贴了文件。政策错误（Policy error）。我说，这是个误会。政策错误。我回去编辑……</li><li><a href="https://x.com/richardsocher/status/1886402401319432353?s=46">来自 Richard Socher (@RichardSocher) 的推文</a>：既然 OpenAI 已经赶上了我们一年多前的旧研究 Agent，我们很快将推出我们的高级研究和智能 Agent，它将比我们现有的以及 OpenAI 的产品好 10 倍……</li><li><a href="https://x.com/teortaxesTex/status/1886526422493143268">来自 Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex) 的推文</a>：DeepSeek V3 类型的模型现在可以在华为昇腾（Huawei Ascend）上进行训练。</li><li><a href="https://x.com/rdolmedo_/status/1886505669622149139">来自 Ricardo Dominguez-Olmedo (@rdolmedo_) 的推文</a>：带有可验证奖励的强化学习（Reinforcement Learning）是否仅适用于最近的模型家族？事实证明，GRPO 对 Llama 2 7B 也非常有效，在 GSM... 中实现了令人印象深刻的 +15 个准确率百分点提升。</li><li><a href="https://x.com/btibor91/status/1885750898200326336">来自 Tibor Blaho (@btibor91) 的推文</a>：据 The Information 报道，OpenAI 告诉部分股东，ChatGPT 付费订阅用户在 2024 年从 580 万增加到 1550 万，而据知情人士透露……</li><li><a href="https://x.com/vllm_project/status/1885837174588989695?s=61">来自 vLLM (@vllm_project) 的推文</a>：我们上线了针对 @deepseek_ai 模型的第一批增强功能，首先是 MLA 和 cutlass fp8 内核。与 v0.7.0 相比，我们提供了约 3 倍的生成吞吐量，约 10 倍的 Token 内存容量……</li><li><a href="https://x.com/karpathy/status/1886192184808149383">来自 Andrej Karpathy (@karpathy) 的推文</a>：有一种我称之为“vibe coding”的新型编程方式，在这种方式中，你完全沉浸在氛围中，拥抱指数级增长，甚至忘记了代码的存在。这之所以成为可能，是因为 LLM（例如……</li><li><a href="https://x.com/TheXeophon/status/1886428233287208973">来自 Xeophon (@TheXeophon) 的推文</a>：我喜欢这个例子，所有 VLM 在计数时都错得离谱，包括 Gemini。如果你指示 Qwen2.5-VL 非常非常努力地计数，它每次都能成功。否则它会差 1。现在这是一个 Qwe...</li><li><a href="https://x.com/MaitrixOrg/status/1885387184557199857">来自 Maitrix.org (@MaitrixOrg) 的推文</a>：🚀使用 LLM Reasoners 解锁高效的推理时间扩展（inference-time scaling）——现在由 SGLang 驱动！ > 🔥比之前版本的 LLM Reasoners（配合 Hugging Face）加速 100 倍 > 只需一行代码……</li><li><a href="https://x.com/tsarnick/status/1885457829974466595">来自 Tsarathustra (@tsarnick) 的推文</a>：Sam Altman：“在开源/开放权重 AI 模型方面，我们一直站在历史错误的一边”</li><li><a href="https://x.com/littmath/status/1885566673844842739">来自 Daniel Litt (@littmath) 的推文</a>：关于使用 o3-mini-high（OpenAI 今天发布的全新推理模型）进行数学用途的一些简短印象。</li><li><a href="https://x.com/charliermarsh/status/1885516807164842010?s=61">来自 Charlie Marsh (@charliermarsh) 的推文</a>：重大但非常小众的新闻：Flash Attention (flash-attn) 现在使用 Metadata 2.2，这意味着 uv 可以在不从源码构建包的情况下解析它。</li><li><a href="https://fxtwitter.com/lexfridman/status/1885435220502991193">来自 Lex Fridman (@lexfridman) 的推文</a>：OpenAI o3-mini 是一个很好的模型，但 DeepSeek R1 性能相近，价格更便宜，并且会展示其推理过程。更好的模型将会出现（迫不及待想看 o3pro），但“DeepSeek 时刻”……</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/comment/maa0dcx">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/Deep-Agent/R1-V">GitHub - Deep-Agent/R1-V：以不到 3 美元的成本见证 VLM 的顿悟时刻（aha moment）。</a>：以不到 3 美元的成本见证 VLM 的顿悟时刻。通过在 GitHub 上创建账户，为 Deep-Agent/R1-V 的开发做出贡献。</li><li><a href="https://x.com/sama/status/1885601623625331162">来自 Sam Altman (@sama) 的推文</a>：很快还会有一个 o3-mini 的好东西带给大家——我想我们把最好的留到了最后！</li><li><a href="https://chat.qwenlm.ai/s/b38028bc-94f0-4c42-89fd-723970d5fb60">Qwen Chat</a>：未找到描述</li>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1335011872516214836)** (32 条消息🔥): 

> `HF_ENABLE_FAST_TRANSFER, Bengali Ghosthunters, TechCrunch's Meme Game, Economic Value Charts, RLHF vs Reasoning Models` 


- **HF_ENABLE_FAST_TRANSFER 提升效率**：一位成员强调了使用 `HF_ENABLE_FAST_TRANSFER` 的效果，据报道它能将 *HF 生态系统的效能提高三倍*。
   - 随后讨论了大文件存储的默认传输速度，有成员表示担心速度似乎较慢。
- **Bengali Ghosthunters 成为焦点**：[Bengali Ghosthunters](https://x.com/qwrk8126/status/1884399348504748149) 引发了笑料，一位成员讲述了 Gemini Flash Thinking 在帮助他学习 LLM 时变得行为异常的经历。
   - 这一话题引发了对技术与幽默体验之间联系的进一步探索和兴趣。
- **TechCrunch 梗图掀起热潮**：一个有趣的反应被记录下来，*TechCrunch 的标题* 受到称赞，一位成员评论说他们像冠军一样在 X 上发布梗图。
   - 另一位成员开玩笑地建议，*现代数学课* 导致了贡献者中广泛出现的 rosette（玫瑰结）。
- **“预估经济价值”图表引起轰动**：一位成员对新的“预估经济价值”图表表示兴奋，称这些图表肯定会让那些喜欢模棱两可的 test time compute 对数图的粉丝们感兴趣。
   - 反应从幽默的怀疑到对这些洞见将如何呈现（类似于 pitch decks 融资演示文稿）的兴奋不等。
- **关于 RLHF 和推理模型的辩论**：围绕 RLHF 出现了一种强烈的观点，一位成员断言，尽管推理模型兴起，RLHF 仍然是流水线（pipeline）中至关重要的一部分。
   - 这引发了热烈的讨论，强调 RLHF 和推理训练都是更广泛的 post-training 策略的组成部分。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/qwrk8126/status/1884399348504748149">来自 sholín (NOAM CHOMSKY SIGUE VIVO) (@qwrk8126) 的推文</a>: Gemini Flash Thinking Exp 2.0 0121 正在教我更多关于 LLM 的技术本质，并为我准备了一个简短的多选题测验以提供正确答案。在我完成后，它停止了思考...</li><li><a href="https://x.com/edwinarbus/status/1885464407104205249">来自 edwin (@edwinarbus) 的推文</a>: 未找到描述</li><li><a href="https://x.com/CFGeek/status/1886113291023659198">来自 Charles Foster (@CFGeek) 的推文</a>: “惨痛的教训是，我们只需要将奖励函数重新更名为验证器”——Rich Sutton，大概吧</li><li><a href="https://x.com/colin_fraser/status/1885584619661103506?s=46">来自 Colin Fraser (@colin_fraser) 的推文</a>: 这让我笑出声来</li><li><a href="https://x.com/arankomatsuzaki/status/1885780409365262653">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>: 未找到描述</li><li><a href="https://x.com/wordgrammer/status/1886278047977934872">来自 wordgrammer (@wordgrammer) 的推文</a>: Anthropic 花费 500 万美元训练 Claude，然后花费 9.95 亿美元对其进行 redteaming。OpenAI 花费 500 万美元训练 o3，然后花费 9.95 亿美元在某些数学基准测试的 test-time compute 上。DeepSeek 花费 500 万美元训练，然后...</li><li><a href="https://x.com/TheXeophon/status/1885834749807128735">来自 Xeophon (@TheXeophon) 的推文</a>: 我就知道。引用 Nathan Lambert (@natolambert)：太多人认为，因为推理模型已经起飞，并且以带有可验证奖励的强化学习为核心要素，RLHF 就...</li><li><a href="https://x.com/colin_fraser/status/1885742511324274947">来自 Colin Fraser (@colin_fraser) 的推文</a>: 这个长达 202 秒的 CoT 中确实有一些亮点</li><li><a href="https://x.com/_xjdr/status/1886220578966966429">来自 xjdr (@_xjdr) 的推文</a>: 如果你喜欢那些模棱两可的 test time compute 对数图，你一定会喜欢新的“预估经济价值”图表</li><li><a href="https://github.com/huggingface/hf_transfer">GitHub - huggingface/hf_transfer</a>: 通过在 GitHub 上创建账户来为 huggingface/hf_transfer 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1334977480825770054)** (7 条消息): 

> `资金问题、GRPO 与 RLVR、Demos、DeepSeek` 


- **资金挑战导致项目关闭**：一位成员对 AI2 作为一个支持性的开源社区表示感谢，同时分享了由于**缺乏资金**和**个人健康**问题而不得不关闭其项目的遗憾消息。
   - 他们强调了在当前环境下，像 AI2 这样的平台对于开源工作的重要性。
- **关于 GRPO 在 RLVR 中作用的提问**：一位成员提出了一个问题，即 **GRPO** 对于 **RLVR** 是否必不可少，或者是否可以用 **DPO** 代替。
   - 另一位成员插话表示，虽然可以使用 **DPO**，但其效果可能比使用 **GRPO** **差一些**。
- **希望 AI2 提供更好的 Demo**：一位成员表示希望 **AI2** 能改进其 **demo** 展示，并建议更好的展示可以增强社区的推广效果。
   - 然而，他们也承认，不专注于快速见效的项目可以留出更多心理空间去追求**更大的胜利**。
- **关于 DeepSeek 机制的讨论**：一位成员寻求澄清，想知道 **GRPO** 是 **DeepSeek 魔力**的关键所在，还是仅仅是一个**实现细节**。
   - 回复指出，GRPO 可能不是必需的，因为可以应用 DPO，尽管效果会有所下降。



**提到的链接**：<a href="https://bsky.app/profile/dorialexander.bsky.social/post/3lh767gswu22q">Alexander Doria (@dorialexander.bsky.social)</a>：如果有人感兴趣，我成功在 Colab 中设置了一个 GRPO RL 训练的 demo。它是对 Will Brown 关于数学推理的即时经典作品的改编。将 llama 1B 替换为 qwen 0.5b 并进行推理...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1335450173140369480)** (10 条消息🔥): 

> `DeepSeek AI R1 模型、关于 AI 是否为科学的讨论、AI 中的思考模型、关于 Post-training 的 NeurIPs 演讲、R1 训练参数` 


- **DeepSeek AI 发布旗舰级 R1 模型**：1 月 20 日，中国的 DeepSeek AI 发布了他们的第一个成熟的推理模型——[R1](https://huggingface.co/d) 模型。
   - 该模型的特点是专注于使用更多数据进行更长时间的训练，引发了社区对推理模型的兴奋。
- **AI 是一门科学吗？深度探讨**：[The Retort](https://retortai.com/episodes/we-ask-again-is-ai-a-science) 的主持人讨论了 AI 是否符合科学的标准，并引用了**库恩（Kuhn'ian）**的视角。
   - 这场辩论突显了关于科学学科与 AI 之间关系的持续哲学讨论。
- **探索 AI 中的思考模型**：一位嘉宾出现在播客中，讨论了“思考模型”的概念及其与 Post-training 和推理方法的交集，内容可见[此处](https://www.aisummer.org/p/nathan-lambert-on-the-rise-of-thinking)。
   - 讨论强调了 AI 方法论的演变，以及它们如何区分各种模型训练方法。
- **NeurIPs 关于 Post-training 的演讲公开**：最近在 NeurIPs 上发表的一场关于 AI 应用的 **Post-training** 策略的演讲现已公开，观看地址在[此处](https://youtu.be/grpc-Wyy-Zg)。
   - 分享的见解旨在指导 AI 从业者优化其训练周期以获得更好的结果。
- **R1 训练的简洁性得到认可**：训练 R1 模型涉及提供更多数据并延长训练时间，且在 Post-training 周期早期优先进行排序。
   - 这种直接的方法因其简洁性而得到了轻松的认可，激发了人们对多样化推理模型的热情。



**提到的链接**：<a href="https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1">DeepSeek R1 复现 o1 的配方以及推理语言模型的未来</a>：是的，为 DeepSeek R1 敲响真正的 o1 复现之钟 🔔🔔🔔。接下来我们将走向何方。

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1335018922164093000)** (8 条消息🔥): 

> `Creator Gravity, AI Self-Assessment, Rejection in Writing Jobs, Sander Land's Substack Commentary` 


- **Sander Land 的幽默评论**：一位成员分享了一篇讨论 Tokenization 概念的 *幽默 Substack 文章*，并评论道“竟然有人关注这玩意儿？”，同时分享了 [文章链接](https://tokencontributions.substack.com/p/whole-words-and-claude-tokenization/comments)。
   - 讨论中流露出对内容的怀疑与娱乐心态，暗示了 AI 讨论中批判性趋势的增长。
- **Creator Gravity 讨论**：一位成员在分享关于 [Creator Gravity](https://open.substack.com/pub/internetly/p/wtf-is-creator-gravity?r=68gy5&utm_medium=ios) 的见解时，表达了对重复收到拒绝邮件的沮丧。
   - “如果没有人雇佣我，我就雇佣我自己”，这句话反映了创意社区内的坚定决心。
- **Rich Sutton 谈 AI 自我验证**：提到 Rich Sutton 的文章 [Key to AI](http://incompleteideas.net/IncIdeas/KeytoAI.html) 引发了关于 AI 自我评估能力的讨论。他认为，AI 验证自身表现的能力对于成功运行至关重要。
   - 一位成员幽默地回应，将 Sutton 的见解称为 *boomer reference*（婴儿潮一代的引用），指出了在 AI 发展认知上的代际差异。
- **文章发现**：成员分享的链接是对之前讨论的回应，强调了社区内分享想法的互联性。另一位成员评论了这类文章在 Twitter 等平台上的传播是多么容易。
   - 这种互动突显了群组中盛行的信息和观点的动态交流。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://incompleteideas.net/IncIdeas/KeytoAI.html">Self-Verification, The Key to AI</a>: 未找到描述</li><li><a href="https://open.substack.com/pub/internetly/p/wtf-is-creator-gravity?r=68gy5&utm_medium=ios">WTF is Creator Gravity? </a>: 在网络上变得具有吸引力的艺术与科学——以及为什么它与传统影响力无关。</li><li><a href="https://tokencontributions.substack.com/p/whole-words-and-claude-tokenization/comments">Whole words and Claude tokenization</a>: 使用新的计数端点显示出……对完整单词的偏好？
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1335263155605078057)** (23 messages🔥): 

> `拟议的 AI 立法、影子图书馆与 AI、富士康关税、AI 研究合作限制` 


- **国会激进的 AI 法案威胁开源**：共和党参议员提议的一项法案旨在禁止从中国进口 AI 技术，可能包括下载如 [DeepSeek](https://x.com/opensauceAI/status/1885483641649979639) 的模型权重，处罚最高可达 **20 年监禁**。
   - 该法案还禁止向“受关注实体”出口 AI，将发布如 **Llama 4** 等产品的行为等同视之，并处以类似处罚。
- **可能将研究合作定为犯罪的法律**：该法案可能使美国公民与清华大学等机构共同撰写机器学习论文成为犯罪，引发了对学术自由的担忧。
   - 批评者认为，这使关于 AI 国际合作的讨论走向了一个危险的方向。
- **富士康出货与贸易关税**：报告显示，在特朗普计划对加拿大和墨西哥征收关税后，所有发往美国的 **Foxconn GB200** 订单将从墨西哥发货，这可能会影响 GPU 的供应。
   - 在持续的供应链担忧中，这种情况对大型数据中心的建设产生了影响。
- **AI 版权与影子图书馆讨论**：人们对使用 **Z-Library** 等非法存档训练中国 LLM 表示担忧，强调需要从国家安全的高度对版权法进行彻底改革。
   - 专家指出，解决这些问题对于保护知识产权和开源发展具有紧迫性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">版权改革对国家安全至关重要</a>：中国 LLM（包括 DeepSeek）是在我的非法书籍和论文存档上训练的——这是全球最大的存档。西方需要从国家安全的高度改革版权法。</li><li><a href="https://x.com/opensauceAI/status/1885483639611531704">来自 Ben Brooks (@opensauceAI) 的推文</a>：哇。国会刚刚提出了一项法案，将 *真正* 杀死开源。这无疑是针对 AI 最激进的立法行动——而且是由曾因 Llama 抨击 @finkd 的共和党参议员提议的...</li><li><a href="https://x.com/aviskowron/status/1885636578309021701">来自 Aviya Skowron (@aviskowron) 的推文</a>：该法案还将使美国公民与清华大学的人员共同撰写机器学习论文成为犯罪。你可能觉得我在夸大其词，但这里有新闻稿和...</li><li><a href="https://x.com/kakashiii111/status/1884443413568971248">来自 Kakashii (@kakashiii111) 的推文</a>：你们都知道所有发往美国的富士康 GB200 订单都计划从墨西哥发货，对吧？引用 unusual_whales (@unusual_whales) 突发：白宫表示特朗普计划履行...</li><li><a href="https://x.com/opensauceAI/status/1885483641649979639">来自 Ben Brooks (@opensauceAI) 的推文</a>：1. 该法案将禁止进口在中国开发的 AI “技术或知识产权”。可以想象，这将包括下载 @deepseek_ai R1 / V3 权重。处罚：最高 20 年...</li><li><a href="https://x.com/opensauceAI/status/1885483645064142915">来自 Ben Brooks (@opensauceAI) 的推文</a>：2. 该法案还将禁止向“受关注实体”出口 AI。出口意味着传输到美国境外，或发布给在美国的外国人。例如：发布 Llama 4。处罚？同样是...</li><li><a href="https://www.bis.gov/ear/title-15/subtitle-b/chapter-vii/subchapter-c/part-740/ss-74027-license-exception-artificial">§ 740.27 许可证例外人工智能授权 (AIA)。 | 工业和安全局</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1334987779393978500)** (181 messages🔥🔥): 

> `Deep Research 发布、OpenAI Agent 讨论、AI 模型发展、LLM 竞争与内部冲突、推理增强生成 (ReAG)`

- **OpenAI 发布 Deep Research**：OpenAI 推出了 Deep Research，这是一款针对网页浏览和复杂推理优化的 autonomous agent，承诺能在几分钟内综合各种来源生成详尽的报告。
   - 早期反馈表明它是一个强大的电子商务工具，尽管一些用户反映其输出质量存在局限性。
- **模型访问权限引发混淆**：用户对在移动设备上访问 Deep Research 感到困惑，许多人指出目前似乎仅限于桌面端使用。
   - 一些用户对不同订阅层级之间的访问差异表示担忧。
- **AI 模型竞争与内部争议**：Yann LeCun 强调了 FAIR 内部的竞争，对比了 Zetta 和 Llama-1 的开发路径，并表示小团队的表现往往优于大型项目。
   - 这引发了关于此类动态对当前 AI 发展影响的讨论，特别是在 DeepSeek 与传统厂商竞争的背景下。
- **推理增强生成 (ReAG) 的引入**：ReAG 旨在改进传统的 Retrieval-Augmented Generation，通过消除检索步骤并直接将原始材料提供给 LLMs 进行综合。
   - 初步反应表明了其潜在的有效性，但也引发了对可扩展性和文档预处理必要性的担忧。
- **社区参与和反馈**：用户积极为 Deep Research 等 AI 模型众包想法和测试，展示了社区对改进与这些技术交互的兴趣。
   - 参与者正在分享经验、发现并参与讨论，反映了 AI 发展领域充满活力且不断演变的格局。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/cursor_ai/status/1885415392677675337">来自 Cursor (@cursor_ai) 的推文</a>：o3-mini 已向所有 Cursor 用户开放！我们目前免费提供该模型，以便让大家体验。Cursor 开发者在大多数任务中仍然更倾向于使用 Sonnet，这让我们感到意外。</li><li><a href="https://x.com/polynoamial/status/1886508534566883663">来自 Noam Brown (@polynoamial) 的推文</a>：OpenAI Deep Research 可能标志着 Wikipedia 终结的开始，我认为这没问题。我们经常讨论 AI 对齐（alignment）问题，但让人类达成一致也很难。Wikipedia 是一个伟大的...</li><li><a href="https://x.com/Teknium1/status/1885485234533392585">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：让我给你们讲个小故事。大约一年前，我联系了一位不愿透露姓名的 OpenAI 员工，他曾暗示他们非常有兴趣做一些开源的事情...</li><li><a href="https://x.com/tsarnick/status/1885457829974466595">来自 Tsarathustra (@tsarnick) 的推文</a>：Sam Altman：“在开源/开放权重 AI 模型方面，我们一直站在了历史错误的一边”</li><li><a href="https://www.wired.com/story/openai-deepseek-stargate-sam-altman/">DeepSeek 激发了 OpenAI 的斗志</a>：在一家中国初创公司扰动行业之后，OpenAI 提前准备好了应对方案。</li><li><a href="https://x.com/ocolegro/status/1886491097716961635?s=46">来自 Owen Colegrove (@ocolegro) 的推文</a>：我询问了 OpenAI Deep Research 什么是 R1 以及如何小规模复制它——结果非常出色（并非炒作）——这是自初代 ChatGPT 以来第一件让我印象如此深刻的事情...</li><li><a href="https://www.latent.space/p/karina">Agent 推理接口：o1/o3、Claude 3、ChatGPT Canvas、Tasks 和 Operator —— 与 OpenAI 的 Karina Nguyen 对谈</a>：来自 OpenAI（此前在 Anthropic）的 Karina Nguyen 讨论了她在 Claude、ChatGPT Canvas 和 Tasks 方面的工作，以及人机协作的新型 AI 交互范式。</li><li><a href="https://x.com/ianand/status/1885467953979940895?s=46">来自 Ishan Anand (@ianand) 的推文</a>：ArrrZero：为什么 DeepSeek R1 不如 R1-Zero 重要。虽然大家都在讨论 DeepSeek R1，但真正的游戏规则改变者是 R1-Zero。在这段视频中，我介绍了这个模型是如何直接从基础模型...</li><li><a href="https://x.com/namangoyal21/status/1886515845133951192?s=46">来自 Naman Goyal (@NamanGoyal21) 的推文</a>：作为唯一一位同时参与过 OPT 和 llama1 的共同作者，并且曾是 zetta 团队的一员，我可以说明这其实要复杂得多，涉及多个视角，并非一个简单的故事...</li><li><a href="https://x.com/polynoamial/status/1886223995877339568?s=46">来自 Noam Brown (@polynoamial) 的推文</a>：o1 发布不到 2 个月。o3-mini 2 天前发布。Deep Research 今天发布。它是一个强大的工具，我迫不及待想看看世界会用它做些什么，但 AI 将继续...</li><li><a href="https://x.com/ianand/status/1885467953979940895?s=">来自 Ishan Anand (@ianand) 的推文</a>：ArrrZero：为什么 DeepSeek R1 不如 R1-Zero 重要。虽然大家都在讨论 DeepSeek R1，但真正的游戏规则改变者是 R1-Zero。在这段视频中，我介绍了这个模型是如何直接从基础模型...</li><li><a href="https://x.com/kimmonismus/status/1885457297230516224">来自 Chubby♨️ (@kimmonismus) 的推文</a>：OpenAI 团队在 Reddit 上的 AMA 精华帖 🧵 第 1 条：递归自我改进（recursive self-improvement）可能会导致智能剧烈爆发（hard take off）</li><li><a href="https://www.superagent.sh/blog/reag-reasoning-augmented-generation">ReAG：推理增强生成（Reasoning-Augmented Generation） - Superagent</a>：Superagent 是一个拥有 AI Agent 的工作空间，这些 Agent 可以学习、执行工作并进行协作。</li><li><a href="https://x.com/sama/status/1885512348234113243?s=46">来自 Sam Altman (@sama) 的推文</a>：抱歉我搞错了。我以为今天就会发布，但那部分很快就会推出！</li><li><a href="https://x.com/hwchung27/status/1886221344662299022?s=46">来自 Hyung Won Chung (@hwchung27) 的推文</a>：很高兴分享 Deep Research，我们全新的 Agent 模型！Deep Research 的一个显著特点是其极度的耐心。我认为这正迅速接近“超人类的耐心”。在工作中意识到...</li><li><a href="https://x.com/openai/status/1886219087627612504?s=46">来自 OpenAI (@OpenAI) 的推文</a>：由针对网页浏览和 Python 分析优化的 OpenAI o3 版本驱动，Deep Research 利用推理能力智能且广泛地浏览互联网上的文本、图像和 PDF。</li><li><a href="https://x.com/afinetheorem/status/1886206439582015870?s=46">来自 Kevin A. Bryan (@Afinetheorem) 的推文</a>：今天发布的 OpenAI 新模型非常疯狂。它本质上是 Google 的 Deep Research 理念，结合了多步推理、网页搜索，*并且* 底层使用了 o3 模型（据我所知）。它有时...</li><li><a href="https://x.com/sama/status/1886221586002489634">来自 Sam Altman 的推文</a>

Altman (@sama)</a>: （注意：这还不是 o3-mini 的“one-more-thing”。还要再等几天。）</li><li><a href="https://x.com/hxiao/status/1886250705415229627?s=46">来自 Han Xiao (@hxiao) 的推文</a>：OpenAI 的 Deep Research 只是在一个 while 循环中进行搜索+阅读+推理，对吧？除非我遗漏了什么，否则这就是我在 nodejs 中使用 gemini-flash 和 jina reader 实现的复刻版本 https...</li><li><a href="https://x.com/nikunjhanda/status/1885410924879839356?s=46">来自 Nikunj Handa (@nikunjhanda) 的推文</a>：o3-mini 是我们迄今为止发布的功能最完整且对开发者最友好的 o-series 模型：支持 function calling, structured outputs, streaming, batch, assistants！它还具备：1. 90% 以上的 c...</li><li><a href="https://x.com/distributionat/status/1886238792870461451?s=46">来自 thomas (@distributionat) 的推文</a>：如果你每次查询能获得 500 美元的价值，那么在 Pro 订阅的第一个月，通过 100 次查询你就能净赚 49,800 美元。到了第二个月，你可以购买 249 个新订阅，净赚 12,450,000 美元。到第 m...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_ma">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/mckaywrigley/status/1886215847481623030?s=46">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>：OpenAI 的 Deep Research 预览版的关键点在于，他们在长期规划（longterm planning）和 tool calling 方面取得了重大进展。这就是你获得“虚拟协作伙伴”的方式。Agent 时代即将来临。</li><li><a href="https://x.com/sama/status/1886220904088162729?s=46">来自 Sam Altman (@sama) 的推文</a>：祝贺团队，特别是 @isafulf 和 @EdwardSun0909，打造了一款令人惊叹的产品。我非常粗略的感觉是，它可以完成所有具有经济价值的任务中个位数百分比的工作...</li><li><a href="https://x.com/edwardsun0909/status/1886216911777919257?s=46">来自 Zhiqing Sun (@EdwardSun0909) 的推文</a>：很高兴终于能分享我自去年 6 月加入 OpenAI 以来一直在做的工作！Deep Research 的目标是让推理模型能够利用工具来解决现实世界中的长程任务（long-horizon tasks）并...</li><li><a href="https://x.com/danhendrycks/status/1886207504037945462?s=46">来自 Dan Hendrycks (@DanHendrycks) 的推文</a>：看起来最新的 OpenAI 模型在许多主题上表现都非常好。我猜 Deep Research 特别有助于包括医学、古典学和法律在内的学科。</li><li><a href="https://x.com/pelaseyed/status/1886448015533089248">来自 homanp (@pelaseyed) 的推文</a>：传统的 RAG 很糟糕，因为它承诺提供“相关块（relevant chunks）”，但实际上返回的是“相似块（similar chunks）”。相关性需要推理。介绍 ReAG —— 推理增强生成（Reasoning Augmented Generation）</li><li><a href="https://x.com/gdb/status/1886229270428848399?s=46">来自 Greg Brockman (@gdb) 的推文</a>：Deep Research 是一个极其简单的 Agent —— 一个可以浏览网页并执行 Python 代码的 o3 模型 —— 并且已经非常有用。令人大开眼界的是，OpenAI 已经有很多人在使...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1ieonxv/ama_with_openais_sam_altman_mark_chen_kevin_weil/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/armenagha/status/1886522896077439187?s=46">来自 Armen Aghajanyan (@ArmenAgha) 的推文</a>：关于 Zetta 发生的事情绝对不是真的。我们真的要公开这里发生的事情吗？引用 Yann LeCun (@ylecun)：你读错了。Meta 内部曾有多个 LLM 项目...</li><li><a href="https://news.ycombinator.com/item?id=42902936">RLHF Book | Hacker News</a>：未找到描述</li><li><a href="https://x.com/clefourrier/status/1886385835324457143?s=46">来自 Clémentine Fourrier 🍊 (🦋 clefourrier.hf.co) (@clefourrier)</a>：嘿 @OpenAI，如果你不提交到私有测试集（PRIVATE TEST SET），你就没有超越 GAIA（顺便说一下，你报告的之前 SOTA（在验证集上）的结果是错误的，见表格 —— 性能类似于...</li><li><a href="https://x.com/danshipper/status/1886203397004783996?s=46">来自 Dan Shipper 📧 (@danshipper)</a>：OpenAI 刚刚推出了自主研究助手 Deep Research。我们在 @Every 已经测试了几天，它就像是为好奇心准备的火箭筒：- 给它一个问题，它就会...</li><li><a href="https://x.com/alexalbert__/status/1886461372223074412?s=46">来自 Alex Albert (@alexalbert__)</a>：在 Anthropic，我们正在为强大的 AI 系统的到来做准备。基于我们关于 Constitutional Classifiers 的最新研究，我们开发了一个演示应用来测试新的安全技术。我们想...</li><li><a href="https://x.com/IterIntellectus/status/1886417619990802826/photo/1">来自 vittorio (@IterIntellectus) 的推文</a>：呃...</li><li><a href="https://x.com/janleike/status/1886452697425137904">来自 Jan Leike (@janleike) 的推文</a>：我们挑战你来破解我们新的越狱（jailbreaking）防御！共有 8 个等级...</li>

<ul>
<li>ls. 你能找到一个击败所有人的单一 jailbreak 吗？https://claude.ai/constitutional-classifiers</li><li><a href="https://x.com/_jasonwei/status/1886213911906504950?s=46">来自 Jason Wei (@_jasonwei) 的推文</a>：非常激动终于能分享 OpenAI 的 “deep research” 模型，它在 Humanity's Last Exam 上的得分是 o3-mini 的两倍，甚至可以执行一些需要 PhD 经验的任务...</li><li><a href="https://x.com/natolambert/status/1886214346893885951?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：很高兴能与 @lexfridman 以及我的哥们 @dylan522p 聊了 5 个多小时，试图深入探讨目前 AI 领域究竟在发生什么。DeepSeek R1 & V3，中国 vs 美国，开源 vs 闭源，减少...</li><li><a href="https://x.com/emollick/status/1886205847803429173?s=46">来自 Ethan Mollick (@emollick) 的推文</a>：OpenAI 的 deep research 非常出色。与 Google 的版本（主要是多源摘要器）不同，OpenAI 更像是在聘请一位有主见的（通常几乎是 PhD 级别！）研究员，他会追踪线索。L...</li><li><a href="https://x.com/sherwinwu/status/1886256203077915071?s=46">来自 Sherwin Wu (@sherwinwu) 的推文</a>：o3（完整版，而非 mini）首次向 OpenAI 以外的用户开放——并且它被包装在一个非常流畅的产品体验中。引用 OpenAI (@OpenAI)：由针对 ... 优化的 OpenAI o3 版本驱动</li><li><a href="https://x.com/ai_for_success/status/1886225225098109376?t=NiIx9IstTXR_OtixvI9yOw&s=19">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>：OpenAI 发布了 Deep Research，而 ChatGPT Plus 用户再次被冷落了。感觉 OpenAI 对待 Plus 用户比免费层级还要差。</li><li><a href="https://x.com/emollick/status/1886205847803429173">来自 Ethan Mollick (@emollick) 的推文</a>：OpenAI 的 deep research 非常出色。与 Google 的版本（主要是多源摘要器）不同，OpenAI 更像是在聘请一位有主见的（通常几乎是 PhD 级别！）研究员，他会追踪线索。L...</li><li><a href="https://marginalrevolution.com/marginalrevolution/2025/02/o1-pro.html">o1 pro - Marginal REVOLUTION</a>：通常我不写特定的帖子，因为我觉得这对每个人来说都是显而易见的。然而事实很少如此。所以这是我关于 o1 pro 的帖子，随后很快会有 o3 pro，而 Deep Research 正在分发...</li><li><a href="https://docs.google.com/document/d/1JpVXX9EmgjPVZLPEXmlBzSDdRuQVmCsIjBi_pyp3xS4/edit?usp=sharing">巴西 Zouk 动作列表请求</a>：巴西 Zouk 动作与移动。起源于 1990 年代巴西充满活力的舞蹈场景，巴西 Zouk 是一种迷人的双人舞，以其感性和表现力风靡全球...</li><li><a href="https://x.com/OpenAI/status/1886149471249264675?t=0O8ujtyOO">来自 OpenAI (@OpenAI) 的推文</a>：Deep Research 东京现场直播，太平洋时间下午 4 点 / 日本标准时间上午 9 点。请关注直播链接。</li><li><a href="https://x.com/OpenAI/status/1886149471249264675?t=0O8ujtyOOzkt3VZ6dk_alg&s=19">来自 OpenAI (@OpenAI) 的推文</a>：Deep Research 东京现场直播，太平洋时间下午 4 点 / 日本标准时间上午 9 点。请关注直播链接。</li><li><a href="https://x.com/sama/status/1886220051092512979?t=Yos9UQnWV_biDiPXfcC1_g&s=19">来自 Sam Altman (@sama) 的推文</a>：它非常耗费计算资源且速度较慢，但它是第一个能够完成如此多样化、复杂且有价值任务的 AI 系统。现在已在我们的 Pro 层级上线，每月 100 次查询。此外，Team...</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1335062963966705664)** (2 条消息): 

> `AI Engineer Summit, Karina Nguyen Keynote, New Online Track` 


- **AI Engineer Summit 门票火热发售中**：[AI Engineer Summit](https://www.latent.space/p/2025-summit) 的赞助和门票正在快速售罄，该活动定于 **2 月 20 日至 22 日在纽约市 (NYC)** 举行。
   - [新网站](https://www.ai.engineer/summit/2025) 提供演讲嘉宾和日程安排的实时更新。
- **Karina Nguyen 将发表闭幕主题演讲**：**Karina Nguyen** 将在 AI Engineer Summit 上发表闭幕主题演讲，重点介绍她在 **Notion**、**Square** 和 **Anthropic** 任职期间的卓越背景。
   - 她的职业历程包括对 **Claude 1, 2, and 3** 开发的重大贡献。
- **为 AIE Summit 创建了特别的线上分会场 (Online Track)**：由于 AI Engineer Summit 的合格申请者数量极多，一名成员将主持一个新的线上分会场。
   - 该活动的前两天在纽约市拉开帷幕，更多详情请访问 [Discord 活动页面](https://discord.gg/2YJghJEN?event=1335731498896199741)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.engineer).">未找到标题</a>: 未找到描述</li><li><a href="https://www.latent.space/p/karina">The Agent Reasoning Interface: o1/o3, Claude 3, ChatGPT Canvas, Tasks, and Operator — with Karina Nguyen of OpenAI</a>: 来自 OpenAI（此前在 Anthropic）的 Karina Nguyen 讨论了她在 Claude、ChatGPT Canvas 和 Tasks 方面的工作，以及人机协作的新 AI 交互范式。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1334991738116378768)** (270 条消息🔥🔥): 

> `Discord 屏幕共享问题, AI 辅导概念, Deepseek API 讨论, 开源 AI 工具, Cline vs RooCline` 


- **Discord 屏幕共享问题持续存在**：成员们讨论了 Discord 屏幕共享功能的各种问题，重点提到了音频质量和视频冻结问题。
   - 虽然有些人成功使用了屏幕共享，但其他人指出了 Discord UX 的挫败感，并建议使用 Zoom 等替代平台。
- **AI 辅导系统不断演进**：AI 辅导的概念被解释为一种系统以交互方式教导用户的方法，而不是一次性提供所有信息，类似于 Cursor。
   - 成员们对 AI 辅导及其在引导用户完成流程而非仅仅自动化任务方面的潜在优势表现出兴趣。
- **对 Deepseek API 的担忧**：围绕 Deepseek API 的可靠性展开了讨论，一些成员分享了他们的经验并强调了访问问题。
   - 成员们对 Deepseek API 的质量和性能提出了担忧，认为其托管和功能尚有欠缺。
- **对开源 AI 工具的兴趣**：成员们表达了对开源工具而非商业选项的偏好，讨论了 Cline 和 RooCline 等项目作为可行的替代方案。
   - 维护和使用这些工具之间的动态关系被强调，突显了社区对可访问和可定制解决方案的倾向。
- **Cline vs RooCline 对比**：对原始 Cline 项目和 RooCline 进行了对比，指出 RooCline 已经分化并包含了新的修复和功能。
   - 成员们对 RooCline 中潜在的差异和改进很感兴趣，认为这两个项目都是值得进一步讨论的话题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.fastht.ml/llms-ctx.txt">未找到标题</a>: 未找到描述</li><li><a href="https://carelesswhisper.app">Careless Whisper - Mac 听写应用</a>: 未找到描述</li><li><a href="https://voicebraindump.com/">Brain Dump - 瞬间成形想法。</a>: 未找到描述</li><li><a href="https://www.youtube.com/@d-squared70">D-Squared</a>: 日常工作：Gradient Labs 的专业 AI Whisperer | 副业：向你展示 AI 自动化技巧</li><li><a href="https://drive.google.com/file/d/1xEyeP7IIojCkTgzkSLmkL0RUvu6RL9xq/view?usp=drive_link">MCP.mp4</a>: 未找到描述</li><li><a href="https://github.com/D-Squared70/GenAI-Tips-and-Tricks">GitHub - D-Squared70/GenAI-Tips-and-Tricks: 我发现有用的各种 GenAI 提示和技巧</a>: 我发现有用的各种 GenAI 提示和技巧。通过在 GitHub 上创建账号为 D-Squared70/GenAI-Tips-and-Tricks 的开发做出贡献。</li><li><a href="https://github.com/D-Squared70/GenAI-Tips-and-Tricks/blob/main/Claude_ImplementationPlan.txt">GenAI-Tips-and-Tricks/Claude_ImplementationPlan.txt at main · D-Squared70/GenAI-Tips-and-Tricks</a>: 我发现有用的各种 GenAI 提示和技巧。通过在 GitHub 上创建账号为 D-Squared70/GenAI-Tips-and-Tricks 的开发做出贡献。</li><li><a href="https://www.dylandavis.net/archieve/">Archive &#8211; D-Squared</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">模型与价格 | DeepSeek API 文档</a>: 下面列出的价格以每 1M tokens 为单位。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是标点符号。我们将根据总额计费...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: 每周即兴会议</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1336066195757076592)** (1 messages): 

> `Probability of Random Language Model Weights, Volume Hypothesis in Deep Learning, Importance Sampling in High Dimensions, Network Complexity and Alignment` 


- **获得功能性语言模型的概率**：根据一个研究神经网络的团队计算，通过随机猜测权重获得一个功能完备的语言模型的概率大约是 **3.6 亿个零**（即 $10^{-360,000,000}$）分之一。
   - 他们强调，这一估算反映了**复杂性**——概率越低，模型就**越复杂**。
- **体积假设揭示深度学习原理**：他们估算网络采样概率的方法有助于阐明**体积假设 (Volume Hypothesis)**，该假设将从权重空间中采样具有低训练损失的网络联系起来。
   - 这项工作旨在有效地衡量体积，以此作为理解深度学习在模糊假设下如何运作的一种手段。
- **采样离群方向的重要性**：研究强调，从**高维空间**收集数据非常棘手，其中微小的离群方向会极大地影响体积测量。
   - 他们引入了利用梯度信息的**重要性采样 (Importance Sampling)**，以增加捕获这些离群情况的机会。
- **高复杂性与过拟合相关联**：研究发现，记忆训练数据的网络表现出较低的**局部体积 (Local Volume)**，这表明与泛化良好的网络相比，它们的复杂性更高。
   - 这种联系表明，过拟合模型拥有额外的未对齐推理，这可能导致对人类价值观的担忧和对齐失配 (Misalignment)。
- **分享进一步探索的资源**：该团队通过分享关于该主题的 GitHub 仓库 [basin-volume](https://github.com/EleutherAI/basin-volume) 和 arXiv 上的研究论文，鼓励大家关注他们的项目。
   - 他们提供了各种资源，如代码、[Twitter 线程](https://x.com/norabelrose/status/1886504219919966320)以及深入了解其探索性发现的链接。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/basin-volume">GitHub - EleutherAI/basin-volume: Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors</a>: Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors - EleutherAI/basin-volume</li><li><a href="https://arxiv.org/abs/2501.18812">Estimating the Probability of Sampling a Trained Neural Network at Random</a>: We present an algorithm for estimating the probability mass, under a Gaussian or uniform prior, of a region in neural network parameter space corresponding to a particular behavior, such as achieving ...</li><li><a href="https://x.com/norabelrose/status/1886504219919966320">Tweet from Nora Belrose (@norabelrose)</a>: What are the chances you&#39;d get a fully functional language model by randomly guessing the weights?We crunched the numbers and here&#39;s the answer:
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1335101019226374145)** (104 条消息🔥🔥): 

> `R1 结果复现、语言模型中的审查、混合专家模型 (MoE)、DeepSeek 的行为、AI 社区参与` 


- **R1 在 SmolLM2 上的复现失败**：目前的测试显示，与在真实模型上训练的 **SAEs** 相比，在随机模型上训练的 **SAEs** 在 **SmolLM2 135M** 上表现出更差的 autointerp 评分和更高的重构误差。
   - 使用 **Pythia** 复现论文原始结果的尝试宣告失败，这引发了对最初结论有效性的质疑。
- **DeepSeek 的审查问题**：讨论者指出，**DeepSeek** 对**天安门广场**等敏感话题的回答会根据提示词（prompt）的语言而有所不同，凸显了其设计中潜在的偏见。
   - 模型的回应似乎受到审查机制的影响，一些人建议通过巧妙的提示词来绕过这些限制。
- **DeepSeek 的民族主义信息传递**：用户观察到，**DeepSeek** 在涉及**台湾**相关问题时给出了明显的民族主义叙述，这与其在**天安门**问题上更为受限的回应形成对比。
   - 这种不一致性引发了对所采用的**审查模型（censorship model）**及其如何根据主题进行调整的担忧。
- **混合专家模型 (MoE) 的教育资源**：一位用户寻求关于 **Mixture of Experts** (MoE) 的资源，随后社区分享了全面的视觉指南链接和解释该概念的 YouTube 视频。
   - 虽然活跃度尚不确定，但存在一个专门讨论 MoE 的关联频道。
- **对 AI 项目的社区贡献**：一位新用户表达了对参与 AI 安全和政策相关项目的兴趣，并指出意大利此类倡议较少。
   - 社区鼓励参与有关实用工具的讨论，例如用于 LLM 组装的拖拽式界面，以促进协作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.17727">Sparse Autoencoders Can Interpret Randomly Initialized Transformers</a>：稀疏自编码器 (SAEs) 是一种日益流行的解释 Transformer 内部表示的技术。在本文中，我们将 SAEs 应用于“解释”随机 Transformer...</li><li><a href="https://arxiv.org/abs/2310.02446">Low-Resource Languages Jailbreak GPT-4</a>：大语言模型 (LLMs) 的 AI 安全训练和红队测试是减轻不安全内容生成的措施。我们的工作揭示了这些安全机制在跨语言方面的固有脆弱性...</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts">A Visual Guide to Mixture of Experts (MoE)</a>：揭秘 MoE 在大语言模型中的作用</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/norabelrose/status/1885894889218732290">Nora Belrose (@norabelrose) 的推文</a>：我们下周准备发布七 (!) 篇论文，当 arXiv 限制你的上传速度时，你就知道自己正处于爆发期</li><li><a href="https://x.com/norabelrose/status/1886444249065075093">Nora Belrose (@norabelrose) 的推文</a>：他们的结果在 SmolLM2 上无法复现。对于 SmolLM2 135M，在随机模型上训练的 SAEs 得到的 autointerp 评分比在真实模型上训练的 SAEs 差得多。以下是部分结果...</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1i7o9xo/deepseek_r1s_open_source_version_differs_from_the/m8n3rvk/">Deepseek R1 的开源版本与官方 API 版本存在差异</a>：由 u/TempWanderer101 发布在 r/LocalLLaMA • 126 个赞和 64 条评论</li><li><a href="https://github.com/EleutherAI/sae">GitHub - EleutherAI/sae: Sparse autoencoders</a>：稀疏自编码器。通过创建账号为 EleutherAI/sae 的开发做出贡献。</li><li><a href="https://arxiv.org/html/2401.13136v1">The Achilles Heel of Large Language Models: Lower-Resource Languages Raise More Safety Concerns</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1335063112067448932)** (119 条消息🔥🔥): 

> `DeepSeek Math 论文指标、用于图像生成的 DRAW 架构、优化中的学习率调度、模型训练中的蒸馏过程、神经网络中的复杂度度量`

- **理解 Pass@K 和 Maj@K 指标**：在 DeepSeek Math 论文中，Pass@K 表示在 K 次重复中是否有任何解析出的答案通过，而 Maj@K 指的是在这些重复中多数答案是否通过。
   - 第二个指标 Maj@K 特别适用于数值输出或像多项选择题这样的简洁输出。
- **DRAW 架构概述**：DRAW 网络架构引入了一种模仿人类注视（foveation）的新型空间注意力机制，增强了图像生成能力。
   - 它显著提高了生成模型在 MNIST 和 Street View House Numbers 等数据集上的性能，生成的图像与真实数据难以区分。
- **非平滑凸优化中的学习率调度 (Learning-rate Schedules)**：最近的研究强调了大模型训练中的学习率调度与非平滑凸优化理论之间惊人的紧密联系。
   - 这些见解为学习率调优提供了实际益处，从而能够更好地训练 Llama-types 等模型。
- **模型训练中的合成数据与蒸馏 (Distillation)**：讨论围绕在训练中潜在使用合成数据集展开，包括强化学习和通用策略优化等方法。
   - 这种方法可能有助于利用源自大模型合成示例的输出来微调较小的模型。
- **神经网络中的复杂度度量**：有人推测使用复杂度度量来检测神经网络中不希望出现的推理；目标是使网络与人类价值观对齐，而没有其他动机。
   - 讨论指出了之前将神经网络的简单性与归纳偏置（inductive biases）联系起来的工作，重点关注局部体积和模型复杂度的演变。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/1502.04623">DRAW: A Recurrent Neural Network For Image Generation</a>: 本文介绍了用于图像生成的 Deep Recurrent Attentive Writer (DRAW) 神经网络架构。DRAW 网络结合了一种模仿人类视觉中央凹（foveation）的新型空间注意力机制...</li><li><a href="https://arxiv.org/abs/2410.05258">Differential Transformer</a>: Transformer 倾向于将注意力过度分配给无关的上下文。在这项工作中，我们引入了 Diff Transformer，它在消除噪声的同时增强了对相关上下文的注意力。具体而言，t...</li><li><a href="https://arxiv.org/abs/2412.20302">EXAdam: The Power of Adaptive Cross-Moments</a>: 本文介绍了 EXAdam ($\textbf{EX}$tended $\textbf{Adam}$)，这是一种基于广泛使用的 Adam 优化器构建的新型优化算法。EXAdam 包含了三个关键增强：(1) 新的 ...</li><li><a href="https://arxiv.org/abs/2501.17161v1">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>: 有监督微调 (SFT) 和强化学习 (RL) 是基础模型广泛使用的训练后期技术。然而，它们在增强模型泛化能力方面的作用仍然 ...</li><li><a href="https://arxiv.org/abs/2412.02975">Theoretical limitations of multi-layer Transformer</a>: Transformer，尤其是 decoder-only 变体，是大多数现代大语言模型的核心；然而，除了简单的 $1$ 层模型外，我们对其表达能力知之甚少...</li><li><a href="https://arxiv.org/abs/2501.18965">The Surprising Agreement Between Convex Optimization Theory and Learning-Rate Scheduling for Large Model Training</a>: 我们展示了大模型训练的学习率调度与非平滑凸优化理论的性能界限表现出惊人的相似性。我们为恒定调度提供了一个界限...</li><li><a href="https://arxiv.org/abs/2410.24159">GPT or BERT: why not both?</a>: 我们提出了一种将掩码语言建模 (masked language modeling) 与因果语言建模 (causal language modeling) 合并的简单方法。这种混合训练目标产生的模型在同一个模型中结合了两种建模范式的优势...</li><li><a href="https://openreview.net/forum?id=I4YAIwrsXa">Harnessing Proof Assistant Feedback for Reinforcement Learning and...</a>: Lean 是一款先进的证明助手，旨在通过提供各种交互式反馈来促进形式化定理证明。在本文中，我们探索了利用证明助手的方法...</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1ifag0a/comment/mafoeup/?utm_source=share&utm_medi">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/mblondel_ml/status/1885353898648441006">Mathieu Blondel (@mblondel_ml) 的推文</a>: 下面的 EBM 论文将对偶变量参数化为神经网络。这个想法（已在 OT 或 GANs 等其他背景中使用）非常强大，可能是对偶性对神经网络有用的 *唯一* 途径...</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1ifag0a/comment/mafoeup/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1335014020918411327)** (26 条消息🔥): 

> `David Chalmers 的新论文、Crosscoder 仓库、Sparse Autoencoders 优化、MoE 模型中的专家评估` 


- **Chalmers 关于命题态度 (Propositional Attitudes) 的观点**：[David Chalmers](https://x.com/norabelrose/status/1885454252656779778) 的一篇新论文认为，从 AI 中**提取命题态度**比追求机械论理解 (mechanistic understanding) 更有影响力。
   - Chalmers 还引用了该团队[之前发表的一篇论文](https://arxiv.org/abs/2501.15740)，并对引用表示感谢。
- **Crosscoder 仓库讨论**：一名成员询问了用于训练和使用 Crosscoder 的**开源仓库**，强调了该领域在可复现性方面面临的持续挑战。
   - 另一名成员分享了 [dictionary_learning](https://github.com/jkminder/dictionary_learning) 的 GitHub 链接作为潜在资源。
- **Sparse Autoencoders 与优化可扩展性**：讨论涉及了**稀疏恢复 (sparse recovery) 的挑战**，认为寻找正确的表示通常需要迭代方法。
   - 有观点指出，有效分析所需的规模可能会使这些方法在实践中变得不可行。
- **评估 Mixture of Experts (MoE) 中的专家**：讨论集中在识别 DeepMind 代码竞赛数据集中的**最活跃专家**，并提供了几个专家的频率和权重数据。
   - 重点突出了排名靠前的专家，并指出其性能评估可能有助于为 MoE 模型中的剪枝策略提供参考。
- **专家权重中的余弦相似度与欧几里得距离**：澄清了用于分析专家分布的**距离度量**是欧几里得距离，而非最初假设的余弦相似度。
   - 余弦相似度实际上是作为一种派生指标被引用的，基于 MoE 门控模块聚合的专家分布向量。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1306.0256">Distributions of Angles in Random Packing on Spheres</a>: 本文研究了当点数 n 趋于无穷大时，R^p 中 n 个随机均匀分布的单位向量之间成对夹角的渐近行为，而维度 p 则是...</li><li><a href="https://arxiv.org/abs/2010.09931">Smooth activations and reproducibility in deep networks</a>: 深度网络因其惊人的成功正逐渐渗透到我们生活的几乎每一个领域。然而，随着实质性性能准确度的提高，代价是不可复现性...</li><li><a href="https://x.com/norabelrose/status/1885454252656779778">Nora Belrose (@norabelrose) 的推文</a>: @davidchalmers42 的这篇新论文很好。从 AI 中提取命题态度比追逐“机械论”理解更有用。此外，他引用了我们团队的一篇论文，谢谢...</li><li><a href="https://github.com/jkminder/dictionary_learning">GitHub - jkminder/dictionary_learning</a>: 通过创建一个账户来为 jkminder/dictionary_learning 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1335012534779908211)** (5 条消息): 

> `Non-overlapping windows, make_disjoint_window modification, Chunked prefill, Data storage in scripts.write_out` 


- **模型长度的非重叠窗口**：讨论确认系统使用 `size==max_model_len` 的非重叠窗口，并引用了 [A.3](https://arxiv.org/pdf/2405.14782) 章节以获取更多细节。
   - 一位参与者提到实现 **strided approaches** 以提高效率。
- **修改 `make_disjoint_window` 函数**：有人建议修改 [`make_disjoint_window`](https://github.com/EleutherAI/lm-evaluation-harness/blob/0bb8406f2ebfe074cf173c333bdcd6cffb17279b/lm_eval/models/vllm_causallms.py#L307) 函数，改为生成重叠对。
   - 代码编写者表示愿意审查具体示例以进行潜在调整。
- **关于 Chunked prefill 的查询**：有人提出了关于在系统与模型操作的完整性中使用 **chunked prefill** 所产生影响的疑问。
   - 针对 chunked prefill 操作策略的提问未得到回应。
- **scripts.write_out 中的数据存储问题**：一名成员希望澄清在调用 **scripts.write_out** 时数据存储在何处。
   - 该查询在对话中未得到解答。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/0bb8406f2ebfe074cf173c333bdcd6cffb17279b/lm_eval/models/vllm_causallms.py#L307),">lm-evaluation-harness/lm_eval/models/vllm_causallms.py at 0bb8406f2ebfe074cf173c333bdcd6cffb17279b · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1335351413521252363)** (16 messages🔥): 

> `NeoX 性能、Fusion 标志、Transformer Engine 加速、Scaling Softmax 函数、Detect NVLink Pairs 标志错误` 


- **NeoX 性能指标咨询**：一名成员报告在 **A100** 上运行 1.3B 参数模型时，达到了约 **10-11K tokens/sec**，这与 OLMo2 论文中提到的 **50K+ tokens** 形成了鲜明对比。
   - 尽管尝试最大化 Batch Size，但每秒 token 数的提升依然微乎其微。
- **对 Fusion 标志的困惑**：关于 Pythia 配置中 **partition-activations** 标志的使用存在疑问，指出论文与 GitHub 设置之间可能存在差异。
   - 成员们对使用某些 Fusion 标志表示担忧，因为这些标志似乎会导致运行挂起且不生成任何日志。
- **对 Transformer Engine 速度的预期**：针对 **Transformer Engine** 集成中提到的训练配置进行了咨询，询问在使用 **Mixed Precision BF16 训练**时是否有潜在的加速。
   - 此外，还讨论了使用 **scaled_masked_softmax_fusion** 与 **scaled_upper_triang_masked_softmax_fusion** 的必要性和适用场景。
- **NVLink Pairs 标志的问题**：一名成员尝试使用 **detect_nvlink_pairs** 标志，但遇到报错称该标志不存在，并指出它仅出现在参数文件中。
   - 提供的截图展示了代码库中的这一差异。
- **承认支持延迟**：一名团队成员承认，由于目前正处于 **NeoX 3.0 功能**的开发冲刺阶段，对用户的支持可能会较慢。
   - 他们承诺将在当天晚些时候提供更详细的回复。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/issues/1334">设置 scaled_upper_triang_masked_softmax_fusion 或 rope_fusion 为 True 时进程挂起 · Issue #1334 · EleutherAI/gpt-neox</a>：你好，在 Discord 讨论后提交此错误报告 - 我正尝试训练 Llama2 模型，仓库中的配置将 scaled_upper_triang_masked_softmax_fusion 设置为 True。这是 YAML 文件...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/configs/1-3B-transformer-engine.yml">gpt-neox/configs/1-3B-transformer-engine.yml at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库的 GPU 模型并行自回归 Transformer 实现 - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1b.yml#L57">pythia/models/1B/pythia-1b.yml at main · EleutherAI/pythia</a>：EleutherAI 关于可解释性和学习动态工作的枢纽 - EleutherAI/pythia</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues/1334#issuecomment-2629893112">设置 scaled_upper_triang_masked_softmax_fusion 或 rope_fusion 为 True 时进程挂起 · Issue #1334 · EleutherAI/gpt-neox</a>：你好，在 Discord 讨论后提交此错误报告 - 我正尝试训练 Llama2 模型，仓库中的配置将 scaled_upper_triang_masked_softmax_fusion 设置为 True。这是 YAML 文件...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1334979471127871563)** (219 messages🔥🔥): 

> `远程 MCP 工具、Discord 服务器混淆、Superinterface 产品、使用 Litellm Proxy 进行负载均衡、开源替代方案`

- **关于远程 MCP 工具的讨论**：成员们表达了对 MCP 工具远程功能的需求，强调现有的大多数解决方案都集中在本地实现上。
   - 成员们对当前 MCP 设置的可扩展性和易用性表示担忧，并建议探索替代方案。
- **关于 Discord 服务器的混淆**：一位成员指出存在两个相似的 Discord 服务器，其中一个是另一个的副本，导致了混淆。
   - 澄清指出，这两个服务器都不是官方的，且都由非 Anthropic 用户运行，尽管此服务器的管理人员中包含 Anthropic 的员工。
- **关于 Superinterface 产品的见解**：Superinterface 的联合创始人澄清说，他们的重点是提供 AI Agent 基础设施即服务，这与开源替代方案有所不同。
   - 该产品被定位为将 AI 功能集成到用户产品中的解决方案，强调了为此目的所需的基础设施的复杂性。
- **Litellm Proxy 中的负载均衡**：成员们讨论了使用 Litellm Proxy 进行负载均衡的技术，包括设置权重和每分钟请求数。
   - 这种方法有助于在工作流中高效管理多个 AI 模型端点。
- **开源与专有工具**：对话强调了对开源模型和工具的偏好，并提到了 Llama 和 DeepSeek 等具体替代方案。
   - 成员们指出，根据工具的开放性以及与用户需求的契合度来评估工具非常重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://myhosted.mcp.server.com"">未找到标题</a>: 未找到描述</li><li><a href="https://myhosted.mcp.server.com",">未找到标题</a>: 未找到描述</li><li><a href="https://modelcontextprotocol.io/development/roadmap">路线图 - Model Context Protocol</a>: 未找到描述</li><li><a href="https://modelcontextprotocol.io/docs/concepts/transports">传输层 - Model Context Protocol</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/client/sampling/">采样 (Sampling)</a>:           ℹ️                  协议修订版本: 2024-11-05      Model Context Protocol (MCP) 为服务器请求 LLM 采样（“补全”或“生成”...）提供了一种标准化的方式。</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/utilities/progress/">进度 (Progress)</a>:           ℹ️                  协议修订版本: 2024-11-05      Model Context Protocol (MCP) 支持通过通知消息对长时间运行的操作进行可选的进度跟踪。无论是服务器...</li><li><a href="https://mcp.run">mcp.run - MCP Servlets 的应用商店：为 AI 应用和 Agent 提供便携且安全的代码。</a>: 未找到描述</li><li><a href="https://sageapp.ai/">Sage - Claude 的原生客户端</a>: 未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2/src/everything/sse.ts">servers/src/everything/sse.ts (位于 fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2) · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge?tab=readme-ov-file#sse-bridge>">GitHub - SecretiveShell/MCP-Bridge: 一个提供兼容 OpenAI 端点以调用 MCP 工具的中间件</a>: 一个提供兼容 OpenAI 端点以调用 MCP 工具的中间件 - SecretiveShell/MCP-Bridge</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2/src/github">servers/src/github (位于 fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2) · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/integrations/tools/openapi/">OpenAPI 工具包 | 🦜️🔗 LangChain</a>: 我们可以构建 Agent 来消费任意 API，这里的 API 符合 OpenAPI/Swagger 规范。</li><li><a href="https://github.com/wong2/mcp-cli">GitHub - wong2/mcp-cli: 一个用于 Model Context Protocol 的 CLI 检查器</a>: 一个用于 Model Context Protocol 的 CLI 检查器。通过在 GitHub 上创建账号，为 wong2/mcp-cli 的开发做出贡献。</li><li><a href="https://github.com/sparfenyuk/mcp-proxy?tab=readme-ov-file#2-sse-to-stdio>">GitHub - sparfenyuk/mcp-proxy: 连接到运行在 SSE 传输层上的 MCP 服务器，或使用 MCP Proxy 服务器将 stdio 服务器作为 SSE 服务器暴露。</a>: 连接到运行在 SSE 传输层上的 MCP 服务器，或使用 MCP Proxy 服务器将 stdio 服务器作为 SSE 服务器暴露。 - sparfenyuk/mcp-proxy</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/fe7f6c7b7620a1c">GitHub - modelcontextprotocol/servers (位于 fe7f6c7b7620a1c83cde181d4d8e07f69afa64f2)</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/specification/discussions/64">身份验证 · modelcontextprotocol/specification · 讨论 #64</a>: 首先 - 感谢开源 MCP 以及迄今为止付出的所有努力。我非常期待该协议带来的集成可能性。当然，其中很多...
</li>
</ul>

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1334987290283606037)** (14 条消息🔥): 

> `MCP Server 项目, Zed 扩展, Goose 自动化, Supergateway v2, FFmpeg 速度调节` 


- **MCP Server 项目兴起**：几位用户展示了他们的 MCP server 项目，包括一个由 Claude 驱动的与 MercadoLibre API 集成的项目，用于产品搜索和详细评论。
   - 另一位成员介绍了一个允许在任何客户端上运行任何 MCP server 的服务器，突显了 MCP 能力的通用性。
- **Zed 扩展显示出使用限制**：最近合并的一个用于 Confluence 上下文服务器的 Zed 扩展引发了关于其有效性的讨论，用户注意到 Zed 目前仅支持带单个参数的 prompt。
   - 这一限制引发了关于未来如何在 Zed 编辑器中实现更广泛工具支持的疑问。
- **Goose 自动化 GitHub 交互**：一位用户分享了一个 [YouTube 视频](https://youtube.com/shorts/TbmQDv3SQOE)，演示了开源 AI Agent Goose 如何在集成任何 MCP server 的同时自动执行任务。
   - 该视频展示了 Goose 在自动化 GitHub 任务方面的可扩展功能，强调了 MCP 的创新用途。
- **Supergateway v2 增强功能**：Supergateway v2 允许用户通过 ngrok 隧道远程运行任何 MCP server，使服务器的设置和访问更加容易。
   - 鼓励成员寻求帮助，展示了社区在增强 MCP server 易用性方面的精神。
- **FFmpeg 让倍速听音更轻松**：用户讨论了一个 FFmpeg 命令，在调整速度的同时应用音高降低，从而提高快速收听时的音频质量。
   - 这个简单的解决方案在处理音频文件时显著提升了用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mtct/journaling_mcp">GitHub - mtct/journaling_mcp: 用于日志记录的 MCP Server</a>：用于日志记录的 MCP Server。通过在 GitHub 上创建账号来为 mtct/journaling_mcp 的开发做出贡献。</li><li><a href="https://youtube.com/shorts/TbmQDv3SQOE">使用 Goose 自动化 GitHub 任务</a>：Goose 是一个开源 AI Agent，可以自动化你的开发任务。它与任何 MCP server 集成，为你提供可扩展的功能。这个例子展示了 Go...</li><li><a href="https://github.com/supercorp-ai/supergateway">GitHub - supercorp-ai/supergateway: 在 SSE 上运行 MCP stdio 服务器，反之亦然。AI 网关。</a>：在 SSE 上运行 MCP stdio 服务器，反之亦然。AI 网关。 - supercorp-ai/supergateway</li><li><a href="https://github.com/mouhamadalmounayar/confluence-context-server">GitHub - mouhamadalmounayar/confluence-context-server</a>：通过在 GitHub 上创建账号来为 mouhamadalmounayar/confluence-context-server 的开发做出贡献。</li><li><a href="https://zed.dev/extensions?query=confluence">Zed - 面向未来的编辑器</a>：Zed 是一款高性能、多人的代码编辑器，由 Atom 和 Tree-sitter 的创建者开发。</li><li><a href="https://github.com/JoshuaC215/agent-service-toolkit/pull/164">[草案] 由 madtank 添加实验性 Model Context Protocol (MCP) 支持 · Pull Request #164 · JoshuaC215/agent-service-toolkit</a>：[草案] 添加实验性 Model Context Protocol (MCP) 支持概述。在 agent service toolkit 中引入了可选的、轻量级的 MCP 能力集成...
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1335008707251011584)** (6 条消息): 

> `Stripe 支付问题, 用户故事文档, 用户层级的 Zapier 变通方案, 即将举行的 Office Hours` 


- **Stripe 支付检测失败**：成员们对 **Bolt** 无法 *成功检测* **Stripe** 支付表示沮丧，导致后续操作无法处理。
   - 一位成员正在寻找有效的 prompt，表明目前的 prompt 并不奏效。
- **追踪用户故事和更新**：有人询问团队在哪里记录 **用户故事 (user stories)** 及其更新，*暗示需要更好的组织方式*。
   - 目前对于追踪这些信息的最佳媒介尚未达成共识。
- **使用 Zapier 更新用户层级**：一位成员提到使用 **Zapier** 作为临时方案，在订阅发生变化时更新用户层级，尽管这仍处于早期阶段。
   - 他们计划在不久的将来根据不同用户群体的需求探索更复杂的 UI 解决方案。
- **记下 Office Hours 时间**：成员们注意到即将于 2 月 12 日举行的 **office hours** 会议，可能会为 Stripe 支付问题提供线索。
   - 这次讨论可能会为那些面临类似挑战的人提供宝贵的见解。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1334980913091575871)** (223 条消息🔥🔥): 

> `Bolt 性能问题, Supabase vs Firebase, 连接到 Supabase, Calendly 的 Iframe 问题, 用户身份验证问题` 


- **Bolt 性能问题**：多位用户报告了 Bolt 的性能问题，包括响应缓慢和操作过程中的错误消息。
   - 用户提到被迫重新加载或清除 cookie 以解决访问问题，这表明可能存在服务器或本地存储问题。
- **Supabase vs Firebase**：关于 Supabase 与 Firebase 偏好的讨论展开，许多人因其直接集成和易用性而青睐 Supabase。
   - 相反，一些人表达了对 Firebase 的欣赏，特别是对于那些已经熟悉其生态系统的用户。
- **连接到 Supabase**：用户在进行更改后遇到了 Supabase 服务的断开连接，需要重新连接。
   - 一位用户通过重新加载项目解决了该问题，这暗示连接问题可能源于前端更改。
- **Calendly 的 Iframe 问题**：一位用户在其 Voiceflow 聊天机器人中遇到了 Calendly 的 iframe 问题，声称其显示错误。
   - 尽管进行了后续检查，Voiceflow 和 Calendly 的代表都认为这是 Bolt 的问题，令人感到沮丧。
- **用户身份验证问题**：出现了关于用户身份验证的担忧，一位用户无法登录，并在不同浏览器中遇到相同的错误。
   - 其他人建议了潜在的解决方法，如清除本地存储，但问题对某些用户仍然存在，表明存在更深层次的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://resonant-gumdrop-21c69f.netlify.app/04209cea1e3c4705aa87df1a75d33136">Vite + React + TS</a>：未找到描述</li><li><a href="https://resonant-gumdrop-21c69f.netlify.app/">Vite + React + TS</a>：未找到描述</li><li><a href="https://bolt.new/">bolt.new</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1334987329927905280)** (189 条消息🔥🔥): 

> `GPT4All 错误报告, 量化与模型效率, 数据隐私担忧, AI 模型中的 LaTeX 支持, 使用 LLM 生成 NSFW 故事` 


- **GPT4All v3.8.0 错误报告**：用户在现代 Intel macOS 机器上遇到 GPT4All v3.8.0 崩溃，导致人们假设该版本在这些系统上是 DOA（发布即失效）。
   - 根据用户的系统规格，正在形成一个工作假设，以缩小受影响配置的范围，因为多位用户报告了类似问题。
- **理解模型量化**：讨论围绕影响模型性能的量化级别展开，特别强调较低的量化可能导致显著的质量下降。
   - 敦促用户在量化设置中寻找平衡，以在不使硬件过载的情况下保持合理的输出质量。
- **隐私与数据收集辩论**：关于数据收集信任的激烈辩论出现，对比了西方和中国的数据实践，用户表达了不同程度的担忧和怀疑。
   - 争论反映了对不同国家如何看待数据收集的感知双重标准的挫败感。
- **AI 模型中的 LaTeX 集成**：用户探索使用 MathJax 在 GPT4All 等 AI 应用程序中集成 LaTeX 支持的潜力，强调与 LaTeX 结构的兼容性。
   - 对话集中在解析 LaTeX 内容并提取数学相关的表达式，以便在 LLM 的输出中更好地呈现。
- **用于 NSFW 内容生成的本地 LLM**：一位用户正在寻找一个能够在离线状态下生成 NSFW 故事的本地可用 LLM，类似于现有的在线工具，但不使用 llama 或 DeepSeek。
   - 该用户指定了他们的系统能力和要求，表达了对德语 LLM 的需求，以满足其内容生成需求。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pangea.stanford.edu/computing/unix/formatting/symbols.php">LaTeX 排版</a>：未找到描述</li><li><a href="https://www.mathjax.org/">MathJax</a>：在所有浏览器中呈现美观的数学公式。</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF">bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/nein-doch-oh-nodding-yes-gif-6030752">Nein Doch GIF - Nein Doch Oh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/nein-doch-shocked-gif-14859933">Nein Doch Shocked GIF - Nein Doch Shocked - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF 量化概览</a>：GGUF 量化概览。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://www.hrw.org/news/2022/01/09/legacy-dark-side">“黑暗面”的遗产</a>：在 2001 年 9 月 11 日袭击事件以及 2002 年 1 月 11 日首批恐怖主义嫌疑人抵达关塔那摩湾二十年后，许多美国人可能已经不记得系统性虐待的细节……</li><li><a href="https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml">text-generation-webui/instruction-templates/Mistral.yaml at main · oobabooga/text-generation-webui</a>：一个支持多种推理后端的 LLM Gradio Web UI。- oobabooga/text-generation-webui</li><li><a href="https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Metharme.yaml">text-generation-webui/instruction-templates/Metharme.yaml at main · oobabooga/text-generation-webui</a>：一个支持多种推理后端的 LLM Gradio Web UI。- oobabooga/text-generation-webui</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3448">[回归] macOS Intel 在 3.8 版本启动时崩溃（3.7 版本正常） · Issue #3448 · nomic-ai/gpt4all</a>：（未发现类似问题）Bug 报告：GPT4ALL 在 3.8 版本启动时崩溃，而 3.7 及之前版本运行正常。复现步骤：下载并安装 GPT4ALL 3.8，双击……</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3451">修复当 tool calling/thinking 激活时 LocalDocs 使用的索引，由 cebtenzzre 提交 · Pull Request #3451 · nomic-ai/gpt4all</a>：修复了 #3445。当使用 tool calling 或 reasoning（例如 DeepSeek）以及 LocalDocs 时，第二次响应会出现如下错误：Error: item at index 3 is not a prompt。这是准确的——该项目……</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3445">在 reasoning 标签页中使用所有模型时出现 LocalDocs 错误 'item at index 3 is not a prompt'。 · Issue #3445 · nomic-ai/gpt4all</a>：我在 reasoning 标签页中使用所有 DeepSeek 模型时遇到了 'item at index 3 is not a prompt' 错误。如果我只问一个问题，它可以正常工作，但如果我再问一个，就会出现该错误。我应该……</li><li><a href="https://github.com/google/minja/blob/76f0d01779aa00b0c68f2117f6cb2c9afc3a0ca8/include/minja/minja.hpp#L2486-L2810">minja/include/minja/minja.hpp at 76f0d01779aa00b0c68f2117f6cb2c9afc3a0ca8 · google/minja</a>：一个用于 LLM 聊天模板的极简 C++ Jinja 模板引擎 - google/minja</li><li><a href="https://ajithp.com/2024/09/30/ai-reasoning-and-lrms/.">AI 规划的进展：OpenAI 的 o1 和大推理模型 (LRMs) - Ajith's AI Pulse</a>：探索 OpenAI 的 o1 大推理模型 (LRM) 如何改变 AI 规划和问题解决，在复杂推理任务中超越传统的 LLM。</li><li><a href="https://jinja.palletsprojects.com/en/stable/templates/)">未找到标题</a>：未找到描述</li><li><a href="https://docs.gpt4all.io/gpt4all_desktop/chat_templates.html#advanced-what-are-gpt4all-v1-templates).">聊天模板 - GPT4All</a>：GPT4All 文档 - 在您的硬件上高效运行 LLM
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1335009529909350572)** (13 条消息🔥): 

> `NotebookLM 用于 JS 面试、Google Workspace Standard 账户、BPO 环境中的 NBLM、利用 NBLM 进行语言学习、播客公告` 


- **将完整教程集成到 NotebookLM 中进行学习**：一位成员建议将整个教程网站（如 [W3School JavaScript](https://www.w3schools.com/js/)）而不仅仅是单个链接整合进来，以便更好地利用 NotebookLM 准备 JS 面试。
   - 另一位成员提到，目前已有 Chrome 扩展程序可以辅助将网页导入 NotebookLM。
- **Google Workspace Standard 账户显示的变化**：一位用户升级到了 Google Workspace Standard，并注意到 NotebookLM 的明显变化是顶栏增加了 'Analytics'，而没有显示 'NotebookLM Plus' 字样。
   - 他们强调，即使整体界面看起来相似，使用限制也有所不同，并分享了截图以示清晰。
- **探索 NotebookLM 在 BPO 场景中的用例**：一位成员询问了 NotebookLM 在 BPO 环境中的使用情况，并向他人寻求潜在用例的见解。
   - 这表明人们对 NotebookLM 如何促进业务流程外包（BPO）运营的兴趣日益浓厚。
- **使用 NotebookLM 精通语言**：一位用户详细介绍了他们使用 NotebookLM 学习日语的方法，通过分析视频转录文本并无缝澄清语法概念。
   - 他们对 NotebookLM 未来一年的功能表示期待，展示了其在语言教育方面的潜力。
- **发布 'Roast or Toast' 播客**：Toastinator 宣布了播客 'Roast or Toast' 的首映，他们以幽默的方式剖析深刻的话题，从生命的意义开始。
   - 邀请听众通过该播客独特的格式，对生命的奥秘进行既滑稽又深入的探索。



**提到的链接**：<a href="https://chromewebstore.google.com/search/notebookLM?utm_source=ext_app_menu&pli=1">Chrome Web Store</a>：为您的浏览器添加新功能并个性化您的浏览体验。

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1335008858426572891)** (104 条消息🔥🔥): 

> `NotebookLM 功能、语言设置、音频自定义、API 发布、AI 模型与能力` 


- **NotebookLM 在语言设置方面存在困难**：多位用户对更改 NotebookLM 的输出语言表示困惑，建议修改 Google 账号设置或使用 prompt 来指定所需语言。
   - 用户注意到，尽管浏览器和操作系统设置为英文，但下载的输出内容有时会默认为德语。
- **关于 API 和功能的查询**：有关于 NotebookLM 计划发布的 API 的咨询，用户表达了对额外功能的渴望。
   - 有迹象表明 NotebookLM 的输出 token 限制低于 Gemini，但具体规格尚不明确。
- **音频概览（Audio Overviews）的自定义问题**：一位新用户寻求关于在 NotebookLM 中自定义音频概览的指导，但在 UI 更新后发现该功能缺失。
   - 另一位用户建议查看 Illuminate 以获取相关功能，希望某些功能可能会迁移到 NotebookLM。
- **分析链接等功能的问题**：用户报告称未看到指示可访问 NotebookLM Plus 的分析链接，质疑其在所在地区的推出状态。
   - 有建议通过特定清单进行验证，并提示 Google One 可能会提供 Plus 功能的访问权限。
- **内容输出的反馈**：用户对 NotebookLM 在笔记中包含脚注编号但没有相应链接表示担忧，这导致了对引用的混淆。
   - 用户指出清晰引用实践的重要性，并表示需要更好地处理笔记中的源材料。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.quantamagazine.org/chatbot-software-begins-to-face-fundamental-limitations-20250131/">聊天机器人软件开始面临根本性限制 | Quanta Magazine</a>：最近的结果显示，大语言模型在组合任务方面表现吃力，这表明它们的能力存在硬性限制。</li><li><a href="https://thedrive.ai">The Drive AI：革新文件管理与知识库</a>：发现 The Drive AI 在智能文件组织方面的突破。我们的平台在尖端 AI 的帮助下，将您的文件转化为动态知识库。提升您的业务运营...</li><li><a href="https://aistudio.google.com">Google AI Studio</a>：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en">了解 NotebookLM 如何保护您的数据 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://illuminate.google.com/home">Illuminate | 以您的方式学习</a>：使用 Illuminate 将研究论文转化为 AI 生成的音频摘要，这是您更快理解复杂内容的生成式 AI 工具。</li><li><a href="https://workspaceupdates.googleblog.com/2024/12/notebooklm-plus-gemini-for-google-workspace-users.html#:~:text=NotebookLM%20Plus.-,Rollout%20pace,-Rapid%20Release%20and">
Google Workspace 更新：NotebookLM Plus 现已面向 Google Workspace 客户开放
</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&ref_topic=13853688&sjid=4099718052900055362-NA">比较 Google Workspace 的 Gemini 插件 - 商务版 / 企业版 - Google Workspace 管理员帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1334984827258142812)** (6 messages): 

> `Mojo 和 MAX 解决方案，失效的 Mojo 示例链接，社区 Mojo 示例，Modular 示例页面更新` 


- **Mojo 和 MAX 作为解决方案**：一位成员对澄清 **Mojo** 和 **MAX** 的复杂细节表示兴奋，认为它们是解决当前挑战的终极方案。
   - *这些不是简单的问题*，强调了解决这些问题需要大量的投入。
- **Mojo 示例链接返回 404**：一位成员报告说 Modular 首页上的 **Mojo Examples** 链接已失效，返回 **404 响应**。
   - 另一位成员指出，该问题已被确认并*据称已修复*，但更新可能尚未在网站上反映出来。
- **Mojo 示例的社区贡献**：一位成员指出，在指定的 Discord 频道中可以找到来自社区的 **Mojo 示例**。
   - 随后有人澄清说，Modular 已经撤下了他们的页面以更换示例。
- **社区展示移至论坛**：据指出，社区展示（community showcase）现在已设为**只读**，并已转移到 **Modular Forum** 以获得更好的可访问性。
   - 成员可以通过 [Community Showcase Forum](https://forum.modular.com/c/community-showcase/8) 访问。



**提到的链接**：<a href="https://forum.modular.com/c/community-showcase/8">Community Showcase</a>：使用 MAX 和 Mojo 的社区项目

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1334984004411199639)** (49 messages🔥): 

> `Mojo 与 Swift 的复杂性对比，Mojo 用于编程教育，Mojo 类型系统的挑战，社区对 Mojo 1.0 的反馈，Mojo 的热重载系统` 


- **在 Mojo 中避免 Swift 的复杂性**：有人担心 Mojo 会重蹈 Swift 追求复杂性的覆辙，强调需要保持清晰并避免仓促开发。
   - 社区专注于稳定 Mojo，并在没有不必要压力的情况下仔细权衡折衷方案。
- **在教育场景中使用 Mojo**：一名学生正考虑将 Mojo 用于编程课项目，并询问其与 Pascal 架构显卡的兼容性及系统要求。
   - 讨论强调了潜在的硬件限制，特别是对于旧代 GPU。
- **在 Mojo 系统中集成类型**：一位用户询问在将参数作为具体类型传递时，如何访问 Mojo 类型系统中的特定 struct 字段。
   - 回复表明，用户可能仍在学习如何有效利用 Mojo 的类型能力。
- **对 Mojo 过度依赖 Magic 的担忧**：一位社区成员对 Mojo 通过 'magic' 进行依赖管理表示担忧，希望对安装过程有更多控制。
   - 普遍认为，更清晰的路线图和减少对 magic 的依赖将增强 Mojo 的可用性和透明度。
- **在 Mojo 中实现热重载的困难**：目前在 Mojo 中实现热重载（Hot reloading）存在问题，主要是由于缺乏稳定的 ABI 以及修改结构的挑战。
   - 社区意识到这一限制阻碍了在用 Mojo 构建的框架中实现动态更新。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1335019173067096144)** (41 messages🔥): 

> `MAX Serving Infrastructure, Ollama 性能对比, LLM 中的内存占用, 权重路径问题, DeepSeek R1 模型性能` 


- **MAX Serving Infrastructure 下载权重**：MAX Serving Infrastructure 利用 `huggingface_hub` 下载并缓存模型权重至 `~/.cache/huggingface/hub`，这与 Ollama 的处理方式不同。
   - 讨论强调用户可以通过更改 `--weight-path=` 来避免重复下载，但 Ollama 的本地缓存处理可能并不直观。
- **Ollama 与 MAX 性能对比**：用户注意到在同一台机器上，Ollama 似乎比 MAX 运行得更快，即便相关指标暗示 MAX 的性能表现较低。
   - MAX 中基于 CPU 的推理服务仍处于积极开发阶段，随着模型的进一步调优，预计性能将得到提升。
- **内存占用影响模型性能**：建议在 16GB RAM 的环境下，用户运行 MAX 时可能会遇到内存限制，因此提议调整模型配置以实现更好的资源管理。
   - 为了缓解性能缓慢的问题，建议用户采用量化（quantization）技术并减小 `--max-length` 设置。
- **模型端点可见性问题**：用户在尝试将 MAX 容器与 open-webui 配合使用时，遇到了容器未暴露 v1/models 端点的问题。
   - 日志和错误信息显示模型暴露无效，需要进一步排查故障并调整命令以优化功能。
- **关于 Uvicorn 错误的观察**：澄清了 `uvicorn.error` 消息的出现仅是日志记录的产物，并非实际错误，这在最初引起了用户的一些困惑。
   - 为了进一步测试能力，鼓励用户将命令从 `magic run serve` 切换到 `magic run generate` 以获取直接的流式输出结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/use-max-with-open-webui-for-rag-and-web-search">Modular: Use MAX with Open WebUI for RAG and Web Search</a>：了解 MAX 和 Open WebUI 如何帮助你在 GPU 上快速上手 RAG、网页搜索和 Llama 3.1。</li><li><a href="https://unsloth.ai">Unsloth AI - Open Source Fine-Tuning for LLMs</a>：针对 Llama 3、Phi 3.5、Mistral 等模型的开源 LLM 微调工具！初学者友好。使用 Unsloth 提升速度。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1335253889079050273)** (32 条消息🔥): 

> `多节点上的 GRPO，不带消息结构的 SFT，自定义数据集类的注意事项，劫持 SFTDataset 转换` 


- **Ariel2137 在 16 个节点上部署 GRPO**：在对多节点 PR 进行微调后，一位成员成功在 **16 个节点**上运行了 GRPO，并对即将进行的奖励曲线验证持乐观态度。
   - 他们幽默地提到，在资金充裕的公司工作在这些尝试中具有显著优势。
- **探索不带消息结构的 SFT**：一位成员询问了在不使用典型消息结构的情况下执行 **SFT** 的最佳方法，并建议采用遵循替代模板的自定义方法。
   - 讨论强调了为了有效训练需要对某些消息进行掩码（mask），特别是在 SFT 过程中关注 Ground Truth。
- **创建自定义数据集类**：由于默认 SFTDataset 会添加不需要的特殊 Token 的限制，建议为 SFT 创建自定义数据集。
   - 成员们讨论了编码方法以及在使用原始字符串时手动生成正确布尔掩码（Boolean masks）的重要性。
- **自定义 SFTDataset 转换（transforms）**：一位成员成功自定义了 SFTDataset 中的两个转换，以修改消息和模型的处理方式，从而实现更定制化的训练格式。
   - 他们表示这种灵活性解决了所面临的问题，但也指出如果此类自定义成为标准做法，则需要更直观的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/059cad9c1c0b684ec095634992468eca18bbd395/torchtune/datasets/_alpaca.py#L84">torchtune/torchtune/datasets/_alpaca.py at 059cad9c1c0b684ec095634992468eca18bbd395 · pytorch/torchtune</a>: PyTorch 原生训练后库。通过在 GitHub 上创建账户为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2324">Grpo &amp; verifiable rewards dataset by ianbarber · Pull Request #2324 · pytorch/torchtune</a>: 上下文此 PR 的目的是什么？是 [x] 添加新功能、修复错误、更新测试和/或文档、其他（请在此处添加）。变更日志：添加带有 LoRA 的 GRPO 微调 Recipe 和...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1334977180790423552)** (32 条消息🔥): 

> `Torchtune 中的 Multinode 支持、DPO Recipe Seed 问题、DPO Loss 归一化、Gradient Accumulation 修复、DataLoader 与 Seed 一致性` 


- **Multinode 支持的最终审批请求**：针对 [multinode support pull request](https://github.com/pytorch/torchtune/pull/2301) 提出了最终审批请求，强调了其基于用户需求的重要性。
   - 讨论中强调了对 API 参数 `offload_ops_to_cpu` 的潜在疑虑，建议可能需要进一步审查。
- **DPO Recipes 中的 Seed 不一致性**：正在调查为什么 `seed` 在 **LoRA** 微调中有效，但在 **LoRA DPO** 中失效，**sampler** 的一致性受到了质疑。
   - 已记录多个与 **seed 管理** 相关的问题，特别关注数据集中 `seed=0` 和 `seed=null` 的影响。
- **DPO Loss 的归一化**：一名成员对 **DPO** recipe 中缺乏按 token 数量进行的 loss 归一化表示担忧，而这在单设备设置中是存在的。
   - 已创建一个 issue 来解决这个归一化问题，描述了它与其它 recipe 中应用的逻辑有何不同。
- **潜在的 Gradient Accumulation 修复**：建议对 **DPO** 和 **PPO** recipe 应用 gradient accumulation 修复，这与提高效率的需求相关。
   - 引用了一篇相关的 [博客文章](https://unsloth.ai/blog/gradient) 作为理解梯度管理的资源。
- **DataLoader Batching 的一致性**：来自 **DataLoader** 的日志显示，batch 在多次运行中保持一致，表明随机性问题并非源于数据检索。
   - 有人担心成对数据集类（paired dataset class）可能会影响 sampler 的功能，强调需要对微调和 DPO recipe 进行彻底比较。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/2333">Distributed DPO loss normalization by amount of tokens · Issue #2333 · pytorch/torchtune</a>：分布式 DPO 未按 token 数量归一化 loss -> https://github.com/pytorch/torchtune/blob/main/recipes/lora_dpo_distributed.py#L710 单设备 DPO 具有此逻辑 -> http...</li><li><a href="https://github.com/pytorch/torchtune/issues/2334">Apply gradient accumulation fix to DPO/PPO recipes · Issue #2334 · pytorch/torchtune</a>：https://unsloth.ai/blog/gradient</li><li><a href="https://github.com/pytorch/torchtune/issues/2335">Seed is not applied for DPO recipes · Issue #2335 · pytorch/torchtune</a>：TL;DR 使用 seed: 42 启动两次相同配置会导致两条不同的 loss 曲线。受影响的 recipe：full_dpo_distributed - seed 未设置。Full DPO 取自 #2275 lora_dpo_distributed - seed...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/lora_dpo_distributed.py#L713">torchtune/recipes/lora_dpo_distributed.py at main · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/lora_finetune_single_device.py#L706">torchtune/recipes/lora_finetune_single_device.py at main · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2301?">Multinode support in torchtune by joecummings · Pull Request #2301 · pytorch/torchtune</a>：正式宣布 torchtune 支持 multi-node！背景：这是多个用户的明确需求（#2161, #2142），虽然一切应该可以相当容易地开箱即用，但我们...
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1336072009058095187)** (2 条消息): 

> `LLM 数据增强，R1-V 模型介绍` 


- **LLM 数据增强深度综述**：该综述揭示了大型预训练语言模型 (LLMs) 在需要大量训练数据集的应用中表现出色，并强调了在数据不足时出现的过拟合问题。讨论了独特的 Prompt 模板如何增强数据生成，以及最近的基于检索的技术如何整合外部知识以获得更可靠的输出。
   - 这使得 LLMs 能够产生**有据可依的真实数据 (grounded-truth data)**，强调了数据增强在其训练中的重要性。
- **R1-V 模型革新计数能力**：一位成员兴奋地介绍了 **R1-V**，该模型利用带有**可验证奖励 (verifiable rewards)** 的强化学习 (RL) 来提升视觉语言模型 (VLMs) 的计数能力。令人印象深刻的是，一个 **2B 模型**在仅 **100 个训练步数**内，以低于 **$3** 的成本超越了 **72B 模型**。
   - 该项目将**完全开源**，邀请社区关注未来的更新。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18845">Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities</a>：预训练语言模型日益增长的规模和复杂性在许多应用中展示了卓越的性能，但它们通常需要大型训练数据集才能得到充分训练...</li><li><a href="https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Liang Chen (@liangchen5518) 的推文</a>：很高兴向大家介绍 R1-V！我们使用带有可验证奖励的 RL 来激励 VLMs 学习通用的计数能力。2B 模型仅需 100 个训练步数即可超越 72B 模型，成本不到 $3...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1336087344167780414)** (1 条消息): 

> `Jason Weston 讲座，LLM 自我改进方法，Jason Weston 背景介绍` 


- **Jason Weston 精彩讲座就在今天！**：我们的第二场讲座由 **Jason Weston** 主讲，将于今天 **下午 4:00 (PST)** 举行！您可以在[这里](https://www.youtube.com/live/_MNlLhU33H0)观看直播。
   - *《学习如何让 LLMs 自我改进与推理》* 将涵盖提高 LLM 在不同任务中性能的各种方法。
- **LLM 自我改进的创新方法**：Jason 将讨论几种最近的 LLM 方法，包括 **Iterative DPO** 和 **Meta-Rewarding LLMs**，并提供了详细论文链接，如 [Iterative DPO](https://arxiv.org/abs/2312.16682) 和 [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020)。
   - 这些技术专注于增强模型在推理、数学和创意任务中的熟练度，展示了 LLM 技术不断进化的能力。
- **Jason Weston 令人瞩目的背景**：Jason Weston 是 AI 研究领域的杰出人物，拥有机器学习博士学位，职业生涯丰富，曾任职于 **Meta AI** 和 **Google**。他曾多次获得最佳论文奖等荣誉，并参与了获得艾美奖的 **YouTube Recommendation Engines** 项目。
   - 他丰富的经验包括一系列声名显赫的职位以及对 AI 和 NLP 领域的贡献，彰显了他在该领域的重大影响力。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1335005215795183647)** (51 messages🔥): 

> `测验完成困惑, MOOC 项目参与, 证书查询, 邮件列表确认, 黑客松结果更新` 


- **测验完成情况引发困惑**：许多成员对已完成的测验是否计入课程完成进度表示不确定，特别是那些不确定截止日期的成员。一位成员确认目前提交作业“还没有截止日期”。
   - *没问题的！* 由于 MOOC 课程详情尚未完全发布，因此无需担心完成状态。
- **MOOC 学生渴望参与项目**：有咨询关于 MOOC 学生是否可以参加课程项目的 Research Track。目前，该课程主要面向支付学费的 UC Berkeley 学生。
   - 不过，MOOC 学生参与的可能性仍在讨论中，预计很快会有更新。
- **关于证书状态的查询**：参与者请求更新上学期课程的证书状态，提到他们已经填写了领取表格。得到的答复是，对于那些正在等待的人来说，“应该很快了”。
   - 其他人确认收到了确认邮件，但对邮件沟通的效率表示担忧。
- **邮件列表确认依然难以获取**：用户对错过用于课程更新的邮件列表邮件表示担忧，并询问如何确保不会错过重要通知。邮件列表的邮件应来自特定地址，以避免被误认为垃圾邮件。
   - 为质疑自己报名状态的人提供了确认和支持，并指出如果收到了来自 Google Forms 的确认，那么确实已经报名。
- **黑客松结果期待**：有人对之前的黑客松结果感到好奇并询问更新。据指出，参与者已收到私下通知，公开宣布预计在下周进行。



**提到的链接**：<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>：MOOC，2025 年春季

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1335299101545332736)** (8 messages🔥): 

> `测验可用性, DeepSeek R1 vs PEFT, 测验邮件提醒, 推理技术学习会议, 课程网站导航` 


- **测验已发布在课程网站上**：一位成员确认 Lecture 1 的测验已在课程网站的教学大纲（syllabus）部分上线。
   - 如需直接访问，请查看[此处](https://llmagents-learning.org/sp25)的教学大纲部分。
- **DeepSeek R1 挑战 PEFT**：一位成员认为 DeepSeek R1 证明了使用 Group Relative Policy Optimization 的强化学习优于 PEFT 或指令微调（instruction fine-tuning）。
   - 这一观点表明，鉴于 DeepSeek R1 的有效性，关注点正从传统的 Prompting 方法发生转移。
- **没有测验的邮件提醒**：注意到新测验或答案库没有邮件提醒；测验通常在相关讲座后的周三左右发布。
   - 答案会在一周后跟进，但课程团队避免发送邮件以保持收件箱整洁。
- **关于推理技术的学习会议**：一场专注于 Lecture 1 和 DeepSeek R1 推理技术的学习会议即将开始。
   - 有兴趣加入的成员可以通过提供的 [Discord 链接](https://discord.gg/uGYPWFsX)进行连接。
- **测验在课程网站上的位置**：成员询问测验位置，课程协调员指出可以在课程网站的教学大纲部分找到。
   - 提供了直接链接以便更轻松地导航到相关材料，确保学生能够高效找到资源。



**提到的链接**：<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>：MOOC，2025 年春季

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1334977894438535282)** (33 messages🔥): 

> `PR 处理, 使用 NVDEC 进行视频解码, WebGPU Autogen 进展, Linux 发行版中的 LLVM 和 Clang 使用` 


- **PR 处理需要细心和细节**：当维护者关闭 PR 时，反思反馈并改进至关重要，正如一位成员指出的 *'拼写错误反映了缺乏对细节的关注'。*
   - 建议在重新提交前多次审查提交内容，以确保清晰度和准确性。
- **NVDEC 视频解码的挑战**：使用 **nvdec** 解码视频可能很复杂，需要注意文件格式，并且由于内部复杂性，可能需要 **cuvid** 二进制文件。
   - **libavcodec** 的实现非常冗长，包含高层抽象，可以进一步简化。
- **WebGPU Autogen 进展报告**：一位成员报告他们即将完成 **WebGPU autogen**，由于超过了行数限制，仅需要进行细微简化。
   - 他们强调了在未安装 **dawn binaries** 时的指令需求，并指出测试在 **Ubuntu** 和 **Mac** 平台上均已通过。
- **Linux 发行版中的 Clang 与 GCC**：虽然很少有 Linux 发行版使用 **clang**，但它受到 **Apple** 和 **Google** 等特定平台的青睐。
   - 然而，主流 Linux 发行版仍普遍使用 **gcc**，这引发了关于发行版是否应切换到 **clang** 以获得更好优化的辩论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Llama-3.2-1B">meta-llama/Llama-3.2-1B · Hugging Face</a>: no description found</li><li><a href="https://github.com/FFmpeg/FFmpeg/blob/c6194b50b1b4001db23e8debab4ac4444e794f90/libavcodec/nvdec.c#L350">FFmpeg/libavcodec/nvdec.c at c6194b50b1b4001db23e8debab4ac4444e794f90 · FFmpeg/FFmpeg</a>: Mirror of https://git.ffmpeg.org/ffmpeg.git. Contribute to FFmpeg/FFmpeg development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1335241788935639141)** (3 messages): 

> `HCQ 执行范式, CPU P2P 传输机制, Math Trait 重构, 多 GPU 执行策略` 


- **HCQ 执行范式简化了对多 GPU 的理解**：讨论强调了 **HCQ-like 执行** 是理解 **multi-GPU execution** 的关键一步，并提到了对 **CPU implementations** 的潜在支持。
   - *有人指出*，优化用于决定 CPU 和 GPU 工作分配的调度器（dispatcher）可以提高性能。
- **CPU Peer-to-Peer 传输解析**：一位成员推测 **CPU 上的 p2p** 可能涉及释放内存块上的锁以便驱逐到 **L3/DRAM**，并思考了 **D2C transfers** 的效率。
   - *人们担心* 在这些复杂的多插槽（multi-socket）传输过程中，执行局部性对性能的影响。
- **Math Trait 重构将两个类变为三个**：一位成员详细介绍了他们在 **math trait refactor** 上的初步尝试，提到类数量意外地从两个增加到了三个。
   - 一个可能的改进是将原地运算符（in-place operators）压缩进一个 **MathTraits class**，并提供了一个 [GitHub comparison](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:math_trait_refactor) 来展示更改。



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:math_trait_refactor">Comparing tinygrad:master...davidjanoskyrepo:math_trait_refactor · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing tinygrad:master...davidjanoskyrepo:math_trait_refactor · tinygrad/tinygrad

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1335199945170354259)** (17 messages🔥): 

> `Cohere 试用 Key 限制, Command-R+ 模型性能, 账户自动登出问题` 


- **Cohere 试用 Key 限制的困惑**：一位成员对 **Cohere trial key** 表示困惑，特别是限制何时重置——是生成后的 30 天，还是每个月初。
   - *通常这并不是一个会被问到的问题*，因为该 Key 是用于评估目的，而非免费使用。
- **对 Command-R+ 模型的赞赏**：一位用户强调了 **Command-R+ model** 在长期使用后如何持续满足他们的需求，且没有测试其他模型的欲望。
   - 他们指出，尽管自己不是高级用户，该模型仍不断给他们带来 **惊喜**。
- **持续的自动登出问题**：一位成员报告了他们的账户 **auto logging out** 的持续问题，导致需要反复登录。
   - 这个问题似乎是该频道用户中普遍存在的困扰。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1335014895116222569)** (9 messages🔥): 

> `Embed API v2.0 errors, Command R and Japanese translations` 


- **Embed API v2.0 中的 HTTP 422 错误**：一位用户报告在尝试使用提供的 cURL 命令调用 Embed API v2.0 时出现“HTTP 422 Unprocessable Entity”错误，并对较长文章是否需要进行必要的预处理表示担忧。
   - 建议包括确保请求中正确包含 API key，因为另一位用户指出该请求在他们那里可以正常工作。
- **日语翻译结果不一致**：一位成员提出了使用 Command R 或 Command R+ 进行日语翻译时结果不一致的问题，称有时翻译会完全失败。
   - 对此，一位成员建议联系支持部门并提供示例以协助多语言团队，而另一位成员则提到利用日语网站来获取上下文。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1335215821433081917)** (4 messages): 

> `Limitations of LLMs in Math, ASLLM - Application Specific Language Models` 


- **LLM 在数学任务上表现不佳**：一位成员指出 **LLM** 并非为数学设计，建议用户应该在 AI 之外创建一个计算器或利用现有的程序。
   - 这种观点强调了对专用工具的需求，而不是仅仅依赖语言模型进行数学计算。
- **Wolfram Alpha 作为 ASLLM 的例子**：一位用户提到 **Wolfram Alpha** 是用于专门任务的 **ASLLM**（Application Specific Large Language Model，特定应用大语言模型）的一个例子。
   - 这突显了使用针对特定应用量身定制的模型价值，特别是对于复杂的数学查询。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1334979429088104581)** (6 messages): 

> `LlamaReport, o3-mini support, SciAgents, PDF to PPT Generator, Contextual Retrieval` 


- **LlamaReport 展示报告生成能力**：分享了一段演示 **LlamaReport** 早期测试版的视频，突显了其在 2025 年生成报告的潜力。你可以在[这里](https://t.co/pYx3O5BpYe)观看。
   - 这一进展旨在为寻求高效解决方案的用户简化报告流程。
- **o3-mini 获得首日支持**：宣布支持 **o3-mini**，并提供了通过 pip 安装的命令：`pip install -U llama-index-llms-openai`。更多详情见[此处](https://t.co/MLfzxCGhbW)。
   - 这使得想要从一开始就使用 o3-mini 的开发者能够更顺畅地进行集成。
- **介绍用于科学发现的 SciAgents**：**SciAgents** 是一个自动化的科学发现系统，具有利用本体图（ontological graphs）的多 **Agent** 工作流。点击[此处](https://t.co/9pBYvN4IQh)了解更多。
   - 该项目展示了协作分析如何推动科学研究的创新。
- **利用 AI 将 PDF 转换为 PowerPoint**：一个开源 Web 应用允许用户轻松地将 **PDF 文档** 转换为动态的 PowerPoint 演示文稿。该项目利用了 **LlamaParse**，可以在[这里](https://t.co/XRgwUrlvA3)进一步探索。
   - 该应用简化了创建演示文稿的过程，对于希望自动化工作流程的用户来说是一个令人兴奋的工具。
- **用于提升 RAG 准确性的 DocumentContextExtractor**：一位 Reddit 用户强调了 **DocumentContextExtractor**，旨在增强检索增强生成（RAG）的准确性，**AnthropicAI** 和 **LlamaIndex** 都对此进行了展示。更多详情请查看[此讨论帖](https://t.co/qoVrgd0ddy)。
   - 这突显了开源社区在改进 AI 上下文理解方面持续做出的贡献。



**提到的链接**：<a href="https://t.co/Vh9kJc3GRZ">GitHub - lesteroliver911/ai-pdf-ppt-generator-openai: A fun project where I use the power of AI to analyze a PDF. The AI extracts key information based on the user&#39;s instructions and selections (see the UI demo). The user then gets a second screen to edit the slides before downloading the final PPT. Simple, fast, and powered by AI to make creating presentations a breeze!</a>：一个有趣的项目，我利用 AI 的力量来分析 PDF。AI 根据用户的指令和选择提取关键信息（见 UI 演示）。然后用户进入第二个屏幕，在下载最终的 PPT 之前编辑幻灯片。简单、快速，并由 AI 驱动，让创建演示文稿变得轻而易举！

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1335262685499228213)** (19 条消息🔥): 

> `Deepseek vs OpenAI, 从向量数据库自动检索, 测试分块策略, 结构化输出的 Token 成本, 多用户内存管理` 


- **Deepseek 声称胜过 OpenAI**：一位成员指出 **Deepseek** 和 **OpenAI** 之间有了明显的赢家，并分享了一个令人惊讶的叙述链接 [点击此处](https://bit.ly/deepseek-audio-narrative)。
   - 这一对话引发了人们对这些工具相对性能表现的兴趣。
- **探索使用 Chroma 进行自动检索**：一位用户询问是否可以将 **summaryextractor** 和 **keyword extractor** 与从 **Chroma** 等向量数据库检索到的 metadata 一起使用。
   - 他们寻求关于当前设置功能限制的澄清，并附上了示例图片。
- **测试分块策略的技巧**：分享了关于测试 **LlamaIndex** 分块策略的建议，包括尝试不同的 chunk sizes 和 overlap values。
   - 指导强调使用 evaluation metrics 和真实查询测试来优化性能，并在 retrieval 和 synthesis chunks 之间取得平衡。
- **结构化输出的 Token 成本**：一位成员担心输出中的 schema 结构（如 keys 和标点符号）是否会在 inference 过程中产生 Token 成本。
   - 澄清了结构确实计入 input tokens，而生成的值也包含在成本中。
- **多用户应用的内存管理**：一位用户讨论了在其应用中为每个用户进行独立内存管理的必要性，并询问了同时使用 **retrievers** 和 **rerankers** 的情况。
   - 他们寻求关于潜在 latency 问题以及共享资源与个人资源之间平衡的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/optimizing/production_rag/#building-performant-rag-applications-for-production>),">为生产环境构建高性能 RAG 应用 - LlamaIndex</a>: 未找到描述</li><li><a href="https://bit.ly/deepseek-audio-narrative">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1335262500119253032)** (1 条消息): 

> `Deepseek vs OpenAI, 音频叙事技术` 


- **Deepseek 领先于 OpenAI**：一次讨论强调了 **Deepseek** 和 **OpenAI** 之间的明显赢家，表明了它们在竞争能力中展现出的新优势。
   - 鼓励听众欣赏展示这场竞争的 [惊人叙述](https://bit.ly/deepseek-audio-narrative)。
- **音频叙事技术受到关注**：音频叙事技术的有效性正成为 AI 能力讨论的焦点。
   - **Deepseek** 和 **OpenAI** 之间的比较揭示了这些平台如何利用叙事来提高用户参与度。



**提到的链接**: <a href="https://bit.ly/deepseek-audio-narrative">未找到标题</a>: 未找到描述

  

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1335046614586359828)** (1 条消息): 

> `DeepSeek Perspectives, Power Objects in AI, AI Boosters vs Skeptics, Open Source vs Proprietary Development, AI Doomsday Concerns` 


- **DeepSeek 反映了我们的希望与恐惧**：文章讨论了 **DeepSeek** 如何作为一个*教科书级的权力客体 (power object)*，揭示了更多关于我们对 AI 的渴望与担忧，而非技术本身，正如[这里](https://www.dbreunig.com/2025/01/31/deepseek-as-a-power-object.html)所强调的。
   - *关于 DeepSeek 的每一条热评都展现了个人对 AI 影响的具体希望或恐惧*。
- **AI 助推者庆祝 DeepSeek 的前景**：**AI boosters** 认为 DeepSeek 表明 **LLM** 的进步将持续不减，增强了他们对 AI 发展的乐观态度。
   - 这补充了他们的叙事，即尽管存在怀疑，AI 创新仍将继续前进。
- **怀疑论者质疑 AI 的竞争优势**：**AI skeptics** 认为 DeepSeek 说明了缺乏任何显著优势，暗示 AI 公司在快速变化的环境中没有防御性定位。
   - 他们的观点指向了关于 AI 可持续性及其在现实世界应用中集成的更广泛担忧。
- **开源倡导者拥护 DeepSeek**：对于 **open source advocates** 而言，DeepSeek 证明了与专有模型相比，协作、透明的开发实践正在蓬勃发展。
   - 他们将 DeepSeek 的出现视为 **open source community** 的胜利，强调了知识共享的好处。
- **围绕 AI 的末日情景**：**AI doomers** 对 DeepSeek 的影响表示警惕，担心不受控制的 AI 发展会导致一个不确定且具有潜在危险的未来。
   - 他们的担忧突显了在 AI 领域进行更稳健的伦理考量和监管的必要性。



**提及的链接**：<a href="https://www.dbreunig.com/2025/01/31/deepseek-as-a-power-object.html">DeepSeek as a Power Object</a>：关于 DeepSeek 的浪潮揭示了更多关于我们自身希望和担忧的内容，而非 DeepSeek 本身。

  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1335411950280118332)** (1 条消息): 

> `SAEs performance, LLM steering methods` 


- **SAE 面临重大挑战**：一位成员对 **SAE** 在可预测地引导 LLM 方面的长期可行性表示失望，引用了最近的一场[讨论](https://x.com/kzslider/status/1885666578429055096)。
   - 另一位成员强调了近期问题的严重性，称：*“该死，一天之内发生了‘三连杀’。SAE 最近真的遭受了重创。”*
- **对 SAE 可预测性的担忧**：根据最近的讨论，有一种观点认为 **SAE** 可能不是长期有效引导 LLM 的最佳方法。
   - 成员们对 SAE 遇到的挑战发声越来越频繁，暗示需要替代的引导方法。



**提及的链接**：<a href="https://x.com/kzslider/status/1885666578429055096">来自 KZ is in London (@kzSlider) 的推文</a>：该死，一天之内发生了“三连杀”。SAE 最近真的遭受了重创。

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1335459661943537664)** (13 messages🔥): 

> `DSPy 2.6 中的 Typed Predictors，将 Chain-of-Thought 与 R1 模型结合，DSPy 中的流式输出，DSPy 导入错误` 


- **DSPy 中不再存在 Typed Predictors**：成员们澄清 **typed predictors** 已被弃用；在 DSPy 2.6 中，普通预测器（normal predictors）已能满足功能需求。
   - 强调在当前版本中**已不再有 typed predictor 这种东西**。
- **对混合 DSPy 技术的兴趣**：一位成员表示有兴趣将 **DSPy Chain-of-Thought** 与 **R1 模型** 结合进行微调，以共同冲刺 **Konwinski Prize**。
   - 他们还邀请其他人加入讨论，并参与该倡议相关的协作努力。
- **DSPy 流式输出的挑战**：一位用户分享了在使用 **dspy.streamify** 增量生成输出时遇到的困难，收到了 **ModelResponseStream** 对象而非预期的值。
   - 他们在代码中实现了条件判断以妥善处理输出类型，并寻求进一步的改进建议。
- **DSPy 的 ImportError 问题**：一位用户报告在尝试使用 `BootstrapFewShot` 指标进行验证时，遇到了与 `passage_has_answers` 相关的 **ImportError**。
   - 该问题专门出现在使用提供的训练集编译 **RAG** 的过程中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dspy.ai/api/utils/streamify/?h=stream#dspy.streamify">streamify - DSPy</a>：用于编程（而非提示）语言模型的框架。</li><li><a href="https://dspy.ai/tutorials/deployment/?h=stream#deploying-with-fastapi">Deployment - DSPy</a>：用于编程（而非提示）语言模型的框架。
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1335321295491170305)** (10 messages🔥): 

> `OpenEuroLLM, 欧盟委员会 AI 倡议, 研究项目挑战` 


- **OpenEuroLLM 为欧盟语言首次亮相**：[OpenEuroLLM](https://openeurollm.eu/) 作为首个面向所有欧盟语言的开源 LLM 家族推出，强调符合欧盟法规。
   - *这些模型将在欧洲强大的监管框架内开发*，确保与**欧洲价值观**保持一致，同时保持技术卓越。
- **欧盟委员会强调 AI 的欧洲根源**：根据 [EU_Commission](https://x.com/EU_Commission/status/1886427917762150427) 的一条推文，OpenEuroLLM 因其卓越表现获得了首个 STEP Seal，旨在汇聚欧盟初创公司和研究实验室。
   - 该倡议专注于在欧洲超级计算机上开发 AI 的同时，保持**语言和文化多样性**。
- **研究工作中的繁忙日程**：一位成员分享了他们在大学和研究项目中的繁忙日程，反映了同行在平衡学术和个人承诺方面的普遍困境。
   - 另一位成员 spirit_from_germany 以友好的提示询问了他们的可用时间。
- **对性能对比的兴趣**：一位参与者对测试新模型表示兴奋，称其据传比 **HunYuan** 更快。
   - 这反映了社区对现有 AI 模型之间性能对比的浓厚兴趣。
- **对未来 AI 发展的怀疑**：一位成员幽默地评论说到 **2030** 年再来看看，反映了对 AI 进步时间线的怀疑。
   - 此前，对话讨论了 OpenEuroLLM 倡议中概述的雄心勃勃的目标。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/EU_Commission/status/1886427917762150427">来自欧盟委员会 (@EU_Commission) 的推文</a>：欧盟制造的 AI 🇪🇺 OpenEuroLLM，首个涵盖所有欧盟语言的开源大语言模型家族，赢得了首个 STEP Seal。它汇聚了欧盟初创公司、研究机构...</li><li><a href="https://openeurollm.eu/">Open Euro LLM</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1334979579575795832)** (4 条消息): 

> `CV Research Collaboration, R1-Llama and R1-Qwen Evaluation, DeepSeek Model Specifications` 


- **CV 研究协作机会**：一名成员表示愿意在 **计算机视觉 (CV) 研究论文** 方面进行协作。
   - *有人在为 CV 项目寻找贡献者吗？*
- **R1-Llama 表现超出预期**：对 **R1-Llama-70B** 的初步评估表明，它与 **o1-mini** 和原始 **R1** 模型旗鼓相当，甚至有所超越，引起了社区的关注。
   - 此次评估涉及解决 **奥林匹克级别的数学和编程问题**，展示了领先模型中潜在的泛化缺陷 [来源](https://x.com/JJitsev/status/1886210118594760744)。
- **DeepSeek 的规格备受关注**：**DeepSeek v3/R1** 模型拥有 **37B 激活参数**，与消耗更多资源的 **Llama 3** 稠密架构模型形成对比。
   - 讨论强调，**Mixture of Experts (MoE)** 方法有助于提高计算效率，并得到了 DeepSeek 团队广泛优化的支持。



**提到的链接**：<a href="https://x.com/JJitsev/status/1886210118594760744">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 🇮🇱 (@JJitsev) 的推文</a>：DeepSeek R1 Distilled Llama 70B & Qwen 32B 模型声称能解决奥林匹克级别的数学和编程问题，与声称同样能力的 o1-mini 相匹配。它们能处理揭示通用性的 AIW 问题变体吗...

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1335060613763698688)** (3 条消息): 

> `Fine-tuning reasoning models, GRPO Colab notebook` 


- **成员对微调推理模型感到困惑**：一名成员表达了对如何 **微调推理模型** 的困惑，幽默地承认自己甚至不知道从哪里开始。
   - *Lol* - 看来他们正在这个复杂的领域寻求指导。
- **分享了用于 GRPO 的 Colab 笔记本**：另一名成员分享了一个 [用于 GRPO 的 Colab 笔记本](https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing)，为对该主题感兴趣的人提供了资源。
   - 对于想要专门学习更多关于 **GRPO** 知识的成员来说，这可能是一个极好的起点。



**提到的链接**：<a href="https://colab.research.google.com/drive/1bfhs1FMLW3FGa8ydvkOZyBNxLYOu0Hev?usp=sharing">Google Colab</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1335044018844991520)** (3 条消息): 

> `o3-mini compatibility, Open Interpreter changes` 


- **关于在 Open Interpreter 中使用 o3-mini 的问题**：一名成员询问 **o3-mini** 是否可以在 **01** 和 **interpreter** 中使用。
   - 成员们对兼容性表示了担忧，表明需要澄清集成的潜力。
- **对 Open Interpreter 更新的预期**：另一名成员询问在即将推出的 **Open Interpreter** 应用层面可以期待什么样的变化。
   - 他们很好奇根据即将到来的更新，这些变化会是 **微小** 的还是 **重大** 的。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1335793348711288854)** (2 条消息): 

> `Cursor AI as Development Tool, Honor of Kings Market Transactions` 


- **掌握 Cursor AI 以提升生产力**：本周二美国东部时间下午 5 点，请加入我们的混合活动，主题是如何 **像专业人士一样使用 Cursor AI**。特邀演讲者 Arnold（一位 10X CTO）将讨论这一强大工具的最佳实践。参与者可以亲临 Builder's Club 或通过 Zoom 参加，链接将在注册后分享。
   - 该活动旨在提高开发者的编码速度和质量，同时也为非技术人员提供无代码选项，以便轻松创建原型。
- **Honor of Kings 市场的高价成交**：**Honor of Kings** 市场今天出现了一笔高价收购，**小蛇糕** 以 **486** 的价格售出。鼓励用户使用提供的市场代码和密码在市场中进行交易。
   - 参与者可以打开游戏并使用代码 **-<344IRCIX>-** 访问市场，并输入密码 **[[S8fRXNgQyhysJ9H8tuSvSSdVkdalSFE]]** 来买卖物品。



**提到的链接**：<a href="https://lu.ma/wqphpn4d">Awesome AI Tool - Use Cursor Like a Professional · Zoom · Luma</a>：你想学习如何像专业人士一样使用 Cursor AI 吗？🚀 我们的特邀演讲者 Arnold 将分享他如何通过精通 Cursor 成为一名 10X CTO。我们将……

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1336053228705550490)** (1 条消息): 

> `Lumigator Live Demo, Firefox AI Platform, Blueprints Update, Builders Demo Day Pitches` 


- **用于模型评估的 Lumigator Live Demo**：参加 [Lumigator Live Demo](https://discord.com/events/1089876418936180786/1331996842568843315) 了解安装和入门流程，运行你的第一次模型评估。
   - 本次活动将引导参与者完成**有效模型性能测试**的关键设置步骤。
- **Firefox AI Platform 发布，支持离线任务**：[Firefox AI Platform](https://discord.com/channels/1089876418936180786/1329145280838500475) 现已上线，使开发者能够在 Web 扩展中利用离线 Machine Learning 任务。
   - 这一新平台为直接在用户友好型环境中提升 **Machine Learning 能力**开辟了途径。
- **开源 Recipes 项目 Blueprints 的最新动态**：查看 [Blueprints Update](https://discord.com/channels/1089876418936180786/1230938514955436242/1332449189715509279) 获取旨在增强开源项目的新 Recipes。
   - 该计划旨在为开发者提供**创建有效软件解决方案**的基本工具。
- **Builders Demo Day Pitches 已发布**：[Builders Demo Day Pitches](https://www.youtube.com/playlist?list=PLgjjGlfBflISGQaljPUkxEWqDYBgfC7TZ) 已在 Mozilla Developers 的 YouTube 频道发布，展示了来自开发者社区的创新成果。
   - 这些 Pitches 提供了一个与**前沿开发项目**和想法互动的绝佳机会。
- **重要更新与公告**：成员可以查看有关社区最新进展的 [重要新闻](https://discord.com/channels/1089876418936180786/1262961704602570832/1333936885566799912)。
   - 随时了解影响社区倡议和协作的关键讨论。


  

---


---


---


{% else %}


> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果你想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}