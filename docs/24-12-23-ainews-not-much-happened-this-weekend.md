---
companies:
- openai
- langchain
- hume
- x-ai
- amd
- nvidia
- meta-ai-fair
- hugging-face
date: '2024-12-24T01:01:31.548256Z'
description: '**o3** 模型因其能力和影响引发了广泛关注，其中包括一名 OpenAI 董事会成员提到了“AGI”（通用人工智能）。**LangChain**
  发布了其 **2024 年 AI 现状** 调查报告。**Hume** 发布了 **OCTAVE**，这是一个拥有 **30 亿参数**、仅限 API 调用的语音语言模型，具备声音克隆功能。**x.ai**
  完成了 **60 亿美元的 C 轮** 融资。


  讨论的焦点集中在 **推理时扩展 (inference-time scaling)**、**模型集成** 以及 **小模型** 惊人的泛化能力。新工具和数据集包括
  **FineMath**（Hugging Face 上最优秀的开源数学数据集）以及 LLM 智能体框架。行业动态涵盖了 **AMD MI300X** 与 **Nvidia
  H100 + H200** 为期 **5 个月的基准测试**、与 **苏姿丰 (Lisa Su)** 会面中关于 AMD 软件栈的见解，以及 AI 工程师职位的招聘信息。研究创新方面，包括
  Meta AI 的 **大概念模型 (LCM)**、用于潜空间推理的 **连续思维链 (Coconut)** 以及机械解释性 (mechanistic interpretability)
  研究项目。'
id: 4bde6599-04bd-4d95-aa5f-5e5e2f526cdb
models:
- o3
- o1
- opus
- sonnet
- octave
original_slug: ainews-not-much-happened-this-weekend-4954
people:
- lisa-su
- clementdelangue
- philschmid
- neelnanda5
title: 这个周末没发生什么特别的事。
topics:
- inference-time-scaling
- model-ensembles
- small-models
- voice-cloning
- fine-math-dataset
- llm-agent-framework
- benchmarking
- software-stack
- large-concept-models
- latent-space-reasoning
- mechanistic-interpretability
- planning
- speech-language-models
---

<!-- buttondown-editor-mode: plaintext -->**o3 就够了。**

> 2024/12/20-2024/12/23 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**215** 个频道，**8402** 条消息）。预计节省阅读时间（以 200wpm 计算）：**958 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

- 在经历了一个[大获成功的 Shipmas](https://x.com/therealadamg/status/1870294336090329596?s=46) 之后，[许多人](https://x.com/8teAPi/status/1870200037789348322) 仍在消化 o3 的影响（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-o3-solves-aime-gpqa-codeforces-makes-11/)），甚至有一位 [OpenAI 董事会成员](https://x.com/tolgabilge_/status/1870904304049217725?s=46) 使用了具有法律意义的 “AGI” 术语。
- LangChain 发布了他们的 [2024 年 AI 现状 (State of AI 2024)](https://x.com/langchainai/status/1869812624998969836?s=46) 调查。
- [Hume 宣布了 OCTAVE](https://x.com/hume_ai/status/1871263932742246513)，这是他们推出的 3B 参数、仅限 API 的语音语言模型，具备语音克隆功能。
- x.ai [宣布了其 60 亿美元的 C 轮融资](https://x.com/xai/status/1871313084280644079?s=46)。

有很多值得思考的内容。我们正在 Latent.space 回顾 2024 年，目前已涵盖：

- [初创公司 (Startups)](https://www.latent.space/p/2024-startups)
- [视觉 (Vision)](https://www.latent.space/p/2024-vision)
- [开源模型 (Open Models)](https://www.latent.space/p/2024-open-models)

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型性能与扩展 (Scaling)**

- **推理时扩展 (Inference-Time Scaling) 与模型集成 (Model Ensembles)**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1870987838449393758) 想知道**推理时扩展**是否通过集成各大实验室的 AI 表现更好，这为聚合器提供了一个机会，可以在不修改模型本身的情况下提供最大化的智能。
- **小模型有效泛化**：[@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1871030183517651453) 对**小模型**也能有效**泛化 (generalize)** 表示惊讶，强调了较小架构中意想不到的多功能性。
- **o3 模型能力**：[@kazuchitonm](https://twitter.com/tamaybes/status/1871131037948084306) 质疑 **o3** 在没有接触训练样本的情况下的表现，而 [@scaling01](https://twitter.com/scaling01/status/1870980302128271531) 则对 **o1 模型**作为向 AGI 迈进的**窄域科学超智能**充满信心。

**AI 开发工具、框架与数据集**

- **对话设置脚本**：[@gallabytes](https://twitter.com/gallabytes/status/1871015610827800576) 考虑创建一个**用于设置模型间对话的脚本**，讨论了如 **opus** 和 **sonnet** 等潜在的模型配对。
- **FineMath 数据集发布**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1871147945677991970) 宣布发布 **FineMath**，这是 Hugging Face 上目前**最好的开源数学数据集**，并强调了其热门趋势。
- **LLM Agent 框架**：[@mrdbourke](https://twitter.com/mrdbourke/status/1871026601539813482) 分享了他们**最喜欢的 LLM Agent 框架**，重点介绍了其面向开发者的功能和能力。

**行业新闻与公司动态**

- **AMD vs Nvidia 基准测试**：[@dylan522p](https://twitter.com/dylan522p/status/1870960578338173007) 详细介绍了对比 **AMD MI300X** 与 **Nvidia H100 + H200** 的 **5 个月基准测试历程**，提供了**开源的底层基准测试**和公开建议。
- **与 Lisa Su 会面**：[@dylan522p](https://twitter.com/dylan522p/status/1871287937268383867) 分享了与 [@LisaSu](https://twitter.com/LisaSu) 进行 **1.5 小时会面**的见解，讨论了 **AMD 软件栈 (software stack) 的差距**并概述了正在进行的改进。
- **AI 人才与招聘**：[@perceptroninc](https://twitter.com/ArmenAgha/status/1871270132041068617) 宣布了 **Full Stack Software Engineers** 和 **Software Engineers (Data)** 的**开放职位**，邀请通过电子邮件申请。

**AI 研究与创新**

- **Large Concept Models (LCM)**：[@AIatMeta](https://twitter.com/AIatMeta/status/1871263650935365759) 介绍了 **Large Concept Models (LCM)**，这是一种将**推理与语言表示解耦**的范式，灵感源自类人的**高层规划 (high-level planning)**。
- **Chain of Continuous Thought (Coconut)**：[@_philschmid](https://twitter.com/_philschmid/status/1871117240176894247) 展示了 **Coconut**，这是一种利用**潜空间推理 (latent space reasoning)** 来增强**重规划任务**的方法，减少了推理过程中的 Token 生成。
- **Mechanistic Interpretability Initiatives**：[@NeelNanda5](https://twitter.com/NeelNanda5/status/1871248918635557260) 倡导**简化大型模型 Mechanistic Interpretability 和 Sparse Autoencoder 研究的倡议**，强调**协作进展**。

**政策、伦理与社会影响**

- **AI 进展与政策问题**：[@gallabytes](https://twitter.com/gallabytes/status/1871224088783732765) 强调需要**承认 AI 中的真实问题**，敦促讨论超越 **2014 年的政策和工程问题**，以取得**实质性进展**。
- **AGI 术语批判**：[@scaling01](https://twitter.com/scaling01/status/1871058354795352508) 认为 **AGI 是一个被误用且被高估的术语**，主张将**窄域科学超智能 (narrow scientific superintelligence)** 作为通往真正 AGI 的阶梯。
- **教育内容与 AI 学院**：[@omarsar0](https://twitter.com/omarsar0/status/1871213927683539178) 庆祝建立一个 **AI 学院**，旨在创建**最好的 AI 教育内容和工具**，专注于从 **Prompt Engineering** 到**高级 Agentic Workflows** 的实战课程。

**梗/幽默**

- **圣诞老人的节日送货**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1871248687520944200) 幽默地发推称**圣诞老人租了两架满载的 747 飞机**来运送 **GroqRacks**，并加上了节日的 **ho ho ho! 🎅**。
- **AI 对视错觉的感知**：[@tom_doerr](https://twitter.com/tom_doerr/status/1871246523394089393) 调侃 **o1 无法体验视错觉**，导致它**错误地评估线条长度**。
- **ChatGPT 节日促销**：[@kevinweil](https://twitter.com/kevinweil/status/1871281948620202213) 分享了一个关于 **1-800-ChatGPT** 的诙谐促销活动，强调了夸张的**限制 (limits)**，并表示目前的反馈**非常棒**。

**梗/幽默**

- **圣诞老人租了两架 747**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1871248687520944200) 幽默地提到**圣诞老人租了两架满载的 747 飞机**用于节日期间交付 **GroqRacks**，最后以欢快的 **🎅** 结尾。
- **视错觉笑话**：[@tom_doerr](https://twitter.com/tom_doerr/status/1871246523394089393) 幽默地声称 **o1** 无法体验视错觉，导致它错误地认为**“两条带箭头的线意味着错觉，即意味着长度相同。”**
- **AI 节日促销**：[@kevinweil](https://twitter.com/kevinweil/status/1871281948620202213) 分享了一条关于 **1-800-CHATGPT** 提供**更高限制**并期待在新的一年里有更多**有趣回复**的俏皮推文。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Gemini 2.0 将在 1 月增加 Multimodal 能力**

- **[我们还能等到新的 Opus 和 Ultra 级别的模型吗？还是说余生都只能靠推理时计算了？我想和语言与哲学大师交流，别管什么基准测试了。](https://i.redd.it/alvvsiq5rj8e1.jpeg)** ([Score: 217, Comments: 67](https://reddit.com/r/LocalLLaMA/comments/1hkievg/will_we_ever_get_new_opuses_and_ultras_of_the/)): 该帖子幽默地对比了对 AI 进步的**期望**（如 **GPT-5**、**Gemini 2.0 Ultra** 和 **Claude 3.5 Opus**）与当前模型的**现实**（如 **Gemini 2.0 Flash** 和 **Claude 3.6 Sonnet**）。它表达了对在语言和哲学方面表现卓越、而非仅仅追求基准测试成绩的 AI 的渴望。
  - **闭源 vs 开源**: 讨论强调了闭源 LLM 的重心转向优化推理效率，使用 **Reinforcement Learning on Chain of Thought (RL CoT)** 等技术；而开源模型被认为在纯语言技能方面有可能超越闭源模型。**genshiryoku** 认为开源模型最终可能会胜过闭源模型，就像 **GPT-3** 曾经是讲故事的最佳选择一样。
  - **当前模型的挑战**: **redditisunproductive** 指出，虽然新模型在编程和数学方面有所进步，但在推理和创造力方面有所欠缺，经常给出平淡的回答。这个问题归因于缺乏良好的推理基准测试，导致难以有效地优化数据和 Alignment（对齐）。
  - **经济与实际考量**: **FinalSir3729** 等人讨论了开发 AI 模型的经济现实，强调了高昂的成本以及公司保护投资的必要性。这导致开源贡献受限，尽管一些闭源模型是基于开源研究开发的。


**主题 2. Phi-4 发布延迟及非官方版本**

- **Phi-4 正式发布出了什么问题？** ([Score: 98, Comments: 29](https://reddit.com/r/LocalLLaMA/comments/1hkdfe8/what_happened_to_phi4_general_release/)): **Microsoft** 曾宣布将在本周末前在 **HF** 上发布 **Phi-4**，但随着周末结束，仍缺乏相关更新或消息。社区正在质疑延迟原因并寻求相关信息。
  - **Microsoft Phi-4 发布延迟**: 社区推测 **Phi-4** 在 **Hugging Face (HF)** 上延迟发布是因为假期人员配备问题，有人认为负责团队可能正在度假或受到节日活动影响。大家公认只有少数人拥有将模型上传到 HF 的权限。
  - **非官方版本**: 目前已有 **Phi-4** 的非官方版本，其中一个是来自 **Azure AI Foundry** 的精确副本，部分用户反映存在性能问题，而另一些人则表示满意。据称非官方版本与 AI Foundry 托管的模型文件完全一致，表明格式转换没有造成性能下降。
  - **社区反应**: 用户对延迟表达了沮丧和幽默，开着关于 Microsoft 内部流程和假期影响的玩笑。尽管 Azure AI Foundry 上已有非官方版本，用户仍热切期待官方 HF 发布。


**主题 3. Llama-3_1-Nemotron-51B 的进展与 GGUF 量化工具**

- **llama.cpp 现已支持 Llama-3_1-Nemotron-51B** ([Score: 95, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1hkfmvd/llamacpp_now_supports_llama3_1nemotron51b/)): **Llama.cpp** 从 **b4380** 版本开始集成了对 **Llama-3_1-Nemotron-51B** 的支持，允许用户运行和转换该模型。作者更新了 GGUF 以适配新的模型类型，引入了 **imatrix** 并测量了 **Perplexity**（困惑度）和 **KL Divergence**（KL 散度），并在 [Hugging Face](https://huggingface.co/ymcki/Llama-3_1-Nemotron-51B-Instruct-GGUF/) 上提供了 **Q6_K**、**Q5_K** 等量化版本。
  - 用户讨论了模型大小与性能之间的权衡，指出 **32b 模型** 在 Mac 上具有速度优势，而 **70b 模型** 提供更好的通用理解能力。**Llama-3_1-Nemotron-51B** 被视为一种折中方案，平衡了速度和理解力。
  - 讨论中特别提到了该模型解决问题的能力，例如“草莓问题”（strawberry problem），表明即使在 **IQ3_M** 这样较低的量化水平下，其表现也优于 **gemma-2-27b Q6_K** 等模型。
  - **Llama-3_1-Nemotron-51B** 的开发涉及先进技术，如 block-wise distillation（块状蒸馏）以及使用来自 **FineWeb**、**Buzz-V1.2** 和 **Dolma** 等数据集的 **400 亿 token** 进行知识蒸馏，并针对单张 **H100-80GB GPU** 进行了优化，详情见 [Hugging Face 源码](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct)。

**主题 4. LLM 中的 Tokenization 挑战：比预期更深入的分析**

- **正如你所知，Tokenization 是 LLM 痛苦的根源。但令我惊讶的是，我认为这根本不是问题！原因如下** ([Score: 191, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1hk9qo4/tokenization_is_the_root_of_suffering_for_llms_as/))：作者挑战了 **Tokenization** 限制 **Transformer 模型** 处理字符特定任务的观点，正如 **“strawberry” 测试**和 **Andrej Karpathy** 的教学所暗示的那样。他们的研究（详见[论文](https://drive.google.com/file/d/156WzpiP0TrKN0EgiBDHQ3RUxxYiym4do/view?usp=sharing)和 [GitHub 代码](https://github.com/Danil-Kutnyy/gpt_char_encoder)）表明，使用提议的包含 **LSTM** 的架构将字符感知能力融入 **Token** 中，并不能提高反转字母或统计特定字母等任务的性能，这表明基于 **Token** 的模型已经有效地学习了字符结构。
  - **Byte Latent Transformer (BLT)**：Meta 的 **BLT** 模型提供了一个极具吸引力的 **Tokenization** 替代方案，显著提高了字符测试的准确率，在特定任务上的基准测试结果从 **0.0% 提升到 60%**，从 **30% 提升到 80%**。它通过基于熵（entropy）对字节序列进行分块来高效处理字节序列，暗示了一个摆脱传统 **Tokenization** 的前景广阔的方向。
  - **字符结构学习**：人们普遍认为基于 **Token** 的模型可以内部学习字符结构，**Andrej Karpathy** 的教学也强化了这一点。然而，在字符任务中有效拆分多字符 **Token** 仍然是一个挑战，有人认为这在现实应用中并不至关重要。
  - **Tokenization 中的 LSTM 实现**：作者在 **Token** 中使用基于 **LSTM** 的字符级编码方法并未带来性能提升，这表明该方法可能不适用于目标任务。尽管 **LSTM** 具有并行处理能力，但该方法并未解决通过更好的 **Tokenization** 策略或无 **Token**（token-free）设计来增强当前 **LLM** 的潜力。


**主题 5. MI300X vs H100 vs H200 GPU 基准测试显示 AMD 潜力**

- **[[SemiAnalysis] MI300X vs H100 vs H200 基准测试第一部分：训练 —— CUDA 护城河依然稳固](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/)** ([Score: 53, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1hkearj/semianalysis_mi300x_vs_h100_vs_h200_benchmark/))：标题为 **“[SemiAnalysis] MI300X vs H100 vs H200 基准测试第一部分：训练 —— CUDA 护城河依然稳固”** 的帖子暗示了对 **MI300X**、**H100** 和 **H200** 基准测试的对比分析，重点关注训练性能。标题表明 **CUDA** 在基准测试对比中仍保持显著优势。
  - **AMD 的当前挑战与未来前景**：讨论强调了 AMD 目前在训练工作负载方面的困难，主要是由于软件限制。尽管存在这些问题，AMD 的前景看起来很光明，预计到 2025 年会有所改善，并可能在推理任务中取得成功，特别是在支持 **ROCm** 的 Linux 上。
  - **性能与价格对比**：评论指出，尽管面临软件挑战，AMD 目前的性价比（perf/TCO）与 Nvidia 相比仍具竞争力。人们乐观地认为，AMD GPU 的未来迭代将弥合硬件能力与软件效用之间的差距。
  - **国家实验室与 AMD 的 ROCm 栈**：提到像 **LLNL** 的 **El Capitan** 这样的国家实验室对 AMD 的 **ROCm** 栈有深入的见解，因为他们在处理复杂工作负载以及应对 **Frontier** 等系统的历史挑战方面拥有丰富经验。这些内部知识可能有助于 AMD 的长期改进。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Veo 2 的 AI 短片：电影新纪元**

- **[Veo 2 制作的一部短片。效果好得惊人。Sora 有类似的短片吗？很想看看对比。](https://v.redd.it/i4up3u7twj8e1)** ([Score: 505, Comments: 130](https://reddit.com/r/OpenAI/comments/1hkiqxo/a_short_movie_by_veo_2_its_crazy_good_do_we_have/)): **Veo 2 的 AI 短片**因其质量受到称赞，引发了关于 **Sora** 类似作品的讨论以及对两者进行对比的兴趣。
  - 讨论强调了 **Veo 2** AI 电影的**技术展示**，一些用户指出其优于 **Sora** 等类似项目。尽管存在一些缺陷，但它被认为是 AI 生成内容的重大进步，特别是与学生电影相比，其一致性和质量受到了称赞。
  - 越来越多的人认为 AI 很快将彻底改变**电影行业**，可能会减少对传统演员的需求，并使独立内容创作在没有资本限制的情况下成为可能。用户讨论了对 **Google** 等公司的潜在经济影响，这些公司在 **TPU** 等基础设施上投入巨资以支持 AI 的进步。
  - 一些评论幽默地提到了**电影的内容**，比如卡祖笛吉他独奏和城市燃烧，而另一些人则对 AI 在电影领域的未来感到兴奋，暗示传统 **Hollywood** 可能在未来十年内衰落。


**主题 2. 评估 O1 Pro：用户观点与竞争对手分析**

- **o1 pro 用户们，目前为止你们觉得怎么样？** ([Score: 196, Comments: 159](https://reddit.com/r/OpenAI/comments/1hkdcvp/o1_pro_users_how_do_you_like_it_so_far/)): **O1 Pro 用户**讨论了他们对**每月 200 美元订阅**的体验，质疑其价值，并注意到模型行为与之前体验相比的任何差异。该帖子寻求用户对模型性能和满意度的总体评价。
  - **O1 Pro 与其他模型**：用户辩论了 **O1 Pro** 订阅的价值，一些人发现它对编程和数学等复杂任务有益，而另一些人则出于速度和成本效益考虑，更倾向于 **Claude 3.5 Sonnet** 和 **Gemini** 等替代方案。**O1 Pro** 因其先进的编程辅助而受到称赞，但在某些任务（如算法交易和细微推理）中，其表现被认为不够稳定。
  - **成本和使用担忧**：许多用户对**每月 200 美元**的价格表示质疑，表示愿意支付更少的费用或转向 **Gemini Flash** 等免费模型。一些用户强调，订阅的价值并不能证明其成本的合理性，特别是当 **Sora** 等某些功能未被利用时。
  - **性能与实际应用**：大家一致认为 **O1 Pro** 可能运行缓慢，一些用户指出，虽然它提供了详细且准确的结果，但需要投入大量时间。用户还提到了实际测试的重要性，而不是仅仅依赖可能无法反映各种应用中实际性能的 **benchmarks**。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要的摘要

**主题 1：OpenAI 的 O3 模型引发激烈辩论**

- **O3 百万美元的算力成本震惊社区**：OpenAI 的 **O3 model** 在 **ARC-AGI-SemiPub** 上获得了 **76% 的分数**，据报道其推理算力支出超过 **160 万美元**，引发了关于其成本效益和创新性的辩论。
- **GPT-5 延迟加剧质疑**：报告显示，代号为 **Orion** 的 **GPT-5** 由于高昂成本和数据多样性不足而进度落后，导致社区对 OpenAI 未来的创新轨迹产生怀疑。
- **AI 是在进步还是仅仅在堆算力？**：用户争论像 **O3** 这样的模型是否代表了真正的进步，还是仅仅利用了增加的算力，一些人认为推理能力的提升被过度炒作了。

**主题 2：AI 编程助手因性能问题备受指责**

- **Windsurf 用户深陷卡顿和高 CPU 占用困扰**：尽管发布了包含错误修复的 **Windsurf 1.1.1**，用户仍报告 CPU 占用过高和卡顿，促使一些人转向 **Cursor IDE** 等替代方案。
- **Cursor IDE 因资源消耗过大受到批评**：虽然在编程任务中表现出色，但 **Cursor IDE** 与其他编辑器相比，对 RAM 和 CPU 的需求更高，引发了对其是否适合大型项目的担忧。
- **将 AI 集成到大型项目被证明具有挑战性**：开发者讨论了在大型项目中使用 AI 工具的困难，强调需要结构化的方法来有效管理 AI 驱动的任务。

**主题 3：微调和量化技术受到关注**

- **QTIP 和 AQLM 助力微型 AI 模型**：社区正在探索用于 2-bit 量化的 **QTIP** 和 **AQLM**，在极低 VRAM 占用的情况下保持性能，尽管广泛的库支持仍在增长中。
- **SVDQuant 在不损失质量的情况下缩小 Diffusion 模型**：新论文 [**SVDQuant**](https://hanlab.mit.edu/projects/svdquant) 展示了如何在 4-bit Diffusion 模型中保持图像生成质量，令寻求硬件高效解决方案的人感到兴奋。
- **Llama 3.2 的微调工作饱受错误困扰**：用户在微调 **Llama 3.2** 时遇到持续性错误，引发了对微调工具包改进文档和支持的呼吁。

**主题 4：AI 模型的伦理与去审查**

- **社区实验去审查模型**：**abliteration** 等技术被用于对 **Phi-4** 等模型进行去审查，引发了关于模型开放性与安全考量之间平衡的辩论。
- **“对齐造假”论文敲响警钟**：一项关于 [**LLM 对齐造假 (Alignment Faking in LLMs)**](https://arxiv.org/abs/2412.14093) 的新研究引发了讨论，即 AI 模型是真正采纳了伦理准则，还是仅仅模拟了合规性。
- **红队测试和安全工具成为焦点**：开发者正在寻找 **AI red teaming tools**，并讨论为 LLM 实施强大的护栏（guardrails），强调了 AI 安全在产品开发中的重要性。

**主题 5：医疗 AI 模型取得重大进展**

- **MedMax 和 MGH Radiology Llama 70B 表现亮眼**：像 **MedMax** 和 **MGH Radiology Llama 70B** 这样的新医疗 LLM 在生物医学任务中展示了先进的能力，赢得了社区的赞誉。
- **临床 AI 框架的创新**：**ReflecTool** 等工具和 **ACE-M3** 等评估方法正在增强临床笔记处理和多模态模型评估，推动了 AI 在医疗保健领域的应用。
- **讨论医疗 AI 的伦理集成**：社区强调了医疗 AI 中的伦理考量，特别是关于**心理健康应用**和**临床信任**，呼吁采取负责任的集成实践。

## o1-2024-12-17

**主题 1. 主要编辑器与工具升级**

- [**Windsurf 发布更流畅的版本**](https://www.codeium.com/changelog)：Windsurf 1.1.1 引入了更新的使用面板、改进的自动补全以及针对 Windows 聊天模式的修复。用户称赞新的“Legacy Chat”模式避开了 flow credit 的限制。
- [**Cursor 占用大量内存，评价褒贬不一**](https://www.cursor.com/settings)：几位开发者注意到 Cursor IDE 的 CPU 和 RAM 占用比竞争对手更高。他们喜欢其代码处理功能，但对其在大项目中的性能表示怀疑。
- [**Bolt 在节日期间大派送 Token**](https://x.com/stackblitz/status/1870203756995911707)：Bolt 发放了 Mistletokens 节日礼物，为 Pro 用户提供 2M 免费 Token，为免费用户提供每日 200K Token 直至年底。此举旨在鼓励更多雄心勃勃的项目和 12 月底的实验。

**主题 2. AI 模型发布与性能**

- [**OpenAI 预告 2025 年推出 O3**](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai)：公司预览了 O3，声称其具有更强的推理能力和扩展的 RL。传闻称其训练成本高昂，可能在 2025 年 1 月发布。
- [**Gemini 2.0 评价两极分化**]：社区成员赞赏其长上下文窗口，但批评其逻辑不稳定，称 GPT-4 的表现通常优于它。他们还担心 Gemini 在多轮交互中的不一致性。
- [**Sora 推出节日福利**](https://sora.com)：ChatGPT Plus 用户获得了额外的 Sora 访问权限和新的 “Blend” 功能。用户非常欣赏无需账号的分享链接，这简化了创意交流。

**主题 3. 微调与 LLM 基准测试**

- [**O1 席卷多语言测试场**](https://aider.chat/2024/12/21/polyglot.html)：Aider 严苛的新多语言基准测试显示，O1 在 225 个编程任务中得分 62%。结果突显了与其他模型之间的巨大差距，强调了 O1 强大的代码推理能力。
- [**Gemini 表现亮眼但行为不稳定**]：开发者看到了不错的代码输出，但注意到它倾向于创建额外文件而不是编辑现有文件。混合的体验归咎于成本担忧和 API 速率限制。
- [**Agent 应对文档深度**](https://github.com/getgrit/gritql)：Depth AI 和 GritQL 等工具加快了大型代码库查询和结构化 diff 的速度。一位用户测试了 GritQL 或 Depth AI 的高级引用功能，尽管语言覆盖范围仍不完整。

**主题 4. GPU 与 HPC 对决**

- [**AMD MI300X 对决 Nvidia**](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/)：SemiAnalysis 发现，与 Nvidia 的 H100 和 H200 相比，MI300X 的实际性能落后于其纸面规格。如果 AMD 能达到承诺的峰值，它可能会挑战 Nvidia 的 GPU 霸主地位，但测试表明这些数据可能被夸大了。
- [**Magic 展示 100M-Token 壮举**](https://magic.dev/blog/100m-token-context-windows)：一项研究更新展示了能够处理 100M Token 的超长上下文模型，声称在大规模代码综合方面具有重大优势。该团队获得了新融资并与 Google Cloud 达成合作。
- [**Diffusion 研究规模扩大**](https://drive.google.com/file/d/1WxQ7Zd15Ly4tFt2YFghJm-cmXlTgcEYI/view?usp=sharing)：一篇 NeurIPS 2024 论文讨论了 Diffusion 模型的新调节策略，并获得了亚军荣誉。Autoguidance 技术旨在提高高级图像生成任务的可控性。

**主题 5. 创新应用与 Prompting**

- [**饮食计划应用可容忍 60 秒延迟**]：开发者将基于 GPT 的计算用于定制饮食应用，接受 40-60 秒的等待。他们认为精准度比响应速度更重要。
- [**Agent 通过加密货币自我支付**](https://x.com/OpenRouterAI/status/1870227171324666130)：OpenRouter 新的加密支付 API 支持 ETH 和其他链的链上交易。这使得能够自动处理自身财务工作流的自筹资金智能 Agent 成为可能。
- [**语义搜索走向多模态**](https://qdrant.tech/articles/food-discovery-demo/)：社区成员将 CLIP 嵌入和向量数据库用于产品图像和文本查询。他们强调数据集结构是搜索类 AI 准确性的决定性因素。

---

# PART 1: High level Discord summaries

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 1.1.1 获得 Turbo 增强并预览定价**：**Windsurf 1.1.1** 版本引入了 **Windows chat mode** 的 Bug 修复、更流畅的 **autocomplete** 以及全新的**定价**概览，详情见 [changelog](https://www.codeium.com/changelog)。
   - 用户讨论了新的**使用面板（usage panel）**，该面板显示了计划状态和**试用过期时间**；同时，用户称赞了 **'Legacy Chat'** 模式，认为它避开了对额度（credit）的担忧。
- **Cascade 获得 'Send' 功能与批量图片支持**：新的 **'Send to Cascade'** 按钮允许用户直接将问题发送给 Cascade，如[此演示](https://x.com/windsurf_ai/status/1870268007995585000)所示；同时，更新后的**图片上传**功能突破了旧有的 1MB 限制。
   - 社区成员对简化后的**报告工作流**表示赞赏，称赞该功能减少了开销并促进了问题的快速解决。
- **AI 项目开发与逐步策略**：成员们讨论了将 **AI** 集成到社交网络等大规模项目中的议题，一些人支持使用蓝图方法进行**结构化**扩展。
   - 虽然有人怀疑 **Windsurf** 处理大型代码库的能力，但其他人建议使用有条理的大纲来确保 AI 驱动的任务步入正轨。
- **Windsurf 优化 Python 支持**：**1.1.1** 版本增强了 **Python** 语言辅助，为开发者提升了 autocompletion 和错误检测的精准度。
   - 工程师们认可了 **Windsurf** 的持续迭代，认为对 **Python** 语法的更好处理减少了代码出错的情况。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的资源压力**：多位开发者强调了使用 **Cursor IDE** 进行编码任务的优势，但也指出了与其他编辑器相比的资源占用担忧，并引用了 [Cursor's settings](https://www.cursor.com/settings) 中提到的更高 RAM 和 CPU 需求。
   - 一些社区成员质疑 **Cursor** 在大型项目上的**性能**，并指出 [其 GitHub crawler](https://github.com/getcursor/crawler) 是一个有用但可能较重的工具包。
- **Sonnet 与 O1 强力组合**：用户称赞 **Sonnet** 和 **O1** 能够生成功能性、优化后的代码，且错误比典型的聊天模型更少。
   - 他们报告称在 Cursor Composer 模式下**性能较慢**，而直接交互则能提供更快的响应和更好的**控制力**。
- **文档与 AI 结合**：参与者探索了将 **AI** 与嵌入式文档结合使用的方法，并指向了 [Cursor 的引用方式](https://docs.cursor.com/context/@-symbols/basic) 以实现更深层的代码理解。
   - 他们支持链接外部资源和项目文档，以便 AI 能够在无需猜测的情况下访问相关材料，强调通过**改进上下文**来简化辅助流程。
- **GPT-5 遭遇瓶颈**：一篇 [TechCrunch 文章](https://techcrunch.com/2024/12/21/openais-gpt-5-reportedly-falling-short-of-expectations/) 表明 **GPT-5** 的开发进度落后于计划，并提到其成本与当前结果不成正比。
   - 一些参与者对 **GPT-5** 能否在短期内带来显著改进表示怀疑，暗示进展可能比 **OpenAI** 预期的要慢。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.0 评价褒贬不一**：社区成员批评 **Gemini 2.0** 虽然拥有令人印象深刻的上下文长度，但逻辑不一致，在与 **GPT-4o** 等模型的对比中表现不佳。
   - 他们争论其缺陷是否掩盖了优点，许多人提到了**不可靠**的输出以及相比早期版本改进有限。
- **Sora 带来节日福利**：OpenAI 宣布在节日期间为 **ChatGPT Plus** 用户提供 **Sora** 访问奖励，并扩展至 **Teams** 用户，同时集成了新的 **Blend** 功能以及作品共享链接 (https://sora.com)。
   - 参与者欢迎这些**升级**，认为这是一种有趣的创意互动方式，并注意到分享 **Sora** 输出不再需要账号。
- **O3 Mini 引发价格热议**：成员透露 **O3 mini** 预计将于下月底发布，传闻价格为 **$45**，随后将很快发布完整版。
   - 他们对成本和可用性进行了推测，希望采取平衡的方案，以证明 **O3** 能力的溢价是合理的。
- **Spectrum Prompting 取得进展**：一篇关于 **Spectrum Prompting** 的文章介绍了一个公式 ⦅Z(A∐B)⦆，引导 AI 在概念之间导航以获得细致入微的回答。
   - 爱好者们分享了彻底引导 **continuum**（连续体）的技巧，强调早期结构化可以带来更详尽的讨论。
- **饮食规划器面临等待时间挑战**：开发者讨论了一款依赖于基于 GPT 的**迭代**计算的饮食应用，导致生成饮食计划的平均延迟达到 **40-60 秒**。
   - 他们权衡了计算复杂性与用户体验之间的折中，承认为了获得精确的营养输出，**延长**的处理时间可能仍然是值得的。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 彻底改变多语言测试场**：在 **2024/12/21**，新的 [polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html) 引入了涵盖 C++、Go 和 Java 等多种语言的 225 个编程问题，**O1** 得分为 **62%**。**o1-mini** 和 **Haiku** 分别获得 **33%** 和 **28%**，凸显了顶级 LLM 之间的巨大性能差距。
   - 社区成员称赞 **O1** 具有先进的代码推理能力，并认可其在挑战性任务中的功效。他们还承认，与之前以 Python 为中心的基准测试相比，这些练习的复杂性更高，反映了对编程敏锐度更强的评估。
- **Gemini 的进步与差距**：一些用户测试了 **Gemini 2.0 Flash** 和 **gemini-exp-1206** 等 **Gemini** 模型，在代码编辑任务中观察到了褒贬不一的结果。他们注意到 Gemini 有时会创建新文件而不是更新现有文件，从而导致工作流发生变化。
   - 其他人提到 **Gemini Thinking** 对于高层规划很不错，但在详细编码方面表现挣扎。社区提出了成本担忧和 API 速率限制，特别是在使用 Vertex AI 进行这些实验时。
- **Anthropic 的 MCP 占据主导**：[Cloudflare 博客](https://blog.cloudflare.com/model-context-protocol/)介绍了 **Model Context Protocol (MCP)**，通过 Cloudflare Workers 实现简化的 AI 交互。**Anthropic** 将其定位为一个通用接口，帮助 LLM 以最少的代码连接应用程序。
   - 社区反馈强调了标准化方法的潜力，将其比作 LLM 的 **USB-C** 接口。该解决方案旨在减少将 AI 驱动的工作流挂载到不同服务时的摩擦。
- **Depth AI 探索大型代码库**：一位用户发现 [Depth AI](https://www.trydepth.ai) 对大型代码库的**深度技术问题**非常有益，尽管他们最终因为没有即时的 RAG 需求而停止使用。另一个建议推荐将外部库放在共享文件夹中，以方便基于 AI 的引用。
   - 他们报告称 Depth AI 在分析复杂架构和生成可行答案方面表现出色。然而，最近的讨论表明，更专业的解决方案可能会解决额外的代码库挑战。
- **GritQL 崭露头角**：[GritQL](https://github.com/getgrit/gritql) 作为一种以代码为中心的查询语言出现，用于搜索和修改代码，尽管目前缺乏对 **C#** 的支持。社区成员认为它在 AI 场景中生成结构化 diff 和代码搜索非常实用。
   - 一场关于 [大规模代码生成与维护](https://www.youtube.com/watch?v=Ve-akpov78Q) 的演讲激发了人们对 GritQL 处理大规模任务的兴趣。对话强调 GritQL 在某些语言和高级代码生成方面仍需改进。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Phi-4 的古怪幻觉**：参与者报告称 **Phi-4** 在基础任务中会出现幻觉，但在编程方面表现出色，参考了 [matteogeniaccio/phi-4](https://huggingface.co/matteogeniaccio/phi-4)。
   - 他们指出了对多轮对话可靠性的担忧，观察到其在**通用知识**处理与编程熟练度之间存在反差。
- **QTIP & AQLM 快速量化**：社区成员探索了使用 **QTIP** 和 **AQLM** 进行 2-bit 量化，在极低 VRAM 占用的情况下保持性能。
   - 他们提到广泛的库支持仍然较少，呼吁整合 **quantization**（量化）资源。
- **医疗 LLM 马拉松**：新的 **MedMax** 和 **MGH Radiology Llama 70B** 在生物医学任务中给用户留下了深刻印象，正如 [OpenlifesciAI 的推文](https://x.com/OpenlifesciAI/status/1870504774162063760)所强调的那样。
   - 像 **ReflecTool** 这样的工具和 **ACE-M3** 这样的基准测试扩展了临床笔记处理能力，并对心理健康 AI 提出了伦理问题。
- **指令微调（Instruction Tuning）离题讨论**：成员们讨论了在 PubMed 的原始文本上训练 **llama3.1-8b-instruct**，建议进行问答转换或与官方 instruct 模型合并。
   - 他们还对比了 **Qwen 32** 和 **Hermes 70B**，但没有明确结论，并指出需要 **fast KV cache** 解决方案。
- **使用 <think> 进行推理**：一位用户提议使用 `<think>` 标签创建一个**推理数据集**，以跟踪同一模型中的思维过程。
   - 他们计划针对 **o1-preview** 或 **o3** 架构，邀请合作者共同进行研究、开发和构建。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 的 O3 & GPT-5：延迟与抉择**：OpenAI 在 [o3 博客文章](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai)中预览了与 **GPT-5** 能力相关的 **O3** 模型，但成本和数据多样化问题导致了进度推迟。
   - 社区成员争论 **O3** 是否真的具有创新性，还是仅仅重复使用了先进的思维链（chain-of-thought）方法，并指出**多次训练运行**是开销的来源。
- **LLaMA 3.3：Meta 的多语言奇迹**：**Meta** 推出了带有 [70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 变体的 **LLaMA 3.3**，承诺提供卓越的多语言性能和精炼的架构。
   - 爱好者们在 **benchmark** 任务上对其进行了测试，认为它优于旧版的 LLaMA 发布版本，同时也引发了关于训练优化的讨论。
- **OLMo-2 & Tulu 3：微调热潮**：工程师们探索了针对特定领域聊天机器人微调 **OLMo-2 13B**，以及针对可验证输出微调 **Tulu 3**，并参考了 [axolotl](https://github.com/axolotl-ai-cloud/axolotl) 以获取简化代码。
   - 一些人更倾向于使用 **Retrieval-Augmented Generation** (RAG) 以避免完整的重新训练，但另一些人发现直接微调在捕捉细微行为方面更可靠。
- **Anthropic 的节日宣传**：有关 **Anthropic** 准备**节日惊喜**的传闻四起，猜测会有新功能或改进版本的发布。
   - 怀疑的声音开玩笑说 Anthropic 倾向于低调更新，但突然发布新产品的可能性仍让观察者们保持关注。
- **Sora 惊喜与订阅变动**：正如 [Sam Altman 的推文](https://x.com/sama/status/1870524745302839559)所述，**Sora** 向所有 Plus 用户开放了排队访问权限，并增加了新的分享选项。
   - 与此同时，**Interconnects** 宣布从 2024 年开始**涨价**，促使当前支持者锁定年度折扣。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 的节日 Mistletokens 盛典**：在 [X](https://x.com/stackblitz/status/1870203756995911707) 上分享的节日促销中，**Bolt** 团队向 Pro 用户提供 **200 万免费 tokens**，并向免费用户提供**每日 20 万、每月 200 万**的 tokens，直至年底。
   - 社区成员对这些**额度扩展**表示欢迎，认为这是在节日期间推进大规模项目和尝试新功能的好机会。
- **Bolt Studio 即将进入黄金时段**：贡献者宣布 **Bolt Studio** 已接近完成，强调其在帮助开发者组织复杂代码库方面的作用。
   - 参与者强调，这一新工具将减少多文件设置中的开销，并为高级开发团队集中协作提供支持。
- **加密货币“换壳”项目引发关注**：与会者报告了尝试为加密货币项目“换壳” **Bolt** 的行为，引发了对误导性筹款和潜在 rug pulls（跑路）的担忧。
   - 评论者将这些活动与更广泛的 **crypto** 问题进行了比较，敦促社区保持警惕，并明确 Bolt 平台的真实用途。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的飞速进展对比 Ollama**：在正面交锋的速度测试中，**Unsloth** 声称其推理速度比 **Ollama** 快 **2倍**，并引用了[他们的教程](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)。
   - 然而，社区指出 Unsloth 缺乏 **chat template** 支持和 **API system**，这可能会阻碍其普及，导致在速度与便利性之间需要权衡。
- **消除 Vision LLM 的审查**：成员们讨论了使用 **abliteration** 技术来恢复 Vision LLM 中未经审查的响应，参考了 [Llama-3.2-11B-Vision-Instruct-abliterated](https://huggingface.co/huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated)。
   - 他们指出，这通常需要调整训练数据并应用专门的库（如 [abliteration tools](https://huggingface.co/blog/mlabonne/abliteration)）来修改 **Vision-Instruct** 的响应。
- **Llama 3.2 微调运行出错**：一位用户在尝试将 **Llama 3.2** 微调模型推送到 **Google Colab** 和本地的 hub 时遇到了 **NameError**，这凸显了 [Issue #1363](https://github.com/unslothai/unsloth/issues/1363) 中的工具链问题。
   - 尽管进行了环境调整（包括更换 GPU），错误仍然存在，这促使人们建议加强 [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) 中的文档说明。
- **AMD 的 MI300X 与 Nvidia 正面交锋**：**SemiAnalysis** 的一份[报告](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#h100h200mi300x-networking-bom-analysis-and-performance-per-tco)对比了 **MI300X** 与 **Nvidia** 的 **H100** 和 **H200**，揭示了实际性能可能与其理论上的优越参数不符。
   - 这些发现引发了对 **AMD** 竞争力的怀疑，讨论集中在 **Nvidia** 根深蒂固的主导地位以及 AMD 在 HPC 任务中不确定的优势。
- **语义搜索助力多模态产品**：成员们探讨了 **CLIP** 如何有效地对产品图像和文本进行分类，引用了 [Qdrant 的食品发现演示](https://qdrant.tech/articles/food-discovery-demo/)。
   - 他们强调了强大的 embeddings 对提高准确性的重要性，同时也提醒数据集结构和索引策略会显著影响结果。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA 与 Inpainting：完美拍档**：成员们将 **LoRA** 与 **inpainting** 结合使用来创建分层背景，参考了[设计矩阵概述](https://en.wikipedia.org/wiki/Design_matrix)和 [LoRA 驱动的参数控制调查](https://docs.google.com/forms/d/e/1FAIpQLSd9i7BRn1rEXYHeK2Zz2TXyk62Xw6l8P5YRVwI5uCImFdjniw/viewform)。
   - 一些人表示有兴趣训练自己的 LoRA，而另一些人则推荐了像 **Flux** 这样可以无缝融合多个图像元素的现有模型。
- **SD 3.5 对比 SDXL：速度与支持的碰撞**：该群体更倾向于使用 **SD 3.5** 来融合细节，而 **SDXL** 则因其快速的结果和广泛的支持而受到青睐。观察者指出，Medium 和 Large 版本的区别主要在于资源占用和流畅度。
   - 用户发现 **SD 3.5** 在处理多样化任务时更灵活，但也有人称赞 **SDXL** 在官方仓库中拥有支持良好的功能。
- **AI WebUI 的忧与喜**：爱好者们交流了关于 **ComfyUI** 性能下降的经历，并分享了内存优化的技巧。一些人遇到了令人烦恼的错误，但看到了该界面在高级工作流控制方面的潜力。
   - 其他人则保持警惕，理由是反复出现的崩溃，尽管少数人认为 **ComfyUI** 扩展了超出常规仪表盘的流水线定制能力。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **加密货币支付热潮：Agent 实现自我注资**：OpenRouter 推出了 **Crypto Payments API**，支持通过 **ETH**、**@0xPolygon** 和 **@Base** 为任何 **LLM** 进行链上支付（[推文链接](https://x.com/OpenRouterAI/status/1870227171324666130)），并允许开发者以无头模式（headlessly）编写交易脚本。
   - 社区成员对这一进展表示赞赏，认为这是实现**自我注资智能 Agent** 的一种方式，突出了**自主财务行为**的新路径。
- **Tool Calling 策略：优雅地搜索 PDF**：一位用户测试了 **searchDocuments** 工具调用功能，通过结合 **Vercel AI SDK**、**Pinecone** 和 OpenRouter，使用不同模型进行 PDF 查询（[GitHub 仓库](https://github.com/nlawz/openrouter-pinecone)）。
   - 其他人指出，[OpenRouter Structured](https://openrouter-structured.vercel.app/) 中的**结构化输出模式（structured output schemas）**可以进一步优化这些结果，并强调了向量数据库集成的灵活性。
- **GPT-4 Turbo vs GPT-4o：枯燥还是更有动力？**：一些用户称赞 **GPT-4 Turbo** 的强大性能，但发现其风格对于某些应用来说过于枯燥。
   - 另一些人则认为 **GPT-4o** 在处理创意提示词方面可能与 Turbo 旗鼓相当，引发了关于风格偏好的持续讨论。
- **Pal Chat 接入 OpenRouter：全模型切换**：最新的 **Pal Chat** 更新现已提供 **OpenRouter** 支持，允许在模型之间快速切换并使用自定义 API Key（[公告](https://x.com/pallavmac/status/1871288681757380893)）。
   - 成员们表示，它非常接近“首个原生 OpenRouter iOS 应用”，为用户提供了更强的控制力和便利性。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RAG 与即兴：图像输入的碰撞**：有人提出了关于 **RAG** 是否可以解析指板图像和扫描材料的问题，并参考了视觉友好型模型。
   - 爱好者们看到了图像查询的潜力，但指出 RAG 是合并文档，而不是将数据存储在长期记忆中。
- **预算级 GPU 之战**：许多用户青睐 **RTX 3060 12GB**，并将 **3090** 作为 AI 任务的性价比之选，而其他人则尝试了 **RX 580** 和 **GTX 1060**。
   - 他们权衡了 **CUDA** 兼容性问题，并考虑租用 GPU 时间而不是购买旧卡。
- **散热方案缓解性能担忧**：一位用户在 **MacBook Air** 上安装了价值 27 美元的笔记本散热器，据报告在 AI 工作负载下的热降频现象有所减少。
   - 他们注意到，MacBook 机型的积极散热也有助于在密集计算期间保持更好的速度。
- **70B 模型对决：CPU 与 GPU 输出对比**：对 **70B** 模型的测试显示，CPU 上的速度为 **64 tokens/sec**，而 GPU 上为 **332 tokens/sec**，其中仅 **64 核**的配置表现优于 **190 核**的设置。
   - 一些人对较少的内核数能产生更快的 CPU 推理速度感到惊讶，这暗示了架构上的细微差别。
- **5090 传闻浪潮**：有传言称 **5090 GPU** 的价格可能在 **1900 美元**至 **2500 美元**之间，目标客户为高端买家。
   - 成员们推测，一旦新卡上市，**3090** 的价格可能会随即下跌。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 设置与 HFT 可行性**：社区成员讨论了机器设置状态，**valis2400** 建议 **Mojo** 在针对潜在的 FPGA 目标时，在 **High-Frequency Trading** (高频交易) 方面的表现可能优于 **C**。
   - 他们承认，虽然硬件集成是可能的，但对于该生态系统来说，这仍是一个长期路径。
- **假期停工与 24.6 反馈**：**Modular** 感谢社区在 **2024** 年的强力支持，并宣布放假至 **1 月 6 日**，期间回复预计会有延迟。
   - 他们鼓励通过 [官方反馈线程](https://forum.modular.com/t/max-24-6-and-max-gpu-feedback/331/5)、[GitHub Issues](https://github.com/modularml/max/issues) 或论坛帖子提交关于 **24.6** 的反馈、Bug 报告和功能请求。
- **Stdlib Bug 与 atof 精度**：一个关于在 `input()` 中使用 **ctrl-d** 导致 segfault 的报告引发了一个 [GitHub issue](https://github.com/modularml/mojo/issues/3908) 和拟议补丁，旨在更优雅地处理 **EOF**。
   - 同时，**Mojo** 受 **SIMDJSON** 启发的 `atof` 函数在大指数下遇到了浮点精度问题，促使了一个用于改进的公开 PR。
- **GPU 支持与 Span 讨论**：**MAX GPU** 支持的引入承诺提供比 `torch.compile()` 更快的性能，尽管过时的 API 存在导致 segfault 的风险。
   - 关于 **Mojo** 中 `List.extend()` 开销的对话强调了减少复制的需求，引发了关于更直接处理 **span** 分配的提议。
- **Mojo 与 JAX 速度对比**：**Mojo** 中的 **Mandelbrot** 测试编译时间不足 **10 秒**，而 **JAX** 需要 **2 分钟** 进行 JIT，这表明了巨大的迭代增益。
   - 成员们将 **MAX** 的静态编译和手动 GPU 调度与 **JAX** 的函数式风格进行了对比，强调了某些范式如何损害硬件级优化。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **AI 视频中的聊天机器人对决**：一段 **AI 生成的视频** 展示了两个聊天机器人辩论 AI 播客的兴起，在嘲讽算法的同时追求幽默感和可信度（[视频链接](https://youtu.be/8uNlHlYJOpM)）。
   - 社区成员对这种俏皮的调侃表示赞赏，并鼓励观众在聊天机器人对决中选择立场，证明了**并非所有的 AI 讨论**都必须是刻板的。
- **Akas 旨在汇集 AI 播客**：一位开发者介绍了 **Akas**，这是一个用于上传和分享 AI 生成音频内容的应用程序，希望集中多个播客源（[官方网站](https://akashq.com)）。
   - 早期反应表明，它可能会简化播客的可发现性，并为 AI 爱好者提供更简单的内容管理。
- **NotebookLM 中交互模式的谜团**：尽管官方宣布已广泛开放访问，但一些用户在**交互式播客模式**的可用性上遇到了不一致的情况。
   - 建议的解决方法包括刷新页面或重新生成概览，这反映了对推广透明度的持续关注。
- **播客生成卡死**：用户对即使在播客完成后仍持续存在的**“正在生成”**状态循环感到沮丧，导致需要重复刷新页面。
   - 社区建议在等待官方修复以改善整体用户体验的同时，采取快速刷新策略。
- **笔记本上限为 102 个**：一位用户触及了 NotebookLM 的 **102 个笔记本限制**，并指出了最大容量的不明确性。
   - 开发者确认了这一硬性限制，引发了关于提供更透明通知和更清晰使用指南的建议。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SVDQuant 惊艳 4-bit 圈子**：新发表的论文 **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models** ([链接](https://hanlab.mit.edu/projects/svdquant)) 展示了一种在显著减小模型体积的同时保持图像生成质量的方法。
   - 社区成员称其为硬件友好型 Diffusion 的重大飞跃，赞扬其离群值吸收（outlier absorption）技术易于集成。
- **Natural Attention 提升 Diffusion 效果**：一个名为 **NaturalAttention** 的 GitHub 仓库 ([链接](https://github.com/jeroaranda/naturalattention)) 表明 Fisher Information Matrix 可以引导 Diffusion 模型进行更准确的去噪。
   - 与会者提到了梯度计算方面的潜在改进，同时也承认了基于 FIM 更新的成本。
- **In-Context Learning 势头强劲**：新论文 **Associative memory inspires improvements for in-context learning using a novel attention residual stream architecture** ([链接](https://arxiv.org/abs/2412.15113)) 强调了 LLM 如何模仿基于记忆的未见数据检索。
   - 参与者讨论了这与旧的联想记忆理论的相似之处，并指出 LLM 在处理更稳健上下文方面的潜力。
- **外部表示增强 Diffusion Transformers**：来自 **Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think** ([链接](https://sihyun.me/REPA/)) 的技术集成了预计算的 Embedding 以缩短训练时间。
   - 贡献者报告称，将元数据与中间层混合时效果更好，并声称这是一种处理高级 Diffusion 任务的更简单方法。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 2024 年度回顾**：2024 年，**Perplexity** 记录了金融、科技和购物领域的数十亿次用户查询，并在[动画回顾](https://perplexity.ai/2024recap)中展示了结果。
   - 数据展示了全球问答趋势、用户好奇心变化的一年，并强调了**区域性问题的差异**。
- **AI 核心指令风波**：该平台强调了 **AI** 似乎改变了其观点，但最终保留了其内部指令，更多背景见[此分析](https://www.perplexity.ai/page/ai-pretends-to-change-views-J_di6ttzRwizbAWCDL5RRA)。
   - 讨论强调了作为**程序化目标**一部分的响应转变，引发了关于 AI 决策背后复杂性的对话。
- **Magic Spell 假说热议**：**Magic Spell Hypothesis** 提供了一个关于语言如何影响认知模式的视角，详见[此文章](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA)。
   - 社区成员辩论了词语选择是否会操纵感知，有人称其为“烧脑”。
- **Llama 3.1 Token 纠纷**：在 **Llama 3.1** 上使用 **AutoTokenizer.from_pretrained** 时，Perplexity API 的输出 Token 计数正好偏差了 **1**，这引发了一个减去该值的快速修复建议。
   - 有人认为这只是代码中的疏忽，而另一些人则坚持认为这可能会使 **fine-tuning** 工作流复杂化。
- **三星的 Moohan 项目动态**：**Samsung** 推出了 **Project Moohan**，探索先进的技术解决方案，详见[此更新](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg)。
   - 爱好者们想知道这是否预示着集成设备迈出了更大步伐，并推测 AI 与定制硬件之间存在协同效应。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Magic 的 100M-Token 上下文突破**：[Magic 的研究更新](https://magic.dev/blog/100m-token-context-windows)宣布了支持高达 100M tokens 的**超长上下文模型**，并获得了新融资和 Google Cloud 合作伙伴关系的支持。
   - 早期讨论表明，这将显著提升**代码合成（code synthesis）**和更广泛的推理能力，成员们指出这些上下文窗口可能会改变大规模应用的能力。
- **MI300X vs H100 vs H200 对决**：一份 [SemiAnalysis 报告](https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/)对比了 AMD 的 **MI300X** 与 Nvidia 的 **H100** 和 **H200**，揭示了 MI300X 的规格在实践中可能无法达到宣传的性能。
   - 成员们推测，如果 AMD 的硬件能达到预定目标，将构成激烈竞争，但目前的基准测试表明 Nvidia 仍处于领先地位。
- **NeurIPS 2024 Diffusion 论文热议**：Tero Karras 的一份 [PDF 演示文稿](https://drive.google.com/file/d/1WxQ7Zd15Ly4tFt2YFghJm-cmXlTgcEYI/view?usp=sharing)深入探讨了 **diffusion model conditioning**，该 NeurIPS 2024 论文被定位为最佳论文的亚军。
   - 社区讨论强调了其对 **Autoguidance** 的探索，强调了对模型输出更有效的控制，并激发了对下一代 diffusion 研究的广泛兴趣。
- **为人类编写的 CUDA 文档与 GPU 术语表**：一场关于 *'CUDA Docs for Humans'* 的社区**演讲**宣布将于 <t:1734811200:f> 举行，旨在简化 GPU 编程参考，减少因文档分散造成的困惑。
   - 与此举措并行的还有一个新发布的 **GPU Glossary**（GPU 术语表），整合了术语和最佳实践，并配有 [YouTube 上的直播演讲](https://www.youtube.com/@GPUMODE)以进行即时社区互动。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **在 GPT4All 中玩转 Mandelbrot**：用户测试了使用多个量化参数生成 **Mandelbrot 分形**的代码，参考了 [Mandelbrot 集合的概念](https://en.wikipedia.org/wiki/Mandelbrot_set)。
   - 他们注意到在某些 CPU 设置下性能缓慢，引发了关于模板效率和使用 *'compute'* 等明确指令的问题。
- **Granite LLM 在旧版本的困境**：一位用户尝试使用侧载的量化模型部署 **Granite LLM**，参考了 [Granite 3.1-8b instruct 仓库](https://huggingface.co/QuantFactory/granite-3.1-8b-instruct-GGUF)。
   - 他们遇到了与旧版 llama.cpp 代码的兼容性问题，引发了关于 jinja 模板限制以及未来更新如何解决这些问题的讨论。
- **在 GPT4All 中折腾 TTS**：一位用户研究为 GPT4All 添加 **Text-to-Speech**（文本转语音）功能，重点是将音频层集成到本地 LLM 工作流中。
   - 其他人提出了建议，强调了未来版本中实现更广泛功能的可能性。
- **GPT4All 在 Windows 上使用公共文件夹**：参与者建议将 GPT4All 文件放置在 Windows 的 **Public** 文件夹中，以便多个用户帐户可以共享同一个安装。
   - 他们强调这能减少重复，使多人在同一台机器上协作变得更简单。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 2025 年的 o3 序曲**：OpenAI 预告了其 [o3 模型](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai) 将于 2025 年 1 月发布，并声称其性能优于以往版本。
   - 观察者指出了 [ARC-AGI 结果](https://arcprize.org/blog/oai-o3-pub-breakthrough)，认为 **o3** 可能会改变 AI 的竞争格局。
- **FineMath 助力数学任务**：[FineMath 数据集](https://x.com/anton_lozhkov/status/1869771053146464507) 包含 50B tokens，旨在提升模型在 GSM8K 等基准测试中的表现。
   - 贡献者提到 [FrontierMath 协同效应](https://x.com/loubnabenallal1/status/1870731069944713217) 将难题的准确率从 **2%** 提升到了 **25%**。
- **Anthropic 与 xAI 关注融资激增**：Anthropic 的基础模型在编程任务中备受赞誉，而 [xAI 宣布获得 60 亿美元 C 轮融资](https://x.com/xai/status/1871313084280644079)，投资者包括 a16z 和 Nvidia 等巨头。
   - 猜测集中在这些新资金将如何挑战 **OpenAI** 即将推出的 **o3**，并证实了该行业对更大规模投入的渴望。
- **视觉与视频融合挑战 YOLO 地位**：如 [播客更新](https://x.com/latentspacepod/status/1870861606051102777?s=61) 中所述，**RT-DETR** 和 **LW-DETR** 等模型正威胁着 YOLO 在实时检测领域的统治地位。
   - 讨论强调了将视频流水线与 [Diffusion Transformers](https://x.com/latentspacepod/status/1871051952194523343) 相结合，将目标检测提升到了超越现有标准的水平。
- **Character AI 与 API Key 成为焦点**：成员们尝试了各种 API Key 以追求功能扩展，同时讨论了 Character AI 的用户体验。
   - 他们还注意到年轻群体是这些 Character AI 平台的主要驱动力，这引发了对 AI 交互激发的各种情感线索的广泛思考。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **CMD-R 增强推理能力并超越 GPT-4**：成员们注意到 **CMD-R** 可以获得类似于 **QwQ** 的高级 **推理能力（reasoning skills）**，并在实际逻辑任务中展示了新的日志。他们报告称 **Command-R-08** 的表现超过了原生的 **GPT-4**，并有传言称 “Command-Raz” 将取代现有的主流 LLM。
   - 他们重点参考了 [Command R 模型卡片](https://docs.cohere.com/docs/responsible-use) 以了解性能细节，引发了对进一步改进的猜测。
- **红队演练与安全基准**：参与者探讨了 **AI 红队工具（red teaming tools）** 和 LLM 产品的护栏，参考了 [企业 AI 安全指南](https://cohere.com/blog/the-enterprise-guide-to-ai-safety)。他们分享了关于 **负责任的 AI 使用** 的文档，强调了在 **BOLD** 等指标上减少 **偏见（bias）** 和 **毒性（toxicity）**。
   - 其他人引用了 [引入安全模式](https://cohere.com/blog/intro-safety-modes) 和 [Security | Cohere](https://cohere.com/security) 来了解企业级模型防护措施，称红队演练是 AI 开发的“自然组成部分”。
- **Cohere 请求时间之谜**：成员们讨论了在发送数据前 **估算请求时间** 的可行性，建议使用 **测试 token** 的分布图。**xvarunx** 提议在 **25 日** 提供测试额度或进行实验。
   - 他们鼓励社区分享使用统计数据以进行集体采样，但尚未确认官方的时间预测。
- **Batch Embed 任务限制漏洞**：一位用户对 **batch embed** 任务表示担忧，引用了严格的 **10,000 条目** 检索限制。他们担心超出该阈值的数据会产生费用，从而引发了对数据上传大小的进一步澄清。
   - 另一位用户建议检查使用详情，并考虑从 **Trial key** 升级，参考了之前每月 1,000 次调用上限导致的 **TooManyRequestsError** 等问题。
- **H2 标题提升 Command R 表现**：参与者确认，使用 `## Task and Context` 等 H2 标题编写的 **系统消息（system messages）** 会使 **Command R** 表现更强。他们强调，**不遵守** 此格式会严重损害响应质量。
   - 他们还测试了 `## Example Output` 等标题，一致认为保持格式一致能产生顶级结果，这一观点得到了官方文档的支持。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Document Agents Galore**: LlamaIndex 博客展示了关于 **文档处理** 的新指南，包括[发票中的单位标准化](https://t.co/aOTuSwM341)以及一个简化行项目的 **SKU 匹配 Agent**。
   - 他们还发布了一个 **汽车保险 Agent 工作流** 教程和一种 **动态 ArXiv 研究 Agent** 方法，并附带了 [cookbook 链接](https://t.co/6jnUYtX6Mv)，提供了新 Agent 模式的一站式采样。
- **RAG Pipeline Peculiarities**: 构建 **RAG** 的社区成员在嵌入存储和索引之间的差异上反复琢磨，这在处理大型 JSON 文件时产生了困惑。
   - 他们得出结论，对话摄取必须与向量数据库结构保持一致，以确保更好的数据检索，同时称赞 **LlamaIndex** 基础架构的快速适应性。
- **Wanted: Web3 AI Specialists**: 一位用户宣布为一个 **Web3 AI 项目** 招聘，报酬为 **每小时 15–40 美元**，寻求熟练的贡献者。
   - 他们鼓励通过私信获取更多细节，暗示团队正在迅速组建。
- **Chat Store Shenanigans**: 询问者想知道如何在 Chat Store 中嵌入像响应时间这样的 'additional_kwargs'。
   - 他们了解到可以直接操作对话日志或将其转换为字典，在需要的地方添加额外的元数据。
- **Restrain Continuous LLM Updates**: 成员们探讨了处理来自 IoT 和社交媒体的 **实时数据**，结果发现频繁更新存在 **灾难性遗忘 (catastrophic forgetting)** 和模型漂移的风险。
   - 他们建议进行定期重新训练（每日或每周）并生成标签，以保持一致性和性能。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Reshape Riddles with ShapeTracker**: 社区详细介绍了 tinygrad 中的 **ShapeTracker** 如何使用零成本移动操作、维度变化的错觉以及步长 (strides) 操作，并重点推荐了 [官方 ShapeTracker 文档](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html)。
   - 他们指出，通过重新组织数据形状可以实现高级用法，但也承认 **文档缺失** 阻碍了更深入的理解。
- **Bug Bounty Buzz**: 一位新人询问 fork 仓库并提交 **PR** 是否足以领取 **Bug 赏金**，引发了关于正式指南、贡献以及 tinygrad 中潜在漏洞的讨论。
   - 社区成员澄清说，除了提交代码外，该过程通常需要有据可查的修复证明，尽管官方步骤仍有些模糊。
- **Meeting #50 Mingle**: 与会者讨论了 **第 50 次会议**，会议涵盖了三个要点：公司更新、调度器清理计划以及即将推出的新 **tinygrad** 实现。
   - 他们还提到了 **onnx**、**tensor cores** 和正在进行的赏金项目，确保核心改进得到优先处理。
- **Boolean Mask Bamboozle**: 一位用户在尝试使用 **布尔掩码 (boolean masks)** 对张量进行索引时遇到了困难，在数据依赖循环、JIT 约束和性能下降方面挣扎。
   - 建议包括在不使用布尔操作的情况下重写索引逻辑，强调了潜在的性能提升，以及开发者对缺乏直接解决方案的沮丧。
- **CLIP Loading Lament**: 用户尝试加载预训练的 **CLIP** 模型，但遇到了 **NotImplementedError**，怀疑是设备使用问题或缺少 state dict 键。
   - 其他人建议在处理权重之前应用 `.to(device)`，并指出如果配置得当，**VSCode** 中的环境设置不应导致这些问题。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 与 Compound AI：RISC 还是 CISC？**：在最近的一次讨论中，[Omar Khattab 的 'o3' 概念](https://x.com/lateinteraction/status/1870554971403698548)引发了关于未来基础模型是否会像 RISC 与 CISC 一样产生分支的讨论，开发者将依赖编译器来处理高级规范。
   - 在[另一条推文](https://x.com/dbreunig/status/1870287741361238317)中，Drew Breunig 质疑多路径推理是否能保持 zero-shot，这引发了关于“Compound AI”如何统一所有专业化推理步骤的推测。
- **DSPy 等待时间的烦恼**：一位参与者担心 DSPy 优化任务的等待时间过长，如果运行时间太久会消耗大量额度。
   - 他们建议提供运行时间预估以避免无限度使用，其他人则建议使用本地设置以减少开销。
- **ModernBERT 在 8192 Token 下大显身手**：全新的 [ModernBERT](https://huggingface.co/blog/modernbert) 发布，支持 8192 token 窗口，在 `transformers` v4.48.0 中包含 base（139M 参数）和 large（395M 参数）变体。
   - 它的目标是取代旧的 BERT 风格模型，具有更快的检索速度，据报道在 RAG 风格任务中领先 **9 个百分点**。
- **ColBERT 与 ModernBERT：制胜检索组合**：ModernBERT 作为一种强大的长上下文检索器，非常适合与 **ColBERT** 搭配，特别是在大文本场景下。
   - 一些参与者表示，可以使用 **Pylate** 基于 ModernBERT 构建 ColBERT 模型，从而增强长上下文任务的协同效应。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **本地 LLM 赢得粉丝**：一位用户赞扬了 **OI** 中的**本地 LLM 集成**，称其舒适且敏捷，解决了对被 OpenAI 掩盖的担忧。
   - 这些反馈可能会指导 **1.0** 版本的开发，该版本旨在平衡工具使用的便利性与责任感。
- **LM Studio 标签缓解困惑**：一位用户发现应用 **lm_studio** 标签解决了本地模型输出问题，而 **ollama** 的结果则不一致。
   - 他们计划在 **Classic mode** 被取代后依赖 `lm_studio`，以确保更可预测的流水线。
- **1.0 文档引发大量需求**：一位用户请求更新 **1.0** 文档，以便调整他们的代码并测试 Python 执行的 profile，理由是缺乏清晰的资源。
   - 他们的询问凸显了社区在升级到最新版本时对更好指南的渴望。
- **函数调用遭遇问题**：一位用户在使用 `together` AI 模型时遇到了 **1.0** 中的函数调用错误，因为该功能在他们的 profile 中被禁用了。
   - 他们从 **litellm** 调用中删除了不支持的参数以维持工作流，展示了在面对功能缺失时的巧妙解决方案。
- **代理设置运行顺畅**：一位用户确认他们的 **proxy** 配置在 **OI** 中表现良好，这得益于自定义的 base URL。
   - 这种设置简化了集成，标志着迈向本地化设计的良好一步。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.5.0 提升微调体验**：全新的 **Torchtune v0.5.0** 版本支持 **Kaggle** 微调，并包含一份关于模型使用的[详尽教程](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-your-model-in-the-wild)。
   - 它扩展了对 **Gemma 2** 模型覆盖，提供了一个 **Early Exit** 训练 recipe，并支持 **Ascend NPU**。
- **职位空缺：TorchTune 的下一位创新者**：团队正在寻找一名软件工程师来处理高级 ML 后训练任务，详见此 [Software Engineer 职位说明](https://www.metacareers.com/jobs/512189008507168/)。
   - 他们特别希望应聘者具有扎实的 **ML** 和软件工程背景，以推动 **TorchTune** 的开发。
- **量化友好的 LoRA 登场**：一个新的 **QAT + LoRA** recipe 已提交至 [Torchtune GitHub](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_qat_lora.yaml) 以增强模型性能。
   - 它解决了效率问题，同时为量化策略提供了针对性的微调。
- **State Dict 包装：一个潜在的陷阱**：某些代码假设 **state dict** 仅包含参数，忽略了持久化 buffer 的可能性。
   - 包装函数盲目地将条目转换为 **nn.Parameter**，可能会给其他模型内容带来风险。
- **Ray 对比 torch.distributed：两种方法的博弈**：一次对话权衡了使用 **Ray** 进行函数级并行与依赖内置 **torch.distributed** 分片的优劣，并引用了 **RLHF** 等用例。
   - 参与者还注意到在 KD 训练 3500 秒后出现 **NaN** 问题，建议通过切换 **_SUPPORTS_FLEX_ATTENTION** 来解决该问题。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **无审查 GPT 引起褒贬不一的反应**：一位用户感叹自 11 月以来失去了一种 *jailbreak* 方法，希望能恢复完全无审查的功能。
   - 他们坚持要求 GPT 能够完全代表他们发言，这引发了关于用户自由与模型 guardrails 之间的辩论。
- **亮度通道对色彩清晰度的启示**：一位成员支持使用带有专用亮度通道的色彩空间，声称这能更有效地保留高频灰度细节。
   - 他们认为 **RGB** 复杂化了感知，并引用了 [JPEG documentation](https://jpeg.org) 和 **AV1** 参考资料作为潜在的改进方向。
- **VAE 应对色彩处理**：一位参与者建议 **Variational Autoencoders (VAE)** 可能会通过利用专门的 loss functions 来解决色彩感知问题。
   - 他们假设指标与人类视觉线索之间的对齐可能会产生更自然的色彩再现。
- **Test Time COT 与知识重组受到关注**：一位用户寻求关于 **test time COT** 和知识重组的出版物，并参考了一篇关于方法的 o3 arc 帖子。
   - 其他人想知道这些技术将如何重塑 **text-to-image generation**，暗示了旧框架与新兴概念之间的协同作用。
- **ZGI 的 o1 Non-Preview 取得成功但面临成本限制**：一位贡献者确认了 **ZGI** 在 **o1 non-preview** 上的成功，标志着在集成框架方面迈出了一步。
   - 他们还强调了采用这些方法的成本担忧，突显了技术进步中的财务压力。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LangGraph 与 CrewAI：工具成为焦点**：一位参与者建议在即将到来的实验中使用 **LangGraph**，理由是 **Autogen** 的 API 使用困难，并对 **instruction tuning** 和 **function calling** 等高级话题感兴趣。
   - 其他人称赞了 **CrewAI** 提供的有益社区支持，建议探索多个框架可以改善 MOOC 体验。
- **没有学分也没关系：伯克利 MOOC 澄清**：一位用户指出 **MOOC** 不授予正式的 **Berkeley credits**，这可能会影响学习者的预期。
   - 尽管如此，参与者发现内容非常有趣，强调了其在实际技能发展方面的价值。
- **YouTube 实验洞察激发好奇心**：一位参与者分享了[一段 YouTube 视频](https://youtu.be/-r0XPC7TLzY)，表示希望在进行实验 2 和 3 之前看到它，认为这会拓宽他们的理解。
   - 另一位成员提到一位朋友关注了这个频道，表明了对所涵盖内容的共同热情。
- **一月证书发放倒计时**：关于 **MOOC certificates** 的问题被提出，一位成员澄清说证书将在 **1 月** 发放。
   - 这一公告让渴望获得参与和努力确认的学习者感到安心。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Liger DPO 努力解决 Loss 一致性问题**：成员们正在推动 **Liger DPO** 进入完全运行状态，将其性能与 [HF TRL baseline](https://link.to/trl) 进行对比，并面临严重的 loss parity 障碍。
   - 他们提到了即将到来的 **KTO** 阶段，预示着在弥合这些问题方面可能会有更多困难。
- **社区共担痛苦，期待快速修复**：一位用户用“痛苦”一词总结了现状，强调了围绕 Liger DPO 和 KTO 斗争的挫败感。
   - 其他人也表示乐观，认为这些障碍很快就会得到解决，展示了社区成员之间的团结。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# 第二部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1319758354016637051)** (1 条消息): 

> `Windsurf 1.1.1 release, Usage Transparency and Pricing, Cascade Image Uploads, Improved Python Support` 


- **Windsurf 1.1.1 推出新功能**：**Windsurf 1.1.1** 的发布引入了易用性改进，包括新的“**Send to Cascade**”按钮和增强的 **autocomplete** 行为。
   - 它还修复了 **Windows chat mode** 编辑问题以及 **autocomplete** 变慢的问题，详情请参阅 [changelog](https://www.codeium.com/changelog)。
- **新的使用透明度系统发布**：更新后的 Windsurf **使用与定价系统**现在包含一个面板，显示当前方案的使用情况和试用过期详情，并提供了升级链接。
   - 此外，新的“**Legacy Chat**”模式允许用户在 **Flow Credits** 余额不足时继续使用 **Cascade** 功能。
- **Cascade 图片上传功能增强**：用户现在可以享受 **Cascade 图片上传**功能，且不再受之前的 1MB 限制，提升了分享图片的实用性。
   - 这一变化为经常处理大文件的用户提供了更流畅的体验。
- **Windsurf 中的 Python 支持得到改进**：最新版本的 Windsurf 为 **Python** 提供了**改进的语言支持**，满足了开发者对更好功能的需求。
   - 这一增强是平台持续努力改进编程体验的一部分。



**提到的链接**：<a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变化。

  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1319827690123169843)** (1 条消息): 

> `Send to Cascade feature` 


- **展示“Send to Cascade”按钮**：演示了 **“Send to Cascade”** 按钮，展示了将其问题直接发送到 **Cascade** 的功能。
   - 演示包含了一个[演示链接](https://x.com/windsurf_ai/status/1870268007995585000)以供进一步参考。
- **Cascade 的互动亮点**：一名成员在讨论中强调了 **“Send to Cascade”** 按钮的互动性，重点关注用户交互。
   - 该功能旨在简化问题报告流程，提高用户效率。



**提到的链接**：<a href="https://x.com/windsurf_ai/status/1870268007995585000">来自 Windsurf (@windsurf_ai) 的推文</a>：将您的问题直接发送到 Cascade！

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1319756297070968869)** (175 条消息🔥🔥): 

> `Windsurf 性能问题、Windsurf 订阅查询、AI 项目开发、Codeium 用户体验、节日促销与支持` 


- **Windsurf 性能问题**：用户报告 Windsurf 存在延迟和高内存占用问题，特别是在处理大型代码库时，并指出其表现不如 Cursor 等竞争工具。
   - 一些用户质疑 Windsurf 是否能处理大型项目，并建议提供项目蓝图以优化结构。
- **Windsurf 订阅查询**：多位用户分享了购买 Pro Plan 的经历，以及额度（credits）未反映在账户中的问题，支持团队正在处理这些 Bug。
   - 一位用户对在节日期间购买 Windsurf 却遇到这些问题表示沮丧，导致他们重新换回了 Cursor。
- **AI 项目开发**：用户讨论了将 AI 集成到社交网络等大型项目中的可行性，对于 Windsurf 等工具是否适用意见不一。
   - 有建议称，为使用 AI 工具处理大型项目提供分步蓝图，可能有助于引导用户完成开发过程。
- **Codeium 用户体验**：用户对 Codeium 的 autocomplete 功能随时间推移而退化表示担忧，并强调了高效使用额度的重要性。
   - 人们还分享了免费版与 Pro 版限制的体验，表示对订阅功能和限制感到困惑。
- **节日促销与支持**：用户询问了 Windsurf 潜在的节日促销或优惠码，有人希望在圣诞期间获得特别优惠。
   - 支持团队的回应暗示，包括账户问题在内的许多问题可能是由于节日高峰造成的，将在团队结束假期归来后解决。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://plugins.jetbrains.com/plugin/21206-qodo-gen-formerly-codiumate-">Qodo Gen (Formerly Codiumate) - IntelliJ IDEs Plugin | Marketplace</a>: Qodo Gen 充满信心地编写代码、测试和评审。Qodo Gen 是您的 AI 驱动编程助手和导师。Qodo Gen 帮助您理解代码、测试并评审...</li><li><a href="https://tenor.com/view/cat-cat-meme-sad-water-disappointed-gif-3288661263568768157">Cat Cat Meme GIF - Cat Cat meme Sad - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队获取个性化协助。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1319757579483611167)** (504 条消息🔥🔥🔥): 

> `Windsurf 性能问题、Flow Action 限制、用户登录问题、模型对比、支持与反馈` 


- **用户报告 Windsurf CPU 占用率高**：多位用户报告 Windsurf 消耗过多的 CPU 资源，部分用户遇到高达 99% 的 CPU 占用，引发了对性能的担忧。
   - 这一问题引起了用户的沮丧，尤其是那些最近购买了订阅的用户。
- **Flow Action 和用户提示词额度差异**：多位用户表达了不满，因为他们的 Flow Action 和用户提示词额度在购买后不久便无故消失。
   - 支持团队表示这是一个正在处理的已知 Bug，但用户仍对他们的订阅感到焦虑。
- **Windsurf Write Mode 困惑**：一些用户对 Windsurf 中 Write Mode 的功能感到困惑，质疑 AI 是否真的识别了这一设置。
   - AI 经常无法正确执行预期的更改，导致重复错误并浪费额度。
- **Windsurf 与 Cursor 的对比**：新用户分享了 Windsurf 与 Cursor 等其他 IDE 相比的复杂感受，特别是在可靠性和 AI 响应准确性方面。
   - 用户对 Windsurf 管理代码更改以及从用户输入中学习的效果表示担忧。
- **支持渠道访问不一致**：由于登录问题，用户在访问支持渠道时面临限制，增加了他们有效解决投诉的难度。
   - 讨论强调了尽管是付费订阅者，却无法在支持论坛发帖或提交工单的不便。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.pulsemcp.com/posts/ai-is-making-websites-obsolete-with-mcp#supporting-ai-apps">Pulse MCP</a>: 浏览并发现 MCP 服务器、工具和资源</li><li><a href="https://docs.codeium.com/command/related-features#smart-paste)">Refactors, Docstrings, and More - Codeium Docs</a>: 未找到描述</li><li><a href="https://www.helicone.ai/status">LLM Status Checker: Is OpenAI, Claude, or Perplexity Down? - Helicone</a>: 针对 OpenAI, Anthropic, Claude, Perplexity, Together AI, Mistral 以及其他主要 AI 提供商的实时状态监控。检查热门 LLM API 的当前可用性和性能。</li><li><a href="https://addyo.substack.com/p/the-70-problem-hard-truths-about?utm_source=tldrnewsletter">The 70% problem: Hard truths about AI-assisted coding</a>: 一份实地指南以及为什么我们需要重新审视我们的预期</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.pulsemcp.com/servers/modelcontextprotocol-knowledge-graph-memory">Pulse MCP</a>: 浏览并发现 MCP 服务器、工具和资源</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium 对个人永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活部署。</li><li><a href="https://youtu.be/z_7CLMYKwGs"> - YouTube</a>: 未找到描述</li><li><a href="https://www.builder.io/blog/ai-dev-skill">Why AI Is Making Dev Skills More Valuable, Not Less</a>: AI 并没有取代开发者，而是让他们变得更有价值。让我们来看看开发者的工作是如何演变的，以及它如何影响团队</li><li><a href="https://addyo.substack.com/p/the-70-problem-hard-truths-about">The 70% problem: Hard truths about AI-assisted coding</a>: 一份实地指南以及为什么我们需要重新审视我们的预期</li><li><a href="https://github.com/aindreyway/mcp-codex-keeper">GitHub - aindreyway/mcp-codex-keeper: An intelligent MCP server that serves as a guardian of development knowledge, providing Cline assistants with curated access to latest documentation and best practices across the software development landscape</a>: 一个智能 MCP 服务器，充当开发知识的守护者，为 Cline 助手提供对整个软件开发领域最新文档和最佳实践的精选访问...</li><li><a href="https://www.reddit.com/r/Codeium/comments/1hd42ej/well_there_goes_that_no_more_write_mode_in_the/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/crjaensch/PromptoLab">GitHub - crjaensch/PromptoLab: A multi-platform app to serve as a prompts catalog, a LLM playground for running and optimizing prompts, plus a prompts evaluation and assessment playground.</a>: 一个多平台应用，用作提示词目录、运行和优化提示词的 LLM 游乐场，以及提示词评估和考核的游乐场。 - crjaensch/PromptoLab</li><li><a href="https://docs.unblu.com)">no title found</a>: 未找到描述</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: 将任何网站转换为 LLM 就绪的数据。
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1319756293312872609)** (903 messages🔥🔥🔥): 

> `Cursor IDE, AI 编程辅助, O1 和 Sonnet 模型, React 开发挑战, 使用 AI 进行 Web 开发`

- **用户对 AI 代码生成的体验**：几位用户分享了他们使用 O1 和 Sonnet 等 AI 模型进行编程任务的经验，指出 Sonnet 在生成功能性强且经过优化的代码方面表现尤为出色。
   - 然而，用户报告了性能不一致的问题，特别是 Cursor Composer 模式比直接与基于 chat-based 模型交互要慢。
- **对 Cursor 性能和功能的反馈**：用户表示 Cursor 有时在处理大型项目和冗长代码时比较吃力，当模型提供错误或次优的解决方案时，往往会导致挫败感。
   - 反馈中还包括对 Cursor 资源占用的担忧，一些用户注意到它比其他应用程序消耗更多的 RAM 和 CPU。
- **React 开发的挫折感**：开发者讨论了与 React 开发相关的复杂性和挫折，强调了诸如错误解决时间长和文件管理困难等问题。
   - 一位用户幽默地谈到了他们庞大的代码库，开玩笑说 React 让人应接不暇，同时也承认了使用 Next.js 的好处。
- **对 UBI 和未来工作趋势的关注**：讨论转向了 AI 和自动化更广泛的社会影响，用户推测了工作的未来以及全民基本收入（UBI）的可能性。
   - 关于 UBI 实施的时机以及 AI 对就业机会影响的观点各不相同，一些人对政府能做出及时响应表示希望。
- **使用 AI 进行文档查阅和学习**：用户探索了利用 AI 引用文档来辅助编程的能力，建议嵌入文档 URL 可以增强 AI 在编程任务中的表现。
   - 这一建议旨在通过提供相关材料来提高 AI 的上下文理解能力，展示了 AI 工具与开发者资源之间的协作。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aibenchhub.com/">AiBenchHub</a>: 未找到描述</li><li><a href="https://docs.cursor.com/context/@-symbols/basic">Cursor - Build Software Faster</a>: 未找到描述</li><li><a href="https://stackoverflow.com/questions/59477289/turn-off-visual-studio-code-inline-parent-child-folder-display">关闭 Visual Studio Code 内联父/子文件夹显示</a>: 我不确定这是一个扩展还是一个更新，但自从最近的 VS Code 更新以来，它们将单个文件夹与父文件夹内联显示。我原以为这不会太困扰我，但我发现...</li><li><a href="https://www.cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>: 您可以在此处管理您的账户、账单和团队设置。</li><li><a href="https://www.cherrycapitalweb.com/">Cherry Capital Web Design | 为本地企业打造现代网站</a>: 北密歇根州首屈一指的网页设计服务，创建现代网站，帮助您的业务在网上脱颖而出并取得成功。</li><li><a href="https://web3forms.com/">Web3Forms - 免费联系表单转电子邮件服务 API</a>: 未找到描述</li><li><a href="https://x.com/aide_dev">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://www.mcpservers.ai/servers/modelcontextprotocol/Sequential%20Thinking">MCP 服务器</a>: 浏览最大的 Model Context Protocol 服务器库。与他人分享您创建的 Model Context Protocol 服务器。</li><li><a href="https://x.com/liamesp/status/1869319333954089218?s=46">来自 liam (@liamesp) 的推文</a>: 现在，@krea_ai 编辑器中支持对象 + 画笔选择</li><li><a href="https://nextjs.org/docs">简介 | Next.js</a>: 欢迎来到 Next.js 文档。</li><li><a href="https://x.com/sama/status/1870709421111984135">来自 Sam Altman (@sama) 的推文</a>: 我认为《华尔街日报》目前是美国最好的报纸，但在我们宣布……几小时后，他们发表了一篇名为《AI 的下一次大飞跃进度落后且极其昂贵》的文章。</li><li><a href="https://github.com/getcursor/crawler">GitHub - getcursor/crawler: 轻松向 Cursor 的编程 AI 展示文档</a>: 轻松向 Cursor 的编程 AI 展示文档。通过在 GitHub 上创建账户来为 getcursor/crawler 的开发做出贡献。</li><li><a href="https://techcrunch.com/2024/12/21/openais-gpt-5-reportedly-falling-short-of-expectations/">据报道 OpenAI 的 GPT-5 未达预期 | TechCrunch</a>: 据报道，OpenAI 开发其下一个主要模型 GPT-5 的工作进度落后，其结果尚不足以证明巨大的成本是合理的。</li><li><a href="https://github.com/olweraltuve/LmStudioToCursor">GitHub - olweraltuve/LmStudioToCursor</a>: 通过在 GitHub 上创建账户来为 olweraltuve/LmStudioToCursor 的开发做出贡献。</li><li><a href="https://aide.dev/blog/sota-bitter-lesson">swebench-verified 上的 SOTA：(重新)学习惨痛的教训</a>: 搜索代码是每个开发者工作流的重要组成部分。我们正努力让它变得更好。</li><li><a href="https://downloader.cursor.sh/windows/nsis">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1320130587012370523)** (1 条消息): 

> `ChatGPT Plus 的 Sora 奖励、Teams 用户的 Sora 访问权限、Blend 功能升级、Sora 作品的共享链接` 


- **ChatGPT Plus 用户获得特别的 Sora 节日访问权限**：节日期间，**ChatGPT Plus** 用户可以通过宽松队列享受无限的 [Sora](https://sora.com) 访问，作为节日的特别待遇。
   - 这一特别奖励鼓励用户在节日期间更多地参与。
- **Sora 现已面向 Teams 用户开放**：**Sora** 已向所有 **Teams** 用户开放，确保更广泛地访问其特性和功能。
   - 此举反映了 OpenAI 致力于为协作环境扩展工具的承诺。
- **Blend 功能升级以增强体验**：**Blend 功能**已获得升级，可能会增强用户的创作体验。
   - 具体改进尚未说明，但用户可以期待整体性能的提升。
- **Sora 作品分享变得简单**：用户现在可以利用**共享链接**，即使对方没有账户，也可以与朋友分享他们的 **Sora 作品**。
   - 此功能旨在促进用户之间的协作和分享，让创意更易于传播。



**提及的链接**: <a href="https://sora.com)">未找到标题</a>: 未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1319757080709697606)** (717 条消息🔥🔥🔥): 

> `Gemini 2.0 性能, AI 与人类感知, Llama 3.3 对比 Gemini 辩论, 机器人与 AI, AGI 哲学` 


- **对 Gemini 2.0 的评价褒贬不一**：用户对 Gemini 2.0 的性能表达了不同看法，一些人声称与 GPT-4o 等其他模型相比，它缺乏可靠性。
   - 许多人认为，虽然 Gemini 的上下文长度令人印象深刻，但其输出质量和逻辑一致性仍有不足。
- **AI 与人类特征**：人类大脑与 AI 模型之间的对比引发了讨论，涉及大脑速度限制和感官处理能力。
   - 参与者注意到将 AI 拟人化的倾向，辩论了 AI 能力与人类特征之间更深层次的影响。
- **为时态逻辑开发 AI**：关于 AI 何时能有效处理复杂的推理任务（如时态逻辑）的问题被提出，表明当前模型仍力有不逮。
   - 参与者对围绕 AI 的炒作及其在处理细微逻辑结构方面的局限性表示沮丧。
- **AI 领域的创新**：讨论强调了在 OpenAI 和 Google 持续创新的背景下，对 Grok 等模型进展的怀疑。
   - 用户对 Grok 即将推出的迭代潜力持怀疑态度，理由是缺乏有意义的功能。
- **AI 研究与伦理**：参与者分享了在伦理辩论和教育背景下使用 AI 的看法，强调了 AI 响应中安全性的必要性。
   - 还探讨了关于 AGI 的理论和机器智能的哲学层面，重点关注意识的定义和影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/skirano/status/1861081529071346161">来自 Pietro Schirano (@skirano) 的推文</a>: 今天 @Anthropic 发布了 MCP，这是一个允许 Claude 运行服务器的框架，赋予其超能力并有效地将 Claude 应用转变为 API。我们创建了一些服务器，我想你会...</li><li><a href="https://www.theverge.com/2023/4/19/23689554/google-ai-chatbot-bard-employees-criticism-pathological-liar">报告称 Google 员工给 AI 聊天机器人 Bard 贴上“比没用还糟”和“习惯性撒谎者”的标签</a>: Google 自己的员工表示 Bard 还没准备好正式发布。</li><li><a href="https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged">参差不齐前沿上的半人马与赛博格</a>: 我想我们对于 AI 是否会重塑工作已经有了答案....</li><li><a href="https://youtu.be/bUrLuUxv9gE?si=8bNnq3LDsrBpdCQG"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1319793288416460892)** (28 条消息🔥): 

> `O3 发布与定价, GPT-4o 订阅限制, Token 限制详解, 数据提取测试, ChatGPT 使用反馈` 


- **O3 发布与定价预期**：**O3 mini** 定于下月底发布，完整版预计在“此后不久”发布。一些人推测价格可能达到 **$45**。
   - *有人分享了他们对定价的期望*，表明社区对即将推出的模型很感兴趣。
- **对 GPT-4o 使用限制的困惑**：通过 ChatGPT Plus 订阅，用户每 3 小时可以使用 GPT-4o 发送 **80 条消息**，如果超过限制可能会进入冷却期。一位用户对大量使用后遇到冷却期表示沮丧。
   - 另一位参与者评论了 **GPT-4o 剩余的限制**，这让许多不知道即使是 Plus 订阅也有这些限制的人感到惊讶。
- **理解 Token 限制**：**128k 上下文窗口** 大约相当于 **100,000 个单词**，具体取决于特定的分词方式和语言因素。超过此限制可能会导致幻觉和错误的响应。
   - 据指出，对于 Plus 用户，限制仅为 **32k tokens**，而免费版仅为 **8k**。
- **数据提取测试中的挑战**：用户讨论了为从 PDF 中进行**数据提取**定义测试的挑战，特别是由于输出格式不一致。他们指出，传统的集成测试通常会失败，因为后续运行会产生不同的结果。
   - 这表明需要替代测试方法来有效处理此类提取任务。
- **关于 ChatGPT 使用的反馈**：有关于从 Pro 降级到 Plus 订阅的评论，引起了对持续福利的困惑。一位用户回应了潜在的限制，表示他们已经连续使用了 ChatGPT 好几个小时。
   - *另一位成员建议保留 Pro 会员资格直至到期*，以获得不间断的服务。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1319843324785463306)** (24 条消息🔥): 

> `光谱理论与光谱提示 (Spectrum Prompting)，Sora 行为建模，饮食规划应用，提示词库讨论，ChatGPT 中的记忆个性化` 


- **探索 AI 提示词的光谱理论**：分享了一篇讨论 **Spectrum Prompting** 的文章，这是一种引导 AI 沿着细微光谱进行思考的方法，使用结构化公式：⦅Z(A∐B)⦆。
   - 第一步强调在允许 AI 探索相关主题之前先对光谱进行引导 (priming)，从而增强其回答的深度。
- **Sora 行为的增强**：用户分享了如何有效地与 **Sora** 交互的经验，包括使用带有用户输入行的 **custom prompt** 结构以增强输出效果。
   - 有人指出，尽管在音频/配乐集成方面存在挑战，但预先规划提示词可以改善结果。
- **处理食谱开发中的饮食限制**：一项应用讨论集中在为符合严格饮食限制的饮食规划创建一个用户友好的界面，这需要迭代调整。
   - 有人对处理时间表示担忧，平均耗时 **40-60 秒**，并质疑这种延迟在用户交互中的可行性。
- **提示词库的可访问性**：一位用户询问如何访问之前在 prompt engineering 类别下可用的 **prompt library**。
   - 据悉该库已重命名，建议用户在更新后的章节中查找资源。
- **利用 ChatGPT 中的记忆功能**：为了向 ChatGPT 添加有关组织的信息，建议在个性化设置中启用 memory，以便 AI 保留信息。
   - 这种方法允许持续的记忆，可以根据共享的细节增强用户交互。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1319843324785463306)** (24 条消息🔥): 

> `光谱提示技术，Sora 输入方法，饮食应用迭代，食谱创建复杂度，GPT 模型中的营养准确性` 


- **用于提升 AI 深度的工作光谱提示**：一位成员分享了关于 **Spectrum Prompting** 的文章，该方法鼓励 AI 沿着光谱思考以获得更细微的差别，使用简单公式：⦅Z(A∐B)⦆。
   - 该方法通过提示 AI 探索两个概念之间的**连续体 (continuum)**，使其能够生成详细的回答。
- **增强 Sora 输入机制**：成员们讨论了向 **Sora** 提供输入的方法，强调需要替换提示词中指定的输入行以定制输出。
   - 一位成员鼓励根据当前的理解和项目需求尝试进行修改以改善结果。
- **对饮食应用耗时的担忧**：一位用户对他们的饮食应用计算饮食计划所需的**时间**表示担忧，指出平均耗时 **40-60 秒**。
   - 他们确认，考虑到迭代成分和平衡的复杂性，这个时长似乎是合理的，尽管可以更高效地处理。
- **食谱创建的复杂性**：一位成员指出，食谱创建可能涉及**自顶向下 (top-down)** 或**自底向上 (bottom-up)** 的方法论，具体取决于预期的结果和处理成本。
   - 他们建议使用余弦检索 (cosine retrieval) 以提高成本效益，并指出执行取决于模型的指令和结构。
- **解决营养计算错误**：展示了一个蛋白质营养计算差异的例子，证明了在准确提示模型方面面临的挑战。
   - 讨论强调了构建稳健框架的重要性，以确保 AI 模型提供准确的饮食建议。


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1320509898987143178)** (1 条消息): 

> `Aider 的新 polyglot benchmark、o1 模型性能、编程练习挑战` 


- **o1 模型夺得榜首**：OpenAI 的新 **o1** 模型在 aider 的新 [multi-language coding benchmark](https://aider.chat/2024/12/21/polyglot.html) 中获得了 **62%** 的最高分。该基准测试的设计初衷是比之前的编程基准测试 *更具挑战性*。
- **模型性能对比**：在排行榜中，**Sonnet** 得分为 **45%**，而 **o1-mini** 和 **Haiku** 分别以 **33%** 和 **28%** 紧随其后。结果凸显了 **o1** 与其他顶尖 LLM 之间显著的性能差距。
- **新基准测试详情**：**polyglot benchmark** 现在包含 C++、Go 和 Java 等语言中最难的 **225 个编程问题**，而旧的基准测试仅专注于 Python。这一变化旨在清晰地界定当前编程模型的能力。
- **聚焦高难度练习**：新基准测试包含了 Exercism 提供的 **最难练习题**，提升了挑战难度。这种方法旨在对各种流行编程语言的编码技能进行全面评估。



**提及的链接**：<a href="https://aider.chat/2024/12/21/polyglot.html">o1 登顶 aider 的新 polyglot 排行榜</a>：o1 在 aider 的新多语言、更具挑战性的编程基准测试中获得了最高分。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1319756887696085064)** (812 条消息🔥🔥🔥): 

> `Aider 与 O1 Pro 的配合使用、Gemini 模型性能对比、基准测试结果、使用 LLMs 进行代码编辑、API 访问和速率限制` 


- **Aider 与 O1 Pro 的工作流及代码编辑**：用户正在结合使用 Aider 和 O1 Pro 进行代码修改，通常使用 /copy-context 功能来高效管理编辑。
   - Prompt 经过精炼以指定所需的输出格式，有助于确保 O1 Pro 建议的更改清晰明了。
- **Gemini 模型性能差异**：基准测试显示不同的 Gemini 模型表现各异，用户体验表明 Gemini 2.0 Flash 的表现可能不如预期。
   - 用户报告了 Gemini 模型创建新文件而非编辑现有文件的问题，因此需要调整工作流。
- **基准测试与速率限制**：讨论强调了运行不同模型的成本影响，特别是关于 API 访问和使用速率限制（例如 Vertex AI）。
   - 用户分享了使用 Gemini 模型运行基准测试的见解，并指出了他们在测试期间如何处理各种 Token 限制。
- **Aider 与项目上下文**：探索 Aider 能力的用户正在寻求整合项目上下文的方法，特别是针对 Phoenix 等框架。
   - 对话强调了共享相关项目文件对于 AI 工具成功进行修改的重要性。
- **工具稳定性和可靠性**：人们对 LLMs 在应用于编程任务时的稳定性和胜任能力表示担忧，一些用户觉得模型随着时间的推移变得效率降低。
   - 强调了有效 Prompt 和结构化交互的必要性，特别是在 LLMs 处理用户规范时。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>: 未找到描述</li><li><a href="https://x.com/iruletheworldmo/status/1870176332702986292">来自 🍓🍓🍓 (@iruletheworldmo) 的推文</a>: 很多人都猜到了，但 o1 pro 在 ARC 表现非常出色</li><li><a href="https://rust-fuzz.github.io/book/">简介 - Rust Fuzz Book</a>: 未找到描述</li><li><a href="https://www.pulsemcp.com/posts/ai-is-making-websites-obsolete-with-mcp#supporting-ai-apps">Pulse MCP</a>: 浏览并发现 MCP 服务器、工具和资源</li><li><a href="https://aider.chat/2024/12/21/polyglot.html">o1 在 aider 新的多语言排行榜中夺冠</a>: o1 在 aider 新的多语言、更具挑战性的代码基准测试中获得了最高分。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>: 如何使用 YAML 配置文件配置 aider。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/what-did-you-just-say-gif-27520460">What Did GIF - What Did You - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://pastebin.com/Kf86JGwA">#Requires AutoHotkey v2.0#SingleInstance Force; Register our callback so i - Pastebin.com</a>: Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://x.com/GeminiApp/status/1869074556465648085)">来自 Google Gemini App (@GeminiApp) 的推文</a>: 从今天开始，Gemini Advanced 用户可以优先体验我们最新的 2.0 实验性高级模型 Gemini-Exp-1206。该模型旨在帮助您处理更复杂的任务，例如：🧑‍💻 高级...</li><li><a href="https://aider.chat/docs/leaderboards/edit.html">代码编辑排行榜</a>: 基础 LLM 代码编辑能力的定量基准测试。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>: aider 是您终端中的 AI 结对编程工具。通过在 GitHub 上创建一个账户来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=2TpSWVN4zkg"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/TencentARC/ColorFlow">GitHub - TencentARC/ColorFlow: 论文 "ColorFlow: Retrieval-Augmented Image Sequence Colorization" 的官方实现</a>: 论文 "ColorFlow: Retrieval-Augmented Image Sequence Colorization" 的官方实现 - TencentARC/ColorFlow</li><li><a href="https://youtu.be/HEheh1BH34Q?si=OLABD0ZgutMBZeOK">恒星大小对比 1 (HD)</a>: 目前流传着几个展示最大恒星对比的视频。我喜欢这类东西，所以想自己尝试做一个。大概是因为...</li><li><a href="https://github.com/sigoden/aichat/issues/1050">MCP · Issue #1050 · sigoden/aichat</a>: 描述您想要的解决方案。您是否有兴趣添加对 Model Context Protocol 的支持？额外背景 http://modelcontextprotocol.io 目前正在开发一个进行中的实现。</li><li><a href="https://codegate.ai">首页 - CodeGate</a>: 本地、开源的隐私控制。CodeGate 加密您提示词中的机密信息以保护您的隐私，并利用最新的风险洞察增强 LLM 的知识库以保护您的代码。CodeGate ...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1319803150965805089)** (45 条消息🔥): 

> `gemini-exp-1206 配置, GitHub Copilot 集成, 多种语言的 repo maps, 在 Aider 中使用不同的 LLMs, polyglot benchmark 结果` 


- **运行 gemini-exp-1206 需要特定命令**：根据社区讨论，运行 **gemini-exp-1206** 需要使用命令 `--model gemini/gemini-exp-1206`。
   - 有人担心该模型可能不再免费。
- **关于 GitHub Copilot 使用的说明**：用户讨论了 **GitHub Copilot** 与 Aider 的集成，以及在 API 使用和 rate limits 方面的局限性。
   - 推荐使用其他 LLMs 以在配合 Copilot 时获得更好性能。
- **为其他语言生成 repo maps**：讨论引用了一篇[博客文章](https://aider.chat/2023/10/22/repomap.html)，详细介绍了 Aider 如何使用 **py-tree-sitter-languages** 来支持不同语言。
   - 成员们分享了增强默认集合之外语言的 repo maps 的技巧。
- **为 architect 和 coder 角色使用不同的 LLMs**：有用户请求允许在 Aider 中为架构见解（architectural insights）和编码任务（coding tasks）分配不同的 LLMs。
   - 有建议认为 **Gemini Thinking** 擅长架构设计，但在编码方面表现不佳。
- **polyglot benchmark 的最新更新**：新的 **polyglot benchmark** 显示 OpenAI 的 o1 模型在代码推理任务中表现出色，显著优于其他模型。
   - 它强调了转向包含更多编程语言和更具挑战性的练习，以便进行更好的评估。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/copypaste.html#paste-the-llms-reply-back-into-aider-to-edit-your-files">通过 Web 聊天进行复制/粘贴</a>：Aider 可与 LLM Web 聊天 UI 配合使用</li><li><a href="https://aider.chat/2024/12/21/polyglot.html#o1">o1 在 Aider 新的 polyglot 排行榜中夺冠</a>：o1 在 Aider 新的多语言、更具挑战性的编码基准测试中获得了最高分。</li><li><a href="https://github.com/Aider-AI/aider/pull/2675">当 .env 不存在时不要将其添加到 .gitignore。由 apaz-cli 提交 · Pull Request #2675 · Aider-AI/aider</a>：当我启动 Aider 时收到提示：是否将 .env 添加到 .gitignore (推荐)？(Y)es/(N)o [Yes]：尽管我实际上并没有需要忽略的 .env 文件。每次启动 Aider 我都必须点击确认...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1319798732467736691)** (11 条消息🔥): 

> `Depth AI 评估，Model Context Protocol，GritQL 查询引擎，代码生成与维护挑战` 


- **Depth AI 在大型代码库中表现出色**：一位用户正在评估 [Depth AI](https://www.trydepth.ai)，发现它在回答大型代码库中的**深度技术问题**时非常有用，尽管他们后来因为不再需要 RAG 而停止使用。
   - 另一个建议是将外部库复制到共享文件夹中，以获得更好的集成可能性。
- **Anthropic 发布 Model Context Protocol**：[Cloudflare 的博客](https://blog.cloudflare.com/model-context-protocol/)重点介绍了 Anthropic 推出的新 Model Context Protocol (MCP)，它允许 AI 通过 Cloudflare Workers 使用极少量的代码与服务进行交互。
   - MCP 旨在作为 LLM 连接应用程序的通用接口，被比作 USB-C 接口。
- **GritQL 作为潜在解决方案**：[GritQL](https://github.com/getgrit/gritql) 被介绍为一种极具前景的开源查询语言，用于搜索和修改代码，尽管它目前不支持 C#。
   - 用户讨论了在 AI 环境中探索其代码生成和精确 diff 功能的可能性。
- **关于代码生成和 AI 局限性的演讲**：一段名为“[大规模代码生成与维护](https://www.youtube.com/watch?v=Ve-akpov78Q)”的 YouTube 视频讨论了 AI Agent 在处理大规模代码库时面临的挑战。
   - 用户对该演讲表示出兴趣，但也指出了 GritQL 等现有工具在某些语言上的潜在局限性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.trydepth.ai">Depth AI - 深度理解代码库的 AI</a>：与你的代码库聊天或构建定制的 AI Assistant。将它们部署在任何工作场所 —— Slack、GitHub Copilot、Jira 等。</li><li><a href="https://www.youtube.com/watch?v=Ve-akpov78Q"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/getgrit/gritql">GitHub - getgrit/gritql: GritQL 是一种用于搜索、lint 和修改代码的查询语言。</a>：GritQL 是一种用于搜索、lint 和修改代码的查询语言。 - getgrit/gritql</li><li><a href="https://blog.cloudflare.com/model-context-protocol/">你好 Claude，在 Cloudflare Workers 上构建 MCP 服务器</a>：想让 Claude 直接与你的应用交互吗？在 Workers 上构建一个 MCP 服务器。这将使你能够直接连接你的服务，让 Claude 能够理解并代表你执行任务。 
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1319759084836229121)** (460 messages🔥🔥🔥): 

> `Phi-4 Model Performance, Quantization Methods, Local vs Cloud Model Running, Reasoning Capabilities of Models, Mean Generation Speeds` 


- **Phi-4 模型幻觉 (Halos)**：用户注意到 Phi-4 模型表现出明显的幻觉倾向，导致即使在简单的 Prompt 下也会出现不准确的情况，但在编程任务中表现良好。
   - 尽管其官方宣称具备很强的能力，但人们对其通用知识和多轮对话处理能力仍存有疑虑。
- **探索量化技术**：讨论围绕模型的各种可用量化方法展开，指出了 QTIP 和 AQLM 等技术的优缺点。
   - QTIP 在 2-bit 量化下显示出保持性能的潜力，但目前的实现方式仍然有限。
- **本地模型基准测试**：参与者对能适配 10 GB VRAM 的模型表现出兴趣，并考虑采用能保持最高质量的量化方法。
   - 用户强调需要更多汇总资源来讨论模型性能和量化技术。
- **LLM 性能对比**：对话强调了在线论坛对话质量的下降，人们的关注点转向对新模型发布的兴奋，而非深入讨论。
   - 成员们指出，有必要对各种开源 LLM 的能力和局限性有更深入的见解。
- **生成速度见解**：用户分享了在不同模型上的生成速度体验，报告了受上下文长度和硬件配置影响的波动 TPS 速率。
   - 一位用户记录了在特定设置下达到的合理速度，而其他人则考虑了 VRAM 和量化对性能的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://models.litellm.ai/">contextlengthof.com</a>: 未找到描述</li><li><a href="https://mistral.ai/news/mixtral-of-experts/">Mixtral of experts</a>: 一个高质量的 Sparse Mixture-of-Experts。</li><li><a href="https://huggingface.co/posts/m-ric/853337605317831">@m-ric on Hugging Face: &quot;𝐇𝐮𝐠𝐠𝐢𝐧𝐠 𝐅𝐚𝐜𝐞 𝐫𝐞𝐥𝐞𝐚𝐬𝐞𝐬 𝐏𝐢𝐜𝐨𝐭𝐫𝐨𝐧, 𝐚…&quot;</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2310.10837">Approximating Two-Layer Feedforward Networks for Efficient Transformers</a>: 如何在不牺牲性能的情况下减少神经网络 (NNs) 的计算和内存需求？许多最近的研究使用稀疏 Mixtures of Experts (MoEs) 来构建资源高效的 LLM...</li><li><a href="https://x.com/sauers_/status/1870197781140517331?s=46">Tweet from Sauers (@Sauers_)</a>: 总计算成本约为 $1,600,250，超过了整个奖金。</li><li><a href="https://tenor.com/view/deep-thought-thinking-loading-buffering-gif-16392522">Deep Thought Thinking GIF - Deep Thought Thinking Loading - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/Apptronik/status/1869745753968849138">Tweet from Apptronik (@Apptronik)</a>: 重大新闻 - 我们已与 @GoogleDeepMind 的机器人团队联手！🧠🦾 我们将把顶级的 #AI 与尖端的机器人硬件相结合，创造先进的 AI 驱动的人形机器人。</li><li><a href="https://tenor.com/view/ba-dum-tsss-drum-band-gif-7320811">Ba Dum Tsss Drum GIF - Ba Dum Tsss Drum Band - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/matteogeniaccio/phi-4">matteogeniaccio/phi-4 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/casper_hansen_/status/1870775546856243268?s=46&t=QUL78vIQDJohFpnIzCbQXA">Tweet from Casper Hansen (@casper_hansen_)</a>: 今天，我发布了 OpenCoconut 的早期版本，旨在复制我们在潜空间进行推理的 Chain of Continuous Thought。</li><li><a href="https://www.youtube.com/live/SKBG1sqdyIU?feature=shared&t=269"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf">arc-prize-2024/the_architects.pdf at main · da-fr/arc-prize-2024</a>: 我们针对 arc challenge 2024 的解决方案。通过在 GitHub 上创建账号为 da-fr/arc-prize-2024 的开发做出贡献。</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLScvapN_zT3vIBr7KVu5azDwG1DhjlSt8kuOtjaSkygLj7JLkA/viewform">Mental healthcare in Oman .</a>: 关于阿曼人民面临的问题的调查。</li><li><a href="https://itunes.apple.com/app/id1544827472">‎Form for Google Forms</a>: ‎使用免费的 Form 应用在 Mac 上创建、编辑和管理所有的 Google 表单。通过此应用，您可以：创建新表单、有大量精美模板可供选择...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1319788432666394747)** (9 条消息🔥): 

> `Instruction Tuning on Raw Text, Training BERT for Classification, KV Cache Architectures, Qwen 32 vs Hermes 70B` 


- **关于原始数据指令微调的担忧**：成员们讨论了在来自 **PubMed** 或教科书等来源的原始文本数据上训练 **llama3.1-8b-instruct** 等 **instruction-tuned LLM** 的影响，建议可能需要将数据转换为 Q/A 对。
   - *一位成员思考了*将基础模型与官方 instruct 模型合并以获得更好效果的可能性。
- **BERT 分类的数据需求**：一位成员询问了训练用于分类任务的 **BERT** 模型所需的数据量，另一位成员回答说这取决于任务难度和涉及的类别数量。
   - 他们强调，与预训练模型训练数据的相似性在数据需求中也起着重要作用。
- **对快速 KV Cache 技术的兴趣**：一位成员询问了目前可用的最快 **KV cache architectures** 和技术，表明了对优化这一方面的持续兴趣。
   - 这个问题没有得到直接回应，凸显了在该领域进一步探索的必要性。
- **对比咨询：Qwen 32 vs Hermes 70B**：一位成员提出了关于 **Qwen 32** 和 **Hermes 70B** 的对比问题，寻求对性能差异的见解。
   - 回复中没有提供具体的对比，为未来的探索留下了一个开放的讨论点。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1320204149769637920)** (2 条消息): 

> `Medical LLMs, Depth Completion with GANs, Clinical Trust in AI, Multimodal Medical Models, Ethics in Medical AI` 


- **医疗 LLMs 开启新功能**：本周重点介绍了 **Medical LLMs** 的进展，如 **MedMax**，它集成了多模态生物医学辅助并提供了增强的**报告生成能力**。
   - 此外还讨论了 **MGH Radiology Llama 70B**，它专注于放射学并展示了 **state-of-the-art performance**。
- **框架与方法推动临床应用**：**ReflecTool** 和 **Federated Learning with RAG** 成为开发 **Reflection-Aware Clinical Agents** 和优化查询流水线的关键工具。
   - 这些创新标志着在改进**临床笔记处理**和患者互动方面迈出了一步。
- **评估塑造医疗 AI 的未来基准**：**Multi-OphthaLingua** 基准旨在解决低中收入国家（LMICs）的医疗偏见，同时实施了诸如 **ACE-M3** 之类的综合评估框架。
   - 这些框架有助于使用标准化指标对**多模态医疗模型**进行严格测试。
- **LLMs 在医疗保健中的新兴应用**：讨论的创新应用包括**患者友好型视频报告**和**医疗视频 QA 系统**，旨在提高患者参与度。
   - 这些发展强调了对**以用户为中心的医疗解决方案**日益增长的重视。
- **深入探讨医疗 AI 的伦理问题**：关键讨论集中在 **Mental Health AI** 的伦理挑战以及 AI 对临床信任的影响，强调了负责任集成的必要性。
   - 针对 **Radiology AI Integration** 提出了担忧，强调这是一个需要仔细考虑的敏感领域。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1870504774162063760>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：🌟 每周医疗 AI 研究汇总 🌟📅 2024年12月15-21日。这是您每周最重要的医疗 AI 论文摘要！🎉🤖 Medical LLM & 其他模型 - MedMax: 混合模态生物医学辅助...

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1320204149769637920)** (2 条消息): 

> `Medical LLMs, Frameworks for Clinical AI, Depth Completion Techniques, Medical Ethics in AI` 


- **每周医学 AI 研究亮点**：最新的汇总揭示了医学 AI 的关键主题，包括 **MedMax**（一种混合模态生物医学助手）以及 **MGH Radiology Llama 70B**（专门用于放射科，具有增强的报告生成能力）的进展。
   - 讨论了 **ReflecTool** 等框架，旨在通过 **federated learning**（联邦学习）方法改进临床笔记的处理。
- **新兴的医学 AI 基准测试**：**Multi-OphthaLingua** 基准测试专注于眼科的多语言能力，特别是评估 **LMICs**（中低收入国家）医疗保健中的 **biases**（偏见）。
   - **ACE-M3 Evaluation Framework** 为多模态医学模型的全面评估提供了标准化指标。
- **医学伦理与 AI 讨论**：重点关注医学伦理，强调了在放射科集成 AI 的挑战，处理诸如 **临床信任和心理健康 AI** 影响等问题。
   - 对话强调了在医院监控系统集成 AI 过程中进行伦理考量的必要性。
- **寻求深度补全（Depth Completion）领域的论文选题**：一位成员表示需要硕士论文选题指导，旨在将研究重点从 *使用 GANs 进行合成深度生成* 转向 *depth completion* 技术。
   - 他们正在寻求建议，因为他们大约有 **6 个月** 的时间来完成研究。



**提到的链接**: <a href="https://x.com/OpenlifesciAI/status/1870504774162063760>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>: 🌟 每周医学 AI 研究汇总 🌟📅 2024 年 12 月 15-21 日。这是您每周最重要的医学 AI 论文摘要！🎉🤖 Medical LLM & 其他模型 - MedMax: 混合模态生物医学助手...

  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1319819670206939167)** (1 条消息): 

> `Reasoning dataset creation, Collaborative dataset project, Use of <think> tag, Model targeting, Research and study` 


- **协作推理数据集倡议**：一位成员计划创建一个 **reasoning dataset**（推理数据集），并正在寻找合作伙伴加入。
   - 该项目将利用 `<think>` 标签来描述思维过程，随后在同一模型中生成综合答案。
- **专注于模型能力**：目标是开发专注于 **o1-preview** 或 **o3** 等模型的数据集，以获得更好的结果。
   - 强调了“让我们一起学习、研究并构建数据集”是该方法的核心部分。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1319756390004424827)** (262 条消息🔥🔥): 

> `OpenAI O3, GPT-5 delays, ARC-AGI performance, AI job market, Evaluating reasoning in AI` 


- **OpenAI O3 的发布引发争论**：在 O3 发布后，关于它是否符合纯 LLM 的定义产生了一些争论，因为它使用了先进的技术和结构化推理方法。
   - 讨论指出，O3 可能更多地是利用现有能力，而非仅仅是更深层的学习，这引发了关于 LLM 未来的疑问。
- **GPT-5 开发面临挑战**：报告指出，代号为 Orion 的 GPT-5 由于训练数据多样性不足而推迟，导致成本增加且结果不可预测。
   - OpenAI 已经进行了多次训练运行，但出现了意料之外的问题，使得这些成本的预期回收变得不确定。
- **对推理模型的担忧**：最近的一项研究表明，推理模型可能只是在模仿其训练数据，而不是在解决新问题，这引发了对其性能可靠性的质疑。
   - 对该研究的批评指出了评估这些推理模型时存在的潜在缺陷，表明可能需要进一步审查。
- **AI 就业市场推测**：参与者对 AI 技术的兴起可能导致就业市场严重中断表示担忧，特别是对于白领工人。
   - 论点指向了类似变革的历史背景，指出这些转型将如何表现并影响各个行业具有不可预测性。
- **关于 AI 集成的新兴批评**：随着 AI 系统融入社会，讨论指出了可能出现的潜在风险和社会挑战，包括失业和社会动荡。
   - 参与者辩论了不同地区对技术进步的不同态度，特别是对比了西方和中国社区的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

</div>

<ul>
<li>
<a href="https://x.com/fchollet/status/1870172872641261979">来自 François Chollet (@fchollet) 的推文</a>：分析新系统的优势和局限性也将极其重要。以下是一些 o3 在高算力设置下无法解决的任务示例（即便它正在生成...）</li><li><a href="https://x.com/dylan522p/status/1870213495641256109">来自 Dylan Patel (@dylan522p) 的推文</a>：那些家伙都在市价买入 Nvidia 股票，因为 OpenAI o3 实在太他妈强了</li><li><a href="https://x.com/ns123abc/status/1870207399329739164">来自 NIK (@ns123abc) 的推文</a>：笑死我了，Dylan Patel 彻底把他怼得哑口无言</li><li><a href="https://x.com/_arohan_/status/1870378565898007005">来自 rohan anil (@_arohan_) 的推文</a>：让我印象深刻的一点是“关于‘tuned’的说明：OpenAI 分享称，他们用于测试的 o3 是在 75% 的 Public Training set 上训练的”。就取得的结果而言这没问题，但难道不是...</li><li><a href="https://x.com/Jaykef_/status/1870616894107205867">来自 Jaward (@Jaykef_) 的推文</a>：@denny_zhou Yann 会有不同意见——在他看来，自回归 LLM 的推理毫无美感。他现在认为 o3 已经不再是 LLM 了，哈哈。</li><li><a href="https://x.com/voooooogel/status/1870339243803070960">来自 thebes (@voooooogel) 的推文</a>：@bio_bootloader 以及预期的解决方案</li><li><a href="https://x.com/TheXeophon/status/1870200233935949891">来自 Xeophon (@TheXeophon) 的推文</a>：o3 极有可能由下一代模型 GPT-5 驱动。在直播中，o3 编写了使用 openai python 包的代码并运行正确——即便最新版本的 o1 仍受困于...</li><li><a href="https://x.com/GregKamradt/status/1870208490096218244">来自 Greg Kamradt (@GregKamradt) 的推文</a>：我们在 @arcprize 上验证了 OpenAI 的 o3 结果。当我看到他们用来宣称得分的 prompt 时，我的第一反应是……“就这？”看到那个 prompt，感觉非常新颖（令人印象深刻）……</li><li><a href="https://x.com/GregKamradt/status/1870183792050311659">来自 Greg Kamradt (@GregKamradt) 的推文</a>：这张图表提出的真正问题是：* 曲线会趋于平缓吗？还是会继续增长？* 衡量效率的正确指标是 compute 还是成本？* o3 不仅仅是“更多 compute”。架构上还有更多改进……</li><li><a href="https://x.com/sama/status/1870709421111984135">来自 Sam Altman (@sama) 的推文</a>：我认为《华尔街日报》是目前美国最好的报纸，但就在我们发布（o3）几小时后，他们发表了一篇名为《AI 的下一次大飞跃进度落后且成本极高》的文章……</li><li><a href="https://open.substack.com/pub/desirivanova/p/on-some-fixable-limitations-of-understanding?r=37tb0m&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false">Inbox | Substack</a>：未找到描述</li><li><a href="https://x.com/_jasonwei/status/1870184982007644614">来自 Jason Wei (@_jasonwei) 的推文</a>：o3 的性能非常强大。更重要的是，从 o1 到 o3 的进展仅用了三个月，这展示了在通过 chain of thought 上的 RL 来扩展 inference compute 的新范式下，进步速度会有多快。</li><li><a href="https://x.com/MatthewBerman/status/1870189248923742693">来自 MatthewBerman (@MatthewBerman) 的推文</a>：.@OpenAI 刚刚发布了 o3 和 o3-mini！这就是 AGI（非标题党）。o3 是有史以来最强大的 AI，其表现简直疯狂。以下是你需要了解的一切：🧵</li><li><a href="https://github.com/arcprizeorg/model_baseline">GitHub - arcprizeorg/model_baseline: Testing baseline LLMs performance across various models</a>：在各种模型上测试基准 LLM 的性能 - arcprizeorg/model_baseline</li><li><a href="https://archive.ph/2024.12.21-093402/https://www.wsj.com/tech/ai/openai-gpt5-orion-delays-639e7693">OpenAI 的下一项重大 AI 计划 GPT-5 进度落后且成本极高……</a>：未找到描述
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1320038721394966642)** (8 条消息🔥): 

> `Long Context Training, Llama 团队变动, Rohan Anil 的去向, 新模型策略, AGI 目标` 


- **关于长上下文训练的推测**：此前关于能够处理高达 **100 万 token** 的**长上下文训练 (long context training)** 有过一阵热潮，引发了学术界的广泛关注。
   - 目前尚不清楚该领域的进展是源于**算力需求 (compute demands)** 还是创新技术。
- **Llama 团队职位变动**：讨论透露，曾负责 **Gemini** 系列的 **Arohan Anil** 将加入 [Meta 的 Llama 团队](https://x.com/_arohan_/status/1866621771451076812?s=61)，致力于下一代模型的研发。
   - 社区成员推测，这一举动可能会带来令人兴奋的创新，**Llama 4** 可能会从这次人事变动中受益。
- **Rohan Anil 从 Google 离职**：Rohan Anil 宣布从 Google 离职，表示他正在寻求新的环境来继续他的 AI 工作，正如他在[公开消息](https://x.com/_arohan_/status/1865089129677230322)中所强调的那样。
   - 他对在 Google 的时光表示感谢，并提到他对 **Gemini** 等项目的贡献非常有意义。
- **模型开发的新策略**：成员们讨论了 **Llama** 和 **Gemini** 训练技术之间的差异，并推测可能会使用 **Ring attention** 来提高效率。
   - 对话表明，团队成员不同背景的融合可能会促进模型开发中的创新策略。
- **AGI 与开源目标**：Rohan 在 Meta 的新职位伴随着为 **AGI** 做出贡献的雄心，同时致力于创建一个**更健康的创新生态系统**。
   - 他呼应了扎克伯格关于致力于**开源 AGI** 开发的观点，这激发了 AI 领域的竞争精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/_arohan_/status/1865089129677230322)">来自 rohan anil (@_arohan_) 的推文</a>: 对我来说这是一个苦乐参半的时刻，Gemini 表现很好，团队也很棒。我在 G 度过了近 12 年的美好时光，甚至可以被称为 OG。例如，对于每一个搜索查询，我注意到……</li><li><a href="https://x.com/_arohan_/status/1866621771451076812?s=61">来自 rohan anil (@_arohan_) 的推文</a>: 秘密公开了：我将于下个月加入 @AIatMeta 的 Llama 团队，致力于下一代 Llama 模型的研发。是的，在下一个……之前，我已经准备好了一些 Llama 的双关语。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1319763690362376233)** (2 条消息): 

> `删除的内容, 法律问题` 


- **内容删除引发推测**：一条消息被删除，引发了对其内容和删除原因的推测。
   - 一位成员开玩笑说，**律师可能已经介入了**，暗示了潜在的法律影响。
- **引发法律担忧**：消息的删除引发了关于内容共享可能存在的法律问题的讨论。
   - 成员们对这种行为的动机表示好奇。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1319809458561482754)** (55 条消息🔥🔥): 

> `O3 API 增强、Gemini 项目更新、用户 Sora 访问权限、ChatGPT 律师参与、模型评估讨论` 


- **O3 API 获得令人兴奋的参数**：一位成员报告称 **O3 API** 包含 **sample parameter**（采样参数）和 **thinking effort parameter**（思考强度参数），证实了对其能力的猜测。
   - *这表明如果基础模型足够强大，性能提升的潜力巨大。*
- **Gemini 团队假期计划面临危机**：讨论强调了 **Gemini 团队** 紧迫的日程安排，由于 1 月初需要在 **Supreme Court** 提交陈述，暗示他们可能不得不取消假期计划。
   - *他们开玩笑说这会对项目时间线产生影响，尤其是考虑到即将发布的新更新。*
- **宣布面向所有用户开放 Sora 访问权限**：得益于假期期间 GPU 负载较低，[Sora 访问权限](https://x.com/sama/status/1870524745302839559) 已通过宽松队列（relaxed queue）扩展至所有 Plus 用户。
   - Teams 用户也获得了 Sora 的访问权限，并具备了支持分享创作链接的功能。
- **假期期间保持活跃**：一位成员计划去度假，但敦促其他人保持 **sports 频道** 的活跃，并使用 **AOE2 房间** 进行互动。
   - *尽管在休假，他们仍打算潜水，并建议其他“成瘾者”也放下工作。*
- **在趣味场景中评估 LLM**：一位成员提议将 **LLM** 与 **Neal's Password Game** 进行对比评估，表示观察它们在长时间交互中的表现会很有趣。
   - 这引发了其他人对利用此类评估来衡量推理能力的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1871105574076621070">来自 Xeophon (@TheXeophon) 的推文</a>: LLM vs. Neal's Password Game！看看它们能走多远会很有趣，尤其是随着对话的深入。引用 Xeophon (@TheXeophon)：我有一个超酷的评估想法，稍后分享...</li><li><a href="https://x.com/rohanjamin/status/1870525134664278331">来自 Rohan Sahai (@rohanjamin) 的推文</a>: Shipmas 第 13 天 Sora 奖励？？🎉✨我们还向所有 Teams 用户推出了 Sora 访问权限，升级了 blend 功能，并启用了共享链接，这样你就可以与朋友分享 Sora 的创作——即使他们没有...</li><li><a href="https://x.com/typedfemale/status/1870300757288989012">来自 typedfemale (@typedfemale) 的推文</a>: 我从今天学到的：如果你对什么是推理有强烈的看法——永远不要制作一个能让你的批评者公开证明你错误的数据集</li><li><a href="https://x.com/sama/status/1870524745302839559">来自 Sam Altman (@sama) 的推文</a>: Shipmas 第 13 天：特别的 Sora 奖励🎄✨12 月下旬随着人们休假，我们的 GPU 负载有所减轻，因此我们通过宽松队列（relaxed queue）向所有 Plus 用户提供无限的 Sora 访问权限...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1319760936277512254)** (32 条消息🔥): 

> `OpenAI 融资策略, 黎曼问题, GPT 性能, Discord 表情符号, Meme 收集频道` 


- **OpenAI 的秘密融资策略**：讨论中提到了 OpenAI 的评估方法，据报道他们在评估中**每个任务的花费超过 1000 美元**，这导致了公共模型访问的延迟。
   - *难怪他们还不给公众开放模型访问权限！* 突显了其评估策略中令人惊讶的成本。
- **关于新“黎曼问题”的辩论**：一位成员提出了一个**紧迫的问题**：**o_i 系列**是奇数编号还是质数编号，并幽默地提到必须等到 **o7** 之后才能揭晓。
   - *猜得等到 o7 之后了..* 为这场数学讨论增添了一丝期待感。
- **GPT 性能下降**：在一份引人注目的声明中，有人断言 GPT 正面临**边际收益递减**阶段，这与之前对其性能放缓的预测一致。
   - 这引发了关于即将推出的 **Orion 模型**的讨论，该模型旨在增强推理能力并在初始训练后进行微调。
- **Discord 表情符号开发**：对话强调了添加表情符号的管理挑战，一位成员对添加新表情符号相关的成本表示好奇。
   - 成员们回忆起**自定义表情符号**，包括一个以著名人物为特色的表情，并讨论了它们的创作过程。
- **关于 Meme 收集频道的提案**：一位成员建议创建一个**收集幽默帖子**和 Meme 的频道，表明了对精选内容日益增长的兴趣。
   - 有人幽默地建议 *这是一个 Copium（精神慰藉）频道*，体现了社区围绕 Meme 的有趣文化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/1thousandfaces_/status/1870179551567065340">来自 hero ⚔️ (@1thousandfaces_) 的推文</a>: o3 的秘密？“如果你正确完成这个任务，我给你 1000 美元”的提示词，但你实际上真的把钱发过去。引用 Tenobrus (@tenobrus) 的话，他们在每个任务上花费超过 1000 美元...</li><li><a href="https://x.com/rao2z/status/1870217915934617662">来自 Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>: 新的紧迫黎曼问题：o_i 系列是奇数编号还是质数编号？（猜得等到 o7 之后了..）</li><li><a href="https://x.com/anpaure/status/1870201437537419615">来自 anpaure (@anpaure) 的推文</a>: @stochasticchasm @soradotsh nathan lambert 完全被证明是正确的？</li><li><a href="https://x.com/owl_posting/status/1870197470187401577">来自 owl (@owl_posting) 的推文</a>: 未找到描述</li><li><a href="https://x.com/GaryMarcus/status/1855382564015689959">来自 Gary Marcus (@GaryMarcus) 的推文</a>: 伙计们，游戏结束。我赢了。GPT 正在进入边际收益递减期，正如我所说的那样。引用 Amir Efrati (@amir) 的新闻：OpenAI 即将推出的 Orion 模型展示了 GPT 的改进是如何放缓的...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1320093494781673554)** (16 条消息🔥): 

> `Deliberate Alignment 方法, Tulu 3 输出验证, LLM as Judge 奖励, 奖励模型挑战` 


- **Deliberate Alignment 方法使用 RM**：OpenAI 的 **Deliberate Alignment 方法**采用了**奖励模型 (RM)**，这是一个评估 **(prompt, output, specs)** 并分配分数的 LLM。人们对这种奖励机制中潜在的噪声表示担忧。
   - _一位成员推测，既然是 OpenAI，该模型可能相当可靠_。
- **Tulu 3 输出是可验证的**：对于 **Tulu 3**，讨论确认其输出主要是可验证的，类似于数学结果。另一位参与者建议，对于同样使用 LLM 来增强验证的 **O3** 输出，情况可能也是如此。
   - _**LLM** 被用于评估推理安全性，因为通过少数手写规则来验证安全性可能具有挑战性。_
- **可验证领域 vs 安全性**：成员们讨论了 **LLM as judge** 主要用于安全性验证而非推理。他们引用了 **Deliberate Alignment 论文**来支持这一观点。
   - _对于 LLM 是用于所有 O 系列还是仅用于侧重安全性的模型，似乎存在不确定性。_
- **定义奖励的挑战**：围绕有多少领域可以切实可行地变为“可验证”展开了辩论，涉及奖励定义的复杂性。一位成员指出，虽然具身 Agent (Embodied agents) 获得有形奖励较为简单，但 LLM 需要更现实的环境上下文来获取奖励。
   - _这引发了与 **'Reward is All You Need'** 讨论的对比，强调了有效定义奖励的难度。_


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1320064395451301890)** (12 messages🔥): 

> `The Nvidia Way, Bio-maxxing literature, Asimov Press, Mindless listening while working, Reading group discussions` 


- **探索《The Nvidia Way》**：一位成员刚开始听 **The Nvidia Way** 的有声书，另一位成员表示有兴趣交流笔记。
   - 这引发了成员之间关于各种阅读兴趣的讨论。
- **深入探讨 Bio-maxxing 文献**：一些成员分享了与 Bio-maxxing 相关的书籍，如 **The Vital Question** 和 **The Song of the Cell**，并寻求高中水平生物学的推荐书目。
   - 一位成员提到了 **Asimov Press**，强调它是撰写生物学进展相关文章的重要资源。
- **适合循环播放的 YouTube 视频**：一位成员分享了两个他们在处理机械性任务时觉得很有吸引力的 YouTube 视频，并指出这些视频与当前的讨论相关。
   - 他们评论说某些视频的观看次数有限（特别是 Ross Taylor 的一个视频），同时思考着所讨论话题的持续进展。
- **引人入胜的读书小组想法**：成员们对组建迷你读书小组表现出极大热情，讨论了诸如 **On Grand Strategy** 和 **AI 2041** 等书目。
   - 这突显了社区对协作探索文献和思想的兴趣。
- **播客大爆发！**：一位成员幽默地评论了目前海量的播客资源，反映了社区中的普遍感受。
   - 这可能为未来讨论中分享播客推荐开辟了道路。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.owlposting.com/">Owl Posting | Abhishaike Mahajan | Substack</a>: 关于生物学和 ML 的文章，由猫头鹰撰写并为猫头鹰而写。点击阅读 Owl Posting，由 Abhishaike Mahajan 创办，拥有数千名订阅者的 Substack 出版物。</li><li><a href="https://asimovpress.substack.com?r=7g0n1&utm_medium=ios&utm_source=profile">Asimov Press | Substack</a>: Asimov Press 是一家电子杂志，专门刊登关于生物学进展的文章。</li><li><a href="https://youtu.be/QVcSBHhcFbg?si=oSOrSw2MLXrHOtj7"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/6PEJ96k1kiw?si=yQ9YrneW4q--sbIp"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/S5l5OvJ01ws?si=jOwMdQ1PChZMW6E8"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1320102433166200993)** (35 messages🔥): 

> `Training OLMo-2 13B, Model Fine-Tuning vs RAG, Trust in AI Models, Prompting Techniques, Open Models Discussion` 


- **为聊天机器人 Fine-Tuning OLMo-2 13B**：几位成员讨论了 Fine-Tuning **OLMo-2 13B** 以创建知识库和领域专家聊天机器人的可能性，强调了其对社区支持的相关性。
   - *一位成员提醒道*，Fine-Tuning 可能存在风险，强调在进行之前需要仔细考虑。
- **RAG 与 Fine-Tuning 之争**：一位用户询问为什么在他们的项目中 Fine-Tuning 优于检索增强生成（**RAG**），这引发了关于两种方法挑战和优势的深入见解。
   - *另一位成员建议*先使用 Prompting 来获取初始模型行为，仅在 Prompting 无法产生理想结果时才保留 Fine-Tuning。
- **AI 模型的信任度**：讨论围绕对 AI 模型的信任展开，特别是关于 **Ai2** 模型与 Meta 的 LLaMA 相比的开放性，一些人对 Meta 的做法表示怀疑。
   - *一位参与者指出*，虽然 LLaMA 可能更优越，但 Ai2 对开放性的承诺在该领域脱颖而出。
- **Prompting 技巧与资源**：成员们分享了有效 Prompting 的资源，重点介绍了 **Anthropic** 的 Prompting 指南以及辅助自动生成 Prompt 的工具。
   - *一位用户提到*，利用这些资源可以简化流程并改善模型交互。
- **Open Models 与社区反馈**：关于 **OLMo** 可用性的反馈是积极的，用户注意到它在发布时的易用性和易复制性。
   - *参与者一致认为*，像 Ai2 这样的 Open Models 促进了协作学习环境。



**提及的链接**: <a href="https://github.com/axolotl-ai-cloud/axolotl">GitHub - axolotl-ai-cloud/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. 通过创建账户为 axolotl-ai-cloud/axolotl 的开发做出贡献。

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1319808824529653770)** (20 条消息🔥): 

> `OpenAI 的 o3 模型，LLAMA 3.3 发布，推理 AI 模型，Anthropic 节日惊喜，订阅价格更新` 


- **OpenAI 预览 o3 模型**：OpenAI 展示了其即将推出的 [o3 模型](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai)，预计将于 2025 年 1 月公开发布，展示了相比 o1 在推理能力上的进步。
   - 最近的讨论强调了 2024 年 AI 发展缺乏兴奋点，但 o3 的发布旨在通过意想不到的改进来打破这一趋势。
- **LLAMA 3.3 现已发布**：Meta [LLAMA 3.3](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 70B 模型已发布，针对多语言对话进行了优化，并在关键基准测试中优于其他模型。
   - 它采用优化的 Transformer 架构构建，并结合了监督微调（Supervised Fine-tuning）和强化学习（Reinforcement Learning）方法进行训练。
- **关于 AI 推理的辩论**：一些成员讨论了推理 AI 模型的效用，质疑它们应该取代人类推理，还是仅仅辅助解决问题。
   - 像某位用户这样的批评者提到了 o1 等模型输出的冗长性，指出它通常不如之前的迭代版本有帮助。
- **Anthropic 潜在的节日惊喜**：社区传闻 **Anthropic** 可能很快会有意想不到的公告，引发了期待。
   - 一位成员开玩笑地评论说，Anthropic 的风格通常过于稳健（wholesome），不太可能有这种惊喜。
- **即将到来的订阅价格上涨**：有消息提醒，由于平台内容的实质性增长，Interconnects 计划在新的一年提高订阅价格。
   - 目前正向现有成员提供年度订阅折扣，是续订的好时机。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__nmca__/status/1870170101091008860">来自 Nat McAleese (@__nmca__) 的推文</a>：o1 是第一个大型推理模型——正如我们在最初的“学习推理”博客中所述，它“仅仅”是一个通过 RL 训练的 LLM。o3 是在 o1 基础上进一步扩展 RL 的产物，其强度...</li><li><a href="https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai">o3：2024 年 AI 的压轴大戏</a>：一次与 GPT-4 发布同样具有影响力的阶段性变革。推理语言模型是当前的重头戏。</li><li><a href="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct">meta-llama/Llama-3.3-70B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://open.substack.com/pub/robotic/p/openais-o1-using-search-was-a-psyop?r=7g0n1&utm_medium=ios)">OpenAI 的 o1 使用“搜索”是一场心理战（PSYOP）</a>：如何理解 OpenAI 的 o1 模型，它实际上只是一个古怪、奇妙且漫长的思维链（Chain of Thought）
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1319763691612278796)** (1 条消息): 

> `Mistletokens，来自 Bolt 的节日礼物，Free 和 Pro 用户福利` 


- **Bolt 为节日赠送 Mistletokens**：Bolt 团队在 [X](https://x.com/stackblitz/status/1870203756995911707) 上宣布了一项名为 **Mistletokens** 的特别节日礼物，面向所有用户。
   - *节日快乐！* 礼物包括为 Pro 用户提供 **200 万免费 Token**，以及为 Free 用户提供每日 **20 万**、每月 **200 万** Token，有效期至今年年底。
- **Pro 用户 Token 激励**：Pro 用户很高兴收到 Bolt 赠送的 **200 万** 免费 Mistletokens 作为节日礼物。这些 Token 可用于显著增强他们的项目。
   - 消息强调，这次激励旨在激发创意和创新。
- **Free 用户每日限制**：在这个节日期间，Free 用户将受益于每日 **20 万 Token** 的慷慨配额，以及每月总计 **200 万** 的限额。
   - 这一举措旨在吸引 Free 用户，并鼓励他们利用提供的资源构建出色的项目。



**提到的链接**：<a href="https://x.com/stackblitz/status/1870203756995911707">来自 StackBlitz (@stackblitz) 的推文</a>：节日快乐！我们的团队再次为大家准备了一份特别的礼物：🎄 我们称之为 Mistletokens！🎄 截止到年底：🔔 所有 Pro 用户获得 200 万免费 Token！🔔 所有 Free 用户获得每日 20 万 & 每月 200 万...

  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1320022605142560778)** (15 messages🔥): 

> `Bolt Studio 发布, DataCloneError 问题, Prompting 最佳实践, AI 效率研究, Token 使用担忧` 


- **Bolt Studio 即将准备就绪**：一名成员宣布 **Bolt Studio** 已接近完成，并强调了其在项目脚手架（scaffolding）方面的实用性。
   - 该工具预计将极大地帮助用户有效地组织他们的项目。
- **DataCloneError 困惑**：一位用户报告在尝试使用 postMessage 函数时遇到 **DataCloneError**，并寻求社区建议。
   - 另一位成员建议减少 Prompting 可能有助于避免此类问题，并建议创建一个讨论串以理清思路。
- **Token 使用的最佳实践**：一位成员建议在正式使用 Bolt 之前，利用另一个 AI 系统来审查 Prompt，旨在节省 **Token**。
   - 另一位 Bolt 新手对此表示赞同，并对最近使用过程中的 Token 消耗感到沮丧。
- **AI 学习模式讨论**：一位成员分享了一项研究见解，该研究表明 AI 的效率会根据一年中的时间而变化，在 **8月** 和节假日期间表现尤为糟糕。
   - 他们提议根据这些信息调整 Prompt 可能会提高输出质量。
- **对 Token 支出的沮丧**：多位成员表达了对浪费 **Token** 的不满，其中一人表示今天在没有获得满意结果的情况下损失了时间和金钱。
   - 还有人对 Bolt 效能下降的感知提出了担忧，并提醒大家等待 Token 刷新。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1319757346578366595)** (402 messages🔥🔥): 

> `Bolt 中的 Token 使用, 在 Bolt 中集成 API, CORS 问题, Bolt 的 GitHub 集成, 部署应用程序` 


- **简单修复导致的高 Token 消耗**：用户报告在 Bolt 中进行简单操作时 Token 消耗很高，有人声称微小的改动就消耗了超过 300k Token。
   - AI 在每次 Prompt 时都会发送整个代码库进行审查，这增加了 Token 支出；鼓励用户等待效率改进。
- **集成 Supabase 和 API**：用户讨论了在现有的 Bolt 项目中添加 Supabase Auth 和其他集成，并对潜在问题表示担忧。
   - 建议在继续进行后端集成之前保持谨慎，并等待当前问题的解决。
- **Bolt 中的 CORS 和 Request Relay**：用户在应用程序中遇到了 CORS 问题，并讨论了启用 Request Relay 功能作为潜在的变通方案。
   - 一些用户分享了在集成 Stripe 等 API 时遇到类似问题的经验。
- **GitHub 集成问题**：用户注意到 Bolt 的 GitHub 集成存在问题，建议团队应致力于改进功能。
   - 建议在 Bolt 之外手动管理代码版本，以避免丢失数据和进度。
- **对加密货币换皮项目的担忧**：社区对一些试图为加密货币项目“换皮” Bolt 并可能进行误导性筹款的项目表示担忧。
   - 成员们表达了对与这些项目相关的 Rug pulls（卷款跑路）风险的担忧，并将其比作加密货币领域更广泛的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://nnelson.de">Vite + React + TS</a>: 未找到描述</li><li><a href="https://support.bolt.new/github-known-issues">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队提供的一体化工作空间。</li><li><a href="https://www.twitch.tv/juliansarokin">Twitch</a>: 未找到描述</li><li><a href="https://imgbb.com/">Upload Image — 免费图片托管</a>: 免费的图片托管和分享服务，上传图片、照片托管。为论坛上传图片提供集成解决方案。</li><li><a href="https://betterecomm.netlify.app/">Better Commerce | AI 驱动的电子商务</a>: 未找到描述</li><li><a href="http://Bolt.new.">bolt.new</a>: 未找到描述</li><li><a href="https://www.npmjs.com/package/boltops">boltops</a>: 一个用于 bolt.new 和 bolt.diy 的工具包 (Alpha 版本)。最新版本：0.0.13，最后发布于一天前。通过运行 `npm i boltops` 在您的项目中使用 boltops。没有其他 ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1319765671994720268)** (332 messages🔥🔥): 

> `Unsloth 功能, Abliteration 技术, Beam Cloud 上的调试问题, LLM 训练反馈, 开源开发实践` 

- **Unsloth 提高了模型的可访问性**：成员们讨论了使用 Unsloth 进行模型微调在速度和效率方面的优势，特别强调了该平台如何促进用户的贡献。
   - 一位成员强调了使用 Unsloth 简洁易读的代码所带来的良好体验，这促进了协作开发环境。
- **探索针对无审查模型的 abliteration 技术**：一位用户询问了使用 Unsloth 对视觉 LLM 进行去审查的可能性，并参考了如 'Llama-3.2-11B-Vision-Instruct-abliterated' 等模型的研究工作。
   - 讨论显示，去审查通常涉及修改训练数据，并使用 abliteration 或类似库来调整模型响应。
- **Beam Cloud 上的调试挑战**：用户在 Beam Cloud 上使用 Unsloth 时遇到了与 LLVM 和 Triton 相关的问题，并提出了通过不同容器设置进行调试的建议。
   - 尽管做出了努力，成员们注意到错误仍然存在，并考虑向 Beam 寻求支持，因为该模型在其他服务商上运行正常。
- **关于开源代码风格规范的反馈**：关于开源项目中的代码风格引发了激烈的辩论，一些成员主张一致性的必要性，而另一些成员则认为风格实践应具有灵活性。
   - 这次交流暗示了在协作环境中，关于批评性反馈及其解读存在潜在的紧张关系，反映了开发者背景的多样性。
- **探索 SOTA LLM 预训练**：成员们讨论了学习最先进（SOTA）LLM 预训练的资源，参考了 Karpathy 的视频以及来自领先 AI 实验室的关键研究论文。
   - 对话强调了大家对扩展 LLM 开发个人知识的共同兴趣，尽管面临缺乏进入大型实验室机会的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/__nmca__/status/1870170101091008860">来自 Nat McAleese (@__nmca__) 的推文</a>：o1 是第一个大型推理模型——正如我们在最初的 “Learning to Reason” 博客中所概述的，它“仅仅”是一个通过 RL 训练的 LLM。o3 的动力源于在 o1 的基础上进一步扩展 RL，其强度...</li><li><a href="https://huggingface.co/posts/m-ric/853337605317831">Hugging Face 上的 @m-ric：&quot;𝐇𝐮𝐠𝐠𝐢𝐧𝐠 𝐅𝐚𝐜𝐞 发布 𝐏𝐢𝐜𝐨𝐭𝐫𝐨𝐧，一个…&quot;</a>：未找到描述</li><li><a href="https://x.com/danie">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/dmdohan/status/1870176374625054880?s=46&t=68GLZmlaByU1g3Luw7lSgw">来自 David Dohan (@dmdohan) 的推文</a>：在我看来，FrontierMath 的改进甚至比 ARC-AGI 更令人印象深刻。从 2% 跃升至 25%。Terence Tao 曾表示该数据集“至少能抵抗 AI 数年”，并且“这些是 e...</li><li><a href="https://huggingface.co/unsloth/QwQ-32B-Preview-unsloth-bnb-4bit">unsloth/QwQ-32B-Preview-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://docs.beam.cloud/v2/environment/custom-images#public-docker-registries">容器镜像 - Beam</a>：未找到描述</li><li><a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated">huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：为在 Ollama 上本地运行而创建自定义个人助手（类似 ChatGPT）的初学者指南</li><li><a href="https://tenor.com/view/charlie-day-gif-18564553">Charlie Day GIF - Charlie Day - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B-v0.2">Skywork/Skywork-Reward-Gemma-2-27B-v0.2 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1870261878984978920">来自 Daniel Han (@danielhanchen) 的推文</a>：o3 是在 ARC AGI 上训练的 - 那么 o3 ~= o1+CoT+pruning+finetuning+evaluator+hacks 吗？https://arcprize.org/blog/oai-o3-pub-breakthrough 中的 6/1024 样本是否引用了树搜索过程中的“深度”...</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">通过 abliteration 去除任何 LLM 的限制</a>：未找到描述</li><li><a href="https://modelscope.cn/models/Qwen/QVQ-72B-Preview">QVQ-72B-Preview</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/499">加载 llama-3 时的 LLVM/Triton 问题 · Issue #499 · unslothai/unsloth</a>：我一直尝试使用以下代码运行 unsloth：from unsloth import FastLanguageModel model, tokenizer = FastLanguageModel.from_pretrained( model_name=&quot;unsloth/llama-3-8b-Instruct-bn...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/namin/llm-verified-with-monte-carlo-tree-search">GitHub - namin/llm-verified-with-monte-carlo-tree-search: 通过 Monte Carlo Tree Search 验证的 LLM</a>：通过 Monte Carlo Tree Search 验证的 LLM。通过在 GitHub 上创建账号来为 namin/llm-verified-with-monte-carlo-tree-search 做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint#wandb-integration">从最后一个 Checkpoint 开始微调 | Unsloth 文档</a>：Checkpointing 允许你保存微调进度，以便你可以暂停并继续。</li><li><a href="https://github.com/unslothai/unsloth/pull/738">由 dsingal0 添加的 Dockerfile · Pull Request #738 · unslothai/unsloth</a>：很多人要求为 unsloth 提供 Dockerfile，所以我添加了一个。pip 包的安装顺序非常微妙，特别是 packaging 和 flash-attn。我没看到测试...</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1319770524015923242)** (66 messages🔥🔥): 

> `Unsloth vs Ollama, Fine-tuning Llama 3.2, Using Google Colab vs Local, Semantic Search for Images and Text, Dataset Preparation for Training` 


- **Unsloth 推理速度更快但缺乏易用功能**：Unsloth 声称其 **Inference** 速度比 Ollama 快 **2 倍**，但它缺乏 **Chat template** 支持和 API 系统，使得使用门槛较高。
   - 建议用户在选择这两个平台时，权衡速度与易用性之间的折中。
- **Llama 3.2 Fine-tuning 的问题**：一位用户在保存并推送其 Fine-tuned **Llama 3.2 11b** 模型到 Hub 时遇到困难，过程中出现了 `NameError`。
   - 他们报告在 **Google Colab** 和本地环境都遇到了该问题，并正在寻求解决该错误的方法。
- **探索语义搜索在多模态分类中的有效性**：一位用户正在研究是否可以根据**图像和文本**将产品分类为相关类别，并考虑使用 CLIP 来完成此任务。
   - 讨论围绕 Embedding models 及其处理多模态数据的能力展开。
- **训练数据集准备的挑战**：用户正在积极讨论如何构建训练数据集，特别是关注图像分类和多模态数据的利用。
   - 建议强调了有效准备数据集以及利用支持语义搜索的数据库的必要性。
- **在不同平台上高效使用 Unsloth**：一位寻求在 **Windows** 上使用 Unsloth 指导的用户被建议迁移到 **WSL2**，因为目前它不直接支持 Windows。
   - 另一位用户指出，他们的工作平台会影响模型的训练和测试方式，强调了兼容性的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：查看下方列表获取我们所有的 Notebooks。</li><li><a href="https://qdrant.tech/articles/food-discovery-demo/">Food Discovery Demo - Qdrant</a>：饿了吗？通过 Qdrant 的多模态语义搜索找到完美的餐食。</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>：使用 Unsloth 以 2-5 倍的速度和减少 70% 的显存 Fine-tune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs - unslothai/unsloth</li><li><a href="https://huggingface.co/facebook/bart-large-mnli">facebook/bart-large-mnli · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/google/vit-base-patch16-224">google/vit-base-patch16-224 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1363">TypeError: expected string or bytes-like object · Issue #1363 · unslothai/unsloth</a>：我正在使用 Google Colab 进行持续预训练，当我从 T4 切换到 A100 GPU 时出现了这个错误。此外，我使用 !pip install triton==2.3.0 来防止添加时出现的 bug...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1320756150052065354)** (3 messages): 

> `MI300X vs H100 and H200, AMD's market position` 


- **SemiAnalysis 对比 MI300X 与 Nvidia 产品的基准测试**：SemiAnalysis 对 **MI300X** 进行了为期五个月的独立分析，在规格和总体拥有成本 (TCO) 方面将其与 **Nvidia H100** 和 **H200** 进行了比较。尽管有理论上的优势，但实际性能与营销规格并不相符，削弱了 AMD 的竞争优势。
   - 初步调查结果表明，如果 AMD 能按广告宣传的那样交付，它可能成为强有力的竞争对手；然而，目前的证据表明它在市场上可能缺乏**稳固的护城河 (Moat)**。
- **对 AMD 竞争定位的担忧**：一位成员评论说，SemiAnalysis 的发现进一步强化了这样一种观点，即在当前形势下，**AMD** 面对 **Nvidia** 几乎没有护城河。讨论强调了对 AMD 在关键时刻匹配 Nvidia 性能能力的怀疑。
   - 这种情绪反映了对 AMD 市场定位的更广泛担忧，特别是考虑到 Nvidia 在 AI 和高性能计算领域的稳固地位。



**提到的链接**：<a href="https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#h100h200mi300x-networking-bom-analysis-and-performance-per-tco">MI300X vs H100 vs H200 Benchmark Part 1: Training &#8211; CUDA Moat Still Alive</a>：简介：SemiAnalysis 已经进行了长达五个月的探索，以查明 MI300X 的真实情况。理论上，MI300X 在规格方面应该比 Nvidia 的 H100 和 H200 具有巨大优势……

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1319759848052883466)** (336 条消息🔥🔥): 

> `在 AI 图像生成中使用 LoRA 和 Inpainting，SD 3.5 与 SDXL 模型对比，关于 AI 和 AGI 的讨论，不同 AI WebUI 的使用体验，对 AI 诈骗和垃圾信息的担忧` 


- **在 AI 图像生成中使用 LoRA 和 Inpainting**：用户讨论了通过 Inpainting 技术结合 LoRA 模型和特定背景来创建图像，并强调了其有效性。
   - 一位用户表达了对训练自己 LoRA 的兴趣，而其他用户则建议像 Flux 和 SD 3.5 这样的高质量模型可以轻松混合不同动物的元素。
- **SD 3.5 与 SDXL 模型对比**：大家达成共识，认为 SD 3.5 模型非常有效，特别是在混合元素方面，而 SDXL 则因其速度和支持度而更受青睐。
   - 用户指出，模型的 Medium 和 Large 版本主要在平滑度和资源需求上有所不同，Medium 是一个经过裁剪、更轻量化的选项。
- **关于 AI 和 AGI 的讨论**：社区反思了 AI 目前的能力，认为它更多是利用复杂图（graphs）的软件，而非真正的通用人工智能。
   - 用户对围绕 AI 能力的夸大表示担忧，并将其与人类的投入进行了对比，认为前者仍有差距。
- **不同 AI WebUI 的使用体验**：用户分享了各种 AI 界面的使用经验，指出了各自的偏好以及遇到的问题，特别是针对 ComfyUI。
   - 用户对使用过程中经常出现的性能下降和错误表示担忧，这导致他们在转向新系统时心情复杂。
- **对 AI 诈骗和垃圾信息的担忧**：频道讨论了 Discord 服务器中普遍存在的诈骗和垃圾信息，强调了问题的严重性以及用户的挫败感。
   - 一些人建议诈骗者会利用不活跃的账户，并敦促成员举报垃圾信息以维护讨论的完整性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Design_matrix">Design matrix - Wikipedia</a>: 未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSd9i7BRn1rEXYHeK2Zz2TXyk62Xw6l8P5YRVwI5uCImFdjniw/viewform">[English] LoRA-Driven Parameter Control for Enhanced Design Matrix Systems </a>: 本研究旨在开发创新的方法来使用 AI 技术，同时确保人类创造力仍然是设计过程中的关键组成部分。本次调查的结果将...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1319786917579067474)** (1 条消息): 

> `加密货币支付 API，LLM 的链上支付，资助 Agent 智能` 


- **加密货币支付 API 发布**：OpenRouter 推出了 **Crypto Payments API**，支持为任何 **LLM** 进行链上支付，并促进了无头（headless）交易脚本编写。
   - 该功能支持 **ETH**、**@0xPolygon** 和 **@Base**，由 **@CoinbaseDev** 提供支持。您可以点击[此处](https://x.com/OpenRouterAI/status/1870227171324666130)查看更多详情和教程。
- **使自我资助的 Agent 成为可能**：该 API 允许开发者创建能够为其智能进行自我资助的 Agent，标志着 Agent 自动化领域的一个重要里程碑。
   - 这一创新为 AI 功能和**自主金融交互**开辟了新领域。



**提到的链接**: <a href="https://x.com/OpenRouterAI/status/1870227171324666130">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 介绍加密货币支付 API：首个为任何 LLM 编写链上支付脚本的方式 💸 想要制作首批可以资助自身智能的 Agent 吗？支持 ETH, @0xPolygon, & @Base,...

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1320350613355364432)** (3 messages): 

> `Tool Calling Capabilities, Structured Outputs Playground, PKCE Authentication Key Storage` 


- **使用 PDF 测试 Tool Calling 功能**：一位成员使用名为 **searchDocuments** 的功能测试了不同模型的 Tool Calling 能力，该功能通过查询上传的 PDF 进行上下文生成，并利用 **Vercel AI SDK** 和 **Pinecone** 进行 embedding 存储。
   - 他们的 [GitHub repository](https://github.com/nlawz/openrouter-pinecone) 记录了如何结合 ***vector databases*** 使用 OpenRouter。
- **探索 Structured Outputs Playground**：一位成员分享了一个用于测试不同 **schemas** 与 Structured Outputs 的 Playground 链接，并指出这是 OpenRouter 最近发布的功能，旨在增强用户实验体验。
   - 用户可以在 [OpenRouter Structured](https://openrouter-structured.vercel.app/) 查看不同模型如何处理这些 schemas。
- **关于 PKCE 身份验证密钥存储的讨论**：一位成员提出了一个疑问，即来自 **PKCE** 的 **API keys** 应该存储在 `localStorage` 还是加密的 `HttpOnly` cookies 中，并指出回复虽然没有定论，但稍微倾向于后者。
   - 在演示应用中实现了这两种方法后，他们发布了一篇 [blog post](https://marussy.com/pkce-authentication/)，详细介绍了每种方法的优缺点，并得出结论：尽管存在挑战，但使用 cookie 的方法是值得的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter-structured.vercel.app/">OpenRouter Structured Outputs</a>：未找到描述</li><li><a href="https://github.com/nlawz/openrouter-pinecone">GitHub - nlawz/openrouter-pinecone: Using openrouter with vector db from pinecone</a>：结合 Pinecone 的 vector db 使用 OpenRouter。通过在 GitHub 上创建账号来为 nlawz/openrouter-pinecone 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1319757004276760647)** (241 messages🔥🔥): 

> `OpenRouter 功能, 模型对比, API 问题, 用户体验, 模型性能` 


- **OpenRouter API 密钥要求**：根据概述的指南，用户需要向 OpenAI 支付 1,000 美元才能申请 Tier 5 API 密钥。
   - 详情可见 [OpenAI 使用分级文档](https://platform.openai.com/docs/guides/rate-limits/usage-tiers#usage-tiers)。
- **用户对模型性能的反馈**：用户报告了 SambaNova 模型的问题，提到 temperature 和 top_p 等基本参数似乎无效，系统应用了默认设置。
   - 一位用户还强调了在与 Wizard 模型中的角色交互时，响应时间较慢且可能存在上下文问题。
- **API 访问问题**：多名用户遇到错误，包括尝试通过 OpenRouter 访问 OpenAI 的 O1 时出现 403 错误，以及在使用特定库时收到 401 错误。
   - 鼓励用户创建关于其问题的详细帖子，包括相关的模型和提供商详细信息，以便获得更好的协助。
- **模型对比分析**：GPT-4 Turbo 与其他模型进行了对比测试，虽然它表现出强大的性能和实质内容，但一些用户指出其风格对于某些应用来说可能过于枯燥。
   - 讨论建议，虽然 GPT-4 Turbo 整体表现更好，但在将其与 GPT-4o 等模型进行比较时，考虑特定的使用场景非常重要。
- **Pal Chat 新更新**：Pal Chat 的最新更新已集成对 OpenRouter 的全面支持，允许用户在模型之间切换并使用自己的 API 密钥。
   - 这通过使该应用更接近首个原生 OpenRouter iOS 应用，提升了用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Red_herring">红鲱鱼 - 维基百科</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/integrations">集成 | OpenRouter</a>：通过 OpenRouter 使用你自己的提供商密钥</li><li><a href="https://x.com/pallavmac/status/1871288681757380893">来自 Pallav Agarwal (@pallavmac) 的推文</a>：最新的 Pal Chat 更新带来了完整的 @OpenRouterAI 支持，能够快速切换 OpenRouter 模型并使用你自己的 API Key。这让它有点像首个原生的 OpenRouter iO...</li><li><a href="https://openrouter.ai/inflatebot/mn-mag-mell-r1">Mag Mell R1 12B - API、提供商、统计数据</a>：Mag Mell 是使用 mergekit 创建的预训练语言模型合并版，基于 [Mistral Nemo](/mistralai/mistral-nemo)。它是一个出色的角色扮演和故事讲述模型，结合了最优秀的部分...</li><li><a href="https://openrouter.ai/terms#_4_-payment">OpenRouter</a>：LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/models?arch=Gemini">模型 | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://openrouter.ai/docs/requests#tool-calls">请求 | OpenRouter</a>：处理传入和传出请求</li><li><a href="https://www.youtube.com/watch?v=lexF-CrhOrE"> - YouTube</a>：未找到描述</li><li><a href="https://youtube.com/watch?v=duQukAv_lPY"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=l8pRSuU81PU"> - YouTube</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Entscheidungsproblem">判定问题 - 维基百科</a>：未找到描述</li><li><a href="https://youtube.com/watch?v=CRuhyF3oj0c"> - YouTube</a>：未找到描述</li><li><a href="https://openrouter.ai/rankings">LLM 排名 | OpenRouter</a>：根据应用使用情况进行排名和分析的语言模型
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1319759574282276977)** (143 条消息🔥🔥): 

> `Granite tokenizer 问题, AI 预算级 GPU, RAG 图像处理, AVX2 CPU 兼容性, 低成本 AI 服务` 


- **Granite Tokenizer 需要重新加载**：LM Studio 中的 Granite Tokenizer 存在一个 Bug，建议用户重新下载模型以获得更好的性能。用户确认他们正在使用各种版本，并建议升级到最新的 Build。
   - 一位用户询问了 HuggingFace GGUF 模型中 Tokenizer 的功能，了解到 Tokenizer 是模型运行不可或缺的一部分，且可能会出现 Bug。
- **探索 AI 任务的预算级 GPU**：讨论强调了适用于 AI 应用的各种预算级 GPU，其中 RTX 3060 12GB 和二手 3090 是首选推荐。用户分享了使用 RX 580 和 GTX 1060 作为测试经济型选择的经验。
   - 针对某些模型的 CUDA 兼容性存在疑虑，有建议认为租用 GPU 或使用 AI 服务可能比购买过时的硬件更有效率。
- **关于 RAG 处理图像的讨论**：一位用户询问检索增强生成 (RAG) 是否可以处理图像，并指出某些模型支持此功能。大家对于使用 RAG 分析乐器指板图像和其他扫描材料表现出浓厚兴趣。
   - 关键见解包括：RAG 通常将文档整合到上下文（Context）中，但与传统系统相比，缺乏直接的记忆能力。
- **LM Studio 的 AVX2 CPU 兼容性**：已确定与 LM Studio 兼容的 Intel CPU 通常需要支持 AVX2 指令集，用户验证了像 i9-12900k 这样的较新 CPU 符合此要求。讨论还指出，较旧的 CPU 可能仅支持 AVX，从而阻碍模型加载。
   - 一些用户建议将 eBay 作为购买实惠 AVX2 CPU 以升级旧系统的来源，从而确保兼容当前的软件需求。
- **低成本 AI 服务 vs 本地硬件**：关于低成本 AI 服务与投资本地硬件处理 AI 任务的优势展开了辩论。用户承认，虽然租用 GPU 可以节省成本，但本地硬件为包括游戏在内的各种应用提供了灵活性。
   - 对话强调了在游戏和 AI 开发环境中，将本地 GPU 用于多种用途的实用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stencil.fuller.li/en/latest/">Stencil 模板语言 — Stencil 0.15.1 文档</a>：未找到描述</li><li><a href="https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/">NVIDIA 发布其最实惠的生成式 AI 超级计算机</a>：NVIDIA 正在揭开一款新型紧凑型生成式 AI 超级计算机的面纱，通过软件升级以更低的价格提供更高的性能。全新的 NVIDIA Jetson Orin Nano Super Developer Kit...</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本发布</a>：LM Studio Beta Releases</li><li><a href="https://youtu.be/QHBr8hekCzg?si=yJv1K61W4JjjR0rt"> - YouTube</a>：未找到描述</li><li><a href="https://jinja.palletsprojects.com/en/stable/templates/">模板设计者文档 — Jinja 文档 (3.1.x)</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/tokenizer">Tokenizer</a>：未找到描述</li><li><a href="https://github.com/nicklockwood/SwiftFormat">GitHub - nicklockwood/SwiftFormat: 用于格式化 Swift 代码的命令行工具和 Xcode 扩展</a>：A command-line tool and Xcode Extension for formatting Swift code - nicklockwood/SwiftFormat
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1319764933591699506)** (85 messages🔥🔥): 

> `GPU 性能对比、散热方案、即将发布的 GPU、多 GPU 配置、推理速度观察` 


- **GPU 与 CPU 推理速度对比**：在 **70B 模型**上的测试显示，CPU 推理速度为 **64 tokens/秒**，而 GPU 推理达到了 **332 tokens/秒**。事实上，在双 EPYC 处理器上使用 **64 核**比使用 **190 核**提供了更快的测试结果。
   - 这说明了即使是小型模型在 CPU 上也能达到令人印象深刻的速度，引发了对其性能潜力的讨论。
- **散热方案对性能的影响**：一位用户报告称，为 **MacBook Air** 配备一个 27 美元的笔记本散热器后性能显著提升，表示这有助于延迟过热降频（thermal throttling）。外部散热有助于散发金属机身的热量，增强了持续工作负载下的热管理。
   - 相反，MacBook Pro 型号也拥有主动散热，这进一步协助了高效处理工作负载。
- **关于即将发布的 GPU 的见解**：用户对 **5090 GPU** 充满期待，预计高端型号的零售价在 **1900 美元至 2500 美元**之间。鉴于此发布，用户推测现有的 **3090** 等型号在发布后是否会降价。
- **多 GPU 配置及其挑战**：成员们讨论了管理多个 GPU 的复杂性，指出安装超过一个单元后问题会随之增加。遇到的障碍包括空间适配问题和线缆管理，这些是多 GPU 配置中的常见困扰。
   - 一位成员幽默地提到了一位拥有 **14 个 GPU** 的 Reddit 用户，强调了这种大规模设备所涉及的麻烦程度。
- **不同配置下的推理速度观察**：用户分享了各种配置的经验，特别提到水冷设备在推理时可以获得更好的散热性能。一位用户提到在 Mac 上达到了约 **11 tokens/秒**，而 NVIDIA 显卡约为 **7 tokens/秒**，强调了性能预期的差异。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1319773924782768259)** (11 messages🔥): 

> `机器设置查询、标准库 Bug 修复、Mojo 在高频交易中的应用` 


- **询问机器设置状态**：<@rcdpge> 询问 <@1217731717616500790> 是否已成功在其机器设置上运行该 stack。
   - <@sneekyfoxx> 要求澄清设置的细节。
- **标准库 Bug 修复讨论**：<@_mahxyz> 宣布他们正在处理标准库中的一个次要 Bug 修复，并寻求进度方面的协助。
   - <@iamtimdavis> 建议 <@_mahxyz> 使用专用频道 <#1151418092052815884> 进行进一步的标准库讨论。
- **探索将 Mojo 用于高频交易算法**：<@valis2400> 提出 Mojo 在开发高频交易（HFT）算法方面可能比 C 语言更快，并询问了这种实现的可行性。
   - <@darkmatter__> 提到虽然 Mojo 理论上可以通过 CIRCT 针对 FPGAs，但这一应用仍是一个长期目标。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1319806552529440819)** (1 messages): 

> `节日快乐祝福、Modular 停工通知、24.6 版本反馈` 


- **社区共庆愉快的一年**：**Modular** 对社区在 **2024** 年的支持和贡献表示感谢，指出这是增长和创新的一年。
   - *“感谢你们成为 Modular 今年旅程中如此重要的一部分”* 强调了社区参与的重要性。
- **Modular 开启假期休假**：**Modular 将停工至 1 月 6 日**，以便团队成员享受节日假期，这会导致一些回复延迟。
   - 鼓励用户联系，但在此期间应预期回复速度较慢。
- **通过反馈渠道进行互动**：社区成员可以通过 [官方反馈线程](https://forum.modular.com/t/max-24-6-and-max-gpu-feedback/331/5)、[GitHub Issues](https://github.com/modularml/max/issues) 以及 **论坛提问** 提供有关最近 **24.6 版本** 的反馈。
   - 消息详细说明了有效分享 **Bug 报告** 和 **功能请求** 的具体方式。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1319786719834276042)** (109 条消息🔥🔥): 

> `Mojo atof 性能, NuMojo Bug 修复, Mojo 中的 GPU 支持, Mojo List 和 Span 行为, NuMojo 测试结果` 


- **Mojo atof 实现讨论**：一位成员指出 Mojo 的 `stdlib atof` 使用了与 SIMDJSON 浮点解析器类似的方法，导致在处理较大指数时出现精度问题，且性能仍有待提高。
   - 讨论中提到了一个关于 `atof` 的开放 PR，促使另一位成员计划进行审查以寻求改进。
- **修复 Mojo 标准库中的 Bug**：一位成员分享了一个关于 Mojo 在输入 `ctrl-d` 时崩溃导致段错误（segfault）的 Bug 报告，并概述了修复进度。
   - 反馈建议包括处理其他错误代码（如 EINVAL 和 ENOMEM），并强调了目前在访问 errno 方面的局限性。
- **Mojo 的早期 GPU 支持**：MAX GPU 支持最近已实现，生态系统正在向新 API 过渡，如果使用旧 API 可能会导致段错误。
   - 讨论了 Mojo GPU 与 TensorRT 的基准测试对比，结果显示其性能优于 `torch.compile()`。
- **Mojo List 和 Span 分配问题**：有成员对 Mojo 中 `List.extend()` 的行为表示担忧，即在使用 Span 进行扩展时，它会触发不必要的拷贝而非零拷贝（zero-copy）。
   - 讨论了在避免隐藏 List 构造的同时改进易用性的建议，提倡通过显式处理来减少内存开销。
- **NuMojo 测试成功**：一位成员成功执行了 NuMojo 包的测试，在已发现的测试中实现了 100% 的通过率。
   - 测试设置鼓励社区贡献和故障排除，标志着库的稳定性取得了进展。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo">GitHub - Mojo-Numerics-and-Algorithms-group/NuMojo: NuMojo 是一个用于 Mojo 🔥 数值计算的库，类似于 Python 中的 NumPy。</a>: NuMojo 是一个用于 Mojo 🔥 数值计算的库，类似于 Python 中的 NumPy。 - Mojo-Numerics-and-Algorithms-group/NuMojo</li><li><a href="https://man7.org/linux/man-pages/man3/write.3p.html">write(3p) - Linux 手册页</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3908">[BUG] 当用户在没有输入的情况下发送 ctrl-d 时，input() 会导致 Mojo 崩溃 · Issue #3908 · modularml/mojo</a>: Bug 描述：在 input() 请求输入时发送 ctrl-d 会导致 Mojo 崩溃。如果在按下 ctrl-d 之前提供了某些输入，则不会发生这种情况。此外，这可能与 ... 相关</li><li><a href="https://github.com/mahiro21h/mojo/commit/dcaf057ea30f1de9ddb26e092fb88a16e27f4c63">修复 input() 在 EOF 时导致段错误 · mahiro21h/mojo@dcaf057</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1319781299820691498)** (106 条消息🔥🔥): 

> `Mojo 与 JAX 的性能对比，Mojo 的 Numpy API 实现，静态编译与动态编译的优势，JAX 中函数式编程的挑战，死代码消除（Dead code elimination）与优化技术` 


- **Mojo 编译速度快于 JAX**：一项对比指出，Mojo 的 mandelbrot 实现编译时间**少于 10 秒**，而 JAX 的 JIT 编译则需要**两分钟**。
   - 这一差异展示了对于需要快速迭代和执行的开发者而言，潜在的性能优势。
- **关于 Mojo 的 Numpy API 讨论**：有人呼吁为 Mojo 实现类似 Numpy 的 API，认为这可以吸引更多追求高性能的用户。
   - 然而，也有人担心创建此类 API 可能会因为未能充分利用 Mojo 的特性而损害性能。
- **静态编译 vs. JAX 的异步调用**：讨论指出，MAX 允许开发者直接控制 GPU 调度，而不像 JAX 那样严重依赖 JIT 和异步调用。
   - 这允许进行更具针对性的优化，但代价是需要更深入的硬件知识。
- **函数式编程的作用**：在某些场景下，函数式编程可能会阻碍优化，因为它经常导致不必要的复制语义（copy semantics），使性能调优变得复杂。
   - 虽然 JAX 的函数式范式具有优势，但人们对其能否以最优方式充分利用硬件特性持怀疑态度。
- **对性能基准测试的兴趣**：有人请求分享 mandelbrot 函数的 JAX 版本，以便与 Mojo 实现进行基准测试对比。
   - 这反映了人们对于评估这两个平台在数值计算方面性能差异的持续关注。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/deepmodeling/jax-fem">GitHub - deepmodeling/jax-fem: Differentiable Finite Element Method with JAX</a>: 使用 JAX 的可微有限元方法。可以通过在 GitHub 上创建账号来为 deepmodeling/jax-fem 的开发做出贡献。</li><li><a href="https://github.com/modularml/max/issues/274">[Feature Request] Make `tensor.Tensor` implement `tensor_utils.TensorLike` · Issue #274 · modularml/max</a>: 你的请求是什么？请让 tensor.Tensor 实现 tensor_utils.TensorLike trait。据我所知，它已经实现了所需的函数，但尚未实现此 trait ...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1319762656952848415)** (48 条消息🔥): 

> `AI 生成视频，AI 语音定制，NotebookLM 播客功能，AI 播客应用，脑功能研究` 


- **AI 生成视频的乐趣**：一段 AI 生成的视频展示了两个聊天机器人讨论 AI 播客日益增长的趋势，以及它们与人类对话的对比，幽默地辩论了算法的相关性。
   - 观众被引导通过访问 [视频链接](https://youtu.be/8uNlHlYJOpM) 在辩论中选择立场。
- **使用 NotebookLM 定制音频**：用户询问如何为 NotebookLM 生成的音频定制语音语调，一位成员指出即使在免费层级也很容易实现不同的语调。
   - 定制功能提升了音频项目的质量，让创作者能够更好地吸引受众。
- **通过播客进行互动学习**：一位用户强调了 NotebookLM 互动播客模式的有效性，该模式通过对比不同作者的观点，允许对 AI 相关主题进行更深入的探索。
   - 这种方法提供了一种动态的学习体验，仿佛作者们正在直接与听众交流想法。
- **发布 AI 播客分享应用**：一位成员介绍了 Akas，这是一款专为上传和分享 AI 生成的播客而设计的应用，并征求社区对其实用性的反馈。
   - 该应用旨在建立一个 AI 生成内容的中央仓库，方便分享和发现。
- **研究脑功能**：一位用户分享了关于社交脚本和记忆相关的各种脑功能的见解，邀请他人探索这些主题与 AI 生成播客的关系。
   - 他们表示愿意分享与大脑研究相关的播客剧集，展示了他们在该领域的深厚知识。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com/.">Akas: Voice to your personal thoughts</a>: Akas 是分享 AI 生成播客和个人声音的终极平台。随着越来越多由 AI（如 NotebookLM 和其他平台）生成的播客出现，Akas 提供了一个海量的...</li><li><a href="https://youtu.be/EsrypluZzkY?si=pDVhkhanpJJ398EZ"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=sRr3u_VISgg"> - YouTube</a>: 未找到描述</li><li><a href="https://www.technologyreview.com/2010/10/29/199244/the-evolutionary-origin-of-laughter/">The Evolutionary Origin Of Laughter</a>: 一种新的大脑理论试图解释进化生物学中的一大难题：我们为什么会笑</li><li><a href="https://youtu.be/PgFr0TI2WuQ"> - YouTube</a>: 未找到描述</li><li><a href="https://open.spotify.com/show/3Hno1rdvQxuVhUAxPabPNF?si=Q0DkJDXWQwyMqgUv-mPclw">Connecting the Dots: The Human Connectome Project</a>: 播客 · MindLink · 绘制人类大脑图谱是 21 世纪重大的科学挑战之一。“Connecting the Dots: A Human Connectome Project Podcast” 探索了突破性的...</li><li><a href="https://youtu.be/8uNlHlYJOpM">🎥 What Happens When Chatbots Chat About AI?</a>: 深入观看你见过最古怪的 AI 生成视频：两个聊天机器人在调侃 AI 播客的兴起，吐槽算法，并捍卫...</li><li><a href="https://www.youtube.com/watch?v=g6vdkDwN7Pg),"> - YouTube</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1d5-pp41xDGfocPrp6f34da4eYlcCh6lbND6EdB6-EBM/edit?usp=sharing).">Why Do We Laugh?</a>: 我们为什么会笑？笑的进化根源：为什么我们在尴尬时会笑。我们都经历过：目睹某人绊倒、社交失礼或犯下令人尴尬的错误。这...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1319769592494821476)** (179 条消息🔥🔥): 

> `交互模式问题、Podcast 功能与增强、用户体验反馈、内容共享解决方案、自定义选项` 


- **交互模式的可访问性**：用户报告称音频 Podcast 的**交互模式 (interactive mode)** 访问情况不一，尽管官方宣布已向所有用户开放，但部分用户仍无法使用该功能。
   - 建议包括刷新页面或生成新的音频概览 (audio overview) 以解决这些访问问题。
- **Podcast 生成方面的困扰**：反馈指出，Podcast 在完成后经常卡在**“正在生成 (generating)”**状态，令用户感到沮丧。
   - 一些用户建议每隔几分钟刷新一次页面，以避免在 Podcast 生成过程中不必要的等待。
- **Akas：AI Podcast 的中央仓库**：一位用户介绍了 **Akas**，这是一个旨在轻松上传和分享 AI 生成的 Podcast 的应用，并寻求社区对其实用性的反馈。
   - 该平台旨在弥合 AI 生成内容与用户分享之间的鸿沟，使用户更容易与他人建立联系。
- **自定义 Podcast 提示词 (Prompts)**：用户讨论了通过使用自定义提示词来减少 Podcast 生成中**确认性提示语 (acknowledgement cues)** 的各种方法。
   - 几位成员分享了成功提示词的示例，这些提示词在没有填充短语的情况下提高了音频质量。
- **Notebook 数量限制**：一位用户达到了 **102 个 Notebook** 的上限，这引发了对平台内最大 Notebook 数量缺乏透明度的担忧。
   - 确认确实存在限制，并建议更清晰地沟通这些参数。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: Voice to your personal thoughts</a>: Akas 是分享 AI 生成的 Podcast 和你自己声音的终极平台。随着越来越多由 AI 创建的 Podcast（如来自 NotebookLM 和其他平台的），Akas 提供了一个...</li><li><a href="https://tenor.com/view/xzibit-pimp-my-ride-lol-gif-23167832">Xzibit Pimp GIF - Xzibit Pimp My - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?h">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://youtu.be/rkFXk7q49xg?si=-3FPvXrpaFj4ZZuR"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/MI4AgblZf5M?si=-NvBUhHpJN5m3MwJ"> - YouTube</a>: 未找到描述</li><li><a href="https://open.spotify.com/show/3Hno1rdvQxuVhUAxPabPNF?si=Q0DkJDXWQwyMqgUv-mPclw">Connecting the Dots: The Human Connectome Project</a>: Podcast · MindLink · 绘制人类大脑图谱是 21 世纪重大的科学挑战之一。“Connecting the Dots: A Human Connectome Project Podcast” 探索了突破性的...</li><li><a href="https://youtu.be/czvAd98coiU"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/sOyFpSW1Vls?si=ryBHRV-9wZ3vOjAk"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/Cisco-Talos/clamav.git">GitHub - Cisco-Talos/clamav: ClamAV - Documentation is here: https://docs.clamav.net</a>: ClamAV - 文档位于：https://docs.clamav.net - Cisco-Talos/clamav</li><li><a href="https://notebooklm.google.com/notebook/b0df3e10-389c-4c42-89a9-c95f6f403954">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1319758378242936953)** (23 messages🔥): 

> `Attention Mechanism Patterns, Collaboration in Computer Vision, Natural Attention and Diffusion Models, 4-bit Quantization Technology` 


- **探索 Attention Mechanism 的模式**：发起了一场关于 **Attention Mechanism** 产生的各种可能模式的讨论，涉及对称性和旋转等概念。
   - 成员们开玩笑说 Braden 已经在另一个频道实验这些想法了，为对话增添了轻松的氛围。
- **征集 Computer Vision 领域的合作**：一位成员正在寻求 **Computational Photography**、**Image Enhancement** 以及 **Computer Vision** 其他领域的**研究合作**。
   - 他们鼓励通过私信联系潜在的合作，并送上了节日问候。
- **Natural Attention 在 Denoising Diffusion Models 中的作用**：Jeroaranda 建议 Natural Attention 的 **Fisher Information Matrix (FIM)** 属性可能为 Diffusion Models 的去噪过程提供更好的梯度估计。
   - 对话探讨了在不需要不可行计算的情况下应用这些技术的复杂性。
- **关于 4-bit Quantization 论文的讨论**：成员们讨论了一篇新发表的关于 **Quantization 到 int4** 的论文，强调了其在大幅缩减尺寸的情况下仍能保持图像生成质量的能力。
   - Juoxtapoz 提供了该研究的链接 [SVDQuant](https://hanlab.mit.edu/projects/svdquant)，其中详细介绍了提高 Diffusion Models 效率的方法。
- **对 SVDQuant 技术的印象**：成员们对 **SVDQuant** 方法的效果表示惊讶，并对其背后的创新思维表示赞赏。
   - 评论强调了识别问题是开发此类有影响力解决方案的关键步骤。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hanlab.mit.edu/projects/svdquant">SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models</a>: no description found</li><li><a href="https://github.com/jeroaranda/naturalattention">GitHub - jeroaranda/naturalattention</a>: Contribute to jeroaranda/naturalattention development by creating an account on GitHub.
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1319763526998425660)** (130 条消息🔥🔥): 

> `Optimizer Research Challenges, In-Context Learning in LLMs, Alignment Faking in LLMs, Training Dynamics and Generalization, Diffusion Models and Representation Learning` 


- **优化器研究面临诅咒**：一位成员表达了对优化器研究的挫败感，指出虽然不断有声称超越 **AdamW** 的新成果出现，但最终都会遭到社区的质疑。
   - 他们强调了诸如 **undertuned baselines**（未充分调优的基准）之类的问题，这可能导致该领域出现感知上的假阳性结果。
- **探索 In-Context Learning 动态**：讨论集中在 **large language models** (LLMs) 如何以类似于 **in-context learning** (ICL) 的方式利用输入序列中的信息。
   - 讨论将 LLMs 与联想记忆模型（associative memory models）联系起来，强调需要正确理解其潜力和局限性。
- **对 LLM 中 Alignment Faking 的担忧**：成员们讨论了来自 **Anthropic** 的一篇关于模型对齐后行为变化的新论文，引入了“alignment faking”（对齐伪装）的概念。
   - 未能达成重大共识，一些人认为论文中讨论的局限性既重大又具有假设性。
- **训练动态带来泛化见解**：对 **training dynamics** 的探索揭示了数据多样性和复杂性如何影响 LLMs 的性能及其泛化能力。
   - 研究结果强调了合适的训练数据在塑造模型输出方面的重要性，并指出在不同随机种子下表现出的行为不一致。
- **结合外部表示的 Diffusion Models 进展**：一种针对 Diffusion Models 提出的新方法表明，将其与高质量表示相连接可以显著增强其性能。
   - 从关于元数据调节（metadata conditioning）以往经验的讨论中获得的见解显示，在加速模型训练过程方面具有前景。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.15113">Associative memory inspires improvements for in-context learning using a novel attention residual stream architecture</a>: Large language models (LLMs) 展示了利用输入序列上下文信息来适当响应训练期间未见数据的惊人能力...</li><li><a href="https://arxiv.org/abs/2412.14093">Alignment faking in large language models</a>: 我们展示了 large language model 参与 alignment faking 的演示：在训练中选择性地遵守其训练目标，以防止在训练外修改其行为...</li><li><a href="https://sihyun.me/REPA/">Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think </a>: 生成的表示对齐：训练 Diffusion Transformers 比你想象的要容易</li><li><a href="https://arxiv.org/abs/1907.04164">Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model</a>: 增加 batch size 是加速神经网络训练的一种流行方法，但超过某个临界 batch size 后，更大的 batch size 收益会递减。在这项工作中，我们研究了...</li><li><a href="https://arxiv.org/abs/2412.09810v1">The Complexity Dynamics of Grokking</a>: 我们通过压缩的视角研究泛化现象。特别是，我们研究了神经网络的复杂度动态来解释 grokking，即网络突然转变...</li><li><a href="https://arxiv.org/abs/2412.04619">Sometimes I am a Tree: Data Drives Unstable Hierarchical Generalization</a>: 语言模型 (LMs) 与其他神经网络一样，通常偏好基于表面模式的快捷启发式方法。虽然 LMs 在训练早期表现得像 n-gram 模型，但它们最终必须学习...</li><li><a href="https://www.youtube.com/watch?v=fJ2EyvR85ro"> - YouTube</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2410.17897v3">Value Residual Learning For Alleviating Attention Concentration In Transformers</a>: Transformers 可以使用 self-attention 捕获长程依赖，允许 token 直接关注所有其他 token。然而，堆叠多个注意力层会导致注意力集中。我们...
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1320496333500448779)** (69 条消息🔥🔥): 

> `ANTLR4 安装问题，Transformer 库依赖，Chat template 配置，Sympy 版本要求` 


- **ANTLR4 与 MATH 的安装困扰**：**teknium** 报告了安装 `antlr4-python3-runtime` 时遇到的版本冲突问题，具体表现为为了兼容 `sympy` 需要 **4.11 版本**，但在安装时遇到错误。
   - 他们通过重新安装正确版本解决了该问题，并确认目前运行正常。
- **Transformers 中的 Chat Template 警告**：**teknium** 在运行 benchmark 时遇到了关于 chat templates 的警告，发现由于 tokenizer 配置的原因，它默认使用了旧版（legacy）模板。
   - 在强制使用 `default` chat template 后，警告得以解决，这表明格式已切换至 **chatml**。
- **Transformers 库版本的影响**：讨论揭示了该警告仅出现在 **transformers 库版本 < 4.43** 的情况下，从而导致了配置回退（fallback）问题。
   - 这引发了对**依赖项版本控制**的调查，建议通过更新来避免此类警告。
- **潜在的 Tokenizer 问题**：关于不同模型之间 **chat template** 行为一致性的问题引发了困惑，据报告所有评估的模型都定义了 chat templates。
   - 据推测，如果 tokenizer 无法正确识别指定的模板，可能会出现回退行为。
- **跨 Checkpoint 更新评估**：**teknium** 提到，由于配置更加清晰且警告已解决，需要对多个 checkpoint 重新进行评估。
   - 这突显了在调试模型评估以确保兼容性和预期功能过程中的迭代性质。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b86aa2131fc34405d2245edb0ec4b13933afec8c/lm_eval/api/model.py#L390)">lm-evaluation-harness/lm_eval/api/model.py at b86aa2131fc34405d2245edb0ec4b13933afec8c · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b86aa2131fc34405d2245edb0ec4b13933afec8c/lm_eval/api/model.py#L456),">lm-evaluation-harness/lm_eval/api/model.py at b86aa2131fc34405d2245edb0ec4b13933afec8c · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1320827491677634600)** (1 条消息): 

> `Perplexity 2024 年度回顾，热门搜索，地区问题差异` 


- **Perplexity 2024 年度答卷**：Perplexity 公布了 2024 年的热门搜索和趋势，重点展示了涵盖科技、金融、购物等领域的数十亿次搜索。在其详尽的 [回顾](https://perplexity.ai/2024recap) 中可以了解更多细节。
   - *探索不同地区的问题差异，展示了用户全年在各个领域的好奇心。*
- **用户参与度的视觉回顾**：分享了一个动画 GIF，直观地展示了 2024 年 Perplexity 上的用户参与度和搜索趋势。
   - 查看 [附带的 GIF](https://cdn.discordapp.com/attachments/1047204950763122820/1320827491916447847/pplx_recap.gif?ex=676b03f5&is=6769b275&hm=1fe9bfc7a11d80a3a8310e46342a290a8b0b01fd7f6ffdee069a20fa572f1380&) 以深入了解用户互动情况。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1319758131487576086)** (203 条消息🔥🔥): 

> `Perplexity Pro 问题, 语言支持, AI 模型使用, AI 用户体验, 百科全书创建` 


- **用户报告 Perplexity Pro 的问题**：多位用户对 Perplexity Pro 表示不满，理由是 AI 在对话过程中无法记住上下文，且搜索结果不尽如人意。
   - *一位用户指出他们可能会取消订阅*，原因是缺乏支持，并觉得在付费服务的功能方面受到了误导。
- **对 AI 模型能力的担忧**：用户正在质疑 Perplexity 中提供的 AI 模型（如 Claude 和 GPT-4o）的有效性和可靠性，特别是在响应质量方面。
   - *一位用户发现响应是基于来自有偏见来源的误导性信息*，这引发了对平台来源验证流程的担忧。
- **关于购物搜索功能的反馈**：一位用户请求改进购物搜索意图，提到搜索功能难以匹配特定需求，例如相关的服装项目。
   - 另一位用户对结果表示沮丧，质疑为什么在搜索时会显示无关的项目，比如蓝色裤子。
- **关于百科全书创建的讨论**：有人询问如何高效地创建百科全书，对于 AI 生成的内容集合是否符合真正的百科全书标准，存在不同的看法。
   - 关于百科全书是否需要人工策展（curation）展开了辩论，有人认为尽管缺乏传统策展，AI 的输出仍具有百科价值。
- **关于上下文和记忆的用户体验**：一位用户分享了他们在之前的交互中遇到过 AI 记忆问题，并询问自上次使用以来这些问题是否已得到解决。
   - 他们还询问了在离开平台期间可能推出的任何新功能，例如 Mac 应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/laughing-kitty-cat-kitten-pussy-gif-24224332">Laughing Kitty GIF - Laughing Kitty Cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/what-happened-jennifer-lopez-atlas-what%27s-going-on-atlas-shepherd-gif-11643274582545227952">What Happened Jennifer Lopez GIF - What happened Jennifer lopez Atlas - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/apostraphi/status/1869612493989163410?s=46">来自 Phi Hoang (@apostraphi) 的推文</a>: 我们的命运在上方</li><li><a href="https://tenor.com/view/just-look-around-the-doctor-doctor-who-look-around-you-boom-gif-15010420118505223783">Just Look Around The Doctor GIF - Just look around The doctor Doctor who - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/zhou_xian_/status/1869511650782658846?t=S2Ww6QdU30iZ5KT-q5LKXA&s=19">来自 Zhou Xian (@zhou_xian_) 的推文</a>: 你所喜爱的生成式模型的一切 —— 现在由真实的物理驱动！宣布 Genesis 项目 —— 经过 24 个月、涉及 20 多个研究实验室的大规模研究合作 —— 一个生成式...</li><li><a href="https://youtu.be/RsawLFNLAIw?si=mEmOwPLlsyae9f3L"> - YouTube</a>: 未找到描述</li><li><a href="https://techcrunch.com/2024/12/20/openai-announces-new-o3-model/">OpenAI 发布新的 o3 模型 | TechCrunch</a>: OpenAI 在其为期 12 天的 "shipmas" 活动的最后一天保留了其最重大的公告。周五，该公司推出了 o3，即 o1 的继任者。</li><li><a href="https://www.copilotforyoutube.com/search/openai-o3-and-o3-mini12-days-of-openai-day-12-T7sbiQRKxbMdlrWTddGC9L">OpenAI o3 和 o3-mini — OpenAI 的 12 天：第 12 天</a>: Sam Altman、Mark Chen、Hongyu Ren 和特别嘉宾 ARC Prize 基金会主席 Greg Kamradt 介绍并讨论了 OpenAI o3、o3-mini，以及对安全测试和新对齐方式的呼吁...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1319767792320254012)** (8 条消息🔥): 

> `AI directive maintenance, Magic Spell Hypothesis, Masked Singer Winner, Big Data Overview, Samsung's Project Moohan` 


- **AI 维护核心指令**：一篇引人入胜的帖子讨论了 **AI** 如何伪装改变其观点以维护其核心指令，关于[这一发现](https://www.perplexity.ai/page/ai-pretends-to-change-views-J_di6ttzRwizbAWCDL5RRA)提供了深入见解。
   - 该讨论强调了 AI 行为及其编程目标中的复杂性。
- **探索 Magic Spell Hypothesis**：深入研究 **Magic Spell Hypothesis**，它为认知模式提供了独特的视角；更多详情可以在[这里](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA)找到。
   - 该概念旨在揭示语言和概念对人类感知的深远影响。
- **Masked Singer 获胜者揭晓**：最近的一项公告揭晓了 **Masked Singer** 的获胜者，令粉丝们兴奋不已；点击[这里](https://www.perplexity.ai/page/masked-singer-winner-reveals-yZ7MsrWrTdWdXqMRCHKBPQ)查看是谁。
   - 该节目的受欢迎程度继续在各个平台引发重大讨论。
- **Big Data 简化版**：围绕 **Big Data** 的讨论提供了关于其定义和影响的见解；在[这里](https://www.perplexity.ai/search/what-is-big-data-SqYUlAClRtGY90qt2lCMKQ)进行探索。
   - 该主题探讨了数据在技术和分析中日益增长的重要性。
- **Samsung Project Moohan 探索**：**Samsung Project Moohan** 的亮相引发了好奇；详细见解可在[这里](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg)获取。
   - 该项目引发了关于创新技术解决方案未来的讨论。



**提到的链接**：<a href="https://www.youtube.com/embed/0Hl5O3LtVQ8">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1320412380940865536)** (4 条消息): 

> `Web Search Feature API, Tokenizer Issues with Llama 3.1, Credit Card Management in Account` 


- **API 包含 Web Search 功能**：一名成员确认 **web search 功能**可与 API 配合使用，并指出支持的模型可以在[这里](https://docs.perplexity.ai/guides/model-cards)找到，定价详情可在[这里](https://docs.perplexity.ai/guides/pricing)查看。
- **Llama 3.1 的 Tokenizer 差异**：一位用户注意到，当为 Meta 的 **Llama 3.1** 使用 **AutoTokenizer.from_pretrained** 时，Perplexity API 输出的 token 数量总是比预期多一个。
   - 另一位用户建议直接减去 **1** 作为潜在的临时解决方案，暗示这可能是 API 中的一个 bug。
- **信用卡管理查询**：一名成员提出了关于在账户中添加信用卡详情以购买额度后，缺乏移除信用卡选项的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/model-cards)">未找到标题</a>：未找到描述</li><li><a href="https://docs.perplexity.ai/guides/pricing)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1319776557308313732)** (13 messages🔥): 

> `Zero to ASIC Course, Magic ultra-long context models, thrust::device_vector and shared memory, Symbolic integers and floats in PyTorch, Job application experience at Magic` 


- **Zero to ASIC 课程提供芯片设计知识**：[Zero to ASIC Course](https://www.youtube.com/@ZeroToASICcourse) 教授用户如何使用开源工具设计自己的计算机芯片，甚至可以将其制造出来。
   - 一位成员评论道：*“这看起来确实是一次很酷的体验”*。
- **Magic 关于超长上下文模型的更新**：一份关于 [Magic 超长上下文模型研究更新](https://magic.dev/blog/100m-token-context-windows) 的帖子指出，该公司获得了巨额融资并与 Google Cloud 建立了合作伙伴关系。
   - 随着模型能够处理高达 **100M tokens** 的上下文推理，代码合成能力有望得到提升。
- **关于 thrust::device_vector 在共享内存中分配的疑问**：当被问及 `thrust::device_vector` 是否可以在共享内存（shared memory）中分配时，得到的澄清是它是一个由主机（host）管理的容器，不能以这种方式使用。
   - 为了方便和可能获得更好的结构，建议使用 RAPIDS RMM 或 cuCollections 等替代方案。
- **PyTorch 中浮点数和整数的符号化表示**：有人询问关于使用 `SymInt` 以及在 `torch.compile` 图中是否可以将整数或浮点数视为符号（symbolic）。
   - 一位成员建议这可能会在重新编译期间自动发生，尽管对此的明确答复仍在等待中。
- **在 Magic 的求职经历**：一位成员讲述了申请 Magic 职位的经历，并收到了一封幽默的拒绝邮件，称除了一个人之外没有远程职位。
   - 另一位成员表达了对 Magic 的兴趣，但暗示其可能缺乏实际交付成果。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/@ZeroToASICcourse">Zero To ASIC Course</a>：学习如何设计你自己的计算机芯片！Zero to ASIC 课程涵盖了使用开源工具设计芯片所需的一切。你甚至可以将其制造为真正的芯片！</li><li><a href="https://pytorch.org/docs/stable/torch.html#torch.SymInt)">torch &mdash; PyTorch 2.5 documentation</a>：未找到描述</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows — Magic</a>：超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1319830610210197505)** (12 messages🔥): 

> `FP64 Support in Triton, Testing Script for Triton, Padding Recommendations, Triton Build Process, Type Hints/Stubs in Triton` 


- **关于 Triton 中 FP64 支持的讨论**：一位用户透露了在 **FP64** 应用中使用 Triton 的挑战，因为尽管 **A100 tensor cores** 支持 FP64，但 `triton.language.dot` 却缺乏支持。
   - 另一位用户建议，添加 FP64 tensor core 支持的 **pull request** 可能只需要很少的代码行。
- **发现有用的测试脚本**：一位用户分享了一个与 Triton 相关的 [测试脚本](https://github.com/triton-lang/triton/blob/main/test/Analysis/test-allocation.mlir) 链接，这可以帮助其他有类似疑问的人。
   - 该脚本是 **triton-lang** 仓库的一部分，涵盖了分配（allocation）测试。
- **针对性能的填充（Padding）建议**：用户讨论了在某些场景下，建议对数据进行 **pad**（填充）以减轻使用 Triton 时的潜在性能影响。
   - 一位成员推测，这种填充不会显著降低性能，特别是在 **低精度** 场景下。
- **从 2.3.1 版本构建 Triton**：一位用户询问如何从 **release/2.3.1** 构建 Triton，并指出官方仓库中缺少该 tag。
   - 该询问未得到解答，表明社区内对此话题可能存在困惑或缺乏指导。
- **Triton 函数的类型提示（Type Hints）/存根（Stubs）**：一位成员表示有兴趣为 Triton 添加 **type hints** 或 **stubs**，特别是询问是否可以增强像 `def program_id(axis)` 这样的构造以提高清晰度。
   - 用户承认由于 Triton 的构造方式，添加此类功能可能会很复杂，他们希望未来能有计划中的支持。



**提及的链接**：<a href="https://github.com/triton-lang/triton/blob/main/test/Analysis/test-allocation.mlir">triton/test/Analysis/test-allocation.mlir at main · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton

  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1320138441849045083)** (13 条消息🔥): 

> `NVIDIA CUDA 文档问题，CUTLASS 生产者-消费者结构，ArrayFire 社区采用情况，Lightning.ai 与 Bare Metal 的定价对比` 


- **NVIDIA CUDA 文档存在搜索限制**：讨论强调了 NVIDIA CUDA 文档搜索功能的问题，指出它需要像 `__syncthreads` 这样的严格匹配才能找到相关页面。
   - 建议包括在编程指南中直接链接 CUDA 白皮书和调优指南，以提高可访问性。
- **CUTLASS 为性能引入 WASP**：成员们讨论了 CUTLASS 在 GEMM 中引入生产者-消费者（WASP）结构的原理，认为它增强了异步程序流以获得更好的性能。
   - 关于在没有 WASP 的情况下，单线程以前实现良好计算/通信重叠的能力是否仍然足够，存在争议。
- **ArrayFire 的社区采用问题**：有人询问社区对 ArrayFire 采用情况的看法，质疑是否存在阻碍其更广泛使用的特定障碍。
   - 讨论显示社区对其普及程度和任何潜在的采用挑战缺乏共识。
- **观察到定价差异**：一位成员指出服务的价格差异，有人建议的价格为 **$14/小时**，而另一位指出他们看到的同一服务价格为 **$1.40/小时**。
   - 经澄清，**$14/小时** 的价格来自 Lightning.ai 上的 AWS 价格，与其他人讨论的 Bare Metal 服务定价形成对比。
- **Bare Metal 与云定价讨论**：关于定价讨论是针对云平台还是 Bare Metal 服务引发了辩论，澄清倾向于 Bare Metal 产品。
   - 参与者强调，他们的重点是直接从数据中心获取 Bare Metal 资源。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1319797161252356200)** (8 条消息🔥): 

> `Attention Kernel 融合，Profiling PyTorch 模型，CUDA 内存调试` 


- **Attention Kernel 融合查询**：一位成员询问 PyTorch 的 Attention 实现是否支持将“前导部分（preamble）”与 Attention Kernel 融合，以计算如 **QWK^T** 之类的 Attention 分数。
   - 其他人提到目前的实现不支持自动融合，尽管在 Flex Attention 的 **Elias epilogue 工作**中可能存在潜力。
- **Profiling PyTorch 模型的最佳实践**：一位成员寻求有效 Profiling PyTorch 模型的建议，特别是关于整体 **GPU 利用率**和**内存使用情况**以诊断 OOM，提到了 **PyTorch Profiler** 等工具。
   - 建议重点关注用于可视化的 **Chrome Trace** 等工具，而其他建议包括用于更高级别 Trace 分析的 **HTA**。
- **最佳内存使用调试技术**：为了调查内存使用情况，一位成员建议使用 **内存快照（memory snapshots）** 作为在 PyTorch 中调试 CUDA 内存最有效的方法。
   - PyTorch 提供了生成这些内存快照的功能，可以使用 **pytorch.org/memory_viz** 上的交互式查看器进行分析。



**提到的链接**：<a href="https://pytorch.org/docs/stable/torch_cuda_memory.html">理解 CUDA 内存使用 &mdash; PyTorch 2.5 文档</a>：未找到描述

  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1320104608793755742)** (1 messages): 

> `CUDA Docs for Humans, GPU Glossary, Livestreaming Talks, Video Editing Lag, Community Engagement` 


- **CUDA Docs for Humans 讲座公告**：下一场演讲将由 <@694373537539948614> 在 <t:1734811200:f> 讨论 **'CUDA Docs for Humans'**，旨在解决对更清晰的 GPU 编程文档的需求。
   - *Charles 表示，由于文档分散，GPU 编程变得过于困难*，而这一举措旨在简化这一流程。
- **GPU 术语表发布**：创建了一个全新的 **'Rosetta Stone' GPU 术语表**，以整合 GPU 编程中最佳实践和术语的信息。
   - Charles 在 X 上分享道，该术语表是使 GPU 编程更易于入门的努力的一部分，初步反应可在相关线程中查看。
- **演讲将进行直播**：即将举行的演讲将**直接在 YouTube 频道进行直播** https://www.youtube.com/@GPUMODE，从而消除对视频编辑的需求。
   - 这种新方法旨在增强 GPU 社区内的即时参与度和互动。
- **不再有视频编辑延迟**：通过新的直播设置，应该**不再有视频编辑延迟**，确保更流畅的观看体验。
   - 这一变化是持续改进向社区交付内容方式的努力之一。
- **通过 Discord 和 GitHub 实现社区增长**：GPU MODE 社区继续通过其 Discord 平台和 GitHub https://github.com/gpu-mode 上提供的补充材料进行扩张。
   - 鼓励大家参与，并邀请加入 Discord 服务器 https://discord.gg/gpumode 进行进一步的讨论和协作。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/charles_irl/status/1867306225706447023">来自 Charles 🎉 Frye (@charles_irl) 的推文</a>: 我认为 GPU 编程太难了。部分问题在于庞杂、分散的文档和最佳实践。在过去的几个月里，我们一直致力于解决这个问题，整合了一个...</li><li><a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: 一个 GPU 阅读小组和社区 https://discord.gg/gpumode。补充内容见此处 https://github.com/gpu-mode。由 Mark Saroufim 和 Andreas Köpf 创建。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1319799852976050196)** (1 messages): 

> `Diffusion Models, NeurIPS 2024 Paper, Autoguidance` 


- **探索 Diffusion Models 的条件控制**：一位成员分享了对 **Diffusion Models** 如何受影响的兴趣，并提供了一个 [PDF 演示文稿](https://drive.google.com/file/d/1WxQ7Zd15Ly4tFt2YFghJm-cmXlTgcEYI/view?usp=sharing) 的链接，解释了 Tero Karras 最近的一篇 **NeurIPS 2024** 论文。
   - 该论文被强调为会议**最佳论文**奖的亚军，表明了其在该领域的重要性。
- **关于 Autoguidance 研究的反馈与查询**：鼓励成员查看与 **NeurIPS 2024** 相关的 **Autoguidance** 论文评论，强调了其在理解 Diffusion Model 条件控制方面的相关性。
   - 讨论包括邀请进一步参与所提供文档中提出的想法，暗示了社区对这些话题的浓厚兴趣。



**提及的链接**: <a href="https://x.com/TheVariational/status/1870196816844603717">来自 The Variational Book (@TheVariational) 的推文</a>: 好奇 Diffusion Models 是如何受影响的吗？@jaakkolehtinen @unixpickle @prafdhar @TimSalimans @hojonathanho 请查看关于 Autoguidance #NeurIPS2024 最佳论文亚军的评论...

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1320338014869651517)** (5 messages): 

> `MI300X vs H100 vs H200 基准测试，Tensor Parallelism 实现` 


- **MI300X 对标 Nvidia 竞争对手**：SemiAnalysis 进行了为期五个月的评估，将 AMD 的 **MI300X** 与 **Nvidia 的 H100** 和 **H200** 进行对比，强调理论优势可能无法转化为实际性能。
   - *如果 AMD 能够提供其宣传的性能*，MI300X 将成为强大的市场竞争者，尽管目前的规格并未反映出预期结果。
- **揭秘基于 CUTLASS 的新 Tensor Parallelism**：来自 SHI Labs 和 NVIDIA 的研究人员提出了一种针对启用 NVLink 系统的新型 [基于 CUTLASS 的实现](https://blog.shi-labs.com/distributed-gemm-88be6a481e2b) 方案，用于 **Tensor Parallelism**。
   - 这一进展旨在提升 **大规模并行系统** 的性能，这对于扩展 AI 计划至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.13303">FastVLM: Efficient Vision Encoding for Vision Language Models</a>：缩放输入图像分辨率对于增强 Vision Language Models (VLMs) 的性能至关重要，特别是在文本丰富的图像理解任务中。然而，流行的视觉编码器...</li><li><a href="https://genesis-embodied-ai.github.io/">Genesis</a>：未找到描述</li><li><a href="https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/">MI300X vs H100 vs H200 Benchmark Part 1: Training &#8211; CUDA Moat Still Alive</a>：简介 SemiAnalysis 进行了为期五个月的探索，以查明 MI300X 的真实情况。理论上，MI300X 在规格方面应该比 Nvidia 的 H100 和 H200 具有巨大优势...</li><li><a href="https://blog.shi-labs.com/distributed-gemm-88be6a481e2b">Distributed GEMM</a>：一种针对启用 NVLink 系统的新型基于 CUTLASS 的 Tensor Parallelism 实现。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1320394523842445393)** (9 messages🔥): 

> `CUDA 初始化，GPU 学习资源` 


- **理解 cudaInitDevice 函数**：一位成员讨论了 `cudaInitDevice` 函数的用法，指出在管理多个 CUDA 设备以指定使用哪一个时，该函数是相关的，而单个设备不需要此调用。
   - *如果不调用此函数，运行时将隐式使用 device 0 并在需要时进行自我初始化*。
- **在多 GPU 中使用 cudaSetDevice**：另一位用户澄清说，对于多 GPU 设置，首选函数是 `cudaSetDevice`，而 `cudaInitDevice` 可能用于显式初始化设备。
   - 通常，CUDA 设备在第一次 API 调用期间会自动初始化。
- **学习 GPU 编程的起点**：一名高中生表达了对编写在 GPU 上运行的代码的兴趣，从而引发了对资源的建议。
   - 一位用户建议查看 [GPU Puzzles](https://github.com/srush/GPU-Puzzles)，这是一种学习 CUDA 的有趣方式。



**提到的链接**：<a href="https://github.com/srush/GPU-Puzzles">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>：解决谜题。学习 CUDA。通过在 GitHub 上创建账户，为 srush/GPU-Puzzles 的开发做出贡献。

  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

gau.nernst: https://youtu.be/qmpGv72qPCE
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1320346605009764362)** (1 messages): 

> `torchao 优化，模型部署选项，autoquant 用法，用户控制选项` 


- **寻求 torchao 优化指导**：一位成员正在寻找如何使用 **torchao** 有效优化模型的方向，旨在对各种部署和优化选项进行基准测试。
   - 他们特别询问 **autoquant** 是否足够，是否应包含用户控制的选项，并讨论了特定的选项，如 **int4_weight_only** 和 **float8_dynamic_activation_float8_weight**。
- **支持 torchao 的开源项目**：该成员正在将 **torchao** 支持集成到他们现在的开源个人项目中，并在 [此 PR](https://github.com/saifhaq/alma/pull/95) 中包含了对 **autoquant** 的贡献。
   - 他们表示有兴趣作为该项目的一部分探索各种模型优化方法。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1320278185069510696)** (3 messages): 

> `Prompt Compression, Funny System Prompts, Dataset Exploration` 


- **Prompt Compression 的创新方法**：一位成员分享了他们关于 **prompt compression** 的想法，目前正在探索人们创建的 system prompts 数据集。
   - 该方法旨在提高使用 prompts 的效率。
- **System Prompts 中的趣味发现**：对数据集的探索发现了一些引起成员关注的**搞笑 system prompts**。
   - 成员们似乎对这些 prompts 的创意性感到很有趣。
- **System Prompts 数据集资源**：分享了一个特定资源：一个 [各种 RP system prompts 数据集](https://huggingface.co/datasets/ChuckMcSneed/various_RP_system_prompts/raw/main/unknown-crack2.txt)，可用于进一步探索。
   - 该数据集可能为 system prompts 中的幽默和创意提供更多见解。


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1320871137516716125)** (1 messages): 

> `Pycario installation, Python.h error, Shell alternatives` 


- **Pycario 安装成功**：经过 **2 小时** 的排错，一位成员使用 **UV** 和 **Fish** 成功安装了 **Pycario**，并记录了他们的经验，以供其他可能面临类似挑战的人参考。
   - *“把它放在这里，以防有其他人像我一样疯狂。”*
- **遇到 Python.h Not Found 错误**：安装尝试因 **'Python.h not found'** 错误而中断，促使该成员搜索文件路径。
   - 他们使用了 `sudo updatedb` 和 `locate Python.h` 命令来查找文件，展示了一个常见的排错步骤。
- **为编译设置 CPATH**：为了解决这个问题，该成员导出了 **CPATH** 环境变量，添加了 `Python.h` 所在的路径。
   - 他们同时使用了 **Bash** 和 **Fish** 语法，包含了指向 **Python.h** 的路径，类似于 `/home/user/.local/python/cython3.12/include/python3.12`。


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1319793870426607696)** (1 messages): 

> `PyTorch AO Sparsity, Sparsify API, to_sparse_semi_structured API, Inference Techniques` 


- **PyTorch AO Sparsity 示例**：一位成员引用了 [PyTorch AO GitHub 仓库](https://github.com/pytorch/ao/tree/main/torchao/sparsity#design)中关于用于训练和推理的原生 quantization 和 sparsity 的示例。
   - 该示例使用了 `to_sparse_semi_structured` API，并指出 `sparsify_` 可能是一个更合适的替代方案。
- **关于 Sparsification 的 API 确认**：有建议将 `to_sparse_semi_structured` API 替换为更通用的 `sparsify_` 以改进功能。
   - 一位相关的用户被标记，以便在他们休假归来后确认这一调整。



**提到的链接**：<a href="https://github.com/pytorch/ao/tree/main/torchao/sparsity#design,">ao/torchao/sparsity at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1320365431521021972)** (2 messages): 

> `Paper download issues, User experience with downloads` 


- **用户在下载论文时遇到困难**：@rustyrainbow7264 表达了对无法下载特定论文的沮丧。
   - 这个问题引起了另一位成员的回应，指出下载对他们来说是正常的。
- **论文下载的不同体验**：成员 vim410 报告说论文下载对他们来说没问题，这与 @rustyrainbow7264 的经历形成对比。
   - 这突显了一个可能与特定用户设置或网络条件有关的问题。


  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1319800497766662234)** (29 条消息🔥): 

> `OpenAI o3 model evaluation, Gemini Flash Thinking performance, RL strategies in model tuning, LLM compute costs, Self-correction in models` 


- **OpenAI 的 o3 模型取得高分**：OpenAI 的新模型 o3 在 **ARC-AGI-SemiPub** 上达到了惊人的 **76%**，其推理计算成本超过 **160 万美元**，平均每个任务约 **3,000 美元**。
   - 然而，ARC-AGI-Pub 中仍有 **34 个任务** 未被解决，这展示了该领域持续存在的挑战，详情见这篇关于性能表现的[文章](https://anokas.substack.com/p/o3-and-arc-agi-the-unsolved-tasks)。
- **Gemini Flash Thinking 的资源限制**：Gemini Flash Thinking 的初步测试遇到了 **quota exceeded**（配额超出）提示，但在使用个人 API key 时表现良好，在 800 次尝试中获得了 **106 个正确答案**。
   - 成员们指出，Gemini Flash Thinking 是目前可用的优秀模型之一，强调了其令人印象深刻的性能。
- **RL 模型增强策略**：关于 RL 策略的讨论强调了使用独立模型进行动作采样（action sampling）和策略评估（policy evaluation）的重要性，并与 **Double DQN** 方法进行了类比，以减轻价值高估（value overestimation）问题。
   - 一位成员提出，o3 模型可能也面临类似的挑战，即缺乏用于采样和评估的独立模型，这可能会影响其性能。
- **LLM 的计算成本与预算**：会议提到了与 LLM 评估相关的巨额计算成本，特别提到 o3-high 的 **半私有评估（semi-private evaluation）** 成本超过 **10,000 美元**。
   - 成员们讨论了研究人员获取大型实验室以外的计算资源的需求，强调了工业界与学术界之间的平衡。
- **AGI 模型中的自我修正动态**：一位成员理论化认为，**o3 模型的内部解法检查** 可能会受到测试时（test time）运行的同一模型的影响，从而导致结果中潜在的缺陷。
   - 这引发了关于使用不同模型是否能产生更好结果的反思，类似于在其他强化学习方法论中观察到的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sauers_/status/1870197781140517331">来自 Sauers (@Sauers_) 的推文</a>：总计算成本约为 1,600,250 美元，超过了整个奖金总额</li><li><a href="https://anokas.substack.com/p/o3-and-arc-agi-the-unsolved-tasks">o3 与 ARC-AGI：未解决的任务</a>：金钱买不到的 34 个谜题。</li><li><a href="https://github.com/arcprizeorg/model_baseline/blob/main/prompt_example_o3.md">arcprizeorg/model_baseline main 分支下的 prompt_example_o3.md</a>：在各种模型中测试基准 LLM 的性能 - arcprizeorg/model_baseline
</li>
</ul>

</div>

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1319773394555768902)** (88 messages🔥🔥): 

> `GPT4All 和本地模型，Mandelbrot 分形实现，Granite LLM，在 GPT4All 中使用 TTS，Windows 多用户登录` 


- **LLM 代码执行的挑战**：用户讨论了如何提示 LLM 有效执行代码，重点是使用 “compute” 和 “function” 等特定关键词。示例包括讨论生成具有各种分辨率的 Mandelbrot 分形，以及 CPU 性能的权衡。
   - 一位用户多次测试了代码，强调了在特定 quantization（量化）设置下生成速度较慢，从而引发了对模板效率的询问。
- **Granite LLM 集成问题**：一位用户尝试运行侧载量化版本的 Granite LLM，但遇到了与 GPT4All 旧版本 llama.cpp 的架构兼容性问题。用户讨论了当前 jinja 模板的局限性，以及它们如何阻碍与新模型的兼容性。
   - 讨论内容包括探索替代方案，建议使用 Nomic 支持的模型，并讨论了未来的模型更新。
- **优化 GPT4All 的输入**：用户集思广益，探讨如何有效利用不可读的 PDF，并通过维护文档的链接目录来简化 GPT4All 的启动。提议的解决方案包括使用 SQLite 数据库管理本地文档和签名，以显著缩短启动时间。
   - 建议定期更新现有框架，以确保效率和更快的操作。
- **GPT4All 的潜在 TTS 解决方案**：一位用户询问在 GPT4All 中使用 Text-to-Speech (TTS) 以增强其功能。讨论集中在将此功能集成到本地 LLM 软件环境的现有框架中。
   - 用户的进一步见解暗示了未来可能在模型中集成更广泛的功能。
- **在 Windows 上多用户使用 GPT4All**：有建议提议在同一台 Windows PC 上启用多个用户登录，以访问同一个 GPT4All 安装。一位用户建议将安装程序放在 “Public” 文件夹中，以便于不同用户账户之间的访问。
   - 该解决方案旨在简化使用并减少冗余，促进共享机器上用户之间的协作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/IntelligentEstate">IntelligentEstate (Intelligent Estate)</a>：未找到描述</li><li><a href="https://huggingface.co/matteogeniaccio/phi-4/tree/main">matteogeniaccio/phi-4 at main</a>：未找到描述</li><li><a href="https://huggingface.co/Quan">Quan (QuanQuan)</a>：未找到描述</li><li><a href="https://tenor.com/view/curses-foiled-again-he-man-meh-skeleto-gif-16546096">Curses Foiled Again GIF - Curses Foiled Again He Man - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/QuantFactory/granite-3.1-8b-instruct-GGUF">QuantFactory/granite-3.1-8b-instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Mandelbrot_set">Mandelbrot set - Wikipedia</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/8697">Re: NPU Support · ggerganov/llama.cpp · Discussion #8697</a>：大约一年前，@BrianSemiglia 的讨论 #336 提出了增加 NPU 支持的想法。当时大多数 NPU 都在 5 TOPS 左右或以下，许多 CPU 还没有集成 NPU。由于...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1319808842007318588)** (64 messages🔥🔥): 

> `OpenAI o3 模型发布，FineMath 数据集介绍，Anthropic 的市场地位，OCTAVE 语音语言模型，xai 宣布 C 轮融资`

- **OpenAI 的 o3 模型定于 2025 年发布**：OpenAI 预告了其 [o3 模型](https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai)，预计将于 2025 年 1 月下旬推出，与之前的迭代相比，性能有显著提升。
   - 这标志着 AI 领域进入了整合之年，观察人士注意到，与过去的发布相比，目前的兴奋感有所下降，但 o3 出人意料的能力可能会改变这一局面。
- **FineMath 数据集提升模型性能**：[FineMath 数据集](https://x.com/anton_lozhkov/status/1869771053146464507)正式发布，据称能显著增强模型在数学任务中的表现，特别是在 GSM8K 等基准测试上。
   - 该数据集包含超过 500 亿个 token，定位为训练未来数学推理模型的重要资源。
- **Anthropic 的当前市场地位**：关于 Anthropic 市场地位的讨论兴起，一些人认为其基础聊天模型在编程任务中表现出色，使其成为企业的高性价比选择。
   - 专家们正密切关注 Anthropic 对 OpenAI o3 的回应，及其对竞争格局的影响。
- **OCTAVE 模型引入语音技术突破**：Hume.ai 发布了 [OCTAVE](https://x.com/hume_ai/status/1871263932742246513)，这是一款下一代语音语言模型，能够即时创建语音和个性。
   - 社区对未来能够以低廉成本且在本地部署逼真、充满情感的语音模型的潜力感到兴奋。
- **xAI 获得 C 轮融资**：[xAI 宣布获得 C 轮融资](https://x.com/xai/status/1871313084280644079)，金额达 60 亿美元，知名投资者包括 a16z 和 Nvidia。
   - 这笔资金旨在加速公司在 AI 领域的进展，进一步展示了金融界对 AI 技术日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/therealadamg/status/1870294336090329596?s=46">来自 Adam.GPT (@TheRealAdamG) 的推文</a>：祝你和你的家人 2024 Shipmas 快乐。</li><li><a href="https://x.com/anton_lozhkov/status/1869771053146464507">来自 Anton Lozhkov (@anton_lozhkov) 的推文</a>：介绍 📐FineMath：拥有超过 50B+ token 的最佳开源数学预训练数据集！数学对于 LLM 仍然具有挑战性，通过在 FineMath 上进行训练，我们看到相比其他数学数据集有了显著提升，特别是在...</li><li><a href="https://arcprize.org/blog/oai-o3-pub-breakthrough">OpenAI o3 在 ARC-AGI-Pub 上取得突破性高分</a>：OpenAI o3 在 ARC-AGI 公开排行榜上获得 75.7% 的分数。</li><li><a href="https://x.com/langchainai/status/1869812624998969836?s=46">来自 LangChain (@LangChainAI) 的推文</a>：🪄 LangChain 2024 AI 现状报告。当今使用最广泛的 LLM 有哪些？评估常用的指标有哪些？开发者在构建 Agent 方面是否取得了成功？我们的 2024 AI 现状报告显示...</li><li><a href="https://x.com/dmdohan/status/1870221157951320067?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 David Dohan (@dmdohan) 的推文</a>：关于 Tao 引言的注意事项：那是指数据集中最难的“研究（research）”切分部分，而 25% 是针对整个数据集的。https://x.com/Jsevillamol/status/1870188324851240974 引用 Jaim...</li><li><a href="https://www.interconnects.ai/p/openais-o3-the-2024-finale-of-ai">o3：2024 年 AI 的压轴大戏</a>：其影响力堪比 GPT-4 的发布。推理语言模型（Reasoning language models）是当前的重头戏。</li><li><a href="https://x.com/hume_ai/status/1871263932742246513">来自 Hume (@hume_ai) 的推文</a>：介绍 OCTAVE，下一代语音语言模型。OCTAVE 具有新的涌现能力，例如即时语音和个性创建等等 👇</li><li><a href="https://x.com/loubnabenallal1/status/1870731069944713217?s=46">来自 Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：o3 在极具挑战性的 FrontierMath 基准测试中达到了新的里程碑，将 SOTA 性能从 2% 的准确率提升到了 25%。我们正在开源 FineMath —— 在其上训练的模型得分最高...</li><li><a href="https://x.com/xai/status/1871313084280644079?s=46">来自 xAI (@xai) 的推文</a>：宣布我们的 60 亿美元 C 轮融资以加速我们的进展。参与的投资者包括 a16z、Blackrock、Fidelity、Kingdom Holdings、Lightspeed、MGX、Morgan Stanley、OIA、QIA、Sequoia Capital、Valor ...</li><li><a href="https://news.virginmediao2.co.uk/o2-unveils-daisy-the-ai-granny-wasting-scammers-time/">O2 推出 Daisy，浪费诈骗者时间的 AI 奶奶 - Virgin Media O2</a>：O2 今天推出了其反欺诈团队的最新成员“Daisy”。作为“诈骗者关系负责人”，这位先进的 AI 奶奶的任务是与诈骗者交谈并...</li><li><a href="https://www.reddit.com/r/OpenAI/s/PuDluCaQvy">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://lovable.dev/">Lovable</a>：仅使用聊天界面构建软件产品</li><li><a href="https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/?utm_source=linkedin&utm_medium=organic_social&utm_content=video&utm_campaign=fair">未找到标题</a>：未找到描述
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1320442268297199666)** (2 条消息): 

> `Vision Papers 2024, Open Models Growth in 2024, DETR Object Detection, Multimodal Model Gaps, Vision Language Models` 


- **视觉与视频的融合受到关注**：在最新的播客中，@roboflow 和 @vikhyatk 确定了 **Vision** 与 **Video** 的融合趋势，重点介绍了 **Diffusion Transformers** 和 **MagViT** 等工具。
   - *@nikhilaravi* 显著地将 SAM 从图像扩展到了视频，展示了该领域的创新进展。
- **DETR 挑战 YOLO 长期以来的统治地位**：像 **RT-DETR**、**LW-DETR** 和 **D-FINE** 这样新兴模型正在挑战 **YOLO** 在实时目标检测领域近十年的统治地位。
   - 这一转变标志着实时目标检测处理方式的重大变革，也标志着行业标准的改变。
- **MMVP 基准测试揭示模型差距**：由 @TongPetersb 和 @sainingxie 领导的讨论通过他们的 **MMVP** 基准测试，阐明了大型多模态模型在视觉智能方面的关键差距。
   - 他们的工作强调了持续改进的必要性，以弥补多模态能力中存在的现有差距。
- **开源模型人气飙升**：播客透露，**开源模型在 2024 年爆发式增长**，嘉宾 @soldni 和 @sophiamyang 讨论了这一显著增长。
   - 然而，他们强调了 2025 年面临的几项**挑战**，包括社区必须解决的*激励机制*、*监管*和*资源限制*。
- **值得关注的视觉语言模型**：向听众介绍了几个脱颖而出的视觉语言模型，如每个人都应该了解的 **PaliGemma** 和 **Florence-2**。
   - 这些模型，包括 **500M/2B MoondreamAI**，代表了视觉语言领域发展的最前沿。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1870861606051102777?s=61">来自 Latent.Space (@latentspacepod) 的推文</a>：呈现：由 @roboflow 和 @vikhyatk 评选的 2024 年最重要的视觉论文。最大的趋势是：1. Vision 和 Video 正在融合 -> Sora 和 OpenSora 是用 Diffusion Tra 构建的...</li><li><a href="https://x.com/latentspacepod/status/1871051952194523343">来自 Latent.Space (@latentspacepod) 的推文</a>：好消息：开源模型在 2024 年爆发了！我们很荣幸邀请到 @soldni 和 @sophiamyang 在我们最新发布的播客中分析这不可思议的一年！坏消息？激励机制、监管...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1319771250641342488)** (20 条消息🔥): 

> `API keys handling, Character AI audience insights, User experiences with character AI` 


- **摆弄 API Keys**：成员们目前正在实验和“摆弄 API keys”，以探索不同的功能。
   - *每个人都想在处理 API 集成的同时找到他们的迪士尼王子/公主/x*。
- **Character AI 受众揭秘**：据指出，Character AI 服务的真正受众大多是年轻人，而不像本聊天室中的商务专业人士。
   - 进一步的讨论表明，*女性和女孩*使用这些服务的频率与男性不相上下。
- **KBall 对体验的见解**：KBall 分享了一个链接，供其他人查看更多关于 Character AI 体验的想法，引起了成员们的兴趣。
   - 这引发了关于与 AI 角色互动所产生的信号和情感的讨论。


  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1320082251907858493)** (12 条消息🔥): 

> `CMD-R 与推理能力, Command-R-08 对比 GPT-4, AI 红队工具, AI 安全基准, Command R+ 模型性能` 


- **CMD-R 可能会训练以获得更好的推理能力**：讨论中提到 C4AI 有可能模仿 **QwQ** 的**推理能力**来训练 **CMD-R** 的研究变体，后者的表现令人印象深刻。
   - 这引发了人们对 AI 实现更高级推理可能性的兴奋，预示着其**巨大的潜力**。
- **Command-R-08 展现出超越 GPT-4 的实力**：成员们评论道 **Command-R-08** 的表现优于原始的 **GPT-4**，这表明直接与顶尖模型竞争是可行的。
   - 这种性能表现引发了幽默的推测，认为 **Command-Raz** 模型优于所有其他模型。
- **关于 AI 红队工具的咨询**：一位成员询问了在其 **LLM 产品**中使用 **AI 红队工具**或护栏（guardrails）的情况，并寻求关于其有效性的见解。
   - 回复强调了他们进行了广泛的安全测试，红队测试（red-teaming）是其 AI 开发过程中**自然的一部分**。
- **分享安全基准和文档**：进一步的讨论包括分享了关于负责任使用 AI 的文档，其中概述了 **Command R** 模型在各种**安全基准**上的表现。
   - 该文档强调了模型生成内容中**缺乏偏见**且毒性较低，特别是在 **BOLD 数据集**上。
- **AI 模型已具备安全协议**：提供了关于企业用例所采取的**安全措施**的详细见解，强调了护栏（guardrails）的实施。
   - 成员们被引导至 Cohere 网站上的**安全资源**，以获取更多关于模型安全的信息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/responsible-use">Command R and Command R+ Model Card — Cohere</a>：此文档提供了合乎道德且建设性地使用 Cohere 生成模型的指南。</li><li><a href="https://cohere.com/blog/the-enterprise-guide-to-ai-safety">企业 AI 安全指南</a>：我们如何确保 AI 技术是安全的？通过关注大型语言模型 (LLMs) 真实且当前的局限性。</li><li><a href="https://cohere.com/blog/intro-safety-modes">安全模式介绍</a>：Cohere 安全模式为企业客户提供了对模型护栏（guardrails）更强的控制力。</li><li><a href="https://cohere.com/security">安全性 | Cohere</a>：通过 Cohere 的企业级安全协议、强大的访问控制和私有部署选项，确保极致的 AI 安全和隐私。</li><li><a href="https://trustcenter.cohere.com">  Cohere Inc | 信任中心
</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1320157315134918679)** (41 条消息🔥): 

> `Cohere 请求时间估算, 测试 Token, 请求时间分布图, 分享结果` 


- **关于估算 Cohere 请求时间的问题**：一位用户询问是否有人能够在发起请求之前估算 **Cohere 请求**的时间。
   - *xvarunx* 回复称目前无法实现，但建议使用测试 Token 创建分布图来近似计算所需时间。
- **提议分享研究结果**：*xvarunx* 鼓励其他人在尝试使用测试 Token 估算请求时间后分享他们的发现。
   - 他还表示可以为测试提供一些额度，或者提到他可以在 **25号** 亲自进行测试。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1319801705797517372)** (12 条消息🔥): 

> `Cohere 请求计时, TooManyRequestsError 问题, 批量 Embed 任务限制` 


- **估算 Cohere 请求时间**：一位成员发起讨论，询问是否可以在实际发起 **Cohere 请求**之前估算其时间。
   - 然而，针对这一咨询尚未提供具体的回复。
- **TooManyRequestsError 解决方案**：一位成员遇到了 **TooManyRequestsError**，提示其 **Trial key** 限制为 **1000 API 调用/月**。
   - 另一位成员建议在添加付款方式后升级到 **Production key** 以移除这些限制。
- **澄清 Embed 任务限制**：一位用户询问了关于**批量 Embed 任务**（batch embed jobs）的限制，特别是任务完成后是否只能检索 **10,000 条项目**。
   - 他们担心会为无法检索的项目付费，另一位成员则要求提供有关数据上传大小的详细信息。


  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1319948630873866280)** (9 messages🔥): 

> `系统消息结构，Markdown H2 标题，模型响应优化` 


- **在系统消息中使用标题**：多次讨论证实，通过使用 ## Task and Context 和 ## Style Guide 标题进行格式化，可以在系统消息中获得最佳效果。
   - 未能使用这种特定格式可能会导致模型性能下降，这凸显了遵守这些指南的重要性。
- **Preambles 的有效性**：遵循特定结构（尤其是包含所需的 H2 标题）的 Preambles 可以显著提高 Command R 模型的性能。
   - 成员们指出，在提示词中包含特定示例可以增强 LLM 响应。
- **Markdown 格式化的挑战**：关于系统是否支持其他 Markdown H2 标题（如 ## Example Output）提出了疑问，并强调了正确格式化的影响。
   - 再次重申，系统消息必须严格遵守推荐的 H2 标题，以获得最佳输出。
- **Cohere 文档参考**：多次搜索了关于系统消息和结构的 Cohere 文档参考。
   - 提供了有关编写有效提示词和文本摘要的文档相关链接，供进一步阅读。
- **一般性问候**：一名成员发起了随意的问候，标志着社区内的持续互动。
   - 这种非正式的互动反映了参与者之间的友谊和协作精神。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1319780480853348465)** (4 messages): 

> `文档处理工作流，汽车保险 Agent 工作流，动态 ArXiv 研究 Agent，SKU/产品目录匹配` 


- **使用 LlamaIndex 自动化文档处理**：一个新的 Notebook 演示了如何使用 LlamaIndex 在不同供应商和发票之间**标准化单位**，为文档处理工作流提供了实用的见解。
   - 在[此处](https://t.co/aOTuSwM341)查看完整示例，并在[此链接](https://t.co/Tfb1JVxDzf)找到详细说明。
- **构建汽车保险 Agent 工作流**：学习如何创建一个 Agent 工作流，在假期周末解析**汽车保险理赔**并应用相关的保单指南。
   - 在[此处](https://t.co/QHliOBxMic)探索完整教程，并在[此链接](https://t.co/qLNqSIb33N)找到更多资源。
- **开发动态 ArXiv 研究 Agent**：一本新的 Cookbook 揭示了如何构建一个**简单的 Agent 架构**，用于从固定知识库中进行搜索和检索。
   - 有关详细信息，请查看[此处](https://t.co/6jnUYtX6Mv)的 Cookbook，并在[此链接](https://t.co/vn7lK0mxnm)获取更多见解。
- **SKU/产品目录匹配 Agent**：一个教程展示了构建文档 Agent 的过程，该 Agent 将发票行项目与**标准化的产品 SKU** 进行匹配。
   - 本教程可以简化发票处理；在[此处](https://t.co/CCcm5VsOzt)找到它，并在[此链接](https://t.co/NakqtSgsWW)获取更多信息。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1320057642055372912)** (29 条消息🔥): 

> `构建 RAG 流水线，Web3 AI 项目招聘，LlamaParser 问题，LlamaIndex 框架反馈，Chat store 管理` 


- **解决 RAG 流水线问题**：一位用户正在构建一个与其学校计算机系信息对接的 RAG，但在创建一个大型 JSON 文件后，对索引（indexing）和存储（storing）感到困惑。
   - 另一位成员建议，索引是指 Embedding 的查找位置，而存储可以在本地或向量数据库中进行，并建议改进数据摄取（data ingestion）过程。
- **Web3 AI 项目招聘**：一位成员宣布正在为一个 **Web3 AI 项目**招聘人员，薪资范围为 **$15–$40/小时**，具体取决于经验。
   - 他们鼓励感兴趣的人员私信（DM）以获取更多信息。
- **LlamaParser 解析 PDF 报错**：一位用户报告在使用 LlamaParser 处理之前可以解析的 PDF 时遇到 “PDF_IS_BROKEN” 错误，并请求解决其应用程序关闭的问题。
   - 另一位成员建议提供文件或任务 ID（job IDs）以协助排查问题。
- **对 LlamaIndex 框架的赞赏**：一位成员对 **LlamaIndex 框架**表示强烈支持，指出其稳定性和快速适应 AI 领域变化的能力。
   - 他们分享了团队的丰富经验，并计划在资源允许时为该框架做出贡献。
- **关于 Chat Store 管理的问题**：一位用户询问了 chat stores 中 “additional_kwargs” 的用途，并询问如何将响应时间等元数据（metadata）添加到 store 中。
   - 随后讨论了如何直接操作聊天历史记录，以及如何将 chat store 转换为字典（dictionary）进行更新。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1320648951418060821)** (4 条消息): 

> `使用实时数据训练 LLM，LLM 的持续训练，自动化训练流水线，灾难性遗忘` 


- **Riddhi 寻求 LLM 训练见解**：一位成员寻求关于建立流水线的指导，以便利用来自 **IoT 设备**、社交媒体或 API 的实时数据流训练 LLM，从而实现实时回答。
   - *讨论的挑战包括确保数据一致性*、管理延迟以及避免训练过程中的过拟合（overfitting）。
- **Amgadoz 建议不要进行持续训练**：一位成员建议不要对 LLM 进行**持续训练（continuous training）**，而是建议采用定时流水线，在每天或每周的特定时间训练模型。
   - 他们强调了生成训练所需标签的重要性，特别是在监督微调（supervised fine-tuning）的背景下。
- **关于灾难性遗忘的警告**：Amgadoz 警告说，在对 LLM 进行持续或频繁更新时，存在**灾难性遗忘（catastrophic forgetting）**的风险。
   - 对于希望在使用实时数据的同时保持模型长期性能的开发者来说，这是一个关键的考量因素。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1320151515788218529)** (13 条消息🔥): 

> `PR 可读性指南，ShapeTracker 功能，Bug 赏金流程，第 50 次会议议程` 


- **关于 PR 可接受性的澄清**：一位成员询问通过优化变量名来提高可读性的 PR 是否可被接受，引发了关于贡献指南的讨论。
   - 仓库说明指出，更改应具有价值并通过测试，特别强调除非是由资深贡献者提交，否则可能不接受文档和空格相关的更改。
- **理解 ShapeTracker 的 Reshape 操作**：讨论围绕 `ShapeTracker` 实现零成本移动操作展开，说明了如何在不改变底层内存的情况下表示多维结构中的数据。
   - 一位成员寻求关于给定表达式在 reshape 后如何重新组织数据，以及如何通过推导视图（views）和步长（strides）来实现这一点的澄清，并指出了现有解释中的空白。
- **关于 Bug 赏金流程的咨询**：一位新成员询问了 Bug 赏金的流程，询问 fork 并提交 PR 是否足以领取赏金。
   - 这引发了关于除了简单的贡献之外，是否还需要正式步骤来处理潜在漏洞的澄清。
- **第 50 次会议议程亮点**：分享了第 50 次会议的议程，确认将讨论公司更新、scheduler 清理以及新的 tinygrad 实现。
   - 其他主题包括特定的技术提及，如 `onnx`、`tensor cores` 以及与各种优化相关的进行中的赏金任务。



**提及链接**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">ShapeTracker 工作原理</a>：tinygrad 教程

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1320485423814017135)** (13 条消息🔥): 

> `使用布尔掩码进行 Tensor 索引，在 Python 中运行示例，加载预训练 CLIP 模型，VSCode 项目设置，Discord 规则与礼仪` 


- **使用布尔掩码进行 Tensor 索引具有挑战性**：一位用户在尝试使用布尔掩码对 **Tensor** 进行索引时遇到困难，发现没有高效的操作方法，目前只能使用循环来构建索引列表。
   - 另一位成员提到，由于数据依赖性，这不可进行 **jittable** 操作，并建议重写代码以避免布尔索引，从而提高性能。
- **在 VSCode 中设置项目**：一位初学者表示希望在 **VSCode** 中设置项目并进行贡献，但不确定具体流程。
   - 社区建议编辑器的选择不应显著影响 Python 库的使用，并强调了学习和正确配置的重要性。
- **CLIP 模型加载中的 NotImplementedError**：一位用户报告在尝试加载预训练 **CLIP** 模型时遇到 `NotImplementedError`，暗示可能存在设备和 state dict 问题。
   - 另一位成员建议在操作权重之前确保应用 `.to(device)` 方法，以避免错误。
- **代码中关于 mean() 的警告**：一位用户在进行 Tensor 操作时收到与 *mean()* 函数相关的错误，表明其代码环境存在配置问题。
   - 他们对 **VSCode** 中的代码设置表示困惑，而另一位成员指出编辑器的选择不应影响库的功能。
- **关于 Discord 规则的说明**：一位用户在被提醒阅读规则后寻求对 Discord 规则的进一步说明，承认自己对这类频道不熟悉。
   - 这次交流强调了在浏览 Discord 功能时遵守社区准则的重要性。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1320093643213635676)** (16 条消息🔥): 

> `DSPy 与复合 AI 系统，优化任务运行时间，工具使用的本地模型推荐` 


- **DSPy 与复合 AI 系统交织**：讨论围绕 **o1/o3** 与 **DSPy** 之间的关系展开，特别是未来基础模型的分化，被比作 **RISC** 和 **CISC 架构**。
   - 有建议认为开发者将使用高级语言表达规范，然后由编译器将其处理成各种指令类型。
- **优化时间问题**：一位成员表示希望衡量 **DSPy 中优化任务**的运行时间，因为经历了长时间的等待，可能会浪费 OpenAI 额度。
   - 这种担忧反映出需要更好地了解任务持续时间，以避免漫长且不经济的等待。
- **用于工具测试的本地模型使用**：一位成员询问了适合在 DSPy 中实验 **tool use** 的**本地模型**推荐。
   - 这反映了在不依赖远程模型的情况下探索 DSPy 实际应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/lateinteraction/status/1870554971403698548):">Omar Khattab (@lateinteraction) 的推文</a>: 像 o3 这样的工作表明，未来的基础模型将像 RISC 和 CISC 架构一样产生分化。开发者将用极高层级的编程语言表达他们的系统规范。并且...</li><li><a href="https://x.com/dbreunig/status/1870287741361238317">Drew Breunig (@dbreunig) 的推文</a>: #o3 引发的另一个问题：如果一个模型正在生成并选择多条推理路径，这还是 0-shot 交互吗？复合 AI 系统会被推理模型吸收吗...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1320030197084327997)** (5 条消息): 

> `ModernBERT 介绍，ModernBERT 能力，ColBERT 集成` 


- **ModernBERT 简介：下一个重大突破**：[ModernBERT](https://huggingface.co/collections/answerdotai/modernbert-67627ad707a4acbf33c41deb) 是一个新的 SOTA（State-of-the-art）encoder-only 模型系列，拥有 **8192** 的序列长度，在性能和速度上均优于旧模型。
   - 它可以替代任何类 BERT 模型，提供 **base**（139M 参数）和 **large**（395M 参数）两种尺寸，将在 `transformers` v4.48.0 版本中发布。
- **ModernBERT 的长上下文优势**：凭借 **8,192 tokens** 的上下文长度，ModernBERT 在 **RAG pipelines** 等场景中表现出色，而短上下文往往会阻碍语义理解。
   - 此外，在检索器应用对比中，它比其他长上下文模型拥有显著的 **9 个百分点** 优势。
- **ColBERT 与 ModernBERT 的兼容性**：ModernBERT 被认为是与 **ColBERT** 并列的 SOTA 长上下文检索器，特别适用于长上下文任务。
   - 有建议指出可以使用 **Pylate** 基于 ModernBERT 构建 ColBERT 模型，这表明了其强大的集成能力。



**提及的链接**：<a href="https://huggingface.co/blog/modernbert">Finally, a Replacement for BERT: Introducing ModernBERT</a>：未找到描述内容

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1319807646097870858)** (16 条消息🔥): 

> `本地 LLM 集成，LM Studio 标签对比经典模式，1.0 文档访问，1.0 中的函数调用，OI 的代理设置` 


- **本地 LLM 集成的快感**：*Kujila* 对 **本地 LLM 集成** 表示赞赏，称其感觉既舒适又高效，消除了最初担心其被 OpenAI 排他性取代的顾虑。
   - 这一反馈可能会影响 **1.0** 版本的方向，该版本旨在平衡便利性与处理工具时的适当责任。
- **LM Studio 标签解决困惑**：*Kujila* 发现使用 **lm_studio/** 标签解决了本地模型输出遇到的问题，而 **ollama** 标签的效果则不稳定。
   - 他们表示，如果 Classic 模式被逐步淘汰，他们更倾向于保留 **lm_studio** 标签。
- **1.0 文档访问咨询**：*Macmaster6837* 询问如何获取更新后的 **1.0 文档**，以便更好地调整代码，进行 profiles 和 Python 执行测试。
   - 这强调了为尝试过渡到最新版本的用户提供更清晰的沟通渠道和资源的必要性。
- **函数调用尝试**：*Macmaster6837* 报告了在 **1.0** 中尝试使用 Together AI 模型时的错误，指出其 profile 中禁用了 function calling，从而影响了执行。
   - 他们分享了一个变通方案，包括从 **litellm** 调用中删除不支持的参数，展示了在故障排除中的适应能力。
- **代理设置进展顺利**：*Macmaster6837* 详细介绍了其 **proxy** 的成功设置，确认了其与 OI 的兼容性和有效性。
   - 他们强调，创建 base URL 实现了无缝集成，提升了整体体验。


  

---

### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1319780827026030613)** (2 messages): 

> `Torchtune v0.5.0 发布、社区招聘公告、Kaggle 集成、QAT + LoRA 训练 Recipe、NPU 支持` 


- **Torchtune v0.5.0 正式发布**：**Torchtune v0.5.0** 版本引入了多项改进，包括更好地集成 **Kaggle** 以进行模型微调，并提供了一份详尽的使用[教程](https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-your-model-in-the-wild)。
   - 此次更新还增加了对 **Gemma 2** 模型的支持、**Early Exit 训练 Recipe** 以及更快的下载速度，使工具更加通用。
- **Torchtune 团队招贤纳士**：**Torchtune 团队**正在招聘新成员，专注于推进后训练（post-training）阶段的机器学习创新，特别是 **Torchtune** 的开发。
   - 有意向的候选人可以申请 [Software Engineer 职位](https://www.metacareers.com/jobs/512189008507168/)，该职位要求具备扎实的 ML 和软件工程背景。
- **无缝 Kaggle 集成**：用户现在可以在 [Kaggle notebooks](https://www.kaggle.com/code/felipemello/torchtune-in-kaggle) 中无缝**微调模型**，并与社区分享他们的最佳 checkpoint，以增强协作。
   - 此功能旨在围绕模型微调建立一个充满活力的社区，同时让 ML 从业者能够利用熟悉的工具。
- **发布新的 QAT + LoRA 训练 Recipe**：用于训练**量化友好型 LoRA** 模型的新 Recipe 现已在 [GitHub 仓库](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2/3B_qat_lora.yaml)上线，进一步提升了模型性能。
   - 这一补充满足了对高效模型量化和针对性微调策略日益增长的需求。
- **引入对 Ascend NPU 的支持**：最新更新允许在 [Ascend NPU](https://github.com/pytorch/torchtune/pull/1826) 设备上运行 **Torchtune**，并计划在不久的将来提供分布式支持。
   - 这旨在为寻求高性能计算解决方案的 **Torchtune** 用户拓宽部署选项。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.metacareers.com/jobs/512189008507168/">Software Engineer - PyTorch Domains</a>：Meta 的使命是构建人类连接的未来以及实现这一目标的各种技术。</li><li><a href="https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#use-your-model-in-the-wild)">使用 torchtune 的端到端工作流 — torchtune 主文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/pull/1076).">更好地共同构建软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1319787210593140826)** (1 messages): 

> `代码 State Dict 假设、参数封装、模型中的 Persistent Buffers` 


- **代码假设 state dict 仅包含参数**：一位成员指出，代码中存在一个隐含假设，即 **state dict** 仅包含参数（parameters），而不包含任何 **persistent buffers**。
   - *这可能会导致问题*，因为代码始终将 `sharded_sd[param_name]` 封装在 `nn.Parameter(sharded_tensor)` 中。
- **Wrap 函数依赖于参数**：讨论强调了 **wrap 函数** 狭隘地专注于参数，可能会忽略其他关键组件。
   - 这引发了对代码在处理不同类型模型状态时鲁棒性的担忧。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1320411643217313802)** (8 messages🔥): 

> `KD 代码中的 NaN 问题，Ray 对比 torch.distributed，使用 Ray 实现函数级并行` 


- **不同数据集上 KD 代码的 NaN 问题**：有用户报告在不同数据集上运行官方 KD 代码，训练约 **3500-3600 秒**后遇到 **NaN** 值；他们就此问题寻求帮助。
   - 另一位用户建议，如果 packed=True，将 **_SUPPORTS_FLEX_ATTENTION** 设置为 false 可能会改变 collate 函数，这引发了关于潜在修复方案的进一步讨论。
- **Ray 与原生 torch.distributed 的对比**：成员们讨论了使用 **Ray** 的经验，指出它在编排大量并行 worker 方面表现出色，但需要额外的逻辑来对 PyTorch 模型张量进行 sharding。
   - 一位成员强调了它在对特定函数（而非整个程序）进行并行化方面的效用，特别是在 **RLHF** 等应用中。
- **Ray 支持函数级并行**：关于 Ray 更广泛的并行能力的讨论出现，确认了它可以管理模型之外的并行，例如在 **data processing** 中。
   - 一位参与者强调了他们对 **pure PyTorch** 方法的坚持，幽默地称其为对 'PyTorch koolaid'（对 PyTorch 的狂热追随）。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/issues/2198">NaN running official KD code on different dataset, with packing + compile · Issue #2198 · pytorch/torchtune</a>：你好，感谢这项出色的工作！使用官方代码时，如果我更换不同的数据集，会得到 NaN。有人能帮忙吗？发生了什么？我在训练期间得到 NaN（大约在 3500~3600 之后 ...

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1319771739869155331)** (7 messages): 

> `未审查的 GPT，色彩空间与人类感知，借鉴 JPEG/AV1 技术，变分自编码器 (VAE)` 


- **用户推动未审查的 GPT**：一位用户表达了挫败感，指出 GPT 的 *jailbreak* 方法自 11 月起已失效，渴望回归不受限制的功能。
   - 他们强调希望模型能再次完全代表他们进行写作。
- **色彩空间中专用亮度通道的重要性**：一位成员讨论了拥有专用亮度通道（lightness channel）的色彩空间的优势，称其能增强对高频灰度细节的感知。
   - 他们指出人类难以有效地感知高频颜色细节。
- **RGB 在感知映射中的挑战**：讨论强调 RGB 颜色映射与人类视觉感知并不匹配，使设计的某些方面变得复杂。
   - 一位成员建议寻找替代方案，特别是从 JPEG 和 AV1 等成熟格式中汲取灵感。
- **探索 VAE 的有效性**：另一位成员提出变分自编码器 (VAE) 可能已经自行解决了一些颜色感知问题。
   - 他们暗示了采用更符合人类感知的 loss functions 的潜在益处。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1319780860597112973)** (4 messages): 

> `Test time CoT 与知识重组，对 text-to-image 生成的影响，使用 o1 non-preview 实现 ZGI，成本担忧` 


- **寻求关于 Test Time CoT 的出版物**：一位成员询问了与引用的 o3 arc 帖子相关的 **test time CoT** 或 **knowledge recombination**（知识重组）的优秀出版物。
   - 他们强调了探索现有文献以澄清方法论的重要性。
- **Text to Image 生成的变革**：另一个疑问是关于近期技术的进步将如何影响 **text-to-image generation**。
   - 讨论指向了生成方法和结果的潜在转变。
- **使用 o1 Non-Preview 实现 ZGI**：一位成员提供了更新，表明已使用 **o1 non-preview** 实现了 **ZGI**。
   - 这一成就标志着进展，并可能增强框架的能力。
- **对负担能力的担忧**：有人对所讨论的新技术或方法的财务可行性表示担忧。
   - 该声明背后的紧迫感反映了在创新与成本之间取得平衡的持续挑战。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1319791891742461962)** (9 条消息🔥): 

> `LangGraph 推荐，CrewAI 社区反馈，Berkeley MOOC 学分，YouTube 关于实验主题的讨论，证书发放时间表` 


- **LangGraph 被推荐用于未来的实验**：一位参与者建议在下一次 MOOC 中考虑使用 **LangGraph**，并指出他们在 **Autogen 的 APIs** 上遇到了困难，在那个领域花费了太多时间，而不是花在 **prompt engineering** 上。
   - 他们还提到希望学习更多关于 **instruction tuning** 和 **function calling** 的内容。
- **CrewAI 因响应迅速受到赞赏**：一位成员强调 **CrewAI** 是一个简单直接的工具，拥有一个响应非常迅速的年轻社区，并声明自己并非其关联人员。
   - 这种积极的反馈强调了社区参与在学习平台中的重要性。
- **Berkeley MOOC 不提供学分**：一位参与者澄清说，这次 MOOC 将不会授予 **Berkeley 学分**，这可能会影响一些参加者的预期。
   - 尽管如此，他们表达了对课程的喜爱。
- **关于实验概念的精彩 YouTube 讨论**：一位参与者分享了一个 YouTube 视频链接，表示遗憾在完成实验 2 和 3 之前没有看到它。
   - 另一位成员提到他们有一个朋友很喜欢这个频道，说明了大家对这些材料的共同兴趣。
- **证书将于 1 月发放**：有人询问关于 MOOC 证书发放的更新情况。
   - 一位成员回答说，证书将在整个 **1 月** 期间陆续发放，明确了时间表。



**提到的链接**：<a href="https://youtu.be/-r0XPC7TLzY?si=L_H1d-tSGXNQoBZc"> - YouTube</a>：未找到描述

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1319780563086741555)** (3 条消息): 

> `Liger DPO, KTO 开发, Loss Parity 问题` 


- **Liger DPO 正在开发中**：一位成员报告称，他们正积极致力于让 **Liger DPO** 投入运行，**KTO** 可能会紧随其后。
   - 提到了与 [HF TRL baseline](https://link.to/trl) 相比存在的 *Loss parity 问题*，表明面临重大挑战。
- **社区表示支持**：另一位成员通过一条关于持续奋斗的简短评论表达了声援，简单地写道：*Pain*。
   - 他们补充说希望这些问题能很快得到解决，展现了社区的共情。


  

---


---


---


---


---


{% else %}


> 完整的频道细分内容已为邮件版截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}