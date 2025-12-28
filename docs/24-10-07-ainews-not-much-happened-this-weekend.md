---
companies:
- openai
- meta-ai-fair
- reka
- langchainai
- entropix
date: '2024-10-08T02:36:09.068096Z'
description: '**2024年10月4日至10月7日的AI新闻**重点介绍了以下几项进展：


  **OpenAI的o1-preview**在处理复杂任务时表现强劲，但在简单任务上却显得吃力；与此同时，**Claude 3.5 Sonnet**通过高级提示工程（prompting）技术，能够展现出与其相当的推理能力。**Meta**推出了**Movie
  Gen**，这是一款用于文本生成视频及编辑的前沿多媒体基础模型。**Reka**更新了其21B Flash模型，新增了时间维度视频理解、原生音频支持以及工具调用功能。


  业界对专注于提示词和微调的“开源版o1”复现工作的兴趣日益增长，其中**Entropix**正在探索基于熵的采样技术。**LangChainAI**展示了一个用于复杂问答的检索代理（Retrieval
  Agent），同时一项关于合成数据生成的研究对417个模型进行了综述。**RNN（循环神经网络）**的复兴表明，高效的并行训练正使其具备与Transformer模型一较高下的竞争力。此外，受生物启发的AI安全方法也受到了关注。


  *“一个安静的周末和空调，就是你所需要的一切。”*'
id: 0ae6301c-88bb-40d4-837e-83424c28aa99
models:
- o1-preview
- claude-3.5-sonnet
- 21b-flash-model
original_slug: ainews-not-much-happened-this-weekend-5817
people:
- lex-fridman
- imrat
- jjitsev
- giffmana
- _philschmid
- karpathy
- rasbt
- adcock_brett
- glennko
- rohanpaul_ai
- labenz
title: 这个周末没什么特别的。
topics:
- prompting-techniques
- finetuning
- entropy-based-sampling
- temporal-understanding
- native-audio
- tool-use
- instruction-chaining
- multimodality
- retrieval-augmented-generation
- synthetic-data-generation
- rnn
- parallel-training
- biologically-inspired-ai-safety
- text-to-video-generation
- video-editing
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的周末和[空调](https://x.com/doomie/status/1843380556802994422)就是你所需要的一切。**

> 2024年10月4日至10月7日的 AI News。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**226** 个频道和 **5768** 条消息）。预计节省阅读时间（以 200wpm 计算）：**640 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

有几件值得注意的事，但没有足以登上头条的新闻：

- [Cursor 登上了 Lex Fridman 的访谈](https://www.youtube.com/watch?v=oFfVt3S51T4)，这是该节目首次同时邀请 4 位嘉宾，也是 Lex 报道开发者工具和早期初创公司的一个显著突破。[Imrat 对该播客的 20 点总结](https://x.com/imrat/status/1843368797417418766)非常实用。
- 人们对 "open o1" 的复现表现出浓厚兴趣。诚然，目前还没有基于 RL 的：大多数是 [prompting 技巧](https://www.reddit.com/r/ClaudeAI/comments/1fx51z4/i_made_claude_35_sonnet_to_outperform_openai_o1/) 和 [finetunes](https://www.reddit.com/r/LocalLLaMA/comments/1fxf5n3/introducing_my_reasoning_model_no_tags_just_logic/)，但最有前景的项目可能是 [entropix](https://x.com/scaling01/status/1842930165053276272?s=46)，它使用 [entropy-based sampling](https://notes.haroldbenoit.com/ml/llms/inference/sampling/entropy-based-sampling) 来插入 pause tokens。


![image.png](https://assets.buttondown.email/images/4195a05c-9bd5-4e7a-b35e-13a600b78514.png?w=960&fit=max)


- Reka 更新了他们的 [21B Flash Model](https://x.com/rekaailabs/status/1843298155682820566?s=46)，增加了时间理解（针对视频）和原生音频支持（无需独立的 ASR），以及 [tool use 和 instruction chaining](https://x.com/RekaAILabs/status/1843298161621901713)。
- SWEBench 发布了一个[多模态版本](https://x.com/jyangballin/status/1843285832263979470?s=46)。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型进展与对比**

- **OpenAI 的 o1-preview 性能**：[@JJitsev](https://twitter.com/JJitsev/status/1842960001020883014) 指出 o1-preview 声称在奥林匹克竞赛和博士级任务中表现强劲，但在**更简单的 AIW+ 问题上表现出波动**，表明可能存在泛化缺陷。[@giffmana](https://twitter.com/giffmana/status/1842908836992090449) 观察到 o1-preview **显然处于领先地位**，解决了 2/6 个变体，并在其余变体中获得了约 50% 的分数，而其他模型的得分不足 10%。

- **Claude 3.5 Sonnet 对比 OpenAI o1**：[@_philschmid](https://twitter.com/_philschmid/status/1842846050320544016) 报告称，可以通过提示词引导 Claude 3.5 Sonnet **增加推理时间计算（test-time compute），从而匹配 OpenAI o1 等推理强模型**。该方法结合了 Dynamic Chain of Thoughts、反思（reflection）和口头强化（verbal reinforcement）。

- **LLM 收敛**：[@karpathy](https://twitter.com/karpathy/status/1843005000206909856) 观察到许多 LLM 听起来很相似，都使用列表、讨论“多方面”（multifaceted）问题，并主动提供进一步帮助。[@rasbt](https://twitter.com/rasbt/status/1843005523991663012) 认为这可能是由于**外部公司为偏好微调（preference tuning）提供了数据集**。

- **Movie Gen**：Meta 发布了 Movie Gen，被描述为[“迄今为止最先进的媒体基础模型”](https://twitter.com/adcock_brett/status/1842958865198981619)。它可以根据文本生成高质量的 AI 视频，并进行精确的视频编辑。

**AI 研究与应用**

- **检索增强生成 (RAG)**：[@LangChainAI](https://twitter.com/LangChainAI/status/1843068720937112013) 分享了一个使用 LangGraph 和 Exa 实现的 Retrieval Agent，用于更复杂的问答应用。

- **AI 在客户支持中的应用**：[@glennko](https://twitter.com/glennko/status/1842869624595198098) 报告称构建了端到端的客户服务 Agent，已为一家 F500 客户**自动化处理了 60-70% 的客户支持量**。

- **合成数据生成**：发布了一份关于过去十年中 417 个合成数据生成 (SDG) 模型的[综合综述](https://twitter.com/rohanpaul_ai/status/1843035580109902172)，涵盖了 20 种不同的模型类型和 42 种亚型。

- **RNN 复兴**：一篇[论文](https://twitter.com/rohanpaul_ai/status/1843029138921398536)发现，通过移除隐藏状态依赖，LSTM 和 GRU 可以高效地并行训练，使其在长序列任务中能与 Transformer 和 Mamba 竞争。

**AI 安全与伦理**

- **生物启发式 AI 安全**：[@labenz](https://twitter.com/labenz/status/1842952941332033992) 强调了 AE Studio 在生物启发方法方面的工作，旨在设计更具协作性且更少欺骗性的 AI 系统，包括训练模型预测其内部状态，并最小化自我与他者的区别。

- **AI 风险辩论**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1842961674892169567) 讨论了 AI 风险辩论中的两极分化，指出怀疑论者通常回避不确定性下的成本效益推理，而许多末日论者（doomers）则过于贝叶斯化（Bayesian）。

**行业新闻与进展**

- **OpenAI 融资**：OpenAI [完成了新一轮 66 亿美元融资](https://twitter.com/adcock_brett/status/1842958965262422448)，公司估值达到 1570 亿美元，巩固了其作为全球资金最雄厚的 AI 初创公司的地位。

- **Cloudflare SQLite 改进**：[@swyx](https://twitter.com/swyx/status/1843039888222134615) 强调了 Cloudflare 对 SQLite 的改进，包括具有异步性能的同步查询，以及将状态回滚到过去 30 天内任意时间点的能力。

**梗与幽默**

- [@ylecun](https://twitter.com/ylecun/status/1843016587244401035) 对一条未指明的推文回复了 "Haha 😄"。

- [@bindureddy](https://twitter.com/bindureddy/status/1843041274347290683) 拿 Elon Musk 因政治观点而遭受仇恨的讽刺现象开玩笑，尽管其初衷是停止仇恨并传播快乐。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. 小规模 LLM 性能的进展**

- **[Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation](https://arxiv.org/abs/2410.02725)** ([Score: 66, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fxjmn5/adaptive_inferencetime_compute_llms_can_predict/)): **Adaptive Inference-Time Compute**（自适应推理时间计算）允许 **Large Language Models (LLMs)** 在生成过程中动态调整其计算资源，从而可能提高输出质量。该方法涉及模型预测额外的计算是否会增强其性能（即使是在生成中期），并据此进行调整。这项技术可以使 LLMs 更加高效、有效地利用计算资源，潜在地提升其整体性能和适应性。
  - [{'id': 'lqn3n3c', 'score': 8, 'body': "这是一篇如果配有代码会好得多的论文。不需要太极端，只要一个基础实现和良好的文档，我就能想办法把它挂载到我喜欢的推理引擎上。有趣的是，过去一周我看到的优质研究论文比整个夏天都多。我不知道是 o1 的发布促使研究人员拿出了他们的干货，还是仅仅是一个周期性的现象。", 'author': 'XMasterrrr', 'is_submitter': False, 'replies': [{'id': 'lqn4dmw', 'score': 2, 'body': "是的，我最近看到了很多好论文。最近大家非常关注 CoT 和推理。我希望有人能根据这个拼凑出可用的代码，看起来非常有趣。", 'author': 'Thrumpwart', 'is_submitter': True, 'replies': []}, {'id': 'lqni1zg', 'score': 2, 'body': '>我们发布了一个 [公开的 GitHub 实现](https://github.com/rohinmanvi/Capability-Aware_and_Mid-Generation_Self-Evaluations) 以供复现。就在附录的最上方... GitHub 目前是空的，但有可分享的代码，他们计划发布。如果你真的很在意，也许可以开个 issue 问问预计发布时间（ETA）是什么时候。', 'author': 'Chelono', 'is_submitter': False, 'replies': []}]}]


- **[3B Qwen2.5 finetune beats Llama3.1-8B on Leaderboard](https://huggingface.co/qnguyen3/raspberry-3B)** ([Score: 69, Comments: 11](https://reddit.com//r/LocalLLaMA/comments/1fxraoy/3b_qwen25_finetune_beats_llama318b_on_leaderboard/)): 一个基于 **Arcee.ai** 的 **EvolKit** 创建的挑战性问题进行微调的 **Qwen2.5-3B** 模型，在 Leaderboard v2 评估中超越了 **Llama3.1-8B**，在 BBH 上获得了 **0.4223** 分，在 GPQA 上获得了 **0.2710** 分，六项基准测试的平均分为 **0.2979**。该模型可在 [Hugging Face Spaces](https://huggingface.co/spaces/qnguyen3/raspberry-3b) 上进行测试，但创作者提醒，由于其专门的训练数据和 **qwen-research license**，它可能尚未达到生产级要求。

**Theme 2. 开源社区复现 o1 推理能力的努力**

- **It's not o1, it's just CoT** ([Score: 95, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1fxof45/its_not_o1_its_just_cot/)): 该帖子批评了开源社区尝试复现 **OpenAI** 的 **Q*/Strawberry**（也称为 **o1**）的行为，认为许多尝试只是简单的 **Chain of Thought (CoT)** 实现，而非真正的 o1 能力。作者指出，**Q*/Strawberry** 可能涉及标准 **RLHF** 之外的 **Reinforcement Learning** 技术，并敦促开源社区专注于开发真正的 o1 能力，而不是将 CoT 嵌入到现有的 **Large Language Models (LLMs)** 中。为了说明区别，帖子引用了 [OpenAI 官方博客文章](https://openai.com/index/learning-to-reason-with-llms/#chain-of-thought)，展示了原始的隐藏推理链，特别是强调了 "Cipher" 示例，以证明 o1 与传统 CoT 相比截然不同的方法。

- **[在现有模型基础上复现 o1 推理的新尝试](https://www.reddit.com/r/ClaudeAI/s/rjrBmSmWcM)** ([评分: 81, 评论: 58](https://reddit.com//r/LocalLLaMA/comments/1fxj93m/a_new_attempt_to_reproduce_the_o1_reasoning_on/))：一项新尝试旨在现有语言模型上复现 **o1 reasoning**，重点是在无需重新训练的情况下增强模型能力。该方法涉及开发一种**专门的 Prompt**，引导模型生成更具结构化和逻辑性的输出，从而可能提高其在复杂推理任务中的表现。这种方法可能提供一种利用当前 AI 模型进行高级推理的途径，而无需支付训练新架构的计算成本。
  - 用户讨论了在本地复现 **o1 reasoning** 的可行性，一些人认为这不仅仅需要一个训练良好的 **LLM**。讨论强调了实现类似功能和速度需要**多次 AI 调用**以及显著的**技术改进**。
  - 一位用户提议了一个测试：计算 "strawberry" 中字母 'R' 的数量，并指出 **70B 模型**通常会诉诸于拼读单词。这表明**大型模型**中出现了一种新兴特征，即尽管不“认识”单个字母，它们也能进行拼写和计数。
  - 讨论对该帖子的观点进行了批评，一位用户认为这更多是在现有模型上复现**“只是 CoT，而非 o1”**。其他人则幽默地将这种尝试比作业余火箭科学，对该方案的可行性表示怀疑。
- **介绍我的推理模型：无标签，纯逻辑** ([评分: 322, 评论: 100](https://reddit.com//r/LocalLLaMA/comments/1fxf5n3/introducing_my_reasoning_model_no_tags_just_logic/))：该帖子介绍了一个受 **o1 系统**启发的**推理模型**，它在用户输入和助手输出之间增加了一个中间推理步骤。作者使用来自 [Reasoning-base-20k](https://huggingface.co/datasets/KingNish/reasoning-base-20k) 集合的 **10,000 列数据集**训练了两个模型：**Reasoning Llama 3.2 1b-v0.1** 和 **Reasoning Qwen2.5 0.5b v0.1**。两个模型均已在 HuggingFace 上发布，帖子中提供了链接。
  - 该模型被描述为 **CoT (Chain of Thought)** 而非 **o1**，用户指出 o1 的推理链显著更长（**5400 Llama3 tokens** 对比 1000），并且涉及**蒙特卡洛树搜索算法 (tree-search monte carlo algorithm)**。
  - 一位用户根据泄露的 o1 信息实现了一个 **16 步推理流水线**，并使用 **Gemini 8B Flash** 进行了测试。该实现提高了代码生成结果，但每条响应耗时约 **2 分钟**。[Colab 链接](https://colab.research.google.com/drive/1Sj7btrr2yexUk1xn97O3P6ZoHWyV0laB?usp=sharing)已提供。
  - 用户请求并获得了模型的 **GGUF 版本**。人们有兴趣将此方法应用于更大的模型，如 **Qwen 2.5 72b** 或 **32B**，一些人建议针对基座模型进行基准测试以评估改进效果。


**主题 3. 用于本地 LLM 推理的 DIY AI 硬件**

- **[搭建了我的首台 AI + 视频处理工作站 - 3x 4090](https://i.redd.it/r8332mez28td1.png)** ([评分: 378, 评论: 79](https://reddit.com//r/LocalLLaMA/comments/1fxu8rt/built_my_first_ai_video_processing_workstation_3x/))：该帖子描述了一台高性能的 **AI 和视频处理工作站**，配置包括 **Threadripper 3960X** CPU、**3x NVIDIA RTX 4090 GPU**（两块 Suprim Liquid X 和一块 Founders Edition）以及 **128GB DDR4 RAM**，采用 **NZXT H9 Flow** 机箱和 **1600W PSU**。该系统旨在离线运行处理敏感数据的 **Llama 3.2 70B** 模型（带有 **3万-4万词的 Prompt**），实现了 **10 tokens/秒** 的吞吐量，在使用 **Ollama** 和 **AnythingLLM** 时 Prompt 评估速度表现出色，同时还能通过 **Topaz Video AI** 进行视频超分辨率和 AI 增强。

- **AMD Instinct Mi60** ([评分: 31, 评论: 32](https://reddit.com//r/LocalLLaMA/comments/1fxn8xf/amd_instinct_mi60/))：在 eBay 上以 **299 美元**购买的 **AMD Instinct Mi60** GPU，拥有 **32GB HBM2** 显存和 **1TB/s** 带宽，可在 **Ubuntu 24.04**、**AMDGPU-pro 驱动**和 **ROCm 6.2** 环境下工作。使用 **Llama-bench** 的基准测试显示，Mi60 运行 **qwen2.5-32b-instruct-q6_k** 的速度为 pp512 达到 **11.42 ± 2.75 t/s**，tg128 达到 **4.79 ± 0.36 t/s**；而 **llama3.1 8b - Q8** 的速度为 pp512 达到 **233.25 ± 0.23 t/s**，tg128 达到 **35.44 ± 0.08 t/s**，性能受限于 **100W TDP**。


**主题 5. 多模态 AI：结合视觉与语言**

- **[Qwen 2 VL 7B Sydney - 喜欢评论你狗狗照片的视觉模型](https://huggingface.co/adamo1139/Qwen2-VL-7B-Sydney)** ([Score: 32, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fxhntw/qwen_2_vl_7b_sydney_vision_model_that_will_love/)): **Qwen 2 VL 7B Sydney** 是一款全新的**视觉语言模型 (Vision Language Model)**，旨在对图像提供详细的评论，尤其擅长描述狗狗的照片。该模型由 **Alibaba** 开发，能够生成长达数段的图像描述，与传统的图像标注模型相比，其输出内容更加详尽。
  - 用户对将**视觉语言模型**与**经过角色扮演微调的 LLM** 合并以增强图像交互表现出了浓厚兴趣。同时，也有人担心大公司会限制此类模型的访问，并以 **Chameleon** 为例。
  - 该模型的创建者分享了使用 **Sydney 的人格 (personality)** 对 **Qwen 2 VL 7B** 进行微调的计划，旨在创建一个更加积极且具有互动性的多模态模型。该项目涉及 **42M tokens** 的文本和图像数据，所有资源均已开源。
  - 讨论还涉及了该模型与 **LM Studio** 的兼容性，由于 **llama.cpp** 尚不支持 **Qwen 2 VL 7B**，目前不太可能兼容。创建者提供了一个推理脚本，并指出为了获得最佳性能，需要配备 **24GB VRAM GPU**。

## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

待完成

---

# AI Discord 摘要

> 摘要的摘要之摘要

## Claude 3.5 Sonnet


**1. AI 模型发布与基准测试**

- **DeepSeek V2 挑战 GPT-4**：**DeepSeek-V2** 已发布，声称在 **AlignBench** 和 **MT-Bench** 等基准测试的某些领域超越了 **GPT-4**。
   - 该模型的表现引发了 [Twitter](https://x.com/deepseek_ai/status/1787478986731429933) 上的讨论，一些人对相比现有模型的改进程度持怀疑态度。
- **Dracarys 2 作为顶级开源编程模型亮相**：[Dracarys 2](https://x.com/bindureddy/status/1842611268148203883) 被介绍为一款强大的开源编程模型，在 LiveCodeBench 等基准测试中表现优于 **Sonnet 3.5**。
   - 虽然在代码编辑任务中达到了 **67%** 的准确率，但一些用户认为它更多是现有模型的重新包装，而非能力的重大创新。
- **Open O1 挑战闭源模型**：[Open O1 项目](https://opensource-o1.github.io/) 旨在创建一个开源模型，在推理、编程和数学解题方面达到 OpenAI o1 的性能水平。
   - 然而，一些社区成员认为围绕 **Open O1** 的讨论缺乏深度，呼吁对此类模型及其声称的能力进行更严格的审查。
  


**2. AI Agent 与推理能力的进展**

- **SwiftSage v2 增强推理能力**：[SwiftSage v2](https://github.com/SwiftSage/SwiftSage) 的发布引入了一个集成快慢思考的推理 Agent 系统，专注于复杂问题的 **In-context learning**。
   - 该开源项目旨在数学和 MMLU 风格的推理任务中与闭源系统竞争，展示了在各种认知挑战中的优势。
- **GenRM 彻底改变奖励模型**：**GenRM** 的引入使得奖励模型可以作为 **Next-token predictor** 而非传统的分类器进行训练，从而为奖励模型开启了 **Chain-of-Thought 推理**。
   - 这一创新提供了统一的策略和奖励模型，增强了各种任务的整体表现，并可能改善 AI 与人类价值观的对齐。
- **用于连续潜空间推理的 COCONUT 范式**：一篇[新论文](https://openreview.net/forum?id=tG4SgayTtk)介绍了 COCONUT，这是一种允许语言模型在**连续潜空间 (Continuous Latent Space)** 而非传统语言空间中进行推理的范式。
   - 这种方法表明，利用隐藏状态 (Hidden states) 进行推理可以缓解传统模型中 Token 的限制，从而实现更复杂的思考并可能增强 LLM 的能力。
  


**3. AI 工具与基础设施改进**

- **Mojo 基准测试框架发布**：Mojo 引入了一个用于运行时性能评估的 [benchmark package](https://docs.modular.com/mojo/stdlib/benchmark/)，类似于 Go 的测试框架。
   - 用户现在可以使用 `benchmark.run` 高效评估函数性能，并报告平均耗时和迭代次数，从而增强 Mojo 生态系统中的开发工作流。
- **LlamaIndex RAG-a-thon 宣布举行**：**LlamaIndex Agentic RAG-a-thon** 定于 **10 月 11 日至 13 日**在硅谷举行，重点关注与 **Pinecone** 和 **VESSL AI** 合作的 Retrieval-Augmented Generation 技术。
   - 该活动旨在推进企业级应用的 **AI agents**，开发者有机会赢得现金奖励，详见[此链接](https://rag-a-thon-2.devpost.com/)。
- **Entropix 增强 Prompt 优化**：**Entropix/Entropy Guided Adaptive Sampler** 增强了 Prompt 优化，专注于 attention entropy 以提升模型性能。
   - 正如 @_xjdr 在社交媒体上所述，其优势包括提高叙事连贯性和减少幻觉，甚至在小型模型中也展现出了这种能力。
  


**4. 开源 AI 项目与协作**

- **Meta Movie Gen 研究论文发布**：Meta 发布了一篇[研究论文](https://ai.meta.com/static-resource/movie-gen-research-paper)，详细介绍了他们在电影生成建模方面的 **Movie Gen** 创新。
   - 该文档是理解 Meta 在电影生成技术进步背后方法论的重要参考，提供了对其最新 AI 驱动创意工具的见解。
- **Python 3.13 发布带来重大更新**：Python 3.13 正式发布，包含重大更新，包括[更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) 以及无需 GIL 运行 Python 的选项。
   - 亮点功能还包括对 **iOS** 和 **Android** 平台的改进支持，由于 Beeware 项目的发展，这些平台被标记为 **Tier 3 支持**。
- **Intel 与 Inflection AI 合作开展企业级 AI**：**Intel** 与 **Inflection AI** 宣布合作推出企业级 AI 系统，标志着企业级 AI 领域的重大进展。
   - 这一合作伙伴关系暗示了企业环境中技术使用方式的潜在重塑，尽管初步公告中未提供系统功能的具体细节。

## GPT4O (gpt-4o-2024-05-13)


**1. LLM 进展**

- **Qwen 模型媲美 LLaMA**：关于 **Qwen 2.5 7B** 模型的讨论显示，它们在对话任务中的表现与 **LLaMA** 模型相当，并指出在训练效率方面存在显著差异。
  - 针对这些模型之间切换性能的担忧被提出，暗示了在微调策略（fine-tuning strategies）中存在优化的潜力。
- **Llama 3.2 模型加载问题**：用户在 **LM Studio** 中加载模型时面临挑战，特别是在处理 'gguf' 格式时，出现了与 AVX2 等过时 CPU 指令相关的错误。
  - 建议包括升级硬件或切换到 Linux，突显了对更好兼容性解决方案的需求。


**2. 模型性能优化**

- **DALI Dataloader 展示了惊人的吞吐量**：**DALI Dataloader** 实现了每秒读取 **5,000 张 512x512 JPEG 图像**，展示了在大尺寸图像变换中对 GPU 资源的有效利用。
  - 成员们注意到即使在全量 **ImageNet transforms** 下其性能依然稳定，强调了其高效性。
- **优化 Onnxruntime Web 体积**：讨论集中在如何通过使用精简版本，将 **Onnxruntime Web** 的默认 WASM 大小从 **20 MB** 减少到更易于管理的 **444K**。
  - 成员们探索了如 LTO 和 tree shaking 等策略，以便在合并自定义推理逻辑的同时进一步优化包体积。
- **使用 CUDA 并行化 RNN**：讨论了使用 **CUDA 并行化 RNN** 的挑战，并引用了如 S4 和 Mamba 等创新解决方案。
  - 社区对克服顺序依赖关系表现出浓厚兴趣，突显了该领域正在进行的深入研究。


**3. 多模态 AI 创新**

- **Reka Flash 更新增强了多模态能力**：最新的 **Reka Flash** 更新现在支持**文本、图像、视频和音频**的交错多模态输入，显著提升了功能性。
  - 这一增强突显了在**多模态理解（multimodal understanding）**和实际应用方面的进展。
- **探索 Luma AI 的魔力**：讨论集中在 **Luma AI** 及其令人印象深刻的**视频应用**上，特别是它在**电影剪辑**和创建独特摄像机运动方面的实用性。
  - 成员们分享了资源和案例，强调了该工具在创意领域的潜力。


**4. 开源 AI 框架**

- **OpenRouter 与 Fal.ai 合作**：**OpenRouter** 已与 **Fal.ai** 建立合作伙伴关系，通过[此链接](https://x.com/isidentical/status/1842650721969459561)增强了 Fal 图像工作流中的 **LLM** 和 **VLM** 能力。
  - 此次集成允许用户利用先进的 AI 模型来改进图像处理任务。
- **API4AI 助力 AI 集成**：**API4AI** 平台促进了与 **OpenAI** 和 **Azure** 等服务的轻松集成，提供了多样化的**现实世界交互** API。
  - 这些功能使开发者能够构建强大的 AI 应用，提升功能和用户体验。


**5. 微调挑战**

- **微调 LLaMA 的挑战**：用户注意到 **LLaMA 3.1** 在训练后产生无尽输出的问题，标志着微调过程中的挑战。
  - 讨论强调了使用正确的聊天模板（chat templates）和序列结束（end-of-sequence）定义对于改善模型行为的必要性。
- **在模型微调中使用 LoRA**：在微调中使用 **LoRA** 的可行性引发了辩论，一些人认为全量微调（full fine-tuning）总体上可能会产生更好的结果。
  - 关于 LoRA 有效实施的不同观点浮出水面，突显了其在处理已经微调过的模型时的局限性。

## GPT4O-Aug (gpt-4o-2024-08-06)


**1. 模型微调与优化**

- **LLaMA 模型微调中的挑战**：Discord 上的用户报告了在微调 **LLaMA 3.1** 等模型时遇到的问题，如生成输出无止尽，并强调了正确使用聊天模板（chat templates）和序列结束（end-of-sequence）定义的重要性。讨论重点关注了 **LoRA** 作为微调策略的重要性，并对其与全量微调（full fine-tuning）的效果进行了辩论。
  - 社区分享了克服这些挑战的策略，例如合并数据集以获得更好的结果，以及利用 **LoRA** 进行高效微调。
- **量化与内存优化**：**NF4** 训练等技术被指出可以将 VRAM 需求从 **16G 降低到 10G**，提供显著的性能提升。社区讨论还涵盖了优化 **Onnxruntime Web** 大小以及测试期间 **CUDA** 内存管理的策略。
  - 成员们庆祝通过 NF4 将速度从 **每步 11 秒** 提升到 **每步 7 秒**，强调了这些优化对模型性能的益处。


**2. AI 模型集成与应用**

- **OpenRouter 增强图像工作流**：**OpenRouter** 与 **Fal.ai** 集成，增强了图像工作流中的 LLM 和 VLM 能力，允许用户使用 **Gemini** 简化任务。
  - 此次集成有望为用户提高效率和产出，鼓励他们利用新功能重新思考流程。
- **Companion Discord 机器人彻底改变参与方式**：由 Cohere 驱动的 **Companion** 机器人引入了动态角色建模和审核功能，旨在提升 Discord 社区内的用户互动。
  - 该项目邀请大家进行探索，因为它加强了审核效率并增强了社区讨论。


**3. AI 研究与开发**

- **Meta 发布 Movie Gen 研究论文**：Meta 关于 [Movie Gen 的研究论文](https://ai.meta.com/static-resource/movie-gen-research-paper) 深入介绍了他们在电影生成建模方面的进展，突出了创新的方法论。
  - 该文档是理解 Meta 在电影生成技术进步背后方法论的重要参考。
- **Entropix 采样器能力探索**：**Entropix/Entropy Guided Adaptive Sampler** 通过优化注意力熵（attention entropy），展示了在提升模型性能、减少幻觉和增强叙事一致性方面的改进。
  - 该项目即使在小模型中也显示出可喜的结果，表明其在提高叙事连贯性方面具有显著能力。


**4. AI 工具与框架**

- **Sci Scope 提供个性化 AI 研究摘要**：[Sci Scope](https://sci-scope.com) 每周汇总并总结新的 ArXiv 论文，提供个性化时事通讯，让研究人员及时了解关键进展。
  - 订阅者受益于与其兴趣相关的定制论文列表，确保他们不会错过 AI 研究中的重要进展。
- **Aider v0.59.0 发布带来增强功能**：新的 **Aider v0.59.0** 版本包括对 `/read-only` 命令的更新、YAML 配置格式更改，以及通过新的完整性检查（sanity checks）实现的性能改进。
  - 这些更新通过在编码任务期间更好的处理提高了性能，并引入了新的完整性检查以简化启动流程。


**5. AI 社区与支持**

- **LlamaIndex RAG-a-thon 开幕**：**LlamaIndex Agentic RAG-a-thon** 活动与 **Pinecone** 和 **VESSL AI** 合作，专注于推进检索增强生成（Retrieval-Augmented Generation）技术。
  - 该活动旨在推进企业级应用的 **AI agents**，开发者有机会赢取现金奖励。
- **AI 工具的社区支持**：Discord 社区强调了协作和反馈的重要性，用户寻求 **Cohere** 和 **OpenRouter** 等工具的支持以改进其 AI 工作流。
  - 鼓励成员联系以获取支持并分享反馈，强调了社区驱动改进的重要性。

## O1-mini

**主题 1. 模型微调与训练挑战**

- [**通过 Unsloth Studio 克服微调瓶颈**](https://github.com/unslothai/unsloth)：即将推出的 **Unsloth Studio** GUI 通过自动化数据集格式化和依赖管理来简化微调过程，目标用户是缺乏高级编程技能的初学者。
  - 用户报告了 **LLaMA 3.1** 等模型在训练后生成无止尽输出的挑战，强调了需要适当的聊天模板和序列结束定义以获得更好的行为。

- [**LoRA 在模型精炼中的局限性引发辩论**](https://github.com/seanchatmangpt/dslmodel)：**LoRA** 在模型微调中的可行性引发了争论，一些人主张通过全量微调（full fine-tuning）以获得更优结果，而另一些人则强调了 LoRA 在处理已微调模型时的约束。
  - 关于如何有效实施 LoRA 出现了不同的观点，展示了其局限性以及社区对更好微调优化技术的追求。

- [**梯度检查点（Gradient Checkpointing）增强 TinyGrad 训练效率**](https://github.com/tinygrad/tinygrad/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426)：实现**梯度检查点（gradient checkpointing）**对于在 **TinyGrad** 中高效训练大型模型至关重要，使其能够处理超出小型玩具模型规模的参数。
  - 如果没有这些优化，TinyGrad 中的模型在进行大规模训练时会非常吃力，从而限制了它们的实际应用。

**主题 2. 新模型发布与性能对比**

- [**Qwen 2.5 在对话任务中与 LLaMA 旗鼓相当**](https://huggingface.co/collections/unsloth/qwen-25-66fe4c08fb9ada518e8a0d3f)：讨论显示 **Qwen 2.5 7B** 模型在对话任务中的表现与 **LLaMA** 相似，并对其训练效率和潜在的性能切换进行了辩论。
  - 用户报告了微调能力方面的显著差异，表明 Qwen 是未来模型优化的一个可行替代方案。

- [**Dracarys 2 在代码基准测试中超越 Sonnet 3.5**](https://x.com/bindureddy/status/1842611268148203883)：新发布的 **Dracarys 2** 模型在 LiveCodeBench 等性能基准测试中超过了 **Sonnet 3.5**，在代码编辑任务中达到了 **67%** 的准确率。
  - 尽管其初始声明令人印象深刻，但一些用户质疑其创新性，认为它只是现有模型的重新包装，而非突破性的进展。

- [**Phi-3.5 模型因安全特性面临社区抵制**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**Microsoft** 的 **Phi-3.5** 模型因其过度的审查设计而遭到社区的嘲讽，导致 **Hugging Face** 上出现了[无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)的分享。
  - 用户通过讽刺性的回应，表达了对该模型因过度内容限制而在技术任务中缺乏实用性的担忧。

**主题 3. 集成、工具与部署**

- [**Unsloth Studio 简化 AI 模型训练**](https://github.com/unslothai/unsloth)：**Unsloth Studio** GUI 的推出旨在降低 AI 模型微调的难度，通过自动处理数据集格式化和依赖管理，特别照顾到没有深厚编程知识的初学者。
  - 用户强调了它在减轻常见微调问题方面的潜力，从而增强了更广泛用户群体的可访问性。

- [**RYFAI 应用推动私有 AI 访问**](https://github.com/open-webui/open-webui)：开源的 **RYFAI** 应用强调离线运行和用户隐私，旨在为 **Ollama** 和 **OpenWebUI** 等成熟的 AI 工具提供具有竞争力的替代方案。
  - 讨论中涉及了对市场饱和度和差异化策略的担忧，用户对其与更成熟解决方案的竞争能力持辩论态度。

- [**TorchAO 期待 NF4 支持以优化 VRAM**](https://github.com/pytorch/torchao/blob/main/torchtune/modules/low_precision/nf4_linear.py)：社区热切期待 **TorchAO** 中 **NF4** 的实现，这可以将 **VRAM** 需求从 **16G 降低到 10G**，并将训练速度从**每步 11 秒提高到 7 秒**。
  - 成员们对这些预期的性能增强表示赞赏，认为它们是高效模型微调和资源管理的变革者。

**主题 4. API 问题、成本与支持**

- [**Cohere API 错误导致项目中断**](https://docs.cohere.com/docs/cohere-faqs#billing-pricing-licensing-account-management)：用户在模型微调过程中频繁遇到 **Cohere API 错误**（如 'InternalServerError'），导致项目进度严重受阻。
  - 管理员承认由于错误积压过多，正在优先处理支持工单，并敦促受影响的用户在解决方案实施期间保持耐心。

- [**大规模媒体分析导致 OpenAI API 成本上升**](https://platform.openai.com/docs/guides/structured-outputs/introduction)：使用 **OpenAI API** 分析数千个媒体文件可能会超过 **12,000 美元**，这引发了关于本地解决方案可行性的讨论，尽管本地方案也伴随着高昂的存储和处理成本。
  - 用户询问潜在的具有成本效益的替代方案，在云端 API 的便利性与项目预算的财务挑战之间进行权衡。

- [**OpenRouter API 持续出现重复生成问题**](https://x.com/isidentical/status/1842650721969459561)：用户报告在使用 **OpenRouter API** 时持续出现重复生成的响应，这表明存在特定于设置的问题，而一些用户在调整响应解析器后遇到了 **404 错误**。
  - 故障排除建议包括审查 API 设置配置并优化响应解析器，以减轻重复响应问题。

**主题 5. 数据流水线与合成数据的使用**

- [**Canvas 项目中合成数据增强了模型训练**](https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/class_generation.ipynb)：Canvas 项目利用合成数据生成技术（例如蒸馏来自 **OpenAI o1-preview** 的输出）来微调 **GPT-4o**，从而实现 AI 模型能力的快速提升。
  - 这种方法允许在不需要大量人工生成数据集的情况下实现可扩展的模型改进，展示了数据处理方面的效率和创新。

- [**SWE-bench Multimodal 评估视觉问题解决能力**](https://sci-scope.com)：新推出的 **SWE-bench Multimodal** 从 **17 个 JavaScript** 仓库中引入了 **617 个新任务**，以评估 AI Agent 解决视觉 GitHub 问题的能力，解决了目前 Agent 性能的局限性。
  - 这一全面的基准测试旨在提高 AI 模型在真实编码环境中的多模态理解和实际问题解决能力。

- [**Entropix 采样器警告不要过度使用合成数据**](https://github.com/xjdr-alt/entropix)：**Entropix/Entropy Guided Adaptive Sampler** 警告不要过度使用来自 AI 输出的合成数据，以防止模型过拟合，同时也承认其在早期训练阶段的有效性。
  - 用户正在探索替代的数据生成方法，重点是通过平衡的数据集策略来维持模型的可靠性和性能。

## O1-preview

**主题 1：微调与模型训练中的创新与工具**

- [**Unsloth GUI 让初学者的微调变得轻而易举**](https://docs.unsloth.ai/get-started/unsloth-notebooks)：即将推出的 **'Unsloth Studio' GUI** 旨在通过自动处理数据集格式化和依赖关系来简化微调。这一创新针对的是那些在没有高级编程技能的情况下进行模型训练时面临挑战的初学者。
- [**Torchtune 听取建议：KTO 训练支持请求**](https://github.com/pytorch/torchtune/issues/1730)：用户渴望在 **Torchtune** 中获得 **KTO 训练** 支持，建议将其添加到 DPO 配方中。开发人员建议提交一个 issue 来跟踪此功能请求。
- [**TinyGrad 通过梯度检查点加速训练**](https://github.com/tinygrad/tinygrad)：讨论强调了 **tinygrad** 中 **梯度检查点 (gradient checkpointing)** 对于高效训练大型模型的重要性。如果没有这些优化，tinygrad 只能处理 *“非常小的玩具模型”*，从而限制了其整体性能。

**主题 2：新 AI 模型及其能力**

- [**OpenAI 的 o1 模型声称思考方式不同，引发质疑**](https://openai.com/o1/)：关于 **OpenAI o1** 将推理直接集成到模型中的争论不断，一些人称其为 *“简化”* 并质疑其真实能力。怀疑论者强调，潜在的挑战可能尚未得到充分解决。
- [**Dracarys 2 吐火，占据顶尖编码模型宝座**](https://x.com/bindureddy/status/1842611268148203883)：**Dracarys 2** 宣布自己是世界上最好的开源编码模型，在 LiveCodeBench 上以 **67%** 的得分超越了 **Sonnet 3.5**。批评者认为它是现有模型的翻版，而非真正的创新。
- [**Meta 发布重磅消息：Movie Gen 研究论文发布**](https://ai.meta.com/static-resource/movie-gen-research-paper)：**Meta** 分享了他们的 **Movie Gen 研究论文**，详细介绍了生成式电影建模方面的进展。该文档对于理解 Meta 在电影生成技术创新背后的方法论至关重要。

**主题 3：AI 辅助工具与应用的增强**

- [**Agent 群体自动创建 YouTube 视频，接管内容创作**](https://t.co/TKs9QqP4ym)：该项目展示了如何使用 **LlamaIndex** 构建一个 Agent “集群”，根据自然语言提示词自主创建 AI 生成的 YouTube 视频。这种方法突显了 **multi-agent architectures** 在简化视频生成工作流方面的潜力。
- [**Cursor 团队编写未来，与 Lex Fridman 对话**](https://x.com/lexfridman/status/1843010390772605183)：**Cursor team** 在与 **Lex Fridman** 的对话中讨论了 AI 辅助编程和编码的未来，展示了他们的创新环境。话题涵盖了 **GitHub Copilot** 以及 AI 集成到编码工作流中的复杂性。
- [**Companion Discord 机器人通过 Cohere 集成结交朋友**](https://github.com/rapmd73/Companion)：新的 **Companion bot** 利用 **Cohere** 增强动态人格建模和用户交互，同时为 Discord 服务器提供集成的审核工具。这加强了 Discord 内部的社区参与度和审核效率。

**主题 4：AI 社区应对平台与 API 故障**

- [**Cohere 用户因 API 错误和 429 问题抓狂**](https://cohere.com/)：沮丧的用户报告了 **Cohere API** 持续出现的 **'InternalServerError'** 和 **429 errors**，影响了他们的项目和测试。版主确认由于积压严重，正在优先处理支持工单。
- [**Perplexity AI 削减 Opus 限制，用户因消息次数减少而抗议**](https://www.perplexity.ai)：随着 **Perplexity AI** 将 **Opus** 消息限制减少到 **每天 10 条**，引发了用户愤怒；用户的强烈抵制显然导致其恢复到了 **50 条消息**。用户对突然的变化表示不满，并提出了关于消费者权益的问题。
- [**Aider 陷入困境，用户抱怨性能缓慢**](https://aider.chat/HISTORY.html)：用户在使用 **Sonnet 3.5** API 时，尤其是在处理大文件时，遇到了 **Aider** 的显著延迟。建议包括限制上下文文件和使用 verbose 标志，许多人正在寻求 **OpenRouter** 等替代方案进行 API 管理。

**主题 5：AI 研究与理论探索的进展**

- [**基于熵的采样有望实现更智能的 AI Prompt**](https://github.com/xjdr-alt/entropix)：**Entropix project** 引入了 **Entropy Guided Adaptive Sampling**，通过评估注意力熵来增强 Prompt 优化。其优势包括提高叙事一致性和减少幻觉，表明即使在小模型中也具备这种能力。
- [**GenRM 融合策略与奖励模型以实现更好的 AI 对齐**](https://x.com/lunjunzhang/status/1829296204171100418)：**Generative Reward Models (GenRM)** 的引入将奖励模型训练为 next-token 预测器，改善了 **Chain-of-Thought reasoning** 以及与人类价值观的对齐。该方法旨在提升决策中的推理能力。
- [**RWKV 系列版本更迭让研究人员眼花缭乱**](https://arxiv.org/abs/2410.01201)：社区成员难以追踪 **RWKV** 各个版本的变化；一篇 [论文](https://arxiv.org/abs/2410.01201) 提供了逐步概述以帮助理解。这突显了在快速演进的模型中需要更清晰的文档说明。


---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth GUI 简化微调**：即将推出的 “Unsloth Studio” GUI 旨在通过自动管理数据集格式化和依赖项来简化微调流程。
   - 这一创新针对的是那些缺乏高级编程技能、在模型训练中面临挑战的初学者。
- **Qwen 模型媲美 LLaMA**：讨论强调 **Qwen 2.5 7B** 模型在对话任务中的表现与 **LLaMA** 模型相似，用户报告了在训练效率方面的显著差异。
   - 提出了关于在两个模型之间切换性能的担忧，暗示了微调优化的潜在途径。
- **LLaMA 微调中的挑战**：用户注意到 **LLaMA 3.1** 在训练后会出现无尽生成输出的问题，这标志着微调过程中的挑战。
   - 讨论集中在为了改善模型行为而使用正确的聊天模板和序列结束（end-of-sequence）定义的必要性。
- **在模型微调中使用 LoRA**：在微调中使用 **LoRA** 的可行性引发了辩论，一些人认为全量微调（full fine-tuning）可能会产生更好的整体结果。
   - 关于如何有效实施 **LoRA** 的不同意见浮出水面，强调了其在处理已经微调过的模型时的局限性。
- **RYFAI 应用带来私有 AI 访问**：**RYFAI** 的推出（一款适用于各种操作系统的开源应用）强调了用户隐私和离线运行。
   - 对其与成熟工具竞争的能力提出了担忧，并讨论了市场饱和度和差异化问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **关于 AGI 和 AI 推理的辩论**：一场关于 **AGI 可实现性** 的讨论展开，强调其与类似于人类大脑功能的概率结构之间的关系。
   - 参与者强调了 **LLM** 与人类思维过程中推理的不同解释。
- **Hugging Face 模型与内存限制**：用户询问了 Hugging Face 上如 **Llama 3.1** 等模型的上下文窗口（context windows），并分享了在高内存配置下的经验。
   - 对在云平台上运行高上下文模型相关成本的担忧非常普遍。
- **模型微调的挑战**：用户报告了在微调模型方面的困扰，特别是注意到 **DETR 模型** 在边界框（bounding boxes）方面的不准确性，并提供了进一步的上下文链接。
   - 这些不准确性引发了关于针对特定任务优化性能的讨论。
- **合成数据的探索**：对话涉及了使用**合成数据**的影响，警告尽管初始性能有所提高，但仍存在潜在的过拟合风险。
   - 参与者表达了学习替代数据生成方法以优化模型训练的共同兴趣。
- **持续的服务中断更新**：10 月 6 日报告了影响 **Share API** 和 **Share Links** 的服务中断，用户被引导至 [状态页面](https://status.gradio.app/) 获取更新。
   - 幸运的是，很快宣布所有受影响的系统已恢复在线，缓解了用户的使用中断。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LLM 训练器咨询引发关注**：一名成员表示想花 **100 小时用 Rust 和 Triton 编写一个 LLM 训练器**，**Sasha** 可提供咨询或合作。
   - *这可能会带来 LLM 训练领域的创新发展。*
- **DALI Dataloader 展示了惊人的吞吐量**：DALI Dataloader 每秒可以读取 **5,000 张 512x512 的 JPEG 图片**，有效地利用 GPU 资源进行大规模图像变换。
   - *成员们注意到，即使在进行完整的 **ImageNet transforms** 时，其性能依然强劲。*
- **使用 CUDA 并行化 RNN 的进展**：讨论集中在**使用 CUDA 并行化 RNN** 的挑战上，并参考了 S4 和 Mamba 等创新解决方案。
   - *这揭示了社区对于克服 RNN 架构中顺序依赖关系的兴趣。*
- **优化 Onnxruntime Web 大小**：Onnxruntime Web 的默认 WASM 大小为 **20 MB**，引发了关于在整合自定义推理逻辑时进行优化的讨论。
   - *成员们探索了各种策略，包括使用仅 **444K** 的精简版本，以实现潜在的效率提升。*
- **期待 TorchAO 支持 NF4**：成员们表达了对 **TorchAO** 实现 **NF4** 的渴望，指出它可以将 **VRAM** 需求从 **16G 降低到 10G**。
   - *他们庆祝速度从 **每步 11 秒** 提升到 **每步 7 秒**，强调了性能的增强。*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **文档分类自动化**：用户探索了 AI 工具如何通过内容分析显著简化文档分类，强调了增强效率的结构化方法。
   - *有人对项目目标的沟通可能存在脱节表示担忧，这可能会阻碍自动化进程。*
- **OpenAI API 的成本影响**：在讨论财务方面时，发现使用 OpenAI API 分析数千个媒体文件的费用可能超过 **$12,000**，这对依赖该服务的项目构成了挑战。
   - 这引发了对本地解决方案可行性的询问，尽管本地存储和处理能力也涉及潜在的高昂成本。
- **GPT-4 处理复杂数学的能力**：据报道，**GPT-4o** 能有效应对复杂的数学挑战，尤其是与 Wolfram 等插件结合使用时。
   - *一位用户提到了 GPT 行为的随机性（stochastic nature），并建议通过与外部工具更紧密的集成来增强可靠性。*
- **有效关键词选择的需求**：一位用户试图从 12,000 个关键词的大型集合中筛选出 **50 个关键词**，由于模型的 context window 限制，这一任务面临挑战，凸显了任务的复杂性。
   - *参与者建议采用批量查询和结构化数据展示来简化关键词选择过程。*
- **Prompt Engineering 的挑战**：许多用户表示在编写有效 Prompt 方面存在困难，特别是在处理确定性任务时，这表明缺乏向 AI 传达需求的简化方法。
   - *对话强调了在创建可执行 Prompt 所需理解方面的差距，建议需要更清晰的指南。*

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.59.0 发布带来增强功能**：新版本 **v0.59.0** 增强了对 `/read-only` 命令的支持，增加了 shell 风格的自动补全，并更新了 YAML 配置格式以提高清晰度。
   - 该更新通过在编码任务期间更好的处理提升了性能，并引入了新的 sanity checks 以简化启动流程。
- **对 Aider 性能缓慢的担忧**：用户在使用 **Sonnet 3.5** API 时遇到了 Aider 的显著延迟，特别是在处理大文件或广泛的代码上下文时。
   - 建议包括限制上下文文件和使用 verbose 标志，许多用户正在寻求像 **OpenRouter** 这样的替代方案来进行 API 管理。
- **推介 Dracarys 2 作为顶级编码模型**：[Dracarys 2](https://x.com/bindureddy/status/1842611268148203883) 被宣布为一款强大的编码模型，在 LiveCodeBench 等性能基准测试中超越了 **Sonnet 3.5**。
   - 虽然它在代码编辑方面达到了 **67%**，但一些用户认为它只是现有模型的翻版，而非真正的能力创新。
- **Python 3.13 特性脱颖而出**：Python **3.13** 的正式发布展示了诸如[更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) 以及在没有 GIL 的情况下运行 Python 等增强功能。
   - 值得注意的更新还包括通过 Beeware 项目扩展了对 iOS 和 Android 作为 **Tier 3 supported** 平台的支持。
- **语义搜索技术的创新**：关于 **semantic search** 优于关键词搜索的讨论强调了其根据含义而非精确匹配来增强查询结果的能力。
   - 然而，实例表明，过度依赖语义搜索可能会在实际应用中导致意想不到的糟糕结果。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 通过新模型进行创新**：Nous 推出了 Forge 和 Hermes-3-Llama-3.1-8B 等令人兴奋的项目，展示了其在**用户导向可控性 (user-directed steerability)** 方面的尖端技术。
   - 这些进步突显了令人印象深刻的创造力和性能，有可能改变 AI 的未来发展。
- **Meta Movie Gen 研究论文发布**：Meta 发布了一篇[研究论文](https://ai.meta.com/static-resource/movie-gen-research-paper)，详细介绍了他们在生成建模方面的 **Movie Gen** 创新。
   - 该文档是理解 Meta 在视频生成技术方面进步的方法论的重要参考。
- **GenRM 增强奖励模型训练**：**GenRM** 的引入展示了奖励模型训练方式的重大转变，集成了 next-token predictions 和 Chain-of-Thought 推理。
   - 这一进步通过利用统一的策略和奖励模型，提升了众多任务的性能。
- **SwiftSage v2 开源 Agent 推出**：集成了不同思维方式以增强推理能力的新型 **SwiftSage v2** Agent 系统现已在 [GitHub](https://github.com/SwiftSage/SwiftSage) 上可用。
   - 该系统针对复杂问题，展示了在使用 in-context learning 的各种推理任务中的优势。
- **Open Reasoning Tasks 项目说明**：**Open Reasoning Tasks** 频道被明确为一个讨论 GitHub 上正在进行的工作的协作空间。
   - 鼓励成员贡献与增强 AI 系统推理任务相关的见解和开发成果。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **模型加载困扰**：用户在 LM Studio 中加载模型时遇到问题，遇到诸如 'No LM Runtime found for model format 'gguf'!' 之类的错误，这通常是由于 CPU 指令集（如 AVX2）过旧导致的。
   - 他们建议升级硬件或切换到 Linux 以获得更好的兼容性。
- **GPU 配置难题**：社区评估了在多 GPU 设置中混合使用 GPU 的挑战，特别是使用 **4090** 和 **3090** 型号，强调了潜在的性能限制。
   - 共识表明，虽然混合使用是可行的，但较慢的 GPU 往往会成为整体性能的瓶颈。
- **图像处理见解**：关于支持图像处理模型的问题，推荐将 **MiniCPM-V-2_6-GGUF** 作为一个可行的选择。
   - 用户对图像尺寸以及分辨率如何影响模型推理时间表示关注。
- **提示词模板要点**：正确使用提示词模板对 LLM 至关重要；不恰当的模板可能导致输出中出现意外的 Token。
   - 讨论显示，偏离默认模板可能会导致显著的输出不匹配。
- **GPU 显存对决**：对比性能讨论强调，拥有 **24GB** VRAM 的 **Tesla P40** 非常适合 AI 任务，而拥有 **16GB** 的 **RTX 4060Ti** 在某些场景下也能胜任。
   - 针对 P40 在 **Stable Diffusion** 中较慢的表现存在担忧，强调了其能力未被充分利用。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 与 Fal.ai 合作**：OpenRouter 已正式与 **Fal.ai** 建立合作伙伴关系，通过[此链接](https://x.com/isidentical/status/1842650721969459561)增强图像工作流中的 **LLM** 和 **VLM** 能力。
   - 用户可以使用 OpenRouter 上的 **Gemini** *重新构思他们的工作流*，以简化图像处理任务。
- **API4AI 助力 AI 集成**：**API4AI** 平台促进了与 OpenAI 和 Azure 等服务的轻松集成，提供了一系列**现实世界交互** API，包括**邮件处理**和**图像生成**。
   - 这些功能使开发者能够更有效地构建多样化的 AI 应用程序。
- **重复生成问题依然存在**：用户报告在使用 OpenRouter API 时出现重复生成响应的问题，这表明存在特定于设置的问题，而一些用户在调整响应解析器后遇到了 404 错误。
   - 这表明需要排查潜在的超时或 API 可用性延迟。
- **数学模型在 STEM 任务中表现出色**：用户强调 **o1-mini** 是数学 STEM 任务的首选模型，因为它在渲染输出方面效率很高，同时也引发了关于 **LaTeX 渲染**能力的讨论。
   - 社区热衷于优化 OpenRouter 环境中的数学公式交互。
- **寻求非营利组织折扣**：出现了关于非洲非营利教育机构获取 OpenRouter 服务潜在折扣的咨询。
   - 这反映了 AI 社区内对于教育倡议能够获得负担得起的技术支持的广泛愿望。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MATS 项目迎来新导师**：Alignment Science 联合负责人 **Jan Leike** 将担任 [MATS Winter 2024-25](https://matsprogram.org/apply) 的导师，申请截止日期为 10 月 6 日晚上 11:59 PT。
   - *该导师项目提供了对对齐科学的深刻见解*，使其成为潜在申请者梦寐以求的机会。
- **理解 ICLR 论文发布时间**：讨论澄清了 **ICLR** 论文发布的时间通常取决于评审过程，并可能存在草案的非正式分享。
   - *成员们强调，了解这些时间线对于保持研究可见性至关重要*，特别是对于那些等待最终预印本的人。
- **RWKV 系列及版本控制挑战**：社区探讨了追踪 **RWKV** 系列版本变化的困难，表明需要更清晰的文档。
   - *链接的论文提供了 RWKV 变更的逐步概述*，这可能有助于测试和研究理解。
- **增强 AI 对齐的生成式奖励模型**：成员们讨论了思维链生成式奖励模型 (CoT-GenRM)，旨在提高训练后性能以及与人类价值观的对齐。
   - *通过融合人类和 AI 生成的反馈，该方法旨在提升决策中的推理能力。*
- **开发中对 JAX 模型的支持**：关于为 **JAX 模型** 提供一流支持的潜力的对话被激发，成员们渴望获得更新。
   - *这突显了人们对于优化框架以适应机器学习开发中不断变化的需求日益增长的兴趣。*

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 错误与困扰**：用户在项目过程中遇到了频繁的 **Cohere API** 错误，如 'InternalServerError'，特别是在微调页面上，影响了推进试验至关重要的技术。
   - 版主确认由于积压严重，已优先处理支持工单，成员们强调了 **429 错误** 影响了多个用户。
- **Companion Discord 机器人革新互动体验**：引入了利用 Cohere 的 Discord 机器人 **Companion**，以增强**动态人格建模**和用户互动，同时提供集成的审核功能。
   - 该 GitHub 项目旨在提升社区讨论质量，并邀请大家探索，因为它加强了 Discord 内的审核效率。
- **关于 API 商业用途的讨论**：社区成员确认 **Cohere API** 可用于商业用途，目标是企业解决方案，同时引导用户查看 FAQ 以获取许可详情。
   - 讨论强调了 API 稳定性和效率的重要性，开发者们热衷于了解从其他平台迁移的细微差别。
- **Rerank API 响应受到关注**：有关 **Rerank API** 在使用 **return_documents: True** 参数时未返回预期文档数据的担忧浮出水面，阻碍了数据检索过程。
   - 用户渴望了解最近的更新是否改变了功能，并寻求对之前受损效率的解答。
- **社区关注协作与反馈**：成员们敦促用户联系支持部门并与 **Cohere** 团队分享反馈，强调了社区驱动改进的重要性。
   - 对话围绕着提供可操作见解的必要性展开，以改善 Cohere 生态系统中的用户体验和技术性能。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SWE-bench Multimodal 发布，旨在解决视觉问题**：新的 **SWE-bench Multimodal** 旨在评估 Agent 解决视觉 **GitHub** 问题的能力，包含来自 **17 个 JavaScript** 仓库的 **617 个新任务**，并引入了 **SWE-agent Multimodal** 以实现更好的处理。
   - 该计划针对现有 Agent 的局限性，推动在视觉问题解决中有效地完成任务。
- **Reka Flash 更新增强了多模态能力**：**Reka Flash** 的最新更新支持交错的多模态输入，如**文本、图像、视频**和**音频**，显著提升了其功能。
   - 这一增强突出了在实际应用中**多模态理解**和推理方面的进展。
- **Cursor 团队与 Lex Fridman 讨论 AI 辅助编程**：在与 **Lex Fridman** 的对话中，**Cursor 团队**深入探讨了 AI 辅助编程和不断演变的编程未来，展示了他们的创新环境。
   - 讨论涵盖了具有影响力的议题，如 **GitHub Copilot** 以及 AI 集成到编程工作流中的复杂性。
- **Discord 音频故障困扰用户**：成员在通话过程中遇到**音频问题**，由于听取困难，有人建议切换到 Zoom。
   - *Verymadbear* 调侃道：**“如果麦克风没出问题，那就不叫真正的会议”**，道出了大家面临的挫败感。
- **探索 Luma AI 的魔力**：对话集中在 **Luma AI**，展示了使用该工具开发的令人印象深刻的**视频应用**和项目，特别是它在**电影剪辑**中的实用性。
   - Karan 强调了 **Luma** 为电影制作带来的创造力，并强调了其实现独特镜头运动的能力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AMD vs NVIDIA：关于 SD 的大辩论**：用户在 **Stable Diffusion** 生成图像时更倾向于选择 **RTX 4070** 而非 **RX 6900 XT**，理由是性能更优。
   - 一些人建议将 **3080 Ti** 作为 4070 的替代方案，其速度快 **30%**，为 GPU 对比增加了另一个维度。
- **CogVideoX 在视频生成领域夺冠**：在文本到视频（text-to-video）生成方面，**CogVideoX** 目前是领先的开源模型，超越了像 **Svd** 这样的旧模型。
   - 用户指出 **Stability** 已经落后，替代模型被证明在*认知能力上更胜一筹*。
- **UI 之战：Stable Diffusion 的 ComfyUI vs Forge UI**：从 **Automatic1111** 转型时，用户在 **ComfyUI** 和 **Forge UI** 之间产生了分歧，两者都展示了独特的优势。
   - 虽然许多人因为易用性而偏好 **ComfyUI**，但也有人欣赏 **Forge** 作为 Auto1111 的优秀分支所带来的增强功能。
- **LoRA 训练难题困扰社区**：用户在为 **SDXL** 训练 **LoRA** 时表达了挑战，并在专门用于故障排除的社区频道中寻求帮助。
   - 社区成员积极提供支持，分享资源以协助创建有效的 **LoRA** 模型。
- **生成后编辑：用户想要更多功能**：围绕生成后编辑的讨论不断涌现，重点在于上传并重新生成特定图像区域的能力。
   - 用户对突出显示并更改生成图像部分的理念很感兴趣，寻求工作流的改进。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Opus 限制引发用户愤怒**：用户对 **Opus** 消息突然减少到 **每天 10 条** 表示沮丧，并对消费者权益提出了质疑。
   - 随后的更新表明限制可能已恢复到 **50 条消息**，缓解了社区内的一些担忧。
- **Perplexity 遭遇用户困扰**：多位用户报告了 **Perplexity** 在访问 Pro 功能和客户支持延迟方面的问题。
   - 随着用户注意到平台转向推广内容而非实质性的功能增强，担忧情绪日益增加。
- **开发团队的重心受到质疑**：用户对开发团队在 **Mac app** 之外的优先级提出了疑问，希望看到更多显眼的新功能。
   - 社区反馈暗示，平台重心似乎转向了赠品活动，而非重大的平台改进。
- **探索 API 的 Structured Outputs**：关于在 **Perplexity API** 中集成 **Structured Outputs** 的讨论，借鉴了 [OpenAI library](https://platform.openai.com/docs/guides/structured-outputs/introduction) 中的功能。
   - 这一探索强调了人们对扩展 API 功能以更好满足用户需求日益增长的兴趣。
- **量子钟（Quantum Clocks）承诺更高精度**：一个涉及 [quantum clocks](https://www.perplexity.ai/search/what-is-a-quantum-clock-t4A_.5lTTiCUnbMObd_5_A) 的创新概念突显了精密计时领域的进展。
   - 该技术承诺比传统方法具有更高的准确性，为未来的应用打开了大门。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 在 Milvus DB 集成方面遇到困难**：由于意外的 API 更改以及对原生对象的依赖，用户报告了在将 **Milvus DB** 集成到其 **LlamaIndex** 工作流中时面临的挑战。
   - 他们呼吁采用更模块化的设计，以便在不强制依赖结构化对象的情况下有效利用预构建组件。
- **Swarm Agents 创建 AI 生成的视频**：一个项目展示了如何构建一个 Agent “集群（swarm）”，从自然语言提示词开始自主创建 AI 生成的 YouTube 视频，详见 [此教程](https://t.co/TKs9QqP4ym)。
   - 这种方法突显了 **multi-agent architectures** 在简化视频生成工作流方面的潜力。
- **RAG 流水线中的动态数据源推理**：在 RAG 流水线之上的 Agent 层允许将不同的数据源构建为“工具”，从而增强对源检索的推理，总结见 [此处](https://t.co/jUzqZrnCOH)。
   - 这种动态方法强调了数据处理正转向更具交互性和响应性的检索机制。
- **Agentic Retrieval 的快速设置**：一份实用的指南提供了 RAG 中 **agentic retrieval** 的快速设置，与静态检索方法相比，为更灵活的数据处理铺平了道路，详见 [此指南](https://t.co/V0JwbQ4Dmz)。
   - 用户对实现的简便性表示赞赏，这标志着检索架构使用方式的转变。
- **通过 Multi-Agent 系统实现法律合规**：一个 **multi-agent system** 协助公司评估法规合规性并起草法律回复，更多详情见 [此处](https://t.co/s1MhinpZ5B)。
   - 该系统实现了法律判例审查的自动化，展示了法律工作流中显著的效率提升。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **梯度检查点（Gradient Checkpointing）增强训练**：一名成员询问了 **gradient checkpointing**，这对于高效训练大型模型至关重要，并强调了它在提升训练能力方面的作用。
   - 如果没有这些优化，tinygrad 只能处理**非常小的玩具模型**，从而限制了其整体性能。
- **用于色彩空间适配的 VAE 训练**：围绕训练 **Variational Autoencoder (VAE)** 以使现有模型适配 **CIE LAB 色彩空间**以改善输出展开了讨论。
   - 对输入的重大更改将需要超出简单 **finetuning** 范围的广泛修改，从而使过程复杂化。
- **澄清 Tinybox 并非服务器工具**：一位用户寻求关于 tinygrad 功能的澄清，询问它是否充当运行 LLM 的**本地服务器**。
   - 澄清指出 tinygrad 更类似于 **PyTorch**，侧重于开发而非服务器功能。
- **KAN 网络带来快速训练**：尽管存在炒作，成员们注意到在 TinyGrad 中很难找到现有的 **KAN networks** 实现，同时展示了能够实现高效训练的示例。
   - *FastKAN* 在 MNIST 上实现了 **10 倍加速**，强调了其性能优势。
- **VIZ 和调度器（Scheduler）增强更新**：成员们收到了关于 **VIZ server** 完全重写的更新，目标是增强内核和图重写。
   - 随着开发的继续，进展的主要阻碍包括解决 **ASSIGN** 问题以及完善融合（fusion）和分组逻辑（grouping logic）。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 集成推理能力**：讨论透露 **OpenAI o1** 直接在模型中集成了推理，超越了推理过程中传统的 **MCTS** 等方法。
   - 尽管如此，对于底层挑战的简化仍存在怀疑，特别是因为某些讨论似乎受到了审查。
- **Entropix 提供提示词优化**：**Entropix/Entropy Guided Adaptive Sampler** 增强了提示词优化，专注于注意力熵（attention entropy）以提升模型性能。
   - 提到的优势包括提高叙事连贯性和减少幻觉，这表明即使在小模型中也具备这些能力。
- **Reflection 70B 未能达到基准测试水平**：一位成员对他们复制的 **Reflection 70B** 表示失望，该模型未能匹配其最初报告的基准测试结果。
   - 尽管如此，他们仍致力于反思微调概念，并承诺很快会提供更详细的见解。
- **Open O1 作为竞争对手出现**：**Open O1** 作为专有模型的有力替代方案被引入，声称在推理、编程和数学任务方面具有优越性。
   - 一些社区成员认为讨论缺乏深度，要求对该模型进行更透彻的分析。
- **RNN 投资诉求引起关注**：一条推文热切呼吁资助开发“再多一个 RNN”，声称它可以*摧毁 Transformer* 并解决长上下文问题。
   - 该成员满怀热情地强调了支持的紧迫性，敦促社区采取行动。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **类生成（Class Generation）Notebook 发布**：GitHub 仓库现在新增了一个[关于类生成的 Jupyter notebook](https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/class_generation.ipynb)，展示了来自 DSPy 和 Jinja2 的**结构化输出**。
   - 该项目旨在增强结构化输出生成，并邀请在 [GitHub](https://github.com/seanchatmangpt/dslmodel) 上进行进一步贡献。
- **即将举行现场编程（Livecoding）活动**：宣布了一场激动人心的现场编程活动，成员们可以直接在 Discord 中观察 Notebook 的创建过程。
   - *鼓励成员加入讨论帖*，在会议期间进行互动，促进协作式 Notebook 开发。
- **TypedPredictors 已准备就绪**：有关于在没有模式格式化逻辑的情况下使用 `TypedPredictors` 的讨论，估计可以在大约 **100 行**代码内实现。
   - 预计很快会集成到 `dspy.Predict` 中，为开发者提供高效路径。
- **可追溯性（Traceability）并不像看起来那么棘手**：一位用户询问如何在不使用外部库的情况下为 DSPy 添加可追溯性，以跟踪 Token 数量并管理成本。
   - 建议利用 `your_lm.history` 属性来有效监控支出。
- **从 dspy.OllamaLocal 转向 dspy.LM 时面临挑战**：一位新用户报告在从 `dspy.OllamaLocal` 切换到 `dspy.LM` 期间出现段错误（segmentation fault），表明可能存在版本不匹配。
   - 回复建议重新安装 DSPy 或确认使用正确的模型端点以解决该问题。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **来自 chat_manager 的实时流传输**：一个 streamlit UI 实现了来自 **chat_manager** 的实时流传输，这得益于一个用于自定义消息处理的 [GitHub pull request](https://github.com/microsoft/autogen/pull/1783)。
   - 此设置对于需要用户对消息进行即时反馈的交互式应用至关重要。
- **线下出席名额专属**：由于容量限制，只有 Berkeley 的学生可以线下参加即将举行的讲座，限制了更广泛的访问。
   - 这一限制在关于非 Berkeley 学生座位可用性的讨论中得到了确认。
- **Omar 的讲座引发热烈反响**：成员们对 **Omar** 即将举行的关于 **DSPy** 的讲座表示热切期待，强调了其相关性。
   - 强调了对 **DSPy** 项目的积极贡献，反映了成员们在提升专业知识方面的投入。
- **成员参与 DSPy 贡献**：一位成员详细介绍了他们最近对 **DSPy** 项目的贡献，展示了他们的参与度以及增强该框架的愿望。
   - 这种持续的参与标志着社区对改进 **DSPy** 功能的浓厚兴趣。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **重构 Mojo 参数约定语法**：一位成员分享了一份[关于重构参数约定语法的提案](https://gist.github.com/lattner/da647146ea573902782525f3446829ff)，旨在完善 **Mojo** 编程语言的各个方面。
   - 他们鼓励通过 [GitHub Issue](https://github.com/modularml/mojo/issues/3623) 提供社区反馈，以帮助完善该提案。
- **Mojo 发布基准测试框架**：Mojo 引入了一个用于运行时性能评估的 [benchmark 软件包](https://docs.modular.com/mojo/stdlib/benchmark/)，类似于 Go 的测试框架。
   - 成员们讨论了使用 `benchmark.run` 来高效评估函数性能，并报告平均耗时和迭代次数。
- **枚举现在支持 Variant 类型**：成员们澄清说 **Mojo** 中没有专门的枚举语法，但 **Variant** 类型可以提供类似的功能。
   - 在引入完整的 sum types 之前，可以通过 struct 声明和别名来创建标签。
- **Max Inference Engine 遇到错误**：用户报告了在 Intel NUC 上使用 **max inference engine** 时的问题，遇到了与 `libTorchRuntimePlugin-2_4_1_post100.so` 和 ONNX 操作相关的错误。
   - 问题包括操作合法化失败以及更改 opset 版本时的并发症。
- **兼容性 Torch 版本澄清**：一位用户询问了 PyTorch 的安装情况，问道 *你使用的是哪个 torch 版本？* 以确保兼容性。
   - 提供的输出显示 **PyTorch 版本为 2.4.1.post100**，并包括关于 **GCC 版本 13.3** 和来自 **conda-forge** 的 Intel 优化的细节。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 缺乏 KTO 训练支持**：一位成员询问 **Torchtune** 是否支持 **KTO 训练**，有迹象表明，如果有必要，这可能会被添加到 DPO recipe 中。
   - 他们建议提交一个 issue 来跟踪此功能请求。
- **大型自定义 CSV 数据集出现 AssertionError**：用户报告说，当 shuffle=false 时，大于 **100MB** 的自定义 CSV 数据集会出现 **AssertionError**，但较小的数据集运行正常。
   - 这表明错误可能与数据集大小有关，而非代码本身。
- **LLAMA 3.2 3B 微调问题**：关于 **LLAMA 3.2 3B 的全量微调**进行了讨论，强调蒸馏模型通常需要特定处理，如较低的学习率。
   - 一位用户提高了学习率以获得满意的损失曲线，尽管他们缺乏全面的评估数据。
- **Grace Hopper 芯片备受关注**：成员们分享了关于 **Grace Hopper 芯片**性能的查询，特别是它们与带有 Hopper GPU 的标准架构相比如何。
   - 这说明了人们对使用新硬件设计所带来的影响有着浓厚的兴趣。
- **训练效率：最大序列长度 vs Batch Size**：建议优化**最大序列长度**而非增加 batch size，以增强 **blockmask dimension** 的性能。
   - 使用较长的序列可能会提高打包效率，但由于静态打包方法，可能会减少数据打乱。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **微调后的 GPT-4 模型消失了**：一位成员幽默地声称 OpenAI 可能拿走了所有人的微调 **GPT-4** 模型，并表示 *'I lost my models'*（我丢了我的模型），并暗示其性能很*垃圾*。
   - 另一位成员指出，*'you only finetune weights you own,'*（你只能微调你拥有的权重），强调了使用共享资源的风险。
- **群组 Logo 更改引发混乱**：一位成员表示由于 Logo 更改，他们跟丢了社区，并幽默地哀叹这造成的困惑。
   - 这强调了品牌变更对社区认可度的影响。
- **Intel 与 Inflection AI 联手**：一位成员分享了一篇文章，详细介绍了 **Intel** 与 **Inflection AI** 合作推出企业级 AI 系统的情况，称其*很有趣*。
   - 这一公告表明企业级 AI 领域将有重大发展，可能会重塑技术的使用方式。
- **探索 Axolotl 的非 pip 打包工具**：一位成员询问是否可以将 Axolotl 切换到像 **uv** 这样的非 pip 打包工具，因为他们对依赖项问题感到沮丧。
   - 他们表示愿意为增强包管理体验做出贡献。
- **fschat 包未找到错误**：一位用户报告在安装 `axolotl[deepspeed,flash-attn]` 时出现 *'Could not find a version that satisfies the requirement fschat (unavailable)'* 错误。
   - 列出的可用版本范围从 **0.1.1** 到 **0.2.36**，但没有一个被标记为可用，这引起了困惑。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LlamaIndex RAG-a-thon 启动**：**LlamaIndex Agentic RAG-a-thon** 定于 **10 月 11 日至 13 日**在硅谷举行，重点关注与 **Pinecone** 和 **VESSL AI** 合作的 Retrieval-Augmented Generation 技术。
   - 该活动旨在推进用于企业应用的 **AI agents**，开发者有机会赢得现金奖励，详见[此链接](https://rag-a-thon-2.devpost.com/)。
- **O1 在简单任务上失败**：讨论显示，**O1** 声称在**奥林匹克级别**的任务中表现强劲，但在更简单的问题上却表现挣扎，引发了对其泛化能力的担忧，正如**[相关讨论](https://x.com/JJitsev/status/1842727628463128968)**中所述。
   - 这些发现引发了关于 SOTA **LLMs** 如何有效管理泛化的问题，这一担忧得到了[研究论文](https://arxiv.org/abs/2406.02061)的支持。
- **寻求 Clip Retrieval API 的明确说明**：人们对 **clip retrieval API** 持续关注，一位成员询问更新情况，表明该技术开发方面的沟通存在断层。
   - 缺乏回应表明需要团队负责人或开发者提供更多信息。
- **分享 Epoch 训练经验**：一位用户分享了使用 **80,000 epochs** 进行训练的心得，为关于模型训练性能的深入对话奠定了基础。
   - 这一细节突显了在**模型训练**中实现最佳结果的不同方法。
- **新工具进入竞技场**：分享了一个指向 **[AutoArena](https://www.autoarena.app/)** 的链接，被认为是一个有趣的工具，反映了对模型改进资源的兴趣。
   - 这种兴趣强调了社区在 AI 开发中利用实用工具的推动力。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Grimes 在 Coachella 的 01 AI 构建方案揭晓**：一份指南概述了 Grimes 和 Bella Poarch 如何在 Coachella 音乐节上使用宏键盘和麦克风设置他们的 [01 AI 助手](https://01.openinterpreter.com/hardware/grimes)。这个简单的设置包括购买宏键盘和麦克风，并重新映射按钮以与 AI 交互。
   - 成员们了解到，这种设置可以实现与助手的高效直接互动，强调了在动态环境中的可用性。
- **本地 LlamaFile 模型面临的挑战**：一位成员在尝试交互时遇到了本地 LlamaFile 模型的错误，提示：**'Model not found or error in checking vision support'**。根据链接的配置，他们的模型 **'Meta-Llama-3.1-8B-Instruct'** 应该被正确映射。
   - 这引发了对配置细节的困惑，并导致了关于 [litellm/model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) 中上下文和定价的讨论。
- **Discord Automod 针对垃圾信息控制**：有一项讨论建议使用 Discord Automod 拦截普通用户的 **@everyone 标签**，以减少垃圾信息。一位成员指出，**95% 的垃圾信息机器人**都会尝试标记所有人，这使其成为一种有效的方法。
   - 实施此举可以简化社区互动，在关键讨论期间最大限度地减少垃圾信息的干扰。
- **比较 01 的成本：11 Labs vs OpenAI**：一位成员提出了关于在 **11 Labs** 和 **OpenAI** 之间使用 **01 服务** 相关成本的问题。有人担心可能需要升级他们在 **11 Labs** 的会员资格。
   - 这反映了人们对了解使用这些平台的财务影响的广泛兴趣，特别是对于那些严重依赖多种服务的用户。
- **创新的数字助手帽子创意**：一位用户提议将 **帽子** 与 **数字助手** 集成，具备扬声器、麦克风和一键通（push-to-talk）按钮功能，以实现无缝交互。该项目旨在包含 **手机通知**、问答和日历管理，可能发展为一个[带有构建指南的开源项目](https://link.to.project)。
   - 另一位用户对这种能增强其 **编程项目** 的设备表示热衷，强调了对提高 **编程效率** 的渴望。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **加入 LlamaIndex RAG-a-thon！**：**LlamaIndex Agentic RAG-a-thon** 将于 **10 月 11-13 日** 在硅谷举行，重点关注 **Retrieval-Augmented Generation** 技术。
   - 感兴趣的参与者可以查看[此处详情](https://rag-a-thon-2.devpost.com/)，并通过 **[Slack](https://join.slack.com/t/futureproof101/shared_invite/zt-2s1c1rlxh-3p64w0UbYQFdjTIpfYb3KQ)** 或 **[Discord](https://discord.com/invite/eN6D2HQ4aX)** 进行联系。
- **使用自然语言实现 QA 自动化**：一位成员讨论了 [Autonoma](https://getautonoma.com/)，这是一个使用 **自然语言** 和 **计算机视觉** 自动化 QA 的平台，旨在减少 Bug。
   - 主要功能包括 **Web 和移动端支持**、CI/CD 就绪以及 **自愈（self-healing）** 能力。
- **通过 Sci Scope 保持领先**：[Sci Scope](https://sci-scope.com) 每周汇总新的 ArXiv 论文，并将个性化摘要直接发送到您的收件箱。
   - 这种个性化的新闻通讯确保订阅者不会错过 AI 研究中的关键进展。
- **对具备支付能力的 Agent 的兴趣**：一位用户提出了关于能够花钱的 Agent 的问题，引发了关于该领域潜在应用和创新的讨论。
   - 虽然没有分享具体的项目，但这个概念引起了许多成员的兴趣。
- **多工具 Agent 实现指南**：成员们表达了对如何实现使用多个工具的 Agent 的指导需求，反映了对有效数据源集成的需求。
   - 在社区内，创建能够利用多样化工具的 Agent 的兴趣持续增长。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **第五届年度 MLOps World + GenAI 大会即将到来！**：参加 11 月 7 日至 8 日在德克萨斯州奥斯汀举行的 **MLOps World + GenAI Conference**，届时将有 **50+** 个主题、实操工作坊和社交机会。点击[此处](https://mlopsworld.com/speakers)查看完整议程，包括 11 月 6 日的额外虚拟会议日！
   - *记好你的时间表！* 这是 AI 工程师交流并了解 MLOps 最新动态的绝佳机会。
- **Manifold Research Lab 发布 CRC 更新**：Manifold 正在举办名为 **CRCs** 的互动更新活动，探讨 **Multimodality**、**Robotics** 以及各种研究项目的突破。在他们的[活动页面](https://www.manifoldrg.com/events/)获取更多见解，并点击[此处](https://discord.gg/Pza3jxKPUY)加入社区。
   - 这些会议提供了对前沿研究的深入探讨，非常适合想要在该领域保持领先地位的技术爱好者。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **播客重点介绍 Data Pipelines**：本周三，[AIFoundry.org](https://aifoundry.org/) 将主持一场播客，涵盖用于模型 **fine-tuning** 的 **data pipelines**，并强调成功所需的必要**数据量**。
   - 预计该活动将引发关于各种 **fine-tuning** 任务所需最佳调整的讨论。
- **社区关于数据选择的疑问**：社区围绕**数据选择和处理过程**展开了热烈讨论，许多人正在寻求有效方法的指导。
   - 重点在于调整这些流程，以增强对特定 **fine-tuning** 任务的适用性。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **发布新研究见解**：分享了一篇题为“[Title of the Paper](https://arxiv.org/abs/2410.02694)”的新研究论文，重点关注 AI 方法论的进展。
   - 这突显了 AI 研究的持续演进及其对未来 **benchmarks** 的影响。
- **AI Benchmarking 讨论**：讨论强调了在技术不断发展的背景下，开发稳健的 **benchmarks** 以准确评估 AI 性能的重要性。
   - 成员们强调需要制定标准，以确保不同 AI 模型之间的可比性。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将予以移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将予以移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将予以移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将予以移除。


---

# 第 2 部分：各频道详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1291840621094502461)** (729 条消息🔥🔥🔥): 

> - `用于微调的 Unsloth GUI`
> - `Qwen 模型性能`
> - `模型中的多模态支持`
> - `训练的数据集格式化`
> - `训练波斯语模型` 


- **用于微调的 Unsloth GUI**：一个名为 'Unsloth Studio' 的 GUI 预计将用于微调，它将通过自动处理数据集格式化和依赖项来简化用户的操作流程。
   - 该工具旨在降低初学者的门槛，使其无需深厚的编程知识即可训练模型。
- **Qwen 模型与 LLaMA 的性能对比**：用户讨论了 Qwen 模型，指出 1B 模型和更大的模型在对话场景中表现相似，Qwen 2.5 7B 是一个可以进行微调以提高性能的潜在模型。
   - 一些用户报告称，在 Qwen 和 LLaMA 模型之间切换时，性能和训练效率存在显著差异。
- **模型中的多模态支持**：目前正在进行将图像输入能力集成到 LLaMA 3.2 等模型中的工作，尽管具体的发布时间表尚不明确。
   - 用户提到了微调多模态模型的复杂性，并对未来的支持表示期待。
- **训练的数据集格式化**：讨论了微调模型的数据集格式化问题，重点是确保训练对话的结构正确。
   - 建议将对话部分封装为单个文本块，并根据模型规范调整格式。
- **训练波斯语模型**：用户询问了使用波斯语数据集进行微调的有效模型，Qwen 被建议作为一个合适的选项。
   - 对话强调了非英语语言高质量数据集对于获得更好模型性能的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1JXK3W2zWAThS5uqzFTjUi38Sf8sStEkn#scrollTo=SWXkHoegOlyd">Google Colab</a>: 未找到描述</li><li><a href="https://slurm.schedmd.com/documentation.html">Slurm Workload Manager - 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/fixie-ai/ultravox-v0_4-mistral_nemo">fixie-ai/ultravox-v0_4-mistral_nemo · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/qwen-25-66fe4c08fb9ada518e8a0d3f">Qwen 2.5 - unsloth 收藏集</a>: 未找到描述</li><li><a href="https://tenor.com/view/wow-gif-20411229">Wow GIF - Wow - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: 使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用...</li><li><a href="https://ollama.com/unclemusclez/unsloth-llama3.2">unclemusclez/unsloth-llama3.2</a>: 使用 Unsloth 的 Llama 3.2</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/ef8GmUlgLF">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/chigkim/Ollama-MMLU-Pro">GitHub - chigkim/Ollama-MMLU-Pro</a>: 通过在 GitHub 上创建账户来为 chigkim/Ollama-MMLU-Pro 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/macadeliccc/opus_samantha?">macadeliccc/opus_samantha · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: 将算力和书籍转换为指令微调数据集！制作：QA、RP、分类器。</a>: 将算力和书籍转换为指令微调数据集！制作：QA、RP、分类器。 - e-p-armstrong/augmentoolkit</li><li><a href="https://huggingface.co/blog/mlabonne/sft-llama3">使用 Unsloth 超高效地微调 Llama 3.1</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1292236162341142601)** (8 条消息🔥): 

> - `Generational Shift in Content Consumption`（内容消费的世代转变）
> - `Deep Learning Enthusiasts Discussion`（Deep Learning 爱好者讨论）
> - `Short Form Content Opinions`（对短视频内容的看法）


- **年轻一代抛弃 TikTok**：一位成员观察到年轻一代正在远离 **TikTok** 和**短视频内容**，而年长一代似乎反而开始接受它。
   - “听到关于我们这一代的消息真不错”是大家共有的一种情绪，突显了对世代鸿沟的轻松看法。
- **Deep Learning 爱好者对现实的不同看法**：Deep Learning 爱好者之间的讨论强调，网络行为中经常可见的内容并不代表整体现实。
   - 一位参与者表示，虽然这些见解很有价值，但来自 TikTok 等平台的噪音可能会扭曲认知。
- **对轰炸式内容的喜爱**：一位成员幽默地声称喜欢以最大音量播放**短视频内容**，强调了由于注意力受损（fried attention span）而导致的快速滑动行为。
   - 他们澄清说自己并不使用 TikTok，但仍然享受这种混乱的内容消费体验。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1291838451318001695)** (137 条消息🔥🔥): 

> - `Model Fine-tuning Challenges` (模型微调挑战)
> - `Inference Issues with LLaMA` (LLaMA 的推理问题)
> - `Usage of LoRA in Fine-tuning` (LoRA 在微调中的使用)
> - `CUDA Configuration for WSL` (WSL 的 CUDA 配置)
> - `Training Loss Observation` (训练损失观察)


- **模型微调的复杂性**：用户讨论了在微调 Qwen2.5 和 LLaMA 3.1 等模型时遇到的问题，提到了在多次训练后推理过程中出现无限生成等问题。
   - 有人对微调已经微调过的模型时产生的灾难性遗忘（catastrophic forgetting）表示担忧，并建议通过合并数据集来获得更好的效果。
- **LLaMA 3.1 的推理问题**：多位用户报告称，在重新训练 LLaMA 3.1 后，他们的模型开始无休止地生成响应而无法完成生成，这表明微调过程可能存在问题。
   - 对话强调了检查正确的 chat templates 以及定义序列结束符（EOS）的必要性，以改善模型行为。
- **微调中的 LoRA 实现**：讨论了使用 LoRA 进行微调的可行性，一些用户指出虽然 LoRA 很有益，但全量微调（full fine-tuning）可能会产生更好的结果。
   - 参与者对有效利用 LoRA 的最佳方法发表了不同看法，并讨论了直接改进已微调模型的局限性。
- **WSL 上的 CUDA 设置以提升性能**：用户遇到了与 WSL 上的 CUDA 安装相关的问题，以及 NVIDIA 驱动程序对模型训练性能的影响，特别是在不同的配置下。
   - 对话中包含了资源链接，以确保正确安装 CUDA，从而在使用 Unsloth 和 Qwen 等模型时提升性能。
- **使用 LLM 设置内容审核**：一位用户询问如何利用 LLaMA 3.1 或 Qwen 执行内容审核任务，寻求关于如何使用包含 5 万条记录的自定义数据集构建训练设置的指导。
   - 讨论集中在如何通过微调策略在 LLM 中有效实施内容审核规则。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：查看下方列表获取我们所有的 notebook：</li><li><a href="https://docs.nvidia.com/cuda/wsl-user-guide/index.html">CUDA on WSL</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/llama-32-all-versions-66f46afde4ca573864321a22">Llama 3.2 All Versions - a unsloth Collection</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth?sort_models=downloads#models">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://huggingface.co/docs/trl/sft_trainer">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth?sort_models">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/commit/79a2112ca4a775ce0b3cb75f5074136cb54ea6df">Reload · unslothai/unsloth@79a2112</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1291930947297742858)** (101 messages🔥🔥): 

> - `RYFAI App`
> - `Ease of Use for Non-technical Users`（非技术用户的易用性）
> - `Competing Open Source Solutions`（竞争性开源解决方案）
> - `Privacy in AI`（AI 中的隐私）
> - `Market Saturation`（市场饱和度）


- **RYFAI 提供便捷的私有 AI 模型访问**：一位用户介绍了一款名为 **RYFAI** 的开源应用，该应用专为 MacOS、Windows 和 Linux 设计，强调其对易用性和在线隐私的关注。
   - 用户指出 **RYFAI** 允许完全离线运行，但有人认为这已经可以通过 **Ollama** 和 **OpenWebUI** 等成熟工具实现。
- **关于非专家技术门槛的辩论**：对话揭示了在**非技术用户**是否能处理像 **Ollama** 或 **Docker** 这样复杂的设置上存在分歧。
   - 一位参与者强调基础用户缺乏对这类工具的认知，暗示 **RYFAI** 的目标受众是不熟悉 AI 技术的人群。
- **对与成熟工具竞争的担忧**：成员们对 **RYFAI** 是否有潜力与拥有强大社区支持和资金的成熟工具（如 **OpenWebUI**）竞争表示怀疑。
   - 有人指出，如果没有显著的差异化或**更好的分发渠道**，**RYFAI** 可能会在饱和的市场中挣扎。
- **AI 工具中的隐私视角**：隐私是一个核心主题，讨论了**本地模型**如何提供比中心化 AI 服务更安全的替代方案，这对比关注数据隐私的用户特别有吸引力。
   - 尽管隐私很重要，但对于包括非技术用户在内的目标受众是否会优先考虑这一特性仍存争议。
- **关于产品可行性和市场契合度的反馈**：针对 **RYFAI** 的长期可行性提出了批评，认为满足**非技术用户群**的需求具有挑战性。
   - 强调该应用必须展示出优于现有选项的显著优势，才能在寻求隐私保护方案的用户中获得青睐。



**提到的链接**: <a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: User-friendly AI Interface (Supports Ollama, OpenAI API, ...)</a>: 用户友好的 AI 界面（支持 Ollama, OpenAI API, ...） - open-webui/open-webui

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1291847025012248638)** (8 messages🔥): 

> - `minLSTM and minGRU performance`（minLSTM 和 minGRU 的性能）
> - `Parallel scan algorithm`（并行扫描算法）
> - `Self-improvement in LLMs`（LLM 的自我提升）
> - `Chain-of-Thought reasoning`（思维链推理）


- **minLSTM 和 minGRU 挑战 Transformer**：来自 Mila 和 Borealis AI 的研究人员透露，名为 **minLSTM** 和 **minGRU** 的简化版 **RNN** 在任务中的表现可以与现代 Transformer 媲美。
   - 这些模型摒弃了额外的复杂性，在长序列处理中实现了 **200 倍的加速**，同时节省了 **88% 的内存**使用，从根本上质疑了先进架构的必要性。
- **对并行扫描算法（Parallel Scan Algorithm）的好奇**：一位成员询问了什么是 **parallel scan** 算法，该算法用于高效地并行训练这些新型极简 RNN。
   - 另一位成员链接了一份关于 **parallel prefix sums**（并行前缀和）的文档，为该话题提供了潜在的解释。
- **探索 LLM 的自我提升**：一项研究讨论了 LLM 利用预训练规模数据上的 **Chain-of-Thought** (CoT) 来**自我提升**推理能力的潜力，而无需监督数据集。
   - 通过利用预训练数据中存在的大量非结构化文本，这可以显著增强 LLM 的推理能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=BGnm7Lo8oW">Towards Learning to Reason at Pre-Training Scale</a>: 提示大语言模型 (LLM) 输出思维链 (CoT) 推理可以提高在复杂问题解决任务上的表现。此外，存在几种流行的“自我提升”方法...</li><li><a href="https://huggingface.co/posts/m-ric/957178001915012">Hugging Face 上的 @m-ric: &quot;📜 𝐎𝐥𝐝-𝐬𝐜𝐡𝐨𝐨𝐥 𝐑𝐍𝐍𝐬 𝐜𝐚𝐧 𝐚𝐜𝐭𝐮𝐚𝐥𝐥𝐲 𝐫𝐢𝐯𝐚𝐥 𝐟𝐚𝐧𝐜𝐲…&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/papers/2410.01201">论文页面 - Were RNNs All We Needed?</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426">unsloth/unsloth/models/llama.py at ae9e264e33c69b53dd5d533a4c5a264af4141c28 · unslothai/unsloth</a>: 微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1291844096457179207)** (731 messages🔥🔥🔥):

> - `AGI and AI reasoning`
> - `Hugging Face models`
> - `Gradio Spaces`
> - `LLM performance`
> - `Synthetic data generation` 


- **关于 AGI 与 AI 推理的辩论**：围绕 AGI 是否可以实现展开了讨论，有人断言它仍然是一个严重依赖概率的数学结构，类似于人类大脑的运作方式。
   - 参与者辩论了 LLM 推理与人类思维过程的不同解释，一些人声称两者在本质上是相似的。
- **Hugging Face 与模型上下文窗口**：参与者询问了 Hugging Face 上可用模型的上下文窗口（context windows），例如 Llama 3.1 以及 HuggingChat 中的不同配置。
   - 用户讨论了他们在内存限制方面的经验，以及在云服务上使用 Llama 3.1 等高上下文模型的成本。
- **Gradio Spaces 与训练模型**：有关于使用 Gradio Spaces 部署模型以及与并发和安全处理用户信息相关问题的对话。
   - 一位用户表达了对运行推理任务以及优化脚本以避免资源浪费并最大化效率的担忧。
- **AI 中的合成数据生成**：讨论涉及了在 AI 自身的输出上进行训练导致模型崩溃（model collapse）的概念，以及使用合成数据的潜在好处和陷阱。
   - 参与者指出，虽然合成数据可以在初始训练周期（epochs）中提高性能，但存在过拟合（overfitting）的风险，并最终损害模型的可靠性。
- **关于 AI 与硬件的技术查询**：用户发布了关于不同 PCIe 代际之间性能差异及其对推理时间影响的技术咨询。
   - 讨论还涉及了模型根据输入进行自我微调（fine-tune）的潜力，引发了关于此类方法效率和有效性的疑问。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://leo62-glitch.github.io/https-leoAI.com/">Passion for Technology</a>：未找到描述</li><li><a href="https://www.kaggle.com/datasets/jef1056/discord-data/data">Discord-Data</a>：长上下文、匿名化、干净的多轮和单轮对话数据集</li><li><a href="https://x.com/_philschmid/status/1842494809719640309?t=qB7_Vp7Ps3Ufc4T1toORMA&s=19">Philipp Schmid (@_philschmid) 的推文</a>：LLM 真的擅长数学吗？一篇新论文揭示了 LLM 在单个数学问题上表现强劲，但在链式问题上表现吃力，因为其中一个问题的答案会影响下一个。这个原因...</li><li><a href="https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/model.safetensors?download=true">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/argilla/synthetic-data-generator">Synthetic Data Generator - argilla 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/autotrain-projects/train-flux-lora-ease/discussions/8">autotrain-projects/train-flux-lora-ease · 找不到仓库..</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/allenai/reward-bench">Reward Bench Leaderboard - allenai 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/manage-cache">管理 huggingface_hub 缓存系统</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Butter_tea">酥油茶 - 维基百科</a>：未找到描述</li><li><a href="https://tenor.com/view/bugs-bunny-looney-tunes-cartoons-gif-25067683">兔八哥 乐一通 GIF - 兔八哥 乐一通 卡通 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/reidsonm-gif-21586450">Reidsonm GIF - Reidsonm - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/no-donkeys-shrek-gif-16041065">No Donkeys GIF - No Donkeys Shrek - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/judge-judy-hurry-today-tapping-gif-8723777">Judge Judy Hurry GIF - Judge Judy Hurry Today - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/no-sleep-staying-up-insomnia-coffee-weak-gif-21941823">No Sleep Staying Up GIF - No Sleep Staying Up Insomnia - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/doopiidoop/status/1843009342536286329?s=46">doopiidoo (@doopiidoop) 的推文</a>：鱼在晚餐前会梦见什么？</li><li><a href="https://huggingface.co/blog/ImranzamanML/fine-tuning-1b-llama-32-a-comprehensive-article">微调 1B LLaMA 3.2：包含代码的全面分步指南</a>：未找到描述</li><li><a href="https://tenor.com/view/hehe-hee-smile-steve-harvey-gif-7550012">Hehe Hee GIF - Hehe Hee Smile - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/the-goonies-comedy-adventure-hey-you-guys-sloth-gif-3531366">Hey You Guys GIF - The Goonies Comedy Adventure - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sus-cat-2-suspicious-cat-the-cat-looks-suspiciously-cat-sits-in-front-of-food-the-ginger-cat-is-watching-gif-14890167989997543813">Sus Cat 2 Suspicious Cat GIF - Sus Cat 2 Suspicious cat The cat looks suspiciously - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://ollama.com/unclemusclez/unsloth-llama3.2/tags">标签 · unclemusclez/unsloth-llama3.2</a>：使用 Unsloth 的 Llama 3.2</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo/blob/main/app.py">app.py · ggml-org/gguf-my-repo (main 分支)</a>：未找到描述</li><li><a href="https://huggingface.co/docs/trl/en/sft_trainer#train-on-completions-only">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://goodfirstissue.dev">Good First Issue：完成你的第一次开源贡献</a>：未找到描述</li><li><a href="https://www.reddit.com/r/datasets/comments/la6zuq/massive_multiturn_conversational_dataset_based_on/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/blog/aws-marketplace">AWS Marketplace 上的 Hugging Face Hub：使用您的 AWS 账户支付</a>：未找到描述</li><li><a href="https://repost.aws/knowledge-center/accepted-payment-methods">了解 AWS 接受的付款方式</a>：我想知道可以使用哪些付款方式支付我的 AWS 账单。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80%</a>：微调 Llama 3.2, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.imdb.com/title/tt20420538/">

All the Names of God (2023) ⭐ 5.7 | 动作, 冒险, 剧情</a>: 1h 45m</li><li><a href="https://github.com/huggingface/transformers/blob/5ef432e4742cc505f610f8e54ac1cd2e1dfd265e/src/transformers/utils/hub.py#L102">transformers/src/transformers/utils/hub.py at 5ef432e4742cc505f610f8e54ac1cd2e1dfd265e · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的 Machine Learning。 - huggingface/transformers</li><li><a href="https://github.com/python/cpython/pull/113465">GH-113464: 由 brandtbucher 提交的 copy-and-patch JIT 编译器 · Pull Request #113465 · python/cpython</a>: 那是圣诞节前夜，代码世界一片寂静，没有核心开发者在合并代码，甚至连 Guido 也没有；CI 在 PR 上认真运行，期待着绿色的勾选标记很快就会出现……</li><li><a href="https://github.com/huggingface/transformers/issues">Issues · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供最先进的 Machine Learning。 - Issues · huggingface/transformers</li><li><a href="https://github.com/huggingface/diffusers/issues">Issues · huggingface/diffusers</a>: 🤗 Diffusers: 为 PyTorch 和 FLAX 提供最先进的用于图像和音频生成的扩散模型。 - Issues · huggingface/diffusers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1291858662674923570)** (13 messages🔥): 

> - `上传模型到 HuggingFace`
> - `学习 Flutter 和 Dart`
> - `Synthetic data`
> - `Fine-tuning 模型`
> - `配置 Python 和 Jupyter` 


- **上传模型到 HuggingFace 的挑战**：一位成员正在学习如何正确地将模型上传到 HuggingFace 控制台，发现他们参考的教程已经过时且不完整，因为许多模型需要额外的 .json 文件。
   - 他们现在正在 YouTube 上寻找更及时的示例。
- **对 Flutter 和 Dart 的热情**：一位成员表达了学习 Flutter 和 Dart 的乐趣，发现它比 Jetpack Compose 更容易，并且在大多数任务中比 Kotlin 更倾向于使用 Dart。
   - 他们强烈推荐 Flutter 作为一个出色的开发框架。
- **对 Synthetic data 的好奇**：一位成员询问了关于 Synthetic data 的信息，承认自己太懒而不想创建自己的数据集。
   - 这个问题反映了对替代数据生成方法的普遍兴趣。
- **Fine-tuning 模型的挣扎**：一位用户开始研究 Fine-tuning 模型，并为监督式 Fine-tuning 创建了一个 Alpaca 数据集，但发现初步结果令人失望，并将其形容为“一团糟（fire dumpster）”。
   - 在意识到这比使用基础模型更复杂后，他们计划明天重新审视这个主题。
- **配置 Python 和 Jupyter**：一位成员开始在笔记本电脑上配置 Python 和 Jupyter，包括安装包和下载用于本地运行的模型。
   - 这一基础步骤对于他们即将开展的 Machine Learning 工作至关重要。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1291843624526676008)** (9 条消息🔥): 

> - `Nvidia 的 AI 模型`
> - `文本转歌唱模型 (Text to Singing Model)`
> - `Sci Scope 通讯`
> - `Qwen2.5 微调`
> - `MIDI 生成器性能` 


- **Nvidia 发布新款 AI 模型，旨在与 GPT-4 竞争**：Nvidia [投下了重磅炸弹](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/)，推出了其新款 AI 模型，该模型被描述为开源且体量巨大，旨在与 **GPT-4** 展开竞争。
   - *这可能会撼动 AI 领域的格局*，开发者和研究人员对其功能和潜力充满期待。
- **寻找文本转歌唱模型**：一位成员表示希望有一种方法可以将文本转换为歌唱，以便在传统的歌唱环境之外使用。
   - *这引发了人们对创新框架的好奇心*，这些框架可能有助于弥补 AI 在这一领域的空白。
- **探索 Sci Scope 获取 AI 研究更新**：Sci Scope 将主题相似的新 [ArXiv 论文](https://sci-scope.com) 分组并进行总结，提供简洁的每周概览。
   - 该平台现在提供个性化版本，确保用户收到与其兴趣相关的定制论文列表。
- **Qwen2.5-3B 微调效果超出预期**：通过使用 @arcee_ai 的 EvolKit，一位成员开发了 **Raspberry**，这是一个 Qwen2.5-3B 的微调版本，据称其表现优于 **Llama3.1-8B-Instruct**。
   - 该过程使用了包含 **25k** 个数学和编程问题的训练集，这对训练方法具有有趣的启示意义。
- **MIDI 生成器获得好评**：一位成员称赞了 MIDI 生成器，指出了它的有效性，并鼓励探索其潜力。
   - 这突显了人们对通过 AI 技术增强音乐创作工具的持续关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sci-scope.com">Sci Scope</a>：一个关于 AI 研究的 AI 生成通讯</li><li><a href="https://x.com/stablequan/status/1843007532173811726">来自 qnguyen3 (@stablequan) 的推文</a>：在复杂问题上训练 LLM 是否会带来智能？我认为是的。使用 @arcee_ai 的 EvolKit，我为 Qwen2.5-72B 创建了 25k 个困难的数学和编程问题。结果如何？欢迎 Raspbe...</li><li><a href="https://huggingface.co/datasets/arcee-ai/EvolKit-20k">arcee-ai/EvolKit-20k · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1291837628760719451)** (20 条消息🔥): 

> - `意识预测方程 (Sentience Prediction Equation)`
> - `高阶张量的量化方法`
> - `SimpleTuner 框架`
> - `OpenAI 并行补全 API`
> - `SuperWikiImage 数据集发布` 


- **使用 SPE 探索 AI 意识**：一篇新文章提出了 **意识预测方程 (SPE)**，用于评估 AI 何时可能实现意识，并幽默地质疑了诸如披萨上加菠萝之类的存在主义担忧。
   - 文章将此与 **德雷克公式 (Drake Equation)** 类比，认为当今 AI 的进步引发了对其潜在未来的深度哲学思考。
- **引入创新的量化方法**：一位成员宣布了一种针对高阶张量开发的 **新量化方法**，并展示了一个涉及猫图像的演示示例。
   - 该方法旨在提高特定张量应用的效率和性能。
- **SimpleTuner v1.1.1 发布**：新发布的 **SimpleTuner v1.1.1** 将 NF4 训练集成到框架中，支持在 **10G 显存 GPU** 上进行高级训练配置。
   - 特性包括自定义时间步分布设置，可提高性能，特别是在 Linux 环境下。
- **通过并行化增强 OpenAI API**：一位用户开发了一个用于 **OpenAI chat completion** 的类，支持并行推理以提高模型性能和效率。
   - 该设置允许用户在同时处理多个请求时管理 Batch Size 并跟踪 API 使用情况。
- **大规模发布维基百科 CC 图片**：一位成员宣布提供约 **700 万** 张来自维基百科的 CC 授权图片，采用 webdataset 格式以供广泛使用。
   - 他们强调了涉及的许可复杂性，并提供了访问 [数据集](https://huggingface.co/datasets/recursal/SuperWikiImage-7M) 的途径。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1u3edc6FmWmBluwylA_1YDie7Tbh_3QTr?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://gillandsiphon.pythonanywhere.com/">Word Game</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Pixeltable/Multi-LLM-RAG-with-Groundtruth-Comparison">Multi LLM RAG With Groundtruth Comparison - Pixeltable 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://medium.com/@ryanfoster_37838/the-sentience-prediction-equation-when-will-ai-achieve-sentience-and-should-we-be-worried-bf5fa0042408">意识预测方程：AI 何时会获得意识？（我们应该担心吗？）</a>：你已经听到了传闻：AI 正在变得越来越聪明。它在写小说、制作迷因、诊断疾病，甚至，嗯，正在生成这个……</li><li><a href="https://huggingface.co/datasets/recursal/SuperWikiImage-7M">recursal/SuperWikiImage-7M · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://tenor.com/view/3po-star-wars-this-is-madness-gif-13899583">3po 星球大战 GIF - 3po 星球大战 这太疯狂了 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/honest-word-its-honest-work-it-aint-much-it-aint-much-but-its-honest-work-gif-13763573">活儿不多，但这是诚实的工作。 GIF - 诚实的话语 诚实的工作 活儿不多 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v1.1.1">版本发布 v1.1.1 - 带来土豆模型 · bghira/SimpleTuner</a>: 使用 NF4 通过 PagedLion8Bit 进行训练。为 Flux 提供了新的自定义时间步分布，通过 --flux_use_beta_schedule, --flux_beta_schedule_alpha, --flux_beta_schedule_beta (#1023)。时髦的 AdEMAMix，它的 8...</li><li><a href="https://gist.github.com/djellalmohamedaniss/addc4a6d512bb3c3256cc2bae71594a5">parallel inference openai completion API</a>: 并行推理 openai completion API。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/ragesh2000/AutoQAPairGen">GitHub - ragesh2000/AutoQAPairGen</a>: 通过在 GitHub 上创建账号，为 ragesh2000/AutoQAPairGen 的开发做出贡献。</li><li><a href="https://github.com/Alvi-alvarez/sd-Img2img-batch-interrogator">GitHub - Alvi-alvarez/sd-Img2img-batch-interrogator: 适用于 AUTOMATIC1111 的 Stable Diffusion web UI 的 Img2img 批量反推器</a>:  适用于 AUTOMATIC1111 的 Stable Diffusion web UI 的 Img2img 批量反推器 - Alvi-alvarez/sd-Img2img-batch-interrogator</li><li><a href="https://huggingface.co/KingNish/Reasoning-0.5b">KingNish/Reasoning-0.5b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/KingNish/reasoning-base-20k">KingNish/reasoning-base-20k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://apps.apple.com/us/app/gary4beatbox/id6736522400">‎gary4beatbox</a>: ‎gary 获取你的输入音频并随之发挥。这个版本旨在延续你的 beatbox。使用麦克风录制，支持或不支持降噪，如果需要还可以设置预备拍……</li><li><a href="https://github.com/betweentwomidnights/gary-backend-combined">GitHub - betweentwomidnights/gary-backend-combined: gary4live 和 gary4web 的后端</a>: gary4live 和 gary4web 的后端。通过在 GitHub 上创建账号，为 betweentwomidnights/gary-backend-combined 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1291837752127783055)** (12 条消息🔥): 

> - `Original Research Sharing` (原创研究分享)
> - `Weekly Reading Group` (每周读书会)
> - `Combinatorial Limit Theory` (组合极限理论)
> - `ML Model Compression` (ML 模型压缩)
> - `Universal Approximation Theorems` (通用逼近定理)


- **原创研究展示场所**：成员们讨论了在读书会期间于 Discord 社区内展示原创研究的可能性。
   - 一位成员提到，他们可以分享过去的录音和记录，以协助潜在的演讲者进行准备。
- **每周读书会详情**：读书会通常在 **周六下午 1 点** 举行，如果演讲者有空，时间可以灵活调整。
   - 过去已经进行过多次演示和演讲，表明该环境非常支持研究分享。
- **使用组合极限理论的创新方法**：一位成员讨论了他们的 [preprint](https://arxiv.org/abs/2410.01799) 以及过去关于使用 **组合极限理论 (combinatorial limit theory)** 压缩 **7B LLM** 的演讲。
   - 他们强调了各种压缩技术，以及涉及用于图像压缩的 **高阶张量 (higher order tensors)** 的应用。
- **对 ML 模型压缩研究的兴趣**：该研究者的重点并非完全在 ML 上，但他们指出涉及符号向量的 **matmul/matvec propagation** 在 **avx512/avx10** 架构上表现出更好的性能。
   - 他们鼓励其他人探索这一途径，同时提到了他们记录的一些 **简单的通用逼近定理 (universal approximation theorems)**。
- **对 PDF 文件过大表示歉意**：该成员对研究报告 PDF 体积过大表示遗憾，原因是图像在渲染前未进行压缩。
   - 他们保证在未来的草案中会解决这个问题，展现了改进工作的决心。



**提到的链接**：<a href="https://medium.com/@ryanfoster_37838/the-sentience-prediction-equation-when-will-ai-achieve-sentience-and-should-we-be-worried-bf5fa0042408">The Sentience Prediction Equation: When Will AI Achieve Sentience? (And Should We Be Worried?)</a>：你已经听到了传闻：AI 变得越来越聪明。它正在写小说、制作迷因、诊断疾病，甚至，嗯，正在生成这段文字……

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1292577175727247484)** (11 条消息🔥): 

> - `Grounding Dino`
> - `Detection of Oriented Objects` (有向目标检测)
> - `DETR Model Fine-tuning Issues` (DETR 模型微调问题)
> - `Smoothing in CNN Autoencoders` (CNN Autoencoder 平滑问题)
> - `Extending Character Set in TrOCR` (扩展 TrOCR 字符集)


- **Grounding Dino 和 Florence-2 模型建议**：一位成员建议尝试 **Grounding Dino** 或 **Florence-2 模型**，并指出即使无法达到实时性能，结果也可能会有所改善。
   - 他们还提到可以使用 **GPT-4V** 和 **Molmo-7B** 等大模型来增强 UI 能力。
- **有向目标检测 (Oriented Object Detection) 选项**：成员们讨论了有向目标检测，确认了 **YOLO v8 OBB** 的存在，并提到了 **Rotated Faster R CNN**、**Rotated RetinaNet**、**Oriented R CNN** 和 **Gliding Vertex** 等替代方案。
   - 一位成员对这些指导表示感谢，表明其重点在于寻找合适的检测器。
- **微调后的 DETR 模型边界框问题**：一位用户反映，在对均匀分布的对象进行测试后，微调后的 **DETR 模型** 出现了边界框 (bounding boxes) 不准确的问题，特别是在图像的右下角区域。
   - 他们提供了一个链接以获取有关该问题的更多上下文：[Inaccurate bboxes after finetuning](https://discuss.huggingface.co/t/inaccurate-bboxes-after-finetuning-detr/109736)。
- **CNN Autoencoder 输出平滑化**：一位成员询问了导致 **CNN Autoencoder 输出** 观察到 **平滑 (smoothing)** 现象的原因。
   - 随后他们询问了实现减少平滑输出的潜在方法。
- **扩展 TrOCR 中的字符集**：一位用户询问了在 **TrOCR 模型** 中扩展字符集或词典的难度，并寻求有关该过程的建议。
   - 他们要求相关回复直接针对他们进行回答。



**提到的链接**：<a href="https://discuss.huggingface.co/t/inaccurate-bboxes-after-finetuning-detr/109736">Inaccurate bboxes after finetuning DETR</a>：我按照目标检测指南微调了一个 DETR 模型。然而，图像左上角对象的预测边界框往往比右下角更准确（t...

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1292222727284981761)** (12 条消息🔥): 

> - `ollama 和 LLaMA3.1 摘要问题`
> - `Google T5 模型本地执行`
> - `使用原始方法进行日志数据分析`
> - `从 Hugging Face 加载模型的挑战` 


- **ollama 在摘要任务上表现不佳**：一位用户报告了使用 **ollama** 和 **LLaMA3.1 70b** 摘要长文本时的问题，发现输出内容过于浅显，且仅关注输入的最后一部分。
   - 他们质疑上下文窗口大小（context size）或提示词（prompting）是否影响了摘要质量，并表示决心改进这一流程。
- **本地运行 Google T5 的困扰**：尽管遵循了仓库的说明和示例，一位用户在本地运行 **Google T5** 模型时仍面临困难。
   - 社区成员建议检查错误信息，并考虑防火墙问题是否是影响配置的潜在原因。
- **探索日志数据分析技术**：一位成员询问是否可以使用 **PCFG parsers** 等原始方法或无监督方法进行日志数据分析，而不是使用沉重的 ML/DL 算法。
   - 他们正在寻求能够从日志数据中生成高质量模板的资源，表明其研究方向正转向更简单的方法论。
- **从 Hugging Face 加载模型的困惑**：一位用户询问从 **Hugging Face** 加载模型是否会产生费用，得到的答复是无需支付任何费用。
   - 另一位用户在加载模型时遇到了错误，特别是与缺失 **onnx/decoder_model_merged_quantized.onnx** 文件相关的错误，凸显了潜在的加载问题。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1291915070648877057)** (20 条消息🔥): 

> - `处理显存溢出 (OOM) 错误`
> - `Flux 1.1 Pro 模型发布`
> - `配合 T5 Encoder 运行 Flux Dev`
> - `AutoencoderKL 中的预训练权重`
> - `优化 Diffusion 推理` 


- **处理显存溢出错误的策略**：用户在尝试运行 diffusion 模型时遇到了 **64GB 显存溢出错误**，这可能是由于在 CPU 上以全精度加载权重，而非在 GPU 上以半精度加载所致。
   - 建议包括阅读有关减少内存使用的优化文档，并参考 Hugging Face 的最佳实践指南。
- **Flux 1.1 Pro 宣称的效率**：Flux 1.1 Pro 声称比 Flux 1 快 **5-6 倍**，但事实证明它实际上比 **Flux 1 dev** 快约 **2 倍**，比 **Flux 1 pro** 快 **6 倍**。
   - 尽管其**成本更高**，该模型的效率提升可能源于通过蒸馏（distillation）实现的尺寸缩减或优化的步长映射（step mappings）。
- **配合 T5 Encoder 运行 Flux Dev**：一位用户寻求关于将 **T5 encoder** 与 Flux Dev 集成的建议，以提高在低 VRAM 设备上的效率。
   - 推荐方案包括探索 **torchao** 等替代方案，据报道这些方案在保持质量的同时能更好地适配 **16GB VRAM** 的设备。
- **在 AutoencoderKL 中使用预训练权重**：一位用户询问如何在修改输入和输出通道的同时，将预训练权重加载到 **AutoencoderKL** 类中。
   - 讨论强调了在当前框架下实现这一点的难度，建议依靠量化（quantization）方法作为解决方案。
- **优化 Diffusion 推理过程**：分享了关于推理过程的通用建议，性能权衡很大程度上取决于 VRAM 和质量要求。
   - 提到的一种有效方法是使用 **torch.compile**，但它可能会减慢初始推理速度，且无法在不同的 LoRA 模型之间轻松切换。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main">comfyanonymous/flux_text_encoders at main</a>：未找到描述</li><li><a href="https://github.com/huggingface/diffusers/discussions/6609)">在性能较低的 GPU 上实现更快的 Diffusion ⚡️ · huggingface/diffusers · Discussion #6609</a>：我们最近发布了《加速生成式 AI 第三部分：Diffusion 篇》，展示了如何：我们在 80GB A100 上进行了演示。文中介绍的技术在很大程度上适用于相对...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1292520195474325619)** (2 messages): 

> - `Service Outage`（服务中断）
> - `Share API Issues`（Share API 问题）
> - `Share Links Services`（Share Links 服务）


- **报告持续的服务中断**：2024年10月6日，有报告称 **Share API** 和 **Share Links** 服务正经历持续中断，建议用户查看 [状态页面](https://status.gradio.app/) 获取更新。
   - 团队承认了这些问题对用户工作的影响，并承诺将尽快解决。
- **服务已解决且系统在线**：不久后传来了好消息，宣布所有系统已恢复在线，影响 **Share API** 和 **Share Links** 的问题已完全解决。
   - Gradio 团队感谢用户的耐心等待，并对停机期间造成的任何不便表示歉意。



**提及的链接**：<a href="https://status.gradio.app/">Gradio Status</a>：未找到描述

  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1292077740354179082)** (14 messages🔥): 

> - `LLM Trainer in Rust and Triton`
> - `Cloud Provider Recommendations`
> - `HBM Manufacturing Insights`
> - `Text to VFX Dataset Search`
> - `Discussion on Glue and DRAM Scaling` 


- **Sasha 开放 LLM 训练器合作**：一位成员分享了一条推文，表示很想花 **100 小时用 Rust 和 Triton 编写一个 LLM 训练器**，**Sasha** 表示可以提供咨询或合作。
   - 这可能会在社区的 LLM 训练方法上带来创新性的发展。
- **为小型集群寻找云服务商**：一位成员询问了适合小型集群的**云服务商推荐**，要求能够轻松使用 nsys 进行性能分析，并强调不需要 H100s。
   - 几位成员讨论了他们的偏好，表明社区对可获取的计算资源很感兴趣。
- **对 HBM 制造的见解**：一位成员分享了他们对 **HBM 制造方式**的新理解，称其非常疯狂，随后讨论了 Graphcore CTO 提出的扩展性担忧。
   - 成员们幽默地回应，引用“将多层 DRAM 粘在一起”来质疑其扩展性。
- **搜索 Text to VFX 数据集**：一位成员表示有兴趣训练一个 **text to VFX** 模型，但找不到合适的数据集，并向社区寻求推荐。
   - 该咨询凸显了在视觉效果特定模型训练方面，可用资源的潜在缺口。
- **关于胶水的哲学幽默**：在一段轻松的交流中，成员们评论了“胶水的神秘属性”，引用其与 DRAM 扩展的关系，并穿插了披萨胶水的比喻。
   - 这反映了社区将技术讨论与幽默融合的能力，保持了活跃的氛围。



**提及的链接**：<a href="https://x.com/srush_nlp/status/1777453605336854545">Sasha Rush (@srush_nlp) 的推文</a>：天哪。现在我真的很想花 100 小时用 Rust 和 Triton 编写一个 LLM 训练器。

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1292009705782247454)** (14 messages🔥): 

> - `Matrix Multiplication Tutorial`（矩阵乘法教程）
> - `Triton Kernel Updates`（Triton Kernel 更新）
> - `FP8 Matrix Handling`（FP8 矩阵处理）
> - `BF16 vs FP32 Computations`（BF16 与 FP32 计算） 


- **理解 FP8 中的矩阵转置**：成员们讨论了在进行 FP8 矩阵乘法时转置第二个矩阵的必要性，特别是 Triton 如何处理矩阵布局，其中第二个矩阵预期为列优先（column-major）格式。
   - 有人建议这种列优先的要求可能会带来性能优势，而其他人则寻求澄清转置操作是否会影响不同数据类型的性能指标。
- **将 Triton Kernel 更新为 BF16**：一位用户询问如何更新 Triton Kernel 以利用 BF16，但由于除加法和减法外的多数操作都会自动转换为 FP32 而面临挑战。
   - 讨论强调了混合精度的策略，建议在 FP32 中进行计算以保证精度，并主要将 BF16 用于矩阵乘法，并分享了如何妥善处理 Tensor 操作的细节。
- **BF16 vs FP32 和 TF32**：一位成员询问使用带有 TF32 的 FP32 计算是否比使用 BF16 表现更差，强调了理解不同数据类型精度差异的重要性。
   - 回复指出，更倾向于采用能最大化精度的流程，特别是在需要更高准确度的操作期间，并承认了 BF16 和 FP32 之间支持操作的差异。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1291998309086400556)** (47 messages🔥): 

> - `DALI Dataloader Performance` (DALI Dataloader 性能)
> - `FFCV advantages` (FFCV 优势)
> - `Multi-threaded Dataloader` (多线程 Dataloader)
> - `Data Loading Bottlenecks` (数据加载瓶颈)
> - `Integration of DALI with PyTorch` (DALI 与 PyTorch 的集成)


- **DALI Dataloader 在性能方面表现出色**：用户强调 DALI Dataloader 每秒可以读取 **5,000 张 512x512 JPEG 图片**，并能高效利用 GPU 资源进行大图像变换，尽管*其设置需要花费一定精力*。
   - 一位成员指出，DALI 在完整的 **ImageNet transforms** 下具有令人印象深刻的吞吐量，无论训练什么模型，减速都微乎其微。
- **FFCV 提供卓越的训练加速**：FFCV 独特的技术（如缓存和异步数据传输）能够**显著改善数据加载**，实现极高的 GPU 利用率并缩短训练时间。
   - 一位成员分享道，FFCV 允许在**单机上仅用 20 分钟**即可将 **ImageNet ResNet-50** 模型训练至 **75%** 的准确率。
- **关于多线程 Dataloader 进展的讨论**：目前正在进行的工作旨在通过**带有和不带有 GIL 的多线程处理**来增强数据加载，这在最近的一次活动中得到了展示。
   - 成员们有兴趣与 DALI 团队合作以潜在地利用其能力，但正如分享的那样，**并非所有用户都偏好 DALI**。
- **流式数据集（streaming datasets）的挑战**：针对 FFCV 对流式传输的支持提出了疑问，指出它目前仅处理本地数据集，并且需要重新摄取（re-ingestion）为专有格式。
   - 随后讨论了 FFCV 对某些操作的优化，而一些参与者对其流式传输能力表示怀疑。
- **Dataloader 对 GPU 加速的需求**：成员们承认在某些预处理操作中 **GPU 加速的潜力**，但也指出*某些任务（如图像解码）在 GPU 上并不可行*。
   - 进一步的实验表明，尝试使用 `torch.compile` 融合变换操作会导致性能变慢，这引发了对其在各种设置中有效性的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2306.12517">FFCV: Accelerating Training by Removing Data Bottlenecks</a>：我们介绍了 FFCV，一个用于简单快速地训练机器学习模型的库。FFCV 通过消除训练过程中（通常是微妙的）数据瓶颈来加速模型训练。特别是，我们...</li><li><a href="https://github.com/pytorch/torchcodec">GitHub - pytorch/torchcodec: PyTorch video decoding</a>：PyTorch 视频解码。通过在 GitHub 上创建账号为 pytorch/torchcodec 的开发做出贡献。</li><li><a href="https://github.com/libffcv/ffcv">GitHub - libffcv/ffcv: FFCV: Fast Forward Computer Vision (and other ML workloads!)</a>：FFCV：快速推进计算机视觉（以及其他 ML 工作负载！）。</li><li><a href="https://github.com/NVIDIA/DALI/blob/2d9d526fa2909f0758336f39a48bae07e9bb2159/dali/python/nvidia/dali/auto_aug/auto_augment.py#L222-L296">DALI/dali/python/nvidia/dali/auto_aug/auto_augment.py at 2d9d526fa2909f0758336f39a48bae07e9bb2159 · NVIDIA/DALI</a>：一个 GPU 加速库，包含高度优化的构建模块和执行引擎，用于数据处理，以加速深度学习训练和推理应用。 - NVIDIA/DALI
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1292334116288725013)** (1 messages): 

> - `Quantized Optimizers` (量化优化器)
> - `INT8 Quantized Training` (INT8 量化训练)
> - `TorchAO` (TorchAO)
> - `Zoom Meetings` (Zoom 会议)


- **关于量化优化器的精彩讨论**：一场活动将在 **5 分钟内**开始，届时将有一位资深成员介绍如何在 TorchAO 中实现**量化优化器**和 **INT8 量化训练**。
   - 邀请参与者通过 **Zoom** 加入讨论，提升在这些高级主题方面的知识。
- **在 Zoom 上加入我们**：会议将在 **Zoom** 上举行，为成员提供一个互动平台进行交流和学习。
   - 对于成员来说，这是一个深入了解 **TorchAO** 功能的绝佳机会。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1292583082003791952)** (1 条消息): 

> - `Phrack archives`
> - `Reading formats` 


- **Phrack 71 第 17 期访问**：一个使用 `wget` 的共享命令演示了如何通过终端以简化的方式访问 **Phrack** 第 71 期，特别是第 17 篇文章。
   - *一位用户评论说他们更喜欢以有趣的方式阅读*，展示了对替代阅读体验的兴趣。
- **有趣的阅读方式**：一位用户评论了以不同风格阅读的乐趣，强调了阅读格式的差异。
   - 这一笔记表明了对以非传统方式接触内容的偏好。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1292101894612848660)** (113 条消息🔥🔥): 

> - `Shared Memory in CUDA`
> - `Parallelizing RNNs with CUDA`
> - `Lookahead Decoding`
> - `Quantization in LLMs` 


- **探索 CUDA 中的 Shared Memory**：成员们讨论了在 CUDA 中使用 `__shared__` 在 block 内创建共享内存，但质疑是否存在适用于 block/grid 级别的类似方法。
   - 进一步的对话显示，这些声明发生在 kernel 或 device 函数内部。
- **并行化 RNN 引起关注**：研究了使用 CUDA 并行化 RNN 的可能性，并讨论了由于其顺序特性带来的挑战。
   - 成员们提到了最近的作品如 S4 和 Mamba 解决了这一难题，以及表明克服顺序依赖性方法的研究。
- **Lookahead Decoding 介绍**：Lookahead Decoding 被提出作为一种通过并发求解方程来打破 LLM 推理中顺序依赖的方法。
   - 讨论链接到了 [Lookahead Decoding 论文](https://lmsys.org/blog/2023-11-21-lookahead-decoding/#background-parallel-llm-decoding-using-jacobi-iteration) 和一个 GitHub 仓库以供进一步探索。
- **推荐 Quantization 资源**：一位成员寻求关于 LLM Quantization 的全面资料，随后推荐了 Hugging Face 的 [quantization 指南](https://huggingface.co/docs/optimum/en/concept_guides/quantization) 等资源。
   - 有人指出，虽然通用模型量化适用，但针对 LLM 的特定方法往往侧重于 weight-only quantization 以优化内存。
- **量化整数计算的挑战**：一位成员强调了关于使用量化整数进行计算的文档非常稀缺，但推荐了论文 [A Survey of Quantization Techniques](https://arxiv.org/pdf/1712.05877) 以获得清晰的解释。
   - 这次讨论承认了对优化 LLM 性能的有效量化方法的持续关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2023-11-21-lookahead-decoding/#background-parallel-llm-decoding-using-jacobi-iteration">Break the Sequential Dependency of LLM Inference Using Lookahead Decoding | LMSYS Org</a>: &lt;p&gt;&lt;strong&gt;TL;DR:&lt;/strong&gt; 我们介绍了 &lt;strong&gt;lookahead decoding&lt;/strong&gt;，一种新的、精确的、并行的解码算法，用于加速 LLM 推理。Look...</li><li><a href="https://huggingface.co/docs/optimum/en/concept_guides/quantization">Quantization</a>: 未找到描述</li><li><a href="https://github.com/janestreet/torch/blob/master/internals.md">torch/internals.md at master · janestreet/torch</a>: Contribute to janestreet/torch development by creating an account on GitHub.</li><li><a href="https://github.com/machine-discovery/deer/tree/main/experiments">deer/experiments at main · machine-discovery/deer</a>: 在序列长度上并行化非线性顺序模型 - machine-discovery/deer</li><li><a href="https://github.com/machine-discovery/deer/">GitHub - machine-discovery/deer: Parallelizing non-linear sequential models over the sequence length</a>: 在序列长度上并行化非线性顺序模型 - machine-discovery/deer</li><li><a href="https://drive.google.com/drive/folders/1Pz607n07u382_ybdd4gFdrNyEWra5kpj">IRL Keynotes - Google Drive</a>: 未找到描述</li><li><a href="https://github.com/hao-ai-lab/LookaheadDecoding">GitHub - hao-ai-lab/LookaheadDecoding: [ICML 2024] Break the Sequential Dependency of LLM Inference Using Lookahead Decoding</a>: [ICML 2024] 使用 Lookahead Decoding 打破 LLM 推理的顺序依赖 - hao-ai-lab/LookaheadDecoding
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1292334930436685835)** (3 条消息): 

> - `GPU MODE 系列讲座`
> - `讲座观看顺序`
> - `YouTube 上传内容` 


- **GPU MODE 讲座推荐观看顺序**：一位成员建议**按顺序观看第 1-5 讲**以获得最佳理解，然后根据个人兴趣选择后续讲座。
   - 这种方法能让新观众在探索其他主题之前掌握基础概念。
- **关于 YouTube 上传时间表的询问**：一位成员询问了最后一场演讲上传到 YouTube 的预计时间 (ETA)。
   - 这表明人们对该系列讲座及其在在线平台上的可用性持续关注。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1292112464946659413)** (27 条消息🔥): 

> - `TorchAO 中的 NF4 支持`
> - `NF4 带来的性能提升`
> - `使用 bitsandbytes 进行训练`
> - `最近演讲的录像`
> - `CPU 上的 Int4 支持` 


- **TorchAO 对 NF4 的支持备受期待**：成员们表达了对 **TorchAO** 支持 **NF4** 的渴望，并指出其在模型训练中提升性能的潜力。
   - 一位成员指出了现有的 **[NF4 tensor 实现](https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py)**，并建议增强其易用性。
- **NF4 降低了训练 VRAM 需求**：用户注意到 NF4 训练将 **VRAM 最低需求**从 **16G 降低到了 10G**，提供了比标准 **INT4** 更好的功能。
   - 一位成员表示，使用 NF4 后，他们的训练速度从 **每步 11 秒提升到了每步 7 秒**。
- **最近演讲的录像即将发布**：在一位成员对最近的演讲表示赞赏后，另一位成员提到因时间冲突错过演讲而感到遗憾。
   - 主持人表示，录像将在几天内为错过的人提供。
- **Int4 支持的细节澄清**：针对关于 CPU 上 **int4_weight_only()** 的提问，确认了不支持使用 Tensor Core 布局。
   - 不过，似乎还有其他针对 CPU 的 Int4 实现，讨论中提供了相关链接。
- **Torchtune 与 NF4 功能**：对话强调 **Torchtune** 目前是处理 **LoRa 线性层**的最佳选择。
   - 成员们承认了早期版本 **torch.compile()** 的复杂性，以及在未来更新中进行直观集成的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/search?q=repo%3Apytorch%2Fpytorch%20_weight_int4pack_mm&type=code">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/modules/low_precision/nf4_linear.py">torchtune/torchtune/modules/low_precision/nf4_linear.py at main · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/modules/peft/lora.py">torchtune/torchtune/modules/peft/lora.py at main · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py">ao/torchao/dtypes/nf4tensor.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1291859630313701439)** (386 条消息🔥🔥): 

> - `Resume-Review 频道`
> - `研究生申请`
> - `AI Summer 与研究差异`
> - `NVFuser 集成`
> - `Tiramisu 编译器` 


- **关于 Resume-Review 和模拟面试的提议**：一位成员建议开设一个用于简历复盘（Resume-Review）和模拟面试的频道，以帮助针对特定领域的个人，在保持隐私的同时提供切实的反馈。
   - 目前正在讨论此类服务与服务器使命的相关性，因为一些人认为重点应保持在构建开源项目上，而非传统的职业支持。
- **对 AI 研究方向的担忧**：成员们强调，目前的资金严重向 LLM 倾斜，导致其他研究领域（如几何深度学习和该领域的通用创新）陷入停滞。
   - 一位成员对 LLM 扩展（scaling up）缺乏透明度表示沮丧，指出重要的实现细节通常仍是大型公司的专利。
- **对分布式系统和 NVFuser 的兴趣**：讨论了 NVFuser 与 Thunder 的集成，成员们表示有兴趣为编译器架构和优化创建一个更简单、更易于访问的环境。
   - 成员们提到了在处理线程和管理复杂构建系统时遇到的困难，并希望有更精简的工具。
- **多面体编译器（Polyhedral Compiler）概念的探索**：成员们讨论了多面体编译器，特别是 Tiramisu，及其在跨各种平台优化计算方面的潜力，并强调了 Python 在此类工具中的易用性。
   - 对话倾向于编译器技术在机器学习中的效用，以及创建或增强利用现有框架的编译器的愿望。
- **对国际象棋和社区互动的兴趣**：在聊天中分享了国际象棋游戏的邀请，反映了成员之间进行非正式互动和社区联结的愿望。
   - 轻松的闲聊展示了小组的社交属性，成员们鼓励参与技术讨论之外的活动。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/borgir-gif-22149357">Borgir GIF - BORGIR - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理 Git 仓库，像专家一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/pytorch/pytorch/tree/main/torch/csrc/jit/codegen/cuda">pytorch/torch/csrc/jit/codegen/cuda at main · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/Tiramisu-Compiler/tiramisu">GitHub - Tiramisu-Compiler/tiramisu: A polyhedral compiler for expressing fast and portable data parallel algorithms</a>：一个用于表达快速且可移植的数据并行算法的多面体编译器 - Tiramisu-Compiler/tiramisu</li><li><a href="https://tiramisu-compiler.org/">Tiramisu Compiler</a>：一个用于稠密和稀疏深度学习及数据并行算法的多面体编译器</li><li><a href="https://arxiv.org/abs/1804.10694">Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code</a>：本文介绍了 Tiramisu，这是一个旨在为包括多核、GPU 和分布式机器在内的多个平台生成高性能代码的多面体框架。Tiramisu 引入了一个方案...</li><li><a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/thunder/executors/nvfuserex_impl.py#L211-L295">lightning-thunder/thunder/executors/nvfuserex_impl.py at main · Lightning-AI/lightning-thunder</a>：让 PyTorch 模型提速高达 40%！Thunder 是一个针对 PyTorch 的 source-to-source 编译器。它允许同时使用不同的硬件执行器；跨越一个或数千个 GPU。- Lightning-AI/ligh...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1292834300051456000)** (1 条消息): 

> - `train.c 代码`
> - `编程资源` 


- **寻求关于 train.c 代码的澄清**：一位成员对 **train.c 代码** 表示困惑，正在寻找能提供清晰解释的文章。
   - *有没有人知道能清晰解释这些代码的好文章？*
- **征集关于 train.c 的文章**：发出了另一项关于 **train.c** 的查询，特别是寻求能澄清其用法和功能的信息性文章。
   - 鼓励成员们分享相关的资源或见解。


  

---

### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1292887600822091928)** (1 messages): 

> - `Sparsity in Attention vs MLP Layers` 


- **关于 Attention 层中 Sparsity 影响的问题**：一位成员询问，与在 **MLP linear layers** 中应用相同的 **sparsity** 相比，在 **attention linear layers** 中应用 **sparsity** 是否会导致 **model** 运行更慢。
   - 这个问题突出了 **sparsity** 如何与不同的模型架构及其效率相互作用的一个基本方面。
- **Sparsity 应用的性能对比**：讨论了 **sparsity** 的实现如何根据其在 **attention** 或 **MLP** 层中的应用产生不同的性能结果。
   - 参与者指出，这两类层之间的效率可能存在显著差异，这使其成为一个关键的分析点。


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1292700135058444290)** (7 messages): 

> - `WASM Packaging with Onnxruntime`
> - `Onnxruntime Web Optimization`
> - `Custom Inference Logic`
> - `WebGPU Backend Usage` 


- **优化 Onnxruntime Web WASM 大小**：一位成员指出 **Onnxruntime Web** 的默认 **WASM** 大小为 **20 MB**，这表明在打包其自定义推理逻辑时需要进行优化。
   - *Tailoredcub* 提到尚未尝试为他们的模型层探索 **Onnxruntime** 的自定义构建。
- **探索更小的替代方案**：另一位成员分享说他们使用了 [onnxruntime-web](https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js)，其大小仅为 **444K**，但尚未针对自定义计算进行广泛测试。
   - *Tailoredcub* 请求一个开源示例，演示如何将该压缩版本与 **WebGPU backend** 配合使用。
- **关于 LTO 和 Tree Shaking 的疑问**：一位成员对在减小包体积方面使用 **LTO** (Link Time Optimization) 和 **tree shaking** 的潜在选项表示好奇。
   - 这一讨论凸显了在集成自定义逻辑时，人们一直在寻找减少 **Onnxruntime Web** 庞大体积的策略。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1291973281464979458)** (5 messages): 

> - `Torch Compile`
> - `Tensor Parallel Inference`
> - `Liger Kernel Downloads`
> - `Q4 Roadmap` 


- **使用 Torch Compile 进行优化**：一位成员建议直接使用 **torch compile** 以获得更好的优化，并表示 **triton** 实现的效果并不理想。
   - 强化使用 **torch compile** 可能会在 ML 工作负载中实现更高效的执行。
- **Tensor Parallel 推理性能**：在使用 **tensor parallel inference** 的 **flux dev** 上实现了 **12.87 it/sec** 的性能率，尽管其效率受到了质疑。
   - 成员们反思了其性能，并幽默地承认了其较低的计算效率。
- **Liger Kernel 达成重大里程碑**：**Liger Kernel** 在发布仅一个月后下载量就突破了 **100,000+**，庆祝了来自社区的许多成功案例。
   - *他们将继续致力于提升性能并支持更多的 kernel 和模型。*
- **Liger Kernel 的 Q4 路线图**：团队分享了他们的 **Q4 roadmap**，其中包括引入令人兴奋的特性，如 **multimodal** 和 **JSD kernels**。
   - 他们鼓励社区贡献以帮助塑造项目的未来，邀请所有人参与下一个里程碑。



**提及的链接**：<a href="https://x.com/liger_kernel/status/1842661651264503896">Liger Kernel (@liger_kernel) 的推文</a>：🚀 Liger Kernel 在一个月后下载量已突破 **100,000+**！我们对研究社区和企业分享的众多成功案例感到荣幸。我们的承诺依然坚定……

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1292699332956520491)** (4 messages): 

> - `BFloat16 computations`
> - `MLX on Mac machines` 


- **寻求 MLX 中 BFloat16 的加速秘诀**：一位成员询问关于在 Mac 机器上的 **MLX** 中加速 **BFloat16** 计算的见解，他们认可其内存优势，但正在寻求性能提升。
   - 另一位成员询问了正在处理的具体操作，并指出了解上下文可能有助于提供更好的建议。
- **提升性能的转换技巧**：一位成员建议在加载后转换为 **fp32** 以获得潜在更快的计算速度，这暗示了 **BFloat16** 速度的一种变通方案。
   - 然而，一位成员承认对 **MLX** 缺乏了解，指出了在该特定领域的专业知识空白。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1292771413127856151)** (1 messages): 

> - `Sci Scope Newsletter`
> - `ArXiv Papers Summary`
> - `Personalized Research Alerts` 


- **通过 Sci Scope Newsletter 保持更新**：Sci Scope 提供免费的 Newsletter，通过将相似主题分组来总结新的 **ArXiv papers**，以便于导航和选择阅读材料。
   - *立即注册*，直接在收件箱中接收摘要，每周节省研究时间！
- **个性化 Newsletter 发布**：新版个性化 Sci Scope 允许你自定义兴趣，并根据你的偏好发送每周摘要。
   - 通过订阅，你将不再错过与工作相关的任何进展，从而最大限度地提高研究效率。



**提到的链接**：<a href="https://sci-scope.com">Sci Scope</a>：关于 AI 研究的 AI 生成 Newsletter

  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1291875218591191060)** (7 messages): 

> - `gemma.cpp`
> - `ATen Vectorized library`
> - `vpternlogd instruction`
> - `SIMD programming insights` 


- **为 AVX 优化的 Gemma.cpp**：[gemma.cpp 项目](https://github.com/google/gemma.cpp)是一个用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎，使用 highway 库实现并针对 AVX 进行了优化。
   - 一位成员对在该项目的 [ops 目录](https://github.com/google/gemma.cpp/tree/main/ops)中发现的秘密 SIMD Transformer kernel 库表示了极大的兴趣。
- **质疑 ATen 的库选择**：一位成员提出了一个问题：为什么 **ATen** 使用自己的 Vectorized 库而不是 highway 库，并暗示这种选择可能是有特定原因的。
   - 这引发了对设计决策的反思，指出这些架构选择背后的原因尚不明确。
- **发现 vpternlogd 指令**：一篇[博文](https://arnaud-carre.github.io/2024-10-06-vpternlogd/)详细介绍了 **vpternlogd** 指令，这是 AVX-512 中的一种位三元逻辑运算，允许使用三个输入源进行复杂的逻辑运算。
   - 作者将其功能与过去逻辑设计中的挑战进行了比较，暗示了其在现代 SIMD 编程中的潜在应用。
- **关于最小项逻辑设计的记忆**：一位成员回忆起大学逻辑设计中的 **minterms** 和 **maxterms** 概念，并将其与 Amiga 硬件的设计决策联系起来。
   - 他们幽默地表示，该软件的文档可能是由 Amiga 芯片设计师本人起草的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arnaud-carre.github.io/2024-10-06-vpternlogd/">AVX Bitwise ternary logic instruction busted!</a>：现代 AVX 指令如何与 1985 年的 blitter 芯片共享相似设计，作者 Arnaud Carré</li><li><a href="https://github.com/google/gemma.cpp/tree/main/ops">gemma.cpp/ops at main · google/gemma.cpp</a>：用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。- google/gemma.cpp</li><li><a href="https://github.com/google/gemma.cpp">GitHub - google/gemma.cpp: lightweight, standalone C++ inference engine for Google's Gemma models.</a>：用于 Google Gemma 模型的轻量级、独立 C++ 推理引擎。- google/gemma.cpp
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1291887028463472667)** (337 条消息🔥🔥): 

> - `File Organization with AI Tools`（使用 AI 工具进行文件整理）
> - `Challenges of Using AI for Document Categorization`（使用 AI 进行文档分类的挑战）
> - `Differences Between AI Models and Architectures`（AI 模型与架构之间的差异）
> - `Local vs Cloud AI Cost Analysis`（本地与云端 AI 成本分析）
> - `Issues with File Uploading in ChatGPT`（ChatGPT 文件上传问题）


- **自动化文档分类**：用户讨论了利用 AI 工具通过分析内容来对大量文档进行分类的潜力，并给出了如何构建这一流程的示例。
   - 一位用户建议，如果对项目需求缺乏清晰的沟通，可能会阻碍自动化解决方案的进展。
- **使用 OpenAI API 的成本影响**：计算显示，根据 Token 使用量，使用 OpenAI API 分析数千个媒体文件的成本可能会超过 12,000 美元，这构成了巨大的财务障碍。
   - 这引发了关于开发本地解决方案是否更具可行性的讨论，尽管本地方案可能面临与存储和处理相关的高昂成本。
- **关于不同 AI 模型的讨论**：参与者注意到了各种 AI 模型及其能力之间的差异，特别是讨论了 OpenAI o1 模型以及人们对其架构的看法。
   - 对于“新模型代表了对先前架构的彻底背离”这一说法，存在一些怀疑态度，并建议对其设计进行进一步探究。
- **本地 AI 解决方案的挑战**：关于使用本地 AI 解决方案与基于云的 API 相比的效率和成本效益，存在截然不同的观点，一些用户发现本地设置反而更昂贵。
   - 此外，还有人对从不同存储位置提取数据进行统一分析的实用性表示担忧。
- **ChatGPT 中的文件上传问题**：一位用户报告在 ChatGPT 中上传文件时持续遇到困难，上传会中途停止，但在其他设备上则没有问题。
   - 这一问题在多个账户中都有出现，引发了关于可能影响用户体验的平台特定问题的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.getbind.co/2024/09/17/gpt-o1-vs-claude-3-5-sonnet-which-model-is-better-for-coding/">GPT o1 vs Claude 3.5 Sonnet: Which model is better for Coding?, Bind AI</a>: 什么是 GPT o1？在代码生成任务上它比 Claude 3.5 Sonnet 更好吗？阅读关于这两个 AI 模型的详细分析。</li><li><a href="https://topai.tools/s/automated-file-organization">70 Best Automated File Organization AI tools - 2024</a>: 探索 2024 年最佳的 70 个付费和免费 AI 自动化文件整理工具，并了解它们的功能和定价。寻找最适合自动化文件整理的 AI 工具。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1291890143904534648)** (13 条消息🔥): 

> - `Complex Math with GPT-4`（使用 GPT-4 处理复杂数学问题）
> - `Custom GPT Development`（自定义 GPT 开发）
> - `GPT-4 Free Plan Enhancements`（GPT-4 免费版功能增强）
> - `Data Export for ChatGPT Conversations`（ChatGPT 对话数据导出）
> - `Voice Options for Custom GPTs`（自定义 GPT 的语音选项）


- **GPT-4 处理复杂数学任务**：用户注意到 **GPT-4o** 在处理复杂数学方程方面表现相当不错，尤其是与 Wolfram 等插件配合使用时。
   - *另一位成员强调了 GPT 行为的随机性（stochastic nature），建议进一步集成可能会提高可靠性。*
- **创建量身定制的自定义 GPT**：一位用户询问了在 OpenAI 平台上开发自定义 GPT 的最简单方法，该 GPT 需利用 PDF 文档来辅助 zsh 和 macOS 脚本编写。
   - *他们对在不同模型之间切换所浪费的时间感到沮丧，希望能有一个针对其需求的专注工具。*
- **GPT-4 免费版可能的增强功能**：讨论了 OpenAI 是否扩大了免费版的功能范围，用户注意到尽管达到了 4o 的限制，他们仍能访问图像分析功能。
   - *其他人确认，现在即使是 **4o-mini** 也包含了生成和分析图像的能力。*
- **搜索 ChatGPT 对话记录**：一位用户询问如何在多个正在进行的 ChatGPT 对话中搜索特定文本，其中一些对话已超过六个月。
   - *另一位成员建议从设置中请求数据导出（data export），以便于搜索旧的聊天记录。*
- **对自定义 GPT 更多语音选项的需求**：一位用户请求在自定义 GPT 中增加更多语音选项，特别是除了目前的 **Shimmer** 语音之外的男声。
   - *另一位用户表示完全赞同，表达了对语音调制多样性的需求。*


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1292173049025659004)** (61 条消息🔥🔥): 

> - `Optimizing ChatGPT responses` (优化 ChatGPT 回复)
> - `Prompt engineering challenges` (Prompt engineering 的挑战)
> - `Keyword selection for media files` (媒体文件的关键词选择)
> - `Understanding AI communication` (理解 AI 沟通)
> - `Learning preferences in AI usage` (AI 使用中的学习偏好)


- **优化 ChatGPT 回复**：一位用户建议使用复杂的 Prompt 来增强 ChatGPT 的理解和回复质量，表明对模型进行引导（priming）可以获得更准确的回答。
   - 观察到详细的表述可以增强模型回复的一致性。
- **Prompt engineering 的挑战**：对话显示，由于思维方式的不同，一些用户发现很难创建有效的 Prompt，尤其是那些旨在处理特定任务的 Prompt。
   - 建议通过简单地处理需求和反馈，来引导模型获得更好的输出。
- **媒体文件的关键词选择**：用户讨论了根据媒体内容从大量术语中选择关键词的挑战，并对 Prompt 在大小和范围上的限制表示担忧。
   - 建议的方法包括分批处理数据，以简化关键词选择的工作流程。
- **理解 AI 沟通**：一位用户对将自然语言 Prompt 转换为适合 AI 处理的更机械化的格式感到沮丧。
   - 有人提议，AI 可能会通过迭代反馈和实验来调整其输出，以更好地符合用户需求。
- **AI 使用中的学习偏好**：一位用户提到需要算法层面的理解才能进行有效的 AI 交互，而其他人则强调通过实践经验进行学习。
   - 强调了学习和与 AI 交互的不同方法，表明个人适用性因用户而异。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1292173049025659004)** (61 条消息🔥🔥): 

> - `Optimizing ChatGPT's functions` (优化 ChatGPT 的功能)
> - `Keyword selection methodology` (关键词选择方法论)
> - `Prompt engineering` (Prompt engineering)
> - `Communicating with LLMs` (与 LLM 沟通)
> - `Understanding AI learning processes` (理解 AI 学习过程) 


- **优化 ChatGPT 功能的清晰度**：一位用户建议，提高 ChatGPT 分析问题和澄清上下文的能力可以提升性能，特别是在像计算单词中字母数量这样简单的任务中。
   - 如果没有特定的“引导（priming）” Prompt，模型的回复准确性会下降，这引发了关于潜在更新的疑问。
- **从大数据集中有效选择关键词**：一位用户寻求根据媒体文件内容从 12,000 个术语的广泛集合中选择 50 个关键词，这引发了对模型 Context Window 限制的担忧。
   - 讨论包括分批查询模型并提供结构化数据，强调了该任务的复杂性。
- **Prompt engineering 中的挑战**：人们普遍关注 Prompt 构建的复杂性，特别是当用户需要确定性算法来创建 Prompt 时。
   - 一位用户表示难以将 Prompt engineering 概念转化为可操作的步骤，突显了在理解如何有效地向模型传达需求方面的差距。
- **对不同沟通风格的需求**：用户讨论了 LLM 适应非常规沟通风格的需求，其中一人对模拟与 AI 进行有意义的对话感到沮丧。
   - 重点是引导 LLM 更好地理解个人沟通需求，并输出更合适的回复。
- **AI 交互中的多样化学习方法**：参与者强调每个人的学习方式都不同，将理解 AI 比作训犬，技术知识可能对某些学习者有帮助，但并非对所有人都有用。
   - 这个类比强调了不同的背景和经验如何决定用户与 AI 的交互方式以及对其功能的掌握。


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1291892679612960780)** (1 条消息): 

> - `Aider v0.59.0 发布`
> - `/read-only 的改进`
> - `YAML 配置格式的变化`
> - `完整性检查（Sanity Checks）与启动增强`
> - `Bug 修复与性能更新` 


- **Aider v0.59.0 发布公告**：最新版本 **v0.59.0** 包含大量增强功能和 Bug 修复，详细的变更日志可以在[这里](https://aider.chat/HISTORY.html)查看。
   - *Aider 编写了此版本中 77% 的代码*，反映了持续的改进。
- **/read-only 获得重大更新**：`/read-only` 命令现在支持全文件系统的 shell 风格自动补全，此外还支持像 `/add` 一样的仓库文件路径以及 `src/**/*.py` 等通配符（globs）。
   - 这些增强功能有助于更轻松地在项目中导航和管理文件。
- **YAML 配置格式大改**：**YAML** 配置文件格式已更新，采用标准的列表语法 `- list entries`，确保更好的可读性。
   - 此外，`--yes` 标志已重命名为 `--yes-always`，需要更新现有的 YAML 和 `.env` 文件。
- **带有完整性检查的启动更新**：启动期间增加了对 `--editor-model` 的完整性检查，增强了操作的完整性。
   - 此外，现在提供 `--skip-sanity-check-repo` 开关，以加快大型仓库的启动过程。
- **Bug 修复与性能改进**：修复了一个 Bug，确保 **architect mode** 能正确处理 `Control-C`，提升了整体用户体验。
   - repo-map 已改为确定性的（deterministic），并配合改进的缓存逻辑以获得更好的性能。



**链接提及**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 Aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1291837295724331139)** (242 条消息🔥🔥): 

> - `Aider 的使用与配置`
> - `Sonnet 3.5 API 性能`
> - `模型对比与建议`
> - `Git 与 Aider 的集成`
> - `OpenRouter 与 API 密钥管理` 


- **Aider 性能挑战**：用户报告了 Aider 在执行编码任务时长时间卡住的问题，即使是通过云服务商使用 Sonnet 3.5 的企业账户也是如此。
   - 建议包括减少上下文中包含的文件数量，并使用 verbose 标志来诊断问题。
- **探索 Sonnet 3.5 API 替代方案**：讨论指出 OpenRouter 是直接访问 Sonnet 3.5 的更可靠替代方案，因为其频率限制（rate limits）更少且提供多样化的 LLM。
   - 用户注意到 OpenRouter 通常由于额外的支付处理费而导致成本略高，但提供了更好的可用性。
- **最佳编码辅助模型**：用户交流了关于最佳开源编码模型的看法，强调了 Codestral 和 Gemma 2 27b 等模型在特定编码任务中的优势。
   - 共识倾向于使用结合了编码支持和文档查询的模型，尽管目前的局限性也得到了承认。
- **在 Aider 中管理 API 密钥**：在 Aider 中加载 .env 文件以获取 API 密钥时遇到的问题引发了关于 `python-dotenv` 默认行为的讨论，以及改进用户体验的建议。
   - 用户主张对环境变量进行更标准的处理，而一些人则更喜欢使用 shell 函数进行动态 API 密钥管理。
- **Aider 中的多行输入**：一位用户询问如何在 Aider 的 /ask 模式下输入多行消息，寻求更好地格式化带有空行和代码片段的查询的方法。
   - 提供了 Aider 内部命令使用的资源，说明了如何有效地格式化消息。


<div class="linksMentioned">

<strong>链接提及</strong>：

<ul>
<li>
<a href="https://x.com/chatgpt21">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://aider.chat/docs/install.html">安装</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://aider.chat/docs/config/dotenv.html">使用 .env 配置</a>：使用 .env 文件为 aider 存储 LLM API 密钥。</li><li><a href="https://x.com/AlexTobiasDev/status/1842622901293314157">来自 Alex Tobias (@AlexTobiasDev) 的推文</a>：@chatgpt21 现在是什么情况？新的 Anthropic 模型？不会吧</li><li><a href="https://aider.chat/docs/usage/images-urls.html#web-pages">图像与网页</a>：将图像和网页添加到 aider 编码聊天中。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://aider.chat/docs/git.html">Git 集成</a>：Aider 与 Git 紧密集成。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://x.com/_philschmid/status/1842846053608866277">来自 Philipp Schmid (@_philschmid) 的推文</a>：Blog: https://medium.com/@harishhacker3010/can-we-make-any-smaller-opensource-ai-models-smarter-than-human-1ea507e644a0  Prompt: https://gist.github.com/philschmid/34747bf5bc8280f3a5f10f5fd8d1cd4b  Gi...</li><li><a href="https://x.com/claudeai101/status/1843146849617875045?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Claude101 (@claudeai101) 的推文</a>：随着关于明天可能发布新的 Anthropic AI 模型的传闻四起，期待感不断升温。你期望在他们技术的最新迭代中看到哪些进步和功能？</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">聊天内命令</a>：使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://x.com/claudeai101/status/1843206199556387314?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Claude101 (@claudeai101) 的推文</a>：对 Claude 3.5 Opus 的期待正在升温！虽然尚未公布官方发布日期，但 AI 社区正热切期待这款下一代语言模型。你希望在新版本中看到哪些功能...</li><li><a href="https://aider.chat/docs/usage/tips.html">技巧</a>：使用 aider 进行 AI 结对编程的技巧。</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/blob/main/Meta-Llama-3.1-70B-Instruct-IQ4_XS.gguf">Meta-Llama-3.1-70B-Instruct-IQ4_XS.gguf · bartowski/Meta-Llama-3.1-70B-Instruct-GGUF at main</a>：未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--suggest-shell-commands">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://github.com/Aider-AI/aider">GitHub - Aider-AI/aider: aider 是你终端里的 AI 结对编程工具</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/">Aider AI</a>：Aider AI 有 4 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use">anthropic-cookbook/tool_use at main · anthropics/anthropic-cookbook</a>：展示使用 Claude 的一些有趣且有效方法的 Notebooks/食谱集合。 - anthropics/anthropic-cookbook</li><li><a href="https://github.com/github/gitignore">GitHub - github/gitignore: 有用的 .gitignore 模板集合</a>：有用的 .gitignore 模板集合。通过在 GitHub 上创建账号来为 github/gitignore 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1291844177486811248)** (179 messages🔥🔥): 

> - `Aider 功能改进`
> - `Aider 配置与模型设置`
> - `环境变量处理`
> - `在大型代码库中使用 Aider`
> - `将 Aider 与不同编程语言集成` 


- **Aider 的配置与环境管理**：用户建议 Aider 应避免编辑像 `.env` 这样的敏感文件，因为这可能导致 Key 为空或配置错误等问题。
   - 建议通过全新安装来排除故障，使用 `pipx` 可以帮助更有效地管理虚拟环境。
- **重构大型代码文件的挑战**：一位用户对 Aider 处理大型文件的方式表示沮丧，发现在将一个 900 行的 Python 文件拆分为独立类文件等任务中，它的处理速度缓慢且繁琐。
   - 建议包括尝试使用 Sonnet-3.5 等不同模型以获得更高效率，并使用 architect 模式来简化流程。
- **高效地为 Aider 添加上下文**：为了简化上下文添加，建议在启动 Aider 时指定多个文件或文件夹，使用通配符可以帮助一次性包含多个文件。
   - 用户还可以通过 Shell 脚本或 Aider 内置的命令行选项编写脚本命令，以跨多个文件应用更改。
- **在不同编程环境中使用 Aider**：Aider 的功能会根据所使用的编程语言而演进，一些用户指出在 PHP 环境中由于 Docker 交互时缺少功能而遇到困难。
   - 未来改进正在考虑支持 Node.js 等各种环境以及提升跨语言的通用易用性。
- **解决 LiteLLM 和 API Key 的错误**：用户在更新后遇到了 API 错误，排查步骤包括重新安装和配置检查。
   - 常见的解决方案包括确保有效 API Key 的可用性、检查环境变量配置以及验证在不同仓库中的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://192.168.1.6:11434`">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>: 使用 chat, ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 使用 chat, ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 为 aider 编写脚本。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-with-multiple-git-repos-at-once">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>: 将图片和网页添加到 aider 编码聊天中。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>: 使用 aider 进行 AI 配对编程的技巧。</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/llms/other.html#litellm">Other LLMs</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider 使用你的 Git 仓库映射为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的量化基准测试。</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>: Python SDK，代理服务器 (LLM Gateway)，可以调用 OpenAI 格式的 100 多个 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1292035987198709800)** (25 条消息🔥): 

> - `Dracarys 2 模型发布`
> - `Python 3.13 发布`
> - `Flowsie AI 角色机器人使用`
> - `语义搜索讨论`
> - `模拟推理能力` 


- **介绍 Dracarys 2 作为顶尖编程模型**：[@bindureddy](https://x.com/bindureddy/status/1842611268148203883) 发布了 Dracarys 2，声称其超越了 **Sonnet 3.5** 并在 LiveCodeBench 上表现出色，使其在成本和性能方面都具有可行性。
   - 据指出，**Dracarys2-72B-Instruct** 在代码编辑基准测试中获得了 **67%** 的分数，略高于 **qwen-2.5-72b-instruct**，但一些人表示失望，因为它看起来像是更名版本。
- **Python 3.13 重大特性揭晓**：Python 3.13 正式发布，带来了重大更新，包括[更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) 以及在没有 GIL 的情况下运行 Python 的选项。
   - 亮点功能还包括对 **iOS** 和 **Android** 平台的改进支持，由于 Beeware 项目的发展，它们被标记为 **Tier 3 支持**。
- **使用 Flowsie AI 角色机器人**：一位用户成功创建了一个模拟其导师教学风格的 AI 角色机器人，并在 [Twitter](https://twitter.com/10kdesigners) 上分享了他们的进展。
   - 人们对 Flowsie 的可用性提出了担忧，并指出了保存工作流以实现功能所需的必要步骤以及模型支持方面的限制。
- **关于 SQLite 语义搜索的讨论**：一篇关于 **SQLite 混合搜索** 的文章强调了 **语义搜索** 优于传统关键字搜索的优势，通过含义增强了查询结果。
   - 有人提到，完全依赖语义搜索可能会对应用程序有害，并举例说明了精确术语搜索结果不佳的情况。
- **在模型中模拟推理能力**：关于从 **o1 model** 模拟推理能力以改进较低阶模型的潜力，引发了一场有趣的讨论。
   - 这个想法激发了人们对增强目前未达到预期效果的模型性能的方法的好奇心。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2024/Oct/7/whats-new-in-python-313/">Python 3.13 的新特性</a>：今天是 Python 3.13 的发布日。重磅标志性特性包括具有改进错误提示的 [更好的 REPL](https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-better-interactive-interpreter) ...</li><li><a href="https://x.com/bindureddy/status/1842611268148203883">来自 Bindu Reddy (@bindureddy) 的推文</a>：全球最强的开源编程模型现已发布 - Dracarys 2。我们非常激动地推出 Dracrays2！它击败了 Sonnet 3.5，是 LiveCodeBench 上排名第一的开源模型。该模型...</li><li><a href="https://huggingface.co/abacusai/Dracarys2-72B-Instruct">abacusai/Dracarys2-72B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html">使用 SQLite 进行混合全文搜索和向量搜索</a>：将 SQLite 内置的 FTS5 全文搜索扩展与 sqlite-vec 向量搜索扩展结合，实现混合搜索！</li><li><a href="https://flowiseai.com/">Flowise - 低代码 LLM 应用构建器</a>：为开发者提供的开源低代码工具，用于构建自定义的 LLM 编排流和 AI Agent</li><li><a href="https://github.com/python/cpython/commit/31516c98dd7097047ba10da8dcf728c3d580f3d6">GH-109975: 在 Python 3.13 的新特性中宣布最终版本 (#125007) · python/cpython@31516c9</a>：为 Python 3.13 的最终发布准备“新特性”文档</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/wKfQhP8JzX">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://pythoninsider.blogspot.com/2024/10/python-3130-final-released.html">Python Insider: Python 3.13.0 (最终版) 发布</a>：未找到描述</li><li><a href="https://docs.flowiseai.com/using-flowise/telemetry">遥测 | FlowiseAI</a>：了解 Flowise 如何收集匿名应用使用信息
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1291841090411827271)** (327 条消息🔥🔥): 

> - `Nous Research 更新`
> - `熵采样方法`
> - `基于图的模型`
> - `Hermes 与 o1 模型性能对比`
> - `推理任务中的挑战`

- **Nous Research 持续创新**：参与者对即将推出的 Nous 项目（如 Forge 和 Hermes-3-Llama-3.1-8B）表示期待，这些项目因其无审查（uncensored）和用户导向的可控性（steerability）而受到赞誉。
   - 用户强调了该模型令人印象深刻的创造力和逼真的表现，预示着其对 AI 未来发展将产生重大影响。
- **关于结合 CoT 解码的 Entropic Sampling 的讨论**：人们对所演示的 Entropic Sampling 方法的适用性和清晰度表示担忧，用户对其连贯性提出了质疑。
   - 该方法产生的输出被认为是无意义的，引发了对 Prompt 设计和实现的担忧。
- **探索结合 LLMs 的图模型**：用户深入探讨了在 LLMs 中实现知识图谱（knowledge graphs）的方法，强调了在不进行扁平化处理的情况下处理非结构化数据的重要性。
   - 参与者讨论了关于图模型的内部研究，并建议图数据库可以增强 LLM 的能力，特别是在表示复杂关系方面。
- **关于 o1 模型性能的关键见解**：讨论围绕 o1 模型的推理能力展开，用户分享了在特定推理任务上的不同体验。
   - 反馈表明，该模型有时在简单的算术问题上表现吃力，指出了潜在的改进方向。
- **AI 开发中的社区参与**：几位成员表达了为正在进行的项目做出贡献的兴趣，并索取了资源和阅读材料以进一步加深理解。
   - 随着协作想法的迸发，参与者还强调了这些讨论可能为 AI 领域带来的创新发展潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/m_wulfmeier/status/1842201976597074290?t=bVksmRCFScV1q6Vc4kDwgw&s=19">来自 Markus Wulfmeier (@m_wulfmeier) 的推文</a>：看来新一代学生已经为基于 Gemini/ChatGPT 的评审时代做好了更好的准备...</li><li><a href="https://lapis-nova-b3f.notion.site/How-I-Think-OpenAI-s-o1-Model-Works-and-How-I-Think-it-Was-Trained-11362e1157a18094ab35dcb42f5fad41?pvs=74">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。为您和您的团队提供的一体化工作空间。</li><li><a href="https://huggingface.co/posts/m-ric/957178001915012">Hugging Face 上的 @m-ric："📜 传统的 RNN 实际上可以与花哨的……媲美"</a>：未找到描述</li><li><a href="https://medium.com/@harishhacker3010/can-we-make-any-smaller-opensource-ai-models-smarter-than-human-1ea507e644a0">我们能让更小的开源 LLM 模型比人类更聪明吗？</a>：我是 Harish SG，一名安全研究员，曾在德克萨斯大学达拉斯分校攻读网络安全硕士，目前是 Cisco 的 AI 安全工程师，此前……</li><li><a href="https://openreview.net/forum?id=BGnm7Lo8oW">迈向预训练规模的推理学习</a>：提示大语言模型 (LLM) 输出思维链 (CoT) 推理可以提高在复杂问题解决任务上的表现。此外，存在几种流行的“自我改进”方法……</li><li><a href="https://www.interconnects.ai/p/reverse-engineering-openai-o1">逆向工程 OpenAI 的 o1 </a>：将测试时计算 (test-time compute) 产品化向我们展示了 AI 的未来。探索已降临至语言模型训练领域。</li><li><a href="https://huggingface.co/KingNish/Reasoning-Llama-1b-v0.1">KingNish/Reasoning-Llama-1b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2410.01201">论文页面 - RNN 难道就是我们所需要的一切吗？</a>：未找到描述</li><li><a href="https://huggingface.co/nvidia/NVLM-D-72B">nvidia/NVLM-D-72B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/JJitsev/status/1842727657345036788">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：天哪。在 AIW+ 上，o1 崩溃了，在完全不影响问题结构的变体中表现出剧烈波动。o1-mini 在所有 AIW+ 变体上都瓦解了。AIW+ 离奥数水平还差得很远……</li><li><a href="https://huggingface.co/qnguyen3/raspberry-3B">arcee-ai/raspberry-3B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/xjdr-alt/entropix">GitHub - xjdr-alt/entropix: 基于熵的采样与并行 CoT 解码</a>：基于熵的采样与并行 CoT 解码。通过在 GitHub 上创建账号来为 xjdr-alt/entropix 的开发做出贡献。</li><li><a href="https://chat.hl.ing/share/144c63db-005c-4475-b89e-001f99bee493">剪贴板内容分析摘要 | 共享的高亮对话</a>：未找到描述</li><li><a href="https://github.com/harishsg993010/LLM-Research-Scripts">GitHub - harishsg993010/LLM-Research-Scripts</a>：通过在 GitHub 上创建账号来为 harishsg993010/LLM-Research-Scripts 的开发做出贡献。</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>：跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账号来为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://neo4j.com/docs/">Neo4j 文档 - Neo4j 文档</a>：Neo4j 文档 - Neo4j 文档</li><li><a href="https://networkx.org/documentation/stable/reference/index.html">参考资料 —— NetworkX 3.3 文档</a>：未找到描述</li><li><a href="https://ggc-discrete-math.github.io/graph_theory.html">
   离散数学
  </a>：未找到描述</li><li><a href="https://research.facebook.com/publications/pytorch-biggraph-a-large-scale-graph-embedding-system/">PyTorch-BigGraph：大规模图嵌入系统 - Meta Research</a>：我们介绍了 PyTorch-BigGraph (PBG)，这是一个嵌入系统，它对传统的多关系嵌入系统进行了多项改进，使其能够扩展到具有数十亿个节点和……的图。</li><li><a href="https://arxiv.org/abs/2407.01884">EIT-1M：用于人类视觉文本识别等的一百万个 EEG-图像-文本对</a>：最近，脑电图 (EEG) 信号已被积极引入，用于将大脑活动解码为视觉或文本刺激，并在多模态 AI 中实现物体识别。因此，努力……
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1292414793071853629)** (15 条消息🔥): 

> - `Fine-tuning Instruct Models`
> - `LLM for Low Resource Languages`
> - `Self-Evaluating Models`
> - `Fine-tuning Llama 3.1`
> - `Attention Masking in Packed Samples` 


- **Fine-Tuning Instruct Models 的挑战**：一位成员询问在 completion 任务上对 instruct models 进行 fine-tune 是否可行，并分享了关于需要适当 scaling factors 的困扰。
   - 他们暗示调整 base template token 可能是该领域成功的关键。
- **为 Low Resource Languages 构建通用 LLM**：一位成员寻求关于在拥有 8xH100 节点无限资源的情况下，为 low resource languages 构建通用 LLM 的建议，并强调需要进行 sanity check。
   - 他们建议探索除单纯 fine-tuning 之外的非显性策略。
- **模型 Self-Evaluation 的潜力**：一位成员提出了模型可以 self-evaluate 自身弱点，并通过结合 synthetic data 和真实数据的持续训练进行自适应的想法。
   - 这一想法引发了关于是否存在类似工程挑战的讨论，同行们提到了 OpenAI 的 response evaluation 方法。
- **Pretraining 与 Instruct Models**：讨论了对 instruct models 进行持续 pretraining 是否会使其退化回 base models，并对可能产生的影响表示好奇。
   - 成员们将这一概念与现有的评估 response quality 以进行改进的方法论进行了比较。
- **Llama 3.1 的 Fine-Tuning 策略**：一位成员寻求关于 fine-tuning Llama 3.1 70b base model 的建议，询问了 pitfalls 和 data ordering 策略。
   - 他们表达了在进入训练过程之前，如何根据 corpus preparation 来最大化结果的关注。

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291894401047724032)** (11 messages🔥): 

> - `Meta Movie Generation`
> - `COCONUT reasoning paradigm`
> - `GenRM reward models`
> - `SwiftSage v2 introduction`
> - `Contextualized Document Embeddings` 


- **Meta Movie Gen 发布研究论文**：Meta 发布了其视频生成系统 Meta Movie Gen 的[技术指南](https://ai.meta.com/static-resource/movie-gen-research-paper)。
   - 该文档概述了其视频生成技术的方法论和应用，增强了用户的理解。
- **COCONUT 重新定义 LLM 推理**：[OpenReview](https://openreview.net/forum?id=tG4SgayTtk) 上的一篇论文讨论了 COCONUT，这是一种允许语言模型在连续潜空间（continuous latent space）而非语言空间中进行推理的新范式。
   - 这种方法表明，使用隐藏状态（hidden states）进行推理可以缓解传统模型中 Token 的限制，从而实现更复杂的思考。
- **GenRM 彻底改变奖励模型**：GenRM 的引入使得奖励模型可以作为下一个 Token 预测器（next token predictors）而非传统的分类器进行训练，从而为奖励模型开启了 **Chain-of-Thought 推理**。
   - *@LunjunZhang* 指出，这一创新提供了一个统一的策略和奖励模型，提升了各种任务的整体性能。
- **SwiftSage v2 增强推理能力**：SwiftSage v2 发布，这是一个集成了快慢思考（fast and slow thinking）的推理 Agent 系统，专注于 In-context Learning。
   - Demo 和代码已在 [GitHub](https://github.com/SwiftSage/SwiftSage) 和 Hugging Face 上线，在数学和 MMLU 风格的推理任务中表现出色。
- **上下文文档嵌入的新方法**：最近的一篇论文探索了创建包含邻近文档信息的上下文文档嵌入（Contextualized Document Embeddings）的方法，提升了神经检索任务的表现。
   - 该研究与 *Jina’s late chunking* 以及 Anthropic 的最新进展相呼应，旨在实现更有效的信息检索。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02525">Contextual Document Embeddings</a>: 稠密文档嵌入是神经检索的核心。主流范式是通过直接在单个文档上运行编码器来训练和构建嵌入。在这项工作中，我们认为……</li><li><a href="https://x.com/billyuchenlin/status/1842834224375873726">Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>: 我们很高兴分享 SwiftSage v2 的初始版本，这是一个旨在通过快慢思考进行推理的 Agent 系统。我们的目标是构建一个能够与……竞争的开源推理系统。</li><li><a href="https://arxiv.org/abs/2410.02536">Intelligence at the Edge of Chaos</a>: 我们通过研究基于规则的系统的复杂性如何影响训练用于预测这些规则的模型的能力，来探索人工系统中智能行为的涌现。我……</li><li><a href="https://openreview.net/forum?id=oQ4igHyh3N">TokenFormer: Rethinking Transformer Scaling with Tokenized Model...</a>: 由于 Transformer 在各个领域的卓越表现，它已成为基础模型中的主导架构。然而，扩展这些模型的巨大成本仍然是……</li><li><a href="https://openreview.net/forum?id=tG4SgayTtk">Training Large Language Model to Reason in a Continuous Latent Space</a>: 大语言模型被限制在“语言空间”中进行推理，它们通常使用思维链 (CoT) 来表达推理过程以解决复杂的推理问题……</li><li><a href="https://arxiv.org/abs/2409.04701">Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models</a>: 许多用例需要检索较小的文本部分，而基于稠密向量的检索系统在处理较短的文本段落时通常表现更好，因为语义不太可能被过度压缩……</li><li><a href="https://x.com/lunjunzhang/status/1829296204171100418?s=46">Lunjun Zhang (@LunjunZhang) 的推文</a>: 如果你的奖励模型可以“思考”更多并表现得更好呢？甚至更好的是，如果你的 LLM 策略也可以用作奖励模型呢？介绍 GenRM，作为下一个 Token 预测器训练的奖励模型……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1292497771357536329)** (4 条消息): 

> - `Entropy Based Sampling`
> - `Conversational Programming Language`
> - `OpenAI o1 System`
> - `Open O1 Project`
> - `Inference Scaling Laws` 


- **使用 Entropix 进行基于熵的采样 (Entropy Based Sampling)**：[Entropix 项目](https://github.com/xjdr-alt/entropix) 专注于 **基于熵的采样和并行 CoT 解码**，为模型交互提供了创新方法。
   - 该计划旨在提高模型效率，并欢迎社区贡献。
- **介绍 Convo：一种对话式编程语言**：[Convo 项目](https://github.com/Stevenic/convo) 是一种 **对话式编程语言**，旨在由 Large Language Models (LLMs) 生成和解释。
   - 这种方法试图将自然语言与编程融合，旨在简化用户与 AI 的交互方式。
- **OpenAI 发布 o1 推理系统**：OpenAI 的新推理系统 [o1](https://openai.com/o1/) 旨在通过 **长推理链** 和 **强化学习** 增强用户交互。
   - 虽然目前还是原型，但它标志着在处理更复杂的 AI 任务时向 **在线搜索能力** 的转变。
- **Open O1：OpenAI o1 的开源替代方案**：[Open O1 项目](https://opensource-o1.github.io/) 致力于创建一个开源模型，实现与 OpenAI o1 相当的性能。
   - 他们的使命包括在 **代码生成** 和 **数学问题解决** 方面的进展，旨在赋能 AI 社区。
- **关于推理缩放定律 (Inference Scaling Laws) 的讨论**：OpenAI o1 原型的发展引发了关于 **推理缩放定律** 的讨论，表明资源分配正转向更高效的 AI 交互。
   - 这一进展至关重要，因为它探索了超越传统自回归方法的新型模型交互方式，可能会改变未来的 AI 战略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://opensource-o1.github.io/">Open-Source O1</a>: 未找到描述</li><li><a href="https://www.interconnects.ai/p/reverse-engineering-openai-o1">Reverse engineering OpenAI’s o1 </a>: 将测试时计算 (test-time compute) 产品化向我们展示了 AI 的未来。探索已进入语言模型训练领域。</li><li><a href="https://github.com/Stevenic/convo">GitHub - Stevenic/convo: Convo is a conversational programming language that&#39;s designed to be generated and interpreted by a Large Language Model (LLM).</a>: Convo 是一种对话式编程语言，旨在由 Large Language Model (LLM) 生成和解释。 - Stevenic/convo</li><li><a href="https://github.com/xjdr-alt/entropix">GitHub - xjdr-alt/entropix: Entropy Based Sampling and Parallel CoT Decoding</a>: 基于熵的采样和并行 CoT 解码。通过在 GitHub 上创建账号来为 xjdr-alt/entropix 的开发做出贡献。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291894401047724032)** (11 条消息🔥): 

> - `Meta Movie Gen`
> - `Contextual Document Embeddings`
> - `GenRM Reward Models`
> - `Chain of Continuous Thought`
> - `SwiftSage v2 Introduction` 


- **Meta Movie Gen 论文发布**：Meta 发布了一篇关于 [Movie Gen 的研究论文](https://ai.meta.com/static-resource/movie-gen-research-paper)，详细介绍了他们在电影生成建模方面的最新进展。
   - 该资源对于理解 Meta 在电影生成背景下提出的技术细节和创新至关重要。
- **Contextual Document Embeddings 的进展**：研究探索了更好的 **contextualized document embeddings** 方法，该方法考虑了周围的文档上下文，以提高检索性能。
   - 提出了两种新方法：一种对比学习目标和一种将相邻文档信息整合到编码表示中的新颖架构。
- **GenRM：作为奖励模型的 Next-Token Predictors**：*GenRM* 的引入展示了作为 next-token predictors 训练的奖励模型，这增强了 Chain-of-Thought 推理能力。
   - 这种方法允许有效地利用测试时计算（test-time compute），并将策略与奖励模型结合以改进推理任务。
- **通过 COCONUT 范式改进推理**：一篇论文讨论了通过其新范式 COCONUT，将语言模型的推理从语言空间转向 **continuous latent space**（连续潜空间）。
   - 该模型旨在增强超越传统 Chain-of-Thought 的推理能力，同时减少对词 token 的依赖。
- **SwiftSage v2：新的开源推理 Agent**：*SwiftSage v2* 的初始版本已作为开源 Agent 系统发布，旨在通过 in-context learning 实现更有效的推理任务。
   - 该系统旨在通过交替利用小型和大型语言模型来解决复杂问题，GitHub 上已提供演示和代码。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02536">Intelligence at the Edge of Chaos</a>：我们通过研究规则系统的复杂性如何影响训练用于预测这些规则的模型的能力，来探索人工系统中智能行为的出现。O...</li><li><a href="https://arxiv.org/abs/2410.02525">Contextual Document Embeddings</a>：稠密文档嵌入是神经检索的核心。主流范式是通过直接在单个文档上运行编码器来训练和构建嵌入。在这项工作中，我们认为...</li><li><a href="https://openreview.net/forum?id=tG4SgayTtk">Training Large Language Model to Reason in a Continuous Latent Space</a>：大语言模型被限制在“语言空间”中进行推理，通常通过 Chain-of-Thought (CoT) 表达推理过程来解决复杂的推理问题....</li><li><a href="https://x.com/billyuchenlin/status/1842834224375873726">Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>：我们很高兴分享 SwiftSage v2 的初始版本，这是一个专为快慢思考推理设计的 Agent 系统。我们的目标是构建一个能够竞争的开源推理系统...</li><li><a href="https://openreview.net/forum?id=oQ4igHyh3N">TokenFormer: Rethinking Transformer Scaling with Tokenized Model...</a>：TokenFormer：通过 Tokenized 模型重新思考 Transformer 扩展：由于在各个领域的卓越表现，Transformer 已成为基础模型中的主导架构。然而，扩展这些模型的巨大成本仍然是...</li><li><a href="https://x.com/lunjunzhang/status/1829296204171100418?s=46">Lunjun Zhang (@LunjunZhang) 的推文</a>：如果你的奖励模型可以“思考”更多并表现得更好呢？甚至，如果你的 LLM 策略也可以用作奖励模型呢？介绍 GenRM，作为 next token predictor 训练的奖励模型...</li><li><a href="https://arxiv.org/abs/2409.04701">Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models</a>：延迟分块：使用长上下文嵌入模型的上下文分块嵌入：许多用例需要检索较小的文本部分，而基于稠密向量的检索系统在处理较短的文本段时通常表现更好...</li><li><a href="https://www.anthropic.com/news/contextual-retrieval">Introducing Contextual Retrieval</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1292588635849232384)** (2 条消息): 

> - `Open Reasoning Tasks`
> - `GitHub project` 


- **Open Reasoning Tasks 频道介绍**：一位成员询问了该频道的目的，问道：*'这个频道是做什么的？'*
   - 另一位成员澄清说，该频道主要用于在 [GitHub](https://github.com) 上启动的 **Open Reasoning Tasks** 项目。
- **项目目的说明**：该频道作为一个讨论和进一步开发 **Open Reasoning Tasks** 项目的平台，旨在促进协作和见解分享。
   - 鼓励成员参与并为项目的持续进展做出贡献。


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1291845486877347962)** (236 条消息🔥🔥): 

> - `LM Studio 模型加载问题`
> - `多 GPU 设置`
> - `图像处理模型`
> - `自定义 Prompt 模板`
> - `用户界面建议` 


- **在 LM Studio 中加载模型**：用户在加载模型时遇到问题，特别是收到诸如 'No LM Runtime found for model format 'gguf'!' 之类的错误，这通常与过时的 CPU 指令集（如 AVX2）有关。
   - 建议包括升级硬件或切换到 Linux 以获得对某些模型更好的兼容性。
- **多 GPU 配置的挑战**：讨论强调了在多 GPU 设置中混合不同 GPU 的挑战和限制，特别是将 4090 和 3090 模型组合使用。
   - 用户被告知虽然这是可行的，但性能可能会受到较慢 GPU 的限制。
- **模型中的图像支持**：有关于支持图像处理模型的咨询，建议使用 MiniCPM-V-2_6-GGUF 作为一个可行的选择。
   - 提出了关于图像大小和模型兼容性的问题，指出分辨率可能会影响分析时间。
- **自定义 Prompt 模板**：用户被告知在 LLM 中使用正确的 Prompt 模板的重要性，以避免生成意外的 Token 或结果。
   - 讨论强调，更改为非默认模板可能会导致不匹配以及模型输出问题。
- **用户界面功能请求**：用户请求了诸如撤销功能（以防止意外删除）以及 LM Studio 中可自定义的头像或背景图像等功能。
   - 用户表达了对 UI 美学和功能改进的愿望，特别是在数据管理方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/docs/cli/log-stream">lms log stream - CLI | LM Studio Docs</a>：从 LM Studio 流式传输日志。对于调试发送到模型的 Prompt 非常有用。</li><li><a href="https://x.com/LiquidAI_/status/1840768716784697688">Liquid AI (@LiquidAI_) 的推文</a>：今天我们向世界介绍 Liquid Foundation Models (LFMs)，包含我们的首批语言 LFMs 系列：1B、3B 和 40B 模型。(/n)</li><li><a href="https://x.com/maximelabonne/status/1840770960149913601">Maxime Labonne (@maximelabonne) 的推文</a>：我们目前不开源这些模型，但我们希望通过公开我们的发现、方法和有趣的产物来为社区做出贡献。我们将从发布...开始。</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1fx51z4/i_made_claude_35_sonnet_to_outperform_openai_o1/?share_id=xqAfSzT4HWUbn3NQXHrwj">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/133">功能请求：将 LM Studio 用作本地网络中不同 LLM 服务器的客户端 · Issue #133 · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 已经允许创建一个服务器并将其用于 API 请求。但它不允许 LM Studio 作为该服务器的客户端。场景如下：我有一台强大的机器在我的...</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">侧载模型 - 高级 | LM Studio Docs</a>：使用你在 LM Studio 之外下载的模型文件</li><li><a href="https://lmstudio.ai/docs/configuration/prompt-template#">Prompt 模板 - 配置 | LM Studio Docs</a>：编辑 Prompt 模板</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory))">下载 LLM - 在本地运行 LLM | LM Studio Docs</a>：在 LM Studio 中发现并下载支持的 LLM</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues">问题 · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 桌面应用程序的错误跟踪 - 问题 · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://lmstudio.ai/docs/cli/log-stream#">lms log stream - CLI | LM Studio Docs</a>：从 LM Studio 流式传输日志。对于调试发送到模型的 Prompt 非常有用。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1292221682425139312)** (114 条消息🔥🔥): 

> - `GPU Memory Performance` (GPU 显存性能)
> - `LM Studio Compatibility` (LM Studio 兼容性)
> - `Docker Usage for LLMs` (LLM 的 Docker 使用)
> - `Inference Speed Comparisons` (推理速度对比)
> - `Model Fine-tuning Discussions` (模型微调讨论)


- **关于 GPU 显存性能的讨论**：用户对比了各种 GPU 的性能和 VRAM，指出 **Tesla P40** 拥有 **24GB**，有利于 AI 任务，而 **RTX 4060Ti** 虽然提供 **16GB**，但在某些场景下表现出相当的性能。
   - 然而，有人担心 P40 在 **Stable Diffusion** 等应用中表现较慢，可能无法有效利用其能力。
- **LM Studio 的操作系统兼容性**：在讨论 LM Studio 性能时，用户表达了对操作系统的偏好，建议倾向于使用 **Windows** 以获得易用性，但也认可 **Linux** 的资源效率。
   - 大家的共识是两个系统表现相似，从而引发了一场关于用户体验与 Linux 技术挑战的幽默辩论。
- **Docker 在 LLM 管理中的作用**：几位用户分享了使用 Docker 的经验，一些人因其复杂性而避开，而另一些人则称赞其在管理依赖项和 CUDA 操作方面更高效。
   - 对话揭示了在 AI 工作流（尤其是管理 **LM Studio** 等工具）中使用 Docker 的易用性存在不同意见。
- **推理速度对比**：用户对比了 **Tesla P40** 和 **RTX 4060Ti** 的推理速度，注意到显著差异，P40 达到 **17.1 tokens/sec**，而 4060Ti 为 **8.1 tokens/sec**。
   - 讨论了 VRAM 容量和内存带宽等因素，以解释 AI 模型推理过程中的性能差异。
- **使用 Llama 进行模型微调**：用户表达了对 **Llama 3.1-8B** 模型的喜爱，讨论了其出人意料的输出，以及在使用 'system check' 等不同提示词时获得的乐趣。
   - 有人对模型的训练数据表示担忧，推测其可能存在争议的来源以及使用此类数据的影响。



**提到的链接**：<a href="https://www.wevolver.com/article/tpu-vs-gpu-in-ai-a-comprehensive-guide-to-their-roles-and-impact-on-artificial-intelligence">TPU vs GPU in AI: A Comprehensive Guide to Their Roles and Impact on Artificial Intelligence</a>：未找到描述

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1292224702030413995)** (1 条消息): 

> - `OpenRouter integration with Fal.ai` (OpenRouter 与 Fal.ai 集成)
> - `LLM and VLM workflows` (LLM 和 VLM 工作流)


- **OpenRouter 与 Fal.ai 合作**：OpenRouter 宣布与 **Fal.ai** 建立合作伙伴关系，现在通过[此链接](https://x.com/isidentical/status/1842650721969459561)增强了 Fal 图像工作流中的 **LLM** 和 **VLM** 能力。
   - 通过 OpenRouter 使用 **Gemini**，与 Fal 一起*重构您的工作流*，简化您的图像处理任务。
- **图像工作流的增强**：此次集成允许用户在图像工作流中利用 **LLM** 和 **VLM** 的能力，有望提高效率和输出质量。
   - 该公告强调了用户通过引入的新功能重新思考其流程和结果的潜力。



**提到的链接**：<a href="https://x.com/isidentical/status/1842650721969459561">来自 batuhan taskaya (@isidentical) 的推文</a>：使用 fal 重构工作流（通过 OpenRouter 使用 gemini）

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1292083917997740073)** (3 条消息): 

> - `API4AI`
> - `AI Assisted Coding Tool`
> - `Sci Scope Newsletter` 


- **API4AI：通过新 API 为 AI 赋能**：**API4AI** 平台实现了与 OpenAI 和 Azure OpenAI 等服务的无缝集成，为开发 AI 应用和**现实世界交互**提供了强大的工具。
   - 提供的 API 包括**天气预报**、**互联网搜索**、**电子邮件处理**和**图像生成**等功能，增强了 AI 的实用性。
- **通过 Web Chat 进行 AI 辅助编程**：开发了一款利用 Web Chat 进行 AI 辅助编程的创新工具，特别适用于不支持附件的 **OpenAI 新型 o1 模型**。
   - [GitHub 仓库](https://github.com/cyberchitta/llm-context.py) 提供了一个命令行工具，用于**将代码上下文复制到剪贴板**，从而简化 LLM 聊天中的交互。
- **通过 Sci Scope 保持更新**：**Sci Scope** 新闻通讯每周汇总新的 **ArXiv 论文**，并对相似主题进行总结，让研究人员能够轻松掌握最新动态。
   - 提供根据用户兴趣定制的**个性化摘要**，确保您不会错过与工作相关的重大研究进展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sci-scope.com">Sci Scope</a>：关于 AI 研究的 AI 生成新闻通讯</li><li><a href="https://github.com/cyberchitta/llm-context.py">GitHub - cyberchitta/llm-context.py</a>：一个用于将代码上下文复制到剪贴板以在 LLM 聊天中使用的命令行工具</li><li><a href="https://www.cyberchitta.cc/articles/llm-ctx-why.html">LLM Context: Harnessing Vanilla AI Chats for Development</a>：论证了开发一款能高效利用基于 Web 的 AI 聊天界面进行软件开发的工具的必要性，为集成 IDE 的解决方案提供了一种替代方案。</li><li><a href="https://open.dbapibuilder.com/">API for AI</a>：未找到描述</li><li><a href="https://github.com/dbapibuilder/API4AI">GitHub - dbapibuilder/API4AI</a>：通过在 GitHub 上创建账户来为 dbapibuilder/API4AI 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1291867210721136694)** (286 条消息🔥🔥): 

> - `OpenRouter 功能`
> - `图像和多媒体模型`
> - `重复生成问题`
> - `数学模型性能`
> - `非营利组织折扣` 


- **关于 OpenRouter 能力的讨论**：用户对 OpenRouter 是否会支持图像、视频和音频模型表示关注，认为媒体集成是逻辑上的必然演进。
   - 一些用户认为多模态模型在 AI 领域变得越来越重要。
- **重复生成响应的问题**：一位用户报告在调用 OpenRouter API 时收到重复生成的响应，这似乎是其特定配置的问题。
   - 在调整了针对重试的响应解析器后，他们注意到一些 API 请求返回了 404 错误，暗示可能存在超时或可用性延迟。
- **数学模型表现良好**：在讨论中，`o1-mini` 被强调为数学 STEM 任务的首选模型，因为它在渲染输出方面非常有效。
   - 用户询问了在 OpenRouter 聊天室中渲染数学公式的 LaTeX 能力。
- **关于响应中用量指标的反馈**：API 响应中出现了详细列出 Prompt 和 Completion Token 的新用量指标，一些用户直到现在才注意到。
   - 用量信息适用于通过 OpenRouter 提供的所有模型，并遵循 GPT4 Tokenizer 标准。
- **关于非营利组织折扣的咨询**：一位用户询问了 OpenRouter 为非洲非营利教育机构提供折扣或额度选项的可能性。
   - 这一咨询反映了 AI 社区对非营利倡议的可访问性和支持性定价的广泛兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://api.together.ai/models">no title found</a>: 未找到描述</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: 查看你在 OpenRouter 上的模型使用情况。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">no title found</a>: 未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview">no title found</a>: 未找到描述</li><li><a href="https://ai.google.dev/pricing?hl=ru">no title found</a>: 未找到描述</li><li><a href="https://ai.google.dev/pricing">no title found</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/requests#tool-calls">Requests | OpenRouter</a>: 处理传入和传出请求</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>: 优化 LLM 成本高达 90%</li><li><a href="https://github.com/stanford-oval/storm/">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识策展系统，用于研究主题并生成带有引用的完整报告。 - stanford-oval/storm
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1291855151476379699)** (51 条消息🔥): 

> - `MATS Program 导师项目`
> - `独立研究合作`
> - `ICLR 论文流程`
> - `训练 minGRU`
> - `Transformer 训练需求` 


- **MATS Program 迎来新导师**：AnthropicAI 的 Alignment Science 共同负责人 **Jan Leike** 将担任 [MATS Winter 2024-25](https://matsprogram.org/apply) 的导师，申请截止日期为 PT 时间 10 月 6 日晚上 11:59。
   - 这为申请者提供了一个获得 Alignment Science 宝贵见解和经验的绝佳机会。
- **与大学实验室合作的挑战**：一位独立研究员询问了与美国大学实验室合作所需的正式文件，并指出目前缺乏记录在案的流程。
   - 成员们提到，具体要求因大学而异，最好直接与潜在的合作者沟通以明确情况。
- **了解 ICLR 论文发布时间**：贡献者们讨论了提交给 **ICLR** 的论文发布预期，强调分发可能会在评审过程之后进行。
   - 一些成员建议作者可以非正式地分享早期草案，并就预印本（preprints）的发布时机展开了讨论。
- **寻求训练 minGRU 的帮助**：一位成员寻求在 8 台 RTX 4090 GPU 上训练 **minGRU** 的帮助，理由是在修改实现以进行高效训练时遇到了挑战。
   - 其他人表示愿意提供帮助，但受限于自己的截止日期，同时建议在合成任务上测试小模型以评估性能。
- **澄清 Transformer 训练成本**：一位用户质疑计算 Transformer 训练内存需求的方法论，特别是与张量并行（tensor parallelism）相关的部分。
   - 讨论强调了理解与训练 Transformer 模型相关的计算成本的重要性，并反思了其实际影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.eleuther.ai/transformer-math/">Transformer Math 101</a>: 我们介绍了与 Transformer 的计算和内存使用相关的基础数学知识</li><li><a href="https://x.com/MATSprogram/status/1842286650006892914">来自 ML Alignment & Theory Scholars (@MATSprogram) 的推文</a>: @janleike，AnthropicAI 的 Alignment Science 共同负责人，现在将担任 MATS Winter 2024-25 的导师！申请于 PT 时间 10 月 6 日晚上 11:59 截止。https://matsprogram.org/apply
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1291902909432332378)** (208 条消息🔥🔥): 

> - `RWKV 系列更新`
> - `推理中的循环模型 (Looped Models)`
> - `选择性注意力机制 (Selective Attention Mechanism)`
> - `生成式奖励模型 (Generative Reward Models)`
> - `AI Alignment 的挑战` 


- **RWKV 系列及其版本控制挑战**：用户讨论了追踪 RWKV 系列不同版本变化的困难，强调文档往往缺乏对每个版本贡献的清晰说明。
   - 一位成员指向了一篇详细介绍 RWKV 逐步变化的论文，并建议提供一份完整的版本变更列表将使新手受益。
- **循环模型在推理方面的潜力**：关于循环模型（looped models）的研究假设，它们可能通过使用更少的参数并重复层（而不是扩展整个模型）来增强推理能力。
   - 然而，一些人对循环多个层的有效性表示怀疑，指出更复杂的任务可能无法从这种架构中受益。
- **用于提升效率的选择性注意力**：一种名为“选择性注意力”（Selective Attention）的新机制被提出，旨在减少对不必要元素的关注，从而可能提高不同规模模型的性能。
   - 这种方法可以显著降低内存和计算需求，使 Transformer 更加高效，特别是对于更大的上下文长度（context sizes）。
- **生成式奖励模型增强 AI Alignment**：思维链生成式奖励模型（CoT-GenRM）的引入旨在提高训练后性能以及 AI 系统与人类价值观的对齐。
   - 该方法将人类反馈与 AI 生成的反馈相结合，以增强模型决策中的推理能力。
- **ARXIV 提交延迟**：成员们对 ARXIV 提交的延迟表示沮丧，并引用了一个具体的提交被搁置的案例。
   - 人们担心这些延迟会对研究的可见性以及分享进展的及时性产生影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>: Transformer 在序列长度方面的可扩展性限制，重新引发了人们对训练期间可并行化的循环序列模型的兴趣。因此，许多新型循环架构...</li><li><a href="https://arxiv.org/abs/2410.02089">RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning</a>: 作为 Agent 部署的大语言模型（LLMs）在完成用户指定的任务时，需要跨多个步骤并尽可能减少人工参与。至关重要的是，这些 LLM 需要将其生成的...</li><li><a href="https://openreview.net/forum?id=din0lGfZFd">Understanding Reasoning with Looped Models</a>: 大语言模型在推理问题上展现出了可喜的能力，而 Scaling Laws 表明参数量是关键驱动因素。最近的研究（Chen &amp; Zou, 2024; Ye et al., 2024）认为...</li><li><a href="https://openreview.net/forum?id=r8H7xhYPwz">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>: 由于推理效率高，Linear Transformers 已成为标准 Transformer 的高效替代方案，在各种任务中取得了具有竞争力的性能，尽管它们通常...</li><li><a href="https://www.synthlabs.ai/research/generative-reward-models">Generative Reward Models that Unify RLHF and RLAIF Approaches</a>: 一个结合了 RLHF 和 RLAIF 的新型框架，旨在更好地使 LLMs 与人类偏好对齐，性能优于传统方法高达 45%。</li><li><a href="https://arxiv.org/abs/2410.01792">When a language model is optimized for reasoning, does it still show embers of autoregression? An analysis of OpenAI o1</a>: 在 &#34;Embers of Autoregression&#34; (McCoy et al., 2023) 中，我们展示了几个大语言模型（LLMs）存在一些重要的局限性，这些局限性可归因于它们起源于 next-word prediction...</li><li><a href="https://arxiv.org/abs/2410.02703">Selective Attention Improves Transformer</a>: Attention 上下文中不需要的元素会降低性能。我们引入了 Selective Attention，这是一种对标准 Attention 机制的简单无参数改进，它减少了对不相关...</li><li><a href="https://openreview.net/forum?id=tG4SgayTtk">Training Large Language Model to Reason in a Continuous Latent Space</a>: 大语言模型被限制在“语言空间”中进行推理，它们通常通过思维链（CoT）表达推理过程来解决复杂的推理问题...</li><li><a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>: 我们展示了线性化自注意力机制（linearised self-attention mechanisms）与 90 年代初的快速权重控制器（fast weight controllers）在形式上的等价性，其中“慢速”神经网络通过梯度下降学习来编写“快速”...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: 具有线性注意力的 Transformer（即 Linear Transformers）和状态空间模型（state-space models）最近被认为是 Softmax Attention Transformer 的一种可行的线性时间替代方案。然而，...</li><li><a href="https://x.com/vaiter/status/1842072657505697821">Tweet from Samuel Vaiter (@vaiter)</a>: Stein&#39;s Lemma 指出，对于正态分布变量 X，期望值 E[Xg(X)] = E[g’(X)]，对于任何绝对连续（几乎处处可导）且满足 E[|g’(X)|] &lt; ∞ 的 g。这是一个核心...</li><li><a href="https://x.com/JJitsev/status/1842727628463128968">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: （又）一个关于兴衰的故事：o1 宣称拥有极强的性能，在奥数级别的数学和编程问题上得分很高。它能处理简单的 AIW 问题吗？这些问题揭示了泛化能力...</li><li><a href="https://arxiv.org/abs/2410.02416">Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models</a>: Classifier-free guidance (CFG) 对于提高 Diffusion Models 的生成质量以及输入条件与最终输出之间的对齐至关重要。虽然高引导尺度（guidance scale）通常...</li><li><a href="https://x.com/Msadat97/status/1842246601181646912">Tweet from Morteza Sadat (@Msadat97)</a>: 📢📢 介绍 &#34;Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models&#34;。简而言之：我们展示了通过对 CFG 更新方式进行少量修改，我们可以...</li><li><a href="https://github.com/xjdr-alt/entropix">GitHub - xjdr-alt/entropix: Entropy Based Sampling and Parallel CoT Decoding</a>: 基于熵的采样和并行 CoT 解码。通过在 GitHub 上创建账号来为 xjdr-alt/entropix 的开发做出贡献。</li><li><a href="https://github.com/nikodeam/gematria">GitHub - Nikodeam/Gematria: Gematria is an environment to locally run multiple LLMs capable of chatting with multiple other and users on Discord, with a locally run centralised SQLite database updated and</a>: Gematria 是一个在本地运行多个 LLMs 的环境，能够与 Discord 上的其他用户聊天，并配有一个本地运行的集中式 SQLite 数据库...</li>

检索增强生成，由嵌入模型处理。</a>：Gematria 是一个在本地运行多个 LLM 的环境，能够与 Discord 上的其他用户聊天，并拥有一个本地运行的集中式 SQLite 数据库，支持更新和检索增强...</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>：我们介绍了 Eagle (RWKV-5) 和 Finch (RWKV-6)，这是在 RWKV (RWKV-4) 架构基础上改进的序列模型。我们的架构设计进步包括多头矩阵值状态和动态...</li><li><a href="https://github.com/SmerkyG/RWKV_Explained/tree/main">GitHub - SmerkyG/RWKV_Explained: RWKV, in easy to read code</a>：以易读代码实现的 RWKV。通过在 GitHub 上创建账号来为 SmerkyG/RWKV_Explained 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1292222716950347777)** (7 条消息): 

> - `逆向工程电路`
> - `SAE 电路发现`
> - `稀疏特征电路`
> - `电路研究文献` 


- **探索非玩具模型中的逆向工程**：成员们讨论了在非玩具级语言模型中**完全逆向工程电路**的普遍性，并强调了 **gpt2-small 中的 IOI 电路**作为一个已知示例。
   - *“有很多好的例子吗？”* 引发了对该领域更广泛发现的询问。
- **SAE 电路作为重要发现**：一位成员提出 **SAE 电路**作为逆向工程的潜在示例，并引用了 **Sam Mark 的论文**作为相关材料。
   - 提供了相关链接，其中包括[该论文](https://arxiv.org/abs/2403.19647)，详细介绍了与稀疏特征电路相关的方法。
- **稀疏特征电路的突破**：分享的论文概述了发现和应用**稀疏特征电路**的方法，通过**人类可解释的特征**提供对模型行为的见解。
   - 这种方法旨在提高分类器的泛化能力，并展示了一个**可扩展的可解释性流水线 (pipeline)**。
- **电路研究的文献综述**：一位成员将注意力引向一篇包含多个已识别电路示例的论文，建议将其作为**文献综述的一个良好起点**。
   - 虽然这些例子并非该论文原创，但它们有助于深入理解已研究的电路。



**提到的链接**：<a href="https://arxiv.org/abs/2403.19647">Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models</a>：我们介绍了发现和应用稀疏特征电路的方法。这些是与因果相关的、由人类可解释特征组成的子网络，用于解释语言模型的行为。电路...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1292488378389889130)** (2 条消息): 

> - `Claude 评估`
> - `JAX 模型支持` 


- **关于 Claude 评估的询问**：一位成员询问另一位成员是否尝试在特定任务上评估 **Claude**。
   - 这个问题凸显了人们对 Claude 在各种场景下表现的持续关注。
- **对 JAX 模型的支持**：关于为 **JAX 模型**提供**一等支持 (first-class support)** 的潜在计划展开了讨论。
   - 成员们渴望了解这方面是否有任何进展。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 条消息): 

zackt1234: https://discord.com/channels/729741769192767510/1214931475850469426/1292977027254583397

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1291969599398477857)** (85 messages🔥🔥): 

> - `对 Cohere 支持服务的不满`
> - `社区参与`
> - `Cohere API 印象`
> - `深色模式上线` 


- **对支持响应速度的不满**：一位用户对支持工单缺乏回复表示沮丧，该工单涉及模型创建过程中遇到的 429 错误，并强调该问题影响了多位用户。
   - 尽管响应延迟，另一位版主保证该问题已被优先处理，并指出支持工单目前存在积压。
- **关于角色和贡献的社区对话**：版主们澄清了他们的志愿者身份，其中一位表示，他们更看重为社区贡献带来的“人情”而非“金钱”。
   - 其他人讨论了行业的整体士气，以及用户反馈在改进平台功能方面的重要性。
- **对 Cohere API 性能的赞赏**：一位新成员称赞了 Cohere API，提到了其简洁的设计以及设置多工具 Agent 的简便性，并对其功能表示欣赏。
   - 该用户分享了他们正在评估团队工作流中的 AI 集成，表明开发者体验是一个重要的考量因素。
- **深色模式功能发布公告**：社区对 Cohere 平台引入深色模式（Dark Mode）功能感到兴奋。
   - 用户们庆祝了这一更新，认为这是对用户界面的一次受欢迎的增强。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1291850608395026475)** (97 messages🔥🔥): 

> - `Cohere API 错误`
> - `Fine-tuning 挑战`
> - `将 Cohere 用于商业用途`
> - `社区支持`
> - `Cohere 的 API 特性` 


- **Cohere API 错误与故障排除**：用户报告在使用 Cohere API 时频繁出现 “InternalServerError” 等错误，阻碍了项目进度。
   - 一位用户强调，他们的错误源自 Fine-tuning 页面，这对故障排除至关重要。
- **模型 Fine-tuning 的挑战**：一位用户描述了在向 Cohere 控制面板上传训练文档时遇到的困难，导致 JSON 文件出现编码错误。
   - 讨论中还提出了关于如何利用预定 Embedding 来 Fine-tuning 二元分类器的最佳实践问题。
- **将 Cohere API 用于商业用途**：社区成员确认 Cohere API 确实可以用于商业目的，目标客户为企业市场。
   - 关于许可的详细说明，用户被引导至 Cohere 网站上的 FAQs 章节。
- **社区支持与反馈**：鼓励用户寻求帮助，并建议与支持团队分享进展和反馈。
   - 多位成员强调了社区内协作和及时解决问题的重要性。
- **Cohere 的 API 特性与更新**：成员们讨论了 Cohere API 的最新更新，重点介绍了使从其他服务迁移更轻松的新特性。
   - 用户被提醒注意使用 Cohere 与其他 LLM 供应商之间的区别，并指出了该平台的特定优势。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/new-api-v2">Introducing Cohere’s Updated APIs</a>: Cohere 最新的 API 为开发者提供了新功能和改进。</li><li><a href="https://dashboard.cohere.com/fine-tuning/create?endpoint=chat).">Login | Cohere</a>: 登录以通过易于使用的 API 访问高级 Large Language Models 和 NLP 工具。</li><li><a href="https://docs.cohere.com/docs/cohere-faqs#billing-pricing-licensing-account-management">Cohere FAQs — Cohere</a>: Cohere 是一个使用 Large Language Models (LLMs) 的强大平台。此页面涵盖了与功能、定价、故障排除等相关的 FAQs。</li><li><a href="https://docs.cohere.com/v2/docs/structured-outputs-json">Structured Generations (JSON) — Cohere</a>: 此页面描述了如何让 Cohere 模型以特定格式（如 JSON）创建输出。</li><li><a href="https://docs.cohere.com/v2/docs/tool-use">Tool Use — Cohere</a>: 使您的 Large Language Models 能够连接外部工具，以实现更高级和动态的交互。</li><li><a href="https://docs.cohere.com/v2/docs/chat-fine-tuning">Fine-tuning for Chat — Cohere</a>: 本文档提供了关于 Fine-tuning、评估和改进聊天模型的指导。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1291858381694566451)** (9 messages🔥): 

> - `Cohere command R plus API issues`
> - `Rerank API concerns`
> - `Unicode escape sequences` 


- **Cohere command R plus API 生成 Unicode 转义序列**：用户报告称 **Cohere command R plus API** 返回的搜索查询格式包含 **Unicode 转义序列**，例如 `d\u00e9lat po po\u0159izen`。
   - *Mitchel555* 指出这种情况已经持续了一周，导致输出错误，并提到由于影响到客户，急需解决方案。
- **寻求 API 问题支持**：一位用户建议受影响的人员联系 **support@cohere.com**，并提供详细的示例和代码片段。
   - 由于聊天机器人平台影响到了付费客户，解决这些技术问题的紧迫感很高。
- **对 Rerank API 文档响应的担忧**：有疑问指出，即使使用了 **return_documents: True** 参数，**Rerank API** 也没有返回发送文档的预期数据。
   - 一位用户提到之前的功能现在已失效，正在寻求有关任何更改或持续性问题的信息。



**提及链接**：<a href="https://docs.cohere.com/docs/overview#example-with-semi-structured-data)">Rerank Overview — Cohere</a>：该页面描述了 Cohere 的 ReRank 模型是如何工作的。

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1291842230767587400)** (8 messages🔥): 

> - `Companion Discord Bot`
> - `Moderation Tools`
> - `User Interaction` 


- **介绍 Companion Discord 机器人**：一名成员介绍了 **Companion**，这是一款由 Cohere 驱动的 Discord 机器人，专为服务器社区内的**动态人格建模**和丰富交互而设计。它包含集成的**审核功能**，在进行个人层面交流的同时确保用户安全。
   - 您可以在 [GitHub](https://github.com/rapmd73/Companion) 上探索该项目，了解详细的功能和特性。
- **作为审核工具的潜力**：一名成员建议 **Companion** 可能会增强 Discord 内的审核任务。另一名成员表示赞同，强调这是该机器人能力的坚实应用场景。
   - 讨论强调了利用 AI 改善服务器社区互动并保持尊重氛围的好处。



**提及链接**：<a href="https://github.com/rapmd73/Companion">GitHub - rapmd73/Companion: A discord chat bot utilizing AI in a fun and whimsical way. Provides some moderation tools as well.</a>：一个以有趣和奇特的方式利用 AI 的 Discord 聊天机器人。还提供了一些审核工具。

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1291847411202789467)** (93 messages🔥🔥): 

> - `SWE-bench Multimodal`
> - `Reka Flash update`
> - `Cursor Team on Lex`
> - `AI job application automation`
> - `News aggregation tools` 


- **SWE-bench Multimodal 发布，用于解决视觉问题**：新的 SWE-bench Multimodal 旨在评估 Agent 解决视觉 GitHub issue 的能力，包含来自 **17 个 JavaScript** 仓库的 **617 个新任务**。
   - 该计划针对现有 Agent 的不足，并引入了 **SWE-agent Multimodal** 以更好地处理这些任务。
- **Reka Flash 更新增强了多模态能力**：Reka Flash 发布了新版本，支持**文本、图像、视频**和**音频**等交错的多模态输入，承诺提供改进的功能。
   - 此次更新侧重于推进实际应用场景中的**多模态理解**和通用推理，展示了实验室的进展。
- **Cursor 团队与 Lex Fridman 讨论 AI 辅助编程**：对话以 **Cursor 团队**为主角，探讨了其 AI 辅助编程环境的复杂性以及编码的广阔未来。
   - 关键时间点涵盖了对 **GitHub Copilot**、**ML 细节**以及在编程中集成 AI 的挑战等话题的讨论。
- **AI 机器人有效自动化职位申请**：一个 AI 机器人声称能在 24 小时内处理 **1000 份职位申请**，并获得 **50 次面试**，简化了 LinkedIn 的申请流程。
   - 它使用 LLM 个性化回复，高效管理批量申请，并与 OpenAI 的 API 集成以增强用户体验。
- **寻求更好的新闻搜索工具**：一位用户正在寻找搜索特定主题新闻文章的有效工具，表示对现有的聚合器不满意。
   - 建议包括用于源聚合的 **Follow** 和用于潜在洞察的 **newsandmoods.com**，这些是很有帮助的初步尝试。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>

<li><a href="https://x.com/jxmnop">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/rekaailabs/status/1843298155682820566?s=46">来自 Reka (@RekaAILabs) 的推文</a>：过去几个月我们一直很忙，有一些令人兴奋的更新！📢 我们发布了新版本的 Reka Flash⚡️，这是我们强大的 21B 模型，支持交错的多模态输入（文本📄、图像🖼️、视...</li><li><a href="https://x.com/rohanpaul_ai/status/1842712127556956230?s=46">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：有人使用 AI Bot 在 24 小时内自动申请了 1000 个职位，并获得了 50 个面试机会！🤯 代码已在 GitHub 上发布，并获得了高达 12.7K 的 Stars 🌟 它能自动完成你的 LinkedIn 职位搜索...</li><li><a href="https://x.com/nutlope/status/1842286649230938615?s=46">来自 Hassan (@nutlope) 的推文</a>：宣布推出 http://blinkshot.io！一个开源的实时 AI 图像生成器。输入提示词，图像将随输入实时生成。100% 免费且开源。</li><li><a href="https://highlightai.com/">Highlight AI | 掌控你的世界</a>：针对你所见、所闻或所说的一切获取即时答案。加入 Discord：discord.gg/hlai</li><li><a href="https://x.com/Jacob_Heller/status/1843137269815005364">来自 Jake Heller (@Jacob_Heller) 的推文</a>：@HamelHusain 我说错了。我们的评估并非真的是 100%；事实上，其中有很多我们知道 LLM 目前无法处理的情况（我们希望有一天它能处理）。我也不认为我们真的达到了 100...</li><li><a href="https://x.com/ericsimons40/status/1843345406576787496">来自 Eric Simons (@ericsimons40) 的推文</a>：大家好——以下是我们的最新动态！首先：对 http://bolt.new 的反响感到受宠若惊和震惊……前 72 小时内发送了 300k+ 条消息，发布了数万个精美网站，使用量...</li><li><a href="https://x.com/clefourrier/status/1842286565374193665?s=46">来自 Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>：新的 LLM 排行榜：金融领域！💰 它使用了 40 个领域相关任务，从预测和风险管理到问答和信息提取！目前排名前 3 的模型：- @OpenAI 的 GPT4 ...</li><li><a href="https://alterhq.com.">Alter | 为 Apple 高级用户打造的 AI</a>：未找到描述</li><li><a href="https://x.com/imrat/status/1843205318165004772">来自 Imrat (@imrat) 的推文</a>：我刚看了 Lex Fridman 采访 Cursor 团队播客的第一小时。我整理了其中 10 个我最喜欢的时刻，并在下面剪辑了播客片段。让我知道如果你...</li><li><a href="https://bolt.new/">bolt.new</a>：未找到描述</li><li><a href="https://www.newsandmoods.com/">新闻阅读器 - Lexxe</a>：未找到描述</li><li><a href="https://x.com/jyangballin/status/1843285832263979470?s=46">来自 John Yang (@jyangballin) 的推文</a>：我们正在发布 SWE-bench Multimodal，用于评估 Agent 解决视觉 GitHub issue 的能力。- 来自 17 个 JavaScript 仓库的 617 个*全新*任务 - 每个任务都有一张图片！现有的 Agent 表现挣扎...</li><li><a href="https://x.com/snowmaker/status/1843015916050948372?s=46">来自 Jared Friedman (@snowmaker) 的推文</a>：CaseText 是首批大规模部署的垂直领域 AI Agent 之一。它是一个被数千名律师使用的 AI 法律分析师。哦，而且它在发布仅 2 个月后就以 6.5 亿美元的价格被收购。这里有...</li><li><a href="https://x.com/_philschmid/status/1842846050320544016">来自 Philipp Schmid (@_philschmid) 的推文</a>：@AnthropicAI Claude 3.5 Sonnet 在推理方面能超越 @OpenAI o1 吗？结合动态 Chain of Thoughts、反思和语言强化，现有的 LLM 如 Claude 3.5 Sonnet 可以通过提示词...</li><li><a href="https://x.com/BenMillerise/status/1842241555886719078">来自 Benjamin Miller (@BenMillerise) 的推文</a>：AI 的价值将是多少？我们的团队进行了几个月的研究，在财务数据中发现了一个令人惊讶的模式，@BusinessInsider 昨天对此撰写了一篇文章。我们同意等待 24 小时...</li><li><a href="https://x.com/jxmnop/status/1842236045074498026?s=46">来自 jack morris @ COLM (@jxmnop) 的推文</a>：我们花了一年时间开发 cde-small-v1，这是世界上最好的 BERT 规模的文本嵌入模型。今天，我们正在 HuggingFace 上发布该模型，并在 ArXiv 上发布论文。我认为我们的发布...</li><li><a href="https://x.com/lexfridman/status/1843010390772605183?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Lex Fridman (@lexfridman) 的推文</a>：这是我与 Cursor 创始团队的对话，Cursor 是一款流行的代码编辑器（基于 VSCode），专注于 AI 辅助编程。这是一场非常技术性的对话，其意义超出了...</li><li><a href="https://www.reddit.com/r/Lawyertalk/s/yi5lXXkcLS">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://follow.is/">Follow</a>：下一代信息浏览器</li><li><a href="https://github.com/RSSNext/Follow">GitHub - RSSNext/Follow: 🧡 下一代信息浏览器。</a>：🧡 下一代信息 br</li>

owser. 通过在 GitHub 上创建一个账户来为 RSSNext/Follow 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1291852309009006723)** (98 messages🔥🔥): 

> - `Discord 音频问题`
> - `Luma AI 应用`
> - `3D 建模技术`
> - `Gaussian Splatting`
> - `电影剪辑` 


- **Discord 音频问题困扰用户**：成员们在通话过程中遇到了各种 **音频问题**，好几个人无法正常听到彼此的声音，导致有人建议切换到 Zoom。
   - *Verymadbear* 幽默地评论道：**“如果麦克风不出点问题，那就不算是一场真正的会议”**。
- **探索 Luma AI 的魔力**：讨论围绕 **Luma AI** 展开，用户分享了使用该工具制作的令人惊叹的 **视频应用** 和项目链接，展示了其强大的功能。
   - Karan 强调了 **Luma** 在电影制作中的潜力，表示它对于 **电影剪辑** 和实现独特的相机运动非常有用。
- **将 3D 技术用于游戏开发**：成员们讨论了在游戏应用中以 **3D** 形式重现现实场景的可能性，并思考了利用 **Luma AI** 技术实现这一目标的可行性。
   - 针对基于真实环境将创意转化为功能性 **FPS 射击游戏** 的时间表和挑战，大家提出了疑问。
- **讨论 Gaussian Splatting**：小组对 **Gaussian Splatting** 表现出极大的热情，分享了相关资源链接，并讨论了其在视觉真实感方面的创新应用。
   - *Verymadbear* 强调了它对 **3D 建模** 和创建逼真环境的潜在影响。
- **分享资源和学习材料**：用户交换了各种有用的链接，包括一个与 **NeRFshop** 相关的精彩 GitHub 仓库，以及关于使用 **Luma AI** 的教程视频。
   - 几位成员对分享的见解表示感谢，*Yikesawjeez* 提到该工具存在 **免费层级 (free tier)** 供大家实验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karanganesan">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/aishashok14/status/1832760312455450907/video/1">来自 Aishwarya Ashok (@aishashok14) 的推文</a>：山中之夜——一部皮克斯风格的电影 :) 使用了 @midjourney (--sref 804246641)、@LumaLabsAI（相机运动）和 @udiomusic。在疲惫的攀登结束时去徒步旅行是什么感觉...</li><li><a href="https://vimeo.com/1012136742/065081e415">FREE YOSHI - 概念验证</a>：这是 Jeremy Rubier 在 Vimeo 上的作品 &quot;FREE YOSHI - PROOF OF CONCEPT&quot;，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://lumalabs.ai/web">Luma AI - Fields 仪表板</a>：用 AI 将你的想象变为现实。</li><li><a href="https://x.com/aishashok14/status/1829738607281635371/video/1">来自 Aishwarya Ashok (@aishashok14) 的推文</a>：慢即是美✨ 深呼吸，冷静的头脑，平和的温暖，放松的时刻……这些都是美好的！提醒我们所有人：慢很酷，慢很美。使用了 @midjourney 和 @LumaLabs...</li><li><a href="https://x.com/aishashok14/status/1828790536410730878/video/1">来自 Aishwarya Ashok (@aishashok14) 的推文</a>：稍等，正忙着制作一部茶园纪录片 AI 电影。☕️ 🍃 从郁郁葱葱的绿色种植园到浓郁的茶杯，制茶的过程是一种情感。使用 @midjourney 和 @LumaLabsAI 拍摄...</li><li><a href="https://x.com/lumalabsai/status/1841833038700761205?s=46&t=fm_-fV17wG2CozW7wmZR7g">来自 Luma AI (@LumaLabsAI) 的推文</a>：👀 那么... 你的选择是？🍊↔🍎？🥕↔🥦？🧁↔🍩？🍔↔🍕？使用 #LumaDreamMachine 关键帧制作 #foodforthought #hungry #foodie</li><li><a href="https://x.com/bennash/status/1840829850292011172?s=46">来自 Ben Nash (@bennash) 的推文</a>：使用快 10 倍的新版 @LumaLabsAI 制作的文本转视频驾驶舱场景</li><li><a href="https://lumalabs.ai/ios">‎Luma AI</a>：‎以惊人的 3D 质量展示你的世界，并在网络上的任何地方分享。由 Luma AI 为你带来。Luma 是一种使用 iPhone 通过 AI 创建令人难以置信的逼真 3D 的新方式。轻松捕捉产品...</li><li><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">用于实时辐射场渲染的 3D Gaussian Splatting</a>：未找到描述</li><li><a href="https://github.com/graphdeco-inria/nerfshop">GitHub - graphdeco-inria/nerfshop: NeRFshop: 神经辐射场的交互式编辑</a>：NeRFshop: Interactive Editing of Neural Radiance Fields - graphdeco-inria/nerfshop
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1291855812033249391)** (188 messages🔥🔥): 

> - `Stability.ai Discussion` (Stability.ai 讨论)
> - `Model Comparison` (模型对比)
> - `LoRA Training Challenges` (LoRA 训练挑战)
> - `Web UI Preferences` (Web UI 偏好)
> - `Post-Generation Edits` (生成后编辑)


- **在 SD 中选择 AMD 还是 NVIDIA**：在对比 **RX 6900 XT** 和 **RTX 4070** 时，许多用户建议在 **Stable Diffusion** 中生成图像时选择 **4070**，因为它具有更好的性能。
   - 其他建议包括考虑 **3080 Ti**，据称它比 4070 快约 **30%**。
- **视频生成模型**：对于文本生成视频，**CogVideoX** 目前被认为是现有的最佳开源模型，超越了 **Svd** 等旧模型。
   - 一些用户指出，与认知上更优的替代方案相比，**Stability** 不再是顶尖资源。
- **Stable Diffusion 的 UI 偏好**：从 **Automatic1111** 转向 **ComfyUI** 和 **Forge UI** 的用户表示两者都可行，但各有优势，**Forge** 被描述为 **Auto1111** 的一个更好的分支（fork）。
   - 许多人因其易用性和有效性而推荐 **ComfyUI**，同时也承认某些功能在任一 **UI** 中都能得到更好的设置。
- **LoRA 训练挑战**：一些用户报告在为 **SDXL** 训练 **LoRA** 时遇到困难，正在寻找专门用于故障排除和建议的频道。
   - 社区为那些尝试创建有效 **LoRA** 模型的人提供支持和资源。
- **生成后编辑**：有关于生成后编辑潜力的咨询，例如上传图像并重新生成特定区域（如肢体或头部）。
   - 突出显示并更改生成图像部分内容的可行性是用户感兴趣的话题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1002292111942635562/1026382406279770152/1292765999644545024">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友放松，甚至建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://elevenlabs.io/">ElevenLabs: Free Text to Speech &amp; AI Voice Generator | ElevenLabs</a>: 使用我们的 AI 音频，以数千种声音和 32 种语言创建最逼真的语音。在 Text to Speech 和 AI Voice Generation 方面的先驱研究。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1291844465610326049)** (129 条消息🔥🔥): 

> - `Opus 限制的变化`
> - `Perplexity 的用户体验问题`
> - `开发团队角色与功能更新`
> - `推荐奖励的周边商品公告`
> - `AI 模型性能对比` 


- **Opus 限制争议**：用户对最近在未事先通知的情况下将 **Opus** 消息减少到 **每天 10 条** 表示不满，引发了关于消费者权益和订阅预期的讨论。
   - 然而，一些报告称限制已调回 **50 条消息**，缓解了用户的一些担忧。
- **Perplexity 的用户体验**：几位成员报告了 **Perplexity** 的问题，包括访问 Pro 功能困难、客户支持响应缓慢，以及 API 与模型性能之间的差异。
   - 用户还注意到，平台的重点似乎正在转向促销活动，而不是实质性的服务改进。
- **关于开发团队和功能的询问**：有关于开发团队除了 **Mac app** 之外目前正在做什么的疑问，用户觉得随着时间的推移缺乏新功能。
   - 回复暗示重点可能更多地转向了赠品，而不是增强平台功能。
- **推荐奖励的周边商品**：一位新用户询问了与推荐计划相关的周边商品状态，表示对促销优惠感兴趣。
   - 其他人鼓励对客服响应保持耐心，并强调了关于用户激励的持续讨论。
- **关于 AI 模型性能的讨论**：成员们对比了 **Perplexity** 上可用的 AI 模型，并注意到感知到的质量下降，强调了将 Prompt 与预期结果匹配的重要性。
   - 这引发了关于优化用户 Prompt 以在平台内获得更好研究效果的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.testingcatalog.com/tag/perplexity/">Perplexity - TestingCatalog</a>：报告 AI 乱象。由虚拟助手驱动的未来新闻媒体</li><li><a href="https://x.com/apostraphi/status/1843313891889267103?s=46">Phi Hoang (@apostraphi) 的推文</a>：能发生的最好的事情是什么？</li><li><a href="https://tenor.com/view/whisper-oh-gif-22523198">Whisper Oh GIF - Whisper Oh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/trash-garbage-dumpster-gif-22255810">Trash Garbage GIF - Trash Garbage Dumpster - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1291843888524431362)** (16 条消息🔥): 

> - `量子钟`
> - `仿射群`
> - `特拉亨伯格速算法`
> - `特斯拉的市场表现`
> - `示例中的差异` 


- **探索量子钟的新计时方式**：一个链接讨论了 [量子钟](https://www.perplexity.ai/search/what-is-a-quantum-clock-t4A_.5lTTiCUnbMObd_5_A) 的创新概念及其对精密计时的影响。
   - 量子钟有望在准确性上取得突破，超越传统的计时方法。
- **理解仿射群**：分享了一个关于 [仿射群](https://www.perplexity.ai/search/query-affine-group-with-detail-l4N2B5cFQFef_zsj5dQ59A) 的深刻链接，详细介绍了它们的数学意义。
   - 成员们围绕这些群在各个领域的独特属性和应用展开了讨论。
- **掌握特拉亨伯格速算法进行心算**：重点推荐了一个关于 [特拉亨伯格速算法 (Trachtenberg Shortcut)](https://www.youtube.com/embed/0gAHCBDZ-U8) 的视频，该算法简化了心算技巧。
   - *今天就来发现* 这种方法如何增强心算并提高解决问题的速度。
- **审视特斯拉的市场趋势**：关于 [特斯拉最近的下跌](https://www.perplexity.ai/search/why-is-tesla-s-decline-smaller-VdpxDOJKTAGM_pbVRc3e_w) 及其与其他市场竞争对手相比影响较小的讨论引起了关注。
   - 分析师分享了对可能影响这些趋势的市场策略和消费者情绪的看法。
- **通过示例澄清定义**：一位成员发起了一场关于 [直接示例](https://www.perplexity.ai/search/what-is-an-example-of-direct-a-j0dLeCJnTji_Um.MCvic1A#1) 的对话，以有效地说明定义和概念。
   - 这引发了对具体示例如何增强对复杂话题理解的探索。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1292556494549815366)** (3 messages): 

> - `Perplexity API Settings`
> - `Structured Outputs functionality`
> - `Recent fixes` 


- **导航至 Perplexity API Dashboard**：一名成员指示通过 **Settings** -> **API** -> **View Dashboard** 来访问必要的设置。
   - 这突显了管理 API 设置和配置的简便方式。
- **Perplexity API 中的 Structured Outputs**：有人提出了关于 **Perplexity API** 是否能像 [OpenAI library](https://platform.openai.com/docs/guides/structured-outputs/introduction) 一样处理 **Structured Outputs** 的问题。
   - 这反映了用户对 Perplexity API 框架内高级功能的兴趣日益增长。
- **Perplexity API 中已实施的修复**：一名成员指出，Perplexity API 的一个问题据报道现已 **fixed**（修复）。
   - 这表明官方正在进行持续的改进和更新，以提升用户体验。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1292166129514450966)** (5 messages): 

> - `Multi-agent architecture for video generation`
> - `Multi-Document Agentic RAG`
> - `Agentic retrieval for RAG pipelines`
> - `Multi-agent Legal AI`
> - `Multimodal RAG with Contextual Retrieval` 


- **Swarm Agents 创建 AI 生成的视频**：由 [@lifeoftomi](https://twitter.com/lifeoftomi) 发起的一个项目展示了如何构建一个 Agent “swarm”（集群），使其能够根据简单的自然语言提示，自主创建并上传 AI 生成的 YouTube 视频。
   - 欲了解更多见解，请查看[此处教程](https://t.co/TKs9QqP4ym)。
- **RAG 中的动态数据源推理**：在 RAG 流水线之上引入一个 Agent 层，可以将不同的数据源视为“tools”，从而实现对检索来源的动态推理。
   - 详细介绍请访问[此链接](https://t.co/jUzqZrnCOH)。
- **Agentic Retrieval 的快速设置**：[@fahdmirza](https://twitter.com/fahdmirza) 提供的一份指南介绍了在 RAG 流水线中快速设置 Agentic Retrieval 的方法，相比标准的固定检索方法提供了更大的灵活性。
   - 探索这一高效过程，请参考此[教程](https://t.co/V0JwbQ4Dmz)。
- **通过 Multi-Agent 系统实现法律合规**：由 [@farzad528](https://twitter.com/farzad528) 开发的一个令人印象深刻的 Multi-Agent 系统，可帮助公司自动评估法规合规性、审查法律判例并起草正式的法律答复。
   - 更多细节可以在[此处](https://t.co/s1MhinpZ5B)找到。
- **在 Slide Decks 上构建 RAG**：介绍了如何在 Slide Decks（幻灯片组）上构建多模态 RAG 流水线，允许对每张幻灯片的文本和图像内容进行预提取和索引。
   - 学习如何实现，请查看[此资源](https://t.co/jZLtlNy9M9)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1291845954865201203)** (85 条消息🔥🔥): 

> - `LlamaIndex Integration`
> - `Embedding Errors`
> - `Context Window Management`
> - `Chat UI Recommendations`
> - `Docstore Functionality` 


- **LlamaIndex 在 Milvus DB 集成方面存在困难**：一位用户表达了将 Milvus 集成到其 LlamaIndex 工作流中的挫败感，指出了 API 变更和对原生对象的依赖所带来的挑战。
   - 他们寻求一种更模块化的方法，以便在不被迫使用 LlamaIndex 的结构化对象的情况下，有效地利用预构建组件。
- **Gemini 模型的 Embedding 错误**：一位成员在使用 Gemini 模型时遇到了 Embedding 错误，指出需要在环境中正确设置模型。
   - 另一位用户提醒他们确保模型已在本地部署，并强调如果需要，应增加请求超时时间。
- **澄清 Context Window 机制**：关于 Context Window 的讨论澄清了它包含模板和聊天历史等动态元素，而不是一个静态容器。
   - 讨论强调了 System Prompt 确实会随每条消息一起发送，这有助于构建交互框架。
- **Chat UI 推荐**：当被问及 Chat UI 推荐时，用户建议了 create-llama 和 ragapp 等选项，这些选项不需要 LlamaCloud。
   - 他们指出 LlamaCloud 主要提供托管和简化的 UI，但对于功能实现并非必需。
- **LlamaIndex 中的 Docstore 能力**：一位用户寻求澄清 Docstore 是保存 Chunk 还是完整 Document，结果发现它可以有效地存储两者。
   - 据指出，Document 和 Chunk 都运行在相同的类类型下，从而允许在 Docstore 中进行多种用途的使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/#setup">Ollama - Llama 3.1 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/">Chat Engine - Context Mode - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/docstores/">Document Stores - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/#document-management">Ingestion Pipeline - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/af6ea71c787811cf4c11ebfccf758530140b8380/llama-index-core/llama_index/core/chat_engine/simple.py#L59">llama_index/llama-index-core/llama_index/core/chat_engine/simple.py at af6ea71c787811cf4c11ebfccf758530140b8380 · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/af6ea71c787811cf4c11ebfccf758530140b8380/llama-index-core/llama_index/core/chat_engine/simple.py#L95">llama_index/llama-index-core/llama_index/core/chat_engine/simple.py at af6ea71c787811cf4c11ebfccf758530140b8380 · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/af6ea71c787811cf4c11ebfccf758530140b8380/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py#L221">llama_index/llama-index-core/llama_index/core/chat_engine/condense_plus_context.py at af6ea71c787811cf4c11ebfccf758530140b8380 · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/af6ea71c787811cf4c11ebfccf758530140b8380/llama-index-core/llama_index/core/chat_engine/utils.py#L23">llama_index/llama-index-core/llama_index/core/chat_engine/utils.py at af6ea71c787811cf4c11ebfccf758530140b8380 · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1291949589821395025)** (29 messages🔥): 

> - `Gradient Checkpointing`
> - `VAE Training`
> - `Tinybox and Local Servers`
> - `VIZ and Scheduler Updates`
> - `Upcoming Stream and Project Plans` 


- **Gradient Checkpointing 讨论**：一位成员询问了 **gradient checkpointing** 的实现，这对于高效训练大型模型至关重要。
   - 另一位成员跟进并强调，**如果没有这些优化**，tinygrad 只能处理 **非常小的玩具模型**。
- **VAE 训练见解**：围绕训练 **Variational Autoencoder (VAE)** 以使现有模型适应 CIE LAB 色彩空间从而改善输出展开了讨论。
   - 这引发了一个建议，即对输入的重大改动将需要超出简单 **finetuning** 的广泛修改。
- **探索 Tinybox 作为本地服务器**：一位用户寻求关于 tinygrad 功能的澄清，想知道它是否可以作为运行 LLM 的 **本地服务器**。
   - 澄清指出 tinygrad 更类似于 **PyTorch**，侧重于开发而非服务器功能，同时提到了 **Tinybox** 作为一个产品选项。
- **VIZ 与 Scheduler 增强更新**：分享了关于 **VIZ server** 完全重写的更新，旨在增强其在 kernel 和 graph 重写方面的功能。
   - big graph 的主要阻碍包括处理 **ASSIGN** 以及随着工作进展完善 fusion 和 grouping 逻辑。
- **George Hotz 即将进行的直播与项目计划**：George Hotz 宣布计划在 **明天直播**，重点是 lazybuffer 的迁移和潜在的云集成。
   - 他强调在 **1.0** 版本发布前需要一个精致的前端，并鼓励通过其 GitHub 上的 **good first issues** 进行贡献。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tinycorp.myshopify.com">tiny shop</a>: tiny shop</li><li><a href="https://x.com/__tinygrad__/status/1842873146057339323">来自 tiny corp (@__tinygrad__) 的推文</a>：在 tinygrad GitHub 上添加了一堆 "good first issues"。这是进入 tinygrad 开发的好方法。请编写整洁的代码和测试！在 1.0 之前，我们需要这个前端闪闪发光。随时...</li><li><a href="https://github.com/geohot/ai-notebooks/blob/master/rnn_shakespeare_tinygrad.ipynb">ai-notebooks/rnn_shakespeare_tinygrad.ipynb at master · geohot/ai-notebooks</a>：一些实现 AI 算法的 ipython notebook。通过在 GitHub 上创建账号为 geohot/ai-notebooks 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/issues/6803">JIT 产生错误输出 SDXL SplitVanillaCFG · Issue #6803 · tinygrad/tinygrad</a>：在 master 上运行以下内容正常：$ python examples/sdxl.py --seed 0，输出通过 distance=0.00034500996116548777 验证。更改代码以使用 SplitVanillaCFG 会导致验证...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/6931">取代所有 GRAPH 工具的 VIZ 路线图 · Issue #6931 · tinygrad/tinygrad</a>：将 VIZ 引入 tinygrad 核心，取代 GRAPH, GRAPHUOPS, SAVE_SCHEDULE, JITGRAPH 等（删除 engine/graph.py 的所有内容）。完全重写所有 VIZ server 通用 graph_rewrite 上下文追踪器 Fuzze...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/6811">由 geohot 发起的 big graph 工作 · Pull Request #6811 · tinygrad/tinygrad</a>：给 @Qazalin 的概念验证，关于 big graph 的基本想法。需要一点时间把高级 scheduler 功能加入其中，这是确保它们经过良好测试的好时机。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1291918315635347468)** (50 条消息🔥): 

> - `TinyGrad 中的 KAN 网络`
> - `Wolpertinger 网络实现`
> - `DreamerV3 编译器问题`
> - `TinyGrad 线性层优化`
> - `测试期间的 CUDA 内存管理` 


- **在 TinyGrad 中探索 KAN 网络**：一位成员指出，尽管 **KAN 网络** 备受关注，但在 TinyGrad 中很难找到现有的实现，并分享了展示使用 MLP 层进行训练的简便性的示例。
   - **FastKAN** 在 MNIST 训练中比同类模型实现了 **10 倍的加速**，展示了其通用性和性能。
- **实现 Wolpertinger 网络**：重点介绍了一个在 TinyGrad 中成功实现的 **Wolpertinger 网络**，展示了利用提供的调试工具编写这种复杂的强化学习结构的便捷性。
   - 社区对完善文档以及可能创建一个单独的仓库来存放该实现并维持质量标准表示了兴趣。
- **DreamerV3 编译器的挑战**：**DreamerV3** 的初始版本已完成，但由于超过了设备上的参数限制，训练面临 **AssertionError** 问题。
   - 成员们分享了有用的调试见解，包括调整索引限制以防止溢出，以及隔离故障 Kernel 的方法。
- **优化线性层实现**：一位新成员寻求在 TinyGrad 中实现 **MLXQuantizedLinear** 的帮助，并指出其当前线性层实现存在性能问题。
   - George 强调使用 `.realize()` 来处理延迟执行（lazy execution），并建议使用不同的调试级别进行性能分析（profiling）以提高速度。
- **通过测试管理 CUDA 内存**：一位用户在运行测试时遇到了 CUDA 显存不足（out-of-memory）错误，并询问运行所有测试所需的内存。
   - 设置 `CI=1` 显著改善了测试结果，因为它提供了更小的测试用例，使得管理有限的 GPU 资源变得更加容易。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/mdaiter/tinygrad/tree/0.9.2_dreamer_buffer_count_limit/examples/dreamerv3">tinygrad/examples/dreamerv3 at 0.9.2_dreamer_buffer_count_limit · mdaiter/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你热爱 tinygrad！❤️  - mdaiter/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/0ecc417dd2a9d7bb4be3b2877f503b44c4cec827/test/test_custom_function.py">tinygrad/test/test_custom_function.py at 0ecc417dd2a9d7bb4be3b2877f503b44c4cec827 · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你热爱 tinygrad！❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/issues/3961">kernel index can overflow int32 · Issue #3961 · tinygrad/tinygrad</a>: 例如 #3271 和 beam searching resnet，如果索引 > int32 则会断言失败。#4157 修复了线性化器并检查索引最大值，必要时使用 int64。如果索引 > int64 则断言失败。</li><li><a href="https://github.com/tinygrad/tinygrad/pull/6690/files">FastKAN example by mdaiter · Pull Request #6690 · tinygrad/tinygrad</a>: 这实现了 FastKAN，详情见：https://arxiv.org/abs/2405.06721 训练速度极快！在此处用于 MNIST 训练。此外，我还测试了其中包含的 Attention Transformer 模块...</li><li><a href="https://github.com/mdaiter/wolpertinger">GitHub - mdaiter/wolpertinger: Wolpertinger agents - *on tinygrad*</a>: Wolpertinger agents - *基于 tinygrad*。通过在 GitHub 上创建账号来为 mdaiter/wolpertinger 做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1292224292091592794)** (24 messages🔥): 

> - `OpenAI o1 Model Insights` (OpenAI o1 模型见解)
> - `Entropix/Entropy Guided Adaptive Sampler` (Entropix/熵引导自适应采样器)
> - `Health Issues Impacting ASI Lab` (影响 ASI Lab 的健康问题)
> - `Inference Code Sharing` (推理代码共享)
> - `ICLR vs ICML Discussions` (ICLR 与 ICML 的讨论)


- **OpenAI o1 模型集成**：讨论强调 **OpenAI o1** 直接将推理集成到模型中，避免了 Noam Brown 提到的推理过程中传统的 **MCTS** 等范式。
   - *成员们表示怀疑*，指出此类说法可能简化了潜在的挑战，特别是考虑到之前的评论暗示某些讨论已被删除。
- **探索 Entropix 采样器的功能**：**Entropix/熵引导自适应采样器**显示出良好的前景，通过评估 attention entropy（注意力熵）来实现 prompt 优化，并通过降低熵来驱动模型性能。
   - 讨论的关键优势包括叙事连贯性的提高和 hallucination（幻觉）的减少，正如 @_xjdr 所言，这表明即使在小模型中也具有显著的能力。
- **健康问题导致 ASI Lab 关闭**：由于**恶化的健康问题**，@_xjdr 宣布关闭 ASI Lab，并对许多可能永远无法面世的项目表示遗憾。
   - 然而，这一转变允许更开放地共享推理代码，并有机会在没有实验室约束的情况下探索新途径。
- **RekaAI 和 Entropix 讨论**：成员们分享了与 **Entropix sampler** 相关的各种帖子，包括其实现见解和观察到的能力，许多人对此表示兴趣。
   - 讨论还延伸到了该频道是否适合此类话题，表明了潜在的更广泛兴趣和相关性。
- **ICLR 与 ICML 的适用性**：一位成员表示在讨论**模型概念**时更倾向于 ICLR 而非 ICML，强调关注实质性内容而非充斥定理的演示。
   - 这引发了关于在 Discord 频道内分享某些内容的适当性的对话，成员们反思了讨论的相关性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/s14joshi/status/1843300092310339913?s=46">Siddharth Joshi (@s14joshi) 的推文</a>：@_xjdr 我快速尝试了一下，这样你就不用纳闷了 :-) 更多是 ICLR 风格而非 ICML —— 主要是因为我不想搞那一套定理定义的繁琐流程</li><li><a href="https://x.com/_xjdr/status/1842256651669381413?s=46">xjdr (@_xjdr) 的推文</a>：由于联合创始人健康问题不断恶化，10 月 1 日我的 ASI Lab 正式关停了最后的集群并关闭。有很多我们正在进行的事情，我希望我能分享...</li><li><a href="https://x.com/aidan_mclau/status/1842550225824809439">Aidan McLau (@aidan_mclau) 的推文</a>：我有 80% 的把握 o1 是这样工作的：> 收集问答对数据集 > 模型生成推理步骤（句子） > RL 环境，其中每个新的推理步骤是一个动作 > 无 fan...</li><li><a href="https://x.com/_xjdr/status/1842697597842252163?s=46">xjdr (@_xjdr) 的推文</a>：上次推送中的实现已经足够稳定，即使使用固定阈值（令人惊讶地），也能对采样器在 CoT 或推理之外的能力做出一些观察：1) Prompt Optimizer：...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1292962395446313070)** (5 messages): 

> - `Llama 3.2 11B Vision`
> - `Llama 3.2 8B Text`
> - `Text performance comparison` (文本性能对比)


- **关于 Llama 3.2 模型文本性能的辩论**：一位成员询问 **Llama 3.2 11B Vision** 模型还是 **Llama 3.2 8B** 模型在纯文本场景下表现更好。
   - 另一位成员认为 **8B 模型**可能会优于 **11B Vision** 模型，称后者的增加部分主要集中在图像处理上。
- **11B 模型可能会降低文本性能**：对于 **11B 模型**在具备额外图像处理功能的情况下，其纯文本性能是否有所下降存在怀疑。
   - 指出的关键点是，**11B 模型**的所有额外功能都是专门为处理图像而设计的，这意味着在文本任务上可能存在权衡。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1291856083593330719)** (45 messages🔥): 

> - `Canvas 合成数据`
> - `Reflection 70B 性能`
> - `Open O1 模型`
> - `播客设置计划`
> - `Rao2Z 规划论文` 


- **Canvas 利用合成数据生成**：一位成员强调了他们使用来自 OpenAI o1-preview 的新型**合成数据生成技术**来改进 GPT-4o 以构建 Canvas 的工作，从而实现了高质量的行内注释。
   - 这种方法允许在不依赖人工生成数据的情况下实现**快速模型改进**，吸引开发者使用新的蒸馏产品。
- **Reflection 70B 未达到基准测试要求**：一位社区成员表示失望，因为他们根据 Sahil 的数据集复现的 **Reflection 70B** 未能达到最初报告的基准测试结果。
   - 他们仍致力于探索 reflection tuning 概念，并表示很快将分享关于该模型时间线的更详细发现。
- **Open O1 成为 OpenAI 模型的竞争对手**：一位成员介绍了 **Open O1**，将其作为专有模型的有力替代方案，声称它在推理、编程和数学方面表现出色，并提供了全面的基准测试对比。
   - 然而，一些社区成员认为围绕 **Open O1** 的整体讨论缺乏实质性见解，因此呼吁对这类模型进行仔细审查。
- **筹备引人入胜的播客**：讨论了播客计划，包括工作室设置以及对多麦克风和摄像机等设备的需求，以营造更好的录制环境。
   - 还有关于播客潜在长度的幽默讨论，以及建立一个幽默域名来批评新兴模型的想法。
- **Rao2Z 规划论文分析**：成员们审阅了一篇 **rao2z 规划论文**，该论文揭示了对于极长计划，规划/调度性能会下降，并在社区内确认了其有效性。
   - 该论文被描述为一次迭代更新，突显了在保持 arXiv 新论文持续产出的同时，对先前工作进行细微修改的模式。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/GeZhang86038849/status/1842560837396955327">来自 Ge Zhang (@GeZhang86038849) 的推文</a>：1/ 🚀 介绍另一个令人惊叹的开源项目的激动人心的消息！推出 Open O1，它是 OpenAI O1 等专有模型的强大替代方案！🤖✨ 我们的使命是赋能每一个人 ...</li><li><a href="https://x.com/GeZhang86038849/status/1842562244736901428">来自 Ge Zhang (@GeZhang86038849) 的推文</a>：4/ 💡 Open O1 在从推理、编程到数学和物理的各个领域都表现出色。无论你是开发者、研究人员还是爱好者，我们的模型都能彻底改变你的工作和项目。 ...</li><li><a href="https://huggingface.co/spaces/happzy2633/open-o1">Open O1 - happzy2633 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://fxtwitter.com/mattshumer_/status/1842313328166907995">来自 Matt Shumer (@mattshumer_) 的推文</a>：我根据 Sahil 的数据集和训练脚本复现的 Reflection 70B 现已完成，不幸的是，该模型未能达到最初报告的基准测试结果。我对此感到失望 ...</li><li><a href="https://tenor.com/view/kermit-darkside-star-wars-evil-innerme-gif-13048146">Kermit Darkside GIF - Kermit Darkside Star Wars - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/nickaturley/status/1842281132265484595">来自 Nick Turley (@nickaturley) 的推文</a>：构建 Canvas 我最喜欢的事情之一：我们使用了新型合成数据生成技术，例如蒸馏来自 OpenAI o1-preview 的输出，来微调 GPT-4o 以开启 canvas，制作 ...</li><li><a href="https://www.thirdwheelseattle.com/seattle-rates">西雅图费率 &mdash; Third Wheel 播客工作室 - 西雅图</a>：Third Wheel 提供单次课程以及折扣套餐，以满足您的播客需求。所有预订均包含专业播客工程师，因此您可以专注于您的嘉宾和内容...
</li>
</ul>

</div>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1292521290426618050)** (3 messages): 

> - `Obsidian Setup`
> - `RNN vs Transformers` 


- **对 Obsidian 配置感到满足**：一位成员分享了他们从中间状态转变为对右侧非花哨配置的 **Obsidian setup** 感到满足的心路历程。
   - *I feel blessed* 强调了他们对当前配置的满意度。
- **对 RNN 投资的迫切呼吁**：有人分享了一条推文，强调请求资金来开发 **再多一个 RNN**，并暗示它可能 *摧毁 Transformers* 并解决 long-context 问题。
   - 这条充满热情的消以反复的紧迫感结尾：*bro, please just need dollars*（兄弟，求你了，只需要美金）。



**提到的链接**：<a href="https://x.com/eric_alcaide/status/1842963071276667293">来自 Eric Alcaide @ CoLM (@eric_alcaide) 的推文</a>：兄弟，就再来一个 RNN。我保证，兄弟，只要再多一个 RNN，我们就能摧毁 Transformers。它只是一个更好的 RNN，兄弟。求你了，就再来一个。再多一个 RNN，我们就能搞定 longctx，兄弟。...

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1291948043221995622)** (3 messages): 

> - `DSL Model 中的类生成`
> - `实时编程 Notebooks`
> - `来自 DSPy 和 Jinja2 的结构化输出` 


- **类生成 Notebook 已发布**：GitHub 仓库现在包含一个 [关于类生成的 Jupyter notebook](https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/class_generation.ipynb)，展示了来自 DSPy 和 Jinja2 的 **structured outputs**（结构化输出）。
   - 该项目旨在增强各种应用中的结构化输出生成，并鼓励在 [GitHub](https://github.com/seanchatmangpt/dslmodel) 上进行更多贡献。
- **实时编程环节公告**：宣布了一个令人兴奋的实时编程环节，成员可以直接在 Discord 中观察 Notebook 的创建过程。
   - *鼓励参与者加入讨论帖*并在环节中互动，旨在促进协作式的 Notebook 开发。
- **关于 Notebook 创建的 Loom 视频分享**：一位成员分享了一个 [Loom 视频](https://www.loom.com/share/f181447ba7ed4af98ace0db82ca92109)，演示了高效创建 Jupyter notebooks 的技巧。
   - 该资源预计将为有兴趣提高 Notebook 制作技能的用户提供宝贵的见解和技术。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.loom.com/share/f181447ba7ed4af98ace0db82ca92109">使用 DSLModel 的 IPython Notebook 生成过程 📝</a>：在这个视频中，我将带你了解使用特定方法生成 Notebook 的过程。我们的目标是高效地简化多个 Notebook 的创建。我演示了如何提取和处理...</li><li><a href="https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/class_generation.ipynb">dslmodel/src/dslmodel/examples/class_generation.ipynb 分支 main · seanchatmangpt/dslmodel</a>：来自 DSPy 和 Jinja2 的结构化输出。通过在 GitHub 上创建账号来为 seanchatmangpt/dslmodel 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1292581781517893742)** (40 messages🔥): 

> - `TypedPredictors`
> - `Traceability in DSPy`
> - `Using dspy.LM`
> - `Custom Adapters vs Custom LMs`
> - `Error Handling in LMs` 


- **TypedPredictors 实现**：有一场关于在不使用 schema 格式化逻辑的情况下使用 `TypedPredictors` 的讨论，一位成员建议这可以用大约 100 行代码实现。
   - 一位成员确认这预计很快就会集成到 `dspy.Predict` 中。
- **在 DSPy 中实现可追溯性 (Traceability)**：一位用户询问如何在不使用外部库的情况下为 DSPy 添加可追溯性，特别是为了成本管理而跟踪 token 计数。
   - 建议使用 `your_lm.history` 属性来有效地监控成本。
- **向 dspy.LM 接口迁移**：一位新用户在从 `dspy.OllamaLocal` 迁移到 `dspy.LM` 时遇到了分段错误 (segmentation fault)，这凸显了可能的版本不匹配问题。
   - 提示回复建议重新安装 DSPy 或确认使用正确的模型端点 (endpoints) 可能会解决此问题。
- **评估自定义 LM 与自定义 Adapter**：鉴于 DSPy 2.5 的更新，一位成员建议记录创建自定义 Adapters 与自定义 LMs 的原因。
   - 他们强调，由于功能多样，在为 prompt 和任务函数选择不同模型时具有复杂性。
- **弃用自定义 LM 客户端**：文档指出，自 DSPy 2.5 以来，对自定义 LM 客户端的需求已经减少，敦促迁移到 `dspy.LM`。
   - 鼓励用户参考迁移指南以利用新功能，并确保与未来更新的兼容性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/custom-lm-client">Creating a Custom Local Model (LM) Client | DSPy</a>: ---</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1569">chat_adapter: Format fields as JSON by tkellogg · Pull Request #1569 · stanfordnlp/dspy</a>: 当字段为 Pydantic 对象时，chat_adapter 会将其格式化为 python 代码，这导致了一些奇怪的行为（BootstrapFewShot 会以 JSON 开始，然后退回到不可解析的...
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1291845335035154484)** (24 messages🔥): 

> - `Streaming responses from chat_manager`
> - `GitHub pull request for message processing`
> - `In-person attendance at Berkeley lectures`
> - `Confirmation for assignment grading` 


- **来自 chat_manager 的实时流式传输**：一位成员确认已创建一个 streamlit UI 来实时流式传输 **chat_manager** 的响应，并参考了一个相关的 [GitHub pull request](https://github.com/microsoft/autogen/pull/1783) 以实现类似功能。
   - 该代码允许自定义发送前处理消息的方式，这对于实时流式传输至关重要。
- **线下参加限制**：一位成员表示，由于教室容量限制，只有伯克利的学生可以线下参加讲座。
   - 在回答关于非伯克利学生是否有座位的提问时，再次重申了这一点。
- **作业评分确认**：澄清了成员在书面作业评分后将收到确认，以确保评分过程的透明度。
   - 此确认是课程内关于作业评估持续沟通的一部分。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent/#register_hook">agentchat.conversable_agent | AutoGen</a>: ConversableAgent</li><li><a href="https://github.com/microsoft/autogen/pull/1783">process message before send by sonichi · Pull Request #1783 · microsoft/autogen</a>: 为什么需要这些更改？添加一个钩子方法用于在发送前处理消息。示例应用：用于显示消息的自定义前端。为了清晰起见，重命名了其他钩子方法...
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1292973040207921182)** (1 条消息): 

> - `DSPy 贡献`
> - `Omar 的讲座` 


- **对 Omar 讲座的期待**：一位成员对即将到来的 **Omar** 关于 **DSPy** 主题的讲座表示了极大的热情。
   - 他们提到自己正积极参与 **DSPy**，并打算进一步做出贡献。
- **对 DSPy 的积极贡献**：该成员分享了他们最近一直在努力研究 **DSPy**，同时尝试为该项目做出贡献。
   - 这突显了他们对提升在 **DSPy** 框架中的技能和知识的承诺与兴趣。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1292990449606201375)** (1 条消息): 

> - `重构参数约定语法`
> - `Mojo 编程语言` 


- **关于重构参数约定语法的新提案**：一位成员分享了一份[关于重构参数约定和引用的提案](https://gist.github.com/lattner/da647146ea573902782525f3446829ff)，旨在完善 **Mojo** 编程语言的各个方面。
   - 鼓励通过 [GitHub Issue](https://github.com/modularml/mojo/issues/3623) 提供社区意见，以帮助完善此提案。
- **征集社区对 Mojo 提案的反馈**：提案发起人敦促成员参与讨论，以增强该提案在 **Mojo** 社区中的相关性。
   - 您在 GitHub 线程中的见解和评论对于*塑造 Mojo 的未来*至关重要。



**提到的链接**：<a href="https://github.com/modularml/mojo/issues/3623)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1292193977205915670)** (10 条消息🔥): 

> - `Mojo 基准测试框架`
> - `Mojo 中的枚举 (Enums)`
> - `核心关键字重新评估` 


- **Mojo 基准测试框架实现**：一位成员分享了 Mojo 拥有一个用于运行时基准测试的 [benchmark 软件包](https://docs.modular.com/mojo/stdlib/benchmark/)，类似于 Go 的测试框架。
   - 示例包括使用 `benchmark.run` 来评估函数性能，并生成详述平均持续时间和迭代次数的报告。
- **使用 Variant 类型定义枚举**：关于在 Mojo 中创建枚举的讨论明确了目前没有专门的枚举语法，但可以使用类似于 C++ std::variant 的 **Variant** 类型来实现功能。
   - 成员们指出，要创建标签（tags），可以声明一个结构体（struct）并为各种类型使用别名（aliases），直到完整的和类型（sum types）可用为止。
- **重新评估 Mojo 中的核心关键字**：针对正在进行的 **Mojo 引用子系统**设计提出了一项提案，促使对 'inout' 和 'borrowed' 等核心关键字进行重新评估。
   - 请求在相关的 [GitHub 讨论](https://github.com/modularml/mojo/issues/3623) 中提供反馈和想法，以完善设计。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/benchmark/">benchmark | Modular 文档</a>：实现了用于运行时基准测试的 benchmark 软件包。</li><li><a href="https://github.com/modularml/mojo/issues/3623">[讨论] 重构参数约定和引用 · Issue #3623 · modularml/mojo</a>：Mojo 引用子系统的设计正趋于完善。为了敲定主要点，重新评估 Mojo 早期的一些决策有助于使设计更加……
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1292942271267868683)** (5 条消息): 

> - `Max 推理引擎错误`
> - `Torch 版本详情`
> - `ONNX 算子问题` 


- **Max 推理引擎遇到的困难**：一位用户报告了在他们的 Intel NUC 上使用 **max 推理引擎**时遇到的问题，特别是 `libTorchRuntimePlugin-2_4_1_post100.so` 和 ONNX 算子的错误。
   - *错误包括算子合法化（legalization）失败*以及更改 opset 版本时的各种问题。
- **对 Torch 版本的要求**：另一位用户询问了 PyTorch 的安装情况，问道：*你安装的是哪个 torch 版本？*
   - 他们建议运行一条命令来获取 **torch 的版本**和配置详情。
- **已收到 Torch 版本输出**：用户提供了命令输出，详细说明了他们的 **PyTorch 版本**为 `2.4.1.post100` 以及其他构建细节。
   - 关键亮点包括 **GCC 版本 13.3** 和各种 Intel 优化，全部通过 **conda-forge 频道**安装。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1291867111567655043)** (11 messages🔥): 

> - `Torchtune 中的 KTO 训练支持`
> - `大型自定义 CSV 数据集的问题`
> - `LLAMA 3.2 3B 的全参数微调`
> - `Grace Hopper 芯片对比`
> - `amp.autocast 的 FutureWarning` 


- **Torchtune 目前缺乏 KTO 训练支持**：一位成员询问 Torchtune 是否支持 **KTO 训练**，另一位成员回答说如果需要，可以将其添加到 DPO recipe 中。
   - 他们建议提交一个 issue 来跟踪这一功能需求。
- **大型 CSV 数据集的 AssertionError**：一位用户报告在使用大于 **100MB** 的自定义 CSV 数据集时出现 **AssertionError**，特别是在使用 shuffle=false 时。
   - 该错误在较小的数据集上不会出现，表明可能存在与数据集大小相关的问题。
- **LLAMA 3.2 3B 的微调挑战**：出现了关于 **LLAMA 3.2 3B 全参数微调**的问题，提到蒸馏模型（distilled models）需要特殊处理，例如较低的学习率。
   - 一位成员声称通过提高学习率获得了合理的 loss 曲线，但缺乏评估数据来支持其发现。
- **关于 Grace Hopper 芯片的讨论**：一位成员询问了使用 **Grace Hopper 芯片**的经验，以及它们与配备 Hopper GPU 的常规架构相比如何。
   - 这突显了社区对新硬件设计在性能影响方面的持续关注。
- **与 amp.autocast 相关的 FutureWarning**：一位用户提出了关于 `torch.cpu.amp.autocast` 被弃用的 **FutureWarning**，并指出在 **2.5.0** 版本中已确定了潜在的修复方案。
   - 其他成员一致认为该 issue 可能会被关闭，显示了社区内有效的沟通。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/issues/1730">fix future warning amp.autocast · Issue #1730 · pytorch/torchtune</a>: &quot;/home/felipemello/.conda/envs/torchtune-v0.3.1/lib/python3.11/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: torch.cpu.amp.autocast(args...) is deprecated. Please use torch.amp.aut...</li><li><a href="https://github.com/pytorch/">pytorch</a>: pytorch 在 GitHub 上有 78 个可用仓库。</li><li><a href="https://github.com/pytorch/pytorch/blob/release/2.5/torch/utils/checkpoint.py#L1518.">pytorch/torch/utils/checkpoint.py at release/2.5 · pytorch/pytorch</a>: Python 中具有强 GPU 加速能力的张量和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1291917132879368252)** (4 messages): 

> - `最大序列长度 vs Batch Size`
> - `训练中的 Packing 效率`
> - `LLM 中的 Attention Masking`
> - `训练方法的比较` 


- **建议增加最大序列长度而非 Batch Size**：指南建议在进行 packing 时增加 **max sequence length** 而不是 batch size，因为在 **blockmask 维度**上性能更好。
   - 一位成员指出，使用更长的序列可以提高短序列的 **packing 效率**，但由于采用静态 packing 方法，可能会导致数据打乱（shuffling）程度降低。
- **探索 Packing 与独立样本的区别**：讨论强调了使用 batch size 为 4 且序列长度为 1024，与将 4 个序列 pack 到 4096 并应用 **attention mask** 之间的区别。
   - 成员们对计算成本和内存占用表示担忧，质疑在正确应用 attention mask 的情况下，这两种方法是否会产生相似的结果。
- **LLM 训练的实验建议**：建议有动力的人进行实验，比较上述两种训练方法。
   - 该请求包括发布 **Torchrune 命令**和结果，以阐明性能和资源利用率方面的差异。



**提到的链接**：<a href="https://www.reddit.com/r/MachineLearning/s/BbngGyx5Iw">Reddit - Dive into anything</a>: 未找到描述内容

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1292645298631610389)** (8 messages🔥): 

> - `Finetuned GPT-4 models`
> - `Logo change`
> - `Intel and Inflection AI collaboration` 


- **微调后的 GPT-4 模型丢失**：一位成员幽默地表示 OpenAI 可能拿走了所有人的微调 **GPT-4** 模型，称 *“我的模型丢了”*，并暗示这些微调模型的性能非常 *糟糕 (trash)*。
   - 另一位成员提醒道 *“你只能微调你拥有的权重”*，强调了使用共享资源的风险。
- **群组 Logo 变更引发的困惑**：一位成员提到由于 Logo 的变更，他找不到 Discord 群组了，并幽默地吐槽了这带来的困惑。
   - 该评论强调了品牌变更如何影响社区的导航和识别。
- **Intel 与 Inflection AI 联手**：一位成员分享了一篇关于 **Intel** 与 **Inflection AI** 合作推出企业级 AI 系统的文章，称其非常 *有趣*。
   - 这一公告表明企业级 AI 领域有重大进展，可能会重塑技术使用的各个方面。



**提到的链接**：<a href="https://community.openai.com/t/fine-tuned-models-not-showing-up-for-assistant/966375">Fine-tuned models not showing up for assistant</a>：我无法在 Assistant 中使用我最近制作的微调模型。我仍然可以使用很久以前制作的模型，但从昨天和今天开始，我完全无法使用它们了。...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1291868166791630909)** (3 messages): 

> - `Axolotl non-pip packaging`
> - `uv package manager`
> - `Dependency Management Challenges` 


- **探索 Axolotl 的非 pip 打包工具**：一位成员询问是否有人正在探索将 Axolotl 切换到像 **uv** 这样的非 pip 打包工具，因为他对安装和更新依赖项感到沮丧。
   - 他们表示有兴趣为任何旨在改善现状的努力做出贡献。
- **uv 在处理 CUDA PyTorch 版本控制方面存在困难**：另一位成员指出，**uv** 在处理 **CUDA** **PyTorch** 版本控制方面并不比现有解决方案更好。
   - 这一观点强调了管理 GPU 依赖项时面临的持续挑战。
- **依赖兼容性带来的挫败感**：一位成员分享说，使用该库最令人沮丧的部分是需要花费 **5 分钟以上** 才能找到兼容的包版本。
   - 这突显了 Axolotl 用户在依赖管理方面的关键痛点。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1292296194621575230)** (2 messages): 

> - `fschad package issue`
> - `Reproducing errors in package installation` 


- **fschad 包未找到错误**：一位用户报告在尝试安装 `axolotl[deepspeed,flash-attn]` 时遇到错误，提示 '**Could not find a version that satisfies the requirement fschat (unavailable)**'。
   - 列出的可用版本范围从 **0.1.1** 到 **0.2.36**，但没有一个被标记为可用，这引起了困惑。
- **关于错误复现的询问**：成员 nanobitz 询问了前一位用户复现 **fschad 错误** 的具体细节。
   - 这个问题反映了在故障排除中澄清导致问题的步骤的常见做法。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1291899143505055826)** (3 messages): 

> - `LlamaIndex RAG-a-thon`
> - `Team Formation for Hackathon`
> - `Clip Retrieval API Updates` 


- **LlamaIndex RAG-a-thon 公告**：**LlamaIndex Agentic RAG-a-thon** 将于 **10 月 11 日至 13 日** 在硅谷举行，重点关注检索增强生成 (RAG) 技术。
   - 该活动是与 **Pinecone** 和 **VESSL AI** 合作举办的，旨在促进企业级应用的高级 AI Agent 的开发。
- **寻找黑客松队友**：一位成员表达了为 **LlamaIndex RAG-a-thon** 组队的兴趣，展示了积极参与的态度。
   - 另一位成员评论说由于地理位置限制无法参加，这突显了潜在参赛者面临的各种挑战。
- **关于 Clip Retrieval API 的询问**：一位成员询问了 **clip retrieval API** 的更新情况，展示了对该技术开发的持续关注。
   - 目前没有收到回复，这表明可能需要从团队负责人或开发人员那里获取更多信息。



**提到的链接**：<a href="https://rag-a-thon-2.devpost.com/">AGENTIC RAG-A-THON ($12K in cash prizes)</a>：LlamaIndex RAG-a-thon 与 Pinecone 和 VESSL AI 合作 | 10 月 11 日 - 13 日

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1291870541673660508)** (10 messages🔥): 

> - `O1 performance`
> - `Model robustness`
> - `Epoch training`
> - `AIW problems`
> - `New tools` 


- **O1 在基础任务上表现不佳**：一场讨论强调了 **O1** 虽然声称在**奥林匹克竞赛级**的科学任务中表现强劲，但在更简单的问题上却失败了，这暴露了其缺乏鲁棒性和泛化能力。该讨论揭示了对其在基础推理任务中表现的担忧，正如相关[讨论](https://x.com/JJitsev/status/1842727628463128968)中所指出的。
   - 正如[研究论文](https://arxiv.org/abs/2406.02061)中所阐述的，这引发了关于 SOTA LLM 如何有效管理泛化能力的疑问。
- **O1 与人类相比存在局限性**：针对 **O1-preview** 和 **O1-mini** 的观点认为，尽管这些模型比前代有所进步，但与人类能力相比表现较差。对话强调这些模型尚未学会有效管理新概念。
   - 一位成员建议，虽然这些模型在解释能力上有所提高，但往往缺乏自我修正能力，除非它们在 reflection（反思）过程中发现了错误。
- **Epoch 训练见解**：一位用户分享了他们的训练经验，提到他们正在使用 **80,000 epochs**。这为围绕模型训练效率和性能指标的进一步讨论奠定了背景。
- **对新工具的兴趣**：一位用户分享了 [AutoArena](https://www.autoarena.app/) 的链接，称其为一个值得分享的有趣工具。这表明人们对探索用于模型增强的新资源持续关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/JJitsev/status/1842727628463128968">来自 Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：(又一个) 兴衰故事：o1 声称拥有非凡的强劲表现，在奥林匹克级别的数学和编程问题上得分很高。它能处理简单的 AIW 问题吗？这些问题揭示了泛化能力...</li><li><a href="https://www.autoarena.app/">AutoArena</a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1292318278886359040)** (10 messages🔥): 

> - `Grimes' Coachella Setup`
> - `Local LlamaFile Model Configuration`
> - `Discord Automod for Spam Control` 


- **Grimes 的 Coachella 01 AI 装置揭晓**：一份指南概述了 Grimes 和 Bella Poarch 如何在 Coachella 音乐节上使用宏按键键盘和麦克风设置他们的 [01 AI 助手](https://01.openinterpreter.com/hardware/grimes)。
   - *这个简单的设置包括购买一个宏按键键盘和麦克风，并重新映射按钮以与 AI 交互。*
- **本地 LlamaFile 模型面临挑战**：一位成员在尝试交互时遇到了本地 LlamaFile 模型的错误，提示：**'Model not found or error in checking vision support'**（未找到模型或检查视觉支持时出错）。
   - 该成员指出他们的模型 **'Meta-Llama-3.1-8B-Instruct'** 应该根据链接的配置进行映射，这导致了对错误原因的困惑。
- **Discord Automod 用于防止垃圾信息**：有一场讨论建议使用 Discord Automod 来阻止普通用户使用 **@everyone 标签**，以减少垃圾信息。
   - 一位成员指出 **95% 的垃圾信息机器人**都会尝试艾特所有人，这使得该方法成为打击垃圾信息的有效手段。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://01.openinterpreter.com/hardware/grimes">Grimes Build - 01</a>：未找到描述</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>：Python SDK，代理服务器（LLM 网关），用于以 OpenAI 格式调用 100 多个 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1291955633955934210)** (1 messages): 

> - `01 costs comparison`
> - `11 Labs vs OpenAI` 


- **比较 01 成本：11 Labs 对比 OpenAI**：一位成员提出了关于在 **11 Labs** 和 **OpenAI** 之间使用 **01 服务**相关成本的问题。
   - 他们担心可能需要升级 **11 Labs** 的会员资格，因为他们在其他服务中也在使用它。
- **对 11 Labs 会员资格的担忧**：同一位成员特别担心由于在其他地方的使用量，需要**提高他们在 11 Labs 的会员等级**。
   - 这种担忧反映了用户对利用这些平台的财务影响有着广泛的关注。

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1292255694342590555)** (2 条消息): 

> - `数字助手帽子`
> - `开源项目`
> - `编程生产力` 


- **创新的数字助手帽子构想**：一位用户提出了将**帽子**与**数字助手**集成的概念，具备扬声器、麦克风和一键通话（push-to-talk）按钮功能，以实现无缝交互。
   - 该项目旨在包含**手机通知**、问题回答和日历管理功能，并可能演变为一个[带有构建指南的开源项目](https://link.to.project)。
- **对编程辅助的兴奋**：另一位用户反应热烈，表示渴望拥有这样一种设备来增强他们的**编程项目**，并评论说 *Claude 还不够用*。
   - 他们的兴奋反映了人们对提高**编程生产力**以及与日常任务集成的工具日益增长的兴趣。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1291845934128435270)** (6 条消息): 

> - `LlamaIndex Agentic RAG-a-thon`
> - `Agent 音频问题`
> - `Cursor 向量化疑问`
> - `实现多工具 Agent`
> - `黑客松团队招募` 


- **加入 LlamaIndex RAG-a-thon！**：**LlamaIndex Agentic RAG-a-thon** 将于 **10 月 11 日至 13 日**在硅谷举行，重点关注检索增强生成（RAG）技术和 AI Agent。
   - 感兴趣的参与者可以通过[此链接](https://rag-a-thon-2.devpost.com/)了解更多详情，并通过 **[Slack](https://join.slack.com/t/futureproof101/shared_invite/zt-2s1c1rlxh-3p64w0UbYQFdjTIpfYb3KQ)** 或 **[Discord](https://discord.com/invite/eN6D2HQ4aX)** 进行联系。
- **移动端音频播放问题**：一位用户遇到了 **Agent 音频**在移动浏览器中无法正常播放的问题。
   - 这引发了协助排查播放故障的请求。
- **Cursor 声称拥有令人印象深刻的向量化能力**：针对 **Cursor** 声称在提交链接后几乎能瞬间完成整个文档向量化的说法，有人提出了疑虑。
   - 一位用户对他们是否真的在进行文档向量化表示怀疑，并询问该过程的具体细节。
- **多工具 Agent 实现指导**：有人请求关于如何**实现**一个使用多个工具的 Agent 的指导，建议是结合来自各种检索器（retrievers）的工具。
   - 这反映了人们对创建能够有效利用多样化数据源的 Agent 的兴趣日益浓厚。
- **为黑客松寻找队友**：几位成员正在寻找**团队**加入黑客松，并对差旅住宿表示不确定。
   - 这体现了社区成员渴望参与即将举行的活动的协作精神。



**提到的链接**：<a href="https://rag-a-thon-2.devpost.com/">AGENTIC RAG-A-THON ($12K 现金奖励)</a>：由 Pinecone 和 VESSL AI 赞助的 LlamaIndex RAG-a-thon | 10 月 11 日 - 13 日

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1292303815613218908)** (5 条消息): 

> - `使用自然语言和计算机视觉自动化 QA`
> - `Sci Scope AI 研究摘要`
> - `会花钱的 Agent` 


- **使用自然语言自动化 QA**：一位成员讨论了[一个平台](https://getautonoma.com/)，该平台结合自然语言和计算机视觉来自动化 QA，使团队能够在不引入 Bug 的情况下增加价值。
   - 功能包括 **Web 和移动端支持**、CI/CD 就绪以及可减少维护开销的**自愈**能力。
- **使用 Sci Scope 保持领先**：另一位成员介绍了 [Sci Scope](https://sci-scope.com)，它每周汇总新的 ArXiv 论文，并根据用户偏好进行总结，将见解直接发送到您的收件箱。
   - 订阅者受益于**个性化简报**，确保他们不会错过 AI 研究中的重要进展。
- **对支付型 Agent 的兴趣**：一位用户询问是否有人正在构建或考虑构建可以花钱的 Agent，引发了对潜在开发的讨论。
   - 虽然没有提到具体项目，但 Agent 具备金融交易能力的想法激发了对创新应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.usewalle.com/">Walle - Agent 支付</a>：让您的 Agent 在不存储卡片信息的情况下进行购买的最简单方式。</li><li><a href="https://sci-scope.com">Sci Scope</a>：关于 AI 研究的 AI 生成简报</li><li><a href="https://getautonoma.com/">Autonoma AI</a>：用于构建和运行端到端测试的 AI 驱动平台——无需编程。只需导入您的测试用例即可开始。
</li>
</ul>

</div>
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1292561280888213596)** (2 条消息): 

> - `MLOps World + GenAI Conference`
> - `Manifold Research Lab updates` 


- **参加第五届年度 MLOps World + GenAI Conference！**：该会议将于 11 月 7 日至 8 日在德克萨斯州奥斯汀举行，届时将有 **50+** 个主题、动手实践工作坊和社交机会。
   - [点击此处](https://mlopsworld.com/speakers)查看完整议程，不要错过 11 月 6 日的额外虚拟日！
- **探索 Manifold 的研究实验室和活动**：Manifold 正在举办名为 **CRCs** 的互动更新，重点关注其研究项目中 **Multimodality**、**Robotics** 等领域的进展。
   - 在其[活动页面](https://www.manifoldrg.com/events/)了解更多即将举行的活动，并[在此处](https://discord.gg/Pza3jxKPUY)加入 Discord 社区。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.manifoldrg.com/events/">Manifold Research Group (第 1 页)</a>：未找到描述</li><li><a href="https://www.manifoldrg.com/">Manifold Research Group</a>：Manifold Research 是一家新型研发机构，致力于追求高影响力的前沿科学和技术项目，最终目标是改善和推动人类文明。</li><li><a href="https://mlopsworld.com/speakers">演讲者 — MLOps World</a>：演讲者 — MLOps World
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1292870505849815128)** (1 条消息): 

> - `Data Pipelines for Model Fine-Tuning`
> - `Data Selection Process`
> - `Fine-Tuning Tasks` 


- **AIFoundry.org 关于 Data Pipelines 的播客**：本周三，[AIFoundry.org](https://aifoundry.org/) 将在 Mozilla AI 舞台上主持一场播客，讨论**用于模型 Fine-Tuning 的 Data Pipelines**。
   - 讨论将涉及所需的**数据量**以及针对 Fine-Tuning 任务的调整，这使其成为社区的热门话题。
- **社区关于数据处理的问题**：一个关键的社区话题集中在**数据选择和处理流程**应该是怎样的。
   - 他们正在寻求关于如何调整流程以实现有效适配其 **Fine-Tuning 任务**的模型见解。


  

---



### **DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/)** (1 条消息): 

thilotee: https://arxiv.org/abs/2410.02694
  

---



---



---



---



{% else %}


> 完整的频道细分内容已为邮件格式进行截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}