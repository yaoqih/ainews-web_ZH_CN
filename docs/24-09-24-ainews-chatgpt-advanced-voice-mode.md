---
companies:
- openai
- anthropic
- scale-ai
- togethercompute
- kyutai-labs
date: '2024-09-25T01:31:24.268326Z'
description: '以下是该文本的中文翻译：


  **OpenAI** 推出了 **ChatGPT 高级语音模式 (Advanced Voice Mode)**，新增了 5 种声音，并改进了口音和语言支持，目前已在美国广泛可用。在
  **Llama 3** 和 **Claude 3.5** 传闻更新之际，**Gemini Pro** 大幅降价，以符合新的“智能前沿”定价标准。**OpenAI
  的 o1-preview 模型**在规划任务中表现出色，在 Randomized Mystery Blocksworld 测试中达到了 52.8% 的准确率。传闻
  **Anthropic** 将发布新模型，引发了社区的广泛关注。**Qwen 2.5** 正式发布，模型参数最高达 32B，支持 128K token，性能可媲美
  GPT-4 0613 基准测试。


  研究亮点包括：针对 o1-preview 的 PlanBench 评估；OpenAI 发布了涵盖 14 种语言的多语言 MMMLU 数据集；以及旨在标准化检索增强生成研究的
  RAGLAB 框架。新的 AI 工具包括：将 PDF 转换为音频的 PDF2Audio；用于本地模型部署的开源 AI 入门套件；以及来自 Kyutai 的语音
  AI 助手 **Moshi**。行业动态方面，**Scale AI** 的年度经常性收入 (ARR) 接近 10 亿美元，同比增长 4 倍；**Together
  Compute** 的企业平台提供了更快的推理速度和更低的成本。此外，文中还分享了 **Sam Altman** 博客文章中的见解。'
id: 20ffb8d1-e347-4153-9868-61e5ed39c855
models:
- o1-preview
- qwen-2.5
- llama-3
- claude-3.5
original_slug: ainews-chatgpt-advanced-voice-mode
people:
- sam-altman
- omarsar0
- bindureddy
- rohanpaul_ai
- _philschmid
- alexandr_wang
- svpino
- ylecun
- _akhaliq
title: ChatGPT 高级语音模式
topics:
- voice-synthesis
- planning
- multilingual-datasets
- retrieval-augmented-generation
- open-source
- speech-assistants
- enterprise-ai
- price-cuts
- benchmarking
- model-performance
---

<!-- buttondown-editor-mode: plaintext -->**耐心是你唯一需要的，Jimmy。**

> 2024年9月23日至9月24日的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**222** 个频道和 **2572** 条消息）。预计节省阅读时间（以每分钟 200 字计）：**294 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在传闻明天发布的 Llama 3 和 Claude 3.5 更新之前：

今天我们看到了 [Gemini Pro 的大幅降价](https://x.com/OfficialLoganK/status/1838611055217385646)，使 Gemini 的定价与我们在本通讯中一直记录的 [新 $/intelligence 前沿](https://x.com/Smol_AI/status/1838663719536201790/photo/1) 保持一致。


![image.png](https://assets.buttondown.email/images/f74bb222-114d-4c7b-844f-4ac7622c4ac2.png?w=960&fit=max)


但头条新闻可能是 [ChatGPT Advanced Voice Mode](https://twitter.com/OpenAI/status/1838642444365369814)，公司高层（如 [Mira](https://x.com/miramurati/status/1838642696111689788)！）宣布其“本周推出”，但似乎美国的大多数人到今天结束时已经获得了访问权限。共有 5 种新语音，并改进了 [口音](https://x.com/chatgpt21/status/1838686328936108294)/语言支持。是的，经过一番努力，[它仍然可以唱歌](https://x.com/swyx/status/1838751593417839089)！


![image.png](https://assets.buttondown.email/images/e90fba09-ca40-47ef-b022-fe908c845338.png?w=960&fit=max)



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

**AI 模型开发与发布**

- **OpenAI 的 o1-preview 模型**：[@omarsar0](https://twitter.com/omarsar0/status/1838353480672563581) 分享了关于 o1-preview 在规划任务中表现的见解，指出其虽有进步，但在长问题和不可解实例上缺乏鲁棒性。该模型在 Randomized Mystery Blocksworld 上达到了 52.8% 的准确率，显著优于其他 LLM。

- **Anthropic 传闻中的新模型**：[@bindureddy](https://twitter.com/bindureddy/status/1838310463194685663) 和 [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838320450927047112) 提到了 Anthropic 即将发布新模型的传闻，引发了 AI 社区的热议。

- **Qwen 2.5 发布**：[@_philschmid](https://twitter.com/_philschmid/status/1838122129792839918) 重点介绍了 Qwen 2.5 的发布，其中 7B 模型在各项基准测试中与 OpenAI 的 GPT-4 0613 旗鼓相当。该模型提供 1.5B、7B 和 32B（即将推出）版本，支持高达 128K tokens。

**AI 研究与基准测试**

- **PlanBench 评估**：[@omarsar0](https://twitter.com/omarsar0/status/1838353480672563581) 讨论了一篇在 PlanBench 上评估 o1-preview 的论文，将其与 LLM 和经典规划器进行了对比。研究揭示了 o1-preview 在规划任务中的优势，同时也指出了其局限性。

- **多语言 MMLU 数据集**：[@_philschmid](https://twitter.com/_philschmid/status/1838230108072476951) 宣布 OpenAI 在 Hugging Face 上发布了多语言大规模多任务语言理解（MMMLU）数据集，涵盖 14 种语言和 57 个类别。

- **RAG 研究标准化**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838259514237464893) 提到了 RAGLAB，这是一个用于标准化检索增强生成（RAG）研究的框架，允许在 10 个基准测试中对 6 种 RAG 算法进行公平比较。

**AI 应用与工具**

- **PDF2Audio**：[@_akhaliq](https://twitter.com/_akhaliq/status/1838219563705532750) 分享了一个将 PDF 转换为音频播客、讲座和摘要的工具。

- **开源 AI 入门套件**：[@svpino](https://twitter.com/svpino/status/1838186602885177835) 介绍了一个自托管的 AI 入门套件，包含低代码开发、本地模型运行、向量存储和 PostgreSQL 等组件。

- **Moshi 语音 AI 助手**：[@ylecun](https://twitter.com/ylecun/status/1838327979203588100) 宣布开源 Moshi，这是来自 Kyutai 的语音 AI 助手。

**AI 行业与商业**

- **Scale AI 动态**：[@alexandr_wang](https://twitter.com/alexandr_wang/status/1838272073652932642) 报告了 Scale AI 的增长情况，其 ARR 提前达到近 10 亿美元，同比增长 4 倍。

- **Together Enterprise Platform**：[@togethercompute](https://twitter.com/togethercompute/status/1838287277010883001) 推出了用于集中化生成式 AI 流程管理的平台，提供快 2-3 倍的推理速度，并可降低高达 50% 的运营成本。

**AI 伦理与社会影响**

- **Sam Altman 的博客文章**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1838287528614318353) 分享了 Sam Altman 博客文章《智能时代》（The Intelligence Age）的见解，讨论了 AI 对人类能力和社会的潜在影响。

- **AI 监管讨论**：[@togelius](https://twitter.com/togelius/status/1838335777446691266) 对拟议的 AI 监管法案表示担忧，认为这些法案可能会阻碍开源开发并将权力集中在私人公司手中。

**梗与幽默**

- [@agihippo](https://twitter.com/agihippo/status/1838220613221306770) 开玩笑说要将订三明治的 YAML 配置标准化，凸显了技术概念在日常生活中的普及。

- [@Teknium1](https://twitter.com/Teknium1/status/1838315311545880929) 幽默地评论了 o1 无法重写代码的问题，调侃了该模型的局限性。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Qwen 2.5：本地 LLM 性能的新基准**

- **Qwen2.5 Bug 与问题 + 修复，Colab 微调笔记本** ([评分: 85, 评论: 15](https://reddit.com//r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/))：该帖子强调了 **Qwen 2.5 模型** 中的 **关键 Bug**，包括错误的 **EOS tokens** 和可能导致 **NaN 梯度** 的 **聊天模板问题**。作者已将 **修复后的模型** 和 **4-bit 量化版本** 上传至 [Unsloth 的 Hugging Face 页面](https://huggingface.co/unsloth)，并提供了使用 [Unsloth](https://github.com/unslothai/unsloth) 微调 **Qwen 2.5 模型**（基础版和对话版）的 **Kaggle 和 Colab 笔记本**。Unsloth 在微调期间可提供 **2 倍的加速** 并减少 **70% 的 VRAM** 占用。

- **[Qwen 2.5 72B 现在可以在 HuggingChat 上免费使用了！](https://huggingface.co/chat/models/Qwen/Qwen2.5-72B-Instruct)** ([Score: 196, Comments: 36](https://reddit.com//r/LocalLLaMA/comments/1fniqym/qwen_25_72b_is_now_available_for_free_on/)): **Qwen 2.5 72B**，一个大型语言模型，现在可以在 **HuggingChat** 上免费访问。该模型由 **Alibaba Cloud** 开发，拥有 **720 亿参数**，是 Qwen（通义千问）系列的一部分，提供包括英语、中文和代码生成在内的多种语言能力。
  - **Qwen 2.5 72B** 现在已在 **HuggingChat** 上线，具有 **32k context window**、改进的 role-playing 能力和结构化数据处理能力。开发者正在寻求关于 tool use 的反馈和资源，以便可能与其 [tools feature](https://huggingface.co/chat/tools) 进行集成。
  - 用户讨论了用 **Mistral Small** 等替代方案替换过时的 **Mixtral models**，后者的性能与 **Llama 3.1 70B** 相当。HuggingChat 提供了慷慨的使用限制，仅有每分钟速率限制，没有每日上限。
  - 一些用户注意到该模型相比更小版本性能有所提升，而另一些用户则指出一个有趣的怪癖：它声称自己是由 **Anthropic** 开发的，而不是承认其作为 Qwen 的真实身份。


- **Qwen 是如何做到的？** ([Score: 235, Comments: 127](https://reddit.com//r/LocalLLaMA/comments/1fnhm67/how_did_qwen_do_it/)): **Qwen 2.5** 模型因其令人印象深刻的性能而获得积极反馈，其中 **32B** 模型的表现与 **70B** 模型相似。这引发了关于当较小模型能达到相当结果时，运行较大模型的效率问题，这可能使 **local LLMs** 更具吸引力。帖子作者询问了 Qwen 成功背后的因素，推测了可能的原因，如改进的 **data quality**、**extended training periods** 或模型开发中的其他 **advancements**。
  - **Qwen2.5** 模型在高达 **18 trillion tokens** 的高质量数据上进行了训练，其中 **32B** 模型的表现与较旧的 **70B** 模型相似。**Apache 2.0 license** 适用于除具有商业价值的 **3B 和 72B** 版本外的大多数模型。
  - 用户报告称，**Qwen2.5 72B** 在大多数任务中优于 **Mistral Large** 和 **Cohere Command R+**，但故事写作除外。**32B** 模型已经为一些用户取代了 **Hermes 3 Llama 3.1 70B**，以更快的性能提供相似或更好的结果。
  - 正如 [Hugging Face 讨论帖](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/discussions/1) 中所讨论的，人们对 **Qwen2.5** 模型缺乏文化知识表示担忧。一些用户认为这种权衡对于专业任务是可以接受的，而另一些人则认为基础知识对于一个全面的 LLM 是必要的。


**主题 2：LLM 效率和量化方面的进展**


- **来自 NVIDIA 的新 Llama-3.1-Nemotron-51B 指令模型** ([Score: 205, Comments: 47](https://reddit.com//r/LocalLLaMA/comments/1fnp2kt/new_llama31nemotron51b_instruct_model_from_nvidia/)): NVIDIA 发布了 **Llama-3.1-Nemotron-51B-instruct**，这是一个拥有 **51.5B parameter** 的 LLM，通过 block-wise distillation 以及针对单个 **H100-80GB GPU** 的优化，从 **Llama-3.1-70B-instruct** 衍生而来。该模型使用来自 **FineWeb**、**Buzz-V1.2** 和 **Dolma** 数据集的 **40 billion tokens** 进行了知识蒸馏，专注于英语单轮和多轮对话用例，可在 [Huggingface](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct) 上获取，仓库大小为 **103.4GB**。
  - 用户对 **width-pruned Qwen 2.5 32B** 和 **Qwen 70B** 模型表示期待。**Qwen 14B** 模型实现了 **~80 的 MMLU 分数**，与大学四年级水平相当，详见 [Qwen blog](http://qwenlm.github.io/blog/qwen2.5/)。
  - NVIDIA 还开发了该模型的 **40B 变体**，在精度损失适中的情况下，比父模型实现了 **3.2 倍的速度提升**。其架构类似于 **DeciLM**，表明 NVIDIA 可能集成了 **Deci 的 AutoNAC** 技术。
  - 该模型的 context size 尚不明确，配置信息存在冲突。`max_position_embeddings` 设置为 **131,072**，但 RoPE scaling 设置中的 `original_max_position_embeddings` 为 **8,192**。

- **以自定义浮点格式运行 LLM（近乎无损的 FP6）** ([Score: 54, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1fo5bbk/running_llms_at_custom_floatingpoints/)): 该帖子讨论了用于 **LLM 运行时量化** 的 **自定义浮点格式** 实现，允许将 **FP16 模型** 直接加载到 **FP4, FP5, FP6 和 FP7** 中，且准确率损失和吞吐量惩罚极小。作者解释了其方案的技术细节，包括 **位级预打包 (bit-level pre-packing)** 和具有 **并行反量化** 功能的 **SIMT 高效 GPU 运行时**，这使得即使在不规则位宽下也能获得极具竞争力的性能。基准测试显示，**FP5 和 FP7** 在 **GMS8K** 上取得了与 **FP8** 相似的结果，而 **FP6** 甚至超过了 **BF16 量化**，这促使作者建议将 **FP6** 作为平衡内存和准确率权衡的潜在标准。
  - 讨论了用于运行时量化的 **自定义浮点格式**，用户注意到其相比于 **exl2 6bpw** 和 **GPTQ** 等分组量化可能具有的 **计算效率** 优势。**5bpw** 格式被强调为某些模型和规模的有意义权衡。
  - 有人对 **GMS8K** 基准测试结果的 **统计显著性** 提出了担忧，建议需要更全面的评估。作者承认了这一点，并提到计划运行 **MMLU-Pro** 以及可能的困惑度 (perplexity)/KL 散度测试。
  - 用户询问了如何将模型转换为 **FP6** 格式，并提供了使用命令行界面的说明。作者指出，目前尚无法导出这些格式的模型，但如果需求增加，可能会集成到 **llm-compressor** 中。


**主题 3. 创意应用中的 AI：游戏与音乐**

- **OpenMusic：出色的开源文本生成音乐项目！** ([Score: 59, Comments: 6](https://reddit.com//r/LocalLLaMA/comments/1fo19nn/openmusic_awesome_opensource_texttomusic/)): **OpenMusic** 是一个开源的 **文本生成音乐 (text-to-music generation)** 项目，可在 [Hugging Face](https://huggingface.co/spaces/jadechoghari/OpenMusic) 上使用。该项目也有一个 [GitHub 仓库](https://github.com/ivcylc/qa-mdt)，允许用户通过文本提示生成音乐。

- **[我正在尝试将小型 LLM 用于《天际》+ AI 配置。Qwen 的推理速度让我感到惊讶。](https://www.reddit.com/gallery/1fo5bt3)** ([Score: 87, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1fo5bt3/im_experimenting_with_small_llms_for_a_skyrim_ai/)): 作者正在尝试将 **小型语言模型** 用于 **《天际》(Skyrim) + AI 配置**，并对 **Qwen 的推理速度** 表示惊讶。虽然没有提供具体的性能指标，但帖子表明，在这一与游戏相关的 AI 应用中，Qwen 的速度在测试的其他模型中脱颖而出。
  - 像 **Mantella** 和 **AI Follower Framework (AIFF)** 这样的 **Skyrim AI 模组** 能够使用 **LLM** 实现 NPC 交互。AIFF 提供更多功能但仅限于随从，而 Mantella 允许与任何 NPC 对话。
  - 用户正在为《天际》尝试各种 **LLM**，包括 **Qwen 2.5 7B**、**Llama 3.1 8B** 和 **Gemma 9B**。推荐使用 **角色扮演微调模型 (Roleplay-tuned models)** 以获得更可信的 NPC 交互。
  - 作者使用的是一台配备 **RTX 3080 mobile 8GB GPU** 的 **MSI GP66 11UH-032** 游戏笔记本电脑，目标是在少于 **6GB 显存 (VRAM)** 的情况下运行 LLM。**量化后的 7b-8b GGUF 模型** 表现出了出色的性能。


**主题 4. 新的 AI 数据集与研究论文**

- **OpenAI 发布开源数据集！** ([Score: 235, Comments: 51](https://reddit.com//r/LocalLLaMA/comments/1fno6d4/open_dataset_release_by_openai/)): OpenAI 在 **Hugging Face** 上发布了 **多语言大规模多任务语言理解 (MMMLU)** 数据集。该数据集现已在 [https://huggingface.co/datasets/openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU) 公开发布，为研究人员和开发人员提供了用于多语言理解任务的新资源。
  - 用户对 **OpenAI** 的动机表示怀疑，一些人认为该数据集可能 **“被投毒” (poisoned)** 或旨在偏向他们的模型。在利用 OpenAI 的输出进行训练时，**GPTslop 泛滥 (GPTslop epidemic)** 被列为需要谨慎的原因。
  - 翻译 **MMLU** 的选择受到了质疑，因为众所周知它存在 **有问题的题目** 和 **无效的答案选项**。一些人建议 **MMLU-Pro** 会是更好的选择，因为许多模型在 MMLU 上的得分已经达到了 **90%** 左右。
  - 尽管存在怀疑，用户仍承认 **开源基准测试** 对于可复现性和模型比较的价值。该数据集的规模（**19.4 万测试集**）被认为对于计算单个分数来说可能过大。

- **[Google 发布了新论文：Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)** ([Score: 229, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1fo6bdg/google_has_released_a_new_paper_training_language/))：Google 研究人员提出了一种名为 **Self-Correction via Reinforcement Learning (SCRL)** 的新方法，旨在改进语言模型的输出。该方法利用 **Reinforcement Learning** 训练模型对其初始输出进行 **Self-Correct**，从而在包括 **问答**、**摘要** 和 **推理** 在内的各种任务中提升了 **性能**。SCRL 展现出比标准微调显著的改进，在某些基准测试中增益高达 **11.8%**。
  - **Self-Correction via Reinforcement Learning (SCRL)** 方法的有效性受到了质疑，用户讨论了如何确保模型是 **真正的自我修正**，而非刻意生成错误。论文中关于泛化自我修正能力的关注被视为一个关键见解。
  - 用户对论文的方法论进行了辩论，指出 Prompt 并没有明确说明解法是错误的。一些人指出 **Qwen 72B** 模型可以 zero-shot 解决所有 8 道数学题，这引发了关于数据泄露和需要新型评估集的疑问。
  - 讨论涉及了论文的理论重点与实际应用，强调研究论文通常是测试特定理论而非生产最终产品。通过一个关于数字加法的 **ELI5 类比** 解释了泛化改进步骤的概念。

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型进展与发布**

- **OpenAI 的 GPT-4 Turbo (o1) 模型**：多篇帖子讨论了 OpenAI 新 o1 模型（尤其是 o1-mini）令人印象深刻的能力。用户报告称其在 [复杂数学问题解决等任务中有显著改进](https://www.reddit.com/r/singularity/comments/1fnxu28/o1mini_is_so_insane/)，一位用户形容在某些应用中它就像“在使用真正的魔法”。

- **Anthropic 的新模型发布**：[Anthropic 预计将发布一款新的 AI 模型](https://www.reddit.com/r/singularity/comments/1fnussc/anthropic_will_likely_drop_a_new_model_tomorrow/)，在 AI 社区引起了兴奋。

- **AI 推理成本降低**：[OpenAI 的 Dane Vahey 报告称，每百万 Token 的成本在 18 个月内从 36 美元降至 0.25 美元](https://www.reddit.com/r/singularity/comments/1fo2nj8/dane_vahey_of_openai_says_the_cost_per_million/)，这代表了 AI 运营成本的大幅下降。

**AI 研究与开发**

- **多智能体 (Multi-agent) AI 研究**：[Google DeepMind 和 OpenAI 都在组建专注于多智能体通用人工智能 (AGI) 研究的团队](https://www.reddit.com/r/singularity/comments/1fnp3pj/google_deepmind_is_now_hiring_for_a_team_in/)，表明了对该领域日益增长的兴趣。

- **图像和视频生成中的 AI**：AI 驱动的图像和视频处理进展受到关注，包括 [CogVideoX-I2V 的工作流](https://www.reddit.com/r/StableDiffusion/comments/1fnn08o/cogvideoxi2v_workflow_for_lazy_people/) 以及 [在视频生成中同时控制多个主体的演示](https://www.reddit.com/r/singularity/comments/1fnu6qi/three_subjects_are_controlled_simultaneously_with/)。

**行业与市场动态**

- **Anthropic 潜在的估值增长**：[据报道，Anthropic 正在与投资者洽谈，计划以 300-400 亿美元的估值筹集资金](https://www.reddit.com/r/singularity/comments/1fnvmst/openai_rival_anthropic_has_started_talking_to/)，这可能会使其之前的估值翻倍。

- **政治层面对 AI 的参与**：[美国副总统 Kamala Harris 在一场筹款演讲中承诺将加大对 AI 的投资](https://www.reddit.com/r/singularity/comments/1fnjo2f/kamala_harris_vows_to_boost_ai_and_crypto/)，表明政治层面对于 AI 发展的关注度日益提高。

**关于 AI 进展的观点**

- **AI 能力的飞速进步**：多篇帖子反映了 [AI 能力的快速进步](https://www.reddit.com/r/singularity/comments/1fnlkof/how_fast_things_change/)，许多以前被认为远未解决的任务现在已经可以实现。

- **未来 AI 预测**：[Yann LeCun 预测，匹配或超越人类智能的 AI 即将到来](https://www.reddit.com/r/singularity/comments/1fnuysf/yann_lecun_says_we_will_soon_have_ai_that_matches/)，同时在一年或两年内，智能眼镜中的 AI 助手将能够翻译数百种语言。


---

# AI Discord Recap

> 由 O1-preview 提供的摘要的摘要总结

**主题 1. AI 模型升级：新发布与重大更新**

- [**Mistral Small 模型发布，拥有 220 亿参数**](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)：全新的 **Mistral Small 模型** 已上线，拥有 **220 亿参数**，旨在提升各项任务中的 AI 性能。用户可以通过 [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) 进行探索。
- [**OpenAI o1 模型引发关注，尽管图表未标注**](https://x.com/hughbzhang/status/1838288923656941860)：OpenAI 发布了 **o1 系列模型**，但其 Scaling Law 图表缺少 x 轴标签，促使用户尝试使用 **o1-mini API** 重构数据。讨论集中在计算量是否仅涉及数万个 Token。
- [**Gemini 模型性能提升并降价**](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/)：**Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002** 获得更新，**速率限制 (Rate Limits) 提升 2 倍以上**，且**价格下调 50%**。开发者对这些变化感到兴奋，称其为“开发者的好日子”。

**主题 2. 语音功能在争议中推出**

- [**OpenAI 高级语音功能支持 50 多种语言**](https://x.com/OpenAI/status/1838642444365369814)：**高级语音 (Advanced Voice)** 功能正向 **Plus** 和 **Team** 用户推出，新增了自定义指令 (Custom Instructions)、记忆功能 (Memory) 以及 **5 种新语音**，并改进了口音。用户现在可以用 **50 多种语言** 进行表达。
- [**欧洲用户因无法使用语音功能感到沮丧**](https://x.com/OpenAI/status/1838642453391511892)：尽管功能已推出，但欧洲用户感到失望，因为 **高级语音功能在多个欧洲国家尚未上线**。许多人表示该功能“不如早期的演示”。
- **用户讨论语音助手的审查与限制**：讨论指出，OpenAI 对安全性的关注导致语音助手功能受限。用户抱怨其缺乏像 **Character.ai** 等角色扮演 AI 产品那样的动态性。

**主题 3. 开发者致力于 AI 集成与优化**

- [**OpenRouter 集成至 Cursor 并提供演示应用**](https://x.com/OpenRouterAI/status/1838627801072562232)：OpenRouter 现在可以在 **[Cursor](https://x.com/OpenRouterAI/status/1838627801072562232)** 中与包括 Anthropic 在内的所有模型无缝协作。他们还在 GitHub 上发布了 [演示应用](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling) 以启动开发。
- **Aider 安装困扰导致用户卸载**：用户在安装 **Aider** 时面临挑战，导致多次尝试使用 **pipx** 重新安装仍未解决问题。一些人选择回退到旧版本以恢复功能。
- **GPU MODE 更名引发复杂反应**：原名为 **CUDA MODE** 的社区更名为 **GPU MODE**，旨在扩大关注范围。成员们反应不一，出现了诸如“*Gigachad Processing*”之类的幽默建议，并对更名进行了辩论。

**主题 4. AI 推理与可靠性备受关注**

- [**LLM 无法规划？OpenAI o1 评估报告**](https://arxiv.org/abs/2409.13373)：一份新的研究简报批判性地评估了 **OpenAI o1** 模型的规划能力，认为尽管它被宣传为**大型推理模型 (Large Reasoning Model)**，但实际上“无法规划”。
- **高 Temperature 输出导致幻觉频发**：用户报告称，将 Temperature 提高到 **1.25** 以上会导致模型产生幻觉，从而质疑输出的可靠性。指示模型不要产生幻觉虽有帮助，但不能完全解决问题。
- **JSON 格式化问题困扰开发者**：API 用户在处理 JSON 格式输出时遇到困难，经常收到不完整或错误的响应（如仅有一个 '{'）。虽然建议使用定义更明确的 Prompt 结构，但问题依然存在。

**主题 5. 协作努力与工具增强 AI 开发**

- [**DSPy 2.5.0 发布，解决问题的速度比说出“Chain-of-Thought”还要快**](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)：**DSPy 2.5.0** 的发布旨在迅速解决 **50-100 个问题**，用户对新功能和即将推出的入门 Notebook 充满热情。
- [**GitHub 仓库涌现大量 AI 工具**](https://github.com/jack-tol/youtube-to-audio)：诸如 **YouTube-to-Audio** 之类的新工具可以轻松从视频中提取音频，而 **LitServe** 等框架则简化了使用 FastAPI 提供和扩展 LLM 服务的过程。
- **社区联合进行微调与模型训练**：成员们分享了微调 **Vit_B16** 和 **Llama3.1** 等模型的经验，强调了高质量数据的重要性。与模型开发者的合作有助于解决 **Qwen 2.5** 中的 Bug 等问题。

---

*注：所有链接和细节均基于各 Discord 频道的讨论，反映了最新的更新和社区情绪。*


---

# 第 1 部分：Discord 高层摘要

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cursor 集成了 OpenRouter！**：OpenRouter 现在可以在 [Cursor](https://x.com/OpenRouterAI/status/1838627801072562232) 中无缝运行，支持包括 Anthropic 在内的所有模型。*感谢 @cursor_ai 修复了此问题！* 🍾
   - 此次集成通过简化操作和扩大模型可访问性提升了用户体验。
- **Gemini 模型升级，性能更佳！**：两个更新的模型 **Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002** 现已上线，具有更低的价格和更高的性能指标。
   - 这些模型针对效率进行了优化，具有更快的输出和更高的 Rate Limits，并将于 2024 年 10 月 8 日前自动更新面向用户的别名。
- **OpenRouter 推出演示应用以快速上手**：OpenRouter 团队宣布在 [GitHub](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling) 上提供基础演示应用，帮助开发者快速启动项目。
   - 这些演示包含一个简单的 'tool calling' 功能，使用户更容易从零开始创建应用程序。
- **关于 Middle-Out Transform 影响的讨论**：用户对默认禁用 Middle-Out Transform 表示担忧，理由是对工作流和基础设施产生了负面影响。
   - 社区强调需要就模型变更进行更清晰的沟通和更新，以减轻干扰。
- **关于 Token 定价结构的见解**：相关讨论强调了不同模型之间 Token 定价的差异，指出 OpenRouter 利用从上游返回的原生 Token 进行成本计算。
   - 用户注意到 GPT-4o 和 Qwen 等模型之间 Tokenizers 的差异会显著影响 Token 计数和价格估算。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Mistral Small 模型发布**：新的 [Mistral Small 模型](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) 已上线，拥有 **220 亿参数**，旨在提升各种任务中的 AI 性能。
   - 用户可以通过 [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) 进一步探索该模型。
- **Gradio 5 设定性能标准**：**Gradio 5 (Beta)** 正式发布，引入了重大的易用性改进和服务器端渲染，以实现更快的应用加载。
   - 在公开发布之前，官方鼓励用户提供反馈，旨在根据社区见解完善功能。
- **FinePersonas 为合成数据增加丰富性**：最新的 [FinePersonas v0.1](https://x.com/reach_vb/status/1836882281434165629) 提供 **2100 万个 Persona**，增强了针对各种应用的合成数据生成。
   - 该数据集旨在提供针对特定 Persona 需求量身定制的真实查询生成，彻底改变大规模数据项目。
- **Hugging Face Token 问题频发**：多位用户报告了 Hugging Face Token 无效的问题，引发了关于潜在 Rate Limit 问题和重新安装 huggingface-hub 包的讨论。
   - 尽管进行了故障排除，许多人仍继续遇到 Token 验证失败的情况。
- **OpenAI 的单词生成数量令人震惊**：据报道，OpenAI 每天生成约 **1000 亿个单词**，有用户质疑 Hugging Face 是否能使用自己的模型接近这一指标。
   - 这一讨论突显了 AI 领域不同实体之间在文本生成能力上的显著差异。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV 架构提供独特见解**：社区深入剖析了 **RWKV 架构**，特别是其相较于**卷积 (convolutions)** 的效率，并强调需要更简单的解释来促进采用。
   - 熟悉 **GLA** 至关重要，因为参与者主张对围绕 RWKV 的复杂性进行更清晰的分解。
- **推出 YouTube-to-Audio 工具**：一位用户发布了 **youtube-to-audio**，这是一个命令行工具，可以从 YouTube 提取 **MP3** 和 **WAV** 等多种格式的音频，提升了用户体验。
   - 该工具还支持播放列表下载和自定义文件名，定位为现有解决方案的无广告替代方案。
- **动态评估 (Dynamic Evaluation) 引发辩论**：ML 中的动态评估提议在测试集上微调模型，引发了对其外部有效性以及与经典实践一致性的担忧。
   - 尽管有效，成员们强调了训练和测试数据集之间相似分布的关键需求。
- **muP 实现得到澄清**：社区致力于简化神经网络的 **muP** (Maximal Update Parameterization) 概念，这对于增加社区参与度至关重要。
   - 推动更清晰的实现以及理论见解，旨在提高开发者的集成便利性。
- **OpenAI 的 LRM 显示出规划局限性**：成员们评估了 OpenAI 的 **o1 (Strawberry)** 作为大型推理模型 (LRM) 的表现，并对其有效性进行了辩论，特别是在特定测试条件下。
   - 对其规划能力的担忧浮现，报告指出**推理时计算 (inference time compute)** 占比已跃升至 **50%** 以上。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 模型迎来重大更新**：**Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002** 的推出带来了 **50% 的降价**和增强的速率限制，尽管编码基准测试保持平稳，但仍给开发者留下了深刻印象。
   - 自 10 月 1 日起，对于 **128,000 tokens** 以下的输入，输入成本从 **$3.50** 降至 **$1.25/million tokens**。
- **RoRF 开源以增强性能**：**Routing on Random Forest (RoRF)** 已开源发布，提供 **12 个预训练模型路由 (model routers)**，显著提升了 MMLU 性能。
   - 此次发布有望推动各种应用中的模型路由技术。
- **对新 Claude 模型的期待升温**：关于即将发布的 **Claude 模型**（如 **Haiku 3.5** 或 **Opus**）的猜测不断，用户对自上次更新以来的等待表示沮丧。
   - 这种延迟让人们对未来几周内的及时发布寄予厚望。
- **Aider 安装不便**：用户在安装 **Aider** 时面临挑战，在功能失效后通常需要使用 **pipx** 重新安装，尽管他们努力尝试修复配置。
   - 在用户应对这些复杂情况时，回退到旧版本是建议的解决方案之一。
- **Prompt Caching 功能正在讨论中**：针对 **Anthropic API** 和 **OpenRouter** 的 Prompt Caching 配置 **AIDER_CACHE_KEEPALIVE_PINGS** 是一个热门话题，用户正在寻求实现方面的澄清。
   - 分享了 **Prompt Caching** [文档](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)链接以方便理解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Advanced Voice 功能全面推出**：**Advanced Voice** 功能现已向 **Plus** 和 **Team** 用户开放，通过 [Custom Instructions](https://x.com/OpenAI/status/1838642444365369814) 等新功能增强了 ChatGPT App 内的交互体验。此次更新还引入了**五种新声音**，并显著提升了多语言能力，支持超过 **50 种语言**的表达。
   - 然而，这一功能的推出在欧洲用户中引起了困惑和失望，一些人指出其表现不及早期的演示。
- **JSON 格式化质量受到抨击**：用户正面临 JSON 格式输出的问题，经常获取到如简单的 '{' 这样令人不满意的响应，导致寻求结构化数据的 API 用户感到沮丧。有人建议通过定义更清晰的结构来提高这些输出的质量。
   - 尽管建议使用更明确的 Prompt，但许多人仍遇到性能不佳的问题，限制了他们获得有效 API 响应的能力。
- **明确 Prompt Engineering 以获得更好的输出**：关于 **Prompt Engineering** 的讨论强调了清晰、详细请求的必要性，以最大化模型的性能和输出的相关性。几位成员强调，在 Prompt 中加入具体示例可以极大地提高生成响应的质量。
   - 提到的挑战包括生成多样化的内容，特别是在 **Minecraft** 提问等利基应用中，面临着内容重复的问题。
- **幻觉与响应可靠性担忧**：参与者对模型在 Temperature 设置超过 **1.25** 时产生幻觉的倾向表示担忧，这表明输出缺乏可靠性。成员们的见解表明，指示模型避免幻觉可能会限制无关内容，但并不能完全解决问题。
   - 这种对幻觉的担忧延伸到了 ChatGPT 的各种功能中，促使用户寻求更好的性能优化方法。
- **生成有趣的 Minecraft 问题**：一位用户测试了一个旨在以 JSON 格式生成有趣且引人入胜的 **Minecraft** 问题的 Prompt，但在实现多样化和吸引人的查询方面遇到了障碍。社区的反馈旨在解决重复输出和幻觉的挑战，从而优化这一过程。
   - 这种对问题生成创意性的追求引发了关于改进 Prompt 策略和有效利用 API 能力的讨论。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU MODE 转型**：社区从 **CUDA MODE** 转型为 **GPU MODE**，旨在更广泛地关注 CUDA 之外的各种 GPU 编程框架。
   - 成员们对更名表达了复杂的情绪，并提出了诸如 **Heterogeneous Computing** 甚至像 *Gigachad Processing* 这样幽默的替代名称。
- **分布式系统的训练优化**：讨论强调了在使用 **2x 4090 GPU** 进行训练时的扩展性问题，指出在低带宽条件下 **DDP** 比 **FSDP** 提供更好的性能。
   - 参与者强调了通信带宽对可扩展性的影响，并分享了优化分布式训练工作负载的经验。
- **WebNN 集成前景**：有人建议为 **WebNN** 创建专门的频道，思考其在集成 **WebGPU** 和 **WASM** 中的作用，这在标准化方面可能面临挑战。
   - 针对 **WebNN** 与 NPU API 交互的能力进行了澄清，展示了其在不同硬件设置下的潜力。
- **Luma 招聘性能工程师**：Luma 正在为其 **Dream Machine** 项目积极寻求工程师，提供的职位专注于多模态基础模型（Multimodal Foundation Models）的性能优化。
   - 应聘者被要求在分布式训练和低级 Kernel 优化方面拥有丰富经验，该公司被强调为正在快速增长。
- **GPU 上的数据处理挑战**：成员们指出， GPU 上有效的数据传输严重依赖于**延迟**和**内存带宽**，并询问了与 CPU 设置的对比情况。
   - 一位参与者对这些因素如何影响 GPU 性能以及在片上系统（SoC）架构中的可用性提出了疑问。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 模型引发关注**：OpenAI 发布了 **o1 系列模型**，并附带了一张展示测试时计算（test-time compute）扩展定律（scaling laws）的图表。尽管 x 轴未标注，但引发了关于利用 o1-mini API 重构数据的讨论。
   - 成员指出，计算可能仅涉及数万个 token，这引发了在没有适当结构的情况下进一步扩展的可行性问题。
- **Anthropic 估值或达 400 亿美元**：报告指出 **Anthropic** 正在讨论融资，这可能将其估值推高至 **300 亿美元** 到 **400 亿美元** 之间，较今年早些时候实际上翻了一番。
   - 这反映了在快速发展的市场中，AI 公司争夺大量资金支持的激烈竞争态势。
- **James Cameron 加入 Stability AI 董事会**：Stability AI 欢迎 **James Cameron** 加入其董事会，旨在利用他的专业知识探索视觉媒体领域的创新。
   - 这一战略举措被视为开发专为创作者量身定制的更全面 AI pipeline 的关键。
- **Gemini 模型增强功能发布**：**Gemini 模型** 的更新包括 **2 倍以上的速率限制（rate limits）提升** 以及 Gemini 1.5 Pro **50% 的降价**，同时还为开发者提供了新功能。
   - 修订版还引入了可选的过滤器以提高安全性和可靠性，从而更好地控制模型设置。
- **Scale AI 财务增长洞察**：Scale AI 报告称，尽管毛利率较低，但上半年销售额增长了近四倍，表明在 AI 服务需求不断增长的背景下增长强劲。
   - 随着行业格局继续向 AI 驱动的解决方案转变，这一财务激增使 Scale AI 处于极具吸引力的地位。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **O1 规划能力评估**：最近关于 **O1** 的一篇 [研究简报](https://arxiv.org/abs/2409.13373) 概述了其规划能力，据报道团队成员为了完成该报告熬夜奋战。
   - 研究结果详细阐述了全面的检查，并承诺在公开发布后提供更多见解。
- **World Simulator API 提供低成本访问**：讨论集中在 **World Sim**，强调了用户注册即可获得积分的机会，同时 API 使用成本较低。
   - 频道中普遍鼓励创建账户以利用免费积分。
- **Hermes 和 Monad 表现出“固执”**：有报告称 **Hermes** 和 **Monad** 在交互中变得不那么有效，特别是在打标签（tagging）能力方面。
   - 一个建议是实施存在惩罚（presence penalty），而其他人则注意到基于托管环境的差异。
- **Gemini 1.5 引发关注**：人们对 9 月份发布的 **Gemini 1.5** 升级充满期待，同时 **GPT-4o** 也进行了小规模推广。
   - 成员们对即将举行的 **Meta Connect** 活动可能出现的突破表示期待。
- **DisTrO 在低带宽下表现高效**：初步研究结果表明，**DisTrO** 在非对称带宽和异构 GPU 环境中运行有效，增强了资源管理。
   - 这使得 **DisTrO** 成为次优网络条件下资源分配的可行选择。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 模型问题已解决**：成员们讨论了 **Qwen 2.5** 模型的问题，报告了崩溃和 Bug，但分享了改进模板和修改训练方法等解决方案。
   - 与 Qwen 团队的合作在解决其中一些问题上取得了进展，从而提高了模型的稳定性。
- **Unsloth Trainer 的内存使用得到优化**：一位用户在初始化 **UnslothTrainer** 时遇到了**内存问题**，并建议通过减少数据集映射进程来解决。
   - 他们的后续反馈表明，减少进程数取得了成功，强调了为了获得更好内存性能进行平衡的重要性。
- **分享模型微调见解**：微调 **Vit_B16** 模型的经验分享强调，**高质量数据**比单纯的数据量更能有效提升结果。
   - 该用户在获得显著的准确率后，计划通过增加高质量图像来进一步增强其模型。
- **Llama3.1 的内存问题已处理**：一位用户在加载 **4-bit 量化 Llama3.1** 时遇到了 **out of memory (OOM)** 错误，当 PyTorch 使用了 **14.75GB** 时，**20GB** 的分配失败了。
   - 社区成员建议调整模型配置，作为解决这些 OOM 问题的排查步骤。
- **探索强化学习的改进**：讨论了 OpenAI 如何应用 **RLHF (Reinforcement Learning from Human Feedback)**，根据用户交互来升级其模型。
   - 参与者指出了利用先前的对话来指导模型改进的挑战，强调了训练方法中缺乏结构化反馈。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **对 Anthropic 新模型发布的期待升温**：消息源确认，**Anthropic** 的**重大 AI 模型升级**预计很快发布，完整细节将在禁令解除后公布。
   - 成员们正热烈讨论这次升级可能对 AI 领域的开发者产生的影响。
- **Perplexity Pro 功能引发用户好奇**：用户在讨论 **Perplexity Pro** 账户的限制时产生了困惑，特别是关于每日搜索限制的问题。
   - 虽然一些用户质疑 Pro 账户的价值，但其他人承认了更个性化搜索体验带来的好处。
- **Merlin 扩展受到关注**：关于 **Merlin 扩展**的讨论突出了其直接与各种 **LLMs** 聊天的能力，提供了无限的模型访问权限。
   - 用户赞赏无限查询功能，但对模型设置中与 **HARPA AI** 相比缺乏透明度表示担忧。
- **引用输出的不一致导致困扰**：成员们对通过 API 获取的**引用输出不一致**表示沮丧，输出在 **HTML** 和 **Markdown** 格式之间交替。
   - 据报道，这种不一致性阻碍了自动化工作，使可靠的输出生成变得复杂。
- **AI 在教育中的角色受到审视**：对 AI 如何影响教育的探索揭示了变革性的益处和挑战，并对其影响进行了持续讨论。
   - 成员们分析了 AI 融入教育环境的不同方面以及学习动态的潜在转变。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **在 Air-Gapped 机器上安装 LM Studio**：成员们讨论了在 **Air-Gapped**（物理隔离）机器上安装 **LM Studio** 的可行性，强调虽然安装本身不需要网络，但初始设置和文件传输是必要的。
   - *Air-Gapped 安装需要周密的计划*，特别是需要分别下载安装程序和模型。
- **模型性能遇到瓶颈**：用户报告了模型在接近其 **Token** 限制时的性能问题，指出随着 **Token** 填满，由于 **VRAM** 限制会导致速度变慢。
   - 这引发了关于*管理 Token 限制*以保持最佳性能的建议。
- **LongWriter 模型引发关注**：**LongWriter** 模型因其生成超长文本的能力而受到称赞，并分享了相关资源供感兴趣的成员进一步探索其特性。
   - 鼓励成员查看 [LongWriter 的 GitHub 页面](https://github.com/THUDM/LongWriter) 以了解其用法和功能的见解。
- **对双 GPU 兼容性的担忧**：关于 **LM Studio** 是否支持双 **GPU** 设置的讨论引发了关于混合使用 **RTX 4070 Ti** 与 **RTX 3080** 以及潜在性能收益的询问。
   - 建议集中在尝试此类配置之前评估*兼容性问题*。
- **高昂的 GPU 价格令欧盟买家沮丧**：成员们对欧盟地区较高的 **GPU** 价格表示不满，价格通常在 **$750** 左右，而美国的定价较低。
   - 区域定价问题归因于 **VAT**（增值税）和税收，同时还讨论了欧洲*消费者保护*政策的优势。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 位居编程语言层级榜首**：一位用户将 **Mojo** 排在个人语言层级列表的首位，认为它高于 **C#** 和 **Rust**，这是一个主观但发自内心的决定。
   - 有人呼吁在 **C++** 类别中进行更清晰的划分，特别强调了纯净的 **C** 互操作性的重要性。
- **Rust 面临编译缓慢的困扰**：用户抱怨 **Rust 缓慢的编译时间**，特别是在像 **4 万行代码的游戏**这样的大型项目中，编译过程可能会拖得很长。
   - **Generics**（泛型）被认为是导致这些减速的主要原因，并建议优化 **Windows** 上的文件系统设置。
- **NixOS 在谨慎中引发兴趣**：围绕迁移到 **NixOS** 的讨论称赞了其包管理，但也对其系统的整体复杂性表示担忧。
   - 成员们辩论了将 **Ansible** 作为小型项目更简单工具的潜力，同时探索 **NixOS** 的可复现性优势。
- **讨论中 MLIR 胜过 LLVM**：关于为什么 **MLIR** 可能优于 **LLVM** 的问题集中在并行编译和高级语义处理的改进上。
   - **MLIR** 保留调试信息的能力被视为一个关键优势，特别是在编译器不断演进的情况下。
- **庆祝 Mojo 的进展与未来**：社区庆祝了 **Mojo** 发布两周年，回顾了它的成长，包括 **Mojo SDK** 发布等关键进展。
   - 对该语言未来的热情显而易见，用户们热烈讨论其演进将如何塑造未来几年。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 2.5.0 发布引发热潮**：**DSPy 2.5.0** 的发布旨在迅速解决 **50-100** 个问题，其新功能和即将推出的入门 notebook 激发了广泛的热情。
   - 成员们建议建立公开周会，以便对发布版本提供进一步的反馈。
- **高效的 GROQ API Key 设置**：关于设置 **GROQ_API_KEY** 并执行 `lm = dspy.LM('groq/llama3-8b-8192')` 的用户指南促进了 **Llama 3** 的集成。
   - 该指令简化了在 GROQ 上托管的模型与 **dspy** 库的使用流程。
- **近期论文对 Chain of Thought 进行评估**：一篇论文重点介绍了对超过 **100 项研究**进行的 Chain-of-Thought (**CoT**) 提示词定量元分析，显示其在数学或逻辑任务中的有效性显著增强。
   - 关键发现表明，在 **MMLU** 的符号运算中，直接回答的效果与 CoT 相当，强调了当问题涉及**等号**时对推理的需求。
- **讨论 LLM 使用的自定义适配器**：成员们探讨了创建**自定义适配器**，以便为 **dspy.LM** 的结构化输出指定 `grammar` 等额外参数。
   - 讨论集中在分享经验以及对参数使用更清晰最佳实践的需求。
- **多模态能力的期待持续升温**：DSPy 备受期待的多模态功能将于下周推出，并收到了关于 **Ultravox** 等音频模型兼容性的咨询。
   - 官方回复指出，最初的重点将放在 Vision Language Models (VLMs) 上。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **今日关于 AI 框架和多模态助手的课程**：Berkeley MOOC 的**第 3 讲**（[直播链接](https://www.youtube.com/live/OOdtmCMSOo4)）将由 Chi Wang 讲解 **Agentic AI Frameworks & AutoGen**，并由 Jerry Liu 介绍构建**多模态知识助手**的步骤，于 **PST 时间下午 3:00** 开始。
   - Chi 将讨论 Agentic AI 编程的**核心设计考量**，而 Jerry 将讨论**结构化输出**和**事件驱动工作流**等要素。
- **澄清课程签到困惑**：直播的签到表仅供 Berkeley 学生使用，这在 MOOC 参与者中引起了一些困惑。
   - 下次二维码将附带更清晰的说明以避免误解。
- **探索开源 Embedding 模型**：成员们认为 Jina AI 的 **jina-embeddings-v3** 是领先的开源 Embedding 模型，提供多语言能力并利用了 **Task LoRA**。
   - 该模型增强了**神经搜索应用**的性能，强调了有效索引和相关性的重要性。
- **AutoGen 与 CrewAI 在自定义和速度上的对比**：在多 Agent 协作方面，成员们指出 **AutoGen** 允许更高程度的自定义，而 **CrewAI** 在快速原型开发方面表现出色，但在往返通信方面有所欠缺。
   - AutoGen 中的 **conversable_agent** 支持更复杂的交互，这是 CrewAI 用户认为受限的一个功能。
- **RAG 的搜索/检索技术**：讨论建议专注于**经典 NLP 技术**来增强信息检索，特别是**排序算法**和**语义理解**。
   - 理解这些技术对于改进 RAG 框架内的搜索至关重要，从而实现更好的索引和相关性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Letta AI 脱离隐身模式**：由创始人 [Sarah Wooders](https://x.com/sarahwooders) 和 [Charles Packer](https://x.com/sarahwooders/status/1838261104864346288?s=46) 创立的 [Letta AI](https://x.com/sarahwooders/status/1838261104864346288?s=46) 正式发布，该公司专注于开发有状态的 LLM Agent。他们目前正在旧金山积极招聘并组建团队。
   - 在 [TechCrunch](https://techcrunch.com/2024/09/23/letta-one-of-uc-berkeleys-most-anticipated-ai-startups-has-just-come-out-of-stealth/) 阅读更多关于 Letta 的信息。
- **Gemini 模型增强**：[Gemini 模型](https://x.com/OfficialLoganK/status/1838611055217385646) 迎来了重大更新，包括速率限制（rate limits）翻倍，以及 Gemini 1.5 Pro 降价超过 50%。过滤器已改为选择性开启（opt-in），并发布了更新后的 Flash 8B 实验性模型。
   - 开发者对这些变化持乐观态度，认为这是开发者的黄金时期，详见 [Google Developers Blog](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/)。
- **语音功能推出**：OpenAI 宣布 [Advanced Voice](https://x.com/openai/status/1838642444365369814?s=46) 正在向 ChatGPT 应用的 Plus 和 Team 用户推出，引入了多项新功能并改进了口音。值得注意的是，它可以表达超过 50 种不同语言的短语。
   - 然而，正如 [OpenAI](https://x.com/OpenAI/status/1838642453391511892) 的公告所强调的，该功能尚未在几个欧洲国家开放。
- **客户服务 Agent 实验**：关于管理 Agent 模拟多轮对话挑战的讨论揭示了维持有效用户交互的重要见解。建议包括实施阶段标记（stage markers）和设定清晰的对话终止指南。
   - 用户正在探索各种方法，将强化学习（reinforcement learning）集成到对话管理中，以提升客户 Agent 的体验。
- **HuggingChat macOS 应用介绍**：新发布的 [HuggingChat](https://x.com/cyrilzakka/status/1838618605648490974?s=61) macOS 应用提供了开源 LLM 的原生集成，具备 Markdown 支持和网页浏览等功能。这标志着面向直接桌面使用的用户友好型 AI 工具迈出了重要一步。
   - 该应用展示了增强 AI 驱动应用程序的可访问性和功能的趋势。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **新人被 Cohere AI 吸引**：机械工程专业的学生 **Nav** 等成员表现出学习 **Cohere** 和 AI 的兴趣，并寻求博客或视频等资源，随后有人分享了关于 [Aya Research](https://cohere.com/research/aya) 计划的链接，该计划旨在推进多语言 AI。
   - 该计划旨在提高可访问性，使跨语言的 AI 应用得到更广泛的理解。
- **社区聊天缓解就业焦虑**：**Milansarapa** 表达了对工作不稳定性的担忧，引发了社区关于已拿到合同的保证，强化了支持的重要性。
   - *“你已经拿到合同了”* 变成了一句令人安心的口号，凸显了社区参与的益处。
- **Cohere Toolkit 获得显著功能更新**：**Cohere Toolkit** 的最新更新修复了各种后端/UI 问题，并引入了置顶聊天以及对 **parquet** 和 **tsv** 文件的支持，并提供了 [YouTube 演示视频](https://youtu.be/gdJ0abx9mvo)。
   - 这些增强功能显著提升了用户体验，并展示了团队对社区反馈的重视。
- **Reranker 面临多语言挑战**：有报告称，**多语言 Reranker** 在 **波兰语** 等语言中的相关性得分较低，过滤掉了有用数据，导致其失效。
   - *“相关性得分太低以至于被过滤掉”* 表明在 Reranker 过程中需要更好地处理多样化语言。
- **探索 Chain of Thought (COT)**：**Milansarapa** 询问了 **Chain of Thought** (COT) 机制，引发了关于它如何增强某些任务性能的讨论。
   - 最终，COT 被认为是一种有价值的问题解决方法，尽管其应用因案例而异。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **James Cameron 加入 Stability AI 董事会**：首席执行官 Prem Akkaraju 宣布，传奇电影制作人 **James Cameron** 已加入 **Stability AI 董事会**。这一加入支持了 Stability AI 利用**前沿技术**变革视觉媒体的使命。
   - 以《终结者》和《阿凡达》闻名的 Cameron 旨在通过创新的 AI 视觉媒体解决方案彻底改变叙事方式。
- **寻求 FNAF Loras 合作者**：一名成员正在寻找 **FNAF** 粉丝协助为该游戏创建 **Loras**。他们正在寻找合作伙伴来共同完成这个项目。
   - *有人有兴趣合作这个项目吗？*
- **使用 3090 EGPU 提升 SDXL 性能**：一位用户报告称购买了 **3090 EGPU** 以增强其 **SDXL** 运行体验，克服了以往同类产品的失败经历。他们分享了对某些 **Aurus** 游戏盒子的挫败感。
   - *注意到了同类产品的质量问题，从而促成了这一决定。*
- **探索 ControlNet 的功能**：一位用户询问了 **ControlNet**，它可以引导图像生成，特别是针对姿势。仅靠语言描述很难明确规格要求。
   - *有效的引导方法是改进图像输出的关键焦点。*
- **排查 OpenPose 编辑器安装问题**：一位用户报告了 Forge 中 **OpenPose 编辑器**的问题，建议可能需要特定的安装命令。针对在虚拟环境中运行 **pip install basicsr** 提供了协助。
   - *分享了关于安装命令的说明，以便更好地进行集成。*

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **警惕虚假 LlamaParse 网站**：已发布关于冒充 **LlamaIndex** LlamaParse 的**欺诈网站**的警告，提醒用户避开。
   - 合法的 LlamaParse 可以通过 [cloud.llamaindex.ai](https://cloud.llamaindex.ai) 访问，以防止混淆。
- **LitServe 简化了 LLM 的部署**：来自 LightningAI 的 **LitServe** 框架简化了使用 FastAPI 部署和扩展 LLM 的过程，正如在 [LlamaIndex](https://t.co/Xikqk20peW) 的演示中所展示的那样。
   - 该设置可以在本地针对 **Llama 3.1** 托管一个简单的 **RAG** 服务器，为开发者提供高效方案。
- **50 行代码创建 AI 产品经理！**：使用 LlamaIndex 和 ComposioHQ 仅需 50 行代码即可构建一个 **AI 产品经理**，其功能包括读取电子邮件反馈。
   - 如果获得批准，它会将反馈整合到 **Linear** 看板中进行编辑，展示了函数调用 Agent (function calling agent) 架构的功效。
- **探索人机回环 (Human-in-the-Loop) 工作流**：成员们讨论了在嵌套工作流中实现人机回环 (HITL) 交互，旨在简化事件后用户控制权的回归。
   - 提出了一种事件驱动的方法，用于在工作流过程中动态管理用户响应。
- **用于 RAG 的有效网页爬取技术**：讨论集中在用于嵌入的网页爬取技术上，询问了关于 Puppeteer 与 Firecrawl 或 Crawlee 等工具的选择。
   - 成员们分享了关于将网页爬取数据集成到检索增强生成 (RAG) 流水线中的有效方法的见解。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **用户反馈推动 Blendtain 改进**：一位用户分享了对 **Blendtain** 的兴奋，但指出其容易切断消息，建议增加调整消息长度的功能。
   - 另一位用户表示赞同，反映了对该反馈的积极反响。
- **dykyi_vladk 发布播放列表生成器**：[Adify.pro](https://adify.pro) 作为一个新的播放列表生成器被推出，它可以根据用户提示词定制播放列表，由 dykyi_vladk 开发。
   - 作者自豪地称其为他“最酷的作品”，表明了对该项目的个人投入。
- **机器学习协作学习提案**：dykyi_vladk 邀请其他人私信他进行 **机器学习 (Machine Learning)** 协作学习倡议，促进社区参与。
   - 这一提议以友好的语气呈现，强调了在追求知识过程中的团队合作。
- **图像处理算法的主导地位转移**：一位成员询问了目前 **GANs**、**CNNs** 和 **ViTs** 在图像处理任务中的主导地位，寻求对这些趋势的确认。
   - 他们对通过视觉时间线来展示这些算法随时间的变化趋势表示感兴趣。
- **EleutherAI 的 muTransfer 合作项目**：EleutherAI 与 [Cerebras](https://cerebras.ai/) 启动了一个项目，旨在提高 **muTransfer** 的可访问性，目标是降低训练成本。
   - 成员们推测这种方法可能已经**过时**，质疑其与新方法相比的关联性。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Nvidia 的 51B 合成数据模型引起关注**：关于 Nvidia 的 **51B 合成数据模型**的讨论十分热烈，据报道该模型表现出强劲的 **MMLU** 性能，并具有增强应用的潜力。
   - *尝试对其进行微调和推理会很有趣，* 这突显了成员们探索实际应用的渴望。
- **自动分块（Auto Chunking）：上下文丢失？**：关于对话中**自动分块**实用性的辩论出现，担忧点在于 *想象一下你的对话在中间被切成两半，上下文就丢失了*。
   - 一名成员指出，像 **ST** 和 **Kobold** 这样的系统通常通过保留初始消息来管理溢出。
- **提议动态上下文管理**：讨论了**动态上下文管理**如何帮助 **LLM** 更有效地处理对话转移。
   - 成员们建议将此策略作为解决超出上下文限制的可能方案。
- **Axolotl 支持 Qwen 2.5**：确认 **Axolotl** 支持 **Qwen 2.5** 的普通文本处理，但视觉功能可能缺乏支持。
   - 这一确认反映了可能影响涉及视觉数据应用的局限性。
- **微调峰值分析**：一名成员报告在 **100K 行数据集**上进行微调时出现显著**峰值 (spike)**，正通过日志寻找相关性。
   - 成员注意到缺乏即时的日志帮助，这影响了故障排除工作。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Pydantic 兼容性损坏**：由于 Pydantic v2 不再支持 `__modify_schema__` 方法，用户在从 `langchain_openai` 导入 `ChatOpenAI` 时遇到错误。建议用户检查其 Pydantic 版本并改用 `__get_pydantic_json_schema__`，详见 [LangChain 文档](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis)。
   - 由于 Pydantic v2 已于 2023 年 6 月发布，开发者应确保使用兼容的方法以避免集成问题。
- **注意 GraphRecursionError！**：当达到 25 的递归限制时，LangGraph 应用中会出现 `GraphRecursionError`，从而阻碍执行。用户可以在配置中增加递归限制，正如相关 [GitHub issue](https://github.com/langchain-ai/langchain/issues/18598) 中所建议的。
   - 这一调整对于防止 LangGraph 中复杂图操作期间的崩溃至关重要。
- **呼吁 LLM 友好型文档！**：一名用户要求提供更多对 LLM 友好的文档，以提高 LangChain 的生产力。持续的讨论表明社区有兴趣改进为使用 LangChain 的开发者提供的资源。
   - 这表明需要更好的、专门用于增强 LLM 集成的指南，反映了社区的重点努力方向。
- **Mistral vs Mixtral：大对决**：开源领域关于 **Mistral** 和 **Mixtral** 自托管方案的对比正在酝酿中。成员们对这些模型在性能指标和易用性方面的表现感到好奇。
   - 这一对话突显了社区在为实际应用优化和选择最佳开源模型方面的兴趣。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **关于优化器中 CPU Offloading 的困惑**：讨论了为什么没有利用优化器的 CPU Offloading，并引用了提到性能下降的[这个旧 issue](https://github.com/pytorch/pytorch/issues/74588)。
   - 一名成员建议使用带有 CPU Offloading 的 **PagedAdam** 来优化性能，同时强调需要一个 PR 来考虑单设备微调。
- **优化器方法的对比分析**：成员注意到使用 **torchao 的 CPUOffloadOptimizer** 与反向传播（backward）中的优化器配合不佳，引发了对 Adam 等更快替代方案的疑问。
   - 建议包括尝试使用 `offload_gradients=True` 以节省梯度内存，同时优化 CPU 和 GPU 处理，详见此 [PR](https://github.com/pytorch/ao/pull/584)。
- **CUDA MODE 社区邀请**：建议对性能优化感兴趣的成员加入 GPU MODE Discord 群组，那里有更多专业人士可以提供帮助。
   - 分享的加入链接在[这里](https://discord.gg/jNsm4H44)，鼓励更广泛地参与优化讨论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **探索“行星大脑”的概念**：成员们戏称 **tinyboxes** 连接起来通过 **distributed training** 实现“行星大脑”的潜力，挑战集体智慧的边界。
   - 这暗示了一个迷人的未来，即先进的 **distributed training** 可能会在全球范围内运行。
- **用于分布式训练的 DisTrO 介绍**：讨论集中在 **[DisTrO 项目](https://github.com/NousResearch/DisTrO)**，该项目促进了 **Distributed Training Over-The-Internet**（互联网分布式训练），旨在实现跨模型的革命性协作。
   - 该倡议强调了模型训练中协作框架的需求，增强了可扩展性和可访问性。
- **AttributeError: 'Tensor' 缺少 cross_entropy**：一位用户在训练步骤中遇到了 'AttributeError'，原因是 **Tensor** 对象缺少 `cross_entropy` 属性，这突显了一个潜在的实现缺陷。
   - 参与者推测了根本原因，指向了 **Tensor** 功能中可能存在的空白。
- **Tinygrad 版本争议**：在一位用户从 **0.9.2** 版本过渡到最新的 master 分支并暴露了功能限制后，引发了关于正确 **Tinygrad** 版本的对话。
   - 建议持续更新以整合提升性能的基本特性。
- **模型架构与训练见解**：一位参与者分享了他们的模型架构，利用多个卷积层，随后进行展平操作和线性层，以增强训练效果。
   - 对话强调了旨在优化训练迭代期间模型性能的设计策略。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 迎来更新**：Open Interpreter 正在 [GitHub](https://github.com/OpenInterpreter/open-interpreter) 上积极接收更新，展示了持续的开发努力。
   - 重点围绕项目 '01'，旨在集成专用的语音助手模式，详见 [此处](https://github.com/OpenInterpreter/01)。
- **LLM 承担浏览器自动化任务**：一位成员讨论了使用 Open Interpreter 进行基于 LLM 的浏览器自动化，确认了其功能，但指出受限于任务复杂度。
   - 他们建议使用 Playwright 进行增强，并分享了一个他们一直在磨练的 [Prompt 示例](https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml)。
- **社区热情高涨**：尽管最初存在疑虑，社区成员仍渴望使用共享的 Prompt 自动向目录提交内容。
   - 随着成员回答问题并交流工具使用经验，参与度保持强劲。
- **即将举行的社区活动**：宣布了一个与 Open Interpreter 相关的即将举行的活动，并分享了 [包含更多详情的 Discord 链接](https://discord.gg/open-interpreter-1146610656779440188?event=1288234745477726238)。
   - 这一消息激发了用户的兴奋，表明社区兴趣依然活跃。
- **探讨项目感知的细微差别**：在回答一个查询时，一位成员幽默地指出，社区中的一些人可能并未完全意识到项目的进展。
   - 这指向了讨论中关于 Open Interpreter 生命力的不同看法。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# 第二部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288186634239742064)** (3 条消息): 

> - `Cursor Integration` (Cursor 集成)
> - `Gemini Update` (Gemini 更新)
> - `Database Downtime` (数据库停机维护)
> - `New Nous Model` (新款 Nous 模型)
> - `Open-source Vision Language Models` (开源 Vision Language Models)


- **Cursor 与 OpenRouter 集成！**: OpenRouter 现在可以在 [Cursor](https://x.com/OpenRouterAI/status/1838627801072562232) 中无缝使用所有模型，包括来自 Anthropic 的模型。
   - *感谢 @cursor_ai 修复了这个问题！* 🍾
- **Gemini 1.5 模型升级！**: [Gemini-1.5-flash](https://openrouter.ai/models/google/gemini-flash-1.5) 和 [gemini-1.5-pro](https://openrouter.ai/models/google/gemini-pro-1.5) 现在已路由到最新的 002 版本。
   - 此次更新使这两个模型都具备了最新的功能和改进。
- **计划中的数据库停机维护**: 发布了一份**停机通知**，指出在东部时间周五上午 10 点，将进行 5-10 分钟的数据库升级停机。
   - 这将确保后续运行更加顺畅。
- **Nous 发布多语言版 Llama 3.1！**: Nous 发布了一个针对多语言对话优化的 **Llama 3.1 8B** 新微调版本，可通过[此链接](https://openrouter.ai/models/nousresearch/meta-llama-3.1-8b-instruct)获取。
   - 该模型旨在增强全球沟通能力。
- **用 VLM 吐槽你自己！**: 几个开源的 **vision language models** 现已上线，包括 **Mistral Pixtral 12B** ([链接](https://openrouter.ai/models/mistralai/pixtral-12b)) 和 **Qwen** 系列模型 ([Qwen2-VL-7B-Instruct](https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct), [Qwen2-VL-72B-Instruct](https://openrouter.ai/models/qwen/qwen-2-vl-72b-instruct))。
   - 一定要在聊天室里让他们吐槽一张你的照片！ 🙂


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1838627801072562232">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 感谢 @cursor_ai 修复了这个问题！OpenRouter 现在可以在 Cursor 中使用所有模型，包括 Anthropic 🍾</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现出色，如视觉理解、分类、摘要，以及从图像、音频和视频中创建内容...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API, Providers, Stats</a>: Google 最新的多模态模型，支持文本或聊天提示中的图像和视频。针对语言任务进行了优化，包括：- 代码生成 - 文本生成 - 文本编辑 - 问题解决...</li><li><a href="https://openrouter.ai/models/nousresearch/meta-llama-3.1-8b-instruct">Llama 3.1 8B Instruct - API, Providers, Stats</a>: [Llama-3.1 8B Instruct](/models/meta-llama/llama-3 的微调版本。通过 API 运行 Llama 3.1 8B Instruct</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: 来自 Mistral AI 的首个图像转文本模型。其权重按照他们的传统通过种子发布：https://x。通过 API 运行 Pixtral 12B</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-7b-instruct">Qwen2-VL 7B Instruct - API, Providers, Stats</a>: Qwen2 VL 7B 是来自 Qwen 团队的多模态 LLM，具有以下关键增强：- 对各种分辨率和比例的图像具有 SoTA 级的理解：Qwen2-VL 实现了最先进的性能...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-vl-72b-instruct">Qwen2-VL 72B Instruct - API, Providers, Stats</a>: Qwen2 VL 72B 是来自 Qwen 团队的多模态 LLM，具有以下关键增强：- 对各种分辨率和比例的图像具有 SoTA 级的理解：Qwen2-VL 实现了最先进的性能...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1288069920734580756)** (1 messages): 

> - `OpenRouter App Development`
> - `Demo Apps on GitHub` 


- **OpenRouter 提供 Demo 应用以启动开发**：OpenRouter 团队宣布为有兴趣构建自己应用程序的用户提供基础 Demo 应用，可在 [GitHub](https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling) 上找到。
   - 这些 Demo 包括一个**简单的 'tool calling' Demo**，旨在引导用户完成应用创建的初始阶段。
- **征集对 Demo 应用的反馈**：OpenRouter 团队乐于接收用户关于 Demo 应用的**反馈和需求**。
   - 他们鼓励社区参与，并表示用户的意见将有助于改进未来的产品。



**提及的链接**：<a href="https://github.com/pxl-research/tai-llm-chat/tree/main/demos/tool_calling">tai-llm-chat/demos/tool_calling at main · pxl-research/tai-llm-chat</a>：包含 LLM Demo 代码的仓库（使用 Azure OpenAI 和 OpenRouter）- pxl-research/tai-llm-chat

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1287852298701242430)** (378 messages🔥🔥): 

> - `OpenRouter's middle-out transforms`
> - `New Gemini Models`
> - `Token Pricing Structures`
> - `Performance of various LLMs`
> - `User Experiences with Models` 


- **关于 OpenRouter Middle-Out 转换的讨论**：用户对取消默认 middle-out 转换表示质疑，理由是对其当前基础设施和工作流产生了负面影响。
   - 用户对模型变更的可访问性和沟通表达了担忧，一些用户强调需要更清晰的更新说明。
- **新 Gemini 模型发布公告**：Google 宣布发布两个更新模型：**Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002**，价格大幅下调且性能指标有所提升。
   - 新模型具有更快的输出速度和更高的 Rate Limits，并将于 2024 年 10 月 8 日前自动更新面向用户的别名。
- **各供应商的 Token 定价结构**：讨论了不同模型之间各异的 Token 定价，指出 OpenRouter 利用从上游返回的原生 Token 进行成本计算。
   - 用户获知 GPT-4o 和 Qwen 等模型之间 Tokenizer 的差异会影响 Token 计数和价格估算。
- **LLM 性能对比**：对比性能分析显示，虽然 Gemini Flash 002 比 GPT-4o Mini 更快，但有时无法满足编码约束。
   - 用户分享了生成式编码任务的经验，强调了 Gemini 在某些领域的优势，同时也指出了在遵循任务要求方面的局限性。
- **用户体验与 Bug 修复**：用户对 SambaNova 和 OpenRouter 等模型供应商快速解决 Bug 表示赞赏，并指出在报告问题后得到了及时的修复。
   - 关于用户体验的反馈强调了平台内的效率和响应速度，这增强了用户对这些技术的信心。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2024/Sep/24/gemini-models/">更新的生产就绪型 Gemini 模型</a>：Google Gemini 今天推出了两个新模型：`gemini-1.5-pro-002` 和 `gemini-1.5-flash-002`。它们的 `-latest` 别名将在“未来几天”内更新为这些新模型，而新的 `-001` 后缀...</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>：查看你在 OpenRouter 上使用模型的情况。</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>：管理来自模型的响应</li><li><a href="https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)">未找到标题</a>：未找到描述</li><li><a href="https://openrouter.ai/models?max_price=0">Models | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://ai.google.dev/gemini-api/docs/safety-settings">未找到标题</a>：未找到描述</li><li><a href="https://x.com/rowancheung/status/1838280020642676802">Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一次关于新的重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是一个重要的日子。禁令解除的那一刻，我将在 X 上发布完整对话...</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/">更新的生产就绪型 Gemini 模型，降低了 1.5 Pro 定价，增加了速率限制等</a>：未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions">未找到标题</a>：未找到描述</li><li><a href="https://github.com/open-webui/open-webui/blob/6b463164f4b129e0ce4bdc9008dd661214fe5eb5/backend/open_webui/apps/openai/main.py">open-webui/backend/open_webui/apps/openai/main.py at 6b463164f4b129e0ce4bdc9008dd661214fe5eb5 · open-webui/open-webui</a>：面向 LLM 的用户友好型 WebUI（原 Ollama WebUI）- open-webui/open-webui</li><li><a href="https://openrouter.ai/models/alpindale/magnum-72b">Magnum 72B - API, Providers, Stats</a>：由 [Goliath](https://openrouter.ai/models/alpindale/goliath-120b) 的制作者开发，Magnum 72B 是新模型系列中的首个成员，旨在达到 Claude 3 模型的散文质量，特别是...</li><li><a href="https://openrouter.ai/models/alpind">Models: 'alpind' | OpenRouter</a>：在 OpenRouter 上浏览模型</li><li><a href="https://github.com/OpenRouterTeam/open-webui/commit/89659df1fa10348f51b389a8fea27b67a71dec5d">默认添加 middle-out · OpenRouterTeam/open-webui@89659df</a>：未找到描述</li><li><a href="https://github.com/OpenRouterTeam/open-webui">GitHub - OpenRouterTeam/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>：面向 LLM 的用户友好型 WebUI（原 Ollama WebUI）- OpenRouterTeam/open-webui
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/)** (1 条消息): 

godling72: 就我而言，这会是我自己运行的东西。
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1288231645341548668)** (1 条消息): 

> - `Mistral Small 模型发布`
> - `Gradio 5 发布`
> - `FinePersonas 数据集介绍`
> - `bitsandbytes 0.44.0`
> - `Wikimedia 结构化维基百科数据集`

- **Mistral Small 模型发布**：全新的 [Mistral Small 模型](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) 现已上线，拥有 220 亿参数，有望在 AI 性能方面取得重大进展。
   - 该模型是可以通过 [HF Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) 探索的集合的一部分。
- **Gradio 5：轻松构建与分享**：[Gradio 5](https://5-0-dev.gradio-website.pages.dev/playground) 简化了创建和分享机器学习应用的过程，只需几行代码即可完成最少的设置。
   - 该工具可与任何 Python 库无缝集成，允许开发人员有效地展示他们的模型并生成易于访问的公开链接。
- **FinePersonas 助力合成数据**：最新的 [FinePersonas v0.1](https://x.com/reach_vb/status/1836882281434165629) 提供了 2100 万个画像（personas），有助于为各种应用创建多样化的合成数据。
   - 该数据集可以生成针对特定画像定制的真实查询和内容，彻底改变了合成数据的生成方式。
- **bitsandbytes 0.44.0 现已发布**：新公布的 [bitsandbytes 0.44.0](https://x.com/mattkdouglas/status/1838403695605690444) 引入了 AdEMAMix 优化器的 8-bit 版本，优化了性能。
   - 它还整合了对推理的 CUDA 图（CUDA graphs）支持，展示了轻量级模型优化器能力的进步。
- **维基媒体结构化维基百科数据集揭晓**：维基媒体发布了一个结构化的 [维基百科数据集](https://enterprise.wikimedia.com/blog/hugging-face-dataset/) 供公众反馈，该数据集源自其 Snapshot API。
   - 该数据集提供了改进的机器可读格式，简化了研究人员和开发人员的访问与分析。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground)">Gradio</a>：构建和分享令人愉悦的机器学习应用</li><li><a href="https://x.com/reach_vb/status/1836882281434165629)">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：介绍 FinePersonas-v0.1 - 许可宽松的 2100 万画像，用于生成大规模（多样且可控）的合成数据！🔥 使用 @AIatMeta Llama 3.1 70B Instruct, @arg... 制作</li><li><a href="https://x.com/mattkdouglas/status/1838403695605690444)">Matthew Douglas (@mattkdouglas) 的推文</a>：宣布 bitsandbytes 0.44.0！我们实现了由 @Apple 研究员 @MatPagliardini, @GrangierDavid 和 @PierreAblin 提出的 AdEMAMix 优化器的 8-bit 版本。</li><li><a href="https://x.com/micuelll/status/1838244638873809125)">Miquel Farré (@micuelll) 的推文</a>：好奇 FineVideo 是如何构建的吗？🍿 我们开源了整个抓取和处理脚本，将约 200 万个 YouTube 视频转换为用于训练视频基础模型的丰富、带注释的数据集。R...</li><li><a href="https://x.com/tomaarsen/status/1837132943728209921)">tomaarsen (@tomaarsen) 的推文</a>：我刚刚发布了 Sentence Transformers v3.1.1 补丁版本，修复了一些模型的硬负样本挖掘（hard negatives mining）工具。这个工具对于从你的 embedding 中获得更高性能非常有用...</li><li><a href="https://x.com/davidberenstei/status/1838482286523601339)">David Berenstein (@davidberenstei) 的推文</a>：为什么即使在使用合成数据时，查看你的合成数据也很重要？DataCraft UX 更新。数据可能包含怪癖，如重复的 prompt、过于困难的措辞和 Markdown 格式...</li><li><a href="https://x.com/gabrielmbmb_/status/1838239658737549797)">Gabriel Martín Blázquez (@gabrielmbmb_) 的推文</a>：好奇你可以用 FinePersonas 中的 2100 万个画像做些什么吗？一个用例是创建全新的数据集——就像我刚才做的那样！FinePersonas 合成电子邮件对话 ✉️ 使用 distilab...</li><li><a href="https://x.com/Gradio/status/1838210842497560971)">Gradio (@Gradio) 的推文</a>：🔥 由 @OzzyGT 提供的 Diffusers 快速局部重绘（Inpaint）。在你想擦除或更改的主体上绘制遮罩，并写下你想用什么来重绘它。使用 Diffusers 和 Gradio 创作有趣的艺术作品 😎</li><li><a href="https://enterprise.wikimedia.com/blog/hugging-face-dataset/)">Hugging Face 上的维基百科数据集：AI/ML 的结构化内容</a>：维基媒体企业版在 Hugging Face 上发布维基百科数据集，包含来自 Snapshot API 的结构化内容测试版，用于 AI 和机器学习应用</li><li><a href="https://x.com/qlhoest/status/1837179483201147279)">Quentin Lhoest 🤗 (@qlhoest) 的推文</a>：FinePersonas 是最丰富的画像数据集。现在你可以重写（ReWrite）它，以使画像适应你的需求（适用于 HF 上的任何数据集！）
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1287857707075764296)** (117 条消息🔥🔥): 

> - `Hugging Face Token 问题`
> - `OpenAI 的文字生成量`
> - `Gradio 与模型查询`
> - `语音频道的反馈`
> - `适用于 Hugging Face 工具的 Agents IDE` 


- **Hugging Face token 问题持续存在**：多位用户报告称，尽管多次尝试生成新的 Hugging Face token，但 token 仍然无效，这在多台机器上都引发了困扰。
   - 后续讨论指出可能存在速率限制 (rate limit) 问题，并建议通过重新安装 `huggingface-hub` 包来进行排查，但许多人仍面临同样的挑战。
- **OpenAI 每天生成海量文本**：一位用户进行了对比，强调 OpenAI 每天生成约 **1000 亿个单词**，直逼全人类每天生成的 **100 万亿个单词**。
   - 这引发了人们的好奇：Hugging Face 旗下的模型总量是否能达到这些统计数据。
- **Gradio Spaces 应用问题**：一位用户报告了其 Space 应用的外部日志问题，提到了本地和网络访问 URL，但在访问外部 URL 时遇到了困难。
   - 排查建议包括检查配置，但尚未讨论出具体的解决方案。
- **语音频道改进建议**：一名成员提议增强语音频道功能，包括增加更多 VC、音乐机器人以及定期举行每周讨论 AI 新闻的活动。
   - 社区成员反应积极，表示对提高语音频道的参与度和功能性很感兴趣。
- **对 Hugging Face 工具链的 Agents IDE 感兴趣**：一名成员表达了开发专门为 TGI 和 Hugging Face 工具设计的“agents IDE”的热情，类似于 langgraph-studio。
   - 他们询问是否有任何正在进行的此类项目或计划，并表示愿意协助开发。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">来自 Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一次关于新的重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是大日子。在禁令解除的瞬间，我将在 X 上发布完整对话...</li><li><a href="https://x.com/sama/status/1756089361609981993">来自 Sam Altman (@sama) 的推文</a>：openai 现在每天生成约 1000 亿个单词。地球上所有人每天生成约 100 万亿个单词。</li><li><a href="https://ai.google.dev/competition/projects/extractcode">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro">MMLU Pro - TIGER-Lab 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/cat-dance-dancing-cat-chinese-dancing-cat-funny-cat-meme-cat-gif-18059553370350307210">猫咪跳舞 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eyhrix/anybody_know_of_arx03_topscoring_model_on_mmlu_pro/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://tenor.com/bBEcB.gif">Dennis Reynolds GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf">flux1-dev-Q4_K_S.gguf · city96/FLUX.1-dev-gguf at main</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1287868297626521691)** (2 条消息): 

> - `Neuralink 的 FP8 性能`
> - `混合精度损失对比` 


- **FP8 损失与 bfloat16 持平**：一名成员指出，使用 **1b FP8** 的损失与 **bfloat16 混合精度**相当，表明两者性能非常接近。
   - *今天，我在测试中确认了这一结果*。
- **Neuralink 性能追踪**：Neuralink 正在积极研究与精度损失相关的性能指标，重点关注 **FP8** 和 **bfloat16**。
   - 用户反馈强调了这些指标对于优化 AI 建模工作的重要性。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288052979051794494)** (8 条消息🔥): 

> - `Llama 3.1 Safety Assessment` (Llama 3.1 安全评估)
> - `Comic Sans FLUX Model` (Comic Sans FLUX 模型)
> - `Qwen/Qwen2.5-72B-Instruct Model` (Qwen/Qwen2.5-72B-Instruct 模型)
> - `AI Font Generators` (AI 字体生成器)
> - `Neural Computation Paper` (Neural Computation 论文)


- **安全评估揭示 Llama 3.1 见解**：一个团队发布了关于 **Llama 3.1** 的[安全评估](https://www.hydrox.ai/blogs/c9fa272b-f982-4598-8e5c-002f9c019782)，强调更大的模型并不一定意味着更安全的模型。
   - 社区成员对此表示怀疑，并指出了之前关于 *ASCII char injection attacks*（ASCII 字符注入攻击）的类似发现。
- **Comic Sans 字体模型加入竞争**：赶在 [Text-Tacular Showdown 竞赛](https://civitai.com/articles/7587) 之前，一个新的 [FLUX model](https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572) 允许在图像生成中准确重现 **Comic Sans** 字体。
   - 尽管该字体经常受到批评，但该模型鼓励以一种有趣的方式在各种应用中使用这种备受诟病的字体。
- **Qwen/Qwen2.5-72B-Instruct 已上线**：**Qwen/Qwen2.5-72B-Instruct** 模型现在可以在 [Hugging Face](https://huggingface.co/chat/) 上访问，这是为社区提供高质量 AI 聊天模型持续努力的一部分。
   - 此次发布包括对 **Meta-Llama** 模型的最新更新，并可能包含增强功能。
- **Neural Computation 的历史性贡献**：一篇来自 **Neural Computation** 的引用讨论了约束在增强学习网络泛化能力方面的作用，特别是应用于手写邮政编码识别。
   - 该论文的作者包括 **Yann LeCun** 和 **Bernhard Boser**，他们是神经网络领域的知名人物。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.hydrox.ai/blogs/c9fa272b-f982-4598-8e5c-002f9c019782">HydroX AI</a>：未发现描述</li><li><a href="https://neurosciencenews.com/nlep-ai-language-26329/">无标题</a>：未发现描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>：让社区最好的 AI 聊天模型对所有人可用。</li><li><a href="https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u-x6o8ySG0sC">Backpropagation applied to handwritten zip code recognition</a>：Y LeCun, B Boser, JS Denker, D Henderson, RE Howard, W Hubbard, LD Jackel, Neural computation, 1989 - 被引用 17,404 次</li><li><a href="https://civitai.com/models/791942/comic-sans-font-for-flux?modelVersionId=885572">Comic Sans Font for Flux - V1 | Stable Diffusion LoRA | Civitai</a>：赶在 Text-Tacular Showdown 竞赛之前，通过使用这个 FLUX 模型生成带有文本的图像，在竞争中获得优势，以准确地...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1287876157001306172)** (175 条消息🔥🔥): 

> - `Hugging Face models`
> - `Audio extraction tools`
> - `Social engineering GPT`
> - `Google Gemini object detection`
> - `Tau LLM training` 


- **新的社会工程学 GPT 模型**：一位用户在 Hugging Face 上分享了一个名为 [Social Engineering GPT](https://huggingface.co/spaces/abdurrahman01234/social_engineering_GPT) 的新模型，强调了其在网络安全应用中的有效性。他们正在寻求合作伙伴以进一步对模型进行 fine-tuning。
   - 该模型展示了 AI 在网络安全领域的潜力，引起了爱好者和专家的共同关注。
- **YouTube 转音频 Python 工具**：一位用户介绍了一个 [YouTube-to-Audio](https://github.com/jack-tol/youtube-to-audio) 软件包，允许用户轻松地从 YouTube 视频和播放列表中提取音频。该工具支持多种音频格式，并可以通过 pip 安装。
   - 该工具简化了音频提取过程，消除了对不可靠在线转换器的需求，从而提升了用户便利性。
- **Gemini 目标检测 Demo**：一位用户展示了使用 [Google Gemini](https://huggingface.co/spaces/saq1b/gemini-object-detection) 进行目标检测的 demo，该 demo 可以从图像中生成 bounding box 坐标。此功能允许对 Gemini 的能力进行实际测试和探索。
   - 该 demo 突出了 Gemini 在计算机视觉任务中的潜力，为用户提供了一种无缝的方式来交互并理解模型的输出。
- **社区参与和协作**：用户表达了对开发和 fine-tuning 模型进行合作的兴趣，强调了社区推动增强 AI 项目的动力。这种氛围鼓励分享知识，并寻求改进现有模型的帮助。
   - 这反映了社区中一种日益增长的趋势，即利用集体专业知识来突破 AI 能力的边界。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/saq1b/gemini-object-detection">Gemini Object Detection - a Hugging Face Space by saq1b</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/abdurrahman01234/social_engineering_GPT">Social Engineering GPT - a Hugging Face Space by abdurrahman01234</a>: 未找到描述</li><li><a href="https://youtube.com/live/ZyiH7IkBr_w?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 21</a>: **欢迎回到我们的 Tau LLM 系列！🌟** 在本集中，我们将深入探讨我们的第四次训练尝试，即 **Series D**。以下是我们的计划：...</li><li><a href="https://tenor.com/view/3po-star-wars-this-is-madness-gif-13899583">3po Star Wars GIF - 3po Star Wars This Is Madness - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/Unity-Technologies/ml-agents/blob/develop/LICENSE.md">ml-agents/LICENSE.md at develop · Unity-Technologies/ml-agents</a>: Unity Machine Learning Agents Toolkit (ML-Agents) 是一个开源项目，使游戏和模拟能够作为使用深度强化学习训练智能 Agent 的环境...</li><li><a href="https://github.com/Unity-Technologies/UnityCsReference">GitHub - Unity-Technologies/UnityCsReference: Unity C# reference source code.</a>: Unity C# 参考源代码。通过创建账号为 Unity-Technologies/UnityCsReference 的开发做出贡献。</li><li><a href="https://github.com/jack-tol/youtube-to-audio">GitHub - jack-tol/youtube-to-audio: A lightweight Python package and command-line interface (CLI) tool that extracts audio from YouTube videos and playlists in multiple formats, such as MP3, WAV, OGG, AAC, and FLAC.</a>: 一个轻量级的 Python 软件包和命令行界面 (CLI) 工具，可从 YouTube 视频和播放列表中提取多种格式的音频，如 MP3, WAV, OGG, AAC 和 FLAC。</li><li><a href="https://pypi.org/project/youtube-to-audio/">youtube-to-audio</a>: 一个轻量级的 Python 软件包和命令行界面 (CLI) 工具，可从 YouTube 视频和播放列表中提取多种格式的音频。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1288164470144634880)** (3 条消息): 

> - `HF Dataset`
> - `Cross-Posting Etiquette` 


- **即将发布的 HF Dataset 针对日本和美国**：一位成员宣布计划很快发布一个 [HF dataset](https://link.to.dataset)，专门针对**日本和美国**。
   - 这一战略重点表明了在扩大不同地区数据集相关性方面的针对性方法。
- **频道礼仪提醒**：一位用户温和地提醒另一位用户不要进行 Cross-posting（交叉发布），并保持讨论集中在当前话题上。
   - *Cross-posting* 可能会打断对话流，强调了遵守频道指南的重要性。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288181660080406529)** (2 条消息): 

> - `GOT OCR 2 Model`
> - `Fine-tuning OCR models`
> - `Text-image datasets`
> - `Language-specific training` 


- **为特定语言项目探索 GOT OCR 2**：一位成员对新的 **GOT OCR 2** 模型表示了兴趣，但注意到它没有在他们的语言上进行预训练，表明需要进行 **Fine-tuning**。
   - 他们请求指导和阅读材料，以协助为 Fine-tuning 过程创建特定语言的 **Text-image dataset**。
- **寻求 Fine-tuning 过程的帮助**：该成员表达了希望在 Fine-tuning **GOT OCR 2** 方面获得支持的愿望，并对提供的任何帮助预先表示感谢。
   - 他们正在积极寻求阅读材料和指导建议，以更好地理解所需的步骤。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1288062730351411273)** (7 条消息): 

> - `SetFit Models Training`
> - `Daily Topic Modeling`
> - `Sentiment Analysis Methods`
> - `BERTTopic`
> - `Zero-shot Topic Definition` 


- **探索适用于 SetFit 模型的在线服务**：一位成员询问了适合训练 **SetFit models** 的在线服务。
   - 这个问题反映了人们对高效模型训练解决方案日益增长的兴趣。
- **每日主题建模中的挑战**：另一位成员讨论了使用 **BERTTopic** 确定合理主题数量的困难，并指出在生产环境中需要手动合并。
   - 他们强调了在保持主题完整性的同时管理不断变化的数据的复杂性。
- **用于主题管理的 Zero-shot 方法**：另一位成员分享了他们部署 **Zero-shot 方法** 来定义主题的经验，发现在限制主题数量上限的情况下在生产环境中取得了成功。
   - 这种方法允许将新主题捆绑为“其他”，或在模型运行后动态生成名称。
- **寻找情感分析的替代方案**：有成员提出，希望在不完全依赖 **OpenAI** API 的情况下寻找最先进的 **Sentiment Analysis**（情感分析）方法。
   - 这表明了在外部外包能力之外寻求自给自足模型的动力。
- **持续主题聚类需求**：一位成员表达了每天进行主题聚类或持续添加新主题的愿望，并承认目前对该过程缺乏经验。
   - 他们指出，依赖条件逻辑（if-else）的解决方案对他们的用例没有吸引力。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1287859753615101973)** (14 条消息🔥): 

> - `ControlNet_Union in SDXL`
> - `Training DiT models`
> - `Sigma_t term importance`
> - `Denoising processes`
> - `Latent variable equations` 


- **ControlNet_Union 的严格条件限制**：一位用户注意到 **SDXL 的 ControlNet_Union** 在输入为 Scribble（涂鸦）时会在输出中保留空白区域，并询问如何解决此问题。
   - 另一位成员建议，修剪图像的部分内容可以帮助模型生成更连贯的背景，特别是在使用 Fill/Inpaint/Outpaint 技术时。
- **Denoising 中 Sigma_t 的细微差别**：一位成员质疑在他们的 **DiT** 模型训练采样过程中 **sigma_t** 项的必要性，想知道它是否会影响输出的连贯性。
   - 他们得出结论，使用 **sigma_t** 项有助于逐步进行 **Denoising**，而不是一次性对整个图像进行去噪，并分享了使用替代方程产生更好结果的实践经验。
- **潜变量方程的探索**：讨论集中在调整潜变量的不同方程上，一位用户发现尽管偏离了原始论文的方程，但修改后的方程改善了结果。
   - 该用户对他们的发现表示不确定，并希望更好地理解数学表示，这暗示了 **Denoising Diffusion Models** 的复杂性。


  

---

### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1287889405935222856)** (2 条消息): 

> - `Gradio 5 Beta Release`
> - `Gradio Performance Improvements`
> - `Modern UI Design in Gradio`
> - `AI Playground Feature`
> - `Office Hours Demo` 


- **Gradio 5 Beta 激动人心发布**：团队宣布 **Gradio 5 (Beta)** 正式推出，解决了开发者在性能和易用性方面的重大关切。
   - *我们希望在正式发布 Gradio 5 之前获得您的反馈！*
- **Gradio 5 通过 SSR 提升性能**：**Gradio 5** 具有显著的性能改进，包括服务器端渲染 (SSR)，从而提升了应用的加载速度。
   - 这旨在解决用户长期反馈的 **Gradio 加载过慢** 的问题。
- **Gradio 焕然一新**：在 Gradio 5 中，许多组件（如 **Buttons**、**Tabs** 和 **Sliders**）都采用了现代设计进行了刷新。
   - 此次更新解决了 **Gradio 外观过时** 的担忧，并增强了整体用户体验。
- **在 Gradio 5 中引入 AI Playground**：**Gradio 5** 配备了一个实验性的 **AI Playground**，用户可以直接在浏览器中生成并预览 Gradio 应用。
   - 该功能旨在克服 **LLM 不了解 Gradio** 的挑战，并鼓励与平台的互动。
- **参加 Gradio Office Hours 观看现场演示**：团队邀请用户参加 Office Hours，演示全新的 **服务器端渲染功能**。
   - 该活动定于 **东部时间明天中午 12:00** 举行，届时将展示最新的增强功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://5-0-dev.gradio-website.pages.dev/playground">疑似钓鱼网站 | Cloudflare</a>: 未找到描述</li><li><a href="https://huggingface2.notion.site/Gradio-5-A-Production-Ready-Web-Framework-for-ML-Applications-a4d7e42c26f4450aa0758d968019d120?pvs=74)">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一体化工作空间。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1287853726094069830)** (143 条消息🔥🔥): 

> - `RWKV 架构`
> - `YouTube 转音频工具`
> - `机器学习中的动态评估 (Dynamic Evaluation)`
> - `muP 实现` 


- **理解 RWKV 的架构**：成员们讨论了 **RWKV 架构**的各个方面，特别是其独特的功能如 **ddlerp** 以及对最近 token 的强调，突出了它们与**卷积 (convolutions)** 相比的效率。
   - 有人指出，理解 **RWKV** 需要熟悉 GLA，虽然存在复杂性，但社区认为简化其解释有助于推广采用。
- **新 YouTube 转音频工具介绍**：一位用户宣布创建了一个名为 **youtube-to-audio** 的新命令行工具，可以从 YouTube 提取各种格式的音频，包括 **MP3** 和 **WAV**。
   - 该工具允许自定义输出文件名和下载播放列表，是现有通常带有广告的方法的更简单替代方案。
- **讨论机器学习的动态评估 (Dynamic Evaluation)**：一位成员提出了**动态评估**的概念，即直接在测试集上对模型进行微调，并因对其外部有效性的担忧而质疑其有效性。
   - 虽然在技术上是可行的，但这种方法可能不符合典型的评估实践，强调了训练集和测试集之间需要保持同分布。
- **muP 实现的进展**：目前正在进行澄清 **muP** (Maximal Update Parameterization) 及其在神经网络中应用的工作，旨在尽管其数学原理复杂，仍能推动社区采用。
   - 在理论解释的同时发布更简单的实现被认为是促进理解的关键，使开发者更容易将 muP 集成到他们的框架中。
- **与 yt-dlp 工具的比较**：围绕现有的 YouTube 音频下载工具展开了讨论，强调了 **yt-dlp** 作为一个功能丰富的下载器已经可用。
   - 该工具与新推出的 **youtube-to-audio** 一起被推荐，进一步丰富了寻求音频提取解决方案的用户选择。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/cloneofsimo/status/1838287517906510026">Simo Ryu (@cloneofsimo) 的推文</a>: 好东西。专业提示：按照我勾选的红圈操作可以完成 99%。（但不要缩放 head dim） https://blog.eleuther.ai/mutransfer/</li><li><a href="https://blog.eleuther.ai/mutransfer/">Maximal Update Parameterization 实战指南</a>: 探索 mutransfer 的实现细节</li><li><a href="https://index.commoncrawl.org/">Common Crawl Index Server 的推文</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/rwkv">RWKV 简介 - 兼具 Transformer 优点的 RNN</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2101.06804">什么构成了 GPT-$3$ 优秀的上下文示例 (In-Context Examples)？</a>: GPT-$3$ 因其在广泛的 NLP 任务中的卓越表现而备受关注，尤其是其强大且通用的上下文少样本学习能力。尽管它...</li><li><a href="https://github.com/cloneofsimo/zeroshampoo/blob/main/distributed_shampoo.py">zeroshampoo/distributed_shampoo.py at main · cloneofsimo/zeroshampoo</a>: 通过在 GitHub 上创建账户为 cloneofsimo/zeroshampoo 的开发做出贡献。</li><li><a href="https://github.com/jack-tol/youtube-to-audio">GitHub - jack-tol/youtube-to-audio: 一个轻量级的 Python 包和命令行界面 (CLI) 工具，可以从 YouTube 视频和播放列表中提取多种格式的音频，如 MP3, WAV, OGG, AAC 和 FLAC。</a>: 一个轻量级的 Python 包和命令行界面 (CLI) 工具，可以从 YouTube 视频和播放列表中提取多种格式的音频，如 MP3, WAV, OGG, AAC 和 FLAC。 - jack-tol/youtub...</li><li><a href="https://pypi.org/project/youtube-to-audio/">youtube-to-audio</a>: 一个轻量级的 Python 包和命令行界面 (CLI) 工具，可以从 YouTube 视频和播放列表中提取多种格式的音频。</li><li><a href="https://github.com/google-research/google-research/blob/master/scalable_shampoo/jax/shampoo.py">google-research/scalable_shampoo/jax/shampoo.py at master · google-research/google-research</a>: Google Research。通过在 GitHub 上创建账户为 google-research/google-research 的开发做出贡献。</li><li><a href="https://github.com/yt-dlp/yt-dlp">GitHub - yt-dlp/yt-dlp: 一个功能丰富的命令行音频/视频下载器</a>: 一个功能丰富的命令行音频/视频下载器 - yt-dlp/yt-dlp
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1287851153417310208)** (43 条消息🔥): 

> - `LLMs 的规划能力`
> - `AI 的可解释性`
> - `FP6 浮点格式性能`
> - `ML 中的 Scaling laws`
> - `隐式指令微调 (Implicit instruction tuning)` 


- **OpenAI 的 LRM 展示规划能力**：讨论强调，虽然 OpenAI 最近的模型 o1 (Strawberry) 声称是一个 Large Reasoning Model (LRM)，但其有效性仍存争议，尤其是在某些测试条件下准确率较低。
   - 成员指出 *“推理时计算 (inference time compute) 使其在预览模型上从 0% 提升到 50% 以上”*，这引发了对其规划能力的质疑。
- **对 AI 可解释性论文的批评**：一位成员分享了一篇论文，批评许多 AI 可解释性方法缺乏有意义的见解，且在没有适当评估的情况下犯了统计错误。
   - 论文指出 *“特征归因解释 (feature attribution explanations) 在我们的任务中对人类决策者提供的效用微乎其微”*。
- **FP6 格式在 H100 上超越 BF16**：有报告称，FP6 在 H100 上的精度与 BF16 持平，同时速度快于 FP8/BF16，且 vLLM 已集成了性能增强功能。
   - Alpin Daley 声称 *“吞吐量非常可观，且精度保持与 FP8 相当”*，指向了多样化浮点格式的利用。
- **ML 中的 Scaling Laws 与正则化**：最近的一项研究质疑，在大型语言模型 (LLMs) 占据主导地位的时代，既定的正则化原则是否仍然适用。
   - 作者提出了一种称为 “Scaling law crossover” 的现象，即传统原则可能不再成立，焦点从泛化误差转向了近似误差。
- **语言模型中的隐式指令微调**：研究结果表明，仅针对响应 (responses) 进行训练就能实现指令遵循，这让人怀疑有效的模型训练是否必须使用“指令-响应对”。
   - 这种 *“隐式指令微调 (implicit instruction tuning)”* 还揭示了窄领域数据仍然可以带来广泛的指令遵循能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.14254">Instruction Following without Instruction Tuning</a>：指令微调通常意味着在指令-响应对上微调语言模型。我们发现了两种与指令微调相比有所缺陷的适配（微调）形式，但仍然……</li><li><a href="https://arxiv.org/abs/2409.15156">Rethinking Conventional Wisdom in Machine Learning: From Generalization to Scaling</a>：大型语言预训练的显著成功和 Scaling laws 的发现标志着机器学习范式的转变。值得注意的是，主要目标已从最小化泛化……演变而来。</li><li><a href="https://arxiv.org/abs/2409.13373">LLMs Still Can&#39;t Plan; Can LRMs? A Preliminary Evaluation of OpenAI&#39;s o1 on PlanBench</a>：规划能够实现预期状态的行动路线的能力长期以来被认为是智能 Agent 的核心能力，并且一直是 AI 研究不可或缺的一部分，自其……</li><li><a href="https://x.com/AlpinDale/status/1838369139288596687">Alpin (@AlpinDale) 的推文</a>：不知何故，FP6 在基准测试中的表现优于 BF16。很快将在 vLLM 中落地。https://github.com/vllm-project/vllm/pull/8751 引用 Alpin (@AlpinDale) 的话：你现在可以在任何浮点格式中加载任何 FP16 模型……</li><li><a href="https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines">Betteridge&#039;s law of headlines - Wikipedia</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2012.02748">Challenging common interpretability assumptions in feature attribution explanations</a>：随着机器学习和算法决策系统越来越多地被应用于高风险的人机交互场景，迫切需要理解其过程的基本原理……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1288008158345957451)** (10 messages🔥): 

> - `Chinchilla 与 matmul 算法`
> - `Strassen 算法讨论`
> - `使用 Strassen 分解模型`
> - `低精度 matmul 性能` 


- **Chinchilla 与 Matmul 算法的关系**：成员们指出 Chinchilla 基本上是基于 **matmul 算法**的，讨论暗示使用更快的变体可以改变模型的**最优点 (optimal points)**。
   - 这引发了关于通过改变精度设置来进行性能调整的讨论。
- **Strassen 算法：并不受欢迎**：服务器中达成了一致共识，即 **Strassen 算法**已被多次讨论，但人们对其有效性缺乏信心。
   - 一些成员推测其在 **inference** 过程中的潜力，但怀疑态度仍然普遍。
- **受 Strassen 启发的模型分解思路**：一位成员建议，可以潜在地以**受 Strassen 启发**的方式分解模型，以减少加法和减法操作的数量。
   - 这种方法可能会产生接近全模型能力的近似结果。
- **低精度 vs Strassen 的辩论**：一位成员指出，与其实现 **Strassen**，不如在更低精度下运行传统的 matmul 以获得类似的结果。
   - 这增加了关于 Strassen 在实际场景中功效的持续争论。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1287862866078334997)** (13 messages🔥): 

> - `Pythia 6.9b-deduped 的 MMLU 分数`
> - `Pile 模型的格式问题`
> - `与 ARC 的性能对比`
> - `引用即将发表的论文` 


- **Pythia 6.9b-deduped 的 MMLU 分数偏低**：一位用户对 **Pythia 6.9b-deduped 模型**得到的 **MMLU 5-shot 分数**非常低（约为 **26%**）表示担忧。
   - 另一位成员质疑这些分数是否远低于已发表的 **Pythia** 分数，从而引发了关于模型性能的讨论。
- **Pile 模型在格式化方面的挑战**：成员们讨论了在 **Pile** 上训练的模型由于格式遵循度较差，在 **MMLU** 上表现尤为吃力，从而影响了它们的性能。
   - 有人指出，如果将风格调整为模仿 **ARC**，性能会显著提高。
- **ARC vs. MMLU 性能讨论**：据观察，当 **GPT-NeoX-20B** 遵循 **MMLU** 的风格时，其在 **ARC easy** 上的得分接近**随机水平**，这突显了格式的关键作用。
   - 尽管如此，两个基准测试之间基于格式风格的性能差异仍然很大。
- **寻求格式问题的参考文献**：一位用户正在寻求有关 **Pile 模型**在格式化方面存在困难这一说法的引用，特别是来自一篇即将发表的论文。
   - 另一位成员提供了论文 '[Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)' 的引用，该论文讨论了这一问题。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1287875397513511026)** (2 messages): 

> - `trunc_normal 初始化`
> - `AllenAI 消融研究`
> - `模型稳定性` 


- **考虑切换到 trunc_normal 初始化**：发起了一场关于是否将函数初始化更改为 **trunc_normal** 以提高模型性能的讨论。
   - 由于大规模模型中潜在的稳定性问题，这次更改的重要性得到了强调。
- **AllenAI 的消融研究强调了稳定性**：引用的 [AllenAI 消融研究](https://arxiv.org/abs/2409.02060) 表明，没有使用 trunc_normal 的模型在**大规模下表现出不稳定性**。
   - 该研究的作者包括 **Niklas Muennighoff** 和 **Luca Soldaini** 等核心研究人员，指出如果不使用 trunc_normal 可能会导致严重的后果。



**提到的链接**：<a href="https://arxiv.org/abs/2409.02060">OLMoE: Open Mixture-of-Experts Language Models</a>：我们介绍了 OLMoE，这是一个利用稀疏 Mixture-of-Experts (MoE) 的完全开放、最先进的语言模型。OLMoE-1B-7B 拥有 70 亿 (B) 参数，但每个输入 token 仅使用 1B。我们对其进行了预训练...

  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1287854453281652848)** (122 messages🔥🔥): 

> - `Gemini 模型更新`
> - `新的 RoRF 开源发布`
> - `Claude 模型预期`
> - `Aider 安装问题`
> - `Prompt 缓存功能`

- **Gemini 模型迎来重大更新**：发布了新的生产级 Gemini 模型 **Gemini-1.5-Pro-002** 和 **Gemini-1.5-Flash-002**，其价格降幅超过 **50%**，并提高了速率限制（rate limits）。
   - 开发者对其性能提升印象深刻，尽管在编程任务的基准测试（benchmarks）中表现似乎没有明显变化。
- **令人兴奋的 RoRF 开源发布**：**Routing on Random Forest (RoRF)** 已开源，超越了以往的方法，并引入了 **12 个预训练模型路由（model routers）**。
   - 此次发布因提升了 MMLU 的性能而受到赞誉，为模型路由开辟了新途径。
- **对新 Claude 模型的期待**：关于发布新 Claude 模型的猜测不断，特别是 **Haiku 3.5** 或 **Opus**。
   - 用户对自上次更新以来的漫长等待表示沮丧，希望很快能有相关公告。
- **Aider 安装挑战**：一些用户报告了 Aider 的问题，导致尝试使用 **pipx** 卸载并重新安装，但功能仍然无法正常使用。
   - 社区提出了回滚到旧版本等建议，凸显了社区面临的挑战。
- **Prompt caching 注意事项**：讨论了 Prompt caching 功能，特别是它如何应用于 **Anthropic API** 和 **OpenRouter**。
   - 用户寻求关于配置 **AIDER_CACHE_KEEPALIVE_PINGS** 及其在不同环境中影响的澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/rowancheung/status/1838280020642676802">Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一个关于全新重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是重要的一天。在禁令解除的瞬间，我将在 X 上发布完整对话...</li><li><a href="https://tenor.com/view/side-eye-cat-gif-8216273864367202904">Side Eye Cat GIF - 翻白眼猫 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>：未找到描述</li><li><a href="https://status.anthropic.com/incidents/4k8gmdx11lcq">3.5 Sonnet 部分停机</a>：未找到描述</li><li><a href="https://x.com/OfficialLoganK/status/1838611055217385646">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：两个新的生产级 Gemini 模型，速率限制提高 2 倍以上，Gemini 1.5 Pro 降价 50% 以上，过滤器改为选择性加入（opt-in），更新了 Flash 8B 实验模型等。今天对...来说是个好日子。</li><li><a href="https://openrouter.ai/docs/prompt-caching">Prompt Caching | OpenRouter</a>：优化 LLM 成本高达 90%</li><li><a href="https://x.com/OpenAIDevs/status/1838611640201162807">OpenAI Developers (@OpenAIDevs) 的推文</a>：OpenAI o1 API 可用性更新：- 我们已向第 4 层级（两个模型均为每分钟 100 次请求）的开发者扩大了访问权限。- 我们为第 5 层级（每分钟 1000 次请求）的开发者将速率限制提高了 5 倍...</li><li><a href="https://x.com/testingcatalog/status/1838358531579285511?s=46">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：看起来新的 Gemini 模型明天发布，而不是 Opus 3.5 👀👀👀 引用 ʟᴇɢɪᴛ (@legit_rumors)：新的更新版 Gemini 1.5 模型可能很快就会发布™ 🚀 很有可能不仅仅是...</li><li><a href="https://tenor.com/view/its-just-gambling-liam-scott-edwards-ace-trainer-liam-betting-gamble-gif-20475304">Its Just Gambling Liam Scott Edwards GIF - 这只是赌博 Liam Scott Edwards Ace Trainer Liam - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/incidents/4hc6130xwxt5">Anthropic API 错误率和延迟升高</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=1iolYLlUBiI">Zinley Berkeley 讲座演示</a>：使用终端的 Multi-Agent 全负载。UI 版本即将推出。</li><li><a href="https://x.com/tomas_hk/status/1838586544657240234">Tomas Hernando Kofman (@tomas_hk) 的推文</a>：今天我们开源了 RoRF (Routing on Random Forests)，这是一个成对模型路由器，击败了所有闭源和开源方法，同时还发布了 12 个预训练模型路由器：Hugging Face: http:/...</li><li><a href="https://cloudonair.withgoogle.com/events/gemini-at-work-24">Gemini at Work</a>：加入 Google Cloud CEO Thomas Kurian 和行业领袖的行列，探索 AI 如何重塑全球业务。</li><li><a href="https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/">更新的生产就绪 Gemini 模型、降低的 1.5 Pro 定价、提高的速率限制等</a>：未找到描述</li><li><a href="https://github.com/Not-Diamond/RoRF">GitHub - Not-Diamond/RoRF: Routing on Random Forest (RoRF)</a>：Routing on Random Forest (RoRF)。通过在 GitHub 上创建账户为 Not-Diamond/RoRF 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/commit/7fa1620f58132ec085a7939a8015bbe7935827a2">feat: 允许在 SEARCH/REPLACE 块前缀中灵活匹配 5-9 个字符 · paul-gauthier/aider@7fa1620</a>：…块前缀</li><li><a href="https://github.com/paul-gauthier/aider/issues/1697">&lt;无响应&gt; Bug · Issue #1697 · paul-gauthier/aider</a>：问题 C:\Users\pierr\Desktop\Github\claude-3-artifacts&gt;aider --model openrouter/anthropic/claude-3.5-sonnet --no-pretty Aider v0.57.1 主模型：openrouter/anthropic/claude-3.5-sonnet 带有 diff ed...</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5">Gemini Flash 1.5 - API、提供商、统计数据</a>：Gemini 1.5 Flash 是一个基础模型，在视觉理解、分类、摘要以及从图像、音频和视频创建内容等各种多模态任务中表现出色...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API、提供商、统计数据</a>：Google 最新的多模态模型，支持文本或聊天提示中的图像和视频。针对语言任务进行了优化，包括：- 代码生成 - 文本生成 - 文本编辑 - 问题解决...
</li>
</ul>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1287851745229148250)** (62 messages🔥🔥): 

> - `Aider 文件操作`
> - `在 Aider 中使用模型`
> - `升级 Aider`
> - `HuggingChat 模型`
> - `Aider 使用教程` 


- **在 Aider 中管理只读文件**：用户可以使用 `AIDER_READ` 配置为 Aider 添加多个只读文件，从而实现文档的高效组织。
   - 使用 `/tokens` 命令可以确认已添加哪些只读文件及其数量，这对初学者来说非常清晰。
- **在 Aider 中调用弱模型**：正如社区成员提到的，目前除了使用 `/model switch` 命令外，没有办法在运行中动态切换到较弱的模型。
   - 一些用户指出，对于简单问题使用低功耗模型具有潜在的成本节约优势。
- **Aider 的升级程序**：用户报告了使用 pipx 从 0.56 版本升级到 0.57 版本时遇到的问题，可能是由于缓存了旧版本。
   - 建议的升级命令包括 `pipx upgrade aider-chat` 或使用 `pipx uninstall` 和 `install` 进行完全重新安装。
- **访问 HuggingChat 模型**：成员们讨论了通过 API 获取 HuggingChat 模型的可行性，并指出了付费模型与免费模型之间的性能差异。
   - 一位用户分享了支持不同类型 Hugging Face 模型 API 访问的 LiteLLM 链接，增强了在 Aider 中的可用性。
- **Aider 使用教程与资源**：分享了各种教程视频，以帮助新用户有效地配置和使用 Aider。
   - 资源包括关于设置 Aider 和构建应用程序的 YouTube 教程链接，促进了社区知识共享。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html#optimizing-the-map">Repository map</a>：Aider 使用 Git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>：由 aider 用户制作的入门和教程视频。</li><li><a href="https://simonwillison.net/2024/Sep/20/introducing-contextual-retrieval/">Introducing Contextual Retrieval</a>：这是 Anthropic 描述的一种有趣的新型 embedding/RAG 技术，它应该适用于任何 embedding 模型针对任何其他 LLM 的场景。实现语义搜索的一大挑战是...</li><li><a href="https://huggingface.co/chat/models">HuggingChat - Models</a>：浏览 HuggingChat 可用的模型。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-turn-on-the-repository-map">FAQ</a>：关于 aider 的常见问题解答。</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>：LiteLLM 支持以下类型的 Hugging Face 模型：</li><li><a href="https://github.com/paul-gauthier/aider/pull/1688">fix: improve automatic upgrade flow for aider by fry69 · Pull Request #1688 · paul-gauthier/aider</a>：修复 #1687（精神上）文档化的安装流程，确保应用程序在升级后退出而不继续运行，引入了 Config 类/模块以访问已解析的命令...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1287859125363015710)** (5 messages): 

> - `Agentic 行为`
> - `OpenRouter 集成`
> - `Gemini 模型更新` 


- **关于 RAG 中 Agentic 行为的讨论**：一位成员强调，许多人投资 **RAG** 是因为对 **Agentic 行为** 感到困惑。
   - 另一位成员表示赞同，并补充了一个轻松的评论。
- **OpenRouter 模型现在可在 Cursor 中使用**：一位成员宣布 **OpenRouter 模型** 现在与 **Cursor** 兼容，并感谢他们的修复。
   - 此次集成现在支持所有模型，包括来自 **Anthropic** 的模型。
- **发布的 Gemini 模型新细节**：Simon Willison 的一篇文章讨论了 **两个新的 Gemini 模型**：`gemini-1.5-pro-002` 和 `gemini-1.5-flash-002`，并指出了它们的基准测试和更新。
   - **Pro 模型** 的价格从 10 月 1 日起大幅下调，对于 **128,000 tokens** 以下的输入，输入成本从 **$3.50** 降至 **$1.25/million** tokens。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1838627801072562232">来自 OpenRouter (@OpenRouterAI) 的推文</a>：感谢 @cursor_ai 修复此问题！OpenRouter 现在可以在 Cursor 中使用所有模型，包括 Anthropic 🍾</li><li><a href="https://simonwillison.net/2024/Sep/24/gemini-models/">更新的生产就绪级 Gemini 模型</a>：今天来自 Google Gemini 的两个新模型：`gemini-1.5-pro-002` 和 `gemini-1.5-flash-002`。它们的 `-latest` 别名将在“未来几天”更新为这些新模型，并且新的 `-001` 后缀...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1288207496518959165)** (1 条消息): 

> - `Advanced Voice rollout`
> - `Custom Instructions update`
> - `Improved Accents`
> - `New Voices Feature`
> - `Multilingual Capabilities` 


- **Advanced Voice 向 Plus 和 Team 用户推出**：**Advanced Voice** 功能本周将向 ChatGPT App 的所有 **Plus** 和 **Team** 用户推出，提升用户体验。
   - 用户可以期待更多功能，包括 [Custom Instructions](https://x.com/OpenAI/status/1838642444365369814) 和 **Memory** 功能。
- **新增令人兴奋的新语音功能**：此次更新在 ChatGPT App 中增加了 **五种新语音**，允许更高程度的个性化。
   - 用户现在可以体验到 **改进的口音**，进一步增强了沟通能力。
- **Advanced Voice 支持 50 多种语言**：新的 **Advanced Voice** 功能可以用 **50 多种语言** 表达“对不起，我迟到了”这句话，展示了其多语言能力。
   - 这为全球用户开启了多样化的互动可能性。



**提到的链接**：<a href="https://x.com/OpenAI/status/1838642444365369814">来自 OpenAI (@OpenAI) 的推文</a>：Advanced Voice 将在本周内向 ChatGPT App 的所有 Plus 和 Team 用户推出。在您耐心等待的同时，我们增加了 Custom Instructions、Memory、五种新语音……

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1287851889379115160)** (108 条消息🔥🔥): 

> - `Advanced Voice Mode`
> - `Voice Generation Performance`
> - `Voice Assistant Competition`
> - `Roleplaying AI Services`
> - `GPU Server Rentals` 


- **Advanced Voice Mode 推出的混乱**：**Advanced Voice Mode** 的推出引发了挫败感，尤其是在欧洲，用户仍在等待访问权限，许多人对功能限制和约束表示失望。
   - *“这与 5 月份的演示相去甚远，”* 一位用户评论道，强调了预期功能与实际能力之间的差距。
- **关于语音生成能力的辩论**：用户正在批评当前语音生成模型的表现，称其缺乏在改变语音和情感表达方面所承诺的灵活性。
   - 一位用户指出，尽管系统声称无法哼唱，但他们还是成功让语音哼唱了起来，这揭示了安全准则中的不一致性。
- **与 Google 的语音助手竞争**：讨论指出 OpenAI 正在试图与 **Google Assistant** 等企业级机器人竞争，这导致其采取了安全优先的方法，限制了更多动态功能的发挥。
   - 一位用户谈到了他们的看法，即 OpenAI 正在平衡企业竞争与构建角色扮演 AI 产品（如 character.ai 上的产品）之间的关系。
- **GPU 服务器租赁推荐**：用户分享了针对短期需求的 GPU 服务器租赁建议，**Vast.ai** 和 **salad.com** 被提及为经济实惠的选择。
   - 有人提到，利用 YouTuber 的赞助链接可以为一次性租赁提供大量额度，这对于训练模型特别有用。
- **对 AI 角色扮演服务的兴奋**：对 character.ai 的隐晦提及引发了兴趣，用户对其在角色扮演场景中的受欢迎程度和能力表示惊讶。
   - 有人强调，许多用户在最初尝试这些 AI 服务时遇到了拒绝，导致一些人形成了 DIY 社区方法。



**提到的链接**：<a href="https://fixupx.com/OpenAI/status/1838642448375206345">来自 OpenAI (@OpenAI) 的推文</a>：见见这五种新语音。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1288015657903853578)** (5 条消息): 

> - `Voice Function in GPT`
> - `Calling GPTs` 


- **关于对 GPTs 进行语音呼叫的澄清**：一位用户询问是否可以通过语音功能呼叫某个 GPT，但得到的回答是断然的 **不**。
   - 另一位成员坚持认为 **ChatGPT** 无法呼叫其他 GPTs，并强调他们已在建议频道中提议启用此功能。
- **启用 GPT 呼叫的建议**：一位成员提到他们提出了一个建议，希望 **ChatGPT** 允许通过语音功能呼叫其他 GPTs。
   - 这突显了对聊天机器人功能内增加互动性的需求。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1287935677115273236)** (21 messages🔥): 

> - `API 的 Prompt engineering`
> - `JSON 的 Structured output`
> - `生成 Minecraft 问题`
> - `Hallucination 问题`
> - `API 使用查询` 


- **JSON 格式响应的困难**：一位成员对 JSON 格式回答的质量表示沮丧，称有时响应仅为一个简单的 '{'。
   - 他们建议使用 Structured output 可能会改善格式。
- **澄清 Prompt engineering**：讨论涉及定义 Prompt engineering，强调了清晰陈述请求并提供足够上下文的重要性。
   - 一位成员指出，Prompt 应该详细，以帮助模型产生更好的响应。
- **创建有趣的 Minecraft 问题**：一位成员分享了他们的 Prompt，旨在生成 JSON 格式的有趣 Minecraft 相关问题，用于游戏内聊天。
   - 他们提到了输出重复的挑战，并寻求改进 Prompt 的建议。
- **关于 Hallucination 和响应质量的问题**：当 Temperature 设置超过 1.25 时，模型产生 Hallucination 的倾向引发了关注。
   - 一位成员指出，指示模型不要产生 Hallucination 似乎能最大限度地减少偏离预期输出的问题。
- **通过 API 强制执行 JSON 输出**：一位成员询问了在使用原始 HTTPS API 时，确保 API 一致生成 JSON 格式输出的方法。
   - 他们澄清说没有使用任何库或 Wrappers，这可能会影响输出的一致性。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1287935677115273236)** (21 messages🔥): 

> - `JSON 格式化挑战`
> - `AI 的 Prompt engineering`
> - `Minecraft 问题生成`
> - `API 使用见解`
> - `避免响应中的 Hallucinations` 


- **对 JSON 格式回答的沮丧**：一位用户担心他们的 JSON 格式响应通常质量较差或不完整，有时甚至只产生一个 '{'。
   - 另一位成员建议使用更结构化的输出（Structured output）可以提高响应质量。
- **理解 Prompt Engineering**：成员们讨论了 Prompt engineering 的概念，强调需要清晰地陈述要求和上下文以改进输出。
   - 一位用户指出，在他们的 Prompt 中提供具体示例可以增强生成的问题质量。
- **生成 Minecraft 相关问题**：一位用户分享了他们专为通过 API 创建有趣的 Minecraft 相关问题而设计的 Prompt，并寻求对其有效性的反馈。
   - 他们报告了获取多样化问题的挑战，并注意到提高 Temperature 会导致生成内容中出现 Hallucinations。
- **使用原始格式的 API**：一位参与者澄清说，他们在 Prompt 中使用的是原始 HTTPS API，没有使用任何库或 Wrappers。
   - 他们询问了确保 API 有效输出 JSON 的方法。
- **对输出中 Hallucinations 的担忧**：讨论中包括对旨在防止 AI 响应中出现 Hallucinations 的 Prompt 指令有效性的怀疑。
   - 用户分享了一些经验，即通用指令未能限制生成问题中无关或刻板的内容。


  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1287944793279696956)** (7 messages): 

> - `旧服务器图标回归`
> - `社区反思`
> - `存在诈骗链接`
> - `工具讨论` 


- **旧服务器图标回归**：成员们注意到**旧服务器图标**回来了，引发了用户的怀旧之情。
   - 一位成员评论道，“回到我们的根基”，强调了这一变化的意义。
- **社区对变化感受复杂**：一位成员对社区现状表达了“痛苦”的感觉。
   - 这种情绪反映了对近期发展的一些不满。
- **关于诈骗链接的警告**：一位成员提醒其他人注意频道中发布的**诈骗链接**，呼吁保持警惕。
   - 此通知提醒社区对潜在威胁保持警觉。
- **新人询问工具**：一位新成员询问“这是什么工具？”，表现出了解社区资源的渴望。
   - 作为回应，成员 **hy3na_xyz** 提到所讨论的工具是 **btop**。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

mobicham: 将此对话移至 hqq 频道
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1287898235020640297)** (30 条消息🔥): 

> - `CUDA Caching Allocator`
> - `Triton Kernel Support`
> - `Segment Anything Model 2 (SAM2)`
> - `Debugging Torch Distributed Training`
> - `Imitation Learning with SAM2-fast` 


- **CUDA Caching Allocator 与 Tensor 对齐**：讨论围绕 **CUDA** caching allocator 的最小块大小为 **512 字节**展开，确认了它会返回对齐的地址，但对 **PyTorch** 中的 Tensor 对齐提出了疑问。
   - 一位成员分享了一段代码片段，演示了 Tensor 切片并引用了 **PyTorch** 中的 Tensor 对齐机制。
- **Torch.compile 的 Triton Kernel 问题**：在使用 autotune 中包含 `prune_configs_by` 的自定义 **Triton** kernel 时，**torch.compile** 会出现故障，这促使人们考虑一种预剪枝（pre-pruning）的变通方案。
   - 成员们讨论了提交 Issue 以解决此限制的必要性，并确保为用户编写的 **Triton** kernel 提供正确的路由。
- **对 Segment Anything Model 2 (SAM2) 的兴趣**：成员们表达了对 **SAM2** 的兴趣，并讨论了其在交互式物体选择和图像分割标注中的应用。
   - 一位成员提议探索一个针对用户需求定制的 **SAM2-fast** 版本，强调了其在协作开发方面的潜力。
- **调试 Torch 分布式训练代码**：一位成员询问了调试 **torch.distributed** 训练代码的最佳实践，强调了在没有 GPU 的情况下进行本地模拟能力的需求。
   - 他们寻求关于使用断点和可视化并行结构的建议，但尚未发现满意的解决方案。
- **使用 SAM2-fast 进行模仿学习**：提出了一个有趣的构想，即利用 **SAM2-fast** 作为 **Diffusion Transformer Policy** 的输入进行模仿学习，涉及从传感器数据到机械臂关节位置的映射。
   - 这激发了进一步探索 **SAM2** 在机器人领域应用的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch-labs/segment-anything-fast">GitHub - pytorch-labs/segment-anything-fast: A batched offline inference oriented version of segment-anything</a>: 一个面向批处理离线推理版本的 segment-anything - pytorch-labs/segment-anything-fast</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: 该仓库提供了运行 Meta Segment Anything Model 2 (SAM 2) 推理的代码、下载已训练模型 Checkpoints 的链接，以及展示如何使用该模型的示例 Notebooks。</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: 一个用于大模型训练的原生 PyTorch 库。可以通过在 GitHub 上创建账号来为 pytorch/torchtitan 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1287894159876948141)** (1 messages): 

> - `GPU MODE`
> - `CUDA MODE Origins`
> - `IRL Meetup Success`
> - `Community Growth`
> - `Future of GPU Programming` 


- **CUDA MODE 转型为 GPU MODE**：曾被称为 **CUDA MODE** 的社区现已更名为 **GPU MODE**，反映了其在 CUDA 之外更广泛的 GPU 编程关注点。
   - 这一转变旨在营造一个包容的环境，让成员可以在 GPU 技术领域进行学习、协作和创新。
- **从读书小组发展到 9,000 名成员**：**CUDA MODE** 最初是 **PMPP 书籍** 的一个读书小组，现已增长到超过 **9,000 名成员**，并创建了超过 **10 个开源项目**，如 *torchao* 和 *Liger*。
   - 该社区对开源生态系统做出了重大贡献，展示了其对协作的承诺。
- **线下见面会（IRL Meetup）取得巨大成功**：首次线下见面会吸引了 **150 名黑客**，在一天之内开发了 **40 多个项目**，展示了充满活力的社区精神。
   - 参与者致力于创新项目，包括移植 **PyTorch FlexAttention** 和优化 CUDA kernel，展示了他们的技能。
- **深度专注与分心：社区身份**：“CUDA MODE”一词起源于 **Tim Dettmers** 的一次病毒式演讲，强调了编程中深度专注的力量，但社区现在也同样重视社交互动。
   - 成员们享受共同协作和实验，为 GPU 编程领域的探索创造了一个支持性的空间。
- **拥抱更广泛的 GPU 编程理念**：向 **GPU MODE** 的转型反映了拥抱 CUDA 之外各种编程语言和框架（如 **Triton** 和 **WebGPU**）的承诺。
   - 社区领导者表示希望将讨论扩展到包括 **Groq** 和 **TPU** 在内的技术，鼓励在性能领域的发展。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1288168618340651008)** (4 messages): 

> - `CUDA Programming Course`
> - `Nvidia GPUs`
> - `High-Performance Computing` 


- **发布新的 CUDA 编程课程**：发布了名为 ["CUDA Programming Course – High-Performance Computing with GPUs"](https://www.youtube.com/watch?v=86FAWCzIe_4&ab_channel=freeCodeCamp.org) 的 YouTube 视频，重点介绍使用 Nvidia CUDA 进行高性能计算和深度学习的编程。
   - 相关的 [代码仓库](https://github.com/Infatoshi/cuda-course) 也已提供，为学习者提供动手实践材料。
- **社区对课程质量的好奇**：一位成员询问是否有人尝试过新发布的 CUDA 课程，以及它是否好用。
   - 另一位成员自信地肯定了课程的质量，表示：*“它很棒，是我制作的。今天刚发布。”*



**提到的链接**：<a href="https://www.youtube.com/watch?v=86FAWCzIe_4&ab_channel=freeCodeCamp.org">CUDA Programming Course – High-Performance Computing with GPUs</a>：学习如何使用 Nvidia CUDA 编程，并利用 GPU 进行高性能计算和深度学习。代码：💻 https://github.com/Infatoshi/cuda-course 💻 h...

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1288253857121570827)** (1 messages): 

> - `Luma job opening`
> - `Performance optimization roles`
> - `Dream Machine product`
> - `Luma's research team` 


- **Luma 寻找顶级性能工程师**：Luma 正在寻找优秀的工程师来优化他们的训练和推理栈，特别是针对多模态基础模型，从 [Dream Machine](https://lumalabs.ai/dream-machine) 开始。他们提供在帕罗奥图（Palo Alto）的实地职位，并可为优秀的远程候选人提供签证赞助。
- **关键岗位需要深厚的专业知识**：候选人应在大型分布式训练、底层 kernel（如 **Triton** 和 **CUDA**）或优化分布式推理工作负载的吞吐量和延迟方面拥有深厚经验。这强调了对调试和编译专业知识的需求。
- **Luma 令人印象深刻的增长和支持背景**：Luma 拥有一支强大的扩散模型研究团队，Dream Machine 在短短 **4 天内就获得了 100 万用户**，展示了极高的产品市场契合度。他们由 a16z 支持，并拥有充足的现金流用于增长。
- **快节奏且精简的工作环境**：公司专注于快速开发和所有权，官僚作风极少，允许快速执行项目。他们强调构建和发布产品的效率。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1287911271076728906)** (8 条消息🔥): 

> - `Cutlass 示例讨论`
> - `将 CUDA 移植到 Python`
> - `PyTorch 的 Custom ops` 


- **对 Cutlass 示例的好奇**：一位成员引用了一个可能符合描述的 [Cutlass 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/13_two_tensor_op_fusion/README.md)，并提到他们还没有仔细阅读。
   - 另一位成员对其性能优势表示感兴趣，尽管他们发现该解释是面向 **Cutlass 用户**而非 **CUDA 初学者**的。
- **将独立 CUDA 代码移植到 Python**：一位成员询问了关于将 **独立 CUDA 代码移植**到 Python 的方案。
   - 作为回应，另一位成员建议使用 PyTorch 中的 **load_inline** 作为一种简单的解决方案。
- **CUDA 到 PyTorch 转换的最佳实践**：在讨论了初步方案后，一位成员询问了用 Python 包装 CUDA 的 **最佳实践**。
   - 有人建议参考 PyTorch 中的 [此资源](https://github.com/pytorch/ao/tree/main/torchao/csrc) 来探索 **custom ops**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVIDIA/cutlass/blob/main/examples/13_two_tensor_op_fusion/README.md">cutlass/examples/13_two_tensor_op_fusion/README.md at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账户为 NVIDIA/cutlass 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/csrc">ao/torchao/csrc at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288070575054520380)** (10 条消息🔥): 

> - `uintx 的 Slice 操作`
> - `uintxTensor 的 Padding 讨论`
> - `uintx 切片的测试示例`
> - `Tensor 的整除性要求` 


- **尝试为 uintx 添加 Slice 操作**：一位成员正尝试添加 slice 操作，并使用在 `pack` 函数中检查 8 整除性的代码进行测试。
   - *我知道关于 padding 和 non-padding 有很多讨论。* 该成员质疑 padding 应该在 `UintxTensor` 处理，还是在 bitpacking 函数内部处理。
- **切片 uintx Tensor 的问题**：该成员指出，尝试使用 `x[2:6]` 进行切片未能按预期工作。
   - 提供了一个示例来演示尝试切片 `uintx` Tensor 时遇到的挑战。
- **关于 Padding 实现的担忧**：该成员正在考虑在 `pack` 函数中实现 padding，并反思其整洁度与影响。
   - 他们征求了建议，表示不确定这种调整的潜在影响。
- **大型 Tensor 的整除性要求**：一位成员指出 `uintx` 的用例是为大型 Tensor 设计的，这表明对 8 整除的要求是一个宽松的限制。
   - 这突显了关于形状维度限制的潜在灵活性。
- **对更大切片的考虑**：另一位成员建议对 sub-byte 数据类型尝试更大的切片，认为限制形状必须被 8 整除以便我们始终可以进行 pack 是合理的。
   - *对于 sub-byte dtypes，限制形状维度必须被 8 整除以便我们始终可以 pack 它们，这看起来合理吗？*


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1287905097468219402)** (18 条消息🔥): 

> - `CUDA Mode vs GPU Mode`
> - `Heterogenous Computing Discussions`
> - `Segment Anything Model 2`
> - `Nickname Proposals for GPU Mode`
> - `Mascots for GPU Mode` 


- **关于 CUDA Mode 与 GPU Mode 的辩论**：成员们对从 **CUDA Mode** 更名为 **GPU Mode** 表达了复杂的情绪，一致认为这两个名称与 “Heterogeneous Computing” 等替代方案相比都不够顺口。
   - **CUDA Mode** 可能更容易读，但有人**担心** **GPU Mode** 过度简化了更广泛的处理单元。
- **介绍 Segment Anything Model 2**：[Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2) 仓库提供了运行 Meta 模型推理的代码，包括模型 checkpoints 的链接和示例 notebooks。
   - 一位成员通过分享其 GitHub 页面链接及相关功能，强调了该工具的潜力。
- **GPU Mode 的创意昵称建议**：针对 **GPU Mode** 的更名出现了一些建议，如 **Parallel Mode** 和 **Accel Mode**，引发了用户的笑声。
   - 有趣的是，一些成员提出了诸如 *Gigachad Processing* 和 *Generic Processing Unit* 之类的绰号，展示了讨论中的幽默创意。
- **GPU Mode 的吉祥物构思**：成员们讨论了 **GPU Mode** 吉祥物的概念，提议中包括 **Goku** 和幽默的 *H100 purse*。
   - 这种俏皮的角度旨在为新名称注入更多个性，突显了社区对有趣身份的渴望。
- **GPU 名称的历史**：关于 **GPU** 名称的一个趣闻浮出水面，回顾了它在被普及为 **Graphics Processing Unit** 之前最初的定位是 **Geometry Processor Unit**。
   - 一位用户澄清说 JHH 创造了 GPU 这个术语，导致一些成员重新思考他们之前对这段历史的理解。



**提到的链接**：<a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>：该仓库提供了运行 Meta Segment Anything Model 2 (SAM 2) 推理的代码，以及下载训练好的模型 checkpoints 的链接和展示如何使用该模型的示例 notebooks...

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1288237546698182707)** (2 条消息): 

> - `Triton Puzzles`
> - `PMP Book and Machine Learning` 


- **关于 Triton Puzzles 的 CUDA 替代方案查询**：*是否有类似 Triton puzzles 但针对原生 CUDA 的资源？* 一位成员对针对 CUDA 编程的类似资源表示好奇。
   - 这一询问凸显了在 CUDA 领域缺乏像 Triton puzzles 那样易于获取的资源。
- **PMP 书籍缺乏 Machine Learning 重点**：一位成员评论说 **PMP 书籍** 完全没有提到 **Machine Learning**，并表示失望。
   - *老实说，它确实没有涉及任何 ML 主题，* 这表明项目管理框架与不断发展的技术趋势之间存在脱节。


  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1288185066534469806)** (5 条消息): 

> - `3-bit attention kernel`
> - `Performance drop-off below 4 bits`
> - `Quantization in Attention Modules`
> - `GemLite for HQQ backend` 


- **推动 3-bit Attention Kernel**：一位成员指出，在 attention modules 中，精度低于 **4-bit** 后收益递减，并强调 **3-bit** 的性能是令人满意的。
   - 这促使人们倡导实现 **3-bit kernel** 以提高效率。
- **低于 4 Bits 时性能急剧下降**：讨论表明，在 **4 bits** 以下，**性能下降非常剧烈**，人们对模型输出的影响表示担忧。
   - 成员们正在分析这如何影响整体模型的有效性，以及是否需要进行调整。
- **Attention Modules 量化的挑战**：Attention modules 在较低的 **n-bits** 下量化更为复杂，且它们通常比 **MLP layers** 小，后者应该进行更激进的压缩。
   - 大家一致认为，专注于 **MLP layer 压缩** 可能会带来显著的性能提升。
- **研究 GemLite 以提升速度**：一位成员目前正致力于将 **GemLite** 作为 **HQQ** 的后端加入，以评估其对端到端速度的影响。
   - 这一集成的结果尚待观察，但被视为整体性能的潜在增强。


  

---

### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288199427521122388)** (9 条消息🔥): 

> - `repkv Kernel 集成`
> - `RoPE 实现细节`
> - `测试框架增强`
> - `RMSNorm 更新`
> - `SwigLU 修改` 


- **repkv Kernel 已添加到 Llama 3**：**repkv kernel** 已集成到 **Llama 3** 分支中，为进一步的进展铺平了道路。
   - 一位成员强调，需要澄清与 **RoPE** pull request 相关的 **q** 和 **k** 的 tensor shapes。
- **彻底测试 RoPE 集成**：重点强调将 **RoPE** 正确集成到 **dev/cuda** 环境中，确保 CPU 和 GPU 实现之间的等效性测试。
   - 对 **swiglu** 和 **rmsnorm** 的额外测试也被认为对于跨实现的一致性至关重要。
- **RoPE 需要额外的文件**：计划向 **dev/cuda** 添加几个必要文件，包括用于前向和反向传播的 **rope.cuh**。
   - 需要包含 **swiglu_forward** 和 **swiglu_backward** 文件，以便在合并这些功能的同时保持代码结构。
- **RoPE 功能需要独立性**：为了减少混淆，建议将 **RoPE** 功能拆分，使其对 **Q** 和 **K** 独立运行。
   - 这一更改旨在简化集成过程并提高开发者之间的清晰度。
- **当前实现缺乏 RoPE Scaling**：目前的 **RoPE** 实现不支持 **3.1** 版本中引入的 **RoPE scaling**。
   - 解决这一疏忽对于保持与最新更新的兼容性至关重要。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1287952766144086108)** (7 条消息): 

> - `使用 xformers 进行训练`
> - `CK GroupedGemm 的使用`
> - `VSCode 中的 CUDA 和 Metal`
> - `Llama3 微调历程` 


- **使用 xformers 训练显示出前景**：成员们讨论了 **xFormer attention backend** 与 Composable Kernel 配合运行，确保了**准确性和速度**。
   - *“很有用的信息，谢谢！”* 一位成员对收到的信息表示感谢。
- **寻求关于 CK GroupedGemm 的建议**：一位用户询问关于使用 **CK GroupedGemm** 的问题，表示在众多可用示例中不确定该遵循哪一个。
   - 另一位成员要求澄清**目标硬件**，表示需要上下文来提供相关建议。
- **在 AMD MI300x 上微调 Llama3**：一位成员分享了他们在 [AMD MI300x 上微调 Llama3 405B](https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey)) 的历程链接，详细介绍了他们的经验。
   - 这一见解可能为从事类似项目的其他人提供宝贵的策略。
- **VSCode 用于 CUDA 和 Metal**：一位用户提到他们专门使用 **VSCode** 编写 **CUDA** 和 **Metal** 代码。
   - 这突显了 VSCode 在适应多种编程环境方面的多功能性。



**提到的链接**：<a href="https://publish.obsidian.md/felafax/pages/Tune+Llama3+405B+on+AMD+MI300x+(our+journey)">Tune Llama3 405B on AMD MI300x (our journey) - Felafax Blog - Obsidian Publish</a>：在 AMD MI300x 上微调 Llama3 405B（我们的历程）- Felafax 博客 - 由 Obsidian Publish 提供支持。

  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1288062309268324392)** (1 条消息): 

> - `Cutlass Sycl 分支` 


- **对 Cutlass Sycl 分支的兴趣**：现在有一个新的 [Cutlass Sycl Fork](https://github.com/codeplaysoftware/cutlass-fork) 可用，专注于线性代数子程序的 CUDA 模板。这个分支可能会引起寻求优化 CUDA 解决方案的开发者的兴趣。
- **贡献潜力**：GitHub 仓库鼓励为 **Cutlass Sycl Fork** 的开发做出贡献，增强线性代数运算的能力。
   - 开发者可以参与该项目，以改进 CUDA 生态系统中现有的资源和工具。



**提到的链接**：<a href="https://github.com/codeplaysoftware/cutlass-fork">GitHub - codeplaysoftware/cutlass-fork: CUDA Templates for Linear Algebra Subroutines</a>：线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账户为 codeplaysoftware/cutlass-fork 的开发做出贡献。

  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1287918558646632520)** (28 messages🔥): 

> - `4090 GPU 上的扩展性`
> - `分布式训练优化`
> - `GPU 点对点通信` 


- **2x 4090 训练的扩展性问题**：在 vast.ai 上使用 **2x 4090** 进行训练时，**FSDP** 仅显示出 **13% 的加速**，而 **DDP** 提供了 **47% 的加速**，这归因于跨 **NUMA 节点** 的通信带宽缓慢。
   - 一位参与者指出，由于更高的通信效率，使用 **A100 实例** 的扩展性通常更好，能实现近乎**完美的加速比**。
- **低带宽下 DDP 优于 FSDP 的优势**：成员们讨论了在低带宽场景下，**DDP** 的表现如何优于 **FSDP**，DDP 能够在前向传播期间聚合梯度以提高效率。
   - 这引发了关于 **torch.compile** 等新特性如何与 **DDP** 协同工作的见解。
- **GPU 中的点对点通信**：一位参与者描述了使用 Geohot 编写的驱动程序，该驱动程序利用 **PCIe DMA 特性**，在不需要 CPU 的情况下实现 GPU 之间的**点对点通信**。
   - 这种方法允许 GPU 通过 PCIe 连接直接通信，创建了一个快速的 GPU 间通信设置。
- **Tinybox 配置挑战**：强调了构建包含多个 GPU 的 **tinybox** 配置的挑战，解释了尽管过程很有趣，但其复杂性不容小觑。
   - 参与者对这种配置的潜在**云端版本**表示感兴趣，以便更容易获取。
- **探索 NVMe 卸载性能**：一位用户正在两个 **4090** GPU 上运行用于 PCIe 通信的内核模块，测试将**数据卸载**到 **NVMe 驱动器**与**系统 RAM** 及 **CXL** 相比的优势。
   - 他们指出了 **resizable BAR** 在无需 CPU 干预的情况下促进高效数据传输的作用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/NVIDIA/">NVIDIA Corporation</a>：NVIDIA Corporation 拥有 509 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/NVIDIA/nccl-tests">GitHub - NVIDIA/nccl-tests: NCCL Tests</a>：NCCL 测试。通过创建账户为 NVIDIA/nccl-tests 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1287860300661522433)** (11 messages🔥): 

> - `WebNN 频道讨论`
> - `WebNN 与 WebGPU 及 WASM 的集成`
> - `活动邀请`
> - `NPU API 接口` 


- **是否应该创建 WebNN 频道？**：一位成员建议为 **WebNN** 创建一个单独的频道，指的是将 **WebGPU** 和 **WASM** 集成到一个统一的 **WebNN 架构**中。
   - 然而，另一位成员对该范围表示怀疑，认为鉴于快速变化的**多模态领域格局**，标准化此类高层级集成具有挑战性。
- **WebNN 与 NPU API 的接口**：成员们讨论了 **WebNN** 的目的，质疑其是否主要旨在与**固定功能 NPU API** 进行接口。
   - 对方澄清说，**WebNN** 也与 **WebGPU** 和 **WASM** 集成，并可以通过 Windows 上的抽象层与 **NVIDIA GPU** 配合工作。
- **活动规划与协调**：一位用户请求通过私信发送活动邀请，表示由于该活动与其**正在进行的项目**相关，因此非常期待。
   - 另一位成员迅速确认邀请已发送，展示了社区的参与度和协调性。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: 我们是否有关于 liger-kernel 的定期会议？
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1288178009328849081)** (2 messages): 

> - `GPU 显存带宽`
> - `谜题完成情况` 


- **延迟影响 GPU 数据处理**：一位成员指出，进出 **GPU** 的有效数据传输取决于**延迟**和 GPU 的显存带宽。
   - 他们质疑在 **SoC (片上系统)** 中，GPU 是否比 **CPU** 具有更优越的内存带宽。
- **解谜者想要比较解决方案**：一位成员询问是否有人完成了谜题，表示他们自己的进度已经过半。
   - 他们表示打算在完成谜题后比较解决方案。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1287861392409362616)** (87 条消息🔥🔥): 

> - `OpenAI's o1 models`
> - `Anthropic's funding talks`
> - `Stability AI board announcement`
> - `Gemini model updates`
> - `Scale AI financials` 


- **OpenAI 的 o1 模型引发关注**：OpenAI 最近发布了 o1 系列模型，以及一张关于推理时计算（test-time compute）扩展定律（scaling laws）的图表，尽管 x 轴未标注，但引发了关于使用 o1-mini API 对其进行重构的讨论。
   - 一位成员指出，所使用的计算量可能仅在数万个 token 范围内，并质疑了在没有树形结构的情况下进行扩展的可行性。
- **Anthropic 潜在的 400 亿美元估值**：有报道称 Anthropic 已开始与投资者讨论融资事宜，这可能使该初创公司的估值达到 **300 亿美元**至 **400 亿美元**，实际上比今年早些时候的估值翻了一番。
   - 这一消息反映了在快速发展的背景下，AI 公司寻求加强财务支持的竞争态势。
- **James Cameron 加入 Stability AI 董事会**：Stability AI 宣布传奇电影制作人 **James Cameron** 已加入其董事会，强调了他有望通过以艺术家为中心的方法为视觉媒体创新做出贡献。
   - 他的加入被视为 Stability AI 迈出的重要一步，旨在为创作者开发更全面的 AI 工作流（pipeline）。
- **Gemini 模型增强功能发布**：Gemini 模型的新生产版本已披露，其特点是 **速率限制（rate limits）提高了 2 倍以上**，Gemini 1.5 Pro **降价 50%**，并为开发者更新了实验性功能。
   - 更新后的设置包括用于管理安全性和可靠性的选择性加入（opt-in）过滤器，增强了开发者对配置的控制。
- **Scale AI 财务增长见解**：据报道，尽管毛利率相对较低，但 Scale AI 增长势头良好，对其上半年财务状况的分析显示，其销售额几乎翻了两番。
   - 随着对 AI 服务需求的持续升级，这使 Scale AI 处于一个有利的地位。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1838611055217385646">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 两个新的生产级 Gemini 模型，速率限制提高 2 倍以上，Gemini 1.5 Pro 降价 50% 以上，过滤器改为选择性加入，更新了 Flash 8B 实验模型等。今天是个好日子...</li><li><a href="https://x.com/hughbzhang/status/1838288923656941860">来自 Hugh Zhang (@hughbzhang) 的推文</a>: OpenAI 最近发布了 o1 系列模型和一张显示推理时计算扩展定律的图表——遗憾的是没有标注 x 轴。仅使用公开的 o1-mini API，我尝试重构了...</li><li><a href="https://x.com/amir/status/1838604896087740645">来自 Amir Efrati (@amir) 的推文</a>: Scale AI 的毛利率相当低，但目前增长看起来很健康。上半年财务数据如下：https://www.theinformation.com/articles/scale-ais-sales-nearly-quadrupled-in-first-half?utm_source=ti_app&rc=...</li><li><a href="https://x.com/aaronpholmes/status/1838580256032116981?s=46">来自 aaron holmes (@aaronpholmes) 的推文</a>: 今日 AI 议程独家：Microsoft AI 负责人 Mustafa Suleyman 调整了部分组织架构，Phi 先驱 Sebastien Bubeck 离职。另一位前 Phi 负责人已跳槽至 Google。以下是...</li><li><a href="https://x.com/OfficialLoganK/status/1838613832790360208">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @TheXeophon 是的，另一个更新版本</li><li><a href="https://x.com/StabilityAI/status/1838584605986951254">来自 Stability AI (@StabilityAI) 的推文</a>: 今天，我们的 CEO @premakkaraju 宣布，传奇电影制作人、技术创新者和视觉效果先驱 James Cameron 已加入 Stability AI 董事会。Cameron 的加入...</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917">来自 Colin Fraser (@colin_fraser) 的推文</a>: 我从未感到如此正确。引用 Colin Fraser (@colin_fraser)：如果它实际上看起来像这样呢？</li><li><a href="https://x.com/kateclarktweets/status/1838319202798538974?s=61">来自 Kate Clark (@KateClarkTweets) 的推文</a>: 独家：OpenAI 的竞争对手 Anthropic 已开始与投资者讨论融资事宜，这笔交易可能使该初创公司的估值达到 300 亿至 400 亿美元，大约是其融资估值的两倍...</li><li><a href="https://x.com/LucasAtkins7/status/1838593579305902217">来自 Lucas Atkins (@LucasAtkins7) 的推文</a>: 他在处理沉船方面确实很有经验。引用 Stability AI (@StabilityAI)：今天，我们的 CEO @premakkaraju 宣布，传奇电影制作人、技术创新者和视觉效果先驱 James Cameron 已加入 Stability AI 董事会...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1288209649262264351)** (16 条消息🔥): 

> - `发布编年史 (Release Chronicle)`
> - `Interconnects Artifacts`
> - `Late Fusion Visual LMs`
> - `GPT-4 性能` 


- **发布编年史查询**：一位成员询问是否存在一个记录发布者和发布时间的网站，得到的回复是 Twitter 和博客目前承担了这一功能。
   - 讨论引申出希望在博客上提供日历或数据视图，以便更高效地追踪信息。
- **Interconnects Artifacts 说明**：一位成员指出 *Interconnects Artifacts* 可能涵盖各种模型、数据集和系统，并暗指了 OpenAI 最近的公告。
   - 他们表示，Hugging Face 上列出的 Artifact 日志可以为不断演进的技术格局提供相关的见解。
- **Late Fusion Visual LMs 性能**：有人询问 Late Fusion Visual LMs 在文本基准测试中的表现，预期不会像 Tulu recipe 那样有显著提升。
   - 成员们对使用视觉模型时可能出现的性能退化表示担忧。
- **GPT-4 图像处理观察**：在实验 GPT-4 时，一位成员观察到该模型在处理图像输入时的路由方式与纯文本提示不同。
   - 该成员质疑这两种输入模式之间是否存在明显的智能差异。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1quMyI4BAx4rvcDfk8jjv063bmHg4RxZd9mhQloXpMn0/edit?usp=sharing">[18 April 2024] Aligning open language models</a>: Aligning open language models Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS25: Transformers United V4</li><li><a href="https://huggingface.co/collections/natolambert/2024-interconnects-artifacts-6619a19e944c1e47024e9988">2024 Interconnects Artifacts - a natolambert Collection</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1287863451028553860)** (26 条消息🔥): 

> - `新子域名公告`
> - `Python 工具 UV 介绍`
> - `Docker 与 UV 集成`
> - `UV 更新的 Cronjob`
> - `PyCharm 兼容性问题` 


- **新子域名引发关注**：关于新子域名的公告引起了热烈讨论，评论反映了其影响以及其中涉及的潜在酷炫因素。
   - 一位用户幽默地指出，这或许代表了脱离传统路径的一步。
- **Philpax 借助 UV 增强战力**：一位用户在表达了对 Python 类 Cargo 工具的需求后，向同事介绍了 **UV**，并声称介绍完后感觉“充满力量”。
   - 另一位成员分享了关于 UV 功能的全面总结（TL;DR），强调了其速度和易用性。
- **利用 UV 优化 Docker**：将 UV 与 Docker 集成非常简便，用户推荐使用官方 Dockerfile 并参考 [uv-docker-example](https://github.com/astral-sh/uv-docker-example) 等资源。
   - 一位用户建议创建一个 cronjob，以确保 **UV** 几乎每天都能更新，因为其发布频率很高。
- **UV 与 PyCharm 的兼容性**：一些用户对 **UV** 与 **PyCharm** 的兼容性表示担忧，并找到了像 [ryecharm](https://github.com/InSyncWithFoo/ryecharm) 这样的变通方案来缓解问题。
   - 共识是用户不应指望 JetBrains 会在短期内提供官方修复。
- **关于 Brew 与 Curl 安装方式的讨论**：关于使用 Brew 还是 Curl 安装 **UV** 展开了辩论，一位成员因其优势将 **UV** 比作 Python 界的 **pnpm**。
   - 讨论突显了在选择安装方法时的困惑，特别是对于 Mac 用户而言。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/karan4d/status/1838292114272325936?s=46">Tweet from mephisto (@karan4d)</a>: -be tencent -make gamegen diffusion model -say &#34;weights and paper soon&#34; on the GH repo -put out a github page showcasing the capability -announce to the world -delete everything  rugpulled aga...</li><li><a href="https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies>.">Running scripts | uv</a>: 暂无描述</li><li><a href="https://docs.astral.sh/uv/guides/integration/docker/#caching>)">Docker | uv</a>: 暂无描述</li><li><a href="https://x.com/colin_fraser/status/1838667677981904917?s=46">Tweet from Colin Fraser (@colin_fraser)</a>: I&#39;ve never been more vindicated  Quoting Colin Fraser (@colin_fraser)   What if it actually looks like this?
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1287852837749129216)** (96 条消息🔥🔥): 

> - `O1 Planning Capabilities` (O1 规划能力)
> - `World Simulator API Usage` (World Simulator API 使用)
> - `Hermes & Monad Dynamics` (Hermes & Monad 动态)
> - `Recent AI Model Upgrades` (近期 AI 模型升级)
> - `Nous Research and Merchandise` (Nous Research 与周边商品)


- **O1 Planning Capabilities Evaluation**: 一份关于 **O1** 规划能力的研究笔记已由团队成员提交至 [arXiv](https://arxiv.org/abs/2409.13373)，据报道他们为此熬夜工作。
   - 该摘要暗示对 **O1** 的能力进行了全面考察，并承诺在公开发布后提供更多细节。
- **World Simulator API Usage Discussion**: 成员们讨论了 **World Sim**，用户注册后可获得积分，API 调用会产生费用，强调了其易用性。
   - 一位用户鼓励创建账户以获取免费积分，并指出其 API 相关成本较低。
- **Hermes & Monad Showing Stubborn Behavior**: 有人担心 **Hermes** 和 **Monad** 在对话中变得固执且效率降低，特别是在其标记（tagging）能力方面。
   - 一位成员建议存在惩罚（presence penalty）可能会阻碍它们的交互，而另一位成员则指出基于托管方式的不同存在差异。
- **Latest AI Model Improvements**: **Gemini 1.5** 9 月升级和 **GPT-4o** 的小范围语音功能推出引发了兴奋，显示出市场的期待。
   - 社区渴望看到在即将举行的 **Meta Connect** 上展示的新进展，包括潜在的 AI 技术进步。
- **Nous Research and Merchandise Joke**: 展开了一场关于 **Nous** 被误认为服装店的轻松讨论，并开玩笑说周边商品（merch）驱动了收入。
   - 随后进行了澄清，**Nous** 是一家 AI 研究公司，尽管有这些幽默言论，团队成员仍致力于持续的研究工作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">worldsim</a>: 未找到描述</li><li><a href="https://x.com/replicate/status/1838346354206347730">来自 Replicate (@replicate) 的推文</a>: 我们正在开源我们的 Flux 代码。社区非常喜欢 FLUX.1 文本生成图像模型。为了服务他们，我们在内部进行了许多改进：图生图模式、NSFW 检查器，以及大多数 ...</li><li><a href="https://x.com/rao2z/status/1838245253171814419?s=46">来自 Subbarao Kambhampati (కంభంపాటి సుబ్బారావు) (@rao2z) 的推文</a>: 一份描述我们对 o1 🍓 规划能力评估的研究笔记现已发布在 @arxiv https://arxiv.org/abs/2409.13373（感谢 @karthikv792 & @kayastechly）。正如所承诺的，这里有一个摘要...</li><li><a href="https://console.cloud.google.com/">Google Cloud Platform</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch">NousResearch (NousResearch)</a>: 未找到描述</li><li><a href="https://github.com/NousResearch/finetuning-subnet">GitHub - NousResearch/finetuning-subnet</a>: 通过在 GitHub 上创建账户，为 NousResearch/finetuning-subnet 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1287856441696190486)** (25 条消息🔥): 

> - `llama.cpp 性能`
> - `Scaling LLMs`
> - `GPT-2 预训练`
> - `Sample packing`
> - `Tokenizers` 


- **llama.cpp 在 Turmex 上遇到困难**：重新构建 [llama.cpp](https://link.to/llama.cpp) 在使用 **3.1** 版本时仍会导致 **Turmex** 崩溃，尽管在 **3** 版本中运行良好。
   - 一位成员指出，他们的 **8GB RAM 手机** 无法有效运行 **Hermes-3-Llama-3.1-8B.Q4_K_M.gguf 模型**。
- **关于 Scaling LLMs 价值的辩论**：一位成员质疑如果 LLMs 不能通向 AGI，投入数十亿美元是否合理，并建议即使是像客户支持这样的基础任务也具有显著价值。
   - 其他人表示赞同，认为即使是初级模型也能在数据录入等岗位上节省人工成本。
- **关于 GPT-2 预训练的见解**：一位成员讨论了在 GPT-2 预训练中仅使用由独立句子组成的数据集的影响，对可能产生的次优结果表示担忧。
   - 另一位成员强调了 **sample packing** 的概念，警告称盲目混合序列可能会降低模型性能。
- **训练中有效的 masking**：讨论了在训练期间将未来 Token 的 Attention 分数归零的重要性，以确保序列的清晰隔离。
   - 有人建议添加像 'endoftext' 这样的特殊 Token 可能有助于提高清晰度，但对于学术目的而言并非严格必要。
- **Tokenizers 与特殊 Token**：有人提出了关于 Tokenizers 是否会自动向数据集添加特殊 Token 的问题，重点在于配置。
   - 成员们提到，虽然一些手动实现可能不包含此功能，但大多数现成的 **Hugging Face** Tokenizers 如果指定了，就会自动完成。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288109275520176148)** (2 条消息): 

> - `DisTrO 资源管理`
> - `RENDER 网络测试` 


- **DisTrO 在低带宽环境下表现出色**：初步论文表明 **DisTrO** 可在较差、非对称带宽和异构 GPU 设置下使用，使其适用于通过区块链进行资源管理。
   - *这是在非理想条件下优化资源分配的一个有前景的途径*。
- **在 RENDER 网络上测试 DisTrO**：一位成员提出 **RENDER 网络** 是否可以作为 DisTrO 的测试平台，因为它专门为不同条件下的资源管理而设计。
   - *利用 RENDER 可以为 DisTrO 的性能和可扩展性提供宝贵的见解*。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

ar02293: 嘿 www.keygunz.com 去测试一下
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1288109275520176148)** (2 条消息): 

> - `DisTrO 功能`
> - `RENDER 网络` 


- **DisTrO 增强带宽资源管理**：初步论文表明 **DisTrO** 可以在较差、非对称带宽和异构 GPU 环境下有效运行，从而通过区块链改进资源管理。
   - 这种能力使其成为应对多变网络条件和资源分配的潜在解决方案。
- **RENDER 网络作为测试场**：一位成员提出了使用 **RENDER 网络** 作为 DisTrO 测试平台的想法，称其专门为此类需求而设计。
   - 这一讨论为考虑利用 RENDER 现有的基础设施来验证 DisTrO 在挑战性条件下的性能铺平了道路。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1287851282652205200)** (100 条消息🔥🔥): 

> - `Qwen 2.5 模型问题`
> - `Unsloth Trainer 内存管理`
> - `微调模型`
> - `在 Ollama 中使用 Lora`
> - `模型的部署选项` 


- **Qwen 2.5 模型问题已解决**：多位成员报告了 **Qwen 2.5** 模型的问题，包括崩溃和 Bug，但讨论了如使用改进的模板或更改训练方法等解决方案。
   - 成员们确认他们已与 Qwen 团队合作，解决了与该模型相关的一些问题和 Bug。
- **管理 Unsloth Trainer 的内存占用**：一位用户在初始化 **UnslothTrainer** 时遇到了内存问题，建议通过减少数据集映射（dataset mapping）的进程数来解决该问题。
   - 他们成功降低了进程数并反馈了影响，表明平衡进程数量有助于初始内存映射。
- **分享微调过程见解**：一位用户分享了微调 **Vit_B16** 模型的经验，强调了高质量数据对于获得更好结果的重要性，而非单纯追求数据量。
   - 在初步获得良好的准确率后，他们还计划继续使用更多高质量图像来微调模型。
- **请求更新在 Ollama 中使用 Lora 的文档**：一位用户询问了关于在 **Ollama** 中使用 **Lora** 指南的更新版本，表示对最新流程感兴趣。
   - 回复指出近期变化不大，并强调了早期在该框架中使用 Lora 的成功案例。
- **微调模型的部署策略**：对话涉及了部署模型的有效方法，建议将 **Runpod** 和 **Llama Labs** 作为托管选项。
   - 一位用户对他们新微调的模型表示兴奋，并感谢社区在过程中的支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0">只需 3 步即可在 Ollama 中使用 Unsloth LORA 适配器</a>：使用 LLama.Cpp 将 Unsloth Lora 适配器转换为 GGML(.bin) 并在 Ollama 中使用 —— 仅需单张 GPU</li><li><a href="https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-">未找到标题</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/September-2024">发布 Qwen 2.5 支持 · unslothai/unsloth</a>：Qwen 2.5 支持已上线！Unsloth 已修复 Qwen 2.5 模型的一些问题！Kaggle 基础模型微调 Notebook：https://www.kaggle.com/code/danielhanchen/kaggle-qwen-2-5-unslo...</li><li><a href="https://youtu.be/UWF6dxQYcbU?feature=shared">只需 5 步，即可使用 Unsloth + Ollama 免费微调 AI 模型！</a>：你准备好训练自己的大语言模型 (LLM) 了吗，但觉得太复杂？再想想！在这段视频中，我将向你展示任何人如何...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fnvlla/qwen25_bugs_issues_fixes_colab_finetuning_notebook/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1287891074123305093)** (22 messages🔥): 

> - `Llama Model Quantization`
> - `Token Addition in Llama`
> - `Model Performance Insights`
> - `Feedback Mechanisms for Model Improvement`
> - `Memory Management in PyTorch` 


- **Llama3.1 的内存问题**：一位用户在尝试加载 4-bit 量化版本的 **Llama3.1** 时遇到了 **out of memory (OOM)** 错误，在 PyTorch 已占用 **14.75GB** 的情况下，难以分配 **20GB** 显存。
   - 社区建议检查模型配置并运行原始示例，以排查 OOM 问题。
- **向 Llama 词表添加新 Token**：一位用户询问如何通过 Unsloth 向 Llama 的词汇表添加新 Token，特别是这些 Token 是否会在 Ollama 进行 Tokenization 时被使用。
   - 这引发了关于此类添加是否需要特定推理引擎的讨论。
- **探索模型替代方案**：一位参与者建议探索 **Qwen 模型**，并根据个人测试强调了其在 Function Calling 方面的卓越性能。
   - 其他人也表示赞同，指出如果小型模型能满足用户需求且不消耗过多资源，使用小型模型会更有效率。
- **从过往案例中教学模型**：一位成员寻求关于如何利用过去对话中的错误（以评论作为反馈）来引导模型的建议。
   - 讨论涉及了 KTO 和 ORPO 等当前方法的挑战，强调需要结构化反馈来增强模型训练。
- **OpenAI 的反馈改进流程**：对话探讨了 OpenAI 如何利用 **RLHF (Reinforcement Learning from Human Feedback)**，根据用户评论和反馈来增强其模型。
   - 这引发了关于多轮对话（multi-turn conversations）在模型训练方法论中作用的更广泛讨论。


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1287858899772244093)** (90 messages🔥🔥): 

> - `New Anthropic Model Release`
> - `Perplexity Pro Features`
> - `Merlin Extension`
> - `User Experiences with Perplexity`
> - `Query Limits Discussion` 


- **Anthropic 新模型可能即将发布！**：有消息确认 Anthropic 预计很快会发布重大的 AI 模型升级，并强调这对开发者来说意义重大。完整细节将在禁令（embargo）解除后公布。
   - 成员们表现出极大的兴奋，推测这次升级的潜力及其在 AI 领域的影响。
- **对 Perplexity Pro 搜索功能的关注**：用户讨论了与 Perplexity Pro 账户相关的限制，特别是每天允许的搜索次数。一些用户不确定 Pro 账户的益处，并对最近的变化表示困惑。
   - 讨论指出 Perplexity Pro 提供了更个性化的搜索体验，能根据用户偏好更深入地挖掘主题。
- **关于 Merlin 扩展程序的见解**：成员们讨论了 Merlin 扩展程序，它允许用户直接在浏览器中与各种 LLM 聊天，并提供无限的高级模型访问权限。用户对比了 Merlin 和 HARPA AI 的功能及用户体验。
   - 用户赞赏 Merlin 的无限查询功能，但也指出其在模型设置方面缺乏透明度，这与 HARPA AI 的可定制功能形成对比。
- **用户对数据隐私表示担忧**：用户对即使没有账号也能访问旧搜索链接的保留问题表示担忧，这引发了隐私焦虑。一位用户报告称，此类链接可能会泄露个人信息，促使他们联系支持部门寻求澄清。
   - 讨论集中在用户登出状态下分享的链接是否应保持可访问，引发了对平台数据处理政策的质疑。
- **查询限制引发对话**：几位用户询问查询限制为何发生变化，特别是讨论了 o1 mini 等特定模型的低限制。大家承认，尽管限制较低，许多用户仍然能有效地管理查询而未达到上限。
   - 对话强调了用户对不断变化的限制的适应能力，以及他们在日常搜索活动中采用的应对策略。



**提到的链接**：<a href="https://x.com/rowancheung/status/1838280020642676802">Rowan Cheung (@rowancheung) 的推文</a>：我刚刚完成了一次关于新的重大 AI 模型升级的独家采访。可以确认，明天对开发者来说将是一个大日子。禁令解除的瞬间，我将在 X 上发布完整对话...

  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1287855962866061373)** (10 messages🔥): 

> - `Cosmic Ambassador` (宇宙大使)
> - `Superintelligence Age` (超级智能时代)
> - `Perplexity AI Differences` (Perplexity AI 的差异化)
> - `AI Impact on Education` (AI 对教育的影响)
> - `OpenAI Reasoning Probes` (OpenAI 推理探测)


- **探索卡尔·萨根：宇宙大使**：分享的一个链接指向 [Carl Sagan: Cosmic Ambassador](https://www.perplexity.ai/page/carl-sagan-s-pale-blue-dot-._z2e7N_TsiT92iP5IGiMQ)，讨论了萨根具有影响力的思想及其对地球在宇宙中地位的反思。
   - [来源线程](https://www.perplexity.ai/search/carl-sagan-s-theory-of-mind-be-QUYnNdrbT9CnRxX5MK7Q7w) 深入探讨了萨根与其标志性的“暗淡蓝点”演讲相关的心理理论。
- **超级智能时代深度解析**：一名成员分享了关于 [The Age of Superintelligence](https://www.perplexity.ai/page/the-age-of-superintelligence-5usatQeSSr68txXaWW3zfg) 的链接，探讨了高级 AI 对社会的潜在影响。
   - 讨论集中在 AI 发展过程中技术进步与伦理考量之间的平衡。
- **Perplexity AI 的差异化讨论**：成员们交流了关于 [Perplexity AI 有何不同](https://www.perplexity.ai/search/how-is-perplexity-ai-different-pljzBCICQzmrh0eMnujxUw?login-source=visitorGate) 的链接，强调了其独特的功能和能力。
   - 对话涵盖了用户交互和 AI 学习过程等方面。
- **AI 对教育的影响分析**：一名成员发布了一个链接，探讨 [AI 如何影响教育](https://www.perplexity.ai/search/describe-how-ai-impacts-educat-E_.Sazb0ReeNYu5PmyIQaQ)，分析了其在学习环境中的变革性作用。
   - 讨论围绕 AI 在教育设置中带来的潜在收益和挑战展开。
- **OpenAI 禁止推理探测 (Reasoning Probes)**：分享的新闻指出 OpenAI 已决定 [禁止推理探测](https://www.perplexity.ai/page/openai-bans-reasoning-probes-cJxbLXEgQPK7itY3LL2mJw)，引发了关于此举后果的辩论。
   - 成员们对禁令背后的逻辑及其对 AI 训练的潜在影响表达了各种观点。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1287957269178486929)** (9 messages🔥): 

> - `Citational Access Requests` (引用权限请求)
> - `API Rate Limits` (API 速率限制)
> - `Output Consistency` (输出一致性)
> - `Alternatives to PPLX` (PPLX 的替代方案)
> - `Exa.ai Exploration` (Exa.ai 探索)


- **引用权限请求未得到回应**：成员们对申请 **引用权限 (citational access)** 或更高的 **API rate limits** 未收到回复表示沮丧，一名成员提到在几个月内尝试了多次。
   - 另一名成员表达了类似的不满，称他们也给支持团队发了邮件但未获成功。
- **引用输出不一致阻碍自动化**：一名成员分享说，虽然他们可以通过提问获取引用，但遇到了 **输出不一致** 的问题，格式在 **HTML** 和 **Markdown** 之间交替。
   - 据报道，这种不一致性阻碍了他们的自动化流程，使得获得可靠输出变得相当困难。
- **探索 PPLX 的替代方案**：鉴于目前存在的问题，一名成员正在考虑将 **Exa.ai** 作为 **PPLX** 的可能替代方案，并指出它的功能更像是一个为其他 **LLMs** 服务的互联网搜索封装器 (wrapper)。
   - 他们强调需要一种能够支持随时间变化的特定领域搜索的解决方案，并认为 Exa.ai 可能会满足这一需求。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1287861393894277284)** (72 条消息🔥🔥): 

> - `在离线（Air-Gapped）机器上安装 LM Studio`
> - `LM Studio 中的模型支持`
> - `模型性能与处理`
> - `LongWriter 模型见解`
> - `即将发布的模型` 


- **在离线机器上安装 LM Studio**：成员们讨论了在离线（air-gapped）机器上安装 LM Studio 的可行性，提到需要从互联网分别下载安装程序和模型。
   - 强调了虽然安装过程不需要互联网，但为了确保功能正常，初始设置和文件传输到目标机器是必要的。
- **不支持的模型会导致错误**：当尝试加载像 **Flux** 这样的图像生成模型时，用户报告了由于 LM Studio 缺乏对此类模型支持而产生的错误。
   - 注意到目前不支持任何图像生成模型，这导致了对模型架构的混淆。
- **关于模型性能的担忧**：当对话接近 Token 限制时，性能问题被提出，用户观察到随着接近模型的 context length，速度会有所减慢。
   - 这种行为被描述为正常现象，归因于处理过程中的 VRAM 占用，并建议管理限制和预期。
- **LongWriter 模型能力**：关于 **LongWriter** 模型的讨论强调了其生成长文本的能力以及微调的潜力。
   - 鼓励成员进一步探索该模型，并提供了相关资源链接以深入了解其实现。
- **未来发布与增强**：提出了关于 **Pixtral** 等模型可用性的问题，预期主要围绕 **llama.cpp** 支持的就绪情况。
   - 对模型可用性和发布的变更进行了推测，社区见解为关于未来发展的持续对话做出了贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/THUDM/LongWriter">GitHub - THUDM/LongWriter: LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs</a>: LongWriter: 从长上下文 LLM 中释放 10,000+ 字的生成能力 - THUDM/LongWriter</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7107">tokenization: no double BOS tokens by JohannesGaessler · Pull Request #7107 · ggerganov/llama.cpp</a>: 相关讨论：#7062 llama.cpp tokenizer 目前无条件添加 BOS token。然而，我认为如果 prompt 已经以 BOS token 开头，不这样做会更有意义……</li><li><a href="https://github.com/abetlen/llama-cpp-python/issues/1501">Llama 3 Double BOS · Issue #1501 · abetlen/llama-cpp-python</a>: 前提条件：在提交 issue 之前，请自行回答以下问题。我正在运行最新代码。开发速度非常快，所以目前还没有标记版本。我……
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1287959860809564323)** (25 messages🔥): 

> - `AMD APU 上的 ROCm`
> - `双 GPU 设置兼容性`
> - `RTX 3090 的性能`
> - `GPU 市场的价格差异`
> - `欧盟的消费者保护` 


- **ROCm 不支持 AMD APU**：成员们确认 ROCm **不支持**像 **5700G** 这样的 AMD APU。有人指出，虽然内置 GPU 可以使用 Vulkan API，但由于它们共享系统 RAM，性能提升可以忽略不计。
   - 共识是，由于硬件限制，在 **APU** 上运行 ROCm 应用仍然是不切实际的。
- **关于双 GPU 设置的讨论**：一位成员询问 **LM Studio** 是否支持 **RTX 4070 Ti** 和 **RTX 3080** 的双 GPU 设置。其他人讨论了同时使用不同 Nvidia 显卡的理论，并指出了潜在的好处。
   - 有建议称，在尝试混合使用不同 GPU 型号之前，应考虑兼容性问题。
- **RTX 3090 性能预期**：人们对 **RTX 3090** 的预期 **TPS** 感到好奇，特别是在**推理训练**场景下。成员们推测了性能指标以及未来购置 GPU 的影响。
   - 成员们对使用像 **A770** 这样的额外 GPU 来处理更大的模型表现出了兴趣。
- **美欧之间的价格差异**：一位成员对欧盟 **GPU 价格较高** 表示沮丧，而美国市场的价格约为 **$750**。讨论强调了买家因地区定价而面临的挑战。
   - 有人指出，**VAT**（增值税）和其他税收导致了欧洲价格的虚高。
- **欧盟的消费者保护**：一位成员承认，虽然欧盟的科技产品价格较高，但也有诸如更优越的**保修覆盖**和支持等好处。讨论强调了确保用户获得顶级标准的消费者保护的重要性。
   - 北美和欧洲的对比揭示了在消费者**政府保护**方面的显著差异。



**提到的链接**：<a href="https://tenor.com/view/income-tax-tax-taxes-gif-11011288">Income Taxes GIF - Income Tax Tax Taxes - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1287904088855547924)** (64 messages🔥🔥): 

> - `Mojo 语言层级列表`
> - `Rust 的编译问题`
> - `NixOS 与包管理`
> - `MLIR vs LLVM`
> - `Mojo 的成长与社区` 


- **Mojo 语言层级列表排名**：一位用户分享了他们的个人语言层级列表，将 **Mojo** 排在首位，其次是 **C#**、**Rust** 等，并将其描述为一种主观感受而非逻辑排名。
   - 另一位用户建议根据项目情况区分 **C++** 类别，并强调了对简洁 C 互操作性的需求。
- **Rust 编译速度挑战**：用户对 **Rust 缓慢的编译时间** 表示沮丧，特别是在处理像 **40k 行代码的游戏** 这样的大型项目时，修改代码需要耗费大量时间。
   - 讨论强调了泛型是如何导致速度变慢的，建议包括优化 **Windows** 上的文件系统设置以提高性能。
- **探索 NixOS 作为替代方案**：对迁移到 **NixOS** 的兴趣引发了关于其优势（主要围绕其包管理器）的对话，但也有人对其项目复杂性提出了警告。
   - 用户讨论了使用 NixOS 复现系统的愿望，同时也权衡了对于较小规模设置使用 **Ansible** 等其他工具的潜在简便性。
- **MLIR 和 LLVM 特性对比**：一位成员询问为什么 **MLIR** 被认为优于 **LLVM**，解释集中在并行编译的改进和对高级语义的更好处理上。
   - 讨论指出，随着编译器的演进，使用 MLIR 不会丢失调试信息，这使其在某些语境下成为更好的选择。
- **Mojo 的发展与社区参与**：用户庆祝了 **Mojo** 的两周年纪念，回顾了在首个 **Mojo SDK** 发布等重大事件期间社区的增长和参与。
   - 成员们对 Mojo 的未来表达了兴奋之情，期待随着语言的成熟，它在未来几年将如何演进。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1287858470401474593)** (31 条消息🔥): 

> - `Mojo 类 vs Python 类`
> - `Mojo 中的 Monkey patching`
> - `Mojo 中模式匹配的未来`
> - `Mojo 与 C 之间以及与 Python 之间的通信速度`
> - `Mojo 中的元类 (Metaclasses) 和高级特性` 


- **Mojo 类因其类 Python 的行为而受到批评**：一位成员表示担心 **Mojo 类**不应模仿 **Python 类**的动态特性，建议由于性能问题，此类功能应保留给 Python。
   - 另一位成员强调，虽然有些人希望 **structs** 能实现类似类的功能，但许多人同意在专注于类实现之前，需要先建立 **trait 系统**。
- **对元类等高级特性的渴望**：一位成员幽默地提议在 Mojo 中加入 **metaclasses** 和 **monkey patching** 等高级特性，暗示了语言设计中一种有趣的混乱。
   - 关于结合 **mutable vs immutable** 反射路径以增强 Mojo 能力的讨论似乎非常激烈。
- **关于在 Mojo 中添加 'match case' 的讨论**：成员们表现出对 Mojo 在未来版本中可能采用 **'match case'** 语句语法的兴趣，这与 Python 3.10 的特性保持一致。
   - 对话还提到，引入 **sum types/enums** 对于实现更高级的模式匹配是必要的，二者缺一不可。
- **通信效率比较**：有人提出了关于 **Mojo 与 C**（通过 **DLHandle**）之间的通信是否比其与 **Python** 的通信更快的问题。
   - 虽然有人提出了推测，但尚未得出确切结论，并指出性能可能取决于 **Python** 如何与 **C** 交互。
- **对 Mojo 中类的分歧观点**：成员们对 Mojo 中 **classes** 的必要性持不同意见，一些人主张采用类式接口，而另一些人则满足于更底层的构造。
   - 讨论显示，一些人支持保持与 Python 动态特性的清晰分离，倾向于专注于更基础的特性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/main/proposals/mojo-and-dynamism.md">mojo/proposals/mojo-and-dynamism.md at main · modularml/mojo</a>: Mojo 编程语言。欢迎在 GitHub 上为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/459">Implement Pattern Matching as expression instead of statement in Mojo · modularml/mojo · Discussion #459</a>: 你是否考虑将模式匹配实现为表达式，以摆脱 C 语言的遗产（大量导致糟糕/可变代码的 return 语句），并使 Mojo 成为具有函数式特性的现代语言...</li><li><a href="https://github.com/modularml/mojo/issues/3534">[Historical Discussion] Mojo and Dynamism · Issue #3534 · modularml/mojo</a>: 在 #466 中讨论，最初由 Mogball 于 2023 年 7 月 20 日发布。Mojo 的宏伟目标是成为像 Python 一样简单、强大且易于使用的语言，但同时具备允许程序员重新...</li><li><a href="https://m.youtube.com/watch?v=sPiWg5jSoZI&pp=ygUYRGF2aWQgbWV0YWNsYXNzZXMgcHl0aG9u">Python 3 Metaprogramming</a>: David Beazley。Python 3 中一些最重要的变化与元编程有关。在本教程中，我将介绍装饰器、类装饰器、des...
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[announcements](https://discord.com/channels/1161519468141355160/1209871299854336060/1287887683976433705)** (2 messages): 

> - `DSPy 2.5.0 release`
> - `Migration to LiteLLM`
> - `Deprecation of pre-2.4 LM clients`
> - `Feedback solicitation`
> - `Upcoming changes` 


- **DSPy 2.5.0 低调发布**：**DSPy 2.5.0** 已经发布，鼓励用户在正式公告发布前分享反馈。
   - 此版本弃用了所有 2.4 版本之前的 LM 客户端，包括来自 OpenAI 的客户端。
- **迁移变得简单**：用户可以在约 **3 分钟**内完成 [迁移](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb)，从而显著提升程序的质量。
   - 这对于聊天 LM 和复杂的 signature 特别有益，确保了更好的一致性。
- **反馈至关重要**：此次发布有意保持低调，依靠 **弃用警告 (deprecation warnings)** 来通知用户，因为反馈对于后续的调整至关重要。
   - 开发人员强调，在未来几天对新版本进行微调时，他们非常欢迎任何建议。
- **通过新的 Adapter 层实现一致的质量**：通过使用 `dspy.LM`，用户的 DSPy 模块将通过 **配置好的 Adapter** 进行路由，默认情况下为 `dspy.ChatAdapter`。
   - 这一改进旨在提升用户体验以及在各种用例中的适应性。
- **令人兴奋的更新即将到来**：在接下来的 **10-15 天**内，用户可以期待一系列的更新和增强。
   - 根据开发团队的计划，这段时间可能会带来许多有价值的变更。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>：了解如何在 LiteLLM 上部署并调用来自不同提供商的模型</li><li><a href="http://localhost:{sglang_port}/v1")">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1287977404068790285)** (2 messages): 

> - `DSPy powered AI code assistant`
> - `Live coding session` 


- **基于 DSPy 的 AI 助手实时编程**：一位成员宣布了预定于 **PST 时间上午 9 点（GMT 时间下午 4 点）** 进行的实时编程环节，旨在构建第一个 **基于 DSPy 的 AI 代码助手**。
   - 他们鼓励其他人加入该环节，以深入了解这一创新工具的开发过程。
- **DSPy 代码助手开发启动**：实时编程环节在指定频道开始，重点是 DSPy 代码助手的设置。
   - 鼓励参与者积极互动并跟随操作，以获得构建过程的实战经验。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1288149512929546250)** (1 messages): 

> - `Chain-of-thought (CoT)`
> - `Performance Benefits of CoT`
> - `Quantitative Meta-Analysis of CoT` 


- **CoT 方法论评估**：最近的一篇论文展示了一项定量元分析，涵盖了超过 **100 篇** 使用 Chain-of-thought (**CoT**) 提示词的论文，分析了其在 **14 个模型** 中的有效性。
   - 研究强调，**CoT 对涉及数学或逻辑任务的益处** 显著高于其他任务类型，并提出了最佳使用策略。
- **MMLU 上的 CoT 与直接回答对比**：在分析 **MMLU** 时发现，不使用 CoT 直接生成答案的准确率与使用 CoT 几乎相同，尤其是在处理 **符号运算** 时。
   - 性能差异主要出现在问题或回答包含 **等号** 时，这表明了对推理的需求。
- **CoT 中的规划与执行**：论文将 CoT 任务中的 **规划与执行** 进行了分离，提供了关于 CoT 与工具增强型 LLM 相比如何运作的见解。
   - CoT 带来的大部分性能提升源于更好的规划，突显了该方法论的细微差别。



**提到的链接**：<a href="https://arxiv.org/abs/2409.12183">To CoT or not to CoT? Chain-of-thought 主要在数学和符号推理方面提供帮助</a>：通过提示词进行的 Chain-of-thought (CoT) 是从大语言模型 (LLM) 中激发推理能力的事实标准方法。但这种额外的“思考”究竟适用于哪类任务...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1287888652550275193)** (84 条消息🔥🔥): 

> - `DSPy 2.5.0 Release`
> - `会议反馈`
> - `LLM 自定义适配器`
> - `多模态能力`
> - `DSPy 中的缓存控制` 


- **DSPy 2.5.0 发布引发热烈反响**：DSPy 2.5.0 的发布正式宣布，承诺将快速解决 **50-100** 个问题。成员们对新特性以及正在创建的入门级 Notebook 表示了极大的热情。
   - 有提议指出，可以通过公开周会来收集进一步的反馈。
- **自定义适配器实现查询**：讨论围绕创建自定义适配器以指定 LLM 调用的额外参数展开，例如用于结构化输出的 `grammar`。成员们分享了以往实现的经验，并希望在新的 `dspy.LM` 结构下有更清晰的最佳实践。
   - 另一位用户确认了向其 LLM 传递额外参数的需求，这体现了功能适配这一共同主题。
- **对多模态特性的关注**：对即将推出的 DSPy 多模态能力的期待正在升温，预计将于下周上线。用户询问了其与各种模型类型的兼容性，包括像 **Ultravox** 这样的音频 LM。
   - 回复澄清说，初始版本将集中在视觉语言模型（VLMs）上，除非其他模型的接口结构类似。
- **DSPy 中的缓存管理**：用户询问如何有效管理缓存，以便在没有缓存效应的情况下衡量真实的推理速度。官方澄清，较新的实现允许通过将环境变量（如 `DSP_CACHEBOOL`）设置为 false 来控制缓存。
   - 这一功能对于在缓存会影响结果的任务中评估性能至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/fullstackwebdev/dc0f4e97df7591ade63f83d27668fe25">XMLAdapter</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb">dspy/examples/migration.ipynb at main · stanfordnlp/dspy</a>: DSPy：编程而非提示基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://gist.github.com/fullstackwebdev/ddf21d55cef58a40471e8925834e6531">test_chat_adapter.py</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/fixie-ai/ultravox">GitHub - fixie-ai/ultravox: A fast multimodal LLM for real-time voice</a>: 用于实时语音的快速多模态 LLM。通过在 GitHub 上创建账户为 fixie-ai/ultravox 的开发做出贡献。</li><li><a href="https://marimo.io/">marimo | a next-generation Python notebook</a>: 使用 marimo（下一代 Python notebook）无缝探索数据并构建应用。</li><li><a href="https://github.com/stanfordnlp/dspy/issues/390#issuecomment-1947542304">[WIP] Major refactor roadmap  · Issue #390 · stanfordnlp/dspy</a>: DSPy 拥有少量（约 5-6 个）极其强大的概念，这些概念在过去一年作为开源项目有机增长。在内部，是时候进行一次重大的重构以简化……
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1287930340815667241)** (2 条消息): 

> - `GROQ API 集成`
> - `Chain of Thought 评估` 


- **GROQ API Key 设置**：一位用户提供了设置 **GROQ_API_KEY** 并运行必要 Python 代码以使用模型的指令：`lm = dspy.LM('groq/llama3-8b-8192')`。
   - 此设置旨在方便在 **Llama 3** 模型中使用 **dspy** 库。
- **Chain of Thought 示例执行**：提供的代码片段包含了一个使用简单数学问题“what is 2+2?”演示 **Chain Of Thought** 功能的示例。
   - 该示例展示了如何利用 dspy 框架的功能执行查询。
- **询问结果**：另一位用户对前一个问题的结果表示好奇，询问是否找到了答案。
   - 这反映了对讨论已实现示例结果的持续兴趣和参与。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1287867352750362748)** (1 条消息): 

> - `Lecture 3`
> - `Agentic AI Frameworks`
> - `AutoGen`
> - `Multimodal Knowledge Assistant` 


- **关于 AI Frameworks 和多模态助手的今日课程**：本课程的 **第 3 讲** 将于今日 **太平洋标准时间 (PST) 下午 3:00** 举行，届时将进行 [这场直播](https://www.youtube.com/live/OOdtmCMSOo4)，并邀请两位重磅嘉宾。
   - Chi Wang 将讨论 **Agentic AI Frameworks 与 AutoGen**，而 Jerry Liu 将涵盖构建 **生产级多模态知识助手** 的步骤。
- **Chi Wang 论 Agentic AI 设计考量**：Chi Wang 的演讲将探讨 Agentic AI 编程框架的 **核心设计考量**，并重点关注 **AutoGen**、其应用以及最新的研究进展。
   - 演讲最后将讨论关于未来 AI 应用和赋能开发者的开放性问题。
- **Jerry Liu 对多模态 AI 流水线的见解**：Jerry Liu 的环节将概述 **多模态知识助手** 的逐步开发过程，讨论用于研究目的的 **高级 RAG 流水线**。
   - 它将整合 **结构化输出 (structured outputs)**、**Agentic 推理 (agentic reasoning)** 和 **事件驱动工作流 (event-driven workflows)** 等元素，以创建一个高效的 Agent 系统。
- **课程助教联系方式**：如有任何疑问，鼓励参与者通过指定的 **Discord 频道** 联系课程助教。
   - 这提供了解决有关课程材料和讲座问题的直接途径。



**提到的链接**：<a href="https://www.youtube.com/live/OOdtmCMSOo4">CS 194/294-196 (LLM Agents) - Lecture 3, Chi Wang and Jerry Liu</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1287854038091825315)** (33 条消息🔥): 

> - `课程出勤问题`
> - `客座讲师请求`
> - `测验链接`
> - `开源 Embedding 模型`
> - `AutoGen 的应用` 


- **关于课程出勤的澄清**：直播的出勤表仅供 Berkeley 学生填写，MOOC 学生不应填写，这引起了一些混乱。
   - *下次在展示二维码时会进行更清晰的说明*，以避免此类误解。
- **客座讲师建议**：有人请求邀请未来的客座讲师，如来自 Orkes.io 的 Viren 和来自 Traversaal.ai 的 Hamza Farooq。
   - 课程助教对创建这些建议的反馈表持开放态度，欢迎提供请求该讲师的详细原因。
- **查找测验链接**：课程的测验链接发布在课程网站的教学大纲 (syllabus) 部分，测验通常在每场讲座后一到两天发布。
   - 成员们讨论了在哪里可以找到这些链接，以确保可访问性。
- **开源 Embedding 模型的现状**：目前认为最好的开源 Embedding 模型是 [jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)，由 Jina AI 开发，提供多语言支持。
   - 该模型具有 *Task LoRA* 功能，可增强神经搜索应用的性能。
- **AutoGen 应用的真实案例**：成员们分享了使用 AutoGen 构建的复杂应用示例，包括一个用于古董电脑的机器人和一个医疗相关的应用 [hospitalgpt](https://github.com/micklynch/hospitalgpt)。
   - 社区表示有兴趣看到使用 AutoGen 开发的更多复杂软件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>：未找到描述</li><li><a href="https://github.com/micklynch/hospitalgpt">GitHub - micklynch/hospitalgpt</a>：通过在 GitHub 上创建账号来为 micklynch/hospitalgpt 的开发做出贡献。</li><li><a href="https://github.com/lamm-mit/SciAgentsDiscovery">GitHub - lamm-mit/SciAgentsDiscovery</a>：通过在 GitHub 上创建账号来为 lamm-mit/SciAgentsDiscovery 的开发做出贡献。</li><li><a href="https://github.com/emooreatx/ccmp_ai">GitHub - emooreatx/ccmp_ai: Classic/Retro Computing LLM bot</a>：经典/复古计算 LLM 机器人。通过在 GitHub 上创建账号来为 emooreatx/ccmp_ai 的开发做出贡献。</li><li><a href="https://huggingface.co/jinaai/jina-embeddings-v3">jinaai/jina-embeddings-v3 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1287881743675953286)** (23 messages🔥): 

> - `Q&A 讨论`
> - `技术设置延迟`
> - `提到 Zinley 项目`
> - `AutoGen 代码详情`
> - `与 AlphaGo 对弈国际象棋` 


- **Q&A 环节需要保留**：参与者表示希望活动中的 **Q&A 环节** 不要被缩减，一名成员保证会就此询问工作人员。
   - *如果演讲者能重复问题就太好了*，以确保讨论过程中的清晰度。
- **技术设置延迟仍在继续**：成员们注意到在初始阶段反复出现延迟，特别是与 **视听（AV）设置** 相关的延迟，一名成员幽默地评论说，前 20 分钟通常都是关于 **AV 设备** 的事。
   - 设置暂时中断，导致看不到 **Feed 源**，直到据报道它恢复为止。
- **Zinley 项目引发好奇**：一名成员询问了提到的一个项目，引发了关于 **Zinley** 的讨论，这是一个旨在为各种技能水平的用户简化软件创作的友好型 AI。
   - 他们分享了关于 **Zinley** 使命的见解，旨在快速将想法转化为软件，正如其 [官网](https://zinley.com/about.html) 所强调的那样。
- **关于正在编译的 AutoGen 详情**：成员们讨论了正在编译的代码可能与 **AutoGen** 有关，特别是能够执行代码并与其他 Agent 交互的 **User Proxy Agent**。
   - 该 Agent 的文档（包含各种功能）可以在 [这里](https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/) 找到。
- **AlphaGo 轻松击败象棋选手**：一名成员对使用围棋 Agent 挑战 **AlphaGo/AlphaZero** 的好奇引发了简短的交流，另一名成员断言 **AlphaGo 会大获全胜**。
   - 这突显了 **AlphaGo** 在策略游戏领域对抗类人竞争对手时的 **统治地位**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/autogen/docs/reference/agentchat/user_proxy_agent/">agentchat.user_proxy_agent | AutoGen</a>: UserProxyAgent</li><li><a href="https://zinley.com/about.html">Zinley | 让每个人都能轻松创作软件</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1287923906337968230)** (33 messages🔥): 

> - `搜索/检索技术`
> - `AutoGen 对比 CrewAI`
> - `O1 API 使用`
> - `多 Agent 协作`
> - `面向 Agent 的高级 DSL` 


- **用于 RAG 的搜索/检索技术**：一名成员建议关注与信息检索相关的 **经典 NLP 技术**，包括 **排序算法** 和 **语义理解**，以改进 RAG 中的 R 部分。
   - 他们强调了掌握 **索引（Indexing）** 和 **相关性（Relevance）** 在增强搜索能力方面的重要性。
- **AutoGen 的定制化 vs. CrewAI 的速度**：一名成员探索了用于多 Agent 协作的 **AutoGen** 和 **CrewAI**，发现 AutoGen 的定制化程度更高，而 CrewAI 在快速原型设计方面表现出色。
   - 成员们对 **CrewAI** 在 Agent 间进行往复通信方面的局限性表示担忧，而这是 AutoGen 的 **ConversableAgent** 所提供的功能。
- **O1 API 调用考量**：一名成员质疑 **O1 API** 是否仍适用于 AutoGen 框架（由于内部 Agent 的采用），推测这可能会增加推理时间。
   - 他们指出，虽然 **O1-mini** 应该用于编程任务，但尝试使用 **O1-preview** 作为对抗性 Agent 或规划器（Planner）可能会产生一些见解。
- **对比 O1-mini 和 O1-preview**：在讨论中，有人指出 **O1-preview** 对于专门的复杂任务可能很实用，尽管它与 **O1-mini** 之间存在奇怪的性能差异。
   - 参与者指出了由于其 **黑盒（Black Box）** 性质而在评估方面面临的挑战，同时也承认了 **O1-mini** 在编程任务中的潜力。
- **关于多 Agent 系统 DSL 的讨论**：有人呼吁建立一种用于定义多 Agent 系统的 **通用 DSL**，以避免图论的复杂性，旨在实现 AI 中的实验哲学。
   - 成员们认同这种 DSL 面临的挑战，但也认识到它可以帮助增强 Agent 之间的协作。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1287887476391940247)** (74 messages🔥🔥): 

> - `Letta AI`
> - `Gemini 模型更新`
> - `语音功能推出`
> - `客户服务 Agent 实验`
> - `HuggingChat App 发布`

- **Letta AI 脱离隐身模式**：由创始人 [Sarah Wooders](https://x.com/sarahwooders) 和 [Charles Packer](https://x.com/sarahwooders/status/1838261104864346288?s=46) 创立的 [Letta AI](https://x.com/sarahwooders/status/1838261104864346288?s=46) 公司正式亮相，该公司专注于开发有状态的 LLM Agent。他们目前正在旧金山积极招聘并组建团队。
   - 在 [TechCrunch](https://techcrunch.com/2024/09/23/letta-one-of-uc-berkeleys-most-anticipated-ai-startups-has-just-come-out-of-stealth/) 阅读更多关于 Letta 的信息。
- **Gemini 模型增强**：[Gemini 模型](https://x.com/OfficialLoganK/status/1838611055217385646) 迎来了重大更新，包括速率限制翻倍以及 Gemini 1.5 Pro 价格降低 50% 以上。过滤器已切换为选择性加入（opt-in），并发布了更新后的 Flash 8B 实验性模型。
   - 开发者对这些变化持乐观态度，认为这是开发者的黄金时期，详见 [Google Developers Blog](https://developers.googleblog.com/en/updated-production-ready-gemini-models-reduced-15-pro-pricing-increased-rate-limits-and-more/)。
- **语音功能推出**：OpenAI 宣布 [Advanced Voice](https://x.com/openai/status/1838642444365369814?s=46) 正在向 ChatGPT 应用中的 Plus 和 Team 用户推广，引入了多项新功能并改进了口音。值得注意的是，它可以用 50 多种不同的语言进行表达。
   - 然而，正如 [OpenAI](https://x.com/OpenAI/status/1838642453391511892) 的公告所强调的，该功能在几个欧洲国家尚不可用。
- **客户服务 Agent 实验**：关于管理 Agent 模拟中多轮对话挑战的讨论揭示了维持有效用户交互的重要见解。建议包括实施阶段标记（stage markers）和设定明确的对话终止指南。
   - 用户正在探索各种将强化学习（reinforcement learning）集成到对话管理中的方法，以提升客户 Agent 的体验。
- **HuggingChat macOS 应用介绍**：新发布的 [HuggingChat](https://x.com/cyrilzakka/status/1838618605648490974?s=61) macOS 应用提供了开源 LLM 的原生集成，具有 Markdown 支持和网页浏览等功能。这标志着直接在桌面端使用用户友好型 AI 工具迈出了重要一步。
   - 该应用展示了增强 AI 驱动应用程序的可访问性和功能的趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/hughbzhang/status/1838288923656941860">来自 Hugh Zhang (@hughbzhang) 的推文</a>：OpenAI 最近发布了 o1 系列模型，并展示了一张关于测试时计算（test-time compute）的 Scaling Laws 图表——遗憾的是没有标注 x 轴。仅使用公开的 o1-mini API，我尝试重建了...</li><li><a href="https://x.com/OfficialLoganK/status/1838611055217385646">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：两个新的生产级 Gemini 模型，超过 2 倍的速率限制，Gemini 1.5 Pro 降价超过 50%，过滤器改为选择性加入（opt-in），更新了 Flash 8B 实验模型等。对于...来说是个好日子。</li><li><a href="https://x.com/hughb">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/sarahwooders/status/1838261104864346288?s=46">来自 Sarah Wooders (@sarahwooders) 的推文</a>：很高兴宣布 @Letta_AI，这是我和 @charlespacker 共同创立的公司，旨在构建有状态的 LLM Agent。我们正在旧金山组建一支优秀的（线下）团队，并积极招聘创始...</li><li><a href="https://x.com/cyrilzakka/status/1838618605648490974?s=61">来自 Cyril Zakka, MD (@cyrilzakka) 的推文</a>：很高兴发布 HuggingChat 💬 - 一个原生的 macOS 应用，将强大的开源语言模型直接带到你的桌面 - 支持 Markdown、网页浏览、代码语法高亮等...</li><li><a href="https://arxiv.org/abs/2402.01662">生成式幽灵：预见 AI 来世的益处与风险</a>：随着 AI 系统在性能的广度和深度上迅速提升，它们有助于创建日益强大且逼真的 Agent，包括以特定人物为原型的 Agent 的可能性...</li><li><a href="https://x.com/natolambert/status/1837232801235755174">来自 Nathan Lambert (@natolambert) 的推文</a>：这段较长的 o1 视频中值得注意的事项（不多）：1. “带有 RL 的模型比人类更擅长发现新的 CoT 步骤” 2. “自我批判的出现是一个强大的时刻” 3. 提到了一段文字...</li><li><a href="https://forecast.safe.ai/?id=66e09f58718ead7507890d82">FiveThirtyNine | 预测 AI</a>：未找到描述</li><li><a href="https://x.com/anushkmittal/status/1837233399209283762">来自 anushk (@anushkmittal) 的推文</a>：@natolambert 有趣。AI 不仅仅是文本生成，它关乎构建能够理解世界并与之互动的 Agent。</li><li><a href="https://x.com/Smol_AI/status/1838663719536201790">来自 AI News by Smol AI (@Smol_AI) 的推文</a>：值得注意的是 Lmsys Elo 与价格曲线的预测性，以及该策略是如何奏效的。今天 Gemini Pro 的降价使其正好符合对数线性定价曲线...</li><li><a href="https://x.com/rohanpaul_ai/status/1838171186229858677?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：Priompt 是 @anysphere 内部使用的 Prompt 设计库，该公司是 @cursor_ai 背后的团队。“Prompting 应该被称为 Prompt 设计。Prompting 就像与一个受时间限制的人类交流...”</li><li><a href="https://x.com/OpenAI/status/1838642453391511892">来自 OpenAI (@OpenAI) 的推文</a>：Advanced Voice 尚未在欧盟、英国、瑞士、冰岛、挪威和列支敦士登提供。</li><li><a href="https://x.com/KateClarkTweets/status/1838319202798538974">来自 Kate Clark (@KateClarkTweets) 的推文</a>：独家：OpenAI 的竞争对手 Anthropic 已开始与投资者讨论融资事宜，该交易可能使这家初创公司的估值达到 300 亿至 400 亿美元，估值几乎翻倍...</li><li><a href="https://x.com/shishirpatil_/status/1837205152132153803">来自 Shishir Patil (@shishirpatil_) 的推文</a>：📣 宣布 BFCL V3 - 评估 LLM 如何处理多轮和多步函数调用（function calling）！🚀 对于 Agent 系统，函数调用至关重要，但模型需要做的不仅仅是单轮任务...</li><li><a href="https://x.com/miramurati/status/1838642696111689788">来自 Mira Murati (@miramurati) 的推文</a>：ChatGPT 中的所有 Plus 和 Team 用户。引用 OpenAI (@OpenAI)：“Advanced Voice 将在这一周内向 ChatGPT 应用中的所有 Plus 和 Team 用户推出。在你们耐心等待的同时，我们...”</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准。</li><li><a href="https://x.com/openai/status/1838642444365369814?s=46">来自 OpenAI (@OpenAI) 的推文</a>：Advanced Voice 将在这一周内向 ChatGPT 应用中的所有 Plus 和 Team 用户推出。在你们耐心等待的同时，我们增加了 Custom Instructions、Memory、五种新声音...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb">stanfordnlp/dspy 仓库中的 dspy/examples/migration.ipynb</a>：DSPy：用于编程（而非 Prompting）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://arstechnica.com/information-technology/2024/09/my-dead-">我的已故...</a>

father-is-writing-me-notes-again/?utm_source=changelog-news">我去世的父亲又在给我“写”便条了</a>：最近的一项 AI 发现复活了我已故父亲的手写体——我希望任何人都能使用它。</li><li><a href="https://arstechnica.com/information-technology/2024/">2024 | Ars Technica</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1287883506957156363)** (29 messages🔥): 

> - `Cohere AI`
> - `Aya initiative`
> - `工作焦虑`
> - `测试假设`
> - `Chain of Thought (COT)` 


- **新人分享对 Cohere AI 的兴趣**：成员如机械工程专业的学生 **Nav** 表达了学习 **Cohere** 和 AI 的兴趣，而 **Sanjeev** 则在寻求相关博客或视频的指导。
   - 作为回应，分享了 [Aya Research](https://cohere.com/research/aya) 的链接，介绍了旨在推进多语言 AI 的倡议。
- **解决工作相关的担忧**：成员 **Milansarapa** 对财务状况和开始新工作表示紧张，得到了其他成员关于已有合同的安慰。
   - *你已经拿到合同了*，这缓解了恐惧并加强了社区支持。
- **LLM 中的假设测试**：**Milansarapa** 询问不同 LLM 之间的相似结果是否表明关于 Recursive Iterative 模型的假设测试成功。
   - **mrdragonfox** 建议使用 benchmarks 和 evaluation harnesses 进行更准确的测试，并探索不同的主题。
- **探索不同的方法论**：**Milansarapa** 讨论了审查答案相似性以及使用他们的方法探索不同主题和 LLM 的必要性。
   - **mrdragonfox** 强调了通过系统测试实现更准确答案的重要性。
- **理解 Chain of Thought (COT)**：**Milansarapa** 询问了 COT 的概念，促使其他成员解释其在改进某些问题解决方法中的功能。
   - '**Chain of Thought**' 指的是一种可以增强某些任务性能的策略，尽管并非普遍适用。



**提到的链接**：<a href="https://cohere.com/research/aya">Aya</a>：Cohere 的非营利研究实验室 C4AI 发布了 Aya 模型，这是一个最先进的、开源的、大规模多语言研究 LLM，涵盖 101 种语言——包括 50 多种以前服务不足的语言...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1287883672674111560)** (8 messages🔥): 

> - `服务器位置`
> - `在 Javascript 中使用 Single Step Tools` 


- **关于用户服务器位置的讨论**：一位成员提到了服务器分布在多个位置，询问是否可以为英国用户确认托管位置。
   - 另一位成员建议，如果存在国家边界限制，可以使用 AWS 或 Vertex 来选择合适的区域。
- **在 Javascript .chatStream() 中使用 Single Step Tools**：一位成员询问关于在 Javascript 方法 .chatStream() 中使用 Single Step Tools 的问题。
   - 回复中强调了一个参数 **force_single_step=True**，可用于促进单步操作，尽管其公开可用性尚不确定。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1288083275289464882)** (5 messages): 

> - `多语言 Reranker 问题`
> - `Embedding 模型选择`
> - `Reranker 最佳实践` 


- **多语言 Reranker 在处理外语时遇到困难**：一位用户报告称，Reranker 的**多语言质量**导致在**波兰语**等语言中的相关性得分较低，即使有潜在有用数据也会被过滤掉。
   - *相关性得分太低以至于被过滤掉*，使得 Reranker 在他们的用例中失效。
- **使用 ada_2 模型进行 Reranking**：团队提到他们正在使用 OpenAI 的 **ada_2** 模型进行 Reranking 测试，并提供了诸如 *'what are the working hours?'* 之类的示例查询来说明他们的实现。
   - 他们分享了 **rerank-multilingual-v3.0** 和 **rerank-english-v3.0** 模型作为其测试设置的一部分。
- **强调顶部结果优于分数**：一位成员强调，在使用 Reranker 时，关注 **top n 结果** 比关注用于过滤数据块的相关性得分更重要。
   - 他们建议将 100 个文档的 **top_n** 定义为 1 或 3，而不考虑相关性得分，以便首先查看最相关的块。
- **多语言数据集的最佳实践**：提供了关于多语言数据集最佳实践的建议，推荐使用 **multilingual rerank v3.0** 以更好地处理各种语言。
   - 他们指出，如果不需要相关性得分，则不应使用它们，因为这可以简化查询。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1288186748253507685)** (2 messages): 

> - `自我推广相关问题`
> - `Cohere 在嵌入式系统中的应用` 


- **不欢迎自我推广**：一名成员强调该频道不是进行自我推广的地方，要求其他人删除其链接。
   - “这里不是为你自己做广告的地方”是表达的核心观点。
- **关于 Cohere 在嵌入式系统中的咨询**：一名成员询问是否有在 **embedded systems** 中使用 **Cohere** 的示例。
   - 该问题表明了对 Cohere 技术在典型用例之外的实际应用的兴趣。


  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1288193044750930014)** (1 messages): 

> - `Cohere Toolkit 更新`
> - `聊天功能`
> - `文件类型支持`
> - `用户反馈`
> - `团队协作` 


- **Cohere Toolkit 迎来令人兴奋的更新**：本月，**Cohere Toolkit** 修复了多个后端/UI 问题，提升了整体用户体验。
   - 显著的新功能包括置顶/取消置顶聊天、重新生成最后一条聊天机器人回复，以及支持 **parquet** 和 **tsv** 文件，并提供了 [YouTube 演示视频](https://youtu.be/gdJ0abx9mvo)。
- **新聊天功能让交流更轻松**：用户现在可以在 **Cohere Toolkit** 中轻松置顶/取消置顶聊天，便于更好地管理对话。
   - 此外，还添加了重新生成最后一条聊天机器人回复的功能，允许对讨论进行快速跟进。
- **引入对多种文件类型的支持**：**Cohere Toolkit** 的最新更新现在支持 **parquet** 和 **tsv** 文件格式，增强了数据处理能力。
   - *这一新支持为处理不同数据结构和格式的用户开辟了更多可能性。*
- **鼓励用户反馈以促进进一步开发**：欢迎用户分享他们的反馈和新想法，以继续改进 **Cohere Toolkit**。
   - *开发团队感谢社区的投入与协作，并对讨论和代码审查表示感谢。*



**提到的链接**：<a href="https://youtu.be/gdJ0abx9mvo)">Cohere Toolkit demo 09.2024</a>：未找到描述

  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1288179701696696421)** (1 messages): 

> - `James Cameron`
> - `Stability AI 董事会`
> - `变革视觉媒体`
> - `Generative AI`
> - `电影技术` 


- **James Cameron 加入 Stability AI 董事会**：首席执行官 Prem Akkaraju 宣布，传奇电影制作人 **James Cameron** 已加入 **Stability AI 董事会**。这一加入标志着 Stability AI 在变革视觉媒体使命中的关键举措。
   - Cameron 在将**尖端技术**与叙事相结合方面的经验，将增强 Stability AI 为创作者构建全面 AI pipeline 的努力。
- **Cameron 对电影技术的影响**：作为视觉特效的先驱，Cameron 以《终结者》（The Terminator）和《阿凡达》（Avatar）等电影闻名，不断推向电影技术的边界。他独特的视角与 Stability AI 专注于技术进步与创意融合的方向高度契合。
   - 通过加入 Stability AI，Cameron 旨在通过针对视觉媒体的创新 AI 解决方案进一步**革新叙事方式**。



**提到的链接**：<a href="https://stability.ai/news/james-cameron-joins-stability-ai-board-of-directors">奥斯卡获奖导演 James Cameron 加入 Stability AI 董事会 — Stability AI</a>：今天我们宣布，传奇电影制作人、技术创新者和视觉特效先驱 James Cameron 已加入我们的董事会。

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1287873638716145827)** (41 messages🔥): 

> - `FNAF Loras 创建`
> - `SDXL 在 GPU 上的性能`
> - `Prompt engineering 策略`
> - `ControlNet 应用`
> - `OpenPose editor 集成` 


- **FNAF Loras 创建请求**：一名成员正在寻求 **FNAF** 粉丝的帮助，为该游戏创建一些 **Loras**。
   - *有人有兴趣合作这个项目吗？*
- **3090 EGPU 提升 SDXL 性能**：一位用户报告称，尽管过去在类似产品上遭遇过失败，但最终购买了 **3090 EGPU** 以增强其 **SDXL** 游戏体验。
   - *分享了对某些游戏盒子质量的挫败感，并指出了过去 **Aurus** 品牌的问题。*
- **关于 Prompt engineering 有效性的讨论**：成员们讨论了在 SDXL 中为 **text_g** 和 **text_l** 使用**相同输入**的效率，一些人对其有效性持怀疑态度。
   - *一位成员建议只关注名词，并引用了一篇 **paper** 指出名词比形容词的影响更显著。*
- **ControlNet 的引导能力**：一位用户询问了 **ControlNet**，另一位用户解释说这是一种引导图像生成的方法，特别是针对姿势（poses）。
   - *注意到仅靠语言很难指定细节。*
- **OpenPose editor 安装问题**：一位用户报告了 Forge 中 **OpenPose editor** 的问题，并收到建议称可能需要特定的安装命令才能正常工作。
   - *提供了关于在虚拟环境中运行 `pip install basicsr` 的说明。*



**提到的链接**：<a href="https://tenor.com/view/insideout-joy-hi-hey-hello-gif-11341448685299692120">Insideout Joy GIF - InsideOut Joy Hi - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1288258847303008297)** (1 messages): 

> - `LlamaParse`
> - `欺诈网站` 


- **警惕欺诈性 LlamaParse 网站**：*警告*：llamaparse dot cloud（我们不提供链接！）是一个冒充 **LlamaIndex** 产品的**欺诈网站**。
   - **真正的 LlamaParse** 可以在 [cloud.llamaindex.ai](https://cloud.llamaindex.ai) 找到。
- **识别正版产品**：建议用户始终验证 **LlamaIndex** 产品的真实性，以避免陷入骗局。
   - 保持知情可以帮助用户确保使用正确的服务并避开欺诈性的替代方案。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1287857686402302045)** (5 messages): 

> - `LitServe 框架`
> - `AI 产品经理`
> - `LlamaIndex workflows 工作坊`
> - `Llamaparse 欺诈网站`
> - `AWS Gen AI Loft` 


- **LitServe 简化了 LLM 的服务部署**：来自 @LightningAI 的 **LitServe** 框架基于 FastAPI 构建，有助于有效地部署和扩展 LLM 模型，并在快速演示中使用了 [LlamaIndex](https://t.co/Xikqk20peW) 进行展示。
   - 该设置在本地针对 **Llama 3.1** 托管了一个简单的 **RAG** 服务器。
- **用 50 行代码创建一个 AI 产品经理！**：使用 @llama_index 和 @composiohq，仅需 50 行代码即可构建一个 **AI 产品经理**，具有读取邮件反馈和 **Slack** 通知功能。
   - 如果获得批准，它可以将反馈无缝集成到 **Linear** 板中进行请求的编辑，展示了 **function calling agent** 架构的力量。
- **关于 Context-Augmented Agents 的工作坊**：由 @AIMakerspace 举办的深入工作坊介绍了用于构建 **context-augmented agents** 的 **LlamaIndex workflows** 架构。
   - 参与者可以学习通过基于步骤的事件驱动工作流构建一个 **agentic corrective RAG** 应用程序。
- **警惕欺诈性 LlamaParse 网站**：已发布关于冒充 **LlamaIndex** LlamaParse 的欺诈网站的警告，指出其**并非正版**。
   - 官方 LlamaParse 可以在[此链接](https://t.co/jM9ioNJuv3)找到，以避免混淆。
- **在 AWS Gen AI Loft 讨论 RAG 和 Agents**：@seldo 将在 AWS Gen AI Loft 讨论 **RAG** 和 **Agents**，就在与 **@elastic** 举行的大型 ElasticON 会议之前。
   - 该会议还将涵盖 **Fiber AI** 如何利用 **Elasticsearch** 进行高效的 B2B 潜在客户挖掘，提供宝贵的交流机会。



**提到的链接**：<a href="https://t.co/jM9ioNJuv3">LlamaCloud</a>：未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1287882844072448043)** (35 messages🔥): 

> - `Approximate Metadata Filtering`
> - `Human-in-the-Loop Workflows`
> - `Postgres and pgvector`
> - `Web Crawling for Embedding` 


- **探讨 Approximate Metadata Filtering**：讨论了在 RAG 的 `workflows` 中使用近似元数据过滤，强调了根据用户查询构建动态过滤器的挑战。
   - 成员们注意到 `MilvusVectorStore` 可能不支持近似过滤器，并建议定义 Pydantic 对象来辅助创建可过滤的查询。
- **Human-in-the-Loop Workflows 的挑战**：成员们探讨了在 websocket 环境下通过嵌套工作流实现人机回环（HITL）交互，解决了在特定事件后如何将控制权交还给用户的问题。
   - 一位成员建议使用事件驱动的方法，在工作流流式传输事件时动态管理用户响应。
- **从 Postgres 与 pgvector 转型**：讨论了从使用 pgvector 的 Postgres 混合搜索转向 pgvector.rs 以获得更好性能，但目前 LlamaIndex 的实现中似乎缺少某些功能。
   - 一位成员估计，如果对 pgvector.rs 有深入了解，实现对 sparse search 选项的支持大约需要一天的量。
- **为 RAG 爬取网页**：成员们寻求关于爬取网页进行 Embedding 的技术建议，询问其他人是使用 Puppeteer 等自定义方案，还是更倾向于 Firecrawl 或 Crawlee 等工具。
   - 这一咨询反映了人们对将网页爬取数据集成到检索增强生成（RAG）流水线的有效技术有着广泛兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/#custom-node-postprocessor">Node Postprocessor - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1287851911361597551)** (14 messages🔥): 

> - `Blendtain feedback`
> - `Playlist generator by dykyi_vladk`
> - `Study Machine Learning together`
> - `Impressions of GANs, CNNs, and ViTs` 


- **用户对 Blendtain 的反馈**：一位用户对 Blendtain 的创意表示兴奋，但指出它会截断消息，并建议增加调整消息长度的设置。
   - 另一位用户给出了积极回应，简单地表示：*'yeah thxxx'*。
- **dykyi_vladk 的播放列表生成器**：[Adify.pro](https://adify.pro) 由 dykyi_vladk 推出，这是一个根据用户 Prompt 创建播放列表的生成器。
   - 创作者对该项目表示自豪，称其为 *'my coolest thing'*。
- **机器学习协作学习**：dykyi_vladk 邀请有兴趣一起学习 Machine Learning 的人私信（DM）他。
   - 这一倡议以友好的语气分享，鼓励成员之间的协作。
- **图像任务算法讨论**：一位成员评论了 **GANs**、**CNNs** 和 **ViTs** 作为图像处理任务顶级算法的地位波动。
   - 他们寻求对这一观察的确认，并请求提供这些算法变迁的视觉时间线。



**提到的链接**：<a href="https://adify.pro">Adify</a>：未找到描述

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1287877652824199278)** (12 messages🔥): 

> - `muP Transfer`
> - `HyperCloning Method`
> - `SDXL Unet`
> - `Positional Encoding in UNet`
> - `Sliding Window Attention` 


- **EleutherAI 的 muTransfer 项目**：EleutherAI 介绍了与 [Cerebras](https://cerebras.ai/) 合作的一个联合项目，旨在推广 [muTransfer](https://github.com/microsoft/mup) 的实现细节，并提供对 [nanoGPT library](https://github.com/EleutherAI/nanoGPT-mup) 的移植。这一努力旨在使 Maximal Update Parameterization (**μP**) 更易于使用并降低训练成本。
   - 然而，一些成员推测 muP 可能已经有些**过时**，可能不是未来的最佳方案。
- **HyperCloning 提升训练效率**：[arXiv 论文](https://arxiv.org/abs/2409.12903)中提出的一种新型 **HyperCloning** 方法展示了如何使用较小的预训练模型初始化大型语言模型（LLM），从而获得更好的训练时间和最终准确率。该方法在保留功能的同时，将小模型的参数扩展到大模型。
   - 成员们强调，将原始权重平铺（tiling）到更大的参数中应该会产生更好、更快的结果，使这种扩展更具**可复现性**。
- **SDXL Unet 中的位置编码担忧**：在关于 **SDXL Unet** 的讨论中，有人指出该模型没有为图像坐标使用位置编码，而是使用 adanorm 处理裁剪坐标。卷积层本质上编码了空间位置，一些人认为这使得显式的位置嵌入（positional embeddings）变得多余。
   - 尽管有这些说法，另一位成员提到滑动窗口注意力（sliding window attention）技术（如 **longformer**）虽然也利用了这些优势，但仍然结合了位置编码。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/mutransfer/">The Practitioner&#39;s Guide to the Maximal Update Parameterization</a>：探索 mutransfer 的实现细节</li><li><a href="https://arxiv.org/abs/2409.12903">Scaling Smart: Accelerating Large Language Model Pre-training with Small Model Initialization</a>：语言模型的预训练阶段通常从随机初始化的参数开始。随着当前模型规模化的趋势，训练其庞大数量的参数可能会非常缓慢 ...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1287911909848256642)** (15 messages🔥): 

> - `Nvidia's new synthetic data model`
> - `MMLU performance`
> - `Fine-tuning and inferencing challenges`
> - `Context management in LLMs`
> - `Run Pod CUDA error` 


- **对 Nvidia 新型合成数据模型的兴奋**：由提及 Nvidia 的 **51B 合成数据模型**引发的讨论，强调了其高 **MMLU** 性能可以增强应用。
   - 一位成员提到：*尝试对其进行 fine-tuning 和 inferencing 会很有趣*，表达了对实际应用的兴趣。
- **对话中自动分块（auto chunking）的挑战**：一位成员反对在对话中进行 **auto chunking** 的实用性，称 *想象一下你的对话在中间被切成两半，上下文就丢失了*。
   - 另一位成员指出，像 **ST** 或 **Kobold** 这样的系统通常通过保留第一条消息并删除最旧的消息来处理溢出的上下文。
- **动态上下文管理的价值**：有人提议动态管理滑动上下文如何帮助 **LLM** 学习有机地处理对话转移。
   - 一位成员分享了潜在的好处，建议这可以为超出上下文限制的情况提供解决方案。
- **询问 Run Pod 的使用经验**：一位成员询问是否有人在 **Run Pod** 上取得过成功，并提到正苦于处理一个 **CUDA error**。
   - **illegal CUDA error** 被记录下来，引起了对该平台潜在技术障碍的关注。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1287911909156454421)** (3 messages): 

> - `Qwen 2.5`
> - `Axolotl support` 


- **确认 Axolotl 支持 Qwen 2.5**：一位成员确认 **Axolotl** 确实支持 **Qwen 2.5**。
   - 他们指出 Qwen 2.5 与其前身 **Qwen 2** 使用**相同的架构**。
- **用户询问 Qwen 2.5 支持情况**：一位用户询问了 **Qwen 2.5** 在 **Axolotl** 上的支持状态。
   - 这引发了讨论并最终得到了成员的确认。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1287881023669010517)** (4 条消息): 

> - `Fine-tuning spikes`
> - `Rope scaling`
> - `Llama3.1 setup`
> - `Qwen2.5 configurations` 


- **观察到 Fine-tuning 中的 Spikes**：一名成员报告在对 **100K row dataset** 进行 **fine-tune** 时遇到了 **spike**，并寻求日志输出来将此 **spike** 与特定的数据行关联起来。
   - 关于日志帮助的回复是：*遗憾的是，目前还不支持*。
- **建议使用 Rope Scaling**：另一名成员建议使用 **rope scaling** 来管理内存效率，并强调仅增加 **seq len** 就会消耗大量的 **vRAM**。
   - 这种方法可能有助于缓解 **fine-tuning** 期间遇到的 **spikes** 相关问题。
- **Llama3.1 设置咨询**：一位用户询问他们针对 **Llama3.1** 的 **fine-tuning** 设置是否正确——即使用 **4K sequence length** 配合 **3x factor** 来实现 **120K window**。
   - 他们寻求确认该配置是否针对其需求进行了优化。
- **Qwen2.5 Context 配置**：对话中包含了一个关于 **Qwen2.5** 的设置咨询，标注为使用 **3K sequence length** 和 **4x factor** 来实现 **120K context**。
   - 这些比例表明在设置配置时，采用了深思熟虑的方法来最大化模型效率。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1287960576475136012)** (2 条消息): 

> - `Qwen 2.5`
> - `Axolotl Support` 


- **Axolotl 支持 Qwen 2.5 文本处理**：一名成员确认 **Axolotl** 应该支持 **Qwen 2.5** 的常规文本处理。
   - 然而，他们提到可能不支持 **vision** 功能。
- **Axolotl 对 Qwen 2.5 的 Vision 支持**：尽管支持文本，该成员仍对 **Axolotl** 上 **Qwen 2.5** 的 **vision** 能力表示怀疑。
   - 这表明在处理视觉输入时，与文本相比可能存在局限性。


  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1288026271795908630)** (17 条消息🔥): 

> - `LangChain Pydantic 兼容性`
> - `LangGraph 中的 GraphRecursionError`
> - `适用于 LLM 的 LangChain 文档`
> - `Mistral 与 Mixtral 的对比` 


- **LangChain Pydantic 兼容性问题**: 用户在从 `langchain_openai` 导入 `ChatOpenAI` 时遇到错误，提示 Pydantic v2 不支持 `__modify_schema__` 方法。
   - 建议检查 Pydantic 版本并改用 `__get_pydantic_json_schema__`，详见 [LangChain 文档](https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis)。
- **LangGraph 中的 GraphRecursionError**: 当 LangGraph 应用达到 25 次的递归限制时会抛出 `GraphRecursionError`，以防止无限循环。
   - 鼓励用户在配置中增加该限制，参考 [GitHub issue 评论](https://github.com/langchain-ai/langchain/issues/18598) 中提供的解决方案。
- **对 LLM 友好型文档的需求**: 一位用户询问是否有适用于 LLM 的上下文文本文件，以提高使用 LangChain 的效率。
   - 另一位成员发起的主题讨论了这一话题，表明社区正在就 LangChain 资源进行持续讨论。
- **Mistral vs Mixtral 开源模型**: 一位成员询问在 **Mistral** 和 **Mixtral** 之间，目前哪种模型是自托管的最佳开源解决方案。
   - 这表明社区对自托管模型的性能对比和可用性非常感兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/how_to/pydantic_compatibility/#1-passing-pydantic-objects-to-langchain-apis>).">如何在不同 Pydantic 版本中使用 LangChain | 🦜️🔗 LangChain</a>: - Pydantic v2 已于 2023 年 6 月发布 (https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/)。</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/chat/openai/#instantiation>).">ChatOpenAI | 🦜️🔗 Langchain</a>: OpenAI 是一个人工智能...</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-3>)">如何从旧版 LangChain Agent 迁移到 LangGraph | 🦜️🔗 LangChain</a>: 本指南假设你已熟悉以下概念：</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#maxiterations>).">如何从旧版 LangChain Agent 迁移到 LangGraph | 🦜️🔗 Langchain</a>: 本指南假设你已熟悉以下概念： -
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1287853629260435629)** (10 messages🔥): 

> - `优化器的 CPU Offloading`
> - `性能优化技术`
> - `Paged Adam vs Torchao CPUOffloadOptimizer` 


- **关于优化器中 CPU Offloading 的困惑**：讨论了为什么优化器的 CPU Offloading 未被利用，并引用了[这个旧 Issue](https://github.com/pytorch/pytorch/issues/74588)，其中提到了性能下降的问题。
   - 一位成员建议结合 PagedAdam 使用 CPU Offloading 以优化性能，同时强调需要一个 PR 来修改单设备微调中优化器的用法。
- **优化器方法的对比分析**：有人指出使用 torchao 的 CPUOffloadOptimizer 与 optimizer in backward 配合效果不佳，从而引发了关于 Adam 等更快替代方案的疑问。
   - 建议包括尝试 `offload_gradients=True` 以节省梯度内存，同时使 CPU 计算与 GPU 处理重叠以获得更好的性能，详见此 [PR](https://github.com/pytorch/ao/pull/584)。
- **CUDA MODE 社区邀请**：建议对性能优化感兴趣的成员加入 GPU MODE Discord 小组，并表示那里有更多专业人士可以提供帮助。
   - 加入链接分享在[这里](https://discord.gg/jNsm4H44)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_single_device.py">torchtune/recipes/full_finetune_single_device.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/issues/74588">[FSDP] 使用 CPUOffload 导致 3-10 倍的速度下降，原因是 CPU 优化器 step/update 缓慢 · Issue #74588 · pytorch/pytorch</a>：描述 Bug：使用 FSDP 创建简单的分布式模型包装器。运行不带 CPUoffload 的有状态优化器（如 AdamW）并进行 profile/计时。然后运行带 CPUOffload 的版本，发现性能……</li><li><a href="https://github.com/pytorch/torchtune/issues/1576">用 8bitpagedadam 或 torchao CPUOffloadOptimizer 替换 adamW 和 pagedadam · Issue #1576 · pytorch/torchtune</a>：显然没有理由使用 paged adam 而不使用 8bit 版本。我们应该替换它。此外，单设备全量微调应该使用 paged adam 而不是 adamw，以获得更好的内存表现。对于……</li><li><a href="https://github.com/pytorch/ao/pull/584">用于单 GPU 训练的优化器 CPU offload，由 gau-nernst 提交 · Pull Request #584 · pytorch/ao</a>：背景：目前没有简单的方法为单 GPU 训练实现优化器 CPU offload，尽管 FSDP 存在此类功能。DeepSpeed ZeRO-Offload 可以配合单 GPU 工作，但它需要……
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1288116767000104960)** (2 messages): 

> - `分布式训练`
> - `行星大脑概念`
> - `DisTrO 项目` 


- **探索行星大脑的概念**：一位成员幽默地询问我们距离实现 **NET=1** 还有多远，即通过分布式训练将 tinybox 连接起来形成一个**行星大脑**。
   - 这暗示了一个集体智慧可以在全球范围内实现**分布式训练**的未来。
- **用于分布式训练的 DisTrO 介绍**：讨论重点介绍了 **[DisTrO 项目](https://github.com/NousResearch/DisTrO)**，该项目专注于实现互联网上的分布式训练（Distributed Training Over-The-Internet）。
   - 该项目旨在彻底改变通过互联网协作训练模型的方式。



**提到的链接**：<a href="https://github.com/NousResearch/DisTrO">GitHub - NousResearch/DisTrO: Distributed Training Over-The-Internet</a>：互联网上的分布式训练。可以通过在 GitHub 上创建账号为 NousResearch/DisTrO 的开发做出贡献。

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1288157299235360832)** (7 messages): 

> - `Tensor 中的 AttributeError`
> - `Tinygrad 版本问题`
> - `模型架构见解` 


- **AttributeError: 'Tensor' 缺少 cross_entropy**: 一位用户遇到了 'AttributeError'，提示 **Tensor** 对象没有 `cross_entropy` 属性。他们分享了一段代码片段，指出了该错误在训练步骤函数中出现的位置。
   - 这引发了关于错误潜在原因的讨论，包括 Tensor 实现中可能存在的问题。
- **Tinygrad 版本争议**: 另一位用户询问了正在使用的 **Tinygrad** 版本，得到的回复是发布者最近从 Git 更新到了 **0.9.2** 版本。
   - 有人指出该版本不支持所需的功能，建议升级到来自 master 分支的最新版本以获取更多功能。
- **模型架构与训练**: 用户分享了一个包含多个卷积层、一个展平操作（flattening）以及后续线性层的模型架构。对话强调了在构建模型时为提高训练性能而做出的设计选择。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1287939914628862105)** (9 messages🔥): 

> - `Open Interpreter 更新`
> - `基于 LLM 的浏览器自动化`
> - `社区参与`
> - `GitHub 资源`
> - `关于项目活跃度的反馈` 


- **Open Interpreter 并未停止维护**: 一位成员确信 Open Interpreter 正在 [GitHub](https://github.com/OpenInterpreter/open-interpreter) 上积极接收更新，展示了持续的开发进展。
   - 此外，围绕项目 '01' 也有大量活动，旨在集成专门的语音助手模式，详见[此处](https://github.com/OpenInterpreter/01)。
- **探索用于浏览器自动化的 LLM**: 一位成员分享了利用 Open Interpreter 进行基于 LLM 的浏览器自动化和表单提交的见解，确认其可行，但由于任务的复杂性存在局限。
   - 他们建议使用 Playwright 以获得更好的效果，并分享了一个他们一直在完善的 [Prompt 示例](https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml)。
- **社区热情依旧**: 尽管对项目活跃度有所担忧，成员们继续讨论实际用例，其中一人表示渴望使用共享的 Prompt 自动向目录提交内容。
   - 另一位成员重申社区保持活跃，积极回应咨询并分享使用该工具的经验。
- **宣布即将举行的社区活动**: 一位成员预告了即将举行的与 Open Interpreter 相关的活动，并分享了 [Discord 链接以获取更多详情](https://discord.gg/open-interpreter-1146610656779440188?event=1288234745477726238)。
   - 这一公告引发了用户的兴奋，表明社区活动仍在持续。
- **讨论项目感知的转变**: 针对有关项目状态的询问，一位成员幽默地指出，最初的提问者可能没有完全掌握项目的最新进展。
   - 这种互动凸显了社区内部对项目生命力的不同看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/morisy/openinterpreter-configs/blob/main/foiaportalassistant.yaml">openinterpreter-configs/foiaportalassistant.yaml at main · morisy/openinterpreter-configs</a>: 存放我在使用 Open Interpreter 进行实验时的笔记、配置文件等的地方。- morisy/openinterpreter-configs</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言界面</a>: 计算机的自然语言界面。通过在 GitHub 上创建账户为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: 适用于桌面、移动和 ESP32 芯片的排名第一的开源语音界面。</a>: 适用于桌面、移动和 ESP32 芯片的排名第一的开源语音界面。 - OpenInterpreter/01
</li>
</ul>

</div>
  

---



---



---



---



---



---



---



{% else %}


> 完整的频道逐条解析已因邮件长度限制而截断。
> 
> 如果你想查看完整解析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}