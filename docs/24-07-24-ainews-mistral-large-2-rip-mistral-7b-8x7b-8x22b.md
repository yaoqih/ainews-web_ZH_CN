---
companies:
- mistral-ai
- meta-ai-fair
- groq
- togethercompute
date: '2024-07-24T23:44:31.500890Z'
description: '**Mistral Large 2** 推出了拥有 **1230 亿参数** 的模型，并在研究许可证下采用 **开放权重** 模式。该模型专注于
  **代码生成**、**数学性能** 以及高达 **128k 的上下文窗口**（相比 Mistral Large 1 的 32k 有显著提升）。据称，其 **函数调用**
  能力优于 **GPT-4o**，且推理能力得到了增强。


  与此同时，**Meta** 正式发布了 **Llama-3.1** 系列模型，包括 **Llama-3.1-70B** 和 **Llama-3.1-8B**，并分享了详细的预训练和后训练见解。研究发现，**Llama-3.1
  8B** 模型在 128k 上下文下的表现与 **Mistral Nemo** 和 **Yi 34B 200K** 相比显得逊色。


  Mistral 正在逐步弃用旧的 Apache 协议开源模型，转而专注于 Large 2 和 **Mistral Nemo 12B**。此外，相关新闻还重点介绍了社区讨论和基准测试对比。'
id: 6da8ca62-994d-48b0-ab35-e69b703e4450
models:
- mistral-large-2
- mistral-nemo-12b
- llama-3.1-8b
- llama-3.1-70b
- llama-3.1
- llama-3-405b
- yi-34b-200k
- gpt-4o
original_slug: ainews-mistral-large-2
people: []
title: 'Mistral Large 2 + 再见（或：安息吧）Mistral 7B, 8x7B, 8x22B


  *(注：这里的 "RIP" 通常指新模型的发布使得旧模型（7B, 8x7B, 8x22B）不再具有竞争力或被取代。)*'
topics:
- code-generation
- math
- function-calling
- reasoning
- context-windows
- model-deprecation
- pretraining
- posttraining
- benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**你需要的是 Mistral 商业许可证。**

> 2024年7月23日至7月24日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**474** 个频道，**4118** 条消息）。预计节省阅读时间（以 200wpm 计算）：**428 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

对比 [2024 年 2 月的 Mistral Large](https://mistral.ai/news/mistral-large/?ref=upstract.com) 与 [今天的 Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 的侧重点是很有启发性的：

- Large 1：重点关注 MMLU 81%，介于 Claude 2 (79%) 和 GPT4 (86.4%) 之间，**仅限 API**，未公布参数量。
- Large 2：关于 MMLU 84% 只有一小段描述（仍然没有超过 GPT4！），123B 参数 **Open Weights**（在研究许可证下），“在开放模型的性能/成本帕累托前沿（Pareto front）上设定了新基准”，但新的重点是使用由 [Mixtral 8x22 推广的“凸包（convex hull）”图表](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/) 展示的 codegen 和数学性能。 
![image.png](https://assets.buttondown.email/images/bdb9414a-a9a6-455e-8e8d-ad1cc1daf2f6.png?w=960&fit=max)
 
- 两者都相当关注多语言 MMLU。
- Large 1：32k 上下文。
- Large 2：**128k 上下文**。
- Large 1：仅顺带提及 codegen。
- Large 2：“继 Codestral 22B 和 Codestral Mamba 的经验之后，我们在 Mistral Large 2 的训练中使用了极高比例的代码。” 
![image.png](https://assets.buttondown.email/images/842708ad-cafb-4fa2-8a71-f392c69214e1.png?w=960&fit=max)
 
- Large 1：“原生支持 function calling”和“JSON 格式”。
- Large 2：“开个玩笑，其实我们的 Function calling 在 v1 中并不好，但现在我们比 GPT4o 更强了”。 
![image.png](https://assets.buttondown.email/images/b7986e07-a81f-4098-9057-8da4c5c75bae.png?w=960&fit=max)
 
- Large 2：“还投入了大量精力来增强模型的推理能力。”
- Llama 3.1：<<长达 90 页关于[如何使用合成数据提高推理和数学能力](https://www.latent.space/p/llama-3)的极详尽细节>>。

Mistral 的 la Plateforme 正在弃用其所有 Apache 开源模型（Mistral 7B, Mixtral 8x7B 和 8x22B, Codestral Mamba, Mathstral），其通用模型仅保留 Large 2 和上周发布的 12B [Mistral Nemo](https://mistral.ai/news/mistral-nemo/)。这一弃用完全符合我们在昨天文章末尾讨论的 [成本-Elo 归一化前沿图表](https://x.com/swyx/status/1815892458519289946/photo/1) 的预测。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，从 4 次运行中取最佳结果。

> 今日暂时停更。明天回归。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Llama 3.1 发布及其能力**

- **Meta 正式发布 Llama-3-405B, Llama-3.1-70B & Llama-3.1-8B** ([Score: 910, Comments: 373](https://reddit.com//r/LocalLLaMA/comments/1ea9eeo/meta_officially_releases_llama3405b_llama3170b/)): **Meta** 已正式发布其 **Llama 语言模型**的新版本，包括 **Llama-3-405B**、**Llama-3.1-70B** 和 **Llama-3.1-8B**。这些模型可从 [Llama 官网](https://llama.meta.com/llama-downloads/) 下载，并可在 [Groq](https://console.groq.com/playground) 和 [Together](https://api.together.xyz/playground) 等云服务商的 Playground 中进行测试。

- **让我们讨论 Llama-3.1 论文（关于预训练、后训练等的大量细节）** ([Score: 109, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eabf4l/lets_discuss_llama31_paper_a_lot_of_details_on/)): **Llama 3.1 论文揭示预训练细节**。可在 [ai.meta.com](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) 获取的 Llama 3.1 论文提供了关于模型预训练和后训练过程的详尽细节。论文包括超参数概览、验证损失图表，以及从 **7B 到 70B** 参数不等的多尺寸模型的各种性能指标。

- **[关于 Llama 3.1 8B 在 128K 上下文下的早期评价](https://reddit.com//r/LocalLLaMA/comments/1eac5a7/early_hot_take_on_llama_31_8b_at_128k_context/)** ([评分: 72, 评论: 49](https://reddit.com//r/LocalLLaMA/comments/1eac5a7/early_hot_take_on_llama_31_8b_at_128k_context/)): **Llama 3.1 8B 模型的 128K 上下文表现令人失望**。作者使用小说风格的故事测试了 **Llama 3.1 8B 模型** 的 **128K 上下文**，发现其能力不如 **Mistral Nemo**，且明显逊于 **Yi 34B 200K** 模型。即使在 **24GB VRAM** 中使用 **exllama** 配合 **Q6 cache** 以 **FP16** 精度进行测试，该 Llama 模型也难以识别之前设定的关于角色假定死亡的上下文并生成适当的反应。尽管作者进一步尝试了 **8bpw 和 Q8 量化**，但最终还是决定放弃 Llama 8B，转而选择 **Mistral Dori**。

**主题 2. 开源 AI 策略与行业影响**

- **[开源 AI 是前进之路 - Mark Zuckerberg](https://reddit.com//r/LocalLLaMA/comments/1eaa0m2/open_source_ai_is_the_path_forward_mark_zuckerberg/)** ([评分: 794, 评论: 122](https://reddit.com//r/LocalLLaMA/comments/1eaa0m2/open_source_ai_is_the_path_forward_mark_zuckerberg/)): **Mark Zuckerberg 倡导开源 AI**。Mark Zuckerberg 认为 **开源 AI** 对于推动 AI 技术进步并确保其负责任的发展至关重要。在他的 [博客文章](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) 中，Zuckerberg 强调了开源 AI 的优势，包括 **更快的创新**、**更高的透明度** 以及 **更广泛地获取** AI 工具和知识。

- **[根据 AI Act，Llama 3 405b 被视为对社会的“系统性风险”](https://x.com/deanwball/status/1815826885663658445)** ([评分: 169, 评论: 68](https://reddit.com//r/LocalLLaMA/comments/1eal9oq/llama_3_405b_is_a_systemic_risk_to_society/)): **Meta 的 Llama 3.1 405B 模型** 已根据 **欧盟 AI Act** 被归类为“**系统性风险**”。这一认定适用于参数量超过 **10^25** 的 AI 系统，这使得 Meta 在该模型的开发和部署方面承担了重大的监管义务。这一分类凸显了人们对大型语言模型潜在社会影响的日益担忧，以及它们在欧洲面临的日益严格的监管审查。

- **[现在的 OpenAI...](https://i.redd.it/h60m9gglyced1.jpeg)** ([评分: 167, 评论: 27](https://reddit.com//r/LocalLLaMA/comments/1eanchg/openai_right_now/)): **OpenAI 的竞争对手正在缩小差距**。Meta 发布 **Llama 3.1** 展示了性能的显著提升，可能挑战 OpenAI 在 AI 语言模型领域的领导地位。这一进展表明 AI 领域的竞争正在加剧，其他公司也在迅速提升其能力。
    - **ChatGPT 性能下降**：用户反映自 2023 年初以来 **ChatGPT 的编程能力有所退化**，**GPT-4** 和 **GPT-4 Turbo** 在生成 PowerShell 脚本等任务中表现出不一致的结果，可靠性降低。
    - **OpenAI 的公信力受到质疑**：批评者指出 OpenAI **游说监管开源 AI**，并邀请前 **NSA 负责人 Paul Nakasone** 加入董事会，这表明其正在背离最初的“开放”使命。
    - **呼吁发布开源版本**：一些用户表达了希望 OpenAI **发布模型权重**（特别是 **GPT-3.5**）以便在本地运行的愿望，认为这是真正推动行业进步并履行其“Open”之名的方式。


**主题 3. 性能基准测试与对比**

- **[LLama 3.1 对比 Gemma 及 SOTA](https://www.reddit.com/gallery/1eaal5s)** ([评分: 140, 评论: 37](https://reddit.com//r/LocalLLaMA/comments/1eaal5s/llama_31_vs_gemma_and_sota/)): **Llama 3.1** 在包括 **MMLU**、**HumanEval** 和 **GSM8K** 在内的多项基准测试中超越了 **Gemma** 和其他最先进（SOTA）模型。Llama 3.1 的 **7B** 和 **13B** 版本较其前代有显著提升，其中 13B 模型的得分已可媲美或超越 **GPT-3.5** 等更大规模的模型。这种性能飞跃表明 Llama 3.1 代表了语言模型能力的重大进步，特别是在推理和基于知识的任务中。

- **[Llama 3.1 405B 在新的 ZebraLogic 推理基准测试中位列第二](https://i.redd.it/o9l7ym58fced1.png)** ([评分: 110, 评论: 9](https://reddit.com//r/LocalLLaMA/comments/1eakv0o/llama_31_405b_takes_2_spot_in_the_new_zebralogic/)): **Llama 3.1 405B** 在新推出的 **ZebraLogic 推理基准测试** 中获得了 **第二名**，展示了其先进的推理能力。这一成绩使该模型仅次于 **GPT-4**，并领先于 **Claude 2** 和 **PaLM 2** 等其他知名模型。ZebraLogic 基准测试旨在评估模型处理复杂逻辑推理任务的能力，为衡量 AI 在这一关键领域的表现提供了新的指标。

- **LMSYS 的最后一根稻草** ([Score: 175, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1ean2i6/the_final_straw_for_lmsys/)): **LMSYS 基准测试的可信度受到质疑**。作者批评了 **LMSYS 的 ELO 排名** 将 **GPT-4o mini** 列为整体排名第二的模型，认为 **GPT-4**、**Gemini 1.5 Pro** 和 **Claude Opus** 等其他模型的能力更强。该帖子指出，**人类对 LLM 的评估** 现在受限于人类的能力而非模型的能力，并建议使用 **ZebraLogic**、**Scale.com leaderboard**、**Livebench.ai** 和 **LiveCodeBench** 等替代基准测试，以进行更准确的模型能力评估。

**主题 4. 社区工具与部署资源**

- **[Llama-3.1 8B Instruct GGUF 已上线](https://huggingface.co/aniljava/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main)** ([Score: 50, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1eabypz/llama31_8b_instruct_gguf_are_up/)): **Llama-3.1 8B Instruct GGUF** 模型已发布，提供多种量化级别，包括 **Q2_K**、**Q3_K_S**、**Q3_K_M**、**Q4_0**、**Q4_K_S**、**Q4_K_M**、**Q5_0**、**Q5_K_S**、**Q5_K_M**、**Q6_K** 和 **Q8_0**。这些量化版本为模型大小和性能之间的权衡提供了不同选择，允许用户根据特定的使用场景和硬件限制选择最合适的版本。

- **在 Colab 免费微调 Llama 3.1 + 速度提升 2.1 倍，VRAM 占用减少 60% + 4bit BnB 量化** ([Score: 85, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1eaitaq/finetune_llama_31_for_free_in_colab_get_21x/)): **Unsloth** 发布了针对 **Llama 3.1** 的工具，使微调速度提升 **2.1 倍**，**VRAM** 占用减少 **60%**，并在不损失精度的情况下将原生 HF 推理速度提高 **2 倍**。此次发布包括一个用于微调 **8B 模型** 的 **免费 Colab notebook**，用于加快下载速度并减少 VRAM 占用的 **4-bit Bitsandbytes 量化模型**，以及在 Colab 中与 Llama 3.1 8B Instruct 进行本地聊天的 **Studio Chat UI** 预览。

- **我们开发了 glhf.chat：运行（几乎）任何开源 LLM，包括 405b** ([Score: 54, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eap9fj/we_made_glhfchat_run_almost_any_opensource_llm/)): **用于运行开源 LLM 的新平台 glhf.chat 上线**。新推出的 [glhf.chat](https://glhf.chat) 平台允许用户运行几乎任何由 **vLLM 项目** 支持的开源 LLM，包括显存需求高达 **~640GB VRAM** 的模型。与竞争对手不同，该平台没有硬编码的模型列表，用户可以通过粘贴 **Hugging Face 链接** 来运行任何兼容的模型或微调版本，支持 **Llama-3-70b** 的微调版以及即将推出的 **Llama-3.1** 版本。
    - 该平台最初在注册时需要邀请码 "405B"（在原帖中提到）。开发者 **reissbaker** 随后完全移除了邀请系统，以简化所有用户的访问。
    - 由于升级身份验证提供商时的疏忽，用户遇到了 "500 用户限制" 错误。另一位 glhf.chat 开发者 **Billy** 承认了该问题并承诺在几分钟内修复。
    - 响应用户请求，**reissbaker** 发布了针对 **Mistral NeMo 架构** 的修复补丁，在该平台上实现了对 [dolphin-2.9.3-mistral-nemo-12b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) 等模型的支持。

## Reddit AI 综合回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型发布与基准测试**

- **Meta 发布 Llama 3.1 405B 模型**：Meta 发布了一个新的 4050 亿参数的 Llama 模型。[基准测试结果](https://www.reddit.com/r/singularity/comments/1eab6b1/llama_31_405b_on_scale_leaderboards/)显示，它在某些任务上的表现与 GPT-4 和 Claude 3.5 Sonnet 相当。

- **Zuckerberg 主张开源 AI 模型**：Mark Zuckerberg [阐述了理由](https://www.reddit.com/r/singularity/comments/1eaej0u/mark_zuckerberg_eloquently_states_the_case_for/)，认为开源 AI 模型是有益的，并辩称封闭模型无论如何都会被窃取。他表示：**“中国能够获得开源权重并不重要，因为如果权重是封闭的，他们无论如何也会窃取权重。”**

- **Google 发布 “AI Agents 系统”**：Google 发布了 [Project Oscar](https://www.reddit.com/r/singularity/comments/1ea1kz9/google_has_released_the_worlds_first_ai_agents/)，这是一个用于创建 AI agents 以管理软件项目的开源平台，特别适用于监控 issue 和 bug。

**AI 能力与基准测试**

- **关于 AI 是否超越人类智能的辩论**：关于目前的 AI 模型是否在某些领域超越了人类水平的智能，目前仍在讨论中。有人认为 [AI 现在已经“聪明到足以愚弄我们”](https://www.reddit.com/r/singularity/comments/1eaud7r/for_the_first_time_in_history_the_ais_are_smart/)，而另一些人则认为 **AI 在简单的逻辑和数学任务上仍然表现吃力**。

- **当前基准测试的局限性**：批评者指出，[当前的 AI 基准测试可能无法准确衡量智能](https://www.reddit.com/r/singularity/comments/1eaud7r/for_the_first_time_in_history_the_ais_are_smart/leogncp/)。例如，Arena 基准测试衡量的是人们更喜欢哪些回答，而不一定是智能。

**AI 伦理与企业实践**

- **OpenAI 因保密协议受到批评**：在 [社交媒体上的社区备注](https://www.reddit.com/r/OpenAI/comments/1eaq40g/openai_got_community_noted/) 指出 OpenAI 此前曾使用保密协议阻止员工进行受保护的披露后，该公司面临批评。

- **关于开源与封闭 AI 开发的辩论**：关于开源 AI 模型与保持封闭的优缺点的讨论仍在继续。有人认为开源促进了创新，而另一些人则担心潜在的滥用。

---

# AI Discord 纪要

> 摘要之摘要的摘要

**1. Llama 3.1 模型性能与挑战**

- **微调困境**：**Llama 3.1** 用户报告了**微调方面的问题**，特别是与模型配置和 Tokenizer 处理相关的错误消息，建议更新 transformers 库。
   - 讨论强调了指定**正确模型版本**和保持正确配置以缓解这些挑战的必要性。
- **性能不一致**：用户注意到 **Llama 3.1 8B** 在推理和编码任务中表现吃力，一些成员对其整体性能表示怀疑。
   - 比较表明，虽然在其尺寸级别表现尚可，但其逻辑能力似乎不足，特别是与 **Gemma 2** 等模型相比。
- **过载问题**：**Llama 3.1 405B** 模型由于请求过载经常显示“服务不可用”错误，这表明需求较高且可能存在基础设施限制。
   - 用户讨论了 405B 变体的特性，提到与 70B 兄弟模型相比，它感觉受到了更多的审查（censored）。
    


**2. Mistral Large 2 模型**

- **Mistral Large 2 发布**：2024 年 7 月 24 日，Mistral AI 推出了 **Mistral Large 2**，拥有令人印象深刻的 **1230 亿参数**和 **128,000-token 上下文窗口**，进一步提升了 AI 能力。
   - 据报道，**Mistral Large 2** 的表现优于 **Llama 3.1 405B**，特别是在复杂的数学任务中，使其成为行业巨头的强劲竞争对手。
- **多语言能力**：与现有模型相比，**Mistral Large 2** 模型拥有更长的上下文窗口和多语言支持，使其成为适用于各种应用的通用工具。
   - 成员们将其与 Llama 模型进行了比较，并注意到在这个不断发展的市场中，性能优化工作仍在持续。
    


**3. 软件开发中的 AI 与职业安全**

- **职业安全担忧**：随着 AI 工具越来越多地集成到编码实践中，参与者讨论了初级开发人员面临的**职业安全不确定性**，这可能会使入门级角色边缘化。
   - 共识认为，经验丰富的开发人员应该适应这些工具，利用它们来提高生产力，而不是取代人类互动。
- **AI 数据处理中的隐私问题**：关于 **AI 的数据处理实践**出现了担忧，特别是人类审核员访问敏感信息的潜在影响。
   - 讨论强调了建立强大的数据管理协议以保护用户隐私的迫切需求。
    


**4. AI 模型基准测试与评估**

- **对基准测试的怀疑**：对 **Llama 405B** 的性能指标存在怀疑，讨论强调了它在与 **Mistral** 和 **Sonnet** 模型对比中的平庸表现。
   - 社区反思了各种基准测试结果和主观体验，将基准测试比作“电影评分”，认为其无法捕捉真实的真实用户体验。
- **评估方法**：强调了在**幻觉预防技术**中需要更好的基准测试，引发了关于改进评估方法的讨论。
   - 与 Meta 工程师的简短对话引发了对基准测试现状的担忧，建议采用协作方式开发更可靠的指标。
    


**5. 开源 AI 进展**

- **Llama 3.1 发布**：**Llama 3.1** 模型正式发布，将上下文长度扩展到 **128K** 并支持八种语言，标志着开源 AI 的重大进步。
   - 用户报告 **Llama 3.1 405B** 模型由于过载经常出现“服务不可用”错误，并认为它比 **70B** 版本感觉更受限。
- **Mistral Large 2 特性**：**Mistral Large 2** 具备最先进的 **function calling 能力**，并对结构化输出和 Agent 提供首日支持。
   - 此次发布与增强的 **function calling** 和**结构化输出**保持一致，为用户提供了如 Cookbook 等实用资源进行探索。
    

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.1 Fine-Tuning 挑战**：用户报告了**微调 Llama 3.1 的问题**，特别是源于模型配置和 tokenizer 处理的错误消息，建议更新 transformers 库。
   - 讨论强调了指定**正确模型版本**和保持正确配置以缓解这些挑战的必要性。
- **AI 开发中的就业安全担忧**：参与者讨论了随着 AI 工具日益融入编程实践，初级开发人员面临的**就业安全不确定性**，这可能会使入门级角色边缘化。
   - 共识认为，经验丰富的开发人员应该适应这些工具，利用它们提高生产力，而不是取代人类互动。
- **关于图像生成偏差的见解**：围绕**图像生成**的讨论强调了在实现多样性和解决 AI 模型固有偏差方面的挑战，这对于教育背景至关重要。
   - 出现了对当前多样性努力的批评，指出了可能扭曲历史准确性的执行缺陷。
- **Mistral Large 2 的性能**：**Mistral Large 2** 模型作为 AI 领域的强力竞争者出现，与现有模型相比，它拥有更长的上下文窗口和多语言支持。
   - 成员们将其与 Llama 模型进行了比较，并注意到在这个不断发展的市场中，性能提升工作正在持续进行。
- **AI 数据处理中的隐私担忧**：人们对 **AI 的数据处理实践**产生了担忧，特别是人类审核员访问敏感信息的潜在影响。
   - 讨论强调了建立强大的数据管理协议以保护用户隐私的迫切需求。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 在运行 Llama 3.1 时遇到困难**：用户发现 **LM Studio** 无法在 OpenCL 显卡上运行 **Llama 3.1**；建议升级到 **0.2.28** 版本以获得更好的支持。
   - 确认来自 **LM Studio** 的更新对于 Llama 3.1 等大型模型的有效性能至关重要。
- **ROCm 0.2.28 导致性能下降**：在 ROCm **0.2.28** 更新后，一位用户经历了性能下降，在双 **7900 XT** 配置上仅看到 **150w 功耗**。
   - 恢复到 **0.2.27** 后恢复了正常性能，引发了对新更新中变化的深入调查要求。
- **Nemo 模型面临上下文和性能问题**：用户报告 **Nemo 模型**在当前版本下可以运行，但受限于上下文长度，且由于 RAM 不足导致输出较慢。
   - 某些特定配置下有成功案例，同时也提出了优化建议。
- **GPU Offloading 问题依然存在**：多位成员报告其系统上的 GPU Offloading 功能异常，特别是在 M3 Max 和 **4080S** GPU 上，通常需要手动调整。
   - 自动设置导致了错误的输出，表明需要更可靠的手动配置以获得更好的性能。
- **Meta-Llama 3.1 70B 进入仓库**：**Meta-Llama 3.1** 的 **70B 量化模型**已发布，可通过 [该仓库](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF) 获取。
   - 频道内的热情显著，预计在重新上传以修复 **tokenizer 错误**后，性能将有所提升。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llama 3.1 405B 掀起热潮**：**Llama 3.1 405B** 模型被誉为最强大的开源模型，现已在 Perplexity 上线，其性能可与 **GPT-4o** 和 **Claude Sonnet 3.5** 媲美。
   - 将其集成到**移动应用程序**中的激动人心计划正在进行中，旨在为移动开发者增强可访问性。
- **Mistral Large 2 开辟新天地**：2024 年 7 月 24 日，Mistral AI 发布了 **Mistral Large 2**，拥有惊人的 **1230 亿参数**和 **128,000-token 上下文窗口**，进一步提升了 AI 能力。
   - 据报道，Mistral Large 2 的表现优于 **Llama 3.1 405B**，特别是在复杂的数学任务中，使其成为行业巨头的强劲竞争对手。
- **AI 模型基准测试受到质疑**：人们对 **Llama 405b** 的性能指标产生怀疑，讨论强调了它在 **Mistral** 和 **Sonnet** 模型面前表现平平。
   - 社区反思了各种基准测试结果和主观体验，将基准测试比作无法捕捉真实用户体验的“电影评分”。
- **NextCloud 集成 OpenAI**：最近 **NextCloud** 与 OpenAI 的集成引发了关注，其特点是社区驱动、开源的方法，促进了清晰的代码标准。
   - 共享了一个 GitHub 仓库，为有志于探索这一新功能及其影响的开发者提供资源。
- **TikTok 的搜索引擎潜力**：关于 TikTok 作为 Z 世代搜索工具的讨论非常热烈，突显了其日益增长的影响力，并对传统搜索引擎发起了挑战。
   - 对该平台可靠性的担忧（尤其是在健康建议方面）表明，在使用 TikTok 获取关键信息时需要谨慎。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Mistral-7B 拥有海量上下文窗口**：**Mistral-7B-v0.3** 模型拥有惊人的 **128k 上下文窗口**并支持多种语言，而 **Mistral Large** 版本在使用 **ollama** 时仅需 **69GB** 即可高效运行。
   - 用户称赞了它的能力，指出了在处理大型数据集的多任务处理中的潜在应用。
- **经济实惠的 GPU 服务器方案出现**：讨论强调 **Runpod** 是运行大型模型的高性价比 GPU 服务器选择，价格仅为 **$0.30/小时**。
   - 参与者建议使用 **LM Studio** 和 **ollama**，以便根据特定模型需求获得更好的性能。
- **Kling AI 提供奇特的图生视频功能**：**Kling AI** 以其将静态图像转换为视频的能力给用户留下了深刻印象，尽管一些人指出了视频质量和服务器过载的问题。
   - 尽管体验褒贬不一，但引人入胜的输出激发了进一步尝试该工具的兴趣。
- **记忆功能的不一致性令用户沮丧**：成员们报告说，**memory feature** 在欧盟的表现不一，有些人只能临时访问五分钟。
   - 这引发了关于该功能运行状态及其整体可靠性的轻松调侃。
- **在 Python 中使用 OpenAI 生成 PDF**：一位用户寻求通过 **Python** 使用 OpenAI 生成 PDF 文档的帮助，寻找根据上传内容自动生成章节描述的方法。
   - 这一讨论推动了关于增强文档生成流程的高效工作流的协作交流。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM Distillation 技术的进展**：成员们强调了 [Minitron GitHub 仓库](https://github.com/NVlabs/Minitron) 在理解利用 **pruning**（剪枝）和 **knowledge distillation**（知识蒸馏）进行 **LLM distillation** 最新进展方面的潜力。
   - 该仓库反映了类似于 **Sonnet**、**Llama** 和 **GPT-4Omini** 等模型的持续努力。
- **LLaMa 3 作为新选手引入**：最近推出的 **LLaMa 3** 模型采用了拥有 **405B 参数** 的稠密 Transformer 结构，并配备了高达 **128K tokens** 的上下文窗口，旨在处理各种复杂任务。
   - 这些模型在多语言和编程方面表现出色，为 AI 应用树立了新基准。
- **Mistral Large 2 的竞争优势**：拥有 **123B 参数** 和 **128k 上下文窗口** 的 **Mistral Large 2** 的发布吸引了用户，尤其是在编程任务方面。
   - 尽管其采用非商业许可证，但其创新设计使其在 API 性能优化方面处于有利地位。
- **微调 Llama 3 面临挑战**：关于 **微调 Llama 3 405B** 的担忧浮出水面，一些人建议仅将 **Lora FTing** 作为可行方法。
   - 这种情况可能会推动 **OSS** 社区在 **DoRA fine-tuning** 方面的进展。
- **道德推理与电车难题**：围绕引入**困难道德查询**（如**电车难题**）的讨论强调了评估模型道德基础的必要性。
   - 这引发了关于这些任务是考察纯粹的推理能力还是伦理框架的辩论。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek Coder V2 推出私有推理提供商**：**DeepSeek Coder V2** 现在提供 [私有提供商](https://openrouter.ai/models/deepseek/deepseek-coder)，可以在 OpenRouter 上处理请求而无需输入训练数据，这标志着私有模型部署的重大进展。
   - 这一新能力反映了 **OpenRouter** 平台在增强用户可用性方面的战略进展。
- **对 Llama 3.1 405B 性能的担忧**：用户对 **Llama 3.1 405B** 的性能表示不满，特别是在处理 NSFW 内容时，它经常拒绝提示或输出训练数据。
   - 反馈表明温度（temperature）设置显著影响质量，一些用户报告在较低温度下输出效果更好。
- **Mistral Large 2 替代版提供更好的多语言支持**：**Mistral Large 2** 现已作为 **Mistral Large** 发布，有效替代了之前的版本，并增强了多语言能力。
   - 用户推测在处理法语等语言时，它的表现可能优于 **Llama 3.1**。
- **用户讨论 OpenRouter API 的限制**：讨论强调了 **OpenRouter API** 的挑战，特别是在速率限制（rate limits）和多语言输入管理方面，这增加了模型使用的复杂性。
   - 虽然某些模型处于免费预览阶段，但用户报告了对使用量和上下文的严格限制，指出需要改进。
- **对开源编程工具的兴趣日益增长**：用户对 **Devika** 和 **Open Devin** 等开源自主编程工具表现出浓厚兴趣，并根据当前的效能寻求建议。
   - 这种转变反映了用户希望尝试主流 AI 编程解决方案之外的替代方案，因为后者的表现参差不齐。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 震撼发布**：**Llama 3.1** 模型正式发布，将上下文长度扩展至 **128K** 并支持八种语言，标志着开源 AI 的重大进展。可以通过 [blogpost](https://huggingface.co/blog/llama31) 详细了解该模型，并在此处进行测试 [here](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct)。
   - 用户报告 **Llama 3.1 405B** 模型因过载频繁出现“服务不可用”错误，并认为它比 **70B** 版本受到的审查（censored）更多。
- **HuggingChat v0.9.1 版本改进**：最新版本 **HuggingChat v0.9.1** 集成了显著增强用户易用性的新功能。用户可以通过模型页面发现更多功能。
   - 此次更新旨在利用新的 [HuggingChat](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct) 功能来改善交互体验。
- **MultipleNegativesRankingLoss 的风险**：有报告称在使用 **MultipleNegativesRankingLoss** 训练句子编码器（sentence encoders）时遇到困难，增加 batch size 会导致模型性能下降。用户正在寻求关于该方法相关的常见数据集陷阱的见解。
   - 一位用户描述了他们的评估指标，重点关注 **recall@5**、**recall@10** 和 **recall@20**，以实现更好的基准测试。
- **Mistral-NeMo 12B 在 Demo 中表现出色**：使用 [llama.cpp](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp) 展示的 **Mistral-NeMo 12B Instruct** Demo 展现了该模型显著的性能提升。鼓励用户尝试以获得更好的聊天体验。
   - 社区对该模型的能力及其在各种 AI 任务中的潜在应用兴趣激增。
- **关于 Rectified Flow 和评估的问题**：成员们对缺乏关于 **Rectified Flow** 和 **Flow Matching** 的讨论表示沮丧，特别是与 **DDPM** 和 **DDIM** 的辩论相比。他们强调很难找到 **Flow** 应用的简单示例，例如生成 **MNIST**。
   - 探讨了生成模型的评估方法，重点是评估 **Stable Diffusion** 与 **GANs** 等模型性能的定性和定量方法。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Kohya-ss GUI 兼容性问题**：用户报告当前版本的 **Kohya-ss GUI** 与 Python **3.10** 存在兼容性问题，需要升级到 **3.10.9** 或更高版本。
   - *一位用户幽默地评论道*，这就像是需要 **180 磅**的体重限制，但又不能超过 **180.5 磅**。
- **即将推出的 Lycoris 功能令人兴奋**：**Onetrainer** 可能会在一个新的开发分支中集成 **Lycoris** 功能，引发了关于功能增强的讨论。
   - 社区成员表示更倾向于使用 **bmaltais 的 UI 封装器**，这可能会改善这些新集成的体验。
- **社区热议艺术模型**：讨论概述了包括 **Kolors、Auraflow、Pixart Sigma** 和 **Hunyuan** 在内的模型性能评分，其中 **Kolors** 因其速度和质量而受到赞赏。
   - 参与者就这些模型的使用体验和具体应用展开了辩论，展示了多元的观点。
- **显微镜下的 Stable Diffusion 模型**：用户检查了 **Stable Diffusion 1.5** 和 **SDXL** 在输出上的差异，重点关注细节和分辨率。
   - 讨论了 **Hidiffusion** 和 **Adaptive Token Dictionary** 等技术，作为提升旧模型输出的方法。
- **欢迎来到 Stable Video 4D！**：新推出的 **Stable Video 4D** 模型允许将单个物体的视频转换为多视角视图，用于创意项目。
   - 该模型目前处于研究阶段，有望在**游戏开发、视频编辑**和**虚拟现实**领域得到应用。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **深入探讨采样模型**：成员们讨论了各种 **sampling methods**（采样方法），如 **greedy**、**top-p** 和 **top-k**，强调了它们各自的权衡，特别是对于大型语言模型。
   - 随机采样因其多样性而受到关注，但会使评估变得复杂，这与生成最可能路径的 **greedy** 方法的可靠性形成对比。
- **Llama 3.1 的采样偏好**：在关于 **Llama 3.1** 的讨论中，参与者建议参考其论文以获取最佳的 **sampling methods**，并倾向于使用概率采样技术。
   - 一位成员指出，**Gemma 2** 有效地使用了模型评估中常见的 **top-p** 和 **top-k** 策略。
- **误导性推文引发讨论**：成员们分析了一条与 **Character.ai** 模型相关的误导性推文，特别是其使用的共享 KV 层对性能指标的影响。
   - 对此类信息准确性的担忧随之而来，凸显了社区在理解 **Transformer** 架构方面不断探索的过程。
- **MoE 与 Dense 模型的辩论**：一场关于偏好 **dense models**（稠密模型）而非 **Mixture-of-Experts (MoE)** 的激烈辩论展开了，理由是处理 **MoE** 在训练中的高成本和工程挑战。
   - 尽管预训练 **MoE** 具有潜在效率，但对于不同组织实施这些模型的能力仍存在担忧。
- **Llama API 评估困扰**：用户报告了使用 `lm_eval` 工具评估 **Llama 3.1-405B** 时遇到的错误，特别是通过 API 处理 **logits** 和多项选择任务时的挑战。
   - 诸如 'No support for logits' 和 'Method Not Allowed' 之类的错误引发了故障排除讨论，并记录了对 `_create_payload` 方法的成功修改。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 安装故障排除**：成员们遇到了 **Torch** 未针对 **CUDA** 编译而导致的导入错误。建议从官方页面安装 CUDA 版本以确保兼容性。
   - 设置 CUDA 后，一位用户在分配 **172.00 MiB** 时遇到了 **torch.cuda.OutOfMemoryError**，建议调整 **max_split_size_mb** 以解决内存碎片问题。
- **探索 Llama-2 和 Llama-3 特性**：一位成员分享了一个经过微调的 [Llama-2 7B model](https://huggingface.co/TheBloke/Llama-2-7B-fp16)，该模型在 **24GB GPU** 上训练了 **19 小时**。同时，关于在 Llama 3 中实现 **blockwise attention** 的讨论集中在相对于旋转位置嵌入（rotary position embeddings）的序列切分阶段。
   - 此外，还提出了关于 **Llama 3.1** 是否比 **3.0** 改进了推理延迟的询问，反映了对模型性能进步的持续关注。
- **AMD 的 FlashAttention 优化**：继 [GitHub Pull Request #1010](https://github.com/Dao-AILab/flash-attention/pull/1010) 中详述的实现之后，FlashAttention 已获得对 **AMD ROCm** 的支持。更新后的库保持了 API 的一致性，同时引入了几个新的 C++ API，如 `mha_fwd`。
   - 目前新版本的兼容性仅限于 **MI200 和 MI300**，这表明未来可能会有更广泛的更新。
- **PyTorch Compile 见解**：用户报告称 `torch.compile` 增加了小型 **Bert models** 的 **RAM 使用量**，并且从 **eager mode** 切换后性能变差。建议使用 PyTorch profiler 分析推理期间的内存轨迹。
   - 观察结果显示，使用 `reduce-overhead` 和 `fullgraph` 编译选项没有带来内存效率的提升，强调了理解配置效果的重要性。
- **ML/AI 求职策略**：一位用户寻求关于制定实习和全职 **ML/AI** 职位路线图的建议，并分享了一份[包含其计划的 Google 文档](https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing)。他们表达了努力工作并在时间表上保持灵活的承诺。
   - 鼓励对其实习策略提供进一步反馈，突显了愿意投入额外时间来实现目标的意愿。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.1 遭遇错误困扰**：用户报告了 **Llama 3.1** 的问题，面临如 *AttributeError* 等错误，这可能源于过时的镜像或配置。
   - 一位用户通过尝试不同的镜像找到了解决方法，并对持续的模型更新表示沮丧。
- **Mistral 发布超大型模型**：Mistral 发布了拥有 **123B 参数** 的 **Mistral-Large-Instruct-2407** 模型，声称具有 SOTA 性能。
   - 该模型提供多语言支持、精通编程以及先进的 Agent 能力，在社区中引起了轰动。
- **多语言能力受到审视**：**Llama 3.1** 与 **NeMo** 的对比凸显了性能差异，特别是在多语言支持方面。
   - 虽然 **Llama 3** 在欧洲语言方面具有优势，但用户指出 **NeMo** 在 **中文** 和其他语言方面表现更出色。
- **训练大模型遭遇 RAM 瓶颈**：训练像 Mistral 这样的大模型对 RAM 的巨大需求引起了关注，用户对其局限性发表了评论。
   - 一些人在训练过程中遇到了梯度爆炸（exploding gradients），并推测该问题是否与 sample packing 有关。
- **Adapter 微调阶段受到关注**：成员们讨论了 Adapter 微调的多个阶段，提出了用前一阶段的结果（包括用于 DPO 训练的 SFT 权重）初始化后续阶段的想法。
   - [GitHub](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) 上的一个功能请求建议通过少量的代码更改来实现这一方法。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o mini 霸榜 Chatbot Arena**：凭借超过 **4,000 名用户的投票**，**GPT-4o mini** 目前在 Chatbot Arena 排行榜上并列第一，性能超越前代版本，且价格便宜 **20 倍**。这一里程碑标志着新应用的**智能成本**显著下降。
   - 开发者们对此成就感到兴奋，并指出了其对未来聊天机器人体验的影响。
- **Mistral Large 2：新的竞争者**：**Mistral Large 2** 拥有 **128k 上下文窗口**和多语言支持，在特定许可条件下，其定位非常适合高复杂度任务。关于该强大模型的*商业用途*缺乏清晰度的讨论浮出水面。
   - 成员们强调需要更好的文档来有效应对**许可（licensing）**环境。
- **OpenAI 预计亏损 50 亿美元**：据估计，OpenAI 今年可能面临高达 **50 亿美元** 的巨额亏损，主要原因是 Azure 成本和训练费用。对盈利能力的担忧引发了关于 API 收入低于预期的讨论。
   - 这种情况对 OpenAI 商业模式在当前环境下的可持续性提出了根本性挑战。
- **Llama 3 正式发布**：Meta 已[正式发布](https://llama.meta.com/)在 **15T tokens** 上训练的 **Llama3-405B**，声称在**所有主要基准测试中均超越了 GPT-4**。这标志着开源 AI 技术的重大飞跃。
   - 此次发布引发了关于在模型后训练能力中集成 **100% RLHF** 的讨论，突显了该方法的关键作用。
- **CrowdStrike 为停机事件提供 10 美元道歉礼品卡**：**CrowdStrike** 向合作伙伴提供 **10 美元的 Uber Eats 礼品卡**，作为对大规模停机事件的道歉，但一些人发现礼品卡在尝试兑换时已被**取消**。这一事件凸显了与技术更新相关的运营风险。
   - 成员们对这一举措在持续的挫败感面前的有效性表达了复杂的情绪。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 编译器版本命名困惑**：一场讨论突显了关于下一个主编译器版本是 **24.5** 还是 **24.8** 的不确定性，理由是随着向 **2025** 年推进，nightly 和 main 版本之间可能存在脱节。
   - 社区成员对遵循不同的发布原则表示担忧，这使未来的更新变得复杂。
- **最新 Nightly 更新解析**：最新的 nightly Mojo 编译器更新 `2024.7.2405` 包含重大更改，例如移除了 **DTypePointer** 并增强了字符串格式化方法，详细信息可以在 [当前变更日志](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 中查看。
   - **DTypePointer** 的移除需要对现有项目进行代码更新，这引发了对更清晰过渡指南的呼吁。
- **SDL 集成问题**：一位用户请求有关将 **SDL** 与 **Mojo** 集成的资源，旨在更好地理解该过程以及如何有效地使用 **DLHandle**。
   - 这反映了通过第三方库增强 Mojo 功能的日益增长的兴趣。
- **关于 Var 与 Let 实用性的讨论**：一位成员发起了一场关于在所有内容都已声明为 **var** 的情况下使用它的必要性的辩论，认为这种用法存在冗余。
   - 另一位成员指出 **var** 有助于编译器，而 **let** 则迎合了那些偏好不可变性的人，突显了开发者之间的偏好之争。
- **探索 SIMD 类型可比性**：成员们讨论了为 **SIMD types** 建立全序关系的挑战，并指出了泛型编程与特定比较之间的张力。
   - 有人提议，新的 **SimdMask[N]** 类型可能会缓解一些与平台特定行为相关的复杂性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Factorio 自动化模组激发创意**：新的 [factorio-automation-v1](https://github.com/naklecha/factorio-automation) 模组允许 Agent 在 *Factorio* 中自动执行合成和采矿等任务，为 Agent 能力提供了一个有趣的测试场。
   - 成员们对该模组为复杂游戏交互带来的可能性感到兴奋。
- **GPT-4o mini 微调开放**：OpenAI 推出了 **GPT-4o mini** 的微调功能，面向第 4 级和第 5 级用户开放，在 9 月 23 日之前每天前 **2M training tokens** 免费。
   - 成员们注意到，在将微调后的 **GPT-4o mini** 与 **Llama-3.1-8b** 进行比较时，性能存在不一致，这引发了关于具体用例的疑问。
- **Mistral Large 2 以 123B 参数给人留下深刻印象**：[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 已发布，拥有 **123 billion parameters**，具备强大的编程能力，并支持多种语言。
   - 然而，有迹象显示它在 **Aider's code editing benchmark** 中仅获得了 **60% score**，略领先于最好的 GPT-3.5 模型。
- **Reddit 的内容政策引发辩论**：关于 Reddit [公共内容政策](https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy) 的讨论非常激烈，主要集中在用户对生成内容的控制权上。
   - 成员们认为模糊的政策造成了重大问题，强调了对更清晰指南的需求。
- **加入 Llama 3 紧急论文俱乐部**：一场关于 [The Llama 3 Herd of Models](https://x.com/latentspacepod/status/1816151808357908698) 的“紧急论文俱乐部”会议定于今天晚些时候举行，该论文是 **POTY Awards** 的有力竞争者。
   - 讨论的主要贡献者包括著名的社区成员，强调了该论文的重要性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 增强 Markdown 功能**：**LlamaParse** 现在展示了对 **Markdown 输出**、**纯文本**和 **JSON 模式**的支持，以便更好地进行元数据提取。**多语言**输出等功能增强了其在工作流中的实用性，正如[此视频](https://t.co/RUWJ0Z2NMn)所示。
   - 此次更新将显著提高各种应用的 **OCR** 效率，将其应用范围从简单的文本扩展到各种任务。
- **MongoDB AI 应用计划现已推出**：新启动的 **MongoDB AI Applications Program (MAAP)** 旨在简化组织构建 **AI 增强型应用**的流程。通过**参考架构**和集成技术栈，它加速了 AI 的部署时间；点击[此处](https://t.co/rCz3DfUe3A)了解更多。
   - 该计划解决了开发者以最小开销实现应用现代化的迫切需求，有助于提高工作流效率。
- **Mistral Large 2 引入 Function Calling**：**Mistral Large 2** 正在推出增强的 **function calling 能力**，其中包括在发布时即支持结构化输出。官方提供了 **cookbooks** 等详细资源来帮助开发者利用这些新功能；点击[此处](https://t.co/ho02wDbGpZ)进行探索。
   - 此次发布强调了 LLM 应用的**功能通用性**，允许开发者有效地实现更复杂的交互。
- **使用 SubQuestionQueryEngine 提高流式传输效率**：成员们讨论了使用 **SubQuestionQueryEngine.from_defaults** 来促进流式响应并降低 LLM 查询中的延迟。虽然在使用 `get_response_synthesizer` 方面提出了一些解决方案，但在实现上仍面临挑战。
   - 尽管在采用过程中存在障碍，但人们对提高 LLM 集成中的用户交互速度持乐观态度。
- **对 Llama 3.1 指标产生质疑**：对于 Meta 发布的 **Llama 3.1** 指标，尤其是其在 **RAG 评估**中的有效性，怀疑情绪正在增加。用户正在质疑某些模型（如 `llama3:70b-instruct-q_5`）在实际任务中的可行性。
   - 这种怀疑反映了社区对 AI 指标在评估各种应用中模型性能的可靠性的广泛关注。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 控制面板刷新问题**：成员们报告了 **Cohere 账户控制面板**不断刷新的问题，而其他人则表示他们那边没有此类问题，引发了关于潜在故障的讨论。
   - 这引发了关于 **rate limiting**（速率限制）可能是导致刷新问题原因的讨论。
- **为 Command R Plus 喝彩**：随着 **Llama 3.1** 等模型的每一次发布，成员们对 **Command R Plus** 的评价越来越高，强调了其与其他模型相比的能力。
   - 一位用户建议专门为**模型对比**创建一个 Playground，以进一步探索这种日益增长的情绪。
- **服务器性能受到关注**：虽然有人担心潜在的服务器宕机，但一些用户确认服务器处于**完全运行状态**。
   - 建议包括调查 **rate limiting** 是否是影响用户体验的一个因素。
- **Cohere 的创新功能建议**：一位成员建议在 **Cohere** 的对话中加入使用工具的能力，例如根据需求触发网页搜索。
   - 最初出现了一些困惑，但随后澄清了其中一些功能已经可以使用。
- **社区欢迎新面孔**：新成员介绍了自己，分享了在 **NLP** 和 **NeuroAI** 方面的背景，引发了社区的兴奋。
   - 讨论还涉及了使用 **Command-R+** 的经验，强调了其相对于 **NovelAI** 等模型的优势。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Zenbase/Core 发布引发关注**：**zenbase/core** 现已上线，使用户能够将 **DSPy 的 optimizers** 直接集成到 Instructor 和 LangSmith 等 Python 项目中。可以通过参与他们的 [Twitter 帖子](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ) 来支持此次发布。
   - 社区成员反应积极，非常愿意推广这一最新版本。
- **Typed Predictors 引发输出担忧**：用户报告 **typed predictors** 无法生成正确结构化输出的问题，并寻求他人帮助。建议包括通过 `dspy.configure(experimental=True)` 启用实验性功能来解决这些问题。
   - 同行的鼓励凸显了完善这些 predictor 使用方法的集体努力。
- **内部执行可见性引发辩论**：关于观察内部程序执行步骤的方法（包括 `inspect_history` 等建议）展开了热烈讨论。用户表示需要更深入地了解模型输出，特别是在类型检查出错期间。
   - 对透明度的共同渴望展示了调试工具在 DSPy 使用中的重要性。
- **推动 Small Language Models 发展**：一位成员分享了一篇关于 **small language models** 优势的文章，指出它们的高效性以及对资源有限的边缘设备的适用性。他们强调了在仅有 **4GB RAM** 的设备上运行模型的 **privacy** 和操作简便性等优点。
   - 阅读文章 [Small Language Models are the Future](https://medium.com/thoughts-on-machine-learning/small-language-models-are-the-future-6e8909567198) 以获取有关该主题的全面解读。
- **呼吁为 DSPy 示例做贡献**：一位用户表示有兴趣向 DSPy 仓库贡献初学者友好的示例，旨在丰富资源库。社区反馈确认了对更多样化示例的需求，特别是在 `/examples` 目录下。
   - 这一倡议反映了增强 DSPy 环境内学习材料的协作精神。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **成员攻克 Tinygrad 学习**：成员们分享了学习 **Tinygrad** 的历程，重点是理解其在 **transformers** 方面的应用。有人指出，“这还在进行中”，表明这是一个循序渐进的掌握过程。
   - 讨论暗示了可能通过集体资源来提升学习曲线。
- **分子动力学引擎正在构建中**：一个团队正在开发一个使用神经网络进行能量预测的 **Molecular Dynamics engine**，并面临梯度使用方面的挑战。建议使用输入梯度跟踪方法来优化 backpropagation 过程中的权重更新。
   - 优化 backpropagation 成为提高训练性能的焦点。
- **在 Tinygrad 中创建自定义运行时**：一位成员分享了为 Tinygrad 实现 **custom runtime** 的见解，强调了为新硬件添加支持是多么简单。他们寻求对 `global_size` 和 `local_size` 等术语的澄清，这些术语对于 kernel 执行至关重要。
   - 针对这些参数的操作上下文提供了技术澄清。
- **神经网络势能讨论**：Molecular Dynamics engine 中的能量依赖于 **Neural Network Potentials (NNP)**，重点在于计算效率。对话围绕优化 backpropagation 的策略展开。
   - 提高计算速度的清晰路径对于改善结果至关重要。
- **对 CartPole 中 PPO 算法的审查**：一位成员探究了在 Beautiful CartPole 环境的 **PPO algorithm** 实现中 `.sum(-1)` 操作的必要性。这引发了关于强化学习细微差别的协作对话。
   - 对代码实现的详细探索促进了社区的理解和知识共享。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **3.1 版本倒计时与精彩访谈**：成员们询问是否会随 **3.1** 版本发布一些精彩的 [访谈](https://github.com/pytorch/torchtune/pull/790)，类似于 **Llama3** 发布时的内容。
   - 这引发了人们对新版本可能伴随的见解和讨论的兴趣。
- **MPS 支持 PR 引起关注**：一个新拉取请求 ([#790](https://github.com/pytorch/torchtune/pull/790)) 受到关注，该 PR 为本地 Mac 电脑增加了 **MPS** 支持，并检查 BF16 兼容性。
   - 上下文表明，该 PR 可能会解决使用 MPS 设备的用户面临的主要测试障碍。
- **LoRA 功能问题依然存在**：讨论了围绕 **LoRA** 功能的问题，指出在之前的尝试中它无法正常工作，且此前受到硬编码 **CUDA** 路径的影响。
   - 成员们就遇到的具体错误交换了意见，突出了实现过程中持续存在的挑战。
- **修复 Pad ID Bug**：一名成员指出 **pad id** 不应出现在生成功能中，并将其确定为一个重要的 bug。
   - 作为回应，一个拉取请求被创建以防止 **pad ids** 和特殊 token 显示，详见 [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211)。
- **优化 Git 工作流以减少冲突**：讨论了改进 git 工作流以尽量减少不断出现的新冲突，强调了协作。
   - 有建议认为新冲突可能源于工作流，表明可能需要进行调整。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hugging Face 模型与 Agents 讨论**：成员们讨论了使用 **Hugging Face 模型构建 Agents** 的经验，包括通过 **Ollama** 使用本地 LLM 以及 **OpenAI** 和 **Azure** 等云端选项。
   - 这场对话激发了人们对 Agents 在各种模型框架中潜在应用的兴趣。
- **Python 开发者求职**：一名成员迫切地表达了他们的处境，称：**“有人想雇佣我吗？我需要付账单。”** 并强调了他们在 **Python** 方面的强大技能。
   - 随着对机会的讨论展开，当前市场中职位可用性的紧迫性显而易见。
- **Aurora 上 HNSW IVFFLAT 索引的挑战**：成员们在 **Aurora PGVECTOR** 上创建 **3072 维度** 的 **HNSW** 或 **IVFFLAT** 索引时遇到问题，并分享了涉及 **halfvec** 的解决方案。
   - 这突出了高性能向量数据库中维度管理方面持续存在的挑战。
- **LangServe 的 OSError 限制**：用户在 LangServe 应用处理约 **1000 个并发请求** 时遇到了 **OSError: [Errno 24] Too many open files**。
   - 他们正在积极寻求处理高流量同时缓解系统资源限制的策略，并已提交 [GitHub issue](https://github.com/langchain-ai/langserve/issues/714) 以寻求支持。
- **AI Code Reviewer 工具介绍**：一名成员分享了一个关于 **AI Code Reviewer** 的 [YouTube 视频](https://youtu.be/g_VRsjpC4e8)，重点介绍了其由 **LangChain** 驱动的功能。
   - 该工具旨在增强 **code review 过程**，暗示了代码评估方法向自动化发展的趋势。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.1 405 B 的易用性令人印象深刻**：**Llama 3.1 405 B** 与 [OpenInterpreter](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587) 配合使用时表现极其出色，提供了轻而易举的使用体验。
   - 相比之下，**gpt-4o** 需要不断提醒其具备的功能，这使得 405b 成为多任务处理的更优选择。
- **Nvidia 提供的性价比 API 使用方案**：一位用户分享到，**Nvidia** 在注册时会提供 **1000 credits**，其中 1 credit 等于 1 次 API 调用。
   - 这一激励措施为尝试 API 提供了更多的可访问性。
- **Mistral Large 2 与 Llama 3.1 405 B 旗鼓相当**：据报道，**Mistral Large 2** 的表现与 **Llama 3.1 405 B** 相当，尤其在速度方面表现突出。
   - 较快的性能可能是由于 Mistral 的端点流量比 Llama 的端点流量低。
- **Llama 3.1 免费连接数据库**：[MikeBirdTech](https://x.com/MikeBirdTech/status/1816163862208766137) 指出，**Llama 3.1** 可以通过 **OpenInterpreter** 免费与你的数据库进行交互，强调了在付费服务上的节省。
   - *它也是完全离线和私密的，无需他人看到你的数据*，突显了其**隐私优势**。
- **对 Llama 3.1 处理复杂数据库的担忧**：一名成员提出担忧，认为对于涉及跨表连接（joins）的**复杂数据库**，该解决方案可能无效。
   - 他们对分享这些信息表示感谢，并评论说尽管存在局限性，但执行得**非常出色**。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Llama 3.1：Meta 的开源突破**：Meta 最近推出了 **Llama 3.1 405B**，被誉为有史以来第一个**开源的前沿 AI 模型**，在各种基准测试中表现优于 GPT-4o 等竞争模型。欲了解更多见解，请查看这段 [YouTube 视频](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61)，其中 Mark Zuckerberg 讨论了其影响。
   - 这一反响突显了该模型对 AI 研究和开源贡献的潜在影响。
- **下载 LAION2B-en 元数据遇到困难**：成员们报告在从 Hugging Face 查找和下载 **LAION2B-en 元数据**时遇到困难，并询问其他人是否面临同样的问题。回复表明这是对可访问性的普遍挫败感。
   - 有人链接到了 [LAION 维护说明](https://laion.ai/notes/laion-maintenance/)，以进一步澄清情况。
- **LAION 数据集处于法律悬而未决状态**：讨论显示 **LAION 数据集**目前处于**法律灰色地带**，官方版本的访问受到限制。虽然有替代方案，但建议仅在紧急研究需求时使用非官方数据集。
   - 成员们注意到了 AI 社区中围绕数据合法性的持续复杂性。
- **YouTube 投票：一场怀旧辩论**：一名成员分享了一个 [YouTube 投票](http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS)，询问**哪部 90 年代的电影拥有最好的原声带**，引发了观众的怀旧之情。这促使成员们反思他们最喜欢的那个时代的电影原声带。
   - 该投票通过共同的文化体验激发了联系。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **关于 ML 数据集版权的法律明确性**：一名成员指出，由 **ML 模型**生成的大多数数据集可能不具备版权，因为它们缺乏真正的创造力。他们强调，非 **GPT-4** 生成的内容可能属于 **MIT licensing**，尽管在当前的法律辩论中这一领域仍然模糊不清。
   - 这开启了关于数据集策划中**数据所有权**和伦理准则影响的讨论。
- **探索非蒸馏数据识别**：围绕在 ML 数据集中定位**非蒸馏数据（non-distilled data）**的方法展开了讨论，突显了对系统化数据管理的兴趣。
   - 成员们寻求更清晰的方法论来增强数据集内容的组织，旨在提高 ML 项目中的可用性。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **翻译模型的 DPO 实验**：一位成员询问了使用 **DPO** 成功微调翻译模型的经验，并参考了 [CPO 论文](https://arxiv.org/abs/2401.08417) 的见解。他们强调 **中等规模的 LLM** 无法达到 SOTA 性能。
   - *“是否有人取得了更好的结果？”* 凸显了社区对微调技术日益增长的兴趣。
- **CPO 增强翻译输出**：**CPO 方法** 针对监督微调的弱点，旨在提高机器翻译输出的质量。它将重点从仅“可接受”的翻译转向更高质量的结果，从而提升模型性能。
   - 通过解决参考数据的质量问题，CPO 带来了显著的增强，特别是有效地解决了数据集利用不足的问题。
- **ALMA-R 证明了竞争力**：尽管仅在 **2.2万个平行句子** 和 **12M 参数** 上进行训练，应用 **CPO** 仍显著提升了 **ALMA-R**。该模型现在可以与传统的 encoder-decoder 架构相媲美。
   - 这展示了即使在有限数据下优化 LLM 的潜力，引发了关于效率和扩展性的讨论。
- **8 月下旬的纽约技术聚会**：成员们对 8 月下旬在纽约举行的技术聚会表现出浓厚兴趣，表达了线下交流的愿望。这一倡议有望促进更深层次的网络联系和协作机会。
   - 围绕这次潜在聚会的讨论凸显了成员们渴望分享见解和经验的社区意识。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **通过 Feature Stores 提升 ML 效率**：一场关于 [利用 Feature Stores](https://tinyurl.com/yfjscesh) 的 **直播会议** 定于 **2024 年 7 月 31 日上午 11:00 (EDT)** 举行，面向 ML 工程师、数据科学家和 MLOps 专业人员。
   - 本次会议将探讨 **自动化流水线 (automated pipelines)**、解决不可靠数据问题，并展示高级用例以增强 **可扩展性 (scalability)** 和 **性能**。
- **解决数据一致性挑战**：网络研讨会将强调对齐 **serving 和 training 数据** 的重要性，以创建可扩展且可复现的 ML 模型。
   - 讨论将重点关注 **数据格式不一致** 和特征重复等常见问题，旨在加强 ML 团队内部的协作。
- **加强特征治理实践**：参与者将学习实施 **特征治理和版本控制 (feature governance and versioning)** 的有效技术，这对于管理 ML 生命周期至关重要。
   - 与会者可以期待获得改进其 ML 流程和推进运营的见解和实用工具。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **加速器申请截止日期临近**：**加速器项目** 的申请截止日期即将到来，该项目为期 **12 周**，为项目提供高达 **100k 的非稀释性资金**。
   - 计划与 Mozilla 共同举行 **Demo Day**，鼓励成员在 [此处](https://discord.com/channels/1089876418936180786/1245083732319408195) 提出他们的 **问题**。
- **即将举行的两场精彩活动**：提醒本月还有两场 **即将举行的活动**，将展示知名参与者的工作，为社区带来新鲜见解。
   - 这些活动由两位成员发起，进一步增强了社区参与度。
- **深入的 Zero Shot Tokenizer Transfer 讨论**：一场名为 **Zero Shot Tokenizer Transfer** 的会议将与 Benjamin Minixhofer 共同举行，旨在探索高级 Tokenizer 实现。
   - 详细信息和参与链接可以在 [此处](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732) 找到。
- **AutoFix：开源问题修复工具发布**：发布了关于 **AutoFix** 的公告，这是一个开源的问题修复工具，可以从 Sentry.io 提交 PR，从而简化开发者的工作流程。
   - 有关该项目的更多信息可以在 [此处](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732) 获取。



---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3.1 论文：开源界的宝藏**：Meta 发布的 [Llama3.1 论文](https://threadreaderapp.com/thread/1815789501026861308) 被誉为对开源社区极具**价值**，引发了关于其深刻见解的讨论。
   - *一位成员开玩笑说*，它包含如此多的 **alpha**，以至于*你必须像看最喜欢的电影一样反复阅读*。
- **使用 15T Tokens 训练 405B 模型**：论文透露，拥有 **4050 亿参数 (405B)** 的模型是使用 **~15 万亿 (15T) Tokens** 训练的，这是通过外推其 Scaling laws 预测出来的。
   - *Scaling law 建议*在 **16.55T Tokens** 上训练一个 **402B 参数模型** 以获得最佳结果。
- **关于网络拓扑的见解**：论文中包含对其 **24k H100 集群** 所使用的**网络拓扑 (Network Topology)** 令人惊讶的详细描述。
   - 帖子中分享的图片展示了其**架构**，体现了基础设施的规模。
- **由于服务器问题导致的训练中断**：Llama3-405b 训练过程中的两次中断归因于 **'服务器机箱 (Server Chassis)'** 故障，有人幽默地暗示这是由某人的失误造成的。
   - 因此，在预训练期间由于这些故障损失了 **148 个 H100 GPU**。
- **关于幻觉预防基准测试的讨论**：与 Meta 工程师的简短对话引发了对**幻觉预防 (Hallucination Prevention)** 技术需要更好**基准测试 (Benchmarks)** 的关注。
   - 该成员分享道，*任何其他从事此项工作的人*都应该参与进一步的讨论。



---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。


---


**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂太久，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1265385753660100608)** (772 条消息🔥🔥🔥): 

> - `Unsloth 与 Llama 3.1 微调 (Fine-Tuning)`
> - `AI 在软件开发中的应用`
> - `图像生成模型`
> - `Mistral 模型`
> - `AI 隐私担忧` 


- **Llama 3.1 微调的挑战**：几位用户报告了在微调 Llama 3.1 模型时遇到的问题，特别是与模型配置和 Tokenizer 处理相关的错误消息。
   - 建议更新 Transformers 库以解决其中一些问题，用户讨论了确保使用正确模型版本的重要性。
- **AI 在软件开发与就业保障中的角色**：参与者讨论了 AI 在软件开发中不断演变的角色，强调了初级开发人员对 AI 工具融入编码实践后就业保障的担忧。
   - 大家达成共识，经验丰富的开发人员可以利用 AI 来提高生产力，而不是取代他们的角色，强调要适应新工具。
- **图像生成与多样性问题**：对话转向图像生成工具，成员们反思了在生成内容中实现多样性的挑战以及 AI 模型中偏见的影响。
   - 虽然一些人认为确保多样性的尝试值得称赞，但也有人批评这些尝试的执行方式及其对历史背景和教育用途的影响。
- **Mistral 模型与竞争**：讨论包括 Mistral Large 2 的新功能，它以广泛的上下文窗口 (Context window) 和多语言支持著称，是现有大模型的强力替代方案。
   - 通过与 Llama 模型的对比，突显了 AI 模型领域的竞争格局以及为提升性能而进行的持续努力。
- **AI 隐私与数据处理**：提出了关于 AI 数据处理相关的隐私问题，特别是人类审核员访问敏感数据的潜在影响。
   - 参与者讨论了实施妥善数据管理实践的必要性，并认为某些 AI 工具可能正在以可能损害用户隐私的方式使用数据。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: 今天，我们宣布推出 Mistral Large 2，这是我们旗舰模型的新一代产品。与前代相比，Mistral Large 2 在代码生成、数学和推理能力方面有显著提升...</li><li><a href="https://open.spotify.com/episode/29TW4HAocRcV71kZYCFay8?si=e7SaXoM7S6ODvInF7lTQDA&t=2816">801: Merged LLMs Are Smaller And More Capable, with Arcee AI&#x27;s Mark McQuade and Charles Goddard</a>: 在 Spotify 上收听来自 Super Data Science: ML &amp; AI Podcast with Jon Krohn 的这一集。合并后的 LLMs 是未来，我们将与来自 Arcee AI 的 Mark McQuade 和 Charles Goddard 一起探讨如何实现...</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn">Multi-turn conversations - QnA Maker - Azure AI services</a>: 使用提示和上下文来管理机器人的多个轮次（称为 multi-turn），引导用户从一个问题进入另一个问题。Multi-turn 是指进行来回对话的能力，其中之前的...</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://deepmind.google/technologies/imagen-3/">Imagen 3</a>: Imagen 3 是我们最高质量的 text-to-image 模型，能够生成比我们之前的模型具有更好细节、更丰富光影且更少干扰伪影的图像。</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit/tree/main">unsloth/Meta-Llama-3.1-8B-bnb-4bit at main</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-finetuning">Continued Pretraining | Unsloth Documentation</a>: 又名持续微调（Continued Finetuning）。Unsloth 允许你进行持续预训练（continued pretraining），使模型能够学习一种新语言。</li><li><a href="https://huggingface.co/datasets">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Instruct-8b-Merged">Replete-AI/Replete-Coder-Instruct-8b-Merged · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets?search=multi%20turn">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://github.com/mixedbread-ai/binary-embeddings/blob/main/mxbai_binary_quantization.ipynb">binary-embeddings/mxbai_binary_quantization.ipynb at main · mixedbread-ai/binary-embeddings</a>: 展示如何使用 mxbai-embed-large-v1 生成 binary embedding。Binary embeddings 可节省 32 倍的存储空间，并使检索速度提高 40 倍。- mixedbread-ai/binary-embeddings</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=s">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/catcathh/UltraPixel">GitHub - catcathh/UltraPixel: Implementation of UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks</a>: UltraPixel 的实现：将超高分辨率图像合成推向新高峰 - catcathh/UltraPixel</li><li><a href="https://x.com/tsarnick/status/1758323312483303443">Tweet from Tsarathustra (@tsarnick)</a>: Sora 的性能随计算量扩展</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">no title found</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/XdrEUyIrgl">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.43.2/en/main_classes/pipelines#transformers.TextGenerationPipeline">Pipelines</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp310-cp310-linux_x86_64.whl">no title found</a>: 未找到描述</li><li><a href="https://download.pytorch.org/whl/cu118/xformers-0.0.26.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl">no title found</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 以 2-5 倍的速度和减少 80% 的内存微调 Llama 3.1, Mistral, Phi &amp; Gemma LLMs</li>

mory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1265401304189239448)** (1 messages): 

> - `Llama 3.1 发布`
> - `性能提升`
> - `新 UI 功能`
> - `Google Colab Notebooks`
> - `4-bit 模型` 


- **Llama 3.1 发布了！🦙**：Unsloth 现在支持 **Llama 3.1**，使训练速度比以前的版本快 **2.1 倍**，且内存占用减少 **60%**。该模型已在 **15.6T tokens** 上进行了训练，并将上下文长度扩展至 **128K**。
   - Meta 的更新将 Llama 3.1 定位为迄今为止**最先进的模型**，支持新语言并增强了性能。
- **适用于 Llama 3.1 的 Google Colab Notebooks**：提供了一个 [Google Colab notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)，用于在免费的 Tesla T4 上微调 **Llama 3.1 (8B)**，为用户简化了访问流程。
   - 还提供了 Kaggle 和推理 UI 的 notebooks，以增强用户交互，邀请大家进行实验和测试。
- **Llama 3.1 的新 UI 功能**：Unsloth 引入了一个[新的推理 UI](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)，用于在 Colab 中与 Llama 3.1 Instruct 模型进行交互。
   - 这一用户友好的功能旨在提升模型的整体体验和参与度。
- **令人兴奋的实验机会！**：团队鼓励用户之间分享、测试和讨论模型及结果，旨在促进协作和反馈。
   - 这种社区驱动的方法是 **Unsloth Studio** 内部更广泛开发推动的一部分。
- **探索 Llama 3.1 的 4-bit 模型**：Llama 3.1 的 4-bit 模型提供多种尺寸，包括 [8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit) 和 [70B](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit)。
   - 模型选项针对 Base 和 Instruct 类别进行了量身定制，增强了开发者的灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/llama3-1">使用 Unsloth 微调 Llama 3.1</a>: 通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，上下文长度增加 6 倍！</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)!">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1265473867305189448)** (77 messages🔥🔥): 

> - `LLaMA3.1 上的 Abliterator`
> - `OpenAI API vs 开源模型`
> - `微调 vs RAG 复杂度`
> - `企业内部知识`
> - `L3-8B-Stheno-v3.2 数据集请求` 


- **关于 Abliterator 和 LLaMA3.1 的讨论**：成员们对 *abliterator* 在 **LLaMA3.1** 上的效果感到好奇，但尚未分享明确的经验。
   - 他们表示需要关于这种集成的成功案例。
- **OpenAI API 与开源模型的成本比较**：对话围绕使用 OpenAI 的 *chat API* 与开源模型的**成本效益**展开，强调了开销和硬件费用。
   - 成员们指出，对于初创公司来说，使用 OpenAI API 通常意味着更低的初始成本和更小的运营风险。
- **微调 vs RAG**：会议强调，虽然**微调**在初期被认为更便宜、更简单，但实施 **RAG** 需要大量的专业知识和时间投入。
   - 成员们一致认为，RAG 需要精心设计以避免复杂性，并仍能在生产环境中提供有效的结果。
- **企业内部知识的重要性**：讨论强调了模型通常缺乏**企业内部知识**，因此在企业应用中需要进行微调以确保准确性。
   - 成员们强调，针对特定企业上下文进行微调对于避免不准确性至关重要。
- **L3-8B-Stheno-v3.2 数据集请求**：一位成员请求 **L3-8B-Stheno-v3.2** 的数据集，并对现有数据集包含过多虚构内容表示失望。
   - 另一位成员指出，现在很少有人分享他们的数据集，这表明了可访问性受限的趋势。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1265390693002776717)** (147 条消息🔥🔥): 

> - `循环训练问题`
> - `Unsloth 与 Hugging Face 模型加载`
> - `Llama 3.1 微调`
> - `使用 FastLanguageModel`
> - `微调模型的推理` 


- **循环训练导致 OOM**：一位用户报告说，在循环中使用 `train()` 会导致 VRAM 爆炸，并在第一次训练迭代后出现显存溢出 (OOM) 错误。
   - 他们提到启用了 gradient checkpointing，并正在通过检查配置进行排查。
- **使用 Unsloth 加载模型**：用户在尝试加载模型 'unsloth/meta-llama-3.1-8b-bnb-4bit' 时遇到了 OSError，这表明需要仔细检查模型路径并确保本地目录不冲突。
   - 对于加载本地模型文件，用户讨论了使用特定路径进行直接加载，而不是从 Hugging Face 拉取。
- **Llama 3.1 微调问题**：一些用户注意到在使用各种数据集格式微调 Llama 3.1 时存在问题，质疑提示词格式是否影响了他们的结果。
   - 此外，还有关于使用适当训练配置以确保微调期间达到预期 loss 的指导。
- **FastLanguageModel 的利用**：已确认必须使用 FastLanguageModel 才能实现 Unsloth 所声称的推理速度提升。
   - 用户对如何在 VLLM 中有效实现该模型以获得更快性能表现出兴趣。
- **微调模型的推理**：一位用户成功微调了模型并将其推送到 Hugging Face，但正在寻求关于如何有效运行推理的建议。
   - 建议包括使用 Unsloth 的推理代码或 VLLM 来简化生产环境测试的部署流程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/">llama-tokenizer-js playground</a>：未找到描述</li><li><a href="http://www.unsloth.ai">Unsloth AI | Finetune Llama 3 &amp; Mistral LLMs</a>：为 AI 和 LLM 提供简便的微调。开源且适合初学者。使用 Unsloth 获得更快速度。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-">Continued Pretraining | Unsloth Documentation</a>：又称持续微调。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-finetuning">Continued Pretraining | Unsloth Documentation</a>：又称持续微调。Unsloth 允许你进行持续预训练，使模型能够学习新语言。</li><li><a href="https://docs.unsloth.ai/get-started/installation/updating">Updating | Unsloth Documentation</a>：要更新 Unsloth，请遵循以下步骤：</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>：微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git#egg=unsloth[colab-new]">GitHub - unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth</a>：微调 Llama 3.1, Mistral, Phi 和 Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/issues/6689">[Model] Meta Llama 3.1 Know Issues &amp; FAQ · Issue #6689 · vllm-project/vllm</a>：请查看 vLLM 中发布 Llama 3.1 支持的公告。所有 Llama 3.1 模型均已开启分块预填充 (Chunked prefill)。然而，目前它与前缀缓存 (prefix caching)、滑动窗口 (sliding window) 和 multi-lora 不兼容...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1265386782464282747)** (17 条消息🔥): 

> - `用于合成数据集的 LLaMa-3.1`
> - `Vision Language Models 中 attention masks 的使用`
> - `推理速度 vs 训练速度`
> - `不同模型尺寸的解码` 


- **使用 LLaMa-3.1 生成合成数据**：成员们讨论了利用 **LLaMa-3.1** 生成合成数据集，但许多人一致认为，使用 **405B 模型** 是此用途的理想选择。
   - *一位成员指出，*
- **澄清 Vision Language Models 中的 Attention Masks**：一位成员描述了使用包含 48 个 patch 的 attention mask，并表示该 mask 有效地结合了句子 mask 和 patch mask。
   - 他们明确指出，在 decoder-only 设置下，图像 patch 的 attention mask 应该与句子 token 保持一致。
- **推理速度慢于模型训练**：一位成员提出了关于为什么推理（Inference）明显慢于训练的问题，并指出数据处理速率存在鲜明对比。
   - 虽然训练每分钟可以处理数百个数据点，但推理通常每秒仅处理 **30-100 tokens/s**。
- **使用 8B 模型进行数据格式化**：讨论显示 **8B 模型** 可能被用于数据格式化或微调，尽管这并不是合成数据生成的主要焦点。
   - 成员们承认，合成的主要目标由更大的模型能更好地完成。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: 旨在用于 Llama 模型的实用工具。通过在 GitHub 上创建账户，为 meta-llama/llama-models 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1265406391074427004)** (192 messages🔥🔥): 

> - `LM Studio 和 Llama 3.1`
> - `Nemo 模型性能`
> - `模型下载问题`
> - `Claude Sonnet 3.5 作为编程模型`
> - `模型推理的 GPU 使用` 


- **LM Studio 与 Llama 3.1 的兼容性**：用户讨论了 LM Studio 无法在 OpenCL 显卡上运行 Llama 3.1 的问题，并建议从官网升级到 0.2.28 版本以获得更好的支持。
   - 几位成员确认，LM Studio 的最新更新对于有效运行大型模型至关重要。
- **Nemo 模型及其性能问题**：据报道 Nemo 模型在当前版本上可以运行，但用户在 context length 方面面临挑战，且由于 RAM 有限导致输出速度较慢。
   - 一名用户确认在特定配置下运行成功，而其他用户则提出了改进和优化的建议。
- **模型的下载与访问问题**：一些用户在从 Hugging Face 下载模型时遇到问题，报告的情况从区域 CDN 问题到浏览器缓存问题不等。
   - 其他人确认特定链接可以访问，而另一些人则遇到了 'Entry not found' 错误。
- **Claude Sonnet 3.5 作为编程任务的基准**：用户表示目前的本地模型在编程能力上还无法与 Claude Sonnet 3.5 相比，特别是在生成完整的可用代码方面。
   - 建议探索替代方案并尝试低量化 (lower quantization) 的 Claude 模型作为潜在解决方案。
- **GPU Offloading 对 AI 模型的重要性**：讨论强调，与 GPU 相比，使用 CPU 进行推理 (inference) 会导致输出速度变慢，强调了需要选择能放入 GPU VRAM 的合适模型。
   - 鼓励用户寻找标记为 'full GPU offload' 的模型，以最大限度地提高性能并减少推理时间。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://pastebin.com/uW5fJtLF">🔎 Read the user prompt carefully, attention to detail👣 Think step by step wh - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式和示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://tenor.com/view/mcmahon-crying-he-was-special-wwe-vince-mcmahon-gif-13313547165599993551">Mcmahon Crying He Was Special GIF - Mcmahon Crying He was special WWE - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1265385202373628015)** (89 条消息🔥🔥): 

> - `Llama 3.1 模型性能`
> - `模型审查与行为`
> - `Mistral Large 2 发布`
> - `模型测试与故障排除`
> - `模型命名趋势` 


- **Llama 3.1 在推理和编码方面表现挣扎**：用户注意到 **Llama 3.1 8B** 在推理和编码任务上表现不佳，一些成员对其整体性能表示怀疑。
   - 对比表明，虽然就其规模而言表现尚可，但其逻辑能力似乎有所欠缺，尤其是与 **Gemma 2** 等模型相比。
- **对顶级模型审查的担忧**：关于模型审查的讨论显示，表现良好的模型通常受到的审查较少，但成员们对此持有争议。
   - 一位成员建议，**censor**（审查）标签更多是为了管理列宽，而不是对模型行为的直接描述。
- **Mistral Large 2 带来显著改进**：拥有 128k 上下文窗口的 **Mistral Large 2** 发布，承诺在多种语言和编码任务中提高性能和效率。
   - 该模型的设计旨在实现单节点推理，拥有 1230 亿参数，为创新的 AI 应用提供了机会。
- **LLM 与 Flash Attention 的故障排除**：用户报告了加载 **Llama 3.1** 等模型时的问题，并建议检查与 **Flash Attention** 相关的配置，这可能会影响模型行为。
   - 许多人根据模型是全新加载还是使用不同配置加载，体验到了不同的性能表现。
- **AI 模型中的外星人命名趋势**：一个幽默的帖子探讨了 AI 使用 **Zorvath** 和 **Elara** 等名字作为角色的现象，好奇这些模式起源于何处。
   - 一位成员指出，文学影响可能会使命名惯例产生偏差，某些名字在特定类型和风格中出现频率更高。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1110598183144399058/1263234070813479063/1263234070813479063">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏和与朋友放松，甚至建立全球社区的好地方。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: 今天，我们宣布推出新一代旗舰模型 Mistral Large 2。与前代相比，Mistral Large 2 在代码生成、数学和推理方面能力显著提升...</li><li><a href="https://embed.wattpad.com/story/372087683-the-cosmic-union-dreamcatchers">Embed - The Cosmic Union: Dreamcatchers  - Wattpad</a>: 未找到描述</li><li><a href="https://x.com/YouJiacheng/status/1815817670954213710">Tweet from YouJiacheng (@YouJiacheng)</a>: 刚看到 deepseek-coder 将在 7 月 24 日 10:00 UTC+8 进行升级。</li><li><a href="https://github.com/THUDM/CodeGeeX4">GitHub - THUDM/CodeGeeX4: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;A and much more.</a>: CodeGeeX4-ALL-9B，一款适用于所有 AI 软件开发场景的全能模型，包括代码补全、代码解释器、网络搜索、函数调用、仓库级问答等等。 - ...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650">Feature Request: Proper Llama 3.1 Support in llama.cpp · Issue #8650 · ggerganov/llama.cpp</a>: 前提条件：我正在运行最新代码。如果可能，请提及版本。我仔细阅读了 README.md。我使用与问题相关的关键词进行了搜索，以确保我正在创建...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650#issuecomment-2246438902">Feature Request: Proper Llama 3.1 Support in llama.cpp · Issue #8650 · ggerganov/llama.cpp</a>: 前提条件：我正在运行最新代码。如果可能，请提及版本。我仔细阅读了 README.md。我使用与问题相关的关键词进行了搜索，以确保我正在创建...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1265478565152161812)** (9 条消息🔥): 

> - `Msty 功能`
> - `LM Studio Server 困惑`
> - `模型迁移担忧`
> - `LM Studio 中的 GPU 配置` 


- **Msty 提供了比 LM Studio 更具吸引力的功能**：在使用 Msty 从另一台设备连接到 LM Studio 时，一位用户强调了它无需完整升级应用即可更新 **Ollama 版本**的能力，这是他们希望 LM Studio 也能拥有的功能。
   - 他们对 **LM Studio** 不支持 endpoint 使用且仅限于本地推理表示恼火，这使得 Msty 成为更实用的选择。
- **关于 LM Studio 服务器功能的争论**：尽管 LM Studio 宣传其服务器功能，但用户认为它缺乏客户端能力，需要像 Msty 这样的额外软件才能在设备之间进行有效连接。
   - 这引起了挫败感，用户认为为了同一个功能安装两个应用显得冗余，并强调了 Msty 兼具服务器和客户端的双重角色。
- **对将模型迁移到 Ollama 的担忧**：一位用户提到他们不愿从 LM Studio 切换到 Msty 是因为 **迁移的痛苦**，特别是在将模型转移到 Ollama 后端方面。
   - 他们更喜欢 LM Studio 的模型管理方式，而不是 Ollama 那种他们认为繁琐的方法。
- **LM Studio 中的 GPU 配置**：讨论指出，LM Studio 中 GPU 负载分配的正确配置并不直观，需要深入设置。
   - 用户可以在 AI Chat > New chat 下找到高级 GPU 选项，允许他们切换设置以获得最大加速。


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1265392130583560234)** (11 条消息🔥): 

> - `Llama 3.1 预设`
> - `模型的 GPU 设置`
> - `Llama 3.1 的上下文长度` 


- **Llama 3.1 缺乏预设可见性**：一位用户表达了挫败感，称他们找不到任何 **Llama 3.1** 的 **presets**（预设），表明他们是这个环境的新手。
   - 另一位成员建议 **Llama 3 v2 预设**在更新到 **v0.2.28** 后可以配合使用。
- **3080ti 的最佳上下文长度**：有人提问是否应该为 **3070ti** GPU 保留 **2048** 的 **context length**（上下文长度）。
   - 建议将其设置为 **32k**，因为该模型支持高达 **128k context**。
- **GPU 加载问题**：一位用户报告了模型加载问题，称模型似乎没有完全加载到他们的 GPU 上。
   - 他们尝试将内存设置为最大 (-1)，但发现重新加载后它又恢复了原样。


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1265460884034031689)** (35 条消息🔥): 

> - `OpenCL 弃用`
> - `流媒体 GPU 对比`
> - `微调 LLaMA 3.1`
> - `适用于 LLM 的低预算 GPU`
> - `台湾电子产品购物` 


- **OpenCL 走向淘汰**：OpenCL 现已被弃用，但在 LM Studio 完全过渡到 Vulkan 之前，用户仍可使用它。
   - 目前，它的功能仍然可用，尽管从长远来看不推荐使用。
- **RTX 4090 vs RX 7900 XTX 之争**：一位成员询问在即将推出的 **Ryzen 9950x** 平台上，使用 **1x RTX 4090** 与 **2x RX 7900 XTX** 进行流媒体和本地 AI 应用的优劣。
   - 建议认为，虽然 RX 7900 XTX 在 VRAM 方面可能具有优势，但与 AI 模型的兼容性可能更倾向于支持更广泛的 Nvidia 选项。
- **微调 LLaMA 3.1 的 VRAM 需求**：社区讨论了微调 **LLaMA 3.1 8B** 模型的 VRAM 要求，估计使用 **32GB VRAM** 即可完成。
   - 上下文长度也引起了讨论，认为在上述 VRAM 容量下可以尝试 **27k tokens**。
- **适用于 LLM 的高性价比 GPU 选择**：成员们交流了关于本地 LLM 经济型 GPU 的看法，推荐 **RTX 3060 12GB** 作为比旧款 AMD 型号更可行的选择。
   - 有建议考虑中国改装的 **RTX 2080 Ti 22GB VRAM** 版本，认为这是一种有风险但可能更强大的替代方案。
- **探索台湾的技术机遇**：一位成员表示有兴趣参观台湾的科技卖场，灵感来自一些展示令人印象深刻的科技购物体验的知名视频。
   - 他们计划在逗留期间寻找廉价的 NVME 驱动器，但也注意到价格与在线选项相比并没有巨大差异。



**提到的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/15hiid1/how_long_does_finetuning_take_and_how_much_vram/">Reddit - 深入了解一切</a>：未找到描述

  

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1265383067401588736)** (87 条消息🔥🔥): 

> - `Beta Release Issues`
> - `Interface Changes and Feedback`
> - `GPU Offloading Problems`
> - `Model Loading Concerns`
> - `Version Confusion` 


- **Windows 上的 Beta 版本问题**：用户在 Windows 10 上遇到 Beta 版本无法正常启动的问题，部分用户虽然在任务管理器中能看到进程，但无法看到 UI 界面。
   - 已知问题包括 Beta 版本无法正确启动；建议的操作包括多次重启应用或等待后续更新。
- **对新界面的正面反馈**：许多用户分享了对 Beta 1 中新 UI 的正面体验，指出 Discovery 和 Collections 等功能特别有用。
   - 虽然对文件夹结构和设置提出了一些担忧，但总体而言，用户对所做的更改表示赞赏。
- **GPU Offloading 无法工作**：多位用户报告 GPU offloading 未能按预期运行，特别是在 M3 Max 和 4080S 上，通常需要依赖手动设置。
   - 自动 GPU 设置受到批评，因为它经常导致输出乱码，而手动设置似乎更可靠。
- **模型加载问题**：用户在尝试加载如 bic llama 等模型时面临挑战，某些设置需要调整以防止崩溃或 RAM 过载。
   - 建议利用 Developer 按钮将模型有效地加载到 RAM 中，绕过与 GPU 的集成。
- **版本混淆与更新管理**：关于 LM Studio 的版本存在混淆，由于频道中的链接过时，用户不确定最新版本。
   - 用户呼吁对版本更新进行更清晰的沟通，建议需要改进发布信息的组织。


  

---


### **LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1265709624120840325)** (1 条消息): 

> - `LangGraph tool binding`
> - `LLM limitations`
> - `LangChain integration issues` 


- **LangGraph 中的 LLM 工具绑定问题**：一位用户在 LangGraph 代码中尝试使用 `llm_with_tools = llm.bind_tools(tools)` 时遇到错误。
   - 他们质疑该错误是由于 **LM Studio** 不支持 tool calling，还是由于所使用的 **LLM** 导致的。
- **工具绑定问题的潜在原因**：讨论强调了故障的可能原因，重点在于 **LLM** 兼容性问题。
   - 目前尚不清楚当前使用的特定 **LLM** 是否支持所请求的 **tool binding** 功能。


  

---


### **LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1265714921103032352)** (2 条消息): 

> - `Krypt Lynx Installation`
> - `Pip Install Success` 


- **Successful Installation of Krypt Lynx**：一位成员表示有兴趣尝试 **Krypt Lynx** 项目，并询问了在 **Windows** 上的安装方法。
   - 他们随后更新称，使用 **pip install** 确实对他们有效，展示了良好的安装体验。
- **Windows 上的安装咨询**：一位成员发起讨论，询问如何在 **Windows** 平台上安装 **Krypt Lynx** 项目。
   - 这一咨询最终明确了在尝试 **pip install** 后安装成功，这让人感到宽慰。


  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1265431270146707576)** (33 条消息🔥): 

> - `ROCm 0.2.28 性能问题`
> - `Llama 3.1 兼容性`
> - `LM Studio 更新流程`
> - `OpenELM 支持`
> - `AppImage 功能` 


- **ROCm 0.2.28 表现出性能下降**：在更新到 ROCm **0.2.28** 后，一位用户报告其 2x 7900xt 系统的推理性能显著下降，单张显卡的功耗从之前的 **300w** 降至仅 **150w**。
   - 他们降级到 **0.2.27** 后发现性能恢复正常，因此请求调查 **0.2.28** 中所做的更改。
- **Llama 3.1 在 AMD 显卡上运行困难**：多位用户讨论了在 AMD 显卡上运行 **Llama 3.1** 的问题，提到在使用 **llama.cpp** 时出现 tokenizer 错误。
   - 一位用户发现问题源于使用了 **OpenCL** 而非 ROCm，而另一位用户则报告了在层可见性（layer visibility）方面的困扰。
- **LM Studio 的更新流程**：一位用户询问了 **0.2.28** 的更新命令，认为目前的说明可能仍指向旧版本。
   - 澄清指出，该版本的更新流程为了简化已进行回退，一位用户提到在 Discord 上很难找到版本详情。
- **对 OpenELM 支持的兴趣**：用户对 **OpenELM** 的潜在支持感到好奇，一位用户想尝试 Apple 的模型，并指出了一项最近相关的 GitHub pull request。
   - 回复指出，所有模型支持都取决于 **llama.cpp**。
- **AppImage 运行顺畅**：一位用户确认，在下载了适用于 Linux 的 **0.2.28** AppImage 后，他们可以在 **7800XT** 上开箱即用地运行 **Llama 3.1** 模型。
   - 这满足了使用 ROCm 运行 Llama 3.1 的要求，展示了兼容性。



**提到的链接**：<a href="https://github.com/ggerganov/llama.cpp/pull/7359">icecream95 提供的 OpenELM 支持 · Pull Request #7359 · ggerganov/llama.cpp</a>：修复了 #6868。感谢 @joshcarp 最初的尝试（#6986），这作为复制粘贴和检查的来源非常有帮助。目前许多配置是硬编码的...

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1265536160571064421)** (7 条消息): 

> - `Meta-Llama 3.1 70B`
> - `Tokenizer Bug 修复` 


- **Meta-Llama 3.1 70B 现已发布**：一位成员宣布 **Meta-Llama 3.1** 的 **70B 量化版本**已发布，并附上了 [repository](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF) 链接。
   - 频道里的兴奋之情溢于言表，其他人评论道：“这就是为什么他是 GOAT（史上最强）”。
- **Tokenizer Bug 请求重新上传**：提到模型将重新上传以修复一个 **tokenizer bug**，预计这将提高性能。
   - 目前，据报告性能“尚可”，预计更新后会有更好的结果。


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1265684901928239115)** (11 条消息🔥): 

> - `LM Studio 兼容性`
> - `AVX2 和 AVX-512 指令集`
> - `Koboldcpp vs LM Studio`
> - `模型下载替代方案` 


- **LM Studio 的需求和兼容性问题**：一位用户报告在 **Windows Server 2012 R2** 设置上运行 **LM Studio** 时出现问题，原因是与 **kernel32.dll** 的未知兼容性。另一位用户确认，如果没有 **AVX2** 指令集，**LM Studio** 将无法安装，而当前的 CPU 缺少该指令集。
   - **Koboldcpp** 在 **AVX-512** 上运行良好，但用户因界面原因更倾向于使用 **LM Studio**。
- **理解 AVX2 和 AVX-512 的区别**：关于 **AVX2** 和 **AVX-512** 指令集的使用存在混淆，一位成员认为更大的指令集可能更好。澄清指出 **Xeon 6138** 不支持 **AVX2**，这使得 **LM Studio** 无论是否存在其他错误都无法兼容。
   - 一位用户对澄清表示感谢，并提到理解指令类型对未来的使用会有帮助。
- **探索 llama.cpp 的替代方案**：另一位用户提到，虽然 **LM Studio** 存在兼容性问题，但可以通过支持 **AVX-512** 的方式构建 **llama.cpp**。这为通过控制台或服务器端点进行模型下载和推理提供了替代方案。
   - 用户被引导查看 **lms-comm** 或 **bartowski** 的 Hugging Face 页面，寻找可用的模型作为潜在替代品。



**提到的链接**：<a href="https://github.com/ggerganov/llama.cpp/issues/160">添加 AVX-512 支持？ · Issue #160 · ggerganov/llama.cpp</a>：不确定，但我认为它可能会运行得更快

  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1265397562408435742)** (1 条消息): 

> - `Llama 3.1 405B`
> - `Open source models` 


- **Llama 3.1 405B 在 Perplexity 上线**：被誉为最强大的 **Open source model**，**Llama 3.1 405B** 模型现已在 Perplexity 上可用，并可与 **GPT-4o** 和 **Claude Sonnet 3.5** 媲美。
   - 团队正在积极将 Llama 3.1 405B 集成到其 **mobile apps** 中，请用户关注后续更新。
- **即将推出的移动端应用集成**：Perplexity 团队宣布了下一步将 **Llama 3.1 405B** 功能添加到其 **mobile applications** 的计划。
   - 建议用户关注有关此集成的更多更新。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1265390147072032882)** (306 条消息🔥🔥): 

> - `Llama 405b 性能`
> - `Mistral Large 2`
> - `AI 模型对比`
> - `TikTok 作为搜索引擎`
> - `语言符号输出问题` 


- **关于 Llama 405b 性能的讨论**：成员们对 Llama 405b 的表现持怀疑态度，认为与 Mistral 和 Sonnet 等其他模型相比表现平平。
   - 一些人注意到不同模型之间的 Benchmark 结果存在不一致，这影响了他们的决策。
- **对 Mistral Large 2 的赞赏**：Mistral Large 2 受到关注，被认为在结果上可能优于 Llama 405b，用户表示相比 Llama 更倾向于选择它。
   - 用户希望 Perplexity 在现有模型基础上增加 Mistral。
- **对 AI 模型 Benchmark 的困惑**：Benchmark 的有效性受到质疑，成员们将其比作电影评分，强调了其不一致性。
   - 用户指出，对模型的个人主观体验差异很大，因此很难仅依赖 Benchmark。
- **TikTok 作为搜索工具的潜力**：成员们讨论了 TikTok 作为 Z 世代新兴搜索引擎的角色，并就其与传统搜索方法相比的价值展开了辩论。
   - 有人对 TikTok 上健康建议的可靠性以及使用此类平台获取信息的后果表示担忧。
- **语言符号输出问题**：用户报告 Llama 模型在正确输出亚洲语言符号方面存在困难，认为这是性能上的一个局限。
   - 有人建议，模型不愿使用符号可能源于其训练方式以及对多语言输入的处理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/minchoi/status/1815812112796565690">Min Choi (@minchoi) 的推文</a>: 使用 Llama 3.1 8B + Groq 的即时智能太疯狂了 🤯 </li><li><a href="https://www.perplexity.ai/search/write-short-poem-in-czech-abou-ksVKF84qQNG2wH8Q7ia2NQ">用捷克语写一首关于 smažák 的短诗</a>: Smažák, zlatý a křupavý, Vonící olejem, chutí nebeskou. Křehký a jemný, jako sníh, Smažák, můj žaludek, potěšíš. 翻译：炸奶酪，金黄且...</li><li><a href="https://www.perplexity.ai/search/write-short-magical-poem-in-cz-v7lRjXgxS5.tDX22.g4F2Q">用捷克语写一首关于美丽飞行热狗的神奇短诗</a>: 这里有一首用捷克语写的关于美丽飞行热狗的短诗：&quot;Létající hot dog, krásný a zlatý, S klouboučkem hořčice, tančí v oblakách. Svírá se v...</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1eb9njj/claude_35_vs_llama_405b_vs_others_tested_by_ai/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/history/3c76952a-c352-4520-95a2-ccf1a7b2b056?share=true">_Think-Lab 修订版</a>: 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精简的搜索查询。ScratchPad 是一个强大的工具，可以帮助你保持连贯性和准确性，尤其是...</li><li><a href="https://github.com/nuprl/MultiPL-E">GitHub - nuprl/MultiPL-E: 一个针对 LLM 的多编程语言 Benchmark</a>: 一个针对 LLM 的多编程语言 Benchmark。通过创建账号为 nuprl/MultiPL-E 的开发做出贡献。</li><li><a href="https://app.wordware.ai/share/8c523d8b-c109-4189-a6ce-cc9bfc5d24a2/history/129be5a3-d4ae-4069-85f5-156052669490?share=true">Sonnet Insight 3.5 - 模型输出排名</a>: 此提示词使用 Sonnet 3.5, Gemini 1.5 Pro, Llama 3.1 70B &amp; 405B, GPT-4o/mini, Sonar Large (联网模型), Claude 3 Opus, Claude 3 Sonnet 以及最后使用 Claude 3 Haiku 处理问题。该应用...</li><li><a href="https://scale.com/leaderboard">SEAL 排行榜</a>: 未找到描述</li><li><a href="https://x.com/rypearts/status/1815868829169328349?s=61">Ryan Putnam (@RypeArts) 的推文</a>: ✧ 　 　 ✧ ˚ * 　 　.　 　　　　 　　 · · 　　 　 + ✧ 　　　 · 　 · ˚ . 𝓈𝓊𝓂𝓂ℯ𝓇 𝓋𝒾𝒷ℯ𝓈</li><li><a href="https://www.euronews.com/next/2023/02/05/gen-z-is-using-tiktok-as-a-search-engine-is-this-the-end-of-google">TikTok 是否即将取代 Google 成为顶级搜索引擎？</a>: TikTok 搜索量的增加引发了 Google 是否很快会过时的疑问。</li><li><a href="https://www.perplexity.ai/search/10-animals-in-kanji-only-respo-CdygXAlIQte9iJXH.fXGwg">仅用汉字列出 10 种动物。仅以日语符号回复</a>: * 河馬 * 山羊 * 栗鼠 * 獅子 * 大猩々 * 麒麟 * 長尾驢 * 子守熊 * 駱駝 * 土竜</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1265395723508383927)** (13 条消息🔥): 

> - `Mistral Large 2`
> - `拜登总统的公开露面`
> - `永旺 (AEON) 的 AI 监控`
> - `世界上最古老的树木`
> - `Meta 的 Llama 3.1`

- **Mistral Large 2 树立了新的 AI 标准**：2024年7月24日，Mistral AI 发布了 Mistral Large 2，拥有 **1230 亿参数**和 **128,000-token 上下文窗口**，增强了在代码生成和数学方面的能力。
   - 据报道，该模型在数学任务中优于 **Llama 3.1 405B**，并与 **GPT-4** 旗鼓相当。
- **拜登总统最后一次公开露面**：乔·拜登总统最后一次公开露面是在 **2024年7月17日**，此前他在拉斯维加斯竞选期间 COVID-19 检测呈阳性。
   - 这标志着他在 **7月23日** 退出总统竞选前的最后一次亮相。
- **AI 系统监控永旺（AEON）门店的微笑**：日本超市连锁店永旺实施了一套名为 **'Mr Smile'** 的 AI 系统，根据 450 多个行为要素来规范员工的微笑。
   - 据报道，在八家门店进行的**试点**在三个月内将服务态度提升了 **1.6 倍**。
- **世界最古老树木汇编**：研究重点介绍了大盆地刺果松（Great Basin bristlecone pine）等树木，其树龄接近 **5,000 年**，被已知是最古老的非克隆树种。
   - 包括树轮计数和放射性碳定年法在内的多种方法有助于确定这些古树的年龄。
- **Meta 发布 Llama 3.1**：Meta 最近发布的 **Llama 3.1 405B** 提供了一个极具竞争力的开源模型，重点挑战现有的专有 AI，如 **GPT-4**。
   - 它拥有 **4050 亿参数**，承诺为开发者提供前所未有的高级 AI 能力访问权限。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.perplexity.ai/search/mistral-large-2-revolutionizin-sVfT0LnmTJ2ER3WS5YqILQ#1">Mistral Large 2: Revolutionizing Language Models with Unprecedented Capabilities</a>: 这是我对使用 scratchpad 格式输出的评论：&lt;scratchpad&gt; [从提示词中提取的关键信息] 关于 Mistral 文章的评审任务...</li><li><a href="https://www.perplexity.ai/page/japans-stores-use-ai-to-track-tIYXMbASQbCClx5sHwrn5g">Japan&#x27;s Stores Use AI to Track Smiles</a>: 日本超市连锁店 AEON 引入了一套名为 "Mr Smile" 的人工智能系统，用于评估和标准化员工的微笑，引发了辩论...</li><li><a href="https://www.perplexity.ai/page/why-didn-t-jack-join-rose-USOLbt8USUqW1eB.kg3pYg">Why Didn&#x27;t Jack Join Rose?</a>: 詹姆斯·卡梅隆的《泰坦尼克号》中杰克牺牲自己拯救露丝的经典场景，几十年来一直在粉丝和专家之间引发争论。虽然...</li><li><a href="https://www.perplexity.ai/search/combien-de-litre-d-eau-il-y-a-ght8Kb8OTS2HcZ0OxNY9jw#0">Combien de litre d&#x27;eau il y a sur terre ?</a>: 地球包含约 13.86 亿立方公里的水 (1,386 \times 10^9 \, km^3)，相当于 1.386 万亿升水 (1,386 \times...</li><li><a href="https://www.perplexity.ai/search/the-oldest-tree-in-the-world-o_vxC.JmTnSeKWNhsi_t0Q">The oldest tree in the world</a>: 世界上已知最古老的单体树是一棵名为“玛土撒拉”（Methuselah）的大盆地刺果松，位于加利福尼亚州的白山。这棵树...</li><li><a href="https://www.perplexity.ai/search/cual-sera-el-proximonpresidente-mqJrnaaAShyafjRFgXH7wQ">cual sera el proximonpresidente de ee.uu?</a>: 2024 年 11 月选举的美国总统竞选随着乔·拜登总统最近的决定发生了重大转折...</li><li><a href="https://www.perplexity.ai/search/when-was-president-biden-last-_.41EsmjTk2glkhKNbSp9A">when was president Biden last seen in public</a>: 根据提供的搜索结果，乔·拜登总统最后一次公开露面是在 2024 年 7 月 17 日星期三。具体而言：1. 拜登最后一次被看到是在离开...</li><li><a href="https://www.perplexity.ai/search/when-you-have-pots-does-your-h-jmae9BaDQQqK0IA0VGvoQw">When you have POTS does your heart feel like it’s beating harder at times</a>: 是的，当你患有体位性心动过速综合征 (POTS) 时，你的心脏有时会感觉跳动得更厉害。这是因为 POTS 会导致你的...</li><li><a href="https://www.perplexity.ai/page/legal-trials-of-inanimate-obje-AGyEpycyQ6qVdMxEBIqUsg">Legal Trials of Inanimate Objects</a>: 纵观历史，法律系统一直在应对对造成人类伤害或死亡的无生命物体进行审判的异常做法。从...</li><li><a href="https://www.perplexity.ai/page/meta-releases-llama-3-1-405b-pFAuGE4GR_id4.zHbNDqyQ">Meta releases Llama 3.1 405B</a>: Meta 发布 Llama 3.1 405B 标志着 AI 领域的重大里程碑，推出了一款足以与闭源巨头抗衡的强大开源模型...</li><li><a href="https://youtu.be/1--eJwi-xQo?si=7rl6NsxQPluT_WR7">Meta&#39;s Lama 3.1, Wiz&#39;s Bold Rejection, XAI Memphis Supercluster, Cocaine Sharks, and Space Debris...</a>: 有没有想过单个 AI 模型如何重塑技术格局？探索 Meta 的 Lama 3.1 背后的秘密，这是一款拥有 4050 亿参数的 AI 奇迹...</li><li><a href="https://www.perplexity.ai/page/mistral-large-2-revolutionizin-kUXugCSjRAevYdq7_cnYkA">Mistral Large 2: Revolutionizing AI</a>: 2024 年 7 月 24 日，Mistral AI 发布了 Mistral Large 2，这是一款强大的新语言模型，拥有 1230 亿参数和 128,000 token 的上下文窗口...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1265401892700426363)** (8 条消息🔥): 

> - `Llama 3 405b API 计划`
> - `Llama 3 405b 的上下文大小`
> - `在 Langchain 中传递 return_citations`
> - `NextCloud 与 OpenAI 的集成`
> - `Microsoft Copilot Studio Perplexity 连接器` 


- **Llama 3 405b 的 API 计划**：一位成员确认 **Llama 3 405b** 很快将通过 API 提供，预示着即将发布的新功能。
   - 这一回应引发了热烈讨论，因为它承诺为用户带来新的能力。
- **Llama 3 405b 上下文大小讨论**：一位成员询问了 **Llama 3 405b** 的上下文大小，暗示其可能为 **128K**，因为无需微调。
   - 他们断言，与 Claude 和 GPT 等现有模型相比，这一特性可以降低成本。
- **在 Langchain LLM 链中返回引用 (return_citations)**：一位用户寻求关于在使用 Perplexity Chat 的 **Langchain** LLM 链中如何传递 **return_citations** 值的指导。
   - 讨论中未分享具体的解决方案，表明该问题需要进一步探索。
- **NextCloud 的 OpenAI 集成查询**：一位成员分享了 **NextCloud** 与 OpenAI 集成的链接，赞扬其社区驱动、免费且开源的特性。
   - 他们为对集成细节感兴趣的人提供了 **GitHub repository** 引用。
- **Microsoft Copilot Studio 连接器问题**：一位用户提出了在将 Perplexity 连接器上传到 **Microsoft Teams** 时出现**未指定错误**的问题。
   - 社区的回应表明，解决此问题可能需要进行故障排除。



**提到的链接**：<a href="https://github.com/nextcloud">Nextcloud</a>: 📱☁️💻 为您的所有数据提供安全的家 —— 社区驱动、免费且开源 👏 - Nextcloud

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1265393328862527591)** (298 条消息🔥🔥): 

> - `模型能力`
> - `用于 AI 模型的 GPU 服务器`
> - `Kling AI 图生视频`
> - `LLM 与 Raspberry Pi 的兼容性`
> - `自定义模型的提示词库` 


- **Mistral 模型及其规格**：用户讨论了 **Mistral-7B-v0.3** 模型的规格，注意到其改进的能力，包括 128k 上下文窗口和多语言支持。
   - 提到 **Mistral Large** 模型大小为 69GB，可以使用 **ollama** 高效运行。
- **探索 GPU 服务器选项**：用户强调了运行大型模型的 GPU 服务器选项，建议将 **Runpod** 作为价格实惠的实例，价格为 **$0.30/小时**。
   - 有人建议使用 **LM Studio** 或 **ollama** 以获得更好的性能和特定模型的兼容性。
- **Kling AI 的图生视频功能**：**Kling AI** 因其从静态照片生成动态图像的卓越能力而受到关注，尽管用户报告了视频质量的一些局限性。
   - 尽管结果有趣且引人入胜，但也有评论称服务器过载导致生成时间较长。
- **LLM 在 Raspberry Pi 上的兼容性和性能**：讨论转向了在 **Raspberry Pi 4B** 上运行 LLM 的可行性，用户对其性能表现尚不确定。
   - 提到可以运行具有不同 RAM 配置的模型，可能包括使用 **ollama** 运行 7B 模型。
- **提示词库访问和自定义模型**：用户询问如何访问提示词库以创建 AI 模型的自定义提示词，并指向了可提供帮助的频道。
   - 对话强调了成功利用某些大模型能力需要特定的模型文件和框架。



**提到的链接**：<a href="https://huggingface.co/mistralai/Mistral-7B-v0.3">mistralai/Mistral-7B-v0.3 · Hugging Face</a>: 未找到描述

  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1265645303868948530)** (9 messages🔥): 

> - `Memory Feature Issues in EU`
> - `Spelling Errors in Mini`
> - `Python PDF Generation with OpenAI`
> - `Debugging Model Output`
> - `User Feedback on Model Mistakes` 


- **Memory feature 在 EU 地区出现又消失**：一名成员报告称，他们仅获得了五分钟的 **Memory feature**，随后其他成员也确认了类似的经历。
   - 另一位用户幽默地评论了这种不一致性，并建议讨论该功能是否已完全投入运行。
- **Mini 频繁将 'composure' 拼错**：一位成员指出，**Mini** 在其提示词中始终将 'composure' 拼错为 'composposure'。
   - 另一位成员无法复现此问题，并分享了他们的提示词链接，显示 'composure' 拼写正确。
- **在 Python 中使用 OpenAI 生成 PDF 内容**：一位用户询问如何使用 Python 和 OpenAI 生成 PDF，表示需要根据上传的文件生成目录和章节描述。
   - 这引发了关于利用 OpenAI 进行文档生成的流程和技术的讨论。
- **针对拼写错误调试模型输出**：一位成员注意到频繁的拼写错误，包括将 'it is' 误写为 'itis'，并计划启用调试以检查模型的提示词。
   - 这促使另一位成员建议分享具体示例，以便更好地理解模型的输出倾向。
- **用户在对话中遇到单词拼写错误**：一位用户分享说，只有在明确要求并提供反馈时，他们才会诱发拼写错误和间距问题。
   - 他们提供了一个分享链接，展示了突出这一现象的交互过程。


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1265447785692663911)** (4 messages): 

> - `LLM Distillation`
> - `LLaMa 3`
> - `Common RAG Challenges` 


- **关于 LLM Distillation 的最新论文**：一位成员建议查看 [Minitron GitHub 仓库](https://github.com/NVlabs/Minitron)，该仓库详细介绍了一系列通过 **pruning**（剪枝）和 **knowledge distillation**（知识蒸馏）获得的压缩模型。
   - 该仓库可能为类似于 **Sonnet**、**Llama** 和 **GPT-4Omini** 等模型的 LLM Distillation 最新进展提供见解。
- **LLaMa 3 模型介绍**：推出了一套名为 **LLaMa 3** 的新基础模型，其特点是拥有 **405B parameters** 的密集 Transformer，以及高达 **128K tokens** 的上下文窗口。
   - 这些模型在多语言能力、代码编写、推理和工具使用方面表现出色，将增强广泛的 AI 应用。
- **生产环境中常见的 RAG 挑战**：一位成员分享了一个 LinkedIn 帖子链接，讨论了常见的 RAG (Retrieval-Augmented Generation) 挑战及潜在解决方案。
   - 该帖子强调了在生产环境中实施 RAG 时面临的各种问题，从业者可能会发现这些内容非常有用。



**提及的链接**：<a href="https://github.com/NVlabs/Minitron">GitHub - NVlabs/Minitron: A family of compressed models obtained via pruning and knowledge distillation</a>：通过剪枝和知识蒸馏获得的一系列压缩模型 - NVlabs/Minitron

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1265386240350621707)** (2 messages): 

> - `PC Agent Demo`
> - `Proprietary Tools` 


- **令人兴奋的 PC Agent Demo 发布**：一位成员分享了一个题为 ["PC Agent Demo" 的 YouTube 视频](https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8)，展示了来自 gate-app.com/research/pc-agent 的新 Agent。
   - 该 Demo 展示了 PC Agent 工具的功能和潜在应用。
- **关于专有工具 (Proprietary Tools) 的讨论**：一位成员暗示，这可能与频道早先讨论的主题相关的专有工具有关。
   - 这一讨论引发了其他成员的参与，大家纷纷思考此类工具的影响和应用。



**提及的链接**：<a href="https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8">PC Agent Demo</a>：gate-app.com/research/pc-agent

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1265387248145272932)** (20 条消息🔥): 

> - `Meta Llama 3.1 能力`
> - `合成数据集创建`
> - `Microsoft GraphRAG`
> - `Aider 的仓库地图 (repo map)`
> - `Wordware 应用` 


- **Meta Llama 3.1 在多语言任务中表现卓越**：**Meta Llama 3.1** 系列包括 **8B、70B 和 405B** 尺寸的预训练模型，针对多语言对话进行了优化，性能超越了许多现有的聊天模型。
   - 该系列提供微调选项，支持**合成数据集创建**，并提供**社区许可 (Community License)** 供商业和研究场景使用。
- **关于合成数据集潜力的讨论**：有人提出疑问，既然现在可以访问 **GPT-4**，NousResearch 是否会开始**大规模生产合成数据集**。
   - 尽管热情高涨，但也有人担心使用 **405B 模型** 进行生成的成本比使用 **Sonnet 3.5 或 GPT-4o** 更高。
- **Microsoft 推出 GraphRAG**：Microsoft 发布了 **GraphRAG**，通过从现有数据集中创建知识图谱，增强了 LLM 解决未见数据问题的能力。
   - 这种方法有望改进语义聚类和概念识别，使 **RAG 技术** 成为数据调查的重要工具。
- **Aider 仓库地图 (repo map) 的局限性**：虽然 **Aider** 映射代码仓库的能力令人印象深刻，但其架构在代码库的语义理解方面存在局限性。
   - 目前的方法侧重于**实体频率权重**，而非真正的演进理解，这引发了关于高级检索替代方案的讨论。
- **Wordware 应用展示 JWST 图像**：Wordware 应用将展示已发布的**詹姆斯·韦伯空间望远镜 (JWST)** 图像，以增强平台的视觉吸引力。
   - Wordware 内的另一个应用可以同时测试多个模型，展示了增强的**搜索引擎**能力以及输出速度追踪功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider 使用你的 git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/playground">_Think-Lab Revised</a>: 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精炼的搜索查询。Scratchpad 是一个强大的工具，可以帮助你保持连贯性和准确性，尤其是...</li><li><a href="https://app.wordware.ai/share/8c523d8b-c109-4189-a6ce-cc9bfc5d24a2/playground">Sonnet Insight 3.5 - Rank Model Outputs </a>: 此 Prompt 使用 Sonnet 3.5, Gemini 1.5 Pro, Llama 3.1 70B&amp;405B, GPT-4o/mini, Sonar Large (在线模型), Claude 3 Opus, Claude 3 Sonnet 以及最后的 Claude 3 Haiku 处理问题。该应用会...</li><li><a href="https://youtu.be/1B50IDUl5D4?si=pPfOvaHGax7t68Y0">Create fine-tuned models with NO-CODE for Ollama &amp; LMStudio!</a>: 👋 大家好，新视频重点介绍了我们刚刚在 AnythingLLM 中添加的一个超酷功能，你可以从中创建一个完整的微调模型...</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/">GraphRAG: A new approach for discovery using complex information</a>: Microsoft 正在通过 GraphRAG 变革检索增强生成，利用 LLM 生成的知识图谱在分析复杂信息时显著改善问答效果，并始终保持优异表现...</li><li><a href="https://news.ycombinator.com/item?id=41013693">无标题</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: 旨在与 Llama 模型配合使用的实用程序。通过在 GitHub 上创建一个账户来为 meta-llama/llama-models 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1265746471966216353)** (1 messages): 

> - `Nous Research subreddit`
> - `AMA announcement`（AMA 公告）


- **Nous Research subreddit launched**: **Nous Research** 的新 subreddit 已正式上线，成员可以加入并讨论 **AI** 领域的最新研究与进展。
   - 鼓励用户在[此处](https://reddit.com/r/NousResearch)发起话题并参与互动。
- **Upcoming AMA with Nous leaders**: 计划在未来几周内与 **Reddit** 上的特定成员进行 **AMA** 环节，以回答社区提问。
   - 更多信息将在临近时发布，请保持关注！



**Link mentioned**: <a href="https://reddit.com/r/NousResearch">Reddit - Dive into anything</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1265382943627677828)** (224 messages🔥🔥): 

> - `Llama 3.1 Performance`（Llama 3.1 性能）
> - `Mistral Large 2 Release`（Mistral Large 2 发布）
> - `Open-Source TTS Models`（开源 TTS 模型）
> - `Autonomous Coding Tools`（自主编程工具）
> - `Synthetic Data in AI`（AI 中的合成数据）


- **Llama 3.1 faces competition from Mistral**: Llama 3.1 模型正面临来自 Mistral Large 2 的竞争，后者拥有相似的架构，但在性能上表现更佳，尤其是在编程任务中。
   - 用户对 Mistral 提升输出质量的潜力以及合成数据日益增强的能力感到兴奋。
- **Mistral Large 2 impresses with capabilities**: Mistral Large 2 已发布，拥有 123B 参数，配备 128k 上下文窗口，并在编程任务中表现强劲。
   - 尽管其非商业许可证限制了托管选项，但凭借其创新设计，预计在 API 平台上会有出色表现。
- **Exploration of open-weight TTS models**: 用户正在讨论各种 Text-to-Speech 模型的体验，重点关注质量、速度和离线能力。
   - 比较了 ElevenLabs 和 Apple 的 Siri 语音等模型，并推荐了 parler-expresso 和 VITS 等新方案。
- **Inquiry about autonomous coding tools**: 社区对 Devika 和 Open Devin 等开源自主编程工具的现状表现出浓厚兴趣。
   - 用户正在寻求建议和对比，以确定哪些工具最适合其开发需求。
- **Potential of synthetic data**: 用户对在训练 AI 模型中使用高质量合成数据充满热情，认为这能增强模型性能和通用性。
   - 有推测认为，未来合成数据生成的进步可能会带来模型能力的重大飞跃。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">大有可为</a>：今天，我们宣布推出新一代旗舰模型 Mistral Large 2。与前代产品相比，Mistral Large 2 在代码生成、数学和推理方面的能力显著提升...</li><li><a href="https://x.com/capeto">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/qresearch/llama-3.1-8B-vision-378">qresearch/llama-3.1-8B-vision-378 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/SillyTilly/Meta-Llama-3.1-70B">SillyTilly/Meta-Llama-3.1-70B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct">meta-llama/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/kermit-suicide-flip-jump-crash-gif-5140737">Kermit Suicide GIF - Kermit Suicide Flip - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/normand1/HyperFeeder/blob/master/audioScripts/ttsLocalScript.sh">HyperFeeder/audioScripts/ttsLocalScript.sh at master · normand1/HyperFeeder</a>：自主播客生成器。通过在 GitHub 上创建账户来为 normand1/HyperFeeder 的开发做出贡献。</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">未找到标题</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format">半精度浮点格式 - 维基百科</a>：未找到描述</li><li><a href="https://tenor.com/view/omegalul-lul-lulw-twitch-emote-gif-13523263">Omegalul Lul GIF - Omegalul LUL LULW - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700">来自 Bill Yuchen Lin 🤖 (@billyuchenlin) 的推文</a>：对 Llama-3.1-405B-Instruct-Turbo（在 @togethercompute 上运行）的快速独立评估 ⬇️ 1️⃣ 它在 GSM8K 上排名第一！2️⃣ 它在 ZebraLogic 上的逻辑推理能力与 Sonnet 3.5 非常相似，而且...</li><li><a href="https://github.com/meta-llama/llama-agentic-system">GitHub - meta-llama/llama-agentic-system: Llama Stack API 的 Agent 组件</a>：Llama Stack API 的 Agent 组件。通过在 GitHub 上创建账户来为 meta-llama/llama-agentic-system 的开发做出贡献。</li><li><a href="https://avian.io">Avian.io</a>：Avian 是一个面向企业的生成式 AI 平台，支持跨 Llama-3.1-405B 的最先进 LLM 推理，并支持拥有超过 100 个数据连接器的 RAG。</li><li><a href="https://en.wikipedia.org/wiki/Activation_function">激活函数 - 维基百科</a>：未找到描述</li><li><a href="https://x.com/capetorch/status/1816110002823745945">来自 Thomas Capelle (@capetorch) 的推文</a>：想免费试用 Llama3.1 405B 模型吗？让我们一起对模型进行红队测试，并协作生成一个数据集来评估 Llama 3.1 系列模型。我们准备了一个简单的 Colab...</li><li><a href="https://huggingface.co/collections/hugging-quants/llama-31-gptq-awq-and-bnb-quants-669fa7f50f6e713fd54bd198">Llama 3.1 GPTQ, AWQ, 和 BNB 量化 - hugging-quants 集合</a>：未找到描述</li><li><a href="https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data">wiki-phrases-tokenizer/data at master · vtempest/wiki-phrases-tokenizer</a>：维基百科大纲关系词典数据集 (WORLD) * 特定领域实体和关键词提取 (DSEEK) * 维基百科重要命名主题实体识别 (WINTER) - vtempest/wiki-phr...
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1265387151412166807)** (24 条消息🔥): 

> - `Fine-tuning Llama 3`
> - `Multi-language fine-tuning`
> - `Custom tool calls`
> - `Hermes function calling`
> - `Generative capabilities of LLMs` 


- **Llama 3 微调面临挑战**：成员们表示，**微调 Llama 3 405B** 将是一个巨大的挑战，并建议可能只有 **LoRA 微调** 能够胜任。
   - 一位成员指出，这种情况可能会推动 **OSS** 领域内 **DoRA 微调** 的进步。
- **普什图语 (Pashto) 微调所需资源**：一位成员正在寻找专门针对 **普什图语** 进行模型 **微调** 的资源，并强调尽管该语言拥有 **6000 万** 使用者，但可用材料却很匮乏。
   - 另一位成员建议查阅最近的 **Aya23 model+ 论文** 以获取相关信息。
- **自定义工具调用需要关注**：讨论围绕在执行 **自定义工具调用**（尤其是像垃圾邮件检查这样的简单任务）时，需要通过微调来获得正确的格式。
   - 一位参与者强调应使用 **Hermes function calling** 的 **GitHub 仓库** 中提供的 **正确系统提示词 (system prompt) 和架构 (schema)**。
- **LLM 在生成复杂代码时表现吃力**：一位成员报告了他们尝试使用 **Llama 405B** 生成 Python 版 **贪吃蛇游戏** 的经历，起初很成功，但未能有效地加入 **DQN** 方法。
   - 他们注意到，尽管提供了错误信息，模型仍反复失败，这表明需要更好的提示策略。
- **关于 Hermes 发布和进展的查询**：几位用户询问了针对 **Llama 3.1** 的 **Hermes 发布** 何时可用，反映出对更新的广泛关注。
   - 成员们讨论了目前正在进行的努力以及用于高级项目的资源，包括 **多节点训练 (multi-node training)** 设置。


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1265473433697910854)** (4 条消息): 

> - `Citizen Sleeper 核心机制`
> - `wiki-phrases-tokenizer`
> - `grounded refusals` 


- **Citizen Sleeper 核心机制解析**：**Citizen Sleeper** 的核心机制围绕掷骰子来分配行动，这会显著影响玩家在游戏中的进度。
   - *每天，玩家掷出的骰子结果受状态系统管辖*，这强化了 **不稳定性 (precarity)** 和 **风险** 的主题。
- **wiki-phrases-tokenizer 数据集介绍**：一位成员分享了 [wiki-phrases-tokenizer GitHub 仓库](https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data) 的链接，强调其作为 RAG 样本数据的潜力，包含 *前 10 万个维基百科页面* 和 *Quora 搜索查询* 等数据集。
   - 该数据集被描述为包含用于实体和关键词的 **特定领域提取 (domain-specific extraction)** 的宝贵信息。
- **对 Meta 团队智能水平的认可**：一位成员对未曾考虑到 **grounded refusals** 表示惊讶，并承认 **Meta 团队** 在这方面更聪明。
   - 这一评论反映了一种谦逊的态度以及对该团队能力的认可。



**提到的链接**：<a href="https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data">wiki-phrases-tokenizer/data at master · vtempest/wiki-phrases-tokenizer</a>: Wikipedia Outline Relational Lexicon Dataset (WORLD) *  Domain-Specific Extraction of Entities and Keywords (DSEEK) * Wikipedia Important Named Topic Entity Recognition (WINTER) - vtempest/wiki-phr...

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1265489424737894473)** (3 条消息): 

> - `亚符号概念空间 (Sub-Symbolic Concept Space)`
> - `GPU 集群上的 Llama 模型`
> - `基于订阅的 AI 访问` 


- **探索亚符号概念空间**：一位成员表达了终于有时间参与 **WorldSim** 并思考 **亚符号概念空间 (sub-symbolic concept space)** 的兴奋之情。
   - 这表明在未来的讨论中，人们有兴趣继续扩展理论 AI 概念。
- **Llama 模型可实现分级访问**：一位成员理论化，在托管 **GPU** 集群上使用 **Llama 模型** 可以创建一个通过订阅或分级访问的 **封闭游乐场 (gated playground)**。
   - 他们建议，如果可行，这将是未来聚会中一个值得讨论的话题。
- **关于代码可用性的提问**：一位成员询问所讨论的应用或模型是否有任何 **可用代码**。
   - 这突显了社区内探索资源可能存在的缺口。


  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1265420101230788739)** (13 messages🔥): 

> - `SMT Solvers 与 LLM 翻译`
> - `更新仓库结构`
> - `困难的道德查询`
> - `电车难题道德辩论` 


- **为 LLM 利用 SMT Solvers**：@SMT_Solvers 建议，教 LLM 将英语/德语的应用题翻译成 SMTLIB 可以产生显著的推理能力，这本质上是一个使用 egraphs 进行探索的 **MADLIBS 合成数据问题**。
   - 这通过有效的翻译方法激发了高级推理任务的潜力，增强了模型的整体性能。
- **仓库结构更新进行中**：@teknium 宣布计划今天更新仓库的结构和 Schema，并邀请社区其他成员协作。
   - @n8programs 对更新表示期待并愿意提供帮助，彰显了社区的参与度。
- **推理任务中的道德困境**：围绕是否应包含**困难/道德**查询（如**电车难题**）作为挑战模型基础道德原则的推理任务展开了讨论。
   - 这引发了关于道德推理与场景的直接逻辑评估之间影响的疑问，值得深入分析。
- **对电车难题的反思**：有人担心电车难题评估的是模型采用了哪些道德基础，而非纯粹的推理能力，@stefangliga 对其目的提出了质疑。
   - @_paradroid 建议，结构化的 Prompt 可以澄清推理过程和思维评估，增强对道德框架的理解。
- **道德查询中的结构化推理**：@_paradroid 分享了一个结构化框架，用于分析自动驾驶汽车决策的道德影响，旨在提高推理的清晰度和准确性。
   - 该框架包括识别初步想法、提供背景以及反思推理过程，展示了处理道德推理任务的综合方法。



**提到的链接**：<a href="https://x.com/SMT_Solvers/status/1815856006427205672">来自 Chad Brewbaker (@SMT_Solvers) 的推文</a>：@halvarflake 正如我告诉 @Teknium1 的那样，如果我们能教 LLM 将英语/德语的应用题翻译成 SMTLIB，我们就可以通过 SMT solvers 获得大量的推理能力。如果你……这就是一个 MADLIBS 合成数据问题。

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1265419710799806627)** (1 messages): 

> - `DeepSeek Coder V2`
> - `私有推理提供商` 


- **DeepSeek Coder V2 推出私有推理提供商**：**DeepSeek Coder V2** 现在拥有一个[私有提供商](https://openrouter.ai/models/deepseek/deepseek-coder)，可以在 OpenRouter 上处理请求且**不进行输入训练**。
   - 这一新功能在 [X](https://x.com/OpenRouterAI/status/1815860614755147961) 上宣布，标志着私有模型部署迈出了一步。
- **推理提供商的新进展**：私有推理提供商的宣布标志着 **OpenRouter** 平台的战略进展。
   - **不进行输入训练**是与以往模型的重要区别，提升了可用性。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1815860614755147961">来自 OpenRouter (@OpenRouterAI) 的推文</a>：DeepSeek Coder V2 现在在 OpenRouter 上有一个私有提供商提供请求服务，且不进行输入训练！在这里查看：https://openrouter.ai/models/deepseek/deepseek-coder

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1265389919266803853)** (273 条消息🔥🔥): 

> - `Llama 3.1 405B`
> - `Mistral Large 2`
> - `OpenRouter API Issues`
> - `Coding Tools Exploration`
> - `Language Model Pricing` 


- **对 Llama 3.1 405B 性能的担忧**：多位用户对 **Llama 3.1 405B** 的性能表示不满，指出它在处理 NSFW 内容时表现挣扎，经常拒绝提示词（Prompts）或输出训练数据。
   - 用户反馈表明，Temperature 设置严重影响输出质量，一些用户报告在较低的 Temperature 下效果更好。
- **Mistral Large 2 的发布与使用**：**Mistral Large 2** 模型现已作为 **Mistral Large** 上线，实际上取代了之前的版本，并更新了增强的多语言能力。
   - 用户推测其与 **Llama 3.1** 相比的性能表现，特别是在处理法语等语言方面。
- **OpenRouter API 的挑战**：用户讨论了 **OpenRouter API** 的局限性，包括 Rate Limits 和多语言输入的处理，并指出了在使用某些模型时面临的挑战。
   - 报告显示，虽然某些模型在预览期间是免费的，但它们可能对使用量和 Context 有严格限制。
- **对开源编程工具的兴趣**：用户的关注点有所转移，开始询问 **Devika** 和 **Open Devin** 等开源自主编程工具，并根据当前的效能寻求建议。
   - 讨论凸显了人们对尝试主流 AI 产品之外的替代编程方案的兴趣日益增长。
- **模型定价对比**：关于定价的讨论显示，**Mistral Large** 提供了极具竞争力的费率，输入每百万 Tokens 为 **$3**，输出为 **$9**，并与其他模型进行了对比。
   - 用户辩论了各种模型提供的未审查输出的价值，并将其与其它供应商采取的更商业化的方法进行了权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1815837707505131699">来自 OpenRouter (@OpenRouterAI) 的推文</a>：🏆 多 LLM 提示词竞赛。在下方回复对 Llama 405B、GPT-4o 和 Sonnet 具有挑战性的提示词！获胜者将获得 15 个免费额度 ✨。示例：</li><li><a href="https://www.cloudflare.com/5xx-error-landing?utm_source=errorcode_520&utm_campaign=openrouter.ai"">5xx 错误</a>：Cloudflare 是一个免费的全球 CDN 和 DNS 提供商，可以加速并保护任何在线网站</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Meta: 由 meta-llama 提供的 Llama 3.1 405B Instruct</a>：备受期待的 400B 级 Llama 3 来了！凭借 128k Context 和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的 c...</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openai/gpt-4-32k">OpenAI: 由 openai 提供的 GPT-4 32k</a>：GPT-4-32k 是 GPT-4 的扩展版本，具有相同的功能，但 Context 长度增加了四倍，允许在单次运行中处理多达 40 页的文本。这对于...特别有利。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama">未找到标题</a>：未找到描述</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md#instruction-tuned-models">llama-models/models/llama3_1/MODEL_CARD.md at main · meta-llama/llama-models</a>：旨在与 Llama 模型配合使用的实用程序。欢迎在 GitHub 上为 meta-llama/llama-models 的开发做出贡献。</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">响应 | OpenRouter</a>：管理来自模型的响应</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: 适用于 LLM 的用户友好型 WebUI（原 Ollama WebUI）</a>：适用于 LLM 的用户友好型 WebUI（原 Ollama WebUI） - open-webui/open-webui
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1265758130596348077)** (1 条消息): 

> - `Llama 3.1 Release`
> - `HuggingChat Updates`
> - `Community Tools`
> - `Usage Guides` 


- **Llama 3.1 震撼发布**：**Llama 3.1** 模型已正式发布，带来了令人兴奋的新特性和功能。查看 [blogpost](https://huggingface.co/blog/llama31) 获取所有详细信息。
   - 用户可以在 [此处](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct) 体验该模型。
- **探索模型与社区工具**：鼓励用户深入了解 [Hugging Face](https://huggingface.co/meta-llama) 上展示最新进展的 **models**。此外，还可以探索社区资源 [Quants](https://huggingface.co/hugging-quants) 以获取协作见解。
   - 提升体验的资源包括 GitHub 上的 [How to use guide](https://github.com/huggingface/huggingface-llama-recipes)。
- **HuggingChat v0.9.1 版本发布**：最新版本 **HuggingChat v0.9.1** 让每个人都能使用最优秀的 AI 聊天模型，提高了可访问性。用户可以查看模型页面以深入了解其功能。
   - 新版本与 [Llama](https://llama.meta.com/) 功能无缝集成，以增强用户交互。



**提到的链接**：<a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8)">HuggingChat</a>：让社区最优秀的 AI 聊天模型惠及每个人。

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1265387375996174358)** (238 条消息🔥🔥): 

> - `Llama 3.1 讨论`
> - `训练模型`
> - `在 ML 中使用 Rust`
> - `机器学习课程`
> - `模型性能与问题` 


- **Llama 3.1 405B 过载问题**：用户报告称，由于请求过载，Llama 3.1 405B 模型频繁显示“服务不可用”错误。
   - 一些用户讨论了 405B 变体的特性，提到与 70B 版本相比，它感觉受到的审查（censored）更多。
- **训练模型的挑战**：关于训练模型有多次讨论，包括 batch size 未能如预期减少训练时间或步数（steps）的问题。
   - 用户探讨了来自 GitHub 的训练脚本可能存在的缺陷，导致 epochs 的表现与 steps 类似。
- **在机器学习中使用 Rust**：一位用户询问了 Rust 在 ML 社区中的实用性，特别是提到用于性能和 GPU 支持的 'candle' 框架。
   - GitHub 上的 'candle' 项目被推荐为一种专注于机器学习应用的基于 Rust 的解决方案。
- **将 ML 加入学术课程**：一位成员分享了在帮助其经济系将机器学习内容加入本科课程时面临的挑战。
   - 参与者讨论了所需的基础概念，强调了逻辑和编程基础对学生的重要性。
- **AI 生成内容的质量**：用户分享了使用 AI 生成图像的经验，指出了模糊和背景不真实等技术问题。
   - 强调了在进行 fine-tuning diffusion 模型等技术时保持图像质量的重要性，并讨论了 AI 的伦理考量。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper Speaker Diarization - Xenova 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/nroggendorff/oak">nroggendorff/oak · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://x.com/karpathy/status/1815842603377779140">Andrej Karpathy (@karpathy) 的推文</a>: 衷心祝贺 @AIatMeta 发布 Llama 3.1！几点笔记：今天，随着 405B 模型的发布，是具有前沿能力的 LLM 首次可供所有人使用和开发……</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781800565791">用于自然语言处理的 Transformers | 数据 | 电子书</a>: 使用 Python, PyTorch, TensorFlow, BERT, RoBERTa 等构建创新的 NLP 深度神经网络架构。即时交付。顶级移动应用开发产品。</li><li><a href="https://github.com/huggingface/candle">GitHub - huggingface/candle: 适用于 Rust 的极简 ML 框架</a>: 适用于 Rust 的极简 ML 框架。通过在 GitHub 上创建账户为 huggingface/candle 的开发做出贡献。</li><li><a href="https://huggingface.co/AiAF/Lightsource-0Lightsource-OLS_PonyXL.safetensors">AiAF/Lightsource-0Lightsource-OLS_PonyXL.safetensors · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/LyliaEngine/assassin_cross_XL-bf16-pony-v1">LyliaEngine/assassin_cross_XL-bf16-pony-v1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1265407549595713730)** (7 条消息): 

> - `PEFT 模型加载方法`
> - `Stein score 函数关系`
> - `用于摘要生成的模型训练`
> - `UAE 概念`
> - `Elastic Search 与网页爬取` 


- **理解 PEFT 模型加载方法**：讨论了加载 PEFT 模型的两种方法，比较了针对 **ybelkada/opt-350m-lora** 模型使用 **AutoModelForCausalLM.from_pretrained** 与 adapter 加载方法的区别。
   - *在第一种方法中，adapter config 是否负责检索整个模型？*
- **探索 Stein Score 函数**：一位成员对 **Stein score 函数**与**概率密度函数**（probability density function）之间的关系表示困惑，特别是关于包含 log (log pdf) 的疑问。
   - 他们正在寻求关于这个对数函数重要性的清晰解释。
- **训练 BERT 进行摘要生成**：一位成员分享了他们学习使用 **BERT**（配合 **flan-t5-base-samsum** 模型）训练文本摘要模型的经验。
   - 分享了摘要指标，其中亮点包括 **Rouge1 分数为 47.2141**。
- **学习 UAE 概念**：一位成员正在深入研究与 **UAE** 相关的概念，并分享了一篇 [arXiv 论文](https://arxiv.org/pdf/2309.12871) 链接作为学习的一部分。
   - 他们表示已经掌握了一些概念，但也欢迎进一步的解释。
- **Elastic Search 与网页爬取**：成员们讨论了学习 **Elastic Search** 和 **Apify**，重点关注网页爬取、抓取和索引技术。
   - 这些方法对于各种应用中的数据检索和管理至关重要。



**提及的链接**：<a href="https://huggingface.co/sharmax-vikas/flan-t5-base-samsum">sharmax-vikas/flan-t5-base-samsum · Hugging Face</a>：未找到描述

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1265401156734029965)** (4 条消息): 

> - `Meta 的 Llama 3.1 模型`
> - `开源 AI`
> - `马克·扎克伯格的愿景` 


- **Meta 发布 Llama 3.1，AI 领域的游戏规则改变者**：[Meta 最新的 Llama 3.1 模型](https://ai.meta.com/blog/meta-llama-3-1/) 将上下文长度扩展到 **128K**，并提供对**八种语言**的支持，标志着开源 AI 的重大进步。
   - 值得注意的是，**Llama 3.1 405B** 模型拥有的能力足以媲美 OpenAI 的 **GPT-4o** 等闭源模型，且完整模型（包括权重）均可下载。
- **扎克伯格对开源技术的承诺**：在[一篇博客文章](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)中，马克·扎克伯格强调，开源 AI 通过促进创新与协作，对**开发者**、**Meta** 以及社会都有益。
   - 他相信 Llama 可以进化为一个强大的开源 AI 生态系统，使开发者能够解锁新的工作流并创建增强其项目的工具。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3-1/">未找到标题</a>：未找到描述</li><li><a href="https://www.neowin.net/news/mark-zuckerberg-explains-why-open-source-ai-is-good-for-developers/">马克·扎克伯格解释了为什么开源 AI 对开发者有好处</a>：马克·扎克伯格认为开源 AI 是 AI 的未来，能够促进不受限制的创新，类似于开源开发在其他领域加速进步的方式。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1265410055986217051)** (4 messages): 

> - `Mistral-NeMo 12B Instruct`
> - `Pony Diffusion v6`
> - `Llama 3.1 Release` 


- **与 Mistral-NeMo 12B Instruct 进行极速对话**：一个展示 **Mistral-NeMo 12B Instruct** 的 Demo 已经发布，该版本使用了 [llama.cpp](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp) 实现，性能表现令人印象深刻。
   - 鼓励用户尝试，体验**清爽**的对话。
- **Pony Diffusion v6 获得每周更新**：最新版本的 **Pony Diffusion v6** 最近发布，为高级用户提供了许多选项，并每周滚动更新。
   - 项目可以在[这里](https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6)找到，它与 artificialguybr 之前的 Demo 有所关联。
- **社区对 Llama 3.1 感到兴奋**：社区对 **Llama 3.1** 的发布反响热烈，基于 **HF Inference API** 构建的新 Space 允许自定义系统指令。
   - 点击[这里](https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8)查看 Space，探索免费提供的功能，让每个人都能轻松上手。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp">Mistral NeMo llama.cpp - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6">HD Pony Diffusion - a Hugging Face Space by Sergidev</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8">Llama 3.1 405B FP8 - a Hugging Face Space by as-cle-bert</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1265619447708647434)** (2 messages): 

> - `Object Detection in Java` 


- **对目标检测应用教程感到兴奋**：一位成员分享了一篇[博客文章](https://blog.stackademic.com/object-detection-app-in-java-a50ca86306ff)，详细介绍了如何使用 **Java** 开发**目标检测应用 (Object Detection App)**。
   - *太棒了！！！！* 是另一位成员的热情回应，表明该主题受到了积极的欢迎和关注。
- **社区对 Java 开发的参与**：成员们对 **Java** 开发技术表现出浓厚兴趣，特别是像分享的这类教程。
   - 这种兴奋感反映了社区对软件开发中实际应用和学习资源日益增长的兴趣。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1265739960204660787)** (1 messages): 

> - `Chameleon models`
> - `Batch processing images` 


- **关于 Chameleon 模型的咨询**：一位成员询问是否有人研究过 **Chameleon 模型**，并表示在如何为批量前向传播（forward pass）进行图像批处理/整理（batch/collate）方面存在疑问。
   - *有人能分享关于这些模型图像处理的见解吗？*
- **提出的批处理问题**：讨论强调了需要明确如何为 **Chameleon 模型** 有效地实现批处理和图像整理。
   - 几位成员表示有兴趣分享他们在前向传播过程中进行批处理的经验和最佳实践。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1265560960404422666)** (8 messages🔥): 

> - `Training Sentence Encoders` (训练句子编码器)
> - `Metrics for Model Evaluation` (模型评估指标)
> - `Fine-tuning Sentence Transformers` (微调 Sentence Transformers)
> - `RAG Pipeline for Q&A` (用于问答的 RAG 流水线)
> - `Text-to-HTML/CSS Generation Model` (文本到 HTML/CSS 生成模型)


- **MultipleNegativesRankingLoss 的挑战**：一位成员表达了在使用 **MultipleNegativesRankingLoss** 训练句子编码器时遇到的困难，并注意到在增加 **CachedMultipleNegativesRankingLoss** 的 batch size 时性能反而下降。
   - 他们询问了在增加 batch size 时可能出现的常见数据集问题，旨在获得更好的模型结果。
- **模型评估指标**：一位成员概述了他们的评估指标，基于微调模型的小型向量数据库使用 **recall@5**、**recall@10** 和 **recall@20**。
   - 他们还提到利用名为 **TripletEvaluator** 的评估器来衡量模型性能。
- **为法律规范微调 Sentence Transformers**：一位初学者寻求关于在针对**法律和金融规范**的数据集上微调 Sentence Transformer 的指导，目标是构建用于问答的 RAG 流水线。
   - 他们请求提供成功完成此任务的步骤和推荐读物。
- **对 Tiktoken 经验的兴趣**：一位成员询问了其他人使用 **tiktoken** 的经验，呼吁分享见解。
   - 这突显了对该工具在相关项目中集成和有效性的好奇。
- **开源文本到 HTML/CSS 生成模型**：一位成员宣布打算获取一个**开源文本到 HTML/CSS 生成模型**并寻求推荐。
   - 这反映了对促进文本内容转换为网页格式工具的持续探索。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1265485302433316914)** (6 messages): 

> - `Rectified Flow`
> - `Flow Matching`
> - `DDPM and DDIM Discussions`
> - `Evaluation of Generative Models`
> - `VAE Model Cards` 


- **对 Rectified Flow 缺乏兴趣**：一位成员表示沮丧，虽然有很多关于 **DDPM** 和 **DDIM** 的讨论，但很少有人谈论 **Rectified Flow** 或 **Flow Matching**。
   - 他们强调很难找到 **Flow** 的最小示例（例如生成 **MNIST**），并质疑大家对该话题的普遍兴趣。
- **Diffusers 中的 Flow 调度器**：另一位成员指出 `diffusers` 库中存在 **FlowMatchEulerDiscreteScheduler** 和 **FlowMatchHeunDiscreteScheduler**，暗示了它们与讨论的相关性。
   - 这些资源可以在 [Hugging Face 文档](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete)中找到。
- **生成模型的评估方法**：一位成员引用了一份讨论评估 **Diffusion 模型**的定性和定量方法的文档，强调了模型选择的复杂性。
   - 他们提到定性和定量评估都为比较 **Stable Diffusion** 和 **GANs** 等模型提供了更强的信号。
- **询问 VAE 模型**：一位成员询问讨论中提到的特定 **VAE**，寻求对其身份的澄清。
   - 他们请求分享其对应的模型卡片以获取更多见解。



**提到的链接**：<a href="https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation">Evaluating Diffusion Models</a>：未找到描述

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1265384287952437370)** (239 messages🔥🔥): 

> - `Kohya-ss GUI 问题`
> - `Lycoris 集成更新`
> - `模型性能评分`
> - `Stable Diffusion 模型对比`
> - `新 AI 视频生成模型发布公告` 


- **Kohya-ss GUI 面临兼容性问题**：用户报告当前版本的 **Kohya-ss GUI** 与 Python 3.10 不兼容，需要升级到 3.10.9 或更高版本。
   - *一位用户幽默地评论道*，这就像是要求体重限制在 **180 磅但不能超过 180.5 磅**，反映了这种限制的荒谬性。
- **Lycoris 集成正在开发中**：提到 **Onetrainer** 可能很快会在新的开发分支中实现 **Lycoris** 功能，社区围绕各种功能展开了讨论。
   - 提到了对 Kohya 脚本的 **bmaltais UI 封装器** 的偏好，这提升了这些集成的用户体验。
- **社区对艺术模型性能的评分**：一场关于 **Kolors, Auraflow, Pixart Sigma 和 Hunyuan** 等模型性能评分的讨论展开，其中 Kolors 因其速度和质量而受到青睐。
   - 参与者强调了不同的用户体验，深入辩论了每个模型的具体特性和应用。
- **评估 Stable Diffusion 模型能力**：几位用户辩论了 **Stable Diffusion 1.5** 和 **SDXL** 在细节和分辨率质量方面的输出及可用性差异。
   - **Hidiffusion** 和 **Adaptive Token Dictionary** 等先进技术被强调为增强旧模型输出的有效手段。
- **推出用于多视角视频生成的 Stable Video 4D**：推出了 **Stable Video 4D** 模型，使用户能够将单个物体视频转换为新的视角，以增强创意项目。
   - 该新模型目前处于研究阶段，预计将应用于**游戏开发、视频编辑和虚拟现实**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://stability.ai/news/stable-video-4d">Stable Video 4D &mdash; Stability AI</a>：我们很高兴宣布推出 Stable Video 4D，这是一个创新模型，允许用户上传单个视频并获得八个新角度/视角的动态新视图视频，提供...</li><li><a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">xinsir/controlnet-union-sdxl-1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://civitai.com/models/207437/ballz">BALLZ - Ballz 3 | Stable Diffusion LoRA | Civitai</a>：Mad Balls！直接来自 80 年代的泡沫玩具。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1265404758043394178)** (58 条消息🔥🔥): 

> - `语言模型中的采样方法`
> - `Llama 3.1 基准测试`
> - `对数似然评估`
> - `Greedy vs stochastic sampling`
> - `采样中的尾部概率` 


- **理解 LLM 的采样方法**：成员们讨论了语言模型中使用的各种采样方法，如 **greedy sampling**、**top-p** 和 **top-k**，并强调了它们之间的权衡。
   - **Stochastic sampling**（随机采样）允许输出的多样性，但需要多次运行才能获得具有统计显著性的结果，这与 **greedy sampling** 的可靠性形成对比。
- **Llama 3.1 采样偏好**：在利用 **Llama 3.1** 进行基准测试的背景下，成员们建议查阅其论文以获取推荐的采样方法，共识倾向于使用概率采样。
   - 一位成员指出 **Gemma 2** 利用了 **top-p** 和 **top-k**，这在模型评估中非常典型。
- **对数似然作为衡量工具**：**Log likelihood** 被强调为评估模型性能的重要指标，可以比较模型在不同采样方法下复制结果的效果。
   - 建议使用 **log likelihood** 可以帮助理解采样选择如何影响输出分布和整体模型的可靠性。
- **Greedy Sampling 作为基准**：**Greedy sampling** 是模型评估中可靠的基准，它在巨大的输出空间中生成概率最高的输出路径。
   - 成员们认为，虽然 **stochastic sampling** 可以产生多样化的输出，但它使评估变得复杂，并且需要大量的运行才能达到统计显著性。
- **长序列生成的挑战**：讨论涉及了衡量长生成序列质量的复杂性，并指出了与采样方法和 **log likelihood** 相关的注意事项。
   - 有人担心尾部概率（tail probabilities）可能导致输出中的复合错误，从而影响模型的长期表现和结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://kipp.ly/transformer-param-count/">LLM Parameter Counting | kipply&#x27;s blog</a>: kipply 关于她所做、所读或所观察事物的博客</li><li><a href="https://github.com/EleutherAI/cookbook/tree/main/calc">cookbook/calc at main · EleutherAI/cookbook</a>: 深度学习入门。包含处理真实模型时的所有实践细节和有用工具。 - EleutherAI/cookbook</li><li><a href="https://x.com/stephenroller/status/1579993017234382849">Stephen Roller (@stephenroller) 的推文</a>: @srush_nlp 我发现不熟悉 Scaling 的人会被这个吓到：</li><li><a href="https://build.nvidia.com/explore/discover#llama-3_1-405b-instruct">尝试 NVIDIA NIM APIs</a>: 立即体验领先模型，构建企业级生成式 AI 应用。</li><li><a href="https://www.lesswrong.com/posts/3duR8CrvcHywrnhLo/how-does-gpt-3-spend-its-175b-parameters)">GPT-3 是如何分配其 175B 参数的？ — LessWrong</a>: [目标受众：一周前的我，以及对 ML 有一定了解但想在技术层面更好理解 Transformer 的人。] …
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1265394224631648428)** (132 条消息🔥🔥): 

> - `关于模型性能的误导性推文`
> - `MoE vs Dense 模型`
> - `Character.AI 的模型架构`
> - `Mixtral 与 Mistral 模型设计`
> - `LLM 训练中的外部数据` 


- **关于模型性能的误导性推文和假设**：讨论围绕一条与语言模型中投影计算相关的误导性推文展开，特别关注 **Character.ai** 如何使用共享 KV 层以及这如何影响性能指标。
   - 成员们对所分享信息的准确性表示困惑，并强调了理解 Transformer 架构的个人探索过程。
- **关于 MoE vs Dense 模型的辩论**：参与者分析了为什么 **Dense 架构** 比 **Mixture-of-Experts (MoE)** 模型更受青睐，理由是在大规模训练中处理 MoE 的高成本和工程要求。
   - 有观点认为，一旦模型完成预训练，MoE 在效率方面应该表现更好，尽管人们对组织内部不同的工程能力表示担忧。
- **Character.AI 模型架构的见解**：分享了关于 **Character.AI** 架构选择的见解，强调了他们如何通过设计优化来高效管理推理，尽管其博客文章中的确切细节仍不清楚。
   - 参与者注意到跨层共享缓存的潜力，暗示该模型可能受益于尚未公开阐明的架构信息。
- **Mistral 和 Mixtral 的模型选择**：讨论中提到，尽管 **Mistral** 和 **Mixtral** 最近的模型有能力实现 MoE，但仍选择了 Dense 架构，这让一些成员感到意外。
   - 训练中持续存在的挑战以及推理过程中的效率问题被认为是这些设计决策的关键原因。
- **在 LLM 训练中利用外部数据**：分享了一篇关于在训练语言模型时利用外部来源的论文，为超越传统方法、提高复杂推理任务的性能铺平了道路。
   - 这激发了成员们的好奇心，去探索新模型如何结合此类创新技术以实现更好的信息检索和任务执行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.03133">Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training</a>：混合专家（MoE）模型促进了高效扩展；然而，训练路由网络带来了优化不可微、离散目标的挑战。最近，一种全微分的...</li><li><a href="https://huggingface.co/papers/2204.05149">论文页面 - The Carbon Footprint of Machine Learning Training Will Plateau, Then
  Shrink</a>：未找到描述</li><li><a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>：在 Character.AI，我们正致力于实现 AGI。在未来的状态下，大语言模型（LLMs）将增强日常生活，提供业务生产力和娱乐，并帮助人们...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>：大型 Transformer 模型在许多任务上取得了令人印象深刻的结果，但训练甚至微调的成本都很高，且解码速度极慢，以至于它们的使用和研究变得遥不可及。我们解决了这个问...</li><li><a href="https://arxiv.org/abs/2112.04426">Improving language models by retrieving from trillions of tokens</a>：我们通过基于与前序 token 的局部相似性，从大型语料库中检索文档块进行条件化，从而增强了自回归语言模型。通过一个 2 万亿 token 的数据库，我们的 Re...</li><li><a href="https://github.com/xuekt98/bbdm">GitHub - xuekt98/BBDM: BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models</a>：BBDM：使用布朗桥扩散模型的图像到图像翻译 - xuekt98/BBDM</li><li><a href="https://arxiv.org/abs/2403.13097">Simple Ingredients for Offline Reinforcement Learning</a>：离线强化学习算法已证明在与目标下游任务高度相关的数据集上是有效的。然而，利用一个新的测试平台（MOOD），其中轨迹来自异构...</li><li><a href="https://www.normalcomputing.com/blog-posts/supersizing-transformers-going-beyond-rag-with-extended-minds-for-llms">Normal Computing</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1265464079003287653)** (21 条消息🔥): 

> - `Llama API 评估`
> - `Chat 格式模型使用`
> - `多选题任务处理` 


- **通过 lm_eval 进行 Llama API 评估**：成员们讨论了在使用 `lm_eval` 工具通过 `llama-api.com` 的 API 评估 **Llama 3.1-405B** 模型时遇到的错误，特别是关于 logits 支持和多选题任务的问题。
   - *“它报错：不支持 logits。”* 引发了一系列的排障尝试，包括检查 URL 格式和 API key 的使用。
- **API 配置问题**：为了解决 'Method Not Allowed' 错误，建议使用完整的 API URL，并确保正确配置了 temperature 和 max tokens 等参数。
   - 一位成员成功修改了 `_create_payload` 方法来解决这些问题，从而在特定配置下实现了模型评估功能的正常运行。
- **多选题处理**：在成功运行 `gsm8k` 等任务的评估后，在处理 `mmlu_college_biology` 等多选题任务时出现了错误，特别是与内容处理相关的 'AttributeError'。
   - 这引发了关于 API 输出与评估框架兼容性的疑问，成员们正在寻求解决方案并分享错误日志以供进一步分析。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gist.github.com/riteshdroid/0ec4525c3a315dcf373f16e9df5d1833">gist:0ec4525c3a315dcf373f16e9df5d1833</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/42dc244867889a19ae80847254a481f446f6e4b7/lm_eval/models/openai_completions.py#L121">lm-evaluation-harness/lm_eval/models/openai_completions.py at 42dc244867889a19ae80847254a481f446f6e4b7 · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L86">lm-evaluation-harness/lm_eval/models/openai_completions.py at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/riteshdroid/lm-evaluation-harness/blob/1a2dc674c3dfcff81e9c6f0bf495ba569106c931/lm_eval/models/api_models.py#L140">lm-evaluation-harness/lm_eval/models/api_models.py at 1a2dc674c3dfcff81e9c6f0bf495ba569106c931 · riteshdroid/lm-evaluation-harness</a>: 一个用于语言模型 few-shot 评估的框架。 - riteshdroid/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/42dc244867889a19ae">GitHub - EleutherAI/lm-evaluation-harness at 42dc244867889a19ae80847254a481f446f6e4b7</a>: 一个用于语言模型 few-shot 评估的框架。 - GitHub - EleutherAI/lm-evaluation-harness at 42dc244867889a19ae80847254a481f446f6e4b7</li><li><a href="https://api.llama-api.com,model=llama3.1-405b">无标题</a>: 未找到描述</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1174">Implementing local OpenAI API-style chat completions on any given inference server by veekaybee · Pull Request #1174 · EleutherAI/lm-evaluation-harness</a>: 此 PR 通过将 base_url 传递给新类 LocalChatCompletionsLM 来解决问题 #1072，该类继承自 OpenaiChatCompletionsLM 并接受本地 HuggingFace 风格的模型...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1265458292726042746)** (25 条消息🔥): 

> - `GPU Bit-Matching`
> - `GPU FLOPS and Data Types`
> - `Non-Deterministic Results in Floating Point Operations`
> - `CUDA Lookback Scan Algorithm`
> - `NCCL Computation Overlap Issues` 


- **理解 GPU Bit-Matching**：有人提出了关于特定 GPU 型号的结果在给定某些输入时是否具有唯一的 bit-matched（位匹配）特性的问题，成员们指出这取决于所使用的 **algorithm**（算法）。
   - 另一位成员评论说，对于大多数算法，如果在同一块 GPU 上运行，结果是一致的。
- **GPU FLOPS 与数据类型的依赖关系**：一位成员澄清说，GPU FLOPS 的数值受 **data types**（数据类型）以及计算是使用 CUDA cores 还是 **tensor cores** 的影响很大。
   - 另一位成员补充说，Nvidia 的规格参数在其 **whitepapers**（白皮书）中提供了不同数据类型的详细性能数据。
- **浮点运算中的非确定性**：讨论指出，使用 **floating point**（浮点）数据有时会根据操作顺序产生有益的非确定性结果。
   - 正如所提到的，kernel 调优或硬件的微小变化都可能导致结果的变化，从而增加调试的复杂性。
- **CUDA Mode 中的 Lookback Scan 算法**：一位成员提到了关于 **lookback scan** 算法的 CUDA mode 视频，并建议该算法有时可以进行切换。
   - 然而，成员们很难找到讨论如何利用该算法的文档或示例。
- **NCCL 计算重叠的挑战**：据报告，关于在 backward pass 期间将计算与 **NCCL** 重叠的建议在实现上并未达到预期的简易程度。
   - 引用了一个 GitHub issue，强调了在 ResNet-50 的多 GPU 训练背景下使用 NCCL 时遇到的困难。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/soumithchintala/status/1815829457858625642">来自 Soumith Chintala (@soumithchintala) 的推文</a>：为什么 16k GPU 任务会失败？Llama3 的论文有很多酷炫的细节——但值得注意的是，它有一个巨大的基础设施章节，涵盖了我们如何进行并行化、保持可靠性等。我们达到了 90% 的整体效率...</li><li><a href="https://github.com/NVIDIA/nccl/issues/338">与 NCCL 重叠的计算变得慢得多 · Issue #338 · NVIDIA/nccl</a>：我使用了来自 https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 的环境，通过多 GPU（使用 horovod 和 nccl）训练 resnet-50，发现...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1265657262085509162)** (1 条消息): 

> - `Profiling Triton kernels`
> - `Accelerating current Triton GPTQ kernels`
> - `Integration of Triton kernels into PyTorch` 


- **寻求 Profiling Triton Kernel 的帮助**：一位用户询问了如何按照 [PyTorch 博客文章](https://pytorch.org/blog/accelerating-triton/) 中描述的方法对 Triton kernels 进行 profiling。他们强调了使用 Triton 实现的加速效果，但需要关于实施 profiling 技术的指导。
- **加速 Triton GPTQ Kernels 的步骤**：该博客概述了一种第一性原理方法，该方法将核心 GPTQ 的 Triton GPTQ kernels 加速了 **3x**，将 AutoGPTQ 加速了 **6x**，对于典型的 Llama 风格推理，时间从 **275us 降低到 47us**。它强调了合并内存访问（coalesced memory access）的有效性以及减少 warp stalling 以提高吞吐量的策略。
- **将 Triton Kernels 集成到 PyTorch**：作为优化工作的一部分，该博客讨论了将 Triton kernels 集成到 PyTorch 代码中，强调了其取代现有原生 CUDA 实现的潜力。随着时间的推移，这种集成旨在超越传统 CUDA 原生 GPTQ kernels 的性能。



**提到的链接**：<a href="https://pytorch.org/blog/accelerating-triton/">为 GPTQ 加速 Triton 反量化 Kernels</a>：TL;DR  

  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1265395795671384195)** (13 条消息🔥): 

> - `torch.compile 性能`
> - `使用 torch.compile 时的 GPU 显存占用`
> - `CUDA kernel 反模式`
> - `PyTorch profiling 工具`
> - `PyTorch 中的 CUDA graphs` 


- **torch.compile 在小型 Bert 模型上表现不佳**：一位用户报告称，在对小型 **Bert model** 测试 `torch.compile` 时，**RAM 占用**显著增加，导致 batch size 从 **512 降至 160**，且速度比使用 eager mode 更慢。
   - 尽管在使用 `full_graph=True` 时编译没有问题，该用户仍在寻求导致观察到的性能下降的潜在原因。
- **针对显存问题建议使用 Profiler**：一位成员建议使用 **PyTorch profiler** 及其 **memory trace tool**，以深入调查模型推理过程中的显存使用情况。
   - 这种方法可以深入了解是否是特定的配置或用法导致了显存需求的增加。
- **CUDA Graphs 与配置查询**：一位用户确认他们没有显式设置 **CUDA graphs**，在使用 `torch.compile` 时保持默认设置；他们提到使用的是 **2.3.1 和 2.4 RC** 版本。
   - 互动中强调了 **Inductor configurations**，以及更改这些配置是否会影响模型编译期间的性能。
- **强调 CUDA Kernel 反模式**：一位成员强调了在 PyTorch 中编写 CUDA kernels 时一个微妙的**反模式 (anti-pattern)**，该模式与 **GMEM scratch space** 分配有关，建议注意 kernel 启动之外的 tensor 生命周期。
   - 这一见解源于对 CI 失败的调试，并与开发新算子 (ops) 时对临时 tensor 的仔细管理有关。
- **开启 `reduce-overhead` 的 `torch.compile` 未见差异**：用户观察到，在他们的 `torch.compile` 设置中，无论是否使用 **`reduce-overhead`** 和 **`fullgraph`** 选项，显存占用都没有变化。
   - 这一稳定的观察结果说明了在实践中理解编译模式与显存效率之间关系的必要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hud.pytorch.org/benchmark/compilers">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/131277">由 drisspg 修复 Flash-Attention splitkv kernel 中的 IMA · Pull Request #131277 · pytorch/pytorch</a>: 摘要：在调试 flash_attention 测试的 CI 失败时，我偶然发现了 flash attention 的 split-kv 变体的 2 个 IMA。在写入 softmax_lse_a 期间发生了非法的全局内存写入...
</li>
</ul>

</div>

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1265391454755487794)** (16 条消息🔥): 

> - `VLM Performance`
> - `CUDA Advancement`
> - `Mistral Large 2`
> - `FP16/FP32 Intrinsics`
> - `Feature Engineering Success` 


- **VLM 在文本生成方面表现更优**：讨论强调，即使在可以使用 VLM 的情况下，它们在文本任务上的表现通常也优于图像处理，正如 [GPT-4o 在 ARC 测试集上的表现](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt)所显示的。
   - Ryan 发现，与完全依赖 GPT-4o 的视觉能力相比，对问题网格进行特征工程（Feature Engineering）可以产生更好的结果。
- **CUDA 旨在超越 cuBLAS**：一名成员宣布了即将到来的 CUDA 进展，声称 *我们将在广泛的矩阵尺寸上超越 cuBLAS*。
   - 这不仅包括对 SGEMM 的潜在增强，还包括对其他操作类型的增强。
- **Mistral Large 2 展示先进特性**：[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 拥有 128k 的上下文窗口（Context Window），支持多种语言和 80 多种编程语言，专为高效的单节点推理而设计。
   - 该模型拥有 1230 亿个参数，面向长上下文应用，并根据研究许可证发布，允许非商业用途。
- **FP16/FP32 对性能的影响**：围绕 NVIDIA 的 FP16/FP32 硬件内建函数（Intrinsics）展开了讨论，这可能会显著影响性能结果。
   - 这引发了对 CUDA 生态系统未来发展的期待。
- **有趣的基准测试对比**：成员们发现 Mistral 最新的基准测试非常吸引人，因为它们 *突破了成本效率、速度和性能的界限*。
   - 提供的新功能有助于为各种场景构建创新的 AI 应用程序。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>：今天，我们发布了 Mistral Large 2，这是我们旗舰模型的新一代产品。与前代产品相比，Mistral Large 2 在代码生成、数学和推理能力方面有了显著提升...</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>：你可以通过增加采样量来实现。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1265752468273234024)** (1 条消息): 

> - `ML/AI career roadmap`
> - `Internship opportunities`
> - `Job search strategies` 


- **寻求 ML/AI 职业路径指导**：一位用户请求帮助设计一份路线图，以获得 **ML/AI** 角色的全职职位和实习机会，并分享了一份 [包含详细信息的 Google 文档](https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing)。
   - 他们强调愿意为了达成目标而长时间工作，并对路线图的任何建议持开放态度。
- **对实习建议持开放态度**：该用户正在寻求关于他们在 ML/AI 领域获得 **实习** 机会的方法反馈，以及他们的计划是否可行。
   - 他们明确表示，不应认为时间表不切实际，因为他们可以投入额外的时间来完成任务。



**提到的链接**：<a href="https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing">ML Roadmap</a>：3 个月 - (9月, 10月, 11月) 路线图。统计学：https://www.youtube.com/watch?v=MXaJ7sa7q-8&amp;list=PL0KQuRyPJoe6KjlUM6iNYgt8d0DwI-IGR&amp;t=11s (1 周) 线性代数 - https://www.youtube.com/wat...

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1265612314711953460)** (10 条消息🔥): 

> - `CUDA 安装问题`
> - `显存溢出 (Out of Memory) 错误`
> - `Llama-2 Chat 模型`
> - `将模型作为 Discord 机器人运行` 


- **Torch 未启用 CUDA 编译**：一位成员发现其 **Torch** 在编译时未启用 **CUDA**，并寻求解决办法。
   - 另一位成员建议直接从安装页面安装 CUDA 版本，页面会提供所需的精确命令。
- **CUDA 运行成功但遇到瓶颈**：在使 CUDA 正常工作后，一位成员遇到了 **torch.cuda.OutOfMemoryError**，表示在尝试分配 **172.00 MiB** 时 GPU 显存不足。
   - 他们收到建议调整 **max_split_size_mb** 以防止内存碎片，并参考相关文档进行解决。
- **探索 Llama-2 7B 模型**：一位成员分享了他们微调的 [Llama-2 7B](https://huggingface.co/TheBloke/Llama-2-7B-fp16) 模型的细节，该模型使用 Wizard-Vicuna 数据集在 **24GB GPU** 上训练了 **19 小时**。
   - 他们提供了该模型多个版本的链接，包括托管在 Hugging Face 上的 **GGML** 和 **GPTQ** 版本。
- **将模型作为 Discord 机器人运行**：一位成员表达了将 Llama-2 模型作为 **Discord 机器人**运行的兴趣，展示了对其功能的极大热情。
   - 这一表态反映了将 AI 模型集成到社区平台的广泛兴趣。



**提到的链接**：<a href="https://huggingface.co/georgesung/llama2_7b_chat_uncensored">georgesung/llama2_7b_chat_uncensored · Hugging Face</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1265489453162434658)** (8 条消息🔥): 

> - `Torch AO 中的 ImportError`
> - `支持的 PyTorch 版本`
> - `剪枝 (Pruning) 与量化 (Quantization) 问题` 


- **ImportError: 从 Torch 导入**：一位新用户在尝试从 *torch.utils._python_dispatch* 导入 'return_and_correct_aliasing' 时遇到了 *ImportError*，这表明存在版本不兼容问题。
   - 建议参考 [此 GitHub issue](https://github.com/pytorch/ao/issues/29) 以进行进一步调查。
- **在 PyTorch 版本上进行测试**：成员们指出他们不会在 *2.2 之前的 PyTorch 版本*上进行测试，这意味着用户应升级版本以获得最佳功能。
   - 一位用户确认他们将根据此建议尝试升级到 *torch 2.2*。
- **关于 Llama 3.1 推理延迟的担忧**：一位用户询问 *Llama 3.1 8b* 相比 *3.0* 是否改进了推理延迟，引发了关于模型性能的持续讨论。
   - 目前没有关于模型具体延迟性能的回复。
- **剪枝与量化教程中的问题**：一位用户在结合量化实现结构化剪枝时，对 *weight_orig* 和 *weight_mask* 的转换感到困惑，并寻求解答。
   - 他们在尝试应用 *detach* 操作时遇到了与 *deepcopy* 协议相关的 *RuntimeError*，这导致模型推理中断。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/pruning_tutorial.html">Pruning Tutorial — PyTorch Tutorials 2.4.0+cu124 documentation</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/issues/29`">Issues · pytorch/ao</a>：用于训练和推理的自定义数据类型和布局 - Issues · pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1265754646509654149)** (1 条消息): 

> - `Llama 3 中的分块注意力 (Blockwise Attention)`
> - `输入序列切分` 


- **在 Llama 3 中实现分块注意力**：一位用户询问在 **Llama 3** 架构中实现 **分块注意力 (blockwise attention)** 时，将输入序列切分为块的正确阶段。
   - 他们具体询问这应该发生在对向量 **Q** 和 **K** 应用 **旋转位置嵌入 (rotary position embeddings)** 之后，还是在 **自注意力模块 (self-attention block)** 之前。
- **关于序列处理的澄清**：该用户对 **Llama 3** 架构的具体实现表示困惑，表明需要明确输入序列的处理方式。
   - 讨论围绕如何将 **分块注意力** 有效集成到模型处理流程中的最佳策略展开。


  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/)** (1 条消息): 

iron_bound: 挺酷的 https://github.com/AnswerDotAI/fsdp_qlora/tree/llama400b
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1265383434662973582)** (71 条消息🔥🔥): 

> - `KV Cache 实现`
> - `ZeRO-2 性能见解`
> - `LLaMA 与 muP 对比`
> - `随机舍入 (Stochastic Rounding) 策略`
> - `GPT-2 训练实验` 


- **KV Cache 逻辑进展**：一名成员报告成功实现了针对 Attention 的部分 KV Cache 逻辑，其中涉及在不改变布局的情况下智能地使用现有 Buffer。
   - 调试发现第二次传递（second pass）期间 Token 结果存在差异，但整体实现显示出显著进展。
- **ZeRO-2 性能见解**：使用 ZeRO-2 和 2 个 GPU 进行的测试显示，在较小模型上梯度内存估计可节省 25%，并考虑了可扩展性计划。
   - 尽管有所改进，但注意到梯度计算在通信阶段需要额外的副本，这带来了挑战。
- **LLaMA 与 muP 技术对比**：围绕 LLaMA 与 muP 的性能对比展开了讨论，特别是关于使用 tanh soft clamping 等技术。
   - 讨论中提出了关于 muP 是提升了性能，还是主要提供了更好的学习率迁移（learning rate transfers）的疑问。
- **梯度累积中的随机舍入 (Stochastic Rounding)**：一名成员强调了一种在梯度累积中提出的随机舍入方法，以提高训练稳定性和效率。
   - 这种方法可能会带来更有效的梯度更新，同时可能允许在训练期间进行更大规模的累积。
- **GPT-2 实验训练结果**：在一个 GPT-2 350M 模型上完成了训练，使用 Fineweb-edu 数据集并交织 OpenHermes 数据进行指令预训练。
   - 尽管出现了一些奇特的训练损失（training loss）模式，但整体结果被认为是稳定的，该模型已在 Hugging Face 上公开可用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: 从小宽度到大宽度模型的鲁棒且有效的缩放通常需要精确调整许多算法和架构细节，例如参数化和优化器选择...</li><li><a href="https://github.com/microsoft/mup/issues/76">Not getting perf improvements from muP at ~1.5B scale · Issue #76 · microsoft/mup</a>: 嘿伙计们，首先感谢你们的出色工作！我在 llm.c 项目中实现了 muP（见此处），坐标检查（coord checks）看起来是平坦/正确的（我一直进行到 15 步仍然平坦！），但我...</li><li><a href="https://github.com/karpathy/llm.c/pull/593/files#diff-c8a8f83fdc5921f95e3e09a1b2f475f8342a20042d8bb4a9eea3e291c8b4ad11R596-R607">Zero 2 - WIP by ngc92 · Pull Request #593 · karpathy/llm.c</a>: 尝试让第一个版本运行起来。代码还不漂亮，我们目前在通信代码中失去了异步性，因为我们需要为下一层重用 Buffer，而且它不...</li><li><a href="https://huggingface.co/jrahn/gpt2_350M_edu_hermes">jrahn/gpt2_350M_edu_hermes · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/307">Improve tanh derivative in backward gelu by akbariyeh · Pull Request #307 · karpathy/llm.c</a>: 计算 tanh 的导数为 1 - tanh^2 比计算 1/(cosh^2) 更便宜。这可能不会产生可衡量的差异。</li><li><a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>: 如果设备内存耗尽，使用 cudaMallocManaged 分配优化器状态，这样即使无法容纳优化器状态，我们仍然可以（缓慢地）进行训练。这是基于 #694，应该会被合并...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1265425478198493258)** (3 条消息): 

> - `AMD 的 FlashAttention 支持`
> - `MI200 和 MI300 兼容性`
> - `GitHub Pull Requests` 


- **FlashAttention 现在支持 AMD ROCm**：最近的 [GitHub Pull Request #1010](https://github.com/Dao-AILab/flash-attention/pull/1010) 在 **FlashAttention 2** 库中实现了对 **AMD ROCm** 的支持，包括 `mha_fwd` 和 `mha_varlen_fwd` 等多个 C++ API。
   - 该实现基于 composable kernel 技术，保持了与原始版本的 **API 一致性**。
- **对 MI200 和 MI300 的有限兼容性**：关于新更新的 FlashAttention 的兼容性，声明中提到：*“目前我们仅支持 **mi200 和 mi300**”*。
   - 这为当前的支持范围划定了明确界限，暗示未来可能会有更广泛兼容性的更新。



**提到的链接**：<a href="https://github.com/Dao-AILab/flash-attention/pull/1010">Support AMD ROCm on FlashAttention 2 by rocking5566 · Pull Request #1010 · Dao-AILab/flash-attention</a>：此 PR 实现了 C++ Flash API 的 AMD / ROCm 版本，包括 mha_fwd、mha_varlen_fwd、mha_bwd、mha_varlen_bwd。Kernel 实现来自 composable kernel，C++ API 与原始版本相同...

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1265387780696182906)** (87 条消息🔥🔥): 

> - `Llama 3.1 错误`
> - `Mistral Large 模型发布`
> - `多语言模型性能`
> - `训练和微调挑战`
> - `模型中的合成数据生成` 


- **Llama 3.1 遇到错误**：用户报告了 **Llama 3** 导致 *AttributeError* 等错误的问题，讨论认为过时的镜像或配置可能是原因。
   - 一位用户提到尝试使用不同的镜像来解决问题，而另一位用户则对频繁的模型更新表示普遍的沮丧。
- **Mistral Large 模型开源**：Mistral 发布了 **Mistral-Large-Instruct-2407** 模型，拥有 **123B 参数**，并声称具有 SOTA 性能。
   - 主要特性包括支持数十种语言的多语言支持、精通编程以及先进的 Agent 能力，引发了用户的兴奋。
- **多语言模型性能讨论**：**Llama 3.1** 和 **NeMo** 的对比显示了性能差异，特别是在多语言能力方面，不同语言各有优势。
   - 用户指出，虽然 **Llama 3** 支持一些欧洲语言，但据报道 **NeMo** 对**中文**和其他语言提供了更好的支持。
- **模型训练和微调的挑战**：有人担心有效训练像 Mistral 这样的大型模型需要大量的 RAM，一些用户评论了他们的局限性。
   - 有人表示在训练过程中遇到了梯度爆炸（exploding gradients）的困难，思考这是否与 sample packing 有关。
- **提到合成数据生成**：**Llama 3.1** 的发布提到了“合成数据生成”，引发了对内部文档脚本的需求。
   - 用户讨论了这一想法对于微调和训练模型可能带来的益处。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/1littlecoder/status/1815768634297405811">来自 1LittleCoder💻 (@1littlecoder) 的推文</a>：Llama 3.1 的发布特别提到了“合成数据生成”（受 @Teknium1 的影响太大了 ;) ）</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1265414397820534805)** (33 条消息🔥): 

> - `Adapter Fine-Tuning`
> - `Llama-3.1 Compatibility`
> - `CUDA Errors`
> - `H100 Configurations` 


- **关于 Adapter Fine-Tuning 阶段的讨论**：成员们讨论了实现多阶段 Adapter Fine-Tuning 的潜力，考虑使用前一阶段的结果初始化后续阶段，例如使用 SFT 权重进行 DPO 训练。
   - 在 [GitHub](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) 上发现了一个相关的 Feature Request，并提出了通过少量代码修改来促进该方法的建议。
- **Llama-3.1 Fine-Tuning 难题**：几位用户报告了在 Fine-Tuning **Llama-3.1-8b** 时遇到的错误，涉及 CUDA 检查失败，并建议使用来自 **Hugging Face** 的**官方权重**。
   - 一位成员确认成功 Fine-Tuning 了 **12b** 模型，而另一位成员发现更新 **transformers** 解决了 Llama 3.1 的问题。
- **关于 CUDA 检查实现错误的见解**：一位用户询问了训练过程中遇到的特定 CUDA 错误，引发了关于 CUDA 安装可能损坏的讨论。
   - 其他成员建议重新安装相关库，并分享了他们的配置作为可能的解决方案。
- **寻求 H100 配置参考**：一位成员询问了适用于单台 **8xH100** 设备进行 Fine-Tuning 的已知 **Axolotl** 配置。
   - 该请求突显了社区对针对特定硬件部署量身定制的有效模型配置的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1095)">Issues · axolotl-ai-cloud/axolotl</a>: 尽管向 axolotl 提问。通过在 GitHub 上创建账户，为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://github.com/axolotl-ai-cloud">Axolotl AI</a>: Axolotl AI 有 4 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1679">Adopt qlora-pipe approaches · Issue #1679 · axolotl-ai-cloud/axolotl</a>: ⚠️ 请检查此 Feature Request 此前是否已被提出。我搜索了 Discussions 中之前的 Ideas，未发现类似的 Feature Request。我搜索了之前的 Issues...</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1689)">Issues · axolotl-ai-cloud/axolotl</a>: 尽管向 axolotl 提问。通过在 GitHub 上创建账户，为 axolotl-ai-cloud/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1265647380729172072)** (1 条消息): 

> - `Request for Help`
> - `Experience Sharing` 


- **寻求相关经验以获得协助**：一位成员就频道中分享的特定主题链接寻求帮助。
   - 他们呼吁任何有相关经验的人站出来协助咨询。
- **频道内的公开支持请求**：同一位成员强调了社区内针对所链接问题的集体知识需求。
   - 他们重申，经验丰富的人士提供的任何意见对于解决他们的疑问都将是无价的。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1265401315215937622)** (69 条消息🔥🔥): 

> - `GPT-4o mini 更新`
> - `Mistral Large 2 详情`
> - `OpenAI 的财务挑战`
> - `AI 许可与使用`
> - `新的 RLHF 讨论` 


- **GPT-4o mini 在 Chatbot Arena 占据主导地位**：凭借超过 4,000 张用户投票，**GPT-4o mini** 目前在 Chatbot Arena 排行榜上并列第一，性能超越前代版本，且价格便宜 **20 倍**。
   - 随着开发者庆祝这一里程碑，兴奋之情溢于言表，他们注意到新应用的**智能成本**在持续下降。
- **Mistral Large 2：新的竞争者**：Mistral Large 2 拥有 **128k 上下文窗口**，支持数十种语言，使其成为处理高复杂度任务的顶级模型，旨在根据其特定许可用于商业和研究用途。
   - 围绕**许可条件**展开了讨论，由于用户正在寻求该技术的实际应用，商业用途方面的清晰度尚显不足。
- **OpenAI 50 亿美元亏损预测**：最近的估计表明，OpenAI 今年可能面临高达 **50 亿美元** 的惊人亏损，主要原因是 Azure 账单和训练费用。
   - 在讨论 API 收入远低于预期时，人们对可持续性和盈利能力提出了担忧。
- **聊天机器人许可与法律挑战**：有人质疑 **EU AI Act** 是否影响了 Mistral 的许可方式，并推测与法律合规相关的潜在商业使用限制。
   - 对话强调了对于新兴模型商业应用需要更清晰的文档和指导。
- **RLHF 方法论的转变**：对话指出 **Llama 3** 标志着从传统 RLHF 方法的重大转变，这对基于外包商的数据标注有效性产生了影响。
   - 人们对未来探索新 RLHF 策略以及可能存在的支持演进方法论的 **data foundries** 的帖子充满期待。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/technology/#pricing">Technology</a>: 掌控前沿 AI</li><li><a href="https://x.com/lmsysorg/status/1815855136318840970?s=46">lmsys.org (@lmsysorg) 的推文</a>: 令人兴奋的 Chatbot Arena 更新——GPT-4o mini 的结果出来了！凭借 4K+ 用户投票，GPT-4o mini 攀升至排行榜榜首，目前与 GPT-4o 并列第一，且价格便宜 20 倍！显著地...</li><li><a href="https://fxtwitter.com/paulgauthier/status/1816018141878620414">Paul Gauthier (@paulgauthier) 的推文</a>: DeepSeek Coder V2 0724 在 aider 排行榜上排名第二！与之前的版本不同，它可以通过 SEARCH/REPLACE 高效地编辑代码。这开启了编辑大文件的能力。Coder (75%) 接近...</li><li><a href="https://github.com/openai/safety-rbr-code-and-data">GitHub - openai/safety-rbr-code-and-data: 论文《Rule Based Rewards for Language Model Safety》的代码和示例数据</a>: Code and example data for the paper: Rule Based Rewards for Language Model Safety - openai/safety-rbr-code-and-data</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: 今天，我们发布了 Mistral Large 2，这是我们旗舰模型的新一代。与前代相比，Mistral Large 2 在代码生成、数学和推理方面能力显著提升...</li><li><a href="https://x.com/lmsysorg/status/1816010015494529540">lmsys.org (@lmsysorg) 的推文</a>: 人们一直在问为什么 GPT-4o mini 在 Arena 上排名这么高！我们非常感谢所有的反馈。有几点需要注意：1. Chatbot Arena 衡量不同领域的人类偏好。我们鼓励...</li><li><a href="https://fxtwitter.com/moyix/status/1815840634013639086?s=46">Brendan Dolan-Gavitt (@moyix) 的推文</a>: 抱歉，OpenAI 现在在做什么？！微调 gpt-4o-mini 每天前 200 万 token 免费？？</li><li><a href="https://x.com/btibor91/status/1816142224138158365?s=46">Tibor Blaho (@btibor91) 的推文</a>: 引用 aaron holmes (@aaronpholmes) 的消息：OpenAI 今年有望亏损 50 亿美元，这是我们根据内部财务数据和消息来源估计的。OpenAI 预计将在 Azure 上花费约 40 亿美元...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1265406872761143356)** (8 条消息🔥): 

> - `词表大小对推理的影响`
> - `Byte Pair Encoding 与 Tokenization`
> - `模型大小与词表的关系`
> - `词表扩充的权衡` 


- **更大的词表可能会减慢推理速度**：有观点认为较大的词表大小可能会*减慢*推理速度，这挑战了“大词表能减少**常用句子**所需前向传播（forward passes）次数”的普遍认知。
   - 一位成员质疑这一假设是否成立，并指出上下文可能很重要，特别是对于**小模型**而言，词表的增加对参数量的影响更大。
- **词表大小设定逻辑中的权衡**：讨论围绕着以下观点展开：较小的词表会压缩序列，而较大的词表可能会增加 Token 数量，这可能会使推理时间变得复杂。
   - 成员们辩论了针对高频短语使用更少 Token 的优势，与保留大词表可能遗漏的更细粒度交互之间的利弊。
- **词表研究的复杂性**：一位成员指出，进行彻底的实验来测试词表对不同模型的影响可能成本很高且适用范围较窄。
   - 他们指出，研究结果可能无法很好地泛化，并强调在对模型能力做出广泛断言时需要谨慎。
- **Byte Pair Encoding 在词表构建中的作用**：一位参与者强调了 **Byte Pair Encoding** (BPE) 如何构建词表：首先为单个单词创建 Token，然后在上下文允许时将它们合并为更大的 Token。
   - 这一过程引发了关于“使用多个 Token 而不是单个复合 Token 是否能增强序列理解和 Attention 指标”的讨论。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1265555815918469140)** (4 条消息): 

> - `IBM 的策略`
> - `魔力象限 (Magic Quadrant)` 


- **IBM 关注点的转移**：一位成员指出，之前的内容已被替换为一个突出**热门供应商**的页面，这引发了对 *IBM 现在在做什么* 的好奇。
   - 这种转变引发了关于 **IBM 在当前技术格局中的策略** 的疑问。
- **关于魔力象限 (Magic Quadrant) 的见解**：一位成员提到了对**魔力象限**的潜在影响，重点关注**执行能力 (Ability to Execute)** 和**愿景完整性 (Completeness of Vision)** 等因素。
   - 这表明了科技行业内持续的竞争和战略定位。
- **关于 AI 和 Midjourney 的讨论**：分享了《纽约时报》一篇题为《致 Midjourney 的一封信》的相关文章，讨论了 AI 的当前趋势。
   - 该文章可能会提供有关公众认知和 AI 技术演变角色的见解。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1265562937632882770)** (11 条消息🔥): 

> - `CrowdStrike 停机道歉`
> - `预训练数据基准测试`
> - `数据中心吞吐量问题` 


- **CrowdStrike 为停机提供 10 美元道歉礼品卡**：据多方报道，在一次由错误更新引起的大规模停机后，**CrowdStrike** 正向合作伙伴提供一张 **10 美元的 Uber Eats 礼品卡**作为道歉。
   - 然而，一些接收者发现，在尝试兑换礼品卡时收到了错误提示，显示该代金券已**取消**。
- **为无基准测试的数据支付奖金**：一位成员强调，如果模型中使用的预训练数据不包含任何基准测试（benchmarks），**他们真的会给员工发放奖金**。
   - 这在 Twitter 上引发了讨论，许多人从论文中发现了其他有趣的细节，这些细节可能值得进一步研究。
- **数据中心微气候影响吞吐量**：参与者讨论了论文中的一个注释，该注释指出由于与**数据中心微气候**相关的问题，中午的**吞吐量下降了 2%**。
   - 这一细节被认为非常重要，展示了微小的环境因素如何影响性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/theemozilla/status/1815989758360744085?s=46">来自 emozilla (@theemozilla) 的推文</a>：就像如果预训练数据不包含任何基准测试，他们真的会给人们发奖金一样</li><li><a href="https://techcrunch.com/2024/07/24/crowdstrike-offers-a-10-apology-gift-card-to-say-sorry-for-outage/">CrowdStrike 提供 10 美元道歉礼品卡以示歉意 | TechCrunch</a>：几位收到 CrowdStrike 优惠的人发现礼品卡无法使用，而其他人则收到错误提示称代金券已被取消。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1265397012917129349)** (3 条消息): 

> - `马克·扎克伯格的 AI 时代`
> - `蜗牛表情符号热潮` 


- **走进马克·扎克伯格的 AI 时代**：一段名为 [Inside Mark Zuckerberg's AI Era | The Circuit](https://www.youtube.com/watch?v=YuIc4mq7zMU) 的共享 **YouTube 视频** 讨论了 AI 战争中开源与闭源模型之间的最新对决，并强调了马克·扎克伯格在最前线扮演的角色。
   - 视频描述指出，它深入洞察了 Meta 在持续的 AI 发展浪潮中的品牌重塑和战略方向。
- **社区庆祝这只谦逊的蜗牛**：一位成员表达了对蜗牛的热爱，并分享了一个描绘该生物的友好表情符号。
   - *We love snail*（我们爱蜗牛）这种热情的态度捕捉到了成员们对这一独特形象的喜爱。



**提到的链接**：<a href="https://www.youtube.com/watch?v=YuIc4mq7zMU">Inside Mark Zuckerberg&#39;s AI Era | The Circuit</a>：如果 AI 战争的最新战场是在开源和闭源模型之间，那么 Meta 的首席执行官兼创始人马克·扎克伯格就正处于最前线。自从更名为 M...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1265730238822875270)** (5 条消息): 

> - `Llama 3 发布`
> - `RLHF 在能力提升中的作用`
> - `用于对齐的合成数据` 


- **Llama 3 正式发布**：今天，Meta [正式发布](https://llama.meta.com/)了**迄今为止最大且最强大的开源模型 Llama3-405B**，该模型在 **15T tokens** 上进行了训练，并在**所有主要基准测试中击败了 GPT-4**。
   - 该模型是一个稠密 Transformer，标志着开源 AI 能力的显著进步。
- **RLHF 引导训练后的能力涌现**：Llama 的一位对齐负责人表示，**100% RLHF** 是训练后能力涌现的原因，强调了该方法的重要性。
   - 这一表态引发了关于模型训练中有效对齐方法的讨论。
- **讨论合成数据在对齐中的作用**：分享了一份关于利用**合成数据进行对齐**的精彩概述，阐明了其潜在益处。
   - 讨论强调了利用合成数据改进 AI 对齐策略日益增长的兴趣。
- **加入紧急 LLM 论文俱乐部**：邀请成员加入 [紧急 LLM 论文俱乐部](https://x.com/latentspacepod/status/1816151808357908698)，深入讨论 Llama 3 论文。
   - 这一倡议反映了社区在分析重要 AI 文献方面的协作努力。
- **AI in Action 俱乐部特邀 Cursor 联合创始人**：为了持续参与，鼓励成员参加 [AI in Action 俱乐部](https://lu.ma/tnmx3pvp)，本次活动重点介绍 Cursor 联合创始人及其最新的编程 Agent —— Composer。
   - 这突显了社区致力于紧跟创新 AI 工具步伐的承诺。



**提到的链接**：<a href="https://www.latent.space/p/llama-3">Llama 2, 3 &amp; 4: Synthetic Data, RLHF, Agents on the path to Open Source AGI</a>：Meta/FAIR 的 Llama 2 负责人及 Llama 3 训练后负责人 Thomas Scialom 谈论 Chinchilla 陷阱、为什么合成数据和 RLHF 有效，以及 Llama 4 对 Agent 的关注将如何引领我们走向开源 AGI...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1265421370112606269)** (4 条消息): 

> - `SnailBot 新闻` 


- **SnailBot 新闻发布**：向标签 <@&1216534966205284433> 发送了关于 **SnailBot News** 的通知。
   - 根据最近的讨论，预计会有*有趣的*更新。
- **45 分钟的时间参考**：一位成员提到了一个与 **45 分钟** 相关的**有趣**观察。
   - 在当前的讨论中，这一时间段的具体背景尚未明确。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1265388704827179039)** (12 条消息🔥): 

> - `MAX 和 Mojo 编译器版本管理`
> - `Nightly 编译器发布`
> - `版本管理中的困惑`
> - `基于功能与基于日历的发布` 


- **MAX 和 Mojo 编译器版本管理困境**：讨论围绕下一个主编译器版本是 **24.5** 还是 **24.8** 展开，考虑到功能/稳定版与 Nightly 版本遵循不同的发布原则。
   - 重点关注了 Nightly 版本与主版本之间未来可能出现的脱节，特别是针对 **2025** 年等未来日期。
- **Nightly 发布遵循日历系统**：澄清了 **Nightly 发布**是基于日历模型的，而主版本发布则是由 **Marketing** 因素驱动，并非严格按日期。
   - 一位成员指出，版本号的偶然一致可能会导致混淆，并举例说明了自己在讨论中混淆版本号的经历。
- **社区对 ML 复杂性的探索**：一位参与者提到深入研究 **Machine Learning** 主题，并将其描述为“一团糟”，分享了对近期遇到的复杂性的惊叹。
   - 这一评论强调了在社区中进行 ML 讨论所面临的持续挑战，以及可能出现的各种困惑。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1265384971636576420)** (17 条消息🔥): 

> - `v24.5 发布推测`
> - `在 Mojo 中使用 SDL`
> - `关于 Var 和 Let 的讨论`
> - `生成艺术 vs AI`
> - `Mojo 中的 Regex 库` 


- **关于 v24.5 发布日期的推测**：目前关于 GPU 功能的讨论不断，导致人们猜测 **v24.5** 的发布可能需要一些时间，因为团队正在稳定其功能。
   - *关于为什么版本管理系统遵循每年递增的方式存在一些争议*。
- **在 Mojo 中使用 SDL 的兴趣**：一位用户询问了关于学习 **SDL** 与 **Mojo** 集成的资源，表达了希望更好地理解该过程的愿望。
   - 与此相关，人们对如何在 SDL 环境下利用 **DLHandle** 感到好奇。
- **关于 Var 和 Let 使用的辩论**：一位用户质疑如果所有内容都声明为 **var**，那么使用 **var** 的必要性，认为这可能是多余的。
   - 作为回应，另一位成员指出 var 对编译器有益，而 **let** 主要服务于那些偏好 *Immutability* 的人。
- **生成艺术与 AI 的性能对比**：一位用户提到他们的电脑创作了一些“艺术”，并表示*它不如 Gen AI 那么好*。
   - 另一位用户建议，比较时应考虑所消耗的计算能力。
- **关于 Regex 库可用性的查询**：一位用户询问 **Mojo** 中是否存在 **Regex 库**，突显了对处理正则表达式的兴趣。
   - 对话并未对该查询提供明确的答案。


  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1265449434188222474)** (54 条消息🔥): 

> - `Mojo 更新`
> - `Git 指令`
> - `DTypePointer 移除`
> - `SIMD 比较`
> - `为 Mojo 贡献` 


- **发布了重大的 Mojo 更新**：发布了新的 nightly Mojo 编译器，更新至 `2024.7.2405`，显著变化包括移除了 `DTypePointer` 以及新增了字符串格式化方法。
   - 完整的变更日志可以在 [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 找到。
- **Git Rebase 的挑战**：几位成员讨论了 Git Rebase 的挑战，并在遵循贡献指南时遇到了未解决的合并冲突等问题。
   - 一位成员表示，由于这些工具链问题，感到自己在贡献代码方面的能力受到了限制。
- **DTypePointer 对 Mojo 项目的影响**：从 Mojo 中移除 `DTypePointer` 要求项目更新其代码，过渡到使用 `UnsafePointer`。
   - 成员们呼吁提供明确的指南来协助开发者完成这一过渡，特别是针对现有 Mojo 项目中普遍存在的使用场景。
- **SIMD 类型的可比性**：围绕为 SIMD 类型建立全序关系的挑战展开了讨论，强调了泛型编程与特定比较之间的冲突。
   - 有建议提出，引入 `SimdMask[N]` 类型可能有助于弥合架构相关行为与编程预期之间的差距。
- **对 Mojo 编译器特性的贡献**：贡献者表示希望通过改进泛型编程和迭代器实现来简化 Mojo 库，同时解决当前的编译器问题。
   - 目前正在努力简化 API，特别是关于排序和类型处理相关的重载。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/parameters/#:~:text=Parameter%20inference%E2%80%8B&text=Mojo%20can%20also%20infer%20the,a%20constructor%20or%20static%20method.&text=Note%20that%20you%20can%20create,it%20from%20the%20value%20argument.">Parameterization: compile-time metaprogramming | Modular Docs</a>：参数和编译时元编程的介绍。</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` doesn&#39;t work at compile time. · Issue #3126 · modularml/mojo</a>：Bug 描述如标题。至少 List.__getitem__ 无法工作。复现步骤：fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # 打印 0。系统信息：Mojo 2024.6.2614 (366c690a) o...</li><li><a href="https://docs.modular.com/mojo/manual/parameters/#:~:text=Parameter%20inference%E2%80%8B&text=Mojo%20">Parameterization: compile-time metaprogramming | Modular Docs</a>：参数和编译时元编程的介绍。
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1265384432429170783)** (57 条消息🔥🔥): 

> - `Factorio 自动化 Mod`
> - `GPT-4o Mini 微调`
> - `Mistral Large 2 发布`
> - `Reddit 内容政策争议`
> - `Arxiv2Video 生成器` 


- **Factorio 自动化 Mod 发布**：发布了一个名为 [factorio-automation-v1](https://github.com/naklecha/factorio-automation) 的新 Mod，允许 Agent 执行各种游戏动作，如合成和采矿。
   - 它为 Agent 在游戏内测试其能力提供了一个极佳的游乐场。
- **GPT-4o Mini 微调上线**：OpenAI 已上线 **GPT-4o mini** 的微调功能，目前对 tier 4 和 5 用户开放，9 月 23 日前每日首个 2M 训练 Token 免费。
   - 成员们讨论了将微调后的 **GPT-4o mini** 与 **Llama-3.1-8b** 进行对比的评估结果，并注意到一些性能上的不一致。
- **Mistral Large 2 的亮眼特性**：[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 已揭晓，拥有 **1230 亿参数**，支持多种语言并具备强大的代码能力。
   - 它为非商业用途提供开放权重，并专为长上下文应用而设计。
- **Reddit 的内容政策引发关注**：围绕 Reddit 的公共内容政策展开了讨论，成员们对 Reddit 对用户生成内容的控制表示担忧。
   - 许多人认为用户应该对自己的内容拥有选择权，并认为该政策的模糊性引发了重大问题。
- **Arxiv2Video 生成器展示**：一个开源的 **Arxiv2Video 生成器** 亮相，并为 **Herd of Llamas Paper Club** 创建了演示。
   - 该工具由 @aditya_advani 展示，可以生成引人入胜的学术论文视频摘要，并欢迎进一步的关注和潜在合作。


<div class="linksMentioned">

_（链接已在正文中包含）_

</div>

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1816161458629271673">来自 Alex Albert (@alexalbert__) 的推文</a>：我们收到了来自 @AnthropicAI 开发者为 Build with Claude 2024 年 6 月竞赛提交的许多优秀作品！以下是 3 个获胜项目，每个项目将获得 1 万美元的 Anthropic API 额度：</li><li><a href="https://x.com/alexalbert__/status/1816161464320942279">来自 Alex Albert (@alexalbert__) 的推文</a>：由 @baygross 和 @MatthewSlotkin 开发的行内文档编辑器。这是一个由 Claude 3.5 Sonnet 驱动的工具，可以阅读您的文档并在您需要的地方直接插入评论和建议。</li><li><a href="https://x.com/aaronpholmes/status/1816102562031927298">来自 aaron holmes (@aaronpholmes) 的推文</a>：新消息：根据内部财务数据和消息来源，我们估计 OpenAI 今年有望亏损 50 亿美元。OpenAI 预计将花费约 40 亿美元用于运行 ChatGPT 和其他推理任务的 Azure 账单...</li><li><a href="https://x.com/paulgauthier/status/1816198047690289518">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Mistral Large 2 (2407) 在 aider 的代码编辑基准测试中仅获得 60% 的分数。这使其仅领先于最好的 GPT-3.5 模型。它似乎无法可靠地使用搜索/替换来高效地...</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>：今天，我们发布了 Mistral Large 2，这是我们新一代的旗舰模型。与前代相比，Mistral Large 2 在代码生成、数学和推理方面能力显著提升...</li><li><a href="https://x.com/openaidevs/status/1815836887631946015?s=46">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>：通过微调（Fine-tuning）为您的应用程序定制 GPT-4o mini。今天已向第 4 层和第 5 层用户开放，我们计划逐步向所有层级扩展访问权限。到 9 月为止，每天前 200 万个训练 Token 免费...</li><li><a href="https://x.com/naklecha/status/1815808346735378487?s=46">来自 naklecha (@naklecha) 的推文</a>：今天，我很高兴发布 factorio-automation-v1。使用这个模组，您的 Agent 可以执行游戏动作，如制作、寻路、采矿、研究等。这个模组可以作为一个很好的游乐场...</li><li><a href="https://x.com/aditya_advani/status/1816187840163987654">来自 Aditya P. Advani (@aditya_advani) 的推文</a>：@latentspacepod @lvdmaaten @swyx @vibhuuuus @picocreator @eugeneyan 秉承快速回顾的精神，我的开源 Arxiv2Paper 生成器 ELDO 为俱乐部制作了这个 2 分钟的视频供大家观看...</li><li><a href="https://x.com/dchaplot/status/1816132981377097883">来自 Devendra Chaplot (@dchaplot) 的推文</a>：非常激动地宣布 Mistral Large 2 - 123B 参数 - 适配单个 H100 节点 - 原生多语言 - 强大的代码和推理能力 - SOTA 函数调用（Function Calling） - 非商业用途开放权重...</li><li><a href="https://x.com/corbtt/status/1815843764960911549">来自 Kyle Corbitt (@corbtt) 的推文</a>：@altryne @eugeneyan 评估运行中</li><li><a href="https://x.com/alexalbert__/status/1816161462248947825">来自 Alex Albert (@alexalbert__) 的推文</a>：Claude + 无限画布，由 @TodePond 开发。一个无限画布 Web 应用，Claude 3.5 Sonnet 在其中生成并解释绘图，结合了文本和视觉提示词（Prompts）。</li><li><a href="https://x.com/corbtt/status/1815891110696477099">来自 Kyle Corbitt (@corbtt) 的推文</a>：好吧，大约有 100 个人私信问我如果与微调后的 4o mini 相比会发生什么。我在下面列出了 4 项任务中 3 项的结果！一些想法：- 微调后，Llama 3.1 8B 仍然胜出...</li><li><a href="https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy">公共内容政策</a>：这是一项关于我们如何处理在 Reddit 上公开的信息的政策。这不是隐私政策。关于我们如何收集、使用和共享您的个人/隐私信息，请咨询我们的隐私政策...</li><li><a href="https://www.reddit.com/r/reddit4researchers/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=41057033">得益于 AI 协议，Google 现在是唯一能在 Reddit 上运行的搜索引擎 | Hacker News</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1265711125941452802)** (1 条消息): 

> - `Llama 3 Paper Club`
> - `Cursor's AI Developer Tools`
> - `Asia LLM Paper Club` 


- **Llama 3 紧急论文俱乐部**：在 2 小时内加入**紧急论文俱乐部**，讨论 @lvdmaaten 等人的 *The Llama 3 Herd of Models*，这是 **POTY Awards**（年度论文奖）的早期有力竞争者。更多详情请见 [Latent Space Pod](https://x.com/latentspacepod/status/1816151808357908698)。
   - 包括 @swyx、@vibhuuuus 和 @eugeneyan 在内的成员将出席，详细探讨这一重要主题。
- **Cursor 联合创始人讨论 AI 工具**：在即将举行的一次会议中，**Cursor** 的联合创始人将进行特别分享，讨论 *Cursor、Composer 和 AI 开发者工具*。这是一个直接从源头获取见解的机会。
   - 虽然尚未提供此次讨论的具体日期和时间，但对于那些对 AI 开发感兴趣的人来说，这将是一个重要的活动。
- **不要错过亚洲 LLM 论文俱乐部**：请务必参加**亚洲 LLM 论文俱乐部**，参与聚焦该领域最新进展和论文的精彩讨论。您可以点击[此处](https://lu.ma/jpyss688)了解有关会议的更多信息。
   - 该俱乐部一直是那些致力于 LLM 研究与合作的人士的关键聚集地。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1816151808357908698">来自 Latent.Space (@latentspacepod) 的推文</a>：🚨 紧急论文俱乐部 @latentspacepod Discord 将在 2 小时内开会，讨论 @lvdmaaten 等人的 The Llama 3 Herd of Models，这是赢得 POTY* Awards 的早期竞争者！加入我们（链接见下文）...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1265397223173394524)** (5 条消息): 

> - `LlamaParse Features`
> - `MongoDB AI Applications Program (MAAP)`
> - `Mistral Large 2 Capabilities`
> - `Structured Extraction in LLMs` 


- **LlamaParse 发布 Markdown 等功能**：在最近的一段视频中，展示了 **LlamaParse** 的功能，包括 **Markdown 和纯文本输出**选项，以及用于更丰富元数据提取的 **JSON 模式**。
   - 该工具还支持**多语言**输出以改进 OCR，使其成为任何工作流的多功能补充。[在此观看视频](https://t.co/RUWJ0Z2NMn)。
- **MongoDB 启动 AI 应用计划**：新宣布的 **MongoDB AI 应用计划 (MAAP)** 旨在协助组织快速构建和部署现代 AI 增强型应用。
   - 它提供参考架构和包含领先技术集成的全面技术栈，使企业能够**加速其 AI 之旅**。[了解更多关于 MAAP 的信息](https://t.co/rCz3DfUe3A)。
- **Mistral Large 2 带来先进的 Function Calling**：**Mistral Large 2** 具备最先进的 **Function Calling 能力**，并提供对结构化输出和 Agent 的首日支持。
   - 此版本与增强的 **Function Calling** 和**结构化输出**保持一致，为用户提供了如 Cookbook 等有用资源以供探索。[查看 Cookbook](https://t.co/ho02wDbGpZ)。
- **LLM 结构化提取功能发布**：最新版本为基于 LLM 的 ETL、RAG 和 Agent 流水线提供了**结构化提取能力**，支持异步和流式处理功能。
   - 通过定义 **Pydantic 对象**并将其附加到 LLM，用户可以显著增强其数据处理工作流。[在此发现更多信息](https://t.co/0wLX2Tf1P6)。



**提到的链接**：<a href="https://t.co/rCz3DfUe3A">MongoDB AI 应用计划</a>：获取加速您的 AI 应用之旅并以信心和速度启动所需的支持。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1265450480474132541)** (52 messages🔥): 

> - `SubQuestionQueryEngine`
> - `Llama 3.1 Testing`
> - `RAG Setup for PDF Display`
> - `Text-to-SQL Pipeline Optimization`
> - `ReAct Agent Behavior` 


- **使用 SubQuestionQueryEngine 的流式响应**：成员们讨论了使用 `SubQuestionQueryEngine.from_defaults` 的方案，目标是实现从 LLM 流式传输最终响应以降低延迟。
   - 分享了一个涉及 `get_response_synthesizer` 并利用 token 打印技术的解决方案，但在实施过程中遇到了挑战。
- **对 Llama 3.1 指标的怀疑**：一些用户对 Meta 提供的 Llama 3.1 指标表示不信任，并讨论了其在 RAG 评估中的可用性。
   - 讨论中提出了关于使用像 `llama3:70b-instruct-q_5` 这样的模型是否对这类任务有益的担忧。
- **通过 PDF 界面优化 RAG**：讨论集中在改进 RAG 设置的策略上，涉及在 Web 界面中通过按钮显示 PDF。
   - 建议包括避免包含大量 PDF 的大型项目，以及使用能直接处理 PDF 文件而无需转换的库。
- **提高 Text-to-SQL 流水线速度**：用户强调了其 Text-to-SQL 流水线响应速度慢的问题，并寻求优化建议。
   - 建议包括使用更快的 LLM 或压缩输入；还探讨了通过流式输出（streaming output）来提升用户体验。
- **ReAct Agent 幻觉**：一位成员报告称，他们的 ReAct Agent 在响应输入时会持续产生幻觉，陷入错误的处理循环。
   - 讨论指向了 LLM 无法遵守预期的输出格式，并建议添加 stop tokens 以改善行为。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/pull/14439#issuecomment-2195513666">Add Context-Only Response Synthesizer by Androbin · Pull Request #14439 · run-llama/llama_index</a>: 描述动机：使用工具的 OpenAIAgent 表现比 ContextChatEngine 差，因为外部 LLM (agent)、内部 LLM (query engine) 和 retriever 实际上在扮演……</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1265511825999532042)** (1 messages): 

> - `RAG pipeline evaluation`
> - `Custom RAG evaluation system`
> - `RAGAS framework`
> - `Improving evaluation methods` 


- **有效评估 RAG 流水线**：一位成员表示在使用了 **RAGAS** 框架后，发现其随机性太强，需要关于评估其 RAG 流水线的**专业建议**。
   - 他们提到目前正在开发一个**类似于 RAGAS 的自定义评估系统**，以便更好地控制指标。
- **寻求 RAG 评估包的建议**：该成员正在寻求对其自定义评估方法的**改进**，并询问其他人是否能推荐更适合其需求的包。
   - 他们对任何关于增强系统或值得考虑的替代方案的**建议**表示感谢。


  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1265473423707082764)** (34 条消息🔥): 

> - `Cohere Dashboard 问题`
> - `模型测试好评`
> - `服务器性能担忧`
> - `工具功能建议`
> - `社区自我介绍` 


- **Cohere Dashboard 重载问题**：一位成员提到他们的 **Cohere account dashboard** 似乎在不断重载，而另一位成员确认他们那边正常。
   - 这引发了关于潜在故障和速率限制（rate limiting）的简短讨论。
- **对 Command R Plus 的赞赏**：随着 **Llama 3.1** 等备受期待的新模型发布，一位成员表达了对 **Command R Plus** 日益增长的喜爱。
   - 另一位用户提到正在创建一个用于 **model comparisons**（模型对比）的 Playground，以进一步探索这种感受。
- **服务器性能查询**：有人担心服务器可能宕机，但其他人确认其处于 **full operational status**（完全运行状态）。
   - 建议包括检查可能影响用户体验的速率限制。
- **Cohere Tools 的功能建议**：一位成员提议在 **Cohere** 对话过程中能够中途使用工具，例如根据请求调用网页搜索。
   - 在最初的一些困惑之后，大家确认其中一些功能已经存在。
- **社区自我介绍**：新成员介绍了自己，讨论了他们在 **NLP 和 NeuroAI** 领域的背景，并表达了对该服务器的兴奋之情。
   - 关于 **Command-R+** 使用体验的讨论强调了其与 **NovelAI** 等其他模型相比的影响力。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1265409293314953239)** (6 条消息): 

> - `zenbase/core 发布`
> - `DSPy 优化器` 


- **Zenbase/Core Python 库发布**：一位成员宣布 **zenbase/core** 现已发布，允许用户在现有的 Python 代码（如 Instructor 和 LangSmith）中使用 **DSPy’s optimizers**。
   - 他们请求通过在其 [Twitter 帖子](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ)上进行**转发、点赞和加星**来提供支持。
- **成员在 Twitter 上的互动**：另一位成员热情回应，确认他们已经**点赞并转发**了发布公告，并对这个新库表示期待。
   - 频道内的整体情绪反映了对**近期发布**和开发的积极反应。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 条消息): 

batmanosama: done
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1265406190473445567)** (20 条消息🔥): 

> - `DSPy 中的 Typed Predictors`
> - `内部步骤执行可见性`
> - `小型语言模型的未来`
> - `为 DSPy 仓库做贡献`
> - `模型微调与蒸馏` 


- **Typed Predictors 导致输出问题**：一位用户在使用 DSPy 的 typed predictors 时遇到麻烦，尽管遵循了设置技巧，但无法返回正确结构化的输出。
   - 另一位用户建议使用 `dspy.configure(experimental=True)` 配置实验性功能，以寻求潜在的改进。
- **检查内部程序执行**：用户讨论了在程序执行期间查看内部步骤和 Prompt 的各种方法，并提出了 `inspect_history` 等建议。
   - 一位用户表示需要更多地了解模型输出，即使是在类型检查失败期间。
- **倡导小型语言模型 (SLM)**：一位成员分享了一篇文章，宣传可以在极低硬件上运行的小型语言模型的效率和优势。
   - 他们强调了 **privacy**（隐私）优势以及小型模型在保持有用智能的同时对边缘设备的适用性。
- **贡献 DSPy 示例的机会**：另一位用户询问如何为 DSPy 仓库贡献示例，表示已准备好创建适合初学者的内容。
   - 回复确认了对多样化示例的需求，贡献可以添加到 `/examples` 目录中。
- **关于使用 DSPy 进行模型微调的问题**：一位用户询问是否可以使用 DSPy 对 Llama 8B 等模型进行微调和蒸馏，而无需额外的神经网络代码。
   - 他们的好奇心凸显了理解 DSPy 在模型训练技术方面能力的重要性。



**提到的链接**：<a href="https://medium.com/thoughts-on-machine-learning/small-language-models-are-the-future-6e8909567198">小型语言模型是未来</a>：我的论点：小型语言模型 (SLM)——体积小到可以在仅有 4GB RAM 的计算机上运行的模型——才是未来。SLMs…

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1265454212305784852)** (4 条消息): 

> - `学习 Tinygrad`
> - `GPU 和 Uops 问题`
> - `OpenCL 和 Python 挑战`
> - `查看已关闭的 PR` 


- **学习 Tinygrad 的基础知识**：一位成员提到，学习 **Tinygrad** 和 **transformers** 仍在他的计划清单上。
   - 他补充道：*这是一个循序渐进的过程*，表达了希望逐步深入理解的意愿。
- **GPU 和 Uops 仍是关注点**：一位成员正努力让 GPU 和 Uops 变**绿**（通过测试），虽然他已经成功使用了 numpy 和 pytorch 的 shapes。
   - 他正在寻求修复 **OpenCL** 和 **Python device** 问题的提示，并表示：*我想应该是全绿才对*。
- **建议查看已关闭的 PR**：一位用户建议，查看**已关闭的 PR** 可以为解决当前问题提供思路。
   - 该用户旨在帮助澄清在取得进展过程中的任何不确定性。
- **理解 OpenCL 和 Python 的局限性**：另一位成员指出，**OpenCL** 和 **Python** 失败是因为它们无法利用 views，这使问题变得复杂。
   - 他们指出：*简单的 'bitcast' 无法处理改变 shape 的 bitcasts*，并提醒可以通过 **DEBUG=x** 检查具体细节。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1265431881017589781)** (19 条消息🔥): 

> - `tinygrad 中的分子动力学引擎`
> - `自定义运行时实现`
> - `神经网络势能`
> - `Beautiful CartPole 中的 PPO 算法` 


- **实现分子动力学引擎**：一名成员正与团队合作在 tinygrad 中实现一个**分子动力学引擎 (Molecular Dynamics engine)**，使用神经网络来预测构型的能量，但在梯度利用方面遇到了问题。
   - 另一名成员建议使用输入梯度跟踪并修改权重更新，以避免在反向传播 (backpropagation) 过程中遇到的问题。
- **tinygrad 自定义运行时指南**：一位用户分享了如何在 tinygrad 中实现**自定义运行时 (custom runtime)** 的指南，强调添加对新硬件的支持应该是很简单的。
   - 他们要求澄清诸如 `global_size` 和 `local_size` 等技术术语，这些术语被解释为操作上下文中内核 (kernel) 执行计数的参数。
- **理解神经网络势能**：讨论揭示了分子动力学引擎中使用的能量是基于**神经网络势能 (Neural Network Potentials, NNP)** 的，并强调了高效计算的必要性。
   - 对话中包含了关于如何优化反向传播过程以提高训练结果的建议。
- **Beautiful CartPole 中的 PPO 算法**：一名成员询问了 Beautiful CartPole 环境的 **PPO** 实现中 `.sum(-1)` 操作的作用，并指向了代码中的特定行。
   - 这体现了社区成员在理解强化学习 (reinforcement learning) 实现细节方面的协作性质。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.tinygrad.org/runtime/overview/">Runtime Overview - tinygrad docs</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/tinygrad/codegen/lowerer.py#L155-L156">tinygrad/tinygrad/codegen/lowerer.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://gist.github.com/python273/0dc136fbc63559188ab279c07329e891">TinyJit vis WIP</a>: TinyJit 可视化进行中 (WIP)。GitHub Gist：即时分享代码、笔记和摘要。</li><li><a href="https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py#L231">spinningup/spinup/algos/pytorch/ppo/ppo.py at 20921137141b154454c0a2698709d9f9a0302101 · openai/spinningup</a>: 帮助任何人学习深度强化学习的教育资源。 - openai/spinningup</li><li><a href="https://mesozoic-egg.github.io/tinygrad-notes/addingaccelerator.html">How to add a custom accelerator?</a>: tinygrad 教程</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_python.py#L31">tinygrad/tinygrad/runtime/ops_python.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/examples/whisper.py#L119">tinygrad/examples/whisper.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/examples/whisper.py#L41-L45">tinygrad/examples/whisper.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1265411491192897536)** (15 messages🔥): 

> - `3.1 release interviews`
> - `MPS support PR`
> - `LoRA issues`
> - `Conflicts in contributions`
> - `Git workflow optimizations` 


- **3.1 版本倒计时与精彩访谈**：成员们询问是否会随 3.1 版本发布一些类似 Llama3 的精彩[访谈](https://github.com/pytorch/torchtune/pull/790)。
   - 这引发了人们对新版本发布时可能伴随的深入见解和讨论的兴趣。
- **MPS 支持 PR 引起关注**：重点介绍了一个新的 Pull Request ([#790](https://github.com/pytorch/torchtune/pull/790))，该 PR 为本地 Mac 电脑增加了 MPS 支持，并检查了 BF16 的兼容性。
   - 上下文表明，该 PR 可能会解决那些使用 MPS 设备的用户所面临的主要测试障碍。
- **LoRA 功能问题依然存在**：讨论了围绕 **LoRA** 功能的问题，指出它在之前的尝试中无法工作，且之前受到硬编码 **CUDA** 路径的影响。
   - 成员们就遇到的具体错误交换了意见，强调了实现过程中持续存在的挑战。
- **与 Git 冲突的无休止战斗**：成员们对贡献中频繁出现的新冲突表示沮丧，认为这像是一个无止境的循环，尤其是在修复了现有冲突之后。
   - 有人建议新冲突可能源于工作流，表明可能需要进行调整。
- **优化 Git 工作流以减少冲突**：讨论了改进 Git 工作流以尽量减少新冲突不断出现的问题，强调了协作的重要性。
   - 建议改进贡献实践可能有助于减轻合并挑战的负担。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>：上下文：出于测试目的，直接在本地 Mac 电脑上运行非常有用。变更日志：检查 MPS 设备对 BF16 的支持。添加了针对 MPS 的配置，路径更改为...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1265412754194301110)** (1 messages): 

> - `Pad ID Bug`
> - `Pull Request #1211` 


- **修复 Pad ID Bug**：一位成员指出 **pad id** 不应出现在 generate 功能中，并将其确定为一个重要的 Bug。
   - 作为回应，创建了一个 Pull Request 以防止 **pad ids** 和特殊 Token 显示，详见 [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211)。
- **Pull Request #1211 详情**：[Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) 旨在通过修改 **utils.generate** 中的实现来解决关于 pad id 的问题。
   - 该 PR 的上下文提到它是为了修复一个 Bug，确保 pad ids 被隐式地正确处理。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/pull/1211">Prevent pad ids, special tokens displaying in generate by RdoubleA · Pull Request #1211 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。在 utils.generate 中，Pad ID 被隐式假设为 0，...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1265526704722677823)** (6 messages): 

> - `Hugging Face Agents`
> - `Job Opportunities in Python`
> - `HNSW IVFFLAT Index Issues`
> - `SQLite Server Storage Management` 


- **探索使用 Hugging Face 模型的 Agents**：一位用户询问是否有人有使用 **Hugging Face 模型（LLM 或 Chat）构建 Agents** 的经验。另一位用户回答说，他们在 **OpenAI** 和 **Azure** 以及通过 **Ollama** 运行的本地 LLM 上做了很多关于 Agents 的工作。
- **Python 开发者的求职**：一位成员表达了对工作的需求，称：**“有人想雇佣我吗？我需要付账单。”** 他们强调了自己在 **Python** 方面的熟练程度。
- **Aurora 中 HNSW 和 IVFFLAT 索引的挑战**：一位成员报告了在 **Aurora PGVECTOR** 上创建 **3072 维度** 的 **HNSW** 或 **IVFFLAT** 索引时遇到的困难。他们随后分享了解决方案，即使用 **halfvec**。
- **管理 SQLite 服务器线程**：一位用户询问如何检查他们的 **SQLite 服务器存储**，以监控消息和线程的使用情况。他们对在使用 **Langgraph** 时如何删除以前的线程感到好奇。

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1265624679138328607)** (1 messages): 

> - `Scaling LangServe`
> - `OSError 处理`
> - `处理并发请求` 


- **扩展 LangServe 触及打开文件限制**：用户在 LangServe 应用接收约 **1000 个并发请求**时遇到了 **OSError: [Errno 24] Too many open files**。
   - 他们在 [GitHub 上分享了一个 issue](https://github.com/langchain-ai/langserve/issues/714) 来反映该问题，并寻求关于如何有效管理请求负载的建议。
- **寻求高请求负载的解决方案**：用户正在寻找在 LangServe 应用中有效处理**高并发请求**的策略。
   - 他们希望找到在扩展应用规模时防止系统资源限制相关错误的方法。



**提及的链接**：<a href="https://github.com/langchain-ai/langserve/issues/714">Scaling to production -&gt; OSError: [Errno 24] Too many open files socket.accept() out of system resource  · Issue #714 · langchain-ai/langserve</a>：问题描述：当我的 LangServe 应用接收约 1000 个并发请求时，会因错误崩溃：OSError: [Errno 24] Too many open files socket.accept() out of system resource。临时修复方案：我已经检查了...

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1265418002770165790)** (2 messages): 

> - `使用 Ollama 进行完全本地的 Tool Calling`
> - `AI 代码审查器` 


- **请求完全本地 Tool Calling 的 Notebook**：一名成员请求获取今天早些时候演示的 **“使用 Ollama 进行完全本地的 Tool Calling”** 环节的 Notebook。
   - 他们对内容表示赞赏，并强调了其卓越性。
- **介绍 AI 代码审查器工具**：一名成员分享了一个名为 [AI Code Reviewer Ft. Ollama & Langchain](https://youtu.be/g_VRsjpC4e8) 的 YouTube 视频，介绍了一个旨在增强代码审查流程的 CLI 工具。
   - 该视频重点介绍了由 **LangChain** 驱动的功能，并展示了它如何彻底改变代码评估方式。



**提及的链接**：<a href="https://youtu.be/g_VRsjpC4e8">AI Code Reviewer Ft. Ollama &amp; Langchain</a>：欢迎来到 Typescriptic！在本视频中，我们介绍了我们的代码审查器，这是一个旨在彻底改变代码审查方式的 CLI 工具。由 LangChain 驱动...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587)** (6 messages): 

> - `Llama 3.1 405 B`
> - `Mistral Large 2`
> - `API 使用`
> - `开发者机会`
> - `对 LM Studio 的期待` 


- **Llama 3.1 405 B 的易用性令人印象深刻**：**Llama 3.1 405 B** 与 OpenInterpreter 配合使用时开箱即用，表现出色，提供了轻松的使用体验。
   - 相比之下，**gpt-4o** 需要不断提醒其能力，这使得 405b 成为多任务处理的更优选择。
- **使用 Nvidia 的 API 具有成本效益**：一位用户分享说，**Nvidia** 在注册时提供 **1000 积分**，其中 1 积分等于 1 次 API 调用。
   - 这种激励措施为尝试 API 提供了更多的可访问性。
- **Mistral Large 2 与 Llama 3.1 405 B 旗鼓相当**：据报道，**Mistral Large 2** 的表现与 **Llama 3.1 405 B** 相当，尤其是在速度方面表现突出。
   - 更快的性能可能是由于 Mistral 的端点流量比 Llama 的端点流量低。
- **对开发者贡献的兴趣**：有人询问资深开发者是否有机会为一个未指明的项目做出贡献。
   - 这突显了人们对扩大开发者支持和协作的持续兴趣。
- **对集成 LM Studio 的兴奋**：一位用户表达了对在 **LM Studio** 中使用 **Llama 3.1 405 B** 的热情，预示着一个前景广阔的集成。
   - 这表明了人们对通过这种组合增强能力和功能的期待。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1265384282298384515)** (1 messages): 

> - `设备发货时间线` 


- **询问设备发货时间线**：一位用户询问了设备的发货时间线，并直接表达了对更新的期待。
   - 这个问题突显了社区内对交付时间表的持续关注。
- **社区对设备交付的期待**：该询问反映了用户渴望获得设备发货日期更新的普遍情绪，这增强了他们与品牌的联系。
   - 关于发货时间线的讨论已成为一个互动点，展示了社区对该产品的投入。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1265726346093396152)** (2 条消息): 

> - `Llama 3.1`
> - `OpenInterpreter Database Integration`
> - `Database Complexities` 


- **Llama 3.1 免费连接数据库**：[MikeBirdTech](https://x.com/MikeBirdTech/status/1816163862208766137) 指出，**Llama 3.1** 可以通过 **OpenInterpreter** 免费与您的数据库进行交互，并强调了这比付费服务更节省成本。
   - *它也是完全离线和私密的，无需他人查看您的数据*，突显了其**隐私优势**。
- **对使用 Llama 3.1 处理复杂数据库的担忧**：一名成员提出担忧，认为对于涉及跨表连接 (joins) 的**复杂数据库**，该解决方案可能无效。
   - 他们对分享这些信息表示感谢，并评论说尽管存在局限性，但其执行得**非常出色**。



**提到的链接**：<a href="https://x.com/MikeBirdTech/status/1816163862208766137">Mike Bird (@MikeBirdTech) 的推文</a>：Llama 3.1 通过 @OpenInterpreter 免费与您的数据库对话。为什么要为数据库对话服务付费？省钱！它也是完全离线和私密的，无需他人查看您的数据。

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1265423923395301518)** (5 条消息): 

> - `Llama 3.1 release`
> - `LAION metadata download issues`
> - `LAION datasets legality`
> - `YouTube polls` 


- **Llama 3.1：Meta 的开源突破**：Meta 最近推出了 **Llama 3.1 405B**，被誉为有史以来第一个**开源前沿 AI 模型**，在多项基准测试中超越了 GPT-4o 等竞争模型。欲了解更多见解，请观看这段 [YouTube 视频](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61)，其中 Mark Zuckerberg 讨论了其影响。
   - 这一反响突显了该模型对 AI 研究和开源贡献的潜在影响。
- **下载 LAION2B-en 元数据遇到困难**：一名成员表示在 Hugging Face 上查找和下载 **LAION2B-en 元数据**时遇到困难，并询问其他人是否面临同样的问题。回复强调了目前在可访问性方面持续存在的挑战，表明这是一个普遍的困扰。
   - 有人链接到了 [LAION 维护说明](https://laion.ai/notes/laion-maintenance/) 以进一步澄清情况。
- **LAION 数据集处于法律待定状态**：讨论透露 LAION 数据集目前处于**法律待定状态 (legal limbo)**，官方版本的访问受到限制。虽然有替代方案，但建议仅在紧急研究需求时使用非官方数据集。
   - 成员们注意到了 AI 社区围绕数据合法性持续存在的复杂性。
- **YouTube 投票：一场怀旧辩论**：一名成员分享了一个 [YouTube 投票](http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS)，询问哪部 **90 年代电影拥有最佳原声带**，引发了观众的怀旧之情。
   - 这促使成员们回顾他们那个时代最喜欢的原声带，通过共同的文化体验建立联系。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS">来自 Innuendo 的投票</a>：哪部 90 年代电影的原声带最好听？</li><li><a href="https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61">Mark Zuckerberg 谈论 Llama 3.1、开源、AI Agents、安全等</a>：Meta 刚刚发布了 Llama 3.1 405B —— 有史以来第一个开源前沿 AI 模型，在多个基准测试中击败了 GPT-4o 等顶级封闭模型。我坐下来...
</li>
</ul>

</div>
  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 条消息): 

spirit_from_germany: https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61
  

---


### **Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1265490712938545196)** (2 条消息): 

> - `Copyright issues in ML datasets`
> - `Identifying non-distilled data`
> - `Legal considerations` 


- **关于 ML 数据集版权的法律澄清**：一名成员讨论了由 **ML 模型**生成的大部分数据集可能不具备版权，表明它不被视为真正的创意作品。
   - 他们指出，**非 GPT-4** 生成的内容应稳妥地属于 **MIT 许可**，但也承认在持续的法律讨论中这仍是一个灰色地带。
- **关于识别非蒸馏数据的疑问**：随后提出了一个关于如何识别数据集中**非蒸馏 (non-distilled) 行**的问题。
   - 这表明人们对确保数据集内容管理的清晰度和组织性有着持续的兴趣。

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1265715877664264353)** (1 messages): 

> - `Translation model fine-tuning` (翻译模型微调)
> - `CPO approach` (CPO 方法)
> - `ALMA models performance` (ALMA 模型性能)


- **尝试在翻译模型中使用 DPO**：一位成员询问是否有人根据 **CPO** 论文的启发，成功使用 **DPO** 微调了翻译模型。
   - 他们特别提到 **中等规模的 LLM** 如何无法与最先进的模型相媲美，并指向 [CPO 论文](https://arxiv.org/abs/2401.08417) 以获取更多细节。
- **CPO 在提升翻译模型中的作用**：**CPO** 方法旨在解决机器翻译中监督式微调（SFT）的缺点，强调了参考数据质量的问题。
   - 通过训练模型避免仅生成“合格”的翻译，CPO 增强了 **ALMA-R** 等模型的性能，这些模型充分利用了有限的数据集。
- **ALMA-R 的卓越性能**：在将 **CPO** 应用于 ALMA 模型时，尽管仅使用了 **2.2万条平行句子** 和 **12M 参数**，仍观察到了显著的提升。
   - 由此产生的模型 **ALMA-R** 可以与传统的 encoder-decoder 模型竞争，甚至超越它们。



**提到的链接**：<a href="https://arxiv.org/abs/2401.08417">Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation</a>：中等规模的大语言模型（LLM）——即那些拥有 7B 或 13B 参数的模型——展现出了极具前景的机器翻译（MT）性能。然而，即使是性能最好的基于 13B LLM 的翻译模型...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

intheclouddan: <@1197944730378588170> <@811015724877217803> 我对八月底在纽约市（NYC）感兴趣
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1265636540768583785)** (1 messages): 

> - `Feature Stores` (特征存储)
> - `ML Operations` (机器学习运维)
> - `Scalability` (可扩展性)
> - `Data Management` (数据管理)
> - `Feature Governance` (特征治理)


- **利用特征存储最大化 ML 效率**：一场关于 [利用特征存储](https://tinyurl.com/yfjscesh) 的 **直播会议** 定于 **2024年7月31日美国东部时间上午11:00** 举行，面向 ML 工程师、数据科学家和 MLOps 专业人员。
   - 会议将涵盖 **构建自动化流水线**、管理不可靠数据，并展示增强可扩展性和性能的高级用例。
- **解决 ML 数据中的不一致性**：网络研讨会将重点消除 **推理（serving）和训练数据** 之间的差异，以开发可扩展且可复现的模型。
   - 还将解决 **不一致的数据格式** 和特征重复等挑战，以改善 ML 团队内部的协作。
- **稳健的特征治理策略**：参与者将学习如何实施 **稳健的特征治理和版本控制策略**，这对于有效的 ML 生命周期管理至关重要。
   - 见解和实用工具将帮助参与者优化其 ML 流程并推动业务运营。



**提到的链接**：<a href="https://tinyurl.com/yfjscesh">Leveraging Feature Stores in ML</a>：加入 Hudson Buzby 的课程，学习如何推进 ML Operations 和可扩展性。

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1265387661103726703)** (1 条消息): 

> - `加速器申请截止日期`
> - `即将举行的活动`
> - `Zero Shot Tokenizer Transfer`
> - `AutoFix 开源问题修复工具` 


- **加速器申请截止日期临近**：加速器计划的申请截止日期即将到来，该计划为项目提供为期 **12 周的课程**以及高达 **10 万美元的非稀释性资金**。
   - Mozilla 还计划举办演示日 (Demo Day)，鼓励成员在[此处](https://discord.com/channels/1089876418936180786/1245083732319408195)提出他们的**问题**。
- **即将举行另外两场精彩活动**：提醒本月即将举行的两场活动，届时将展示知名参与者的工作，为社区带来新鲜见解。
   - 这些活动由两位成员发起，进一步增强了社区参与度。
- **深入探讨 Zero Shot Tokenizer Transfer**：一场名为 **Zero Shot Tokenizer Transfer** 的会议将与 Benjamin Minixhofer 共同举行，旨在深入研究 Tokenizer 实现中的先进技术。
   - 详细信息和参与链接可以在[此处](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)找到。
- **AutoFix：开源问题修复工具发布**：发布了关于 **AutoFix** 的公告，这是一个可以从 Sentry.io 提交 PRs 的开源问题修复工具，旨在帮助开发者简化工作流。
   - 有关该项目的更多信息可以在[此处](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732)获取。


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1265590495103680593)** (1 条消息): 

> - `Meta 的 Llama3.1 论文`
> - `Llama3 训练见解`
> - `幻觉预防技术` 


- **Llama3.1 论文：开源界的宝藏**：来自 [Meta 的新 Llama3.1 论文](https://threadreaderapp.com/thread/1815789501026861308)被誉为对开源社区极具**价值**，引发了关于其深刻见解的讨论。
   - *一位成员开玩笑说*，它包含如此多的 **Alpha**，以至于*你必须像看最喜欢的电影一样反复阅读*。
- **使用 15T Tokens 训练 405B 模型**：论文透露，拥有 **4050 亿参数**的模型是使用 **约 15 万亿 (15T) Tokens** 训练的，这是通过推导其 Scaling Laws 预测出来的。
   - *Scaling Law 建议*在 **16.55T Tokens** 上训练一个 **402B 参数模型**以获得最佳结果。
- **网络拓扑见解**：论文中包含了对其 **24k H100 集群**所使用的**网络拓扑**惊人详细的描述。
   - 推文中分享的图片展示了其**架构**，体现了基础设施的规模。
- **服务器问题导致的训练中断**：Llama3-405b 训练过程中的两次中断归因于**“服务器机箱 (Server Chassis)”**故障，有人幽默地暗示这是由于某人的失误造成的。
   - 结果，在预训练 (Pre-training) 期间由于这些故障损失了 **148 块 H100 GPU**。
- **幻觉预防基准测试讨论**：与 Meta 工程师的简短对话引发了对**幻觉预防**技术中需要更好**基准测试 (Benchmarks)**的关注。
   - 该成员分享说，*任何其他从事这一重要领域工作的人*都应该参与进一步的讨论。



**提到的链接**：<a href="https://threadreaderapp.com/thread/1815789501026861308">Thread by @jphme on Thread Reader App</a>：@jphme：正在直播推特 Meta 新 Llama3 论文中最有趣的见解 1. 他们是如何得出使用约 15T Tokens 训练的 405b 模型的？“将所得的 Scaling Law 推导至 3....

  

---



---



---



{% else %}


> 完整的频道细分内容已在邮件中截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}