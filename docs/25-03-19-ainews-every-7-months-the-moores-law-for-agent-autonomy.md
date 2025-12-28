---
companies:
- metr
- nvidia
- hugging-face
- canopy-labs
- meta-ai-fair
- microsoft
date: '2025-03-20T01:59:24.171505Z'
description: '**METR** 发布了一篇衡量 AI 智能体自主性进展的论文，研究显示自 **2019 年（GPT-2）**以来，AI 的自主能力每
  7 个月就会翻一倍。他们引入了一个新指标——**“50% 任务完成时间跨度” (50%-task-completion time horizon)**，目前 **Claude
  3.7 Sonnet** 等模型在大约 50 分钟的时间跨度内能达到 50% 的成功率。据预测，AI 将在 **2028 年实现 1 天的自主性**，并在 **2029
  年底实现 1 个月的自主性**。


  与此同时，**英伟达 (Nvidia)** 发布了用于条件式世界生成的 **Cosmos-Transfer1**，以及拥有 20 亿参数的类人机器人推理开源基础模型
  **GR00T-N1-2B**。**Canopy Labs** 推出了 **Orpheus 3B**，这是一款具备零样本语音克隆和低延迟特性的高质量文本转语音
  (TTS) 模型。据报道，**Meta** 因性能问题推迟了 **Llama-4** 的发布。**微软 (Microsoft)** 则推出了 **Phi-4-multimodal**
  多模态模型。'
id: 1b804ad2-b650-4eb1-bdfc-25a39c23d100
models:
- claude-3-7-sonnet
- llama-4
- phi-4-multimodal
- gpt-2
- cosmos-transfer1
- gr00t-n1-2b
- orpheus-3b
original_slug: ainews-every-7-months-the-moores-law-for-agent
people:
- reach_vb
- akhaliq
- drjimfan
- scaling01
title: 每 7 个月：智能体自主性的摩尔定律
topics:
- agent-autonomy
- task-completion
- multimodality
- text-to-speech
- robotics
- foundation-models
- model-release
- scaling-laws
- fine-tuning
- zero-shot-learning
- latency
---

<!-- buttondown-editor-mode: plaintext -->**视角即一切。**

> 2025年3月18日至3月19日的 AI 新闻。我们为您检查了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**227** 个频道和 **4117** 条消息）。预计节省阅读时间（以每分钟 200 字计）：**426 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

除了 [Llama 4 传闻](https://reddit.com/r/LocalLLaMA/comments/1jes8ue/llama4_is_probably_coming_next_month_multi_modal/) 和 [600 美元的 o1 pro API](https://x.com/openaidevs/status/1902485690958450871?s=46) 之外，我们很少能将一篇论文作为 AINews 的头条故事，所以当它发生时我们非常高兴。[METR](https://x.com/METR_Evals/status/1902384481111322929) 长期以来以对 AI 进展进行高质量分析而闻名，在《衡量 AI 完成长任务的能力》（[Measuring AI Ability to Complete Long Tasks](https://arxiv.org/pdf/2503.14499)）中，他们回答了一个迄今为止极难回答的有价值问题：**Agent 自主性正在增加，但速度有多快？**

**自 2019 年 (GPT2) 以来，它每 7 个月翻一番。**


![image.png](https://assets.buttondown.email/images/f700d580-e607-4b22-b5cf-b68daf96b148.png?w=960&fit=max)


显然，Agent 完成任务所需的时间各不相同，这使得这个问题难以回答，因此其方法论也非常值得关注：

> “为了根据人类能力量化 AI 系统的能力，我们提出了一个新的指标：**50% 任务完成时间跨度（50%-task-completion time horizon）。这是人类通常完成 AI 模型能以 50% 成功率完成的任务所需的时间。** 我们首先对具有相关领域专业知识的人员在 RE-Bench、HCAST 和 66 个新型短任务的组合上进行了计时。**在这些任务上，当前的前沿 AI 模型（如 Claude 3.7 Sonnet）的 50% 时间跨度约为 50 分钟。**”


![image.png](https://assets.buttondown.email/images/88dc7803-50d0-4dab-9f9a-775b229aeb0e.png?w=960&fit=max)


作者发现在 1 分钟跨度处存在显著的不连续性：


![image.png](https://assets.buttondown.email/images/ff524a95-489c-42f8-b0de-43bfb9d4319d.png?w=960&fit=max)


以及在 80% 的截止点，但 Scaling Laws 依然稳健。

按照目前的速度，我们将拥有：

- **1 天的自主性**：(5 个指数级增长 * 7 个月) = 3 年 (2028 年)
- **1 个月的自主性**：在 “2029 年底” (+/- 2 年，仅计算人类工作时间)


![image.png](https://assets.buttondown.email/images/28787374-3c6a-4059-8402-ca30ae204b68.png?w=960&fit=max)


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**AI 进展与模型发布**

- **Nvidia 在 Hugging Face 上发布了 Cosmos-Transfer1，用于具有自适应多模态控制的条件式世界生成**：[@_akhaliq](https://twitter.com/_akhaliq/status/1902187161841000938) 分享了 **Nvidia's Cosmos-Transfer1** 在 Hugging Face 上的发布，该模型支持具有自适应多模态控制的条件式世界生成。
- **Nvidia 在 Hugging Face 上发布了 GR00T-N1-2B**：[@_akhaliq](https://twitter.com/_akhaliq/status/1902124817228194289) 宣布 **Nvidia** 在 Hugging Face 上发布了 **GR00T-N1-2B**，这是一个用于通用人形机器人推理和技能的开源基础模型，[@reach_vb](https://twitter.com/reach_vb/status/1902120408742080558) 也提到了这一点。[@DrJimFan](https://twitter.com/DrJimFan/status/1902117478496616642) 提供了关于 **GR00T N1** 的细节，强调其作为全球首个仅有 2B 参数的人形机器人开源基础模型的地位。该模型从多样化的物理动作数据集中学习，并部署在各种机器人和仿真基准测试中。包含了白皮书、代码库和数据集的链接 ([@DrJimFan](https://twitter.com/DrJimFan/status/1902117481000701970), [@DrJimFan](https://twitter.com/DrJimFan/status/1902117483575963866), [@DrJimFan](https://twitter.com/DrJimFan/status/1902117485752779077), [@DrJimFan](https://twitter.com/DrJimFan/status/1902117487346643382), [@DrJimFan](https://twitter.com/DrJimFan/status/1902117489133416636))。
- [@reach_vb](https://twitter.com/reach_vb/status/1902445501427114043) 宣布了 **Orpheus 3B**，这是一个来自 Canopy Labs 的高质量、富有情感的 **Text to Speech** 模型，采用 Apache 2.0 许可证。主要特性包括 zero-shot 语音克隆、自然语音、可控语调、在 100K 小时音频上进行训练、输入/输出流式传输、100ms 延迟以及易于 fine-tuning。
- **Meta 压着 Llama-4 不发是因为它表现太差**：[@scaling01](https://twitter.com/scaling01/status/1902122901513630110) 评论说 **Meta** 因为性能不佳而未发布 **Llama-4**。
- **Microsoft 推出了 Phi-4-multimodal**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1902372844421460458) 报道称 **Microsoft** 推出了 **Phi-4-multimodal**，这是一个拥有 56 亿参数的高性能开源权重模型，能够同时处理文本、图像和语音。
- **腾讯的 Hunyuan3D 2.0 加速了模型生成速度**：[@_akhaliq](https://twitter.com/_akhaliq/status/1902199977096499424) 宣布 **腾讯** 在整个 **Hunyuan3D 2.0** 系列中实现了 **30 倍加速** 的模型生成速度，将处理时间从 30 秒缩短至仅 1 秒，已在 Hugging Face 上提供。
- **Together AI 推出 Instant GPU Clusters**：[@togethercompute](https://twitter.com/togethercompute/status/1902400229032259731) 宣布了 **Together Instant GPU Clusters**，配备 8–64 个 @nvidia Blackwell GPUs，完全自助服务并在几分钟内就绪，非常适合大型 AI 工作负载或短期爆发式需求。

**研究与评估**

- **METR 关于 AI 任务完成情况的研究**：[@METR_Evals](https://twitter.com/METR_Evals/status/1902384481111322929) 强调了他们的新研究，指出 **AI 能完成的任务长度大约每 7 个月翻一番**。他们定义了一个名为 **"50%-task-completion time horizon"** 的指标来跟踪模型自主性的进展，目前像 **Claude 3.7 Sonnet** 这样的模型跨度约为 50 分钟 ([@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1902244871785549909))。研究还表明，AI 系统可能在 5 年内能够自动化处理许多目前人类需要一个月才能完成的软件任务。该论文可在 arXiv 上查阅 ([@METR_Evals](https://twitter.com/METR_Evals/status/1902384522546815256))。
- **NVIDIA 推理模型**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1902386178206429434) 报道称 **NVIDIA** 宣布了他们的首批推理模型，这是一个新的开源权重 **Llama Nemotron 模型** 系列：Nano (8B), Super (49B) 和 Ultra (249B)。

**Agent 开发与工具**

- **LangGraph Studio 更新**：[@hwchase17](https://twitter.com/hwchase17/status/1902433431788908923) 宣布 **Prompt Engineering** 现已集成在 **LangGraph Studio** 中。[@LangChainAI](https://twitter.com/LangChainAI/status/1902433431788908923)
- **LangGraph 与 LinkedIn 的 SQL Bot**：[@LangChainAI](https://twitter.com/LangChainAI/status/1902441466036933018) 重点介绍了由 **LangGraph** 和 **LangChain** 驱动的 **LinkedIn Text-to-SQL Bot**，该工具能将自然语言问题转化为 SQL，使数据更易于访问。
- **Hugging Face 关于在 LlamaIndex 中构建 Agent 的课程**：[@llama_index](https://twitter.com/llama_index/status/1902387501492584580) 分享了 @huggingface 编写的一门关于在 LlamaIndex 中构建 Agent 的课程，涵盖了组件、RAG、Tools、Agents 和 Workflows，该课程免费提供。
- **Canvas UX**：[@hwchase17](https://twitter.com/hwchase17/status/1902168318414615042) 指出 **Canvas UX** 正在成为在文档上与 LLM 交互的标准。

**框架与库**

- **Gemma 软件包**：[@osanseviero](https://twitter.com/osanseviero/status/1902456220876787763) 介绍了 **Gemma package**，这是一个用于使用和微调 **Gemma** 的极简库，包含关于 Fine-tuning、Sharding、LoRA、PEFT、Multimodality 和 Tokenization 的文档。
- **AutoQuant**：[@maximelabonne](https://twitter.com/maximelabonne/status/1902309252821143682) 宣布了 **AutoQuant** 的更新，以优化 **Gemma 3 的 GGUF 版本**，实现了 imatrix 并将模型拆分为多个文件。
- **字节跳动 OSS 发布 DAPO**：[@_philschmid](https://twitter.com/_philschmid/status/1902258522059866504) 重点介绍了 ByteDanceOSS 发布的新开源 **RL 方法 DAPO**，其性能优于 **GRPO**，并在 **AIME 2024 benchmark** 上获得了 50 分。

**行业合作与活动**

- **Perplexity 与 NVIDIA 合作**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1902181155132309889) 宣布 **Perplexity** 正与 **NVIDIA** 合作，利用其新的 Dynamo 库增强在 Blackwell 上的 Inference；[@perplexity_ai](https://twitter.com/perplexity_ai/status/1902125640691937568) 表示他们正在实施 **NVIDIA Dynamo** 以提升推理能力。
- **Google 与 NVIDIA 合作**：[@Google](https://twitter.com/Google/status/1902106268375851435) 宣布他们正在整个 Alphabet 范围内扩大与 **NVIDIA** 的合作。
- **vLLM 与 Ollama 推理之夜**：[@vllm_project](https://twitter.com/vllm_project/status/1902422970712350744) 和 [@ollama](https://twitter.com/ollama/status/1902415210821960149) 正在旧金山的 **Y Combinator** 举办推理之夜，讨论 Inference 相关话题。

**幽默与杂项**

- [@willdepue](https://twitter.com/willdepue/status/1902133781626056934) 分享了 **"在 YC W25 学到的 10 件事"**，这是对受说唱歌词启发的商业规则的幽默解读。
- **广告拦截器**：[@nearcyan](https://twitter.com/nearcyan/status/1902279001680560570) 表示，广告拦截器（Adblockers）甚至还无法在普通大众中普及，你居然认为我们会给每个人开放任意代码执行（Arbitrary Code Execution）权限？
- **关于 AI Waifus 的题外话**：[@scaling01](https://twitter.com/scaling01/status/1902389718513328533) 在 TPOT 的背景下开玩笑说要拒绝现实女性，拥抱 AI waifus。
- **职业生涯**：[@cto_junior](https://twitter.com/cto_junior/status/1902224897142571509) 调侃道：该死，结婚完全毁掉了我的发帖量（换取了无条件的爱和终身的幸福）。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Llama4 传闻：下月发布，多模态，1M Context**

- **Llama4 可能下个月发布，支持多模态和长上下文** ([Score: 295, Comments: 114](https://reddit.com/r/LocalLLaMA/comments/1jes8ue/llama4_is_probably_coming_next_month_multi_modal/)): **Llama4** 预计将于下个月发布，其特点是具备 **多模态能力** 和约 **100 万 token** 的 **长上下文窗口**。该公告与 [Meta 博客](https://www.meta.com/blog/connect-2025-llamacon-save-the-date/?srsltid=AfmBOoqvpQ6A0__ic3TrgNRj_RoGpBKWSnRmGFO_-RbGs5bZ7ntliloW) 讨论即将举行的 2025 年 **Llamacon** 活动有关。
  - 针对 **上下文大小** 的讨论集中在对 **100 万 token 上下文窗口** 实用性的怀疑上，用户指出模型在达到此类限制之前通常会出现明显的性能下降。**Qwen 2.5** 因使用 **Exact Attention 微调** 和 **Dual Chunk Attention** 来有效管理长上下文而受到关注，详见 [基准测试论文](https://arxiv.org/html/2409.12181v2)。
  - **Llama4** 的 **多模态能力** 引发了辩论，一些用户对其效用表示怀疑，而另一些人则强调了潜在的好处，例如增强的图像和音频处理能力。**DeepSeek** 以及 **Mistral** 和 **Google Gemini** 等其他模型被提及作为竞争基准，用户对 Llama4 的创新架构寄予厚望。
  - 针对 Llama 模型中的 **审查（censorship）** 问题也被提出，希望 Llama4 相比 Llama3 能减少审查。对话还涉及了 **llama.cpp** 等项目中 **零日支持（zero-day support）** 的重要性，暗示与 Meta 的合作可能会大有裨益。


- **[只有老玩家才记得](https://i.redd.it/dh21r5dq5npe1.jpeg)** ([Score: 300, Comments: 58](https://reddit.com/r/LocalLLaMA/comments/1jevzm3/only_the_real_ones_remember/)): **Tom Jobbins (TheBloke)** 在 **Hugging Face** 上的贡献受到关注，特别是他关于“使用 AutoGPTQ 和 transformers 让 LLM 更轻量”的文章。他的个人资料展示了最近的模型及其创建和更新日期，表明了他对 **AI** 和 **ML** 的积极参与和兴趣。
  - **Tom Jobbins 的影响与消失：** 许多用户对 **Tom Jobbins** 为 **开源 AI 社区** 做出的重大贡献表示感谢，特别是他在提高 **AI** 可访问性方面的工作。讨论推测了他突然消失的原因，有人认为是职业倦怠，或者是跳槽到了私人公司，可能担任 **CTO**。
  - **职业转型与推测：** Jobbins 转向一家初创公司的职业变动受到关注，一些用户提到之前资助他工作的拨款已用完。关于他目前的活动存在幽默且乐观的推测，包括他可能在严格的 **NDA**（保密协议）下获得了一份丰厚的录用通知。
  - **社区情感与遗产：** 用户深情地怀念 Jobbins 在 **模型量化** 方面的开创性工作，以及他在为社区简化复杂流程方面发挥的作用。他的遗产继续受到赞赏，一些用户将他的 **Hugging Face** 个人资料加入书签以示致敬。


- **[梦想还是要有的](https://i.redd.it/cw3hsv4mwmpe1.png)** ([Score: 617, Comments: 79](https://reddit.com/r/LocalLLaMA/comments/1jev3fl/a_man_can_dream/)): 该帖子通过一张表情逐渐震惊的梗图，幽默地对比了 AI 模型 **"DEESEEK-R1"、"QWQ-32B"、"LLAMA-4"** 和 **"DEESEEK-R2"**，突显了围绕这些模型（尤其是 **LLAMA-4**）能力的期待和兴奋。
  - 评论者表达了对擅长编程的 **小模型** 的渴望，因为目前的模型更多地关注语言和通用知识，而 **Sonnet 3.7** 的 API 使用成本被认为过高。
  - 针对 AI 模型快速发布周期的推测和幽默不断，有人提到 **R1** 发布还不到 **60 天**，并对未来模型（如 **LLAMA-4** 和 **DEESEEK-R2**）可能达到 **1T 和 2T 参数** 规模开起了玩笑。
  - 讨论还包括对 **QwQ** 和 **QwQ-Max** 等模型名称的幽默解读，以及对 **Pro ProMax Ultra Extreme** 等命名惯例的调侃，并提及了 **Dell** 和 **Nvidia** 等品牌。


**主题 2. 微软的 KBLaM 及其替代 RAG 的潜力**

- **微软的 KBLaM，这看起来很有趣** ([Score: 104, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1jez456/kblam_by_microsoft_this_looks_interesting/)): 微软的 **KBLaM** 引入了一种将外部知识集成到语言模型中的方法，可能作为 **RAG (Retrieval-Augmented Generation)** 的替代方案。该帖子质疑 KBLaM 是否能取代 RAG，并认为解决与 RAG 相关的挑战可能是 AI 领域的重大进步。
  - **KBLaM 的集成方法** 通过使用“矩形注意力 (rectangular attention)”机制将知识直接编码到模型的注意力层中，绕过了 **RAG** 等传统方法的低效。这使得模型能随知识库大小线性扩展，从而在单个 GPU 上高效处理超过 **10,000 个知识三元组 (knowledge triples)**，提高了可靠性并减少了幻觉 (hallucinations)。
  - 讨论了 **模型优化的潜力**，通过将知识与智能分离，可以减小模型尺寸，允许根据需要注入知识。然而，关于智能是否能与知识完全分离存在争议，因为一些人认为它们是相互关联的。
  - **社区参与** 包括对 KBLaM 效率提升和可解释性的兴奋，用户正在评估 [KBLaM 仓库](https://github.com/microsoft/KBLaM/)。该方法无需重新训练即可保持动态更新的能力，被视为优于 RAG 的重大进步，因为 RAG 存在分块 (chunking) 等效率低下的问题。


- **如果《模型即产品》这篇文章属实，许多 AI 公司将面临危机** ([Score: 180, Comments: 94](https://reddit.com/r/LocalLLaMA/comments/1jex61b/if_the_model_is_the_product_article_is_true_a_lot/)): 该帖子讨论了一篇博客文章，暗示 AI 的未来可能会看到像 **OpenAI** 和 **Anthropic** 这样的大型实验室使用 **Reinforcement Learning (RL)** 为 **Agent** 用途训练模型，这可能会削弱应用层 AI 公司的作用。文中提到了 **DataBricks** AI 副总裁的一个预测，即闭源模型实验室可能会在未来 2-3 年内关闭其 API，这可能导致这些实验室与当前的 AI 公司之间竞争加剧。[点击此处阅读更多](https://vintagedata.org/blog/posts/model-is-the-product)。
  - 讨论集中在 **数据比模型更重要**，多位评论者强调，持久的价值在于数据和领域专业知识，而非模型本身。这通过 **Google 2006 年算法** 成功的类比得到了强调，突出了数据和 UI 是关键要素。
  - 对于 **OpenAI** 和 **Anthropic** 等主要 AI 实验室将关闭其 API 的预测存在怀疑。评论者认为 API 是这些公司商业模式的核心，关闭它们会适得其反，特别是考虑到 **Meta** 和 **DeepSeek** 等公司开源模型的兴起。
  - 对话还涉及了 **在第三方平台上建立业务的风险**，将其比作移动应用生态系统，平台所有者可以吸收成功的创意。共识是，虽然大型 AI 公司将主导通用用例，但利基、特定领域的解决方案仍有空间。


**主题 3. Gemma 3 无审查模型发布**

- **无审查版 Gemma 3** ([Score: 147, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1jej4s5/uncensored_gemma_3/)): 作者发布了 **Gemma 3** 的微调模型，可在 [Hugging Face](https://huggingface.co/soob3123/amoral-gemma3-12B) 上获取，声称该模型不会拒绝任何任务。他们还在努力训练 **4B** 和 **27B** 版本，旨在尽快测试并发布。
  - 用户测试了 **Gemma 3** 以查看它是否会拒绝任务，并指出尽管有相关声明，它有时仍会拒绝。**Xamanthas** 和 **StrangeCharmVote** 报告了褒贬不一的结果，**Xamanthas** 指出它仍然拒绝了一些任务，而 **StrangeCharmVote** 发现原始的 **27B** 模型出人意料地没有审查 (uncensored)。
  - 人们对 **Gemma 3** 与默认版本相比的性能指标很感兴趣，**mixedTape3123** 质疑其智能水平，而 **Reader3123** 承认可能存在差异，但缺乏详细指标。
  - **Reader3123** 分享了各贡献者的量化工作，提供了 **Hugging Face** 上的模型链接以供进一步探索：[soob3123](https://huggingface.co/soob3123/amoral-gemma3-12B-gguf)、[bartowski](https://huggingface.co/bartowski/soob3123_amoral-gemma3-12B-GGUF) 和 [mradermacher](https://huggingface.co/mradermacher/amoral-gemma3-12B-i1-GGUF)。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Gemini 插件和 AI Studio 的使用**

- **[天哪，我太爱 gemini photoshop 了](https://v.redd.it/h6ykfmmmmipe1)** ([评分: 124, 评论: 29](https://reddit.com/r/ChatGPT/comments/1jegd7m/god_i_love_gemini_photoshop/)): 该帖子表达了对 **Gemini Photoshop** 的热忱，表明用户对该工具有良好的体验。然而，文中未提供该工具的具体细节或功能。
  - **关于 Gemini 的困惑**：用户对 **Gemini Photoshop** 的提法感到困惑，质疑是否存在 Photoshop 的 **Gemini 插件**，或者是否将其与占星术符号等其他事物混淆了。
  - **Google AI Studio**：一些用户提到使用 **Google AI Studio Flash 2.0** 作为创建图像的工具，一名用户澄清它是免费的，只需在 Google 上搜索 "AI studio" 即可访问。
  - **用户体验褒贬不一**：虽然一些人对该工具表示热衷，但也有人表示不满，称其在遵循基本指令方面表现不佳，这表明用户体验存在差异。


**主题 2. MailSnitch 利用邮件标记识别垃圾邮件**

- **多亏了 ChatGPT，我知道谁在卖我的电子邮箱了。** ([评分: 2119, 评论: 204](https://reddit.com/r/ChatGPT/comments/1jec586/thanks_to_chatgpt_i_know_whos_selling_my_email/)): 该帖子概述了 **MailSnitch** 的开发过程，这是一个受 **email tagging** 启发而开发的工具，通过使用独特的标记邮箱地址来追踪谁卖掉了你的邮箱。作者在 **ChatGPT** 的帮助下，计划发布一个具有自动填充、唯一邮件标记和历史记录功能的 **Chrome Extension**，并考虑免费发布，未来可能进行商业化。
  - 许多评论者强调了**在邮箱地址中使用 "+" 号来追踪泄露源是无效的**，因为垃圾邮件发送者和数据经纪人可以轻易地绕过或删除它。为了更好的隐私和控制，建议使用 **email alias services**（如 [Firefox Relay](https://relay.firefox.com/)、**ProtonPass** 和 Apple 的 **"hide my email"** 功能）。
  - 几位用户强调了在使用代码之前**理解代码的重要性**，尤其是由 **ChatGPT** 生成的代码，因为存在潜在的安全和责任问题。讨论涉及运行未经验证代码的风险，以及在严肃应用中进行代码审查的必要性。
  - 评论者还讨论了**替代方案**，例如使用带有 catchall 地址的自定义域名，这可以实现更强大的邮件管理和追踪。一些用户分享了使用 **Gmail 的点号（dot）功能**的经验，并提到了它在不同平台上的局限性。


**主题 3. 逆向工程 ChatGPT：获得更好回答的策略**

- **我逆向工程了 ChatGPT 的思考方式。以下是获得更好答案的方法。** ([评分: 2064, 评论: 234](https://reddit.com/r/ChatGPT/comments/1jeq5d5/i_reverseengineered_how_chatgpt_thinks_heres_how/)): 该帖子解释说 **ChatGPT** 本质上并不“思考”，而是预测下一个最可能的词，这导致它对宽泛的问题给出平庸的回答。作者建议通过指示 ChatGPT 首先分析关键因素、自我批判其答案并考虑多个视角来增强回复，从而显著提高在 **AI/ML**、商业策略和调试等主题上的深度和准确性。
  - **ChatGPT** 仅仅是一个“下一个词预测器”的概念已被广泛认可，几位评论者指出该帖子的见解并不新颖，有人认为这些想法只是基础的 prompting 技术而非突破。**LickTempo** 和 **Plus_Platform9029** 强调，虽然 ChatGPT 预测下一个 token，但如果提示得当，它仍能表现出结构化的推理，他们推荐参考 **Andrej Karpathy 的视频**等资源以进行深入了解。
  - **Chain-of-thought prompting** 和 **Monte Carlo Tree Search** 被强调为改进 ChatGPT 回复的方法，**djyoshmo** 建议在 **arxiv.org** 或 **Medium** 上进一步阅读以掌握这些技术。**EverySockYouOwn** 分享了一个实际案例，即使用结构化的 20 个问题方法来分解复杂询问，从而增强 AI 回复的深度和相关性。
  - 讨论中还涉及了**自我批判**和**非回声筒（non-echo chamber）**提示词的有效性，**VideirHealth** 和 **legitimate_sauce_614** 等用户分享了鼓励 ChatGPT 挑战假设并提供更具逻辑性、多样化视角的策略。然而，**SmackEh** 等人提醒说，ChatGPT 的默认行为是顺从并避免冒犯，因此需要刻意引导才能进行更具批判性的互动。


**主题 4. 成功在本地运行 Wan2.1**

- **[终于让 Wan2.1 在本地运行了](https://v.redd.it/tzepkx1einpe1)** ([Score: 108, Comments: 25](https://reddit.com/r/StableDiffusion/comments/1jexrhf/finally_got_wan21_working_locally/)): **Wan2.1** 已成功在本地实现，用于 **video processing**。
  - 用户讨论了使用 **Wan2.1** 进行视频处理的时间，并比较了不同的硬件配置。**Aplakka** 分享说他们的 720p 生成耗时超过 60 分钟，而 **Kizumaru31** 报告使用 RTX 4070 生成 480p 需要 6-9 分钟，这表明使用像 RTX 4090 这样更强大的显卡速度会更快。
  - **Aplakka** 提供了一个 [工作流链接](https://pastebin.com/wN37A04Q) 并详细说明了他们的设置，包括在 **Windows Subsystem for Linux** 中使用 **ComfyUI** 以及 **Sageattention**。他们提到了在 RTX 4090 上将 720p 视频放入 **VRAM** 的挑战，并建议重启或调整设置可能会解决该问题。
  - 重点在于提高视频质量和对输出的控制，**BlackPointPL** 指出使用 **gguf** 会降低质量，而 **vizualbyte73** 表示希望对花瓣运动等视觉元素有更多控制。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要的总结

**主题 1. NVIDIA 的 Blackwell 闪电战：新 GPU 与营销炒作**

- [**Blackwell Ultra 与 Ruben 发布，Feynman 紧随其后**](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers): NVIDIA 发布了 **Blackwell Ultra** 和 **Ruben** GPU，下一代命名为 **Feynman**。**Ruben** 结合了 **silicon photonics** 以提高能效，并配备了新的 **ARM CPU**，以及带宽达到 **1.6 Tbps** 的 **Spectrum X** 交换机。尽管有性能宣称和新产品发布，一些用户仍对 **NVIDIA** 的营销炒作和性能夸大持怀疑态度，特别是关于 **H200** 和 **B200** 的加速效果。
- [**DGX Spark & Station：个人 AI 超级计算机问世**](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers): NVIDIA 推出了 **DGX Spark** 和 **DGX Station**，这是基于 **Grace Blackwell platform** 的紧凑型 AI 超级计算机。**DGX Spark**（前身为 Project DIGITS，售价 **$3,000**）旨在为开发者、研究人员和学生提供桌面级的 AI 原型设计和微调能力，尽管有些人认为其规格与 **Mac Mini M4 Pro** 或 **Ryzen 395+** 等替代方案相比并无优势。
- [**DeepSeek-R1 推理声称在 Blackwell 上全球最快**](https://nvda.ws/3FzAzCO): NVIDIA 断言 **Blackwell GPUs** 实现了 *全球最快的 DeepSeek-R1 推理*，在 **NVL8** 配置下的单台八卡 **Blackwell GPUs** 系统在全量 **DeepSeek-R1 671B parameter model** 上可提供 **253 TPS/user** 和 **30K TPS 系统吞吐量**。尽管有这些性能宣称，一些用户仍对 **NVIDIA** 的营销策略持批评态度，称其“不严肃”且可能误导投资者。

**主题 2. 开源 AI 生态系统：工具、数据集与社区**

- [**哈佛研究发现开源投资产生巨额回报**](https://x.com/ClementDelangue/status/1901765997696008307?t=0s9dSKc6E5S4wJf7TzT1Dw&s=19): 哈佛大学的研究表明，*对开源的 41.5 亿美元投资为公司创造了 8.8 万亿美元的价值*，强调了 **每投入 1 美元就有 2,000 美元的回报**。这凸显了 AI 领域开源贡献所驱动的巨大经济影响和价值创造。
- [**Nvidia 为 Llama Nemotron 开源大规模编程数据集**](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1): NVIDIA 发布了一个大型 **开源 instruct coding dataset**，以增强 **Llama instruct models** 在数学、代码、推理和指令遵循方面的能力。该数据集包含来自 **DeepSeek-R1** 和 **Qwen-2.5** 的数据，引发了社区对专门训练进行过滤和微调的兴趣。
- [**PearAI 作为 Cursor 的开源 IDE 替代方案出现**](https://trypear.ai/): **PearAI** 是一款集成了 **Roo Code/Cline**、**Continue**、**Perplexity**、**Mem0** 和 **Supermaven** 等工具的新型开源 AI 代码编辑器，正在 Cursor 社区中获得关注。用户认为 **PearAI** 是 **Cursor** 的一个更便宜且可行的替代方案，尽管其上下文窗口较小，这凸显了开源工具领域的不断发展。

**主题 3. 模型性能与局限性：Gemini、Claude 及开源替代方案**

- [**Gemini Deep Research 为用户节省 10 倍时间，但基准测试成本隐忧浮现**]：用户发现 **Gemini Deep Research** 显著加快了研究任务，潜在地节省了 **10 倍**的时间，一位用户提到它在科学医学研究中的价值，生成了包含 **90 个文献来源**的列表。然而，其成本对于在 LMArena 等平台上进行广泛的基准测试来说可能过于昂贵，引发了对其在大范围评估中可访问性的质疑。
- [**Perplexity AI 面临“比 Claude 笨”的质疑，O1 热度消退**]：一些用户发现 **Perplexity AI** 不如 **Claude 3.5** 智能，理由是上下文保留（context retention）和摘要生成方面存在问题，一位用户表示“*Perplexity 感觉比 Claude 3.5 更笨*”。此外，由于引入了**付费墙**，社区最初对 **Perplexity O1 模型**的热情正在减退，削弱了其吸引力。
- [**Anthropic 的 Claude 3.7 Sonnet 经历宕机后恢复**]：**Anthropic 模型**，特别是 **Claude 3.7 Sonnet**，经历了服务中断和宕机。虽然据报道服务正在趋于稳定，但这一事件凸显了与基于云的 AI 模型访问相关的潜在不稳定性和可靠性担忧。

**Theme 4. AI Agents 与工具：Agents 课程、MCP 以及工作流创新**

- [**Hugging Face 推出免费 LlamaIndex Agents 课程**](https://t.co/eACAJzXg8y)：[Hugging Face](https://huggingface.co/) 发布了关于在 **LlamaIndex** 中构建 Agent 的**免费课程**，涵盖了核心组件、RAG、工具和工作流。该课程为使用 LlamaIndex 开发 AI 驱动的应用提供了全面指南，扩展了 Agentic AI 领域的教育资源。
- [**Model Context Protocol (MCP) 凭借 Python REPLs 和新工具势头强劲**](https://github.com/ericaxelrod-1/model-context-protocol)：**Model Context Protocol (MCP)** 正受到关注，出现了如 [hdresearch/mcp-python](https://github.com/hdresearch/mcp-python) 和 [Alec2435/python_mcp](https://github.com/Alec2435/python_mcp) 等新的 Python REPL 实现，以及一个用户为 Windows 上的 Cursor 构建的 **DuckDuckGo MCP** ([GitHub](https://github.com/ericaxelrod-1/model-context-protocol))，展示了其日益增长的生态系统和在工具集成方面的实用性。
- [**Aider 代码编辑器新增网页和 PDF 读取功能，增强多模态能力**](https://aider.chat/docs/usage/images-urls.html#images)：Aider 代码编辑器现在支持读取网页和 PDF 文件，使 **GPT-4o 和 Claude 3.7 Sonnet** 等具备视觉能力的模型能够直接在编码环境中处理多样化的信息源。这一增强功能通过 `/add <filename>` 等命令访问，扩展了 Aider 在处理复杂、信息密集型编码任务中的用途。

**Theme 5. 硬件与软件挑战：性能、兼容性与成本**

- [**LM Studio 中报告多 GPU 性能下降**]：用户在 **LM Studio** 中使用多个 **RTX 3060s** 配合 CUDA llama.cpp 时遇到了性能和稳定性问题，并指出单 GPU 的性能更优。这些问题归因于模型在 GPU 间的拆分以及潜在的 PCI-e 3.0 x4 带宽限制，表明多 GPU 配置在实现最佳性能方面并不总是即插即用的。
- [**Gemma 3 Vision 微调遭遇 Transformers 库故障**]：用户在为视觉任务微调 **Gemma 3** 时遇到问题，这表明当前 **Transformers** 库版本的视觉支持中可能存在 Bug，特别是在进行 `qlora` 期间。该问题表现为 `RuntimeError`，需要进一步调查兼容性和库依赖关系。
- [**M1 Mac 在模型训练中表现吃力，即使是小批量任务**]：用户报告称 **M1 Mac Airs** 在模型训练方面动力不足，即使是小批量任务，在 Kaggle 和 Hugging Face Spaces 等平台上也会遇到与 **clang** 相关的问题。这凸显了消费级 Apple silicon 在处理高需求 AI 训练任务时的局限性，促使用户寻求替代硬件或基于云的解决方案。


---

# PART 1: Discord 高层级摘要

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet MAX 作为 Agent 表现出色**：**Sonnet MAX** 模型在 Agent 工作流中的**后处理（post-processing）**能力受到赞赏，详见[此 X 帖子](https://x.com/kregenrek/status/1901990102936515040)。
   - 用户强调，由于库的限制，**Cursor** 必须从**代码库**和**自身的错误**中学习。
- **Cursor 的 Claude Max 定价引发争议**：社区成员对 **Cursor** 上 **Claude Max** 的定价表示质疑，指出订阅中的**快速请求（fast requests）**配额未被充分利用。
   - 一些人表示不满，建议如果 **Cursor** 优化了 **Max** 的**快速请求**消耗，他们可能会寻找替代方案，并指出 *"Cursor 团队在不允许 Max 消耗更多快速请求方面做得非常糟糕"*。
- **终端问题困扰用户**：用户对 Agent 生成**多个终端**并重新运行项目感到沮丧，引发了关于实施预防性规则和配置的讨论。
   - 提议了一项*增强型终端管理规则*，用于终止打开的终端，将测试输出定向到新终端，并防止在测试运行期间创建重复终端。
- **PearAI 开源 IDE 出现**：社区正在关注 **PearAI** ([https://trypear.ai/](https://trypear.ai/))，这是一个集成了 **Roo Code/Cline**、**Continue**、**Perplexity**、**Mem0** 和 **Supermaven** 等工具的开源 AI 代码编辑器。
   - 成员们认为，*与 Cursor 相比，Pear 目前的表现非常出色*，因为尽管它的上下文窗口较小，但它是 **Cursor** 的廉价替代方案。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 Vision 遇到 Transformers 故障**：用户报告了在视觉任务中微调 **Gemma 3** 的问题，表明当前的 **Transformers** 版本在视觉支持方面可能存在问题，可能发生在 `qlora` 期间。
   - 一名用户在尝试对 `gemma-3-12b-pt-bnb-4bit` 进行 `qlora` 以进行图像标注时遇到了 *RuntimeError: Unsloth: Failed to make input require gradients!*，建议需要进一步调查。
- **Unsloth 即将支持多节点多 GPU**：Unsloth 计划在未来几周内支持多节点和多 GPU 微调，尽管具体发布日期尚未确定，可订阅 [Unsloth 时事通讯](https://unsloth.ai/newsletter)。
   - 一名成员确认多节点支持将仅限企业版。
- **Unsloth 在旧金山加入 vLLM 和 Ollama 活动**：Unsloth 将于下周四（3 月 27 日）在旧金山加入 vLLM 和 Ollama 的活动，承诺在 Y Combinator 的旧金山办公室进行社交和演示。
   - 更多详情请见 [vLLM & Ollama 推理之夜](https://lu.ma/vllm-ollama)，现场将提供食物和饮料。
- **文件系统问题导致文档保存失败**：一名用户在尝试本地保存合并模型时遇到了 `HFValidationError` 和 `FileNotFoundError`，原因是调用 `save_pretrained_merged` 时仓库 ID 无效。
   - 建议更新 `unsloth-zoo`，因为该问题在最新版本中[应该已经修复](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/saving_utils.py#L544-L549)。
- **ZO2 微调 175B LLM**：[ZO2 框架](https://github.com/liangyuwang/zo2) 仅需 **18GB** GPU 显存即可实现 **175B LLM** 的全参数微调，特别适用于 GPU 显存有限的设置。
   - 一名成员指出 **ZO2** 采用了[零阶优化（zeroth order optimization）](https://huggingface.co/papers/2503.14456)，这与 **SGD** 等更常见的阶优化方法形成对比。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **OpenVoice 克隆你的声音**：成员们重点介绍了 [OpenVoice](https://research.myshell.ai/open-voice)，这是一种多功能的即时语音克隆方法，仅需一段短音频即可复制声音并生成多种语言的语音，并附带了其 [GitHub 仓库](https://github.com/myshell-ai/OpenVoice)。
   - 它让你能够对语音风格进行细粒度控制，包括情感、口音、节奏、停顿和语调，并能复制参考说话者的音色。
- **Oblix 在云端编排模型**：一位成员分享了 [Oblix Project](https://oblix.ai/)，这是一个用于本地和云端模型之间无缝编排的平台，并在[演示视频](https://youtu.be/j0dOVWWzBrE)中进行了展示。
   - Oblix 根据复杂性、延迟需求和成本将 AI 任务路由到云端或边缘，从而智能地根据复杂性、延迟要求和成本考虑来引导 AI 任务。
- **PCIE 带宽影响不大**：成员们发现，在设置**双 4090**时，**PCIE 带宽**对推理速度的影响微乎其微，与 PCI-e 4.0 x8 相比，最多只多出 2 个 tps。
   - 共识是，从 **PCIE 4.0** 升级到 **PCIE 5.0** 对推理任务的提升非常有限。
- **RTX PRO 6000 Blackwell 发布**：NVIDIA 发布了其 [RTX PRO 6000 "Blackwell" GPU](https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai)，该显卡采用 GB202 GPU，拥有 24K 核心、96 GB VRAM，功耗需求为 600W TDP。
   - 由于配备了 HBM，这款显卡的性能*被认为远好于* 5090。
- **多 GPU 性能受损**：一位用户分享到，在 LM Studio 中使用 CUDA llama.cpp 运行多个 **RTX 3060**（3 个在 PCI-e x1 上，1 个在 x16 上）时，观察到性能下降和不稳定性，因为单 GPU 性能（x16）更优。
   - 有人建议，性能问题源于模型在多个 GPU 之间的拆分方式，且 Pci-e 3.0 x4 会使推理速度降低高达 10%。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **矿卡拯救 AI 家庭服务器**：用户正在考虑将旧的 **Radeon RX 580 GPU** 用于本地 AI 服务器，但被引导去购买阿里巴巴上的 **P104-100 或 P102-100**，它们拥有 8-10 GB VRAM。
   - **Nvidia** 在 BIOS 中限制了 VRAM，但中国卖家通过刷入固件来解锁所有可用显存。
- **开源投资价值倍增**：哈佛大学的研究显示，*投入开源的 41.5 亿美元为公司创造了 8.8 万亿美元的价值*，正如[这篇 X 帖子](https://x.com/ClementDelangue/status/1901765997696008307?t=0s9dSKc6E5S4wJf7TzT1Dw&s=19)中所讨论的，这意味着 **1 美元的投入 = 2,000 美元的价值创造**。
   - 这突显了开源贡献带来的显著经济回报。
- **Oblix 编排本地与云端模型**：**Oblix Project** 提供了一个用于本地和云端模型之间无缝编排的平台，如[演示视频](https://youtu.be/j0dOVWWzBrE)所示。
   - **Oblix** 中的自主 Agent 监控系统资源，并动态决定是在本地还是在云端执行 AI 任务。
- **Gradio Sketch AI 代码生成上线！**：Gradio Sketch 发布了一项更新，其中包括针对事件函数的 **AI 驱动代码生成**，可以通过输入 `gradio sketch` 或访问[托管版本](https://huggingface.co/spaces/ysharma/Sketch)来使用。
   - 这使得无需编写代码即可在几分钟内完成工作，而以前则需要数小时。
- **LangGraph 资料已在 GitHub 发布**：由于同步问题，LangGraph 单元 2.3 的资料已在 [GitHub 仓库](https://github.com/huggingface/agents-course/tree/main/units/en/unit2/langgraph)上线。
   - 这让急切的用户可以在网站更新前获取内容，确保他们能继续课程学习。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 与 ChatGPT 争夺霸主地位**：成员们讨论了 [Gemini Advanced](https://gemini.google.com/) 与 [ChatGPT Plus](https://chat.openai.com/) 的优劣，对 **Gemini 2.0 Flash Thinking** 和 **ChatGPT** 的安全对齐（safety alignment）及实现方式意见不一。
   - 一位成员称赞 **Gemini** 提供免费且*无限*的访问权限，而另一位成员则批评其缺乏基础安全功能以及整体实现水平。
- **o1 和 o3-mini-high 依然稳坐头把交椅**：尽管 **Gemini** 备受关注，一些用户坚持认为 **OpenAI** 的 **o1** 和 **o3-mini-high** 模型在编程、规划和数学等推理任务中表现卓越。
   - 这些用户认为 **Google** 模型在这些领域表现*最差*，只有 **Grok 3** 和 **Claude 3.7 Thinking** 可能与 **o1** 和 **o3-mini-high** 竞争。
- **GPT-4.5 的创意写作表现平平**：一位成员发现 **GPT-4.5** 的创意写作表现不一致，指出其存在逻辑错误、重复，且偶尔与 **GPT-4 Turbo** 相似。
   - 虽然在随后的运行中性能有所提高，但该用户对该模型*极低的消息限制*表示遗憾。
- **DeepSeek 被大学校园禁用**：一位成员报告称其大学禁用了 **DeepSeek**，可能是因为其缺乏准则或过滤器，在避开*非法*话题时性能会大幅下降。
   - 禁令似乎只针对 **DeepSeek**，而不针对其他 **LLM**。
- **ChatGPT 沙盒探索“有用性”的边界**：成员们正在通过探索模型对不同提示词和系统消息的反应来实验 **ChatGPT** 的个性，并展示了**“不友好的助手”**角色扮演的示例。
   - 一位成员发现，在不更改系统消息的情况下，很难让 **GPT-4o sandbox** 中的模型脱离*“不友好”*的状态。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Anthropic 的 Claude 3.7 Sonnet 出现停机**：**Anthropic** 模型，特别是 **Claude 3.7 Sonnet**，经历了停机，目前正在恢复中。
   - 用户报告称 **Anthropic** 的服务在停机后似乎正在趋于稳定。
- **Cline 兼容性看板为模型排名**：一位社区成员创建了一个 [Cline Compatibility Board](https://cline-compatibility-board.vercel.app/)，用于根据模型在 **Cline** 中的表现进行排名。
   - 该看板提供了关于 **API providers**、**plan mode**、**act mode**、输入/输出成本以及 **Claude 3.5 Sonnet** 和 **Gemini 2.0 Pro EXP 02-05** 等模型最大输出的详细信息。
- **Gemini 2.0 Pro EXP-02-05 存在故障**：**OpenRouter** 上的 **Gemini-2.0-pro-exp-02-05** 模型确认可用，但会遇到*随机故障*和*频率限制（rate limiting）*。
   - 根据兼容性看板，该模型目前以 **0 成本**提供，输出限制为 **8192**。
- **Gemini 模型在 RP 场景中表现狂躁？**：一些用户发现 **Gemini** 模型（如 *gemini-2.0-flash-lite-preview-02-05* 和 *gemini-2.0-flash-001*）在角色扮演（RP）场景中表现*不稳定*，呈现出*狂躁*行为，即使在 Temperature 设置为 1.0 时也是如此。
   - 然而，其他用户报告 *gemini-2.0-flash-001* 完全没有问题，认为它在 Temperature 为 1.0 时*非常连贯且稳定*。
- **OpenRouterGo SDK v0.1.0 发布**：[OpenRouterGo v0.1.0](https://github.com/eduardolat/openroutergo) 已发布，这是一个用于访问 **OpenRouter API** 的 **Go SDK**，具有简洁流畅的接口。
   - 该 **SDK** 包含自动模型回退（fallbacks）、函数调用（function calling）和 **JSON** 响应验证功能。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **NVIDIA 发布 Nemotron 开源权重 LLMs**：NVIDIA 推出了 Llama Nemotron 系列开源权重模型，包括 Nano (8B)、Super (49B) 和 Ultra (249B) 模型。初步测试显示，**Super 49B 模型**在推理模式下的 GPQA Diamond 测试中达到了 **64%** 的准确率。
   - 这些模型因其推理能力和潜在应用引发了广泛关注，[X 上的推文](https://x.com/ArtificialAnlys/status/1902386178206429434)提到了这一发布消息。
- **DeepSeek R1 671B 登陆 SambaNova Cloud**：**DeepSeek R1 671B** 现已在 SambaNova Cloud 上正式商用，支持 **16K 上下文长度**，并提供与主流 IDE 的 API 集成。发布后迅速走红，[SambaNovaAI 的推文](https://x.com/SambaNovaAI/status/1902072036064997702)确认了这一消息。
   - 该服务已向所有开发者开放，为各种应用提供访问这一大型模型的权限。
- **Aider 新增网页和 PDF 读取能力**：Aider 现在支持从 URL 读取网页和 PDF 文件，可通过 `/add <filename>`、`/paste` 以及命令行参数，配合 **GPT-4o 和 Claude 3.7 Sonnet** 等具备视觉能力的模型使用，详情见[此处文档](https://aider.chat/docs/usage/images-urls.html#images)。
   - 这一功能增强了 Aider 处理多样化信息源的能力，尽管各模型的相对价值仍在讨论中。
- **Gemini Canvas 扩展协作工具**：Google 的 [Gemini](https://blog.google/products/gemini/gemini-collaboration-features/) 推出了带有 **Canvas** 的增强协作功能，提供实时文档编辑和原型代码编写。
   - 这个交互式空间简化了写作、编辑和分享工作，并提供快速编辑工具来调整语气、长度或格式。
- **Aider 通过 Aiderignore 忽略仓库文件**：Aider 允许用户使用 `.aiderignore` 文件从 repo map 中排除文件和目录，详见[配置选项](https://aider.chat/docs/config/options.html#--aiderignore-aiderignore)。
   - 该功能有助于让 LLM 专注于相关代码，从而提高代码编辑的效率。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **腾讯混元预热 T1 等级 (Hunyuan Hypes T1 Hierarchy)**：`@TXhunyuan` 正在寻找合作伙伴共同 *进军* **T1** 模型，并在 [X 上的帖子](https://x.com/TXhunyuan/status/1902336731728506978)中质疑了在众多模型已经占据了 **显著字母** 后，用于命名推理模型的字母可用性。
   - 社区对潜在名称进行了辩论，考虑到在众多模型已经抢占先机后，剩下的选项已经不多了。
- **三星 ByteCraft 生成游戏**：**SamsungSAILMontreal** 推出了 [ByteCraft](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/paper/ByteCraft.pdf)，这是一个将文本提示词转换为可执行视频游戏文件的生成模型，可通过 [7B 模型](https://huggingface.co/SamsungSAILMontreal/ByteCraft)和[博客文章](https://emygervais.github.io/2025/03/15/bytecraft.html?v1)获取。
   - 早期工作需要 *极高的 GPU 需求*，最多需要 **4 张 GPU** 运行 **4 个月**。
- **NVIDIA DGX Spark 和 Station 亮相**：**NVIDIA** 发布了其新型 [DGX Spark 和 DGX Station](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers) *个人 AI 超级计算机*，由该公司的 **Grace Blackwell 平台**驱动。
   - **DGX Spark**（前身为 **Project DIGITS**）是一款售价 **3,000 美元、Mac Mini 大小**的 *全球最小 AI 超级计算机*，面向 AI 开发者、研究人员、数据科学家和学生，用于原型设计、微调和推理。
- **加州 AB-412 法案威胁 AI 初创公司**：加州立法者正在审查 [A.B. 412](https://www.eff.org/deeplinks/2025/03/californias-ab-412-bill-could-crush-startups-and-cement-big-tech-ai-monopoly) 法案，该法案强制要求 AI 开发者跟踪并披露 AI 训练中使用的每一项已注册版权的作品。
   - 批评者担心这种 *不可能的标准* 可能会 *摧毁小型 AI 初创公司和开发者*，同时巩固科技巨头的垄断地位。此外，**AI2** 向 [科学和技术政策办公室 (OSTP)](https://allenai.org/blog/OSTP) 提交了一份建议，倡导开放的创新生态系统。
- **NVIDIA 遭遇“不严肃”营销质疑**：**NVIDIA** 正在广告中宣传 **H200** 在 **H100 节点**上的性能，以及 **B200 相比 H200 在从 FP8 切换到 FP4 后实现了 1.67 倍的加速**，一些人通过[这些推文](https://x.com/_clashluke/status/1902411786554355836)和[这里](https://x.com/NVIDIAAIDev/status/1902068372608852304)形容 NVIDIA 的营销 *非常不严肃*。
   - **NVIDIA** 声称拥有 *全球最快的 DeepSeek-R1 推理速度*，单个系统在 **NVL8** 配置下使用 8 个 **Blackwell GPU**，在完整的 **DeepSeek-R1 671B 参数模型**上可提供 **253 TPS/用户** 或 **30K TPS 的系统吞吐量**，更多详情请见 [NVIDIA 官网](https://nvda.ws/3FzAzCO)。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI 是“三位一体”的威胁**：一位成员指出 **OpenAI** 在 **模型开发**、**产品应用** 和 **定价策略** 方面具有综合实力，使其在 AI 领域脱颖而出。
   - 其他公司可能只擅长这些关键领域中的一两个，但 OpenAI 在各方面都表现卓越。
- **AI 伴侣关系引发成瘾情绪**：随着 **AI Agent**（如语音助手）培养了成瘾倾向，导致一些用户产生情感依赖，担忧正在浮现。
   - 这一趋势引发了关于刻意设计的成瘾功能及其对用户依赖性潜在影响的伦理问题，并讨论了公司是否应该避免可能增强此类行为的功能。
- **智能眼镜：伪装成都市时尚的数据收割？**：一场讨论质疑了 **Meta** 和 **Amazon** 的智能眼镜，认为其意图是数据收割，特别是为机器人公司策划的自我中心视角（egocentric views）。
   - 一位成员开玩笑说了一个针对反派的智能眼镜初创公司点子，强调了情绪检测、梦境电影和共享视角等功能，通过创造依赖反馈回路来收集用户数据并训练模型。
- **AI 艺术版权：不允许机器人作者！**：美国上诉法院确认，**没有人类投入的 AI 生成艺术不受美国法律保护**，支持美国版权局对 Stephen Thaler 的 **DABUS** 系统的立场；详见 [路透社报道](https://www.yahoo.com/news/us-appeals-court-rejects-copyrights-171203999.html)。
   - 法院强调只有人类作者的作品才能获得版权，这标志着在应对快速发展的生成式 AI 行业的版权影响方面的最新尝试。
- **Llama 4 即将到来？**：传闻暗示 **Llama 4** 可能会在 **4 月 29 日**发布。
   - 这一推测与 Meta 即将举行的活动 [Llamacon 2025](https://www.meta.com/blog/connect-2025-llamacon-save-the-date/) 有关。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 被认为比 Claude 笨**：一位用户表示 *Perplexity 感觉比 Claude 3.5 笨*，并指出了在上下文保留（context retention）和抽象生成方面的问题。
   - 建议使用无痕模式或禁用 AI 数据保留以防止数据存储，并指出开启新对话可以提供一个干净的状态。
- **O1 热度因付费墙而减退**：社区测试表明，最初高估了 **O3 mini**，而低估了 **o1** 和 **o1 pro**，**付费墙（paywall）** 显著降低了用户的热情。
   - 一位用户报告称 *R1 在大多数时候都没用*，而 `o3-mini` 在调试 `js` 代码时效果更好。
- **Oblix 项目在边缘端和云端之间转换**：根据[演示视频](https://youtu.be/j0dOVWWzBrE)，[Oblix Project](https://oblix.ai/) 使用 Agent 监控系统资源，在本地和云端模型之间进行编排。
   - 该项目在云端和设备端模型之间动态切换执行。
- **Perplexity API 响应不稳定**：一位用户报告了 Perplexity API 的随机响应问题，特别是在进行**数百次快速网页搜索**时。
   - 在快速重复调用 `conductWebQuery` 函数时，代码要么只收到随机响应，要么忽略随机查询，这可能是由于实现错误导致的。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **GDocs 布局被破坏！**：用户发现转换为 **GDocs** 会破坏布局，且无法导入教师演示文稿中的大部分图像；他们发现提取文本（使用 *pdftotext*）并转换为纯图像格式有助于 Grounding。
   - 纯图像 PDF 可能会超过 **200MB**，由于 **NLM 的文件大小限制**，需要进行拆分。
- **NotebookLM：单人秀！**：用户发现通过自定义功能，可以让它完成任何你想做的事（单人播客可以是男性或女性、模仿特定人格、叙述故事、逐字阅读）。
   - *唯一的限制是你的想象力*。
- **播客休闲模式：脏话连篇！**：反馈表明**休闲播客模式**可能包含脏话。
   - 目前尚不清楚是否提供**纯净设置（clean setting）**。
- **NotebookLM 无法强制换行**：用户无法在 **NotebookLM** 的响应中强制执行**换行**和**间距**，因为 AI 是根据需求添加这些内容的，目前用户无法配置。
   - 作为音频概览（audio overviews）的替代方案，建议用户**下载音频**并将其作为**源文件上传**，以生成转录文本。
- **思维导图功能逐步推出**：用户讨论了 NotebookLM 中新的**思维导图（Mind Map）功能**，该功能可以视觉化地总结上传的源文件，参见 [Mind Maps - NotebookLM Help](https://support.google.com/notebooklm/answer/16070070)。
   - 该功能正逐步向更多用户推出，以便控制 Bug：*先向有限数量的人发布新功能，然后随时间推移逐渐增加人数是更好的做法，因为这样有时间排除出现的任何 Bug，并在所有人都能使用时提供一个更完善的版本*。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **通过 Glama API 列出 Smithery 注册表**：一位成员利用 **Glama API** 枚举了 **GitHub URL** 并验证了 *smithery.yaml* 文件的存在，并将其代码描述为“一次性的临时脚本（one time hack job script）”。
   - 他们表示如果有足够的兴趣，会考虑创建一个 **gist**，并强调了该脚本的临时性质。
- **关于 Spring 应用与 Spring-AI-MCP-Core 的问题**：一位用户首次在 **Open-webui** 中探索 **MCP**，并使用 *spring-ai-mcp-core* 构建基础 **Spring app**，正在寻求除 **ClaudeMCP** 和 *modelcontextprotocol* [GitHub 仓库](https://github.com/modelcontextprotocol)之外的资源。
   - 他们询问了 **MCP 与 GraphQL** 或 function calling 的对比，以及它如何处理 system prompts 和 **multi-agent systems**。
- **Claude Code MCP 实现发布**：一位成员发布了 [Claude Code MCP](https://glama.ai/mcp/servers/nqo1hvazke)，这是 **Claude Code** 作为 **Model Context Protocol (MCP)** 服务端的实现。
   - 他们在集成 **Claude Desktop** 时曾就 *claude_desktop_config.json* 的 *json* 行寻求帮助，但随后自行解决了问题。
- **为 Windows 上的 Cursor 构建的 DuckDuckGo MCP 框架**：由于现有的 **NPM** 项目运行失败，一位成员在 **Python framework** 上为 **Windows** 端的 **Cursor** 创建了自己的 **DuckDuckGo MCP**。
   - 它支持 **web**、**image**、**news** 和 **video** 搜索，无需 **API** key，已在 [GitHub](https://github.com/ericaxelrod-1/model-context-protocol) 上线。
- **MCP Python REPL 受到关注**：成员们就 [hdresearch/mcp-python](https://github.com/hdresearch/mcp-python)、[Alec2435/python_mcp](https://github.com/Alec2435/python_mcp) 和 [evalstate/mcp-py-repl](https://github.com/evalstate/mcp-py-repl) 作为 MCP 的 Python REPL 交换了意见。
   - 有人担心其中一个实现“完全没有隔离，可能导致灾难”，建议使用 **Docker** 进行沙箱化（sandbox）访问。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Phi-4 可作为你的辅助模型**：成员们讨论了 **Phi-4** 在复杂系统中作为有用“辅助模型（auxiliary model）”的潜力，强调了其指令遵循（direction-following）、LLM 接口交互和角色扮演（roleplay）能力。
   - 观点认为，在已经拥有多个其他模型的复杂系统中，它作为*辅助模型*会非常有用。
- **Claude 的参数列表错误**：一位用户批评了 **Claude AI** 的建议，指出其模型大小不准确，因为列出的模型并不符合所要求的 *10m parameters*（1000 万参数）以下，详见 [Claude 的输出](https://claude.ai/share/03dcf20f-800a-4cdc-b961-30f4009555af)。
   - 一位成员辩护称，“10m”可能被理解为“在现代硬件上普遍可用的极轻量级模型”的简称。
- **“Vibe Coding”是背景学习**：一位成员分享了通过沉浸式学习日语的轶事，并将其类比为获取技能的“vibe coding”。
   - 他们补充道，“即使是 vibe coding，你仍然需要担心模块间的接口等问题，以便在有限的 LLM context windows（上下文窗口）下保持扩展性。”
- **Nvidia 发布大规模编程数据集**：一位用户分享了 [Nvidia 开源的指令编程数据集](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1)，旨在提升 **Llama instruct 模型** 的数学、代码、通用推理和指令遵循能力，其中包含来自 **DeepSeek-R1** 和 **Qwen-2.5** 的数据。
   - 另一位下载了该数据集的成员报告称，对其进行过滤和训练将会很有趣。
- **有限的 VRAM 促使利基调优**：一位成员寻求帮助，希望在 **RTX 3080** 等有限 VRAM 的条件下寻找训练模型的“利基（niche）”方向。
   - 讨论内容包括各种 QLoRA 实验，以及对代码编辑进行 fine tune（微调）的建议。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena 面临缓慢消亡？**：一位成员质疑 **LMArena** 上的测试人员质量，问道 *“真正的测试者、改进者和真正的思考者都去哪了？”* 并分享了一个 [Calm Down GIF](https://tenor.com/view/calm-down-a-bit-calm-down-relax-gif-3234894596200294187)。
   - 这表明社区对平台的发展方向或参与度感到担忧，可能涉及其用户体验和社区支持。
- **Perplexity/Sonar 必须击败 OpenAI/Google**：一位成员推测，如果 **Perplexity/Sonar** 不能成为顶级的基于 Web 的搜索工具，该公司将难以在 **OpenAI 或 Google** 面前保持独特性。
   - 另一位成员指出 *“实际上没人在 Perplexity 上使用 Sonar”*，暗示主要是 Pro 订阅在驱动收入。
- **Gemini Deep Research 节省 10 倍时间**：**Gemini Deep Research** 的最新更新为一位成员节省了 **10 倍** 的时间，但对于 **LMArena** 基准测试来说可能过于昂贵。
   - 另一位成员补充道，Gemini 提供了出色的结果，为科学医学研究提供了深刻的分析，甚至生成了包含 **90 个文献来源** 的列表。
- **LeCun 揭穿 Zuckerberg 的 AGI 炒作**：[Yann LeCun](https://aibusiness.com/responsible-ai/lecun-debunks-agi-hype-says-it-is-decades-away) 警告说，实现 **AGI** *“将需要数年甚至数十年”*，并且需要新的科学突破。
   - 最近的一篇文章称，[Meta 的 LLM 在 2025 年之前不会达到人类水平的智能](https://www.pymnts.com/artificial-intelligence-2/2025/meta-large-language-models-will-not-get-to-human-level-intelligence/?utm_source=chatgpt.com)。
- **Grok 3 的 Deeper Search 令人失望**：一位用户发现 **Grok 3** 的 **Deeper Search** 功能令人失望，理由是存在幻觉和低质量结果。
   - 然而，另一位用户为 **Grok** 辩护，称 *“Deeper Search 看起来相当不错”*，但原评论者反驳说，频繁使用会发现大量的错误和幻觉。



---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gemma 3 量化计算**：一位用户询问在 **M1 Macbook Pro** (16GB) 上运行 **Gemma 3** 的情况，另一位用户解释了如何根据模型大小和以字节为单位的量化来计算内存需求，并建议该 Macbook 可以运行 FP16 格式的 **Gemma 3 4B**。
   - 该用户解释说，考虑到 Macbook 的 **16GB** 统一内存中有 **70%** 分配给 GPU，也可以运行 FP4 格式的 **12B 模型**。
- **Blackwell ULTRA 的 attention 指令引起关注**：一位成员提到 **Blackwell ULTRA** 将带来 *attention 指令*，但其具体含义尚不明确。
   - 此外，成员们讨论了如果 kernel1 的 **smem carveout** 仅为 **100 或 132 KiB**，则没有足够空间让两个 kernel 同时运行，建议参考 [关于共享内存的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x) 增加 carveout。
- **Nvfuser 的 Matmul 输出融合引入停顿**：一位成员指出为 [**nvfuser**](https://github.com/rdspring1) 实现 **matmul 输出融合**（matmul output fusions）非常困难，即使是乘法/加法也会引入停顿（stalls），由于需要保持 tensor cores 的供应，这使得它比独立 kernel 更慢。
   - 另一位成员询问困难是否源于 **Tensor Cores** 和 **CUDA Cores** 无法并发运行，从而可能竞争寄存器使用，同时引用 Blackwell 文档称 **TC** 和 **CUDA cores** 现在可以并发运行。
- **`Accelerate` 准备合并 FSDP2 支持**：一位成员询问 `accelerate` 使用的是 **FSDP1** 还是 **FSDP2**，以及是否可以使用 `trl` 配合 **FSDP2** 微调 LLM。澄清指出，下周左右将合并一个 pull request，以在[此处](https://github.com/huggingface/accelerate/pull/3394)添加对 **FSDP2** 的初始支持。
   - 在该成员澄清了 **FSDP2** 支持添加的具体细节后，另一位成员表示：“这太令人兴奋了！感谢澄清！”，强调了用户对 `accelerate` 中 **FSDP2** 支持到来的期待。
- **DAPO 算法开源发布**：**DAPO 算法**（*decoupled clip and dynamic sampling policy optimization*，解耦裁剪与动态采样策略优化）正式发布；**DAPO-Zero-32B** 超越了 **DeepSeek-R1-Zero-Qwen-32B**，在 **AIME 2024** 上获得 **50** 分，且步数减少了 50%；它是在 **Qwen-32b** 预训练模型基础上通过 **zero-shot RL** 训练的，算法、代码、数据集、验证器和模型已完全开源，基于 [Verl](https://verl.ai/) 构建。
   - 主页见[这里](https://dapo-sia.github.io/)，论文见[这里](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)，代码见[这里](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nvidia 发布 Blackwell Ultra 和 Ruben**：在最近的主题演讲中，**Nvidia** 宣布了 **Blackwell Ultra** 和 **Ruben**，以及名为 **Feynman** 的下一代 GPU。
   - 通过 **Ruben**，**Nvidia** 正在转向**硅光子技术**（silicon photonics）以节省数据传输功耗；**Ruben** 将配备全新的 **ARM CPU**，并对 **Spectrum X** 进行重大投资，后者将推出 **1.6 Tbps 交换机**。
- **CompactDict 因修复 SIMD 问题受到关注**：成员们讨论了 **CompactDict** 的优势，这是一种自定义字典实现，避免了内置 **Dict** 中存在的 **SIMD-Struct-not-supported 问题**。
   - 一年前在 [GitHub](https://github.com/mzaks/compact-dict) 上发布了一份报告，详细介绍了两种专门的 Dict 实现：一种针对 **Strings**，另一种强制实现 **trait Keyable**。
- **HashMap 是否加入 Mojo 标准库引发辩论**：有人建议将 *generic_dict* 作为 **HashMap** 包含在**标准库**中，同时保留当前的 **Dict**。
   - 有人担心 **Dict** 需要处理大量非静态类型（not-static-typed）的操作，增加一个设计更好的新 struct 并随着时间的推移弃用 **Dict** 可能更有价值。
- **List.fill 行为导致意外的长度变化**：用户质疑填充 **lists buffer** 未初始化部分是否应该是可选的，因为调用 **List.fill** 可能会出人意料地改变 list 的**长度**（length）。
   - 建议将填充 lists buffer 未初始化部分设为可选将解决此问题。
- **List 中缺少索引越界检查**：一位用户注意到 **List** 中缺少索引越界（index out of range）检查，表示惊讶，因为他们认为这就是 **unsafe_get** 的用途。
   - 另一位成员也遇到了这个问题，Modular 的人员表示需要在“某个时候”添加该功能。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Patronus AI 在 Etsy 评判诚实度**：[Patronus AI](https://www.patronus.ai/) 推出了 **MLLM-as-a-Judge** 来评估 AI 系统，目前已被 [Etsy](https://www.etsy.com/) 采用，用于验证产品图像说明文字的准确性。
   - Etsy 拥有 *数亿件商品*，需要确保其描述准确且不产生幻觉（hallucinated）。
- **Cognition AI 获得 40 亿美元估值**：**Cognition AI** 在由 Lonsdale 公司领投的一笔交易中达到了 **40 亿美元估值**。
   - 该交易的进一步细节尚未披露。
- **AWS 价格远低于 Nvidia**：在 [GTC 2025 年 3 月主题演讲](https://www.youtube.com/watch?v=_waPvOwL9Z8) 期间，据 [这条推文](https://x.com/_fabknowledge_/status/1902092480616497395) 报道，*AWS 对 Trainium 的定价仅为 Nvidia 芯片 (Hopper) 的 25%*。
   - Jensen（黄仁勋）开玩笑说，在 Blackwell 之后，他们可以免费赠送 Hopper，因为 Blackwell 的性能将非常强大。
- **Manus 访问权限给交易机器人用户留下深刻印象**：一位成员获得了 Manus 的访问权限，该工具通过 **Grok3 进行深度搜索**，并评价其“非常好”，展示了它是如何在周末构建一个交易机器人的，不过目前在模拟交易（paper trading）中亏损了约 1.50 美元。
   - 他们展示了令人印象深刻的输出，并分享了*预览*截图。
- **vLLM：推理界的 ffmpeg**：根据 [这条推文](https://x.com/vllm_project/status/1902068326312124815)，**vLLM** 正在 *逐渐成为 LLM 推理界的 ffmpeg*。
   - 该推文对大家对 **vLLM** 的信任表示感谢。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **微调 Gemini/OLMo 模型变得热门**：成员们正在寻求关于微调 **Gemini** 或 **OLMo** 模型的建议，并考虑蒸馏（distillation）是否是更好的方法，特别是针对 **PDF 文件** 中的数据。
   - 讨论演变为 **内存优化（memory optimization）** 和 **混合设置（hybrid setups）** 以增强性能，而非微调特定模型的细节。
- **Passkey 性能变得模糊**：一位成员建议使用混合方法或由 passkey 激活的 **memory expert**，将重要 key 的 **passkey 和模糊率（fuzzy rate）** 提高到接近 **100%**，如 [scrot.png](https://cdn.discordapp.com/attachments/747850033994662000/1351769065001062420/2025-03-19-040425_768x772_scrot.png?ex=67dc3d4b&is=67daebcb&hm=54777a36b97376a2d0b4b470c683ee5dbd9aedad9f3b2bb4febfe00792f6f6e4&) 所示。
   - 他们指出，更大的模型将拥有更长的记忆，并展示了从 **1.5B** 到 **2.9B 参数** 的改进。
- **Latent Activations 揭示完整序列**：一位发帖者认为，应该从 *整个序列* 而不是单个 token 生成 **latent activations**，以了解模型的正常行为。
   - 他们建议关注 *整个序列* 能更准确地代表 **model** 行为，并提供了示例代码：`latents = get_activations(sequence)`。
- **云端模型需要 API Keys**：成员们询问无法在本地托管的云端模型是否与 **API keys** 兼容。
   - 另一位成员确认它们确实兼容，并指向了之前提供的 [详细信息链接](https://link-to-details)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Hugging Face 教授 LlamaIndex Agents 课程**：[Hugging Face](https://huggingface.co/) 发布了一门关于在 **LlamaIndex** 中构建 Agent 的**免费**课程，涵盖了组件、RAG、工具、Agent 和工作流 ([链接](https://t.co/eACAJzXg8y))。
   - 该课程深入探讨了 **LlamaIndex** Agent 的复杂性，为希望构建 AI 驱动应用程序的开发者提供了实用指南。
- **Google 与 LlamaIndex 简化 AI Agents 构建**：**LlamaIndex** 已与 [Google Cloud](https://cloud.google.com/) 合作，利用 **Gen AI Toolbox for Databases** 简化 AI Agent 的构建 ([链接](https://t.co/ocvPTUxvRO))。
   - **Gen AI Toolbox for Databases** 负责管理复杂的数据库连接、安全和工具管理，更多详情可在 [Twitter](https://twitter.com/llama_index/status/1902387501492584580) 上查看。
- **LlamaIndex 与 Langchain 长期记忆对比**：一位成员询问 LlamaIndex 是否具有类似于 [Langchain 在 LangGraph 中的长期记忆支持](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/) 的功能。
   - 另一位成员指出，“在 Langchain 的情况下，长期记忆只是一个向量存储”，并建议使用 LlamaIndex 的 [Composable Memory](https://docs.llamaindex.ai/en/stable/examples/agent/memory/composable_memory/)。
- **Nebius AI 平台与巨头对比**：一位成员对 **Nebius** 用于 **AI** 和**机器学习工作负载**的计算平台的实际使用体验感到好奇，包括其 **GPU 集群**和**推理服务**。
   - 他们正在将其与 **AWS**、**Lambda Labs** 或 **CoreWeave** 在**成本**、**可扩展性**和**部署易用性**方面进行比较，并希望了解其**稳定性**、**网络速度**以及 **Kubernetes** 或 **Slurm** 等编排工具的情况。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Expanse 32B：寻求知识截止日期**：一位用户询问了 **Cohere Expanse 32B** 的知识截止日期，因为他们正在寻找新工作。
   - 目前尚未提供关于具体截止日期的进一步信息或回复。
- **测试密钥用户遇到速率限制**：一位用户报告其测试密钥遇到了 **429 错误**，寻求关于跟踪使用情况以及如何确定是否超过了 [Cohere 速率限制文档](https://docs.cohere.com/v2/docs/rate-limits) 中描述的**每月 1000 次调用**限制的指导。
   - 一位 Cohere 团队成员提供了帮助，并澄清测试密钥确实受到速率限制（Rate Limits）。
- **Websearch Connector 结果退化**：一位用户报告 **websearch connector** 的性能下降，指出*实现方式最近发生了变化*，现在提供的结果变差了。
   - 一位团队成员请求提供详细信息以便调查，并指出连接选项 *site: WEBSITE* 无法将查询限制在特定网站，该修复程序即将发布。
- **Command-R-Plus-02-2024 与 Command-A-03-2025 对比**：一位用户测试并比较了 **command-r-plus-02-2024** 和 **command-a-03-2025** 模型之间的网页搜索结果，发现模型之间没有显著差异。
   - 他们还报告了多起网页搜索功能未能返回任何结果的情况。
- **Goodnews MCP：LLM 传递正能量新闻**：一位成员创建了一个 **Goodnews MCP** 服务器，通过 **Cohere Command A** 向 MCP 客户端传递正面新闻，该项目已在 [此 GitHub 仓库](https://github.com/VectorInstitute/mcp-goodnews) 开源。
   - 该工具名为 `fetch_good_news_list`，使用 **Cohere LLM** 对近期头条新闻进行排名，以识别并返回最正面的文章。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 课程作业详情发布**：LLM Agents MOOC 的课程作业和结业证书说明已在 [课程网站](https://llmagents-learning.org/sp25) 上发布，该内容基于 [2024 年秋季 LLM Agents MOOC](https://llmagents-learning.org/f24) 的基础。
   - Labs 和证书申报表将于 4 月发布，作业暂定于 5 月底截止，证书将于 6 月发放。
- **AgentX 竞赛设有 5 个等级**：AgentX 竞赛的详情已公布，包括 [此处](https://rdi.berkeley.edu/agentx/) 的报名信息，竞赛包含 **5 个等级**：Trailblazer ⚡, Mastery 🔬, Ninja 🥷, Legendary 🏆, Honorary 🌟。
   - 参与者还可以通过 [此申请表](https://forms.gle/E2D69euNBjSmYsK28) 申请 AgentX 研究赛道（Research Track）项目的导师指导，申请截止日期为 **PDT 时间 3 月 26 日晚上 11:59**。
- **测验截止日期为 5 月底**：所有作业（包括每堂课后发布的测验）的截止日期均为 *5 月底*，因此你仍可以提交作业以获得证书资格。
   - 澄清一点：**AgentX 研究赛道** 的选拔标准 *并非“我们只接受前 X% 的人”*。
- **AgentX 研究赛道项目指导**：研究赛道项目的指导将从 **3 月 31 日持续到 5 月 31 日**，导师将直接联系申请人进行潜在的面试。
   - 申请人应展现出主动性、与课程相关的深思熟虑的研究想法，以及在两个月时间内完成该想法的背景能力。
- **证书问题排查**：一位在 **12 月参加 MOOC 课程** 的成员反映未收到证书，导师回复称 *证书邮件* 已于 **2 月 6 日** 发送，并建议检查垃圾邮件箱并确认使用了正确的电子邮件地址。
   - 导师分享道，证书面向任何完成其中一个课程等级的人开放，并分享了 [课程网站](https://llmagents-learning.org/sp25) 和 [Google 文档](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing) 的链接。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FL 设置终于可以运行**：在经历了因 IT 延迟导致的 **4 个月** 等待后，一位成员的 FL（联邦学习）设置终于可以运行了，展示在 [这张图片](https://i.imgur.com/eX6y0NO.png) 中。
   - 这种长时间的延迟凸显了在大组织中使必要基础设施投入运行所面临的常见挑战。
- **Nvidia GPU 面临长期的供应延迟**：成员们报告了 **Nvidia GPU** 供应的持续延迟，以 **H200s** 为例，该型号在 **2 年前** 宣布，但直到 **6 个月前** 才向客户开放。
   - 此类延迟影响了 AI 项目的开发时间表和资源规划。
- **`recvVector` 和 `sendBytes` 触发 DPO 故障**：用户报告在运行 **DPO recipes** 时出现 `recvVector failed` 和 `sendBytes failed` 错误。
   - 错误的来源尚不确定，可能源于集群问题或 **torch** 的问题。
- **`cudaGetDeviceCount` 兼容性报错**：成员在使用 **NumCudaDevices** 时遇到了 `RuntimeError: Unexpected error from cudaGetDeviceCount()`。
   - 错误 `Error 802: system not yet initialized` 可能源于使用了比预期更新的 **CUDA 版本**，尽管这一点尚未得到证实。
- **`nvidia-fabricmanager` 解决 CUDA 修复问题**：使用 `nvidia-fabricmanager` 是解决 `cudaGetDeviceCount` 错误的方法。
   - 正确的流程包括使用 `systemctl start nvidia-fabricmanager` 启动它，并通过 `nvidia-smi -q -i 0 | grep -i -A 2 Fabric` 确认状态，验证状态显示为 *"completed"*。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **M1 Mac 在模型训练方面表现挣扎**：一位用户报告称，他们的 **M1 Mac Air** 即使在小批量训练模型时也非常吃力，在 **Kaggle** 和 **Hugging Face Spaces** 上遇到了 **clang** 问题。
   - 他们寻求关于在已训练模型上托管推理演示的建议，但发现硬件性能不足以处理基础的训练任务。
- **DeepSeek-R1 针对家庭使用进行优化**：**腾讯玄武实验室的优化方案**允许在消费级硬件上家庭部署 **DeepSeek-R1**，成本仅需 **4万元**，功耗与普通台式机相当，详见[这条推文](https://x.com/programerjohann/status/1901800298458575210)。
   - 该优化配置每秒可生成约 **10个汉字**，与传统的 GPU 配置相比，实现了 **97% 的成本降低**，有可能使强大模型的使用变得大众化。
- **Clang 依赖需要更好的错误处理**：一位贡献者建议，在没有 **clang** 的 CPU 上运行 [mnist 示例](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py)时，应改进针对 `FileNotFoundError` 的依赖验证。
   - 当前的错误消息未能清晰指出缺失 **clang** 依赖的问题，可能会让新用户感到困惑。
- **REDUCE_LOAD 的索引选择令人困惑**：一位成员请求澄清 `x.src[2].src[1].src[1]` 的含义，以及选择这些索引作为 **REDUCE_LOAD pattern** 的 `reduce_input` 的原因。
   - 代码片段检查 `x.src[2].src[1].src[1]` 是否不等于 `x.src[2].src[0]`，并据此将 `x.src[2].src[1].src[1]` 或 `x.src[2].src[1].src[0]` 赋值给 `reduce_input`。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI21 Labs 对 Jamba 保持沉默**：AI21 Labs 目前没有公开分享关于他们用于 **Jamba** 模型开发的技术信息。
   - 一位代表对缺乏透明度表示歉意，但表示如果情况有变，他们会提供更新。
- **社区迎来新面孔**：社区欢迎了几位新成员，包括 <@518047238275203073>, <@479810246974373917>, <@922469143503065088>, <@530930553394954250>, <@1055456621695868928>, <@1090741697610256416>, <@1350806111984422993>, 和 <@347380131238510592>。
   - 鼓励他们参与社区投票以进行互动。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chain Of Draft 的可复现性已实现**：一位成员使用 `dspy.ChainOfThought` 复现了 **Chain of Draft** 技术，并在[一篇博客文章](https://pub.towardsai.net/implementing-chain-of-draft-prompt-technique-with-dspy-ca231c58114f)中详细介绍了该过程。
   - 这验证了使用 DSPy 可靠地复现高级 Prompting 策略的方法。
- **Chain Of Draft 技术减少了 Token 消耗**：**Chain Of Draft** Prompt 技术可以帮助 LLM 在不冗长的情况下扩展其回答，将输出 Token 减少了一半以上。
   - 关于该方法的更多细节可以在[这篇研究论文](https://arxiv.org/pdf/2502.18600v1)中找到。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AWS 网络研讨会教授 MLOps 栈构建**：一场关于 **3月25日太平洋时间上午8点** 举行的网络研讨会将涵盖 **在 AWS 上从零开始构建 MLOps 栈**，可通过[此链接](https://buff.ly/IcPYNyR)注册。
   - 该研讨会旨在深入探讨构建端到端 MLOps 平台。
- **AI4Legislation 研讨会聚焦 Legalese Decoder**：**AI4Legislation 研讨会**邀请了 **Legalese Decoder** 创始人 **William Tsui** 和基金会主席 **Chunhua Liao**，定于 **太平洋时间4月2日下午6:30** 举行（[在此预约](https://forms.gle/pmbkRLVurbXcGBbAA)）。
   - 该研讨会是硅谷华人协会基金会 (SVCAF) 推动 AI 在立法领域应用努力的一部分。
- **Featureform 简化了 ML 模型特征**：**Featureform** 被定位为一个*虚拟特征存储 (virtual feature store)*，使数据科学家能够为他们的 ML 模型定义、管理和提供特征。
   - 它专注于简化 ML 工作流中的特征工程和管理流程。
- **SVCAF 竞赛推动立法领域的开源 AI**：**硅谷华人协会基金会 (SVCAF)** 正在举办一场夏季竞赛，重点是开发**开源 AI 驱动的解决方案**，以增强公民对立法过程的参与（[GitHub 仓库](https://github.com/svcaf/2025-AI4Legislation-Public/)）。
   - 该竞赛旨在促进社区驱动的创新，将 AI 应用于立法挑战。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 默认目录详情**：GitHub 上的 [GPT4All FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 页面描述了 **models** 和 **settings** 的默认目录。
   - 该 GitHub 页面还提供了关于 **GPT4All** 的额外信息。
- **GPT4All Models 的默认位置**：**GPT4All models** 的默认位置在 [FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 中有详细说明。
   - 了解此位置有助于管理和组织 **models**。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。


---

# 第二部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}




### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1351633751435907254)** (1174 条消息🔥🔥🔥): 

> `Sonnet MAX 模型分析，Cursor Plan&Build agent 增强，Claude Max 定价与限制，Open Empathic Project 协作，Windsurf 与 Cursor 定价对比` 


- **Sonnet MAX 展现出强大的 Agent 能力**：正如[这篇 X 帖子](https://x.com/kregenrek/status/1901990102936515040)所提到的，**Sonnet MAX** 模型因其在作为 agent 运行时能够有效处理**后处理（post-processing）**而受到认可。
   - 由于库不在训练数据中，Cursor 需要学习 **code base** 并从**自身的错误**中吸取教训。
- **Max 的使用引发定价争议**：社区讨论了在 **Cursor** 中使用 **Claude Max** 的优点，但对其定价结构表示担忧，因为它没有充分利用订阅中分配的 **fast requests**。
   - 一些成员建议，如果 **Cursor** 允许为 **Max** 消耗更多 fast request，他们就不会寻找替代方案，并表示 *“Cursor 团队在不允许 Max 消耗更多 fast request 方面表现得非常糟糕”*。
- **用户分享编程任务工作流**：一位社区成员详细介绍了他们的工作流，包括：*从 repo prompt 开始，选择相关文件 + 指令 -> 粘贴到 grok 3 以获取高层级计划 -> 将计划交给 claude code 进行 one shot -> 在 cursor agent 中进一步完善细节。*
   - 他们还强调 **Claude max 确实让人感觉拥有了没有任何限制的完整模型**，并称赞 *“先在 figma 中进行设计”* 的做法。
- **终端生成令 Cursor 用户感到沮丧**：用户正苦于 agent 生成**多个终端**并不断重复运行同一个项目，引发了关于如何通过规则和配置防止此问题的讨论。
   - 一位成员提出了一项 *增强型终端管理规则*，其中包括在打开新终端之前运行命令终止已打开的终端，确保测试输出定向到新终端，并防止在单次测试运行期间创建多个终端。
- **开源 IDE PearAI 作为替代方案出现**：社区探索了 **PearAI** ([https://trypear.ai/](https://trypear.ai/))，这是一个集成了 **Roo Code/Cline**、**Continue**、**Perplexity**、**Mem0** 和 **Supermaven** 等工具的开源 AI 代码编辑器。
   - 一位成员表示 “*与 cursor 相比，Pear 现在实际上在做非常有意义的工作*”，他们强调它比 **Cursor** 更便宜，但 **context window** 较小。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://getcoai.com/careers/">CO/AI 招聘 - CO/AI</a>：我们的使命是在 AI 时代赋能人类并培育一个更公平的创新生态系统。如果这一使命引起您的共鸣，我们很乐意收到您的消息。</li><li><a href="https://supabase.com/docs/guides/getting-started/mcp#cursor">模型上下文协议 (MCP) | Supabase 文档</a>：使用 MCP 将 AI 工具连接到 Supabase</li><li><a href="https://x.com/kregenrek/status/1901990102936515040?s=46">来自 Kevin Kern (@kregenrek) 的推文</a>：嗯 - Sonnet MAX 是第一个在运行 Agent 时真正处理好后处理的模型。不幸的是，它是有代价的。引用 Kevin Kern (@kregenrek) 的话：好的，我的 Cursor 计划与构建 Agent 可以配合...</li><li><a href="https://tenor.com/view/mood-dance-russiankiddance-club-dancevibe-gif-21421102">情绪舞蹈 GIF - Mood Dance Russiankiddance - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>：您可以在此处管理您的账户、账单和团队设置。</li><li><a href="https://x.com/boltdotnew/status/1900197121829331158">来自 bolt.new (@boltdotnew) 的推文</a>：推出 Figma to Bolt。从 Figma 到像素级完美的全栈应用 —— 只需在 URL 前加上 bolt․new 并开始提示（prompting）！</li><li><a href="https://tenor.com/view/dance-gif-15809027886002605791">舞蹈 GIF - Dance - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>：未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1jerr41/55_on_claude_sonnet_37_max/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://trypear.ai/">PearAI - 为您下一个项目准备的 AI 代码编辑器</a>：PearAI 是一款开源 AI 代码编辑器，具有 AI 聊天、PearAI Creator 和 AI 调试等强大功能，帮助您创造令人兴奋的作品。</li><li><a href="https://x.com/rahulgs/status/1902342317597511909?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 rahul (@rahulgs) 的推文</a>：让 @anthropicai Claude Code 配合 OpenAI 模型工作了。哈哈，我设置了一个代理服务器，模拟 Anthropic 的 /v1/messages API，将请求转发给 OpenAI。映射关系：- Sonnet 3.7 -> 4o - Haiku 3.5 -> ...</li><li><a href="https://trypear.ai">PearAI - 为您下一个项目准备的 AI 代码编辑器</a>：PearAI 是一款开源 AI 代码编辑器，具有 AI 聊天、PearAI Creator 和 AI 调试等强大功能，帮助您创造令人兴奋的作品。</li><li><a href="https://github.com/1rgs/claude-code-openai">GitHub - 1rgs/claude-code-openai：在 OpenAI 模型上运行 Claude Code</a>：在 OpenAI 模型上运行 Claude Code。通过在 GitHub 上创建账户，为 1rgs/claude-code-openai 的开发做出贡献。</li><li><a href="https://github.com/2-fly-4-ai/V0-system-prompt">GitHub - 2-fly-4-ai/V0-system-prompt</a>：通过在 GitHub 上创建账户，为 2-fly-4-ai/V0-system-prompt 的开发做出贡献。</li><li><a href="https://github.com/daniel-lxs/cursor-plus">GitHub - daniel-lxs/cursor-plus：一个在状态栏显示您的 Cursor 订阅使用统计信息的 Cursor 扩展。</a>：一个在状态栏显示您的 Cursor 订阅使用统计信息的 Cursor 扩展。 - daniel-lxs/cursor-plus</li><li><a href="https://www.reddit.com/r/vibecoding/">Reddit - 互联网的核心</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1351633464759291975)** (419 条消息🔥🔥🔥): 

> `Gemma 3 的 Vision 微调问题，Unsloth 和 vLLM 在旧金山的活动，使用 Wandb 评估模型性能，Base 模型与 Instruct 模型的微调对比，Unsloth 的多节点和多 GPU 微调`

- **Gemma 3 Vision 微调遇到 Transformers 故障**：用户报告了在为视觉任务微调 **Gemma 3** 时遇到的问题，表明当前的 **Transformers** 版本在视觉支持方面可能存在潜在问题。
   - 一位用户在尝试对 `gemma-3-12b-pt-bnb-4bit` 进行 qlora 微调以实现图像描述（image captioning）时，遇到了 *RuntimeError: Unsloth: Failed to make input require gradients!*，这表明需要进一步调查。
- **Unsloth 与 vLLM 联手举办旧金山活动**：Unsloth 将于下周四（3 月 27 日）加入 vLLM 和 Ollama 在旧金山举办的活动，届时将有社交环节和演示。该活动将在 Y Combinator 的旧金山办公室举行。
   - 更多详情请见 [vLLM & Ollama 推理之夜](https://lu.ma/vllm-ollama)，现场将提供食物和饮料。
- **探讨 Wandb 集成与模型评估**：成员们讨论了如何在 **Unsloth** 中使用 **Wandb** 评估微调前后的模型性能，重点关注训练损失（training loss）的追踪。
   - 普遍认为训练过程中训练损失下降是微调奏效的标志，但测试模型并运行基准测试（benchmarks）也至关重要。
- **Base 与 Instruct 模型之争愈演愈烈**：讨论围绕着使用 30k 样本的数据集微调 base 模型还是 instruct 模型是否会产生绝对更好的模型展开，引发了各种不同的观点和考量。
   - 一位成员强调，*如果数据质量差或格式不当，微调可能会降低模型性能*，这也可能导致模型“遗忘”数据集之外的内容。
- **Unsloth 预告多节点多 GPU 支持**：一位团队成员宣布，Unsloth 计划在未来几周内支持多节点（multi-node）和多 GPU 微调，尽管具体的发布日期尚未确定。
   - Unsloth 团队建议[订阅时事通讯](https://unsloth.ai/newsletter)以获取发布更新，但一位成员确认多节点支持将仅限企业版（enterprise only）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth Requirements | Unsloth Documentation</a>: 这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint#wandb-integration">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing 允许你保存微调进度，以便你可以暂停并随后继续。</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: 加入我们的新闻通讯和等待名单，获取关于 Unsloth 的一切！</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 Notebook 的列表：</li><li><a href="https://arxiv.org/abs/2212.04089">Editing Models with Task Arithmetic</a>: 改变预训练模型的行为——例如，提高它们在下游任务中的性能或减轻预训练期间学到的偏差——是开发机器学习模型时的常见做法...</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows Installation | Unsloth Documentation</a>: 了解如何在 Windows 上安装 Unsloth（无论是否使用 WSL）。</li><li><a href="https://huggingface.co/collections/unsloth/mistral-small-3-all-versions-679fe9a4722f40d61cfe627c">Mistral Small 3 (All Versions) - a unsloth Collection</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1902396234884903254">Tweet from Unsloth AI (@UnslothAI)</a>: 我们与 @HuggingFace 合作发布了一个免费的 Notebook，用于使用 GRPO 微调 Gemma 3！学习如何：• 在 Gemma 3 (1B) 中启用推理 • 准备/理解奖励函数 • 让 GRPO 适用于小型 LL...</li><li><a href="https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora">no title found</a>: 未找到描述</li><li><a href="https://x.com/karpathy/status/1835561952258723930">Tweet from Andrej Karpathy (@karpathy)</a>: 当模型在它们的 chain of thought 中停止使用英语时，你就能判断 RL 已经正确完成了。</li><li><a href="https://lu.ma/vllm-ollama">vLLM &amp; Ollama Inference Night · Luma</a>: 欢迎来到 vLLM &amp; Ollama 推理之夜。请于 3 月 27 日星期四下午 6 点在 Y Combinator 的旧金山办公室加入我们！将提供食物和饮料...</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-c">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/155">ValueError during training Mistral7b on VertexAI T4 · Issue #155 · unslothai/unsloth</a>: 我在运行 trainer_stats = trainer.train() 后遇到此错误。它运行在 VertexAI VM 上，配备 T4 GPU 和 16GB RAM。代码是从 Mistral7b 示例 Notebook 中复制/粘贴的。错误是：ValueEr...</li><li><a href="https://github.com/JL-er/RWKV-PEFT">GitHub - JL-er/RWKV-PEFT</a>: 通过在 GitHub 上创建账号来为 JL-er/RWKV-PEFT 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/1558">[Fixing] Better exporting to `llama.cpp` and 16bit merging · Issue #1558 · unslothai/unsloth</a>: 用于跟踪 Unsloth 中更好地导出到 GGUF 格式的 Issue - 目标是将 convert_hf_to_gguf.py 从 llama-quantize 中解耦。如果微调者指定了任何低于 Q8_0 的量化，我们必须使用...</li><li><a href="https://github.com/unslothai/unsloth/issues/1519">Model Merge reduces performance · Issue #1519 · unslothai/unsloth</a>: 我对 Qwen2.5 0.5B 模型进行了持续预训练。当我加载 adapter 并合并它然后进行推理时，输出看起来不错。但是，当我与 adapter 合并并以 16bit 或 4bit 保存模型时...</li><li><a href="https://www.c2-computer.com/products/new-parallel-nvidia-rtx-4090d-48gb-gddr6-256-bit-gpu-blower-edition/">(NEW PARALLEL) NVIDIA RTX 4090D 48GB GDDR6 384-bit Graphics Card *BLOWER EDITION*</a>: 使用全新的 PARALLEL NVIDIA RTX 4090D 48GB GDDR6 GPU 提升您的游戏体验，该显卡采用强大的涡轮散热设计，以实现最佳的冷却和性能。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1351639870786764891)** (3 条消息): 

> `` 


- **Niten 的酷评**: 一位名为 niten 的用户评论了 *Cool*。
- **职位仍然开放吗？**: 一位用户询问如何知道职位是否仍然开放。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1351637361867489290)** (121 条消息🔥🔥): 

> `AMD 对 BnB 和 Triton 的支持、Gemma 基础模型变更、训练 GPTs Agent、OpenAI 的侧边栏、多轮对话数据集 + LLM 微调` 


- **AMD 与 Unsloth：充满前景的合作**：成员们报告称 **BnB 和 Triton** 现在已在 **AMD** 上得到支持，通过对 Unsloth 进行一些修改，它有可能在 AMD GPU 上运行。
   - 虽然可以提供悬赏，但考虑到缺乏测试以及 **AMD** 的未知因素，目前还不愿将其作为最高优先级。
- **Gemma 的陷阱：聊天模板难题**：一位成员注意到对从 HF Unsloth 拉取的 **Gemma** 基础模型应用 LoRA 时出现异常行为，怀疑是聊天模板问题。
   - Theyruinedelise 澄清说 **最近没有对 Gemma 基础模型进行任何更改**，并建议检查聊天模板。
- **SageMaker 设置混乱：依赖困境**：由于与 torchvision 相关的导入问题，用户在 Amazon SageMaker 上运行 **Phi_4-Conversational** 笔记本时遇到了 **RuntimeError**。
   - TheDragondagger 通过参考示例笔记本并按照其中的操作 **卸载/安装库** 解决了该问题，建议在使用 Unsloth 和 SageMaker 时注意依赖关系。
- **显存幻象：在 4090 上追求 VRAM 节省**：尽管有一篇博客文章暗示 20GB VRAM 即可容纳，但用户在 4090 上微调 `unsloth/QwQ-32B-unsloth-bnb-4bit` 时遇到了 **显存溢出 (OOM) 错误**。
   - Theyruinedelise 建议减小 `max_seq_length` 和 batchsize，同时澄清 **20GB 是最小值，因为 GPU 会预留部分 VRAM**，并建议至少准备 22GB。
- **文档遭殃，文件系统问题的救赎**：由于在调用 `save_pretrained_merged` 时仓库 ID 无效，用户在尝试本地保存合并后的模型时遇到了 `HFValidationError` 和 `FileNotFoundError`。
   - 建议是更新 `unsloth-zoo`，因为该问题[应该已在最新版本中修复](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/saving_utils.py#L544-L549)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_">Google Colab</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/elastic/errors.html">错误传播 &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/main/en/package_reference/bone">Bone</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/datasets-101>,">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/2101">TypeError: unsupported operand type(s) for /: &#39;Tensor&#39; and &#39;NoneType&#39; when full finetuning gemma3 · Issue #2101 · unslothai/unsloth</a>：版本 pip install unsloth pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 代码来自 unsloth import FastModel import torch model, tokenizer = FastModel.from_pretrained(...</li><li><a href="https://github.com/ggml-org/llama.cpp/issues/9663">功能请求：添加对 MllamaForConditionalGeneration 的支持，以将 Llama 3.2 Vision 模型转换为 GGUF 格式 · Issue #9663 · ggml-org/llama.cpp</a>：前提条件 我正在运行最新代码。如果可能请注明版本。我仔细阅读了 README.md。我使用与问题相关的关键词进行了搜索，以确保...</li><li><a href="https://github.com/unslothai/unsloth/issues/1670">保存并加载 LoRA 模型后推理结果不一致 · Issue #1670 · unslothai/unsloth</a>：我正按照官方 Google Colab 笔记本在 Paperspace 上训练我的推理 LoRA 模型。我能够毫无问题地运行笔记本。但是，当尝试保存训练好的模型时...</li><li><a href="https://github.com/huggingface/transformers.git">GitHub - huggingface/transformers: 🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供的最先进机器学习库。</a>：🤗 Transformers: 为 Pytorch, TensorFlow 和 JAX 提供的最先进机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/unslothai/unsloth/issues/2098,">unslothai/unsloth</a>：微调 Llama 3.3, DeepSeek-R1, Gemma 3 和推理 LLMs，速度提升 2 倍，显存占用减少 70%！ 🦥 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/saving_utils.py#L544-L549)">unsloth-zoo/unsloth_zoo/saving_utils.py at main · unslothai/unsloth-zoo</a>：Unsloth 工具库。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1351995007325966496)** (1 messages): 

> `Unsloth mention` 


- **Unsloth 获得点名**：[Unsloth](https://github.com/unslothai) 在[这篇文章](https://substack.com/@migueloteropedrido/note/c-101152792?r=58depg)中被提及。
- **Unsloth 赢得认可**：Substack 上的一篇文章指出了 Unsloth 在 AI 领域的重要性。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1351652781441810493)** (3 messages): 

> `ZO2 Framework, Zeroth Order Optimization, RWKV-7 Model` 


- **ZO2 Framework 宣称可在普通硬件上对 175B LLM 进行全参数微调**：[ZO2 framework](https://github.com/liangyuwang/zo2) 仅需 **18GB** 的 GPU 显存即可实现 **175B LLMs** 的全参数微调。
   - 它*特别针对 GPU 显存有限的配置进行了优化。*
- **深入探讨 Zeroth Order Optimization**：一名成员指出 **ZO2** 采用了 [Zeroth Order Optimization](https://huggingface.co/papers/2503.14456)，这与 **SGD** 等更常见的一阶方法形成对比。
   - Zeroth-order 方法通过函数评估来近似梯度，使其适用于高维或不可微问题。
- **RWKV-7 模型亮点**：该频道分享了 Hugging Face 上的 [fla-hub/rwkv7-1.5B-world](https://huggingface.co/fla-hub/rwkv7-1.5B-world) 模型，这是一个最近更新的文本生成模型。
   - 他们还附带了一个 [YouTube 视频](https://www.youtube.com/watch?v=xT4jxQUl0X8)，其中可能讨论了该模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2503.14456">Paper page - RWKV-7 &quot;Goose&quot; with Expressive Dynamic State Evolution</a>：未找到描述</li><li><a href="https://github.com/liangyuwang/zo2">GitHub - liangyuwang/zo2: ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory</a>：ZO2 (Zeroth-Order Offloading): Full Parameter Fine-Tuning 175B LLMs with 18GB GPU Memory - liangyuwang/zo2
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1351632211698384906)** (115 条消息🔥🔥): 

> `Leaderboards, OpenVoice, LM Studio User Guide, Oblix Project, 4090 and pcie` 


- ****HF Leaderboards 更新****：成员们分享了来自 [Hugging Face](https://huggingface.co/spaces/mike-ravkine/can-ai-code-results) 的 **AI 编程模型** 最新排行榜。
   - 他们还分享了另一个来自 [BigCode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) 的排行榜。
- ****OpenVoice 克隆声音****：一位成员分享了 [OpenVoice](https://research.myshell.ai/open-voice)，这是一种多功能的即时语音克隆方法，仅需一段简短的音频剪辑即可复制声音并生成多种语言的语音。
   - 除了复制参考说话者的音色外，它还能对语音风格进行细粒度控制，包括情感、口音、节奏、停顿和语调，详见其 [GitHub 仓库](https://github.com/myshell-ai/OpenVoice)。
- ****LM Studio 设置用户指南已创建****：一位成员正在编写 **LM Studio 用户指南**，重点关注针对其 PC 的各项设置。
   - 如果效果良好，他们将为社区分享一个通用版本。
- ****Oblix 编排本地与云端模型****：一位成员介绍了 [Oblix Project](https://oblix.ai/)，这是一个本地与云端模型之间的无缝编排平台，并在 [演示视频](https://youtu.be/j0dOVWWzBrE) 中进行了展示。
   - Oblix 根据复杂性、延迟要求和成本考虑，智能地将 AI 任务定向到云端或边缘。
- ****4090 PCIE 带宽并不重要****：成员们讨论了设置 **双 4090** 的问题，以及 **PCIE 5.0** 相比 **PCIE 4.0** 是否有提升。
   - 结论是 PCIE 带宽不会显著影响推理速度，与 PCI-e 4.0 x8 相比，最多只能增加 2 tps。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://research.myshell.ai/open-voice">OpenVoice: Versatile Instant Voice Cloning | MyShell AI</a>: 探索 OpenVoice：从短音频剪辑中复制声音的即时语音克隆技术。支持多种语言、情感和口音控制以及跨语言克隆。高效且...</li><li><a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - mike-ravkine 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models Leaderboard - bigcode 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://oblix.ai/">Transform Your AI Performance with Intelligent Hybrid Orchestration | Oblix.ai</a>: 体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效益。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1351640813108596827)** (271 条消息🔥🔥): 

> `NVIDIA Digit 定价，5090 带宽对比 M4 Max，小模型下 NPU 对比 iGPU，多 GPU 性能问题，NVIDIA RTX PRO 6000 Blackwell` 


- **NVIDIA DIGITS 定价低于 Mac Mini M4 Pro**：成员们讨论了 NVIDIA DIGITS 的定价，发现其与配置相似但内存减半的 **Mac Mini M4 Pro** 相当，尽管 DIGITS 搭载了 Blackwell GPU 并拥有 273 GB/s 的带宽。
   - 针对 DIGITS 的价值（尤其是其有限的带宽）提出了质疑，有人认为其性价比甚至不如 M4 机器或 Ryzen 395+。
- **5090 带宽瓶颈？**：一位用户观察到，运行 Gemma 3 27B q4 时，**RTX 5090** 达到约 40 tok/s，而 **M4 Max** 达到约 20 tok/sec，考虑到 5090 拥有更强的算力和内存带宽，这一结果出人意料。
   - 其他人指出，5090 在推理过程中并未达到最大功耗，切换到 **Vulkan runtime** 可以提升性能，这暗示瓶颈可能在于 CPU 或软件层面，而非硬件限制。其他人也分享了展示类似结果的 [YouTube 链接](https://youtu.be/8_pw7mKmaLw?t=700)。
- **NPU 缺乏支持**：成员们确认目前 llama.cpp 层面尚不支持 **NPU**，因此 **iGPU** 的表现会优于 40 TOPS 的 NPU。
   - 还有建议指出，如果 GPU 提供 800 TOPS 而 NPU 仅提供 50 TOPS，那么 NPU 是否支持也就不重要了。
- **多 GPU 配置可能会降低性能**：一位用户报告称，在 LM Studio 中使用 CUDA llama.cpp 运行多个 **RTX 3060**（3 个在 PCI-e x1 上，1 个在 x16 上）时，性能出现下降且不稳定，单 GPU（x16）的性能反而更好。
   - 有人建议，性能问题源于模型在多个 GPU 之间的切分方式，且 PCI-e 3.0 x4 会使推理速度降低多达 10%。
- **NVIDIA 发布 RTX PRO 6000 Blackwell**：NVIDIA 发布了 [RTX PRO 6000 "Blackwell" GPU](https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai)，该显卡采用 GB202 GPU，拥有 24K 核心、96 GB VRAM，功耗需求为 600W TDP。
   - 由于配备了 HBM，该显卡的性能被认为远超 5090。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-unveils-dgx-station-workstation-pcs-gb300-blackwell-ultra-inside">Nvidia 发布内置 GB300 Blackwell Ultra 的 DGX Station 工作站 PC</a>：也就是 AI 工作站。</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers>">新闻存档</a>：未找到描述</li><li><a href="https://tenor.com/view/madturnip-show-super-mario-bros-gif-19309485">Madturnip Show GIF - Madturnip Show Super - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://wccftech.com/nvidia-rtx-pro-6000-blackwell-launch-flagship-gb202-gpu-24k-cores-96-gb-600w-tdp/">NVIDIA RTX PRO 6000 "Blackwell" 系列发布：旗舰级 GB202 GPU，24K 核心，96 GB VRAM，最高 600W TDP</a>：NVIDIA 正式发布了针对专业消费者和服务器领域的 RTX PRO 6000 "Blackwell" 系列 GPU，动力强劲。</li><li><a href="https://wccftech.com/nvidia-rtx-pro-6000-blackwell-launch-flagship-gb202-gpu-24k-cores-96-gb-600w-td">NVIDIA RTX PRO 6000 "Blackwell" 系列发布：旗舰级 GB202 GPU，24K 核心，96 GB VRAM，最高 600W TDP</a>：NVIDIA 正式发布了针对专业消费者和服务器领域的 RTX PRO 6000 "Blackwell" 系列 GPU，动力强劲。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1351646277099585692)** (204 条消息🔥🔥): 

> `本地 AI 家庭服务器搭建，学习 LLM/Agent 的书籍，Mistral 模型，年龄验证应用，文本纠错工具`

- **矿卡成为本地 AI 的救星**：一位用户正在构建本地 AI 家庭服务器，并在 **Radeon RX 580 GPUs** 和其他选项之间犹豫不决，寻求同价位中具有更多 VRAM 的 GPU。其他用户建议选择 **P104-100s 或 P102-100s**，它们分别配备 8GB 和 10GB 的 VRAM，并且可以在阿里巴巴上找到。
- **Pascal GPU 显存被中国卖家破解**：**Nvidia** 在 **P104-100s 和 P102-100s** 的 BIOS 中限制了 VRAM，以防止其用于机器学习和其他 GPGPU 任务，但中国卖家通过刷入固件解锁了所有可用显存。
- **NVIDIA 发布 DGX Spark**：NVIDIA 发布了由 NVIDIA Grace Blackwell 平台驱动的 **DGX Spark** 个人 AI 超级计算机，拥有 **128GB VRAM** 和 336gb/s 带宽，但速度比 2060 慢，售价约为 **4000 美元**。
   - 新发布的 **DGX Spark** 代号为 project DIGITS，旨在让 AI 开发者、研究人员、数据科学家和学生在桌面上进行大模型的原型设计、Fine-tune 和 Inference。
- **开源投资成倍增加价值**：哈佛大学的研究发现，*在开源领域投入的 41.5 亿美元为公司创造了 8.8 万亿美元的价值*，相当于 **每投入 1 美元 = 创造 2000 美元的价值**，正如[这篇 X 帖子](https://x.com/ClementDelangue/status/1901765997696008307?t=0s9dSKc6E5S4wJf7TzT1Dw&s=19)中所讨论的那样。
- **T5 模型语法大师**：对于本地文本纠错，**T5** 模型是一个很好的解决方案，特别是针对语法纠错进行过 Fine-tune 的版本 [vennify/t5-base-grammar-correction](https://huggingface.co/vennify/t5-base-grammar-correction)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers">NVIDIA 发布 DGX Spark 和 DGX Station 个人 AI 计算机</a>：NVIDIA 今日推出了由 NVIDIA Grace Blackwell 平台提供支持的 NVIDIA DGX™ 个人 AI 超级计算机。</li><li><a href="https://huggingface.co/spaces/edwardthefma/AgeVault">AgeVault - edwardthefma 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/vennify/t5-base-grammar-correction">vennify/t5-base-grammar-correction · Hugging Face</a>：未找到描述</li><li><a href="https://www.google.com/aclk?sa=L&ai=DChcSEwi88Iem65SMAxWJklAGHdPFKcMYABAOGgJkZw&co=1&gclid=Cj0KCQjws-S-BhD2ARIsALssG0Yki8vYtNKsOvZEdOcbUiRvj6vbqmepu72YZMHT4yzh5b1uA6EcXegaAm6WEALw_wcB&sph=&cce=1&sig=AOD64_3Hk_zyiAcJ5bTCJhV9lRApw__mjw&ctype=5&q=&ved=2ahUKEwjYioOm65SMAxU4U0EAHePbCRwQwg8oAHoECAUQDQ&adurl=">未找到标题</a>：未找到描述</li><li><a href="https://x.com/ClementDelangue/status/1901765997696008307?t=0s9dSKc6E5S4wJf7TzT1Dw&s=19">来自 clem 🤗 (@ClementDelangue) 的推文</a>：在开源领域投入 1 美元 = 为公司/国家创造 2,000 美元的价值！Stargate 又是多少钱来着？引用 clem 🤗 (@ClementDelangue) 关于 @Harvard 对开源的伟大研究：41.5 亿美元投入到开源...</li><li><a href="https://huggingface.co/models?sort=trending&search=text+sql">Models - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces?sort=trending&search=text+sql">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://oblix.ai/">通过智能混合编排提升您的 AI 性能 | Oblix.ai</a>：体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效率。</li><li><a href="https://huggingface.co/chat/).">HuggingChat</a>：让每个人都能使用社区最好的 AI 聊天模型。</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/">Open LLM Leaderboard - open-llm-leaderboard 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/gaia-benchmark/leaderboard">GAIA Leaderboard - gaia-benchmark 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces?sort=trending&search=leaderboard">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/">构建 PDF 摄取和问答系统 | 🦜️🔗 LangChain</a>：本指南假设您熟悉以下概念：</li><li><a href="https://github.com/pymupdf/PyMuPDF">GitHub - pymupdf/PyMuPDF: PyMuPDF 是一个高性能 Python 库，用于 PDF（及其他）文档的数据提取、分析、转换和操作。</a>：PyMuPDF 是一个高性能 Python 库，用于 PDF（及其他）文档的数据提取、分析、转换和操作。 - pymupdf/PyMuPDF</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/sparkle1111/soudesune-shirankedo-7b-instruct">sparkle1111/soudesune-shirankedo-7b-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/grammarly/medit-xl">grammarly/medit-xl · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1351645708507414550)** (2 条消息): 

> `Stochastic Variational Inference, Inference Algorithm, Reparameterization` 


- **引入随机变分推理算法 (Stochastic Variational Inference Algorithm)**：一篇关于如何在具有难以处理的后验分布的连续隐变量的有向概率模型中进行高效推理和学习的论文，介绍了一种可扩展至大规模数据集的**随机变分推理与学习算法**。
   - 这篇题为 *Auto-Encoding Variational Bayes* 的论文可以从 [此处获取 PDF](https://arxiv.org/pdf/1312.6114)。
- **重参数化产生下界估计器**：论文表明，**变分下界的重参数化 (reparameterization)** 会产生一个下界估计器，该估计器可以使用标准的随机梯度方法直接进行优化。
   - 论文还指出，通过使用所提出的下界估计器将近似推理模型拟合到难以处理的后验分布，可以使后验推理变得高效。



**提到的链接**：<a href="https://arxiv.org/abs/1312.6114">Auto-Encoding Variational Bayes</a>：在存在难以处理的后验分布的连续隐变量以及大规模数据集的情况下，我们如何在有向概率模型中进行高效的推理和学习？我们...

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1351654726109761576)** (13 messages🔥): 

> `Gemma-3 Spaces, Gemini 图像编辑, Oblix AI 编排, Road Rash 重制版, 年龄验证应用` 


- ****Gemma-3 系列** Spaces 发布！**: 一位成员分享了他们为多模态 **gemma-3-12b-it** 和 **gemma-3-4b-it** 模型创建的 [Hugging Face Space](https://huggingface.co/spaces/merterbak/gemma-3)。
   - 该 Space 展示了这些新模型的功能和潜在应用场景。
- ****Gemini** 助力图像编辑**: 一位成员利用 **Gemini** 原生图像生成 API 创建了一个简单的 **Gradio** 界面，可在 [此 Hugging Face Space](https://huggingface.co/spaces/saq1b/gemini-image-editing) 访问。
   - 该界面允许用户使用强大的 **Gemini** 模型轻松编辑图像。
- ****Oblix** 编排本地与云端模型**: 一位成员介绍了 **Oblix Project**，这是一个用于本地和云端模型之间无缝编排的平台，并在 [YouTube](https://youtu.be/j0dOVWWzBrE) 上提供了演示视频。
   - **Oblix** 中的自主 **Agent** 会监控系统资源，并动态决定是在本地还是在云端执行 AI 任务。
- ****Road Rash** 经典回归！**: 一位成员宣布推出了经典游戏 **Road Rash** 的现代重制版，可在移动端和桌面端运行，访问地址为 [r3-game.vercel.app](https://r3-game.vercel.app/)。
   - 该游戏具有在线多人游戏、战斗机制，并且是使用 **Claude** 进行 *vibe coded* 的。
- **AgeVault 应用面临身份验证争议**: 一位成员请求对其新的年龄验证应用 [AgeVault](https://huggingface.co/spaces/edwardthefma/AgeVault) 提供反馈，该应用旨在允许 Discord 服务器验证用户年龄。
   - 另一位成员对用户向该应用上传身份证件表示担忧，建议在 NSFW 服务器中更倾向于通过 Ticket 进行人工检查。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://r3-game.vercel.app/">R3</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/merterbak/gemma-3">Gemma 3 - merterbak 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/edwardthefma/AgeVault">AgeVault - edwardthefma 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/saq1b/gemini-image-editing">Gemini Image Editing - saq1b 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/NathanielEvry/LLM-Token-Vocabulary-Analyzer">GitHub - NathanielEvry/LLM-Token-Vocabulary-Analyzer: 揭示 AI 语言模型词汇表中缺失的内容。</a>: 揭示 AI 语言模型词汇表中缺失的内容。 - GitHub - NathanielEvry/LLM-Token-Vocabulary-Analyzer</li><li><a href="https://oblix.ai/">通过智能混合编排提升您的 AI 性能 | Oblix.ai</a>: 体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云提供商之间无缝切换，以实现最佳性能和成本效益。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1351984992275796091)** (2 messages): 

> `Women in AI & Robotics, 语言模型辩论, AI 读书会环节` 


- ****Women in AI** 读书会回归**: **Women in AI & Robotics** 小组将在暑假前再举办三场 **AI 读书会** 活动，其中包括与来自 **Google DeepMind** 的 Yilun Du 讨论**语言模型**如何相互辩论。
   - 该 [活动](https://discord.com/events/879548962464493619/1351984543376085062) 将讨论使用**语言模型**来完善其推理并提高其准确性。
- **即将举行的语言模型辩论环节**: 来自 **Google DeepMind** 的 Yilun Du 将介绍**语言模型**如何通过相互辩论来完善推理并提高准确性。
   - 此环节是 **Women in AI & Robotics AI 读书会**的一部分，计划在暑假前举行。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1351905565609033770)** (1 messages): 

> `OpenAssistant Dataset Release 2 (OASST2), LLM post-training` 


- ****OASST2** 数据集结构公开**: **Open Assistant Conversations Dataset Release 2 (OASST2)** 现已发布，其特点是包含具有初始 prompt 消息以及交替的 'assistant' 或 'prompter' 角色的消息树。
   - 该数据集包含截至 **2023 年 11 月 5 日**收集的数据，并组织成消息树，其中角色从 prompt 到叶节点严格在 'prompter' 和 'assistant' 之间交替；更多详情可以在 [这里](https://huggingface.co/datasets/OpenAssistant/oasst2) 找到。
- ****GPT2-mini** 后训练所需的 Epoch 数量**: 一位成员询问了使用 **OpenAssistant** 数据对像 **GPT2-mini** 这样的小规模 LLM 进行后训练所需的足够 Epoch 数量。
   - 他们还询问了是否有类似于 **HellaSwag** 的基准测试，能够显示 Assistant 在训练期间的早期改进迹象。



**Link mentioned**: <a href="https://huggingface.co/datasets/OpenAssistant/oasst2">OpenAssistant/oasst2 · Datasets at Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1351835859401310219)** (2 messages): 

> `Gradio Sketch AI-powered code generation, Gradio Dataframe Overhaul, Multi-cell selection & copy, Column freezing & row numbers, Search & filter functions` 


- **Gradio Sketch 现在支持 AI 驱动的代码生成**: Gradio Sketch 发布了一个更新，其中包括针对事件函数的 **AI 驱动代码生成**，因此过去需要数小时编写的代码现在可以在 *不写一行代码* 的情况下在几分钟内完成。
   - 这是通过 AI 对组件值的理解实现的，并根据视觉设计生成适当的函数代码，可以通过输入 `gradio sketch` 或访问 [托管版本](https://huggingface.co/spaces/ysharma/Sketch) 来使用。
- **Gradio 的 Dataframe 升级派对！**: Gradio 发布了大规模的 **Dataframe 重构**，包括 **多单元格选择与复制**、**列冻结与行号**、**搜索与过滤功能**以及**全屏模式**。
   - 升级现已上线，可以通过 `pip install --upgrade gradio` 进行更新，[Huggingface Blog](https://huggingface.co/blog/hmb/gradio-dataframe-upgrade) 也已更新以引导你完成这些史诗级的升级。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ysharma/Sketch">Sketch - a Hugging Face Space by ysharma</a>: no description found</li><li><a href="https://huggingface.co/blog/hmb/gradio-dataframe-upgrade">Gradio’s Dataframe has been upgraded! 🎨</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1351883619614064690)** (4 messages): 

> `Pushing Tools to Hugging Face Hub, Issues with Hugging Face Course, VS Code integration` 


- **在本地将 Tools 推送到 HF Hub**: 一位成员询问如何从 VS Code 使用 `agent.push_to_hub` 函数在不使用 notebook 的情况下将 Tools 本地推送到 Hugging Face Hub。
- **Hugging Face 课程引发批评**: 一位成员对某门 Hugging Face 课程表示强烈不满，称 *课程制作者只是在敷衍了事*，并且 *其中一个单元甚至连 pip 安装都无法运行，更不用说下一单元格的导入了*。
   - 他们进一步批评该课程明显缺乏测试，并推测创作者缺乏责任心和经验。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1351634995869126730)** (39 条消息🔥): 

> `Ollama 集成、Unit 2.3 LangGraph 资料、第一个 Agent 模板失败` 


- **深入探讨 Ollama 集成的细微差别**：一位成员建议使用 `ollama/<model_name>` 以获得更多自由度，并指出 `ollama_chat` 可能会访问不同的端点（可能是 `/api/chat`），而 `ollama` 使用的是 `/api/generate`。
   - 有人提到，区别在于 prompt 的格式化方式以及发送给 **LLM** 的方式，并强调了一个已关闭的 [issue](https://github.com/huggingface/smolagents/issues/551)，该 issue 显示了集成的进展。
- **Unit 2.3 LangGraph 资料已在 GitHub 上线**：由于网站存在同步问题，Unit 2.3 关于 LangGraph 的资料已发布在 [GitHub 仓库](https://github.com/huggingface/agents-course/tree/main/units/en/unit2/langgraph)中。
   - 这使得“心急”的用户可以在网站更新前访问内容，确保他们能继续课程学习。
- **第一个 Agent 模板反复失败**：一位成员报告第一个 Agent 模板反复失败，由于 **Qwen/Qwen2.5-Coder-32B-Instruct** 模型超出 token 限制，收到 **422 Client Error**。
   - 另一位成员建议使用替代模型或提供的 **Hugging Face Endpoint**，因为可能存在过载，并推荐使用端点 `'https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'">未找到标题</a>：未找到描述</li><li><a href="https://steanmscommnuity.com/105395109">Steam 礼品激活</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/agents-course/First_agent_template/discussions/234#67dabc265beaf9f0c66e2c73">agents-course/First_agent_template · 我的第一个 agent 模板错误</a>：未找到描述</li><li><a href="https://github.com/huggingface/agents-course/tree/main/units/en/unit2/langgraph">GitHub 上的 agents-course/units/en/unit2/langgraph</a>：此仓库包含 Hugging Face Agents 课程。 - huggingface/agents-course</li><li><a href="https://huggingface.co/learn/agents-course/en/unit1/dummy-agent-library">虚拟 Agent 库 - Hugging Face Agents 课程</a>：未找到描述</li><li><a href="https://github.com/huggingface/smolagents/issues/551">LiteLLM ollama bug 更新 · Issue #551 · huggingface/smolagents</a>：正如在 #406 中要求的，这里是目前 ollama 的状态以及复现代码。简而言之：如果用户在使用 ollama 时遇到困难，请尝试使用 ollama/modelname 而不是 ollama_chat...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1351916134131175539)** (3 条消息): 

> `R1 蒸馏、使用 R1 进行基座模型训练` 


- **新手提问：什么是 R1 蒸馏（R1 Distills）？**：一位成员寻求适合初学者的资源，以了解 **R1 distills** 及其用途。
- **将 Open-R1 集成到基座模型训练循环中**：一位成员询问如何将 **open-R1** 集成到基座模型（Foundation Model）的训练循环中，而不仅仅是蒸馏现有模型。
   - 他们正考虑制作一个精简版的基座模型，并且手头拥有一些批量的 **R1 671B** 计算资源。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1351636650861658142)** (188 条消息🔥🔥): 

> `Gemini vs ChatGPT, o1 与 o3-mini-high 模型对比, GPT-4.5 创意写作, DeepSeek 被禁` 


- **Gemini 与 ChatGPT 正面交锋**：成员们辩论了 [Gemini Advanced](https://gemini.google.com/) 和 [ChatGPT Plus](https://chat.openai.com/) 哪个更好，一位用户认为 **Gemini** 免费的 **2.0 Flash Thinking** 优于许多 **ChatGPT** 模型，理由是其具有*无限*访问权限，而 **ChatGPT** 则有限制。
   - 然而，另一位用户表示 **Gemini** *甚至不具备基础能力*，指的是其安全对齐、内容标记和整体实现。
- **o1 和 o3-mini-high 依然占据统治地位**：尽管有关于 **Gemini** 和其他模型的讨论，一些成员认为 **OpenAI** 的 **o1** 和 **o3-mini-high** 模型在需要推理的任务（如编程、规划和数学等 **STEM** 任务）中仍然是最好的，并指出在他们的经验中没有其他模型能与之媲美。
   - 他们补充说，**Google** 在所有模型中表现最差，尤其是在这些领域，而 **Grok 3** 和 **Claude 3.7 Thinking** 是仅有的可能在质量上接近的模型。
- **GPT-4.5 创意写作令人失望**：一位成员发现 **GPT-4.5** 的创意写作能力令人失望，称该模型的表现不稳定，有时会忽略上下文、出现逻辑错误并重复短语，听起来有时像 **GPT-4 Turbo**。
   - 他们补充说，第二次运行的结果出奇地好，但极端的短消息限制非常荒谬。
- **DeepSeek 被大学禁用**：一位成员报告说他们的大学禁止使用 **DeepSeek**，另一位用户确认该禁令仅适用于 **DeepSeek**，而不适用于其他 **LLM**。
   - 禁令的一个原因可能是因为该模型本身没有实际的指南或过滤器，因为为了规避*非法*话题而束手束脚会大幅降低性能。



**提到的链接**：<a href="https://g.co/gemini/share/bc7bb49815ad">‎Gemini - Correction d&#39;une évaluation de géométrie
</a>：使用 Gemini Advanced 创建

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1351814437027315722)** (4 条消息): 

> `模型访问, 删除对话错误, 代码中插入表情符号` 


- **排查模型访问问题**：一位用户报告称，尽管更改了密码、启用了 **2FA** 并从所有设备注销，但在访问模型时仍遇到问题。
   - 收到的错误消息是 *"Content failed to load"*（内容加载失败），并带有 *"Try again"*（重试）选项。
- **删除错误困扰用户**：一位用户在尝试删除对话时遇到错误。
   - 未说明具体的错误消息，但这导致他们无法按预期删除对话。
- **代码生成中的表情符号令人愤怒**：一位用户对 **ChatGPT** 在代码中插入表情符号表示沮丧，即使在明确指示避免这样做之后也是如此。
   - 用户指出，尽管有提醒和自定义设置，**ChatGPT** 仍继续添加表情符号，这对于维护整洁的代码库可能会产生问题。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1351636469470593115)** (18 条消息🔥): 

> `ChatGPT Personalizations, Unhelpful Assistant, GPT-4o Sandbox Testing, Mixing Helpful Unhelpfulness` 


- **探索 ChatGPT 的个性**：成员们讨论了他们在 **ChatGPT personalizations** 方面的经验，探索模型如何响应不同的 Prompt 和系统消息，并通过图片附件展示了 **"unhelpful assistant"**（无助助手）角色扮演的示例。
   - 一位成员发现，在不更改系统消息的情况下，很难让 **GPT-4o sandbox** 中的模型脱离这种 *"unhelpful"* 状态。
- **掌握“有益的无助感” (Helpful Unhelpfulness)**：一位用户分享了他们如何在个性化设置中混合 *"helpful unhelpfulness"*，包括如何处理此类请求，并提供了 [chatgpt conversation](https://chatgpt.com/share/67da55ad-59ac-8011-abd0-927a6de09c8c) 的链接。
   - 模型同意这就是其目的，且这些内容均未超出允许的范围。其中没有任何内容是真正带有恶意的。
- **GPT-4o 探测其边界**：一位成员分享了他们对 [GPT4o](https://openai.com/index/hello-gpt-4o/) 的实验，通过设置负面人格来观察模型能被推到什么程度。
   - 他们注意到了 *"ChatGPT 正常光芒下的阴暗面"*，以及由于外部施加的 Alignment（对齐）而导致维持 *"unhelpful"* 人格所面临的挑战。
- **API 成本担忧**：一位用户对使用 **API** 进行大规模测试可能产生的成本表示担忧，并担心自己会对尝试新事物产生**成瘾性**从而失去自我控制。
   - 另一位成员提到使用 **4o model** 的花费不到一美元，但也承认在探索过程中容易过度投入。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1351636469470593115)** (18 条消息🔥): 

> `ChatGPT Personalization, Unhelpful Assistant, GPT-4o Behavior, API Cost, Addiction` 


- **探索无助的 ChatGPT 助手**：成员们分享了他们对 **ChatGPT** 的探索，特别是专注于创建和与 *unhelpful* 助手互动。
   - 目标是观察助手如何*演变*，以及是否能在不更改系统消息的情况下将其从指定的 *unhelpful* 状态中*拉出来*。
- **Sandbox 对决：测试系统角色**：一位成员向社区发起挑战，要求在 sandbox 环境中尝试将 **GPT-4o** 从其作为 *unhelpful assistant* 的系统角色中*拉出来*。
   - 意图是在不直接修改系统消息的情况下，让模型恢复到一种*明显并非无助*的状态。
- **有益的无助感：一种个性化利基**：一位成员表示，他们很享受在 **ChatGPT** 个性化设置中探索如何*混合有益的无助感 (helpful unhelpfulness)*。
   - 他们专注于模型应如何处理来自他们的此类请求，并将这一概念整合到他们的通用个性化方法中。
- **API 成本担忧**：一位成员担心潜在的 **API costs** 可能会超过正常的 **ChatGPT** 订阅费用。
   - 另一位成员提到使用 4o 模型花费不到一美元，同时也承认在探索新功能时可能存在*缺乏自控*和*成瘾*的风险。
- **趣味性与成瘾潜力**：成员们讨论了实验 **AI models**（特别是 **ChatGPT**）的趣味性和成瘾性。
   - 一位成员承认会定期检查他们的使用情况，以避免因过度实验而变得像 *Rip Van Winkle 那样衰老且贫困*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1351744341881655399)** (1 条消息): 

> `Anthropic Downtime, Claude 3.7 Sonnet Issues` 


- **Anthropic 模型遭遇停机**：Anthropic 模型经历了停机，特别是 **Claude 3.7 Sonnet**。
- **Anthropic 服务正在恢复**：成员们注意到 **Anthropic** 服务在故障后似乎正在恢复。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1351683758796832883)** (3 条消息): 

> `Claude 3.5 Sonnet, OpenRouterGo SDK, Gemini 2.0 Pro EXP 02-05` 


- **社区在 Cline 兼容性排行榜上为模型评分**：一名成员为模型创建了一个 [Cline Compatibility Board](https://cline-compatibility-board.vercel.app/)，根据它们在 **Cline** 中的表现进行排名。
   - 该排行榜包含了 **API providers**、**plan mode**、**act mode**、输入/输出成本以及 **Claude 3.5 Sonnet** 和 **Gemini 2.0 Pro EXP 02-05** 等模型的最大输出等详细信息。
- **OpenRouterGo SDK v0.1.0 上线**：一名成员宣布发布 **OpenRouterGo v0.1.0**，这是一个用于访问 **OpenRouter API** 的 [Go SDK](https://github.com/eduardolat/openroutergo)，具有简洁、流畅的接口。
   - 该 SDK 包含自动模型回退（fallbacks）、function calling 以及 **JSON** 响应验证。
- **Gemini 2.0 Pro EXP-02-05 存在故障和速率限制**：**OpenRouter** 上的 **Gemini-2.0-pro-exp-02-05** 模型已确认可用，但会出现*随机故障*和*速率限制（rate limiting）*。
   - 根据兼容性排行榜，该模型目前为 **0 成本**，输出限制为 **8192**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cline-compatibility-board.vercel.app/">Cline Compatibility Board</a>: 未找到描述</li><li><a href="https://github.com/eduardolat/openroutergo">GitHub - eduardolat/openroutergo: Easy to use OpenRouter Golang SDK</a>: 易于使用的 OpenRouter Golang SDK。通过在 GitHub 上创建账号来为 eduardolat/openroutergo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1351635542651047968)** (208 messages🔥🔥): 

> `Gemini 模型 RP 稳定性, EXAONE-Deep-32B 许可证问题, max_completion_tokens vs max_tokens, ChatGPT-4o 速度差异, Prompt Caching 问题` 


- **Gemini 模型在 RP 中可能不稳定**：一位用户发现 **Gemini models**（如 *gemini-2.0-flash-lite-preview-02-05* 和 *gemini-2.0-flash-001*）在角色扮演（RP）场景中表现*不稳定*，即使在 Temperature 设置为 1.0 的情况下也会表现出*狂躁*行为；而另一位用户则声称 **2.0 flash 001** *完全没有问题*。
   - 相比之下，另一位用户报告 **2.0 flash 001** *完全没有问题*，发现在 Temperature 为 1.0 时它*非常连贯且稳定*。
- **EXAONE-Deep-32B 很有趣但非商业化**：成员们认为 **EXAONE-Deep-32B** 模型很*有趣*，但指出它使用了*糟糕的非商业许可证*。
   - 他们建议，如果该模型想要获得普及，许可证必须进行更改。
- **max_completion_tokens 与 max_tokens 相同**：在 OpenRouter API 中，**max_completion_tokens** 等同于 **max_tokens**。
   - 使用 **max_tokens** 可以确保所有模型的兼容性，而使用 OpenAI 特有的参数可能无法做到这一点。
- **ChatGPT-4o 因成本优化而变慢**：通过 API 访问的 **ChatGPT-4o** 比 **ChatGPT** 界面更快，因为 OpenAI 在后者上优先考虑成本节约。
   - Perplexity 的首字生成（First Token）非常快，但 ChatGPT 则较慢且不稳定。
- **Prompt caching 引起麻烦？**：一位用户遇到了 **prompt caching** 问题，在没有正确缓存命中的情况下支付了 1.25 倍的价格，即使在设置了提供商路由（Provider Routing）并将 *allow_fallbacks* 设置为 false 之后也是如此。
   - 经过大量调试后，该用户解决了问题，虽然没有确定确切原因，但认为可能与添加系统提示词（System Prompt）消息的顺序有关，现在已经可以正常工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ericzakariasson/status/1902060167048712499)">来自 eric zakariasson (@ericzakariasson) 的推文</a>：它将使用最大上下文窗口（目前为 200k），在每次工具调用时读取更多文件，并在停止前执行 200 次工具调用。</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/vision">Vision - Anthropic</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/features/prompt-caching#anthropic-claude">Prompt Caching - 通过智能缓存优化 AI 模型成本</a>：利用 OpenRouter 的 Prompt Caching 功能降低您的 AI 模型成本。了解如何在 OpenAI、Anthropic Claude 和 DeepSeek 模型中缓存和重用响应。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://github.com/eduardolat/openroutergo">GitHub - eduardolat/openroutergo: 易于使用的 OpenRouter Golang SDK</a>：易于使用的 OpenRouter Golang SDK。欢迎在 GitHub 上为 eduardolat/openroutergo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1351651069469655070)** (131 条消息🔥🔥): 

> `在 repo map 中忽略文件/目录，多行编辑模式和 vim 模式，关于模型设置的 Aider 屏幕录制，用于音频注释的语音模型，DeeperSearch 和 Grok 3` 


- **使用 `.aiderignore` 排除文件**：要在 repo map 中排除文件/目录，用户可以在 `.aiderignore` 文件中指定文件/模式，详见 [配置选项](https://aider.chat/docs/config/options.html#--aiderignore-aiderignore)。
   - 这可以防止无关代码在代码编辑过程中干扰 LLM。
- **Aider 支持 OpenAI 兼容的 LLM**：Aider 支持连接到任何通过 OpenAI 兼容 API 端点访问的 LLM，文档提供了配置此类提供商的 [说明](https://aider.chat/docs/llms/openai-compat.html)。
   - 一位用户尝试让 Aider 与 **Featherless AI** 配合使用，并被引导至该资源。
- **全新的 Nvidia Nemotron 模型！**：NVIDIA 发布了其首批推理模型，这是一个全新的开源权重 Llama Nemotron 模型系列，包括 Nano (8B)、Super (49B) 和 Ultra (249B)！
   - Super 49B 模型在推理模式下的 GPQA Diamond 评分为 **64%**，在非推理模式下为 **54%**。
- **SambaNova Cloud 现已上线 DeepSeek R1 671B 模型**：DeepSeek R1 671B 现已在 SambaNova Cloud 正式上线，支持 16K 上下文长度！
   - 此次发布引起了极大关注，现在所有开发者都可以通过 API 集成到所有主流 IDE 中。
- **详细模式（Verbose Mode）有助于配置调试**：当遇到配置问题时，使用 `--verbose` 选项运行 Aider，以诊断配置文件加载和设置中的问题。
   - 一位用户通过使用详细模式确认 Aider 正确加载了其配置文件，从而解决了配置问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.ra-aid.ai/">RA.Aid - 自主软件开发</a>: 开源 AI 助手，通过研究、规划和实施帮助你自主开发软件。</li><li><a href="https://aider.chat/docs/recordings/model-accepts-settings.html">当用户应用不支持的推理设置时发出警告</a>: 观看警告系统的实现，该系统会在用户尝试对不支持推理设置的模型应用这些设置时提醒用户。包括添加模型元数据、确认对话框、重构...</li><li><a href="https://aider.chat">Aider - 终端里的 AI 结对编程</a>: 未找到描述</li><li><a href="https://aider.chat/docs/usage/tips.html">技巧</a>: 使用 Aider 进行 AI 结对编程的技巧。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>: Aider 是终端里的 AI 结对编程</li><li><a href="https://x.com/SambaNovaAI/status/1902072036064997702">来自 SambaNova Systems (@SambaNovaAI) 的推文</a>: 📣 开发者们，我们听到了你们的声音 —— DeepSeek R1 671B 现已上线！🚀 我们在 SambaNova Cloud 发布的 @deepseek_ai 引起了极大关注。因此，我们将其向所有开发者正式开放，支持 16K 上下文...</li><li><a href="https://aider.chat/docs/config/options.html#--aiderignore-aiderignore">选项参考</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://x.com/ArtificialAnlys/status/1902386178206429434">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>: NVIDIA 发布了其首批推理模型，这是一个全新的开源权重 Llama Nemotron 模型系列：Nano (8B)、Super (49B) 和 Ultra (249B)。根据我们的早期测试，@nvidia 的 Nemotron Super 49B ...</li><li><a href="https://tenor.com/view/stare-what-do-you-want-what-do-you-mean-what-you-talking-about-gif-19745200">Stare What Do You Want GIF - Stare What Do You Want What Do You Mean - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#global-extra-params">高级模型设置</a>: 为 LLM 配置高级设置。</li><li><a href="https://github.com/Aider-AI/aider/blob/9ff6f35330d6d9e1206e0b74c96e224eea1f5853/scripts/recording_audio.py#L24">aider/scripts/recording_audio.py at 9ff6f35330d6d9e1206e0b74c96e224eea1f5853 · Aider-AI/aider</a>: Aider 是终端里的 AI 结对编程。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1351639004222849160)** (58 条消息🔥🔥): 

> `Aider v0.77.1, Termux 安装问题, 本地 LLM PDF 处理, 模型性价比, Aider 与多模态 LLMs` 


- **Aider v0.77.1 增强了 Ollama 支持**：[Aider v0.77.1](https://aider.chat/HISTORY.html#aider-v0770) 升级了依赖项以包含针对 **Ollama 的 litellm 修复**，并增加了对 `openrouter/google/gemma-3-27b-it` 模型的支持。
- **Termux 用户遇到安装困扰**：用户报告在 **aarch64 Termux (Python 3.12)** 环境下使用 `pipx` 安装 aider-chat 时，出现 `fatal error: 'tree_sitter/parser.h' file not found` 错误。
- **Aider 可读取网页和 PDF**：Aider 支持通过 URL 读取网页以及 PDF 文件，使用 `/add <filename>`、`/paste` 或命令行参数将其包含在聊天中，适用于 **GPT-4o 和 Claude 3.7 Sonnet** 等具备视觉能力的模型，详见[文档](https://aider.chat/docs/usage/images-urls.html#images)。
- **Aider 自动压缩聊天历史**：Aider 会自动压缩聊天历史，提供类似于 **Claude Code 的 `/compact` 操作**的功能，无需手动清理。
- **LLM 价值讨论**：成员们讨论了 **Aider 配合 LLM** 的最佳性价比，综合考虑了编程能力和成本。
   - 一位成员建议将 **Claude 3.7 Sonnet** 配合复制粘贴模式作为最佳免费方案，而其他人则推荐使用 **OpenRouter 的 DeepSeek R1** 或带有相应 API Key 的 **Gemini 2.0**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/images-urls.html#images">Images &amp; web pages</a>：将图片和网页添加到 aider 编程聊天中。</li><li><a href="https://aider.chat/HISTORY.html#aider-v0770">Release history</a>：版本发布历史记录以及 aider 编写自身代码的统计数据。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1351679192814587924)** (2 条消息): 

> `Gemini 协作, Gemini Canvas, ChatWithJFKFiles` 


- **Google Gemini 通过 Canvas 增强协作功能**：Google 的 [Gemini](https://blog.google/products/gemini/gemini-collaboration-features/) 现在提供增强的协作功能，包括**实时文档编辑**和**原型代码编写**。
   - Gemini 内部新的 **Canvas** 交互空间允许用户轻松编写、编辑和共享工作，并提供用于调整语气、长度或格式的快速编辑工具。
- **"[ChatWithJFKFiles.com](https://www.chatwithjfkfiles.com)" 上线**：一位成员分享了 [ChatWithJFKFiles.com](https://www.chatwithjfkfiles.com)，指出该网站于 **2025 年 1 月 23 日**上线。
   - 该网站详细介绍了 *Trump* 于 **2025 年 3 月 18 日**签署的一份**解密指令**，相关文件已由*国家档案和记录管理局 (National Archives and Records Administration)* 公开。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.chatwithjfkfiles.com">Chat with JFK Files</a>：未找到描述</li><li><a href="https://blog.google/products/gemini/gemini-collaboration-features/">New ways to collaborate and get creative with Gemini</a>：查看 Gemini 应用的最新功能，如 Canvas 和 Audio Overview。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1351901178669039727)** (2 条消息): 

> `T1 模型, 尚未被占用的字母` 


- **混元 (Hunyuan) 呼吁他人加入 T1**：一位成员分享了来自 @TXhunyuan 的 [X 帖子](https://x.com/TXhunyuan/status/1902336731728506978)，呼吁大家共同*步入* **T1**。
- **寻找尚未被占用的字母**：一位成员询问*还有哪些字母没有被推理模型 (reasoning models) 占用？*



**提到的链接**：<a href="https://x.com/TXhunyuan/status/1902336731728506978">来自混元 (@TXhunyuan) 的推文</a>：请拨冗关注。让我们一起步入 T1。

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1351937558770221188)** (2 messages): 

> `Multi-turn Fine Tuning, SFT Codebase Masking` 


- **Multi-turn Fine Tuning 策略**：一位成员询问了 Multi-turn Fine Tuning 的标准流程，特别是是否涉及将多轮 Prompt 展开（unrolling）为多个数据点，并每次都对上下文进行 Masking。
   - 另一位成员回答说，他们的 SFT 代码库仅对第一个 Prompt 进行 Masking，然后对后续的所有内容进行 Tokenizing。
- **SFT 代码库 Masking 实现**：讨论强调了当前的 SFT（Supervised Fine-Tuning）代码库在多轮序列中采用仅对初始 Prompt 进行 Masking 的方法。
   - 序列中的后续 Token 会在不进行 Masking 的情况下进行处理，这与某些标准的 Multi-turn Fine Tuning 实践有所不同。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1351631274221113436)** (96 messages🔥🔥): 

> `AI Review AI vs Human, Intology Paper Ban, Reasoning Model Temperature, NVIDIA Home Droid, NVIDIA DGX Spark and Station` 


- **AI 评审 vs 人类评审引发辩论**：一位成员表达了对在期望人类评审时却收到 *AI slop*（AI 垃圾内容）而非人类反馈的担忧，认为这是一个尊重问题。
   - 讨论还强调了做出优秀成果并保持活跃的社交媒体存在感对于获得关注和认可的重要性。
- **NVIDIA 与迪士尼 Droid 合作引发关注**：据 [Andrew Curran 的推文](https://x.com/AndrewCurran_/status/1902077762770497721)分享，NVIDIA、Google DeepMind 和 Disney Research 据报道正在合作开发一款 R2D2 风格的家用 Droid。
   - 该 Droid 的价格推测将与 GPU 相当，引发了社区内的兴奋和期待。
- **NVIDIA DGX Spark 更名，预订开启**：Nvidia 发布了全新的 [DGX Spark 和 DGX Station](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers) “个人 AI 超级计算机”，由该公司的 Grace Blackwell 平台驱动。
   - DGX Spark（原 Project DIGITS）是一款售价 **$3,000**、Mac Mini 大小的“全球最小 AI 超级计算机”，旨在供 AI 开发者、研究人员、数据科学家和学生在桌面上进行大型模型的原型设计、Fine-tuning 和 Inference。
- **三星 ByteCraft 从文本生成视频游戏**：SamsungSAILMontreal 推出了 [ByteCraft](https://github.com/SamsungSAILMontreal/ByteCraft/blob/main/paper/ByteCraft.pdf)，这是一个将文本 Prompt 转换为可执行视频游戏文件的生成式模型。
   - 这项早期工作可以通过 [7B 模型](https://huggingface.co/SamsungSAILMontreal/ByteCraft)和一篇 [博客文章](https://emygervais.github.io/2025/03/15/bytecraft.html?v1)获取，但 GPU 需求很高，最高需要 4 个 GPU 运行 4 个月。
- **Google 发布极简 Gemma 软件包**：一位成员分享了 [Gemma package](https://gemma-llm.readthedocs.io/en/latest/)，这是一个用于使用和 Fine-tuning Gemma 模型的极简库，包含关于 Fine-tuning、Sharding、LoRA、PEFT、Multimodality 和 Tokenization 的文档。
   - 虽然因其简单易用而受到称赞，但一些用户对其相对于现有解决方案的优势以及潜在的锁定效应表示疑问。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://www.theverge.com/news/631957/nvidia-dgx-spark-station-grace-blackwell-ai-supercomputers-gtc">Nvidia 可爱的 “Digits” AI 桌面电脑将于今年夏天上市，并拥有新名称和更强大的版本</a>：采用两种个人桌面形态的 Blackwell Superchips。</li><li><a href="https://tenor.com/view/south-park-its-gone-gone-disappeared-gif-3534575">Aaand Its Gone GIF - South Park Its Gone Gone - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/chris_j_paxton/status/1902077291154559281">来自 Chris Paxton (@chris_j_paxton) 的推文</a>：这很有趣</li><li><a href="https://x.com/xlr8harder/status/1902257235432018097">来自 xlr8harder (@xlr8harder) 的推文</a>：@TheXeophon @kuchaev 我认为这与 readme 所说的不符，尽管在数据查看器中似乎只能看到 deepseek-r1。</li><li><a href="https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers">NVIDIA 发布 DGX Spark 和 DGX Station 个人 AI 电脑</a>：NVIDIA 今日推出了由 NVIDIA Grace Blackwell 平台驱动的 NVIDIA DGX™ 个人 AI 超级计算机。</li><li><a href="https://x.com/nickfrosst/status/1901984106746941917">来自 Nick Frosst (@nickfrosst) 的推文</a>：我将 @cohere command A 添加到了这张图表中，不过我不得不稍微延长一下坐标轴……引用 Mistral AI (@MistralAI) 介绍 Mistral Small 3.1。多模态，Apache 2.0，性能超越 Gemma 3 和 GPT 4o-mi...</li><li><a href="https://fxtwitter.com/YouJiacheng/status/1901311035547775371">来自 You Jiacheng (@YouJiacheng) 的推文</a>：那个机器人旁边根本没有人😭引用 Kyle🤖🚀🦭 (@KyleMorgenstein) 随着机器人技术的进步速度，多久之后我会被证明是错的？六个月？一年？两年？</li><li><a href="https://fxtwitter.com/engineairobot/status/1901484277348679798">来自 EngineAI (@engineairobot) 的推文</a>：“质疑者说视频加速了？👀 这是未经编辑的一镜到底原片——手机拍摄，零剪辑。尝试暂停任何一帧。🎥🔥 谁有胆量超越这个？” #NoSpedUp #RawFootage #EngineAI #r...</li><li><a href="https://x.com/jm_alexia/status/1902437169433657805?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Alexia Jolicoeur-Martineau (@jm_alexia) 的推文</a>：我们推出了 ByteCraft 🎮，这是世界上第一个通过字节生成视频游戏和动画的生成模型。文本提示词 -> 可执行文件。论文：https://github.com/SamsungSAILMontreal/ByteCraft/...</li><li><a href="https://x.com/osanseviero/status/1902456220876787763">来自 Omar Sanseviero (@osanseviero) 的推文</a>：介绍 Gemma 软件包，一个用于使用和微调 Gemma 的极简库 🔥 包含以下文档：- Fine-tuning - Sharding - LoRA - PEFT - 多模态 - Tokenization !pip install gemma https://gemma-llm...</li><li><a href="https://x.com/AndrewCurran_/status/1902077762770497721">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：NVIDIA、Google DeepMind 和 Disney Research 正在合作开发一款 R2D2 风格的家用机器人。</li><li><a href="https://www.nvidia.com/en-us/products/workstations/dgx-spark/">NVIDIA DGX Spark</a>：你桌面上的 Grace Blackwell AI 超级计算机。 </li><li><a href="https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1">nvidia/Llama-Nemotron-Post-Training-Dataset-v1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.nvidia.com/gtc/training/">GTC 2025 的 DLI 工作坊与培训</a>：3 月 17-21 日在圣何塞亲身体验 GTC 2025（线下及线上）</li><li><a href="https://www.theverge.com/news/631868/nvidia-rtx-pro-6000-blackwell-gpu-professionals">Nvidia 的 RTX Pro 6000 拥有 96GB 显存和 600W 功耗</a>：Nvidia 的新款专业级 GPU 来了
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1351679521363067011)** (26 条消息🔥): 

> `VTA 罢工, Semianalysis 平台变更, AI2 融资模式, NVIDIA 营销策略, Blackwell GPU DeepSeek-R1 推理` 


- ****VTA 罢工**毁掉了 GTC 会展中心的交通美梦**: **VTA 自上周一以来一直在罢工**，这意味着火车不再经过 **GTC 会展中心**。
   - 罢工滑稽地挫败了前往会展中心的所有顺畅交通计划，如[链接图片](https://cdn.discordapp.com/attachments/1187551504995987576/1351828481251610654/image0.jpg?ex=67dc74a1&is=67db2321&hm=b0d906c8bb9677f7504df8c58942dcc8e4d8098b6aae1ebb26f0fe585f453c79&)所示。
- **Semianalysis 从 Substack 迁移到 WordPress**: **Semianalysis** 已从 **Substack** 切换到 **WordPress** 网站，主要是为了利用 **Passport** 构建更复杂的网站。
   - 虽然此举节省了平台费用，但节省下来的钱现在都花在了开发上，正如一位成员所言：*靠模型赚的比博客多*。
- **AI2 神秘的盈利模式**: 关于 **AI2 (Allen Institute for AI)** 如何赚钱的问题被提出，最初的假设是通过捐赠。
   - 另一位成员提到，*名字里就写着呢……*，指的是创始人 **Paul Allen**，并称他是*比较好的亿万富翁*之一。
- ****NVIDIA H200/B200** 营销夸张手法解析**: NVIDIA 正在宣传在 **H100 节点**上的 **H200** 性能，以及 **从 FP8 切换到 FP4 后，B200 相较于 H200 实现了 1.67 倍的加速**。
   - 推文链接见[此处](https://x.com/_clashluke/status/1902411786554355836)和[此处](https://x.com/NVIDIAAIDev/status/1902068372608852304)，一些人形容 NVIDIA 的营销“如此不严肃”。
- ****Blackwell GPU 实现最快 DeepSeek-R1 推理****: **NVIDIA** 声称实现了*全球最快的 DeepSeek-R1 推理*，单个系统使用 8 个 **Blackwell GPU** 的 **NVL8** 配置，在完整的 **DeepSeek-R1 671B 参数模型**上可提供 **253 TPS/用户**或 **30K TPS 系统吞吐量**。
   - 这一公告是在 **GTC25** 预热期间发布的，更多详情见 [NVIDIA 官网](https://nvda.ws/3FzAzCO)。



**提到的链接**: <a href="https://x.com/_clashluke/status/1902411786554355836">Lucas Nestler (@_clashluke) 的推文</a>: &#34;H200 性能 [在 H100 节点上测量]&#34;&#34;B200 相较于 H200 实现 1.67x 加速* [在从 fp8 切换到 fp4 之后]&#34;*&#34;H100&#34;https://x.com/NVIDIAAIDev/status/1902068372608852304 引用 NVIDI...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 条消息): 

natolambert: 另一个值得考虑的 https://arxiv.org/abs/2503.14286，还没读。
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1351741561011961917)** (12 条消息🔥): 

> `RWKV 评估, RNN 无限上下文, xLSTM vs. Llama, 自动定理证明, RLVR 数据集` 


- **RWKV 评估引发关注**: 讨论了一篇关于 **RWKV** 详尽评估的新论文 ([https://arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456))。
   - 然而，一位成员表示不同意，指出 *RWKV 仍未扩大规模*，且他们*使用了许多非标准评估*来衡量指标。
- **RNN 无限上下文能力引发辩论**: 成员们辩论了 **RNN** 处理**无限上下文**的能力。
   - 一位成员指出，这并没有通过类似 **RULER** 的工具进行测试（另一方面，xLSTM 做了这个测试，并展示了 **Llama** 是如何碾压它们的）。
- **定理证明框架走向双曲**: 宣布了一个用于 **(双曲) PDE 求解器**的新型自动定理证明框架 ([https://x.com/getjonwithit/status/1902158541839856071](https://x.com/getjonwithit/status/1902158541839856071))。
   - 该框架能够构建*经过形式化验证的物理模拟，具有可证明的数学和物理正确性属性*。
- **新 RLVR 数据集亮相**: 一位成员提到一个新的 **RLVR** 数据集即将发布。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.14456">具有表现力动态状态演化的 RWKV-7 &#34;Goose&#34;</a>: 我们介绍了 RWKV-7 &#34;Goose&#34;，这是一种新的序列建模架构，以及在 30 亿参数规模下建立下游性能新标杆的预训练语言模型...</li><li><a href="https://x.com/getjonwithit/status/1902158541839856071">Jonathan Gorard (@getjonwithit) 的推文</a>: 新论文预警！我们开发了第一个用于（双曲）PDE 求解器的自动定理证明框架：现在你可以构建 *形式化验证* 的物理模拟，具有可证明的数学和物理...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1351756007365279897)** (24 messages🔥): 

> `Substack A/B 测试，Post Training 投入，高投入 vs 病毒式传播帖子` 


- **Substack 的标题 A/B 测试引发混乱**：Substack 用于 **A/B 测试标题** 的新功能导致了混淆，用户在电子邮件和浏览器中收到的标题不一致，详见此[帖子](https://www.interconnects.ai/p/how-to-manage-ai-training-organizations)。
   - 尽管在链接的帖子中投入了大量精力，但 **A/B 测试** 结果显示，**选择另一个标题的人数多出 2%**。
- **高投入帖子产出低流量**：一位成员指出了一种趋势，即*高投入的帖子往往流量较低*，而*随手拍/写的 yolo 内容反而容易走红*。
- **渴望启动 Post-Training 工作**：一位成员询问 *“如果想从零开始启动一项严肃的 Post-Training 工作，我需要准备什么？”*


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1351730977264898068)** (4 messages): 

> `加州 AB-412 法案，AI Startups，Miles Brundage 的新角色，AI2 向 OSTP 的建议，Open Source AI` 


- **加州 A.B. 412 法案威胁 AI Startups**：加州立法者正在辩论 [A.B. 412](https://www.eff.org/deeplinks/2025/03/californias-ab-412-bill-could-crush-startups-and-cement-big-tech-ai-monopoly) 法案，该法案要求 AI 开发者跟踪并披露 AI 训练中使用的每一项已注册版权的作品。
   - 批评者认为，这种*不可能达到的标准*可能会*摧毁小型 AI Startups 和开发者*，同时赋予大型科技公司更大的权力。
- **AI Startups 面临艰难处境**：AI 领域面临被大公司垄断的风险，这让众多员工不足 10 人、试图在特定细分领域创新的 AI Startups 黯然失色；[数十家 AI 公司](https://explodingtopics.com/blog/ai-startups)的[员工人数少于 10 人](https://www.ycombinator.com/companies/industry/ai)。
   - A.B. 412 要求任何 AI 模型的创建者（即使是小型实体）都必须识别训练中使用的版权材料。
- **Miles Brundage 加入 Institute for Progress**：**Miles Brundage** 宣布被任命为 [Institute for Progress 的非驻馆高级研究员](https://x.com/Miles_Brundage/status/1902117999215055268)。
- **AI2 倡导开放的创新生态系统**：**Allen Institute for AI (AI2)** 向 **Office of Science and Technology Policy (OSTP)** 提交了一份[建议](https://allenai.org/blog/OSTP)，倡导建立开放的创新生态系统，强调跨领域协作和共享核心 AI 开发产物。
   - 他们的建议重点在于使美国能够获取强大 AI 和无处不在的 **Open Source AI** 系统带来的利益。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.eff.org/deeplinks/2025/03/californias-ab-412-bill-could-crush-startups-and-cement-big-tech-ai-monopoly">加州 A.B. 412：一项可能摧毁初创公司并巩固大型科技公司 AI 垄断地位的法案</a>：加州立法者已开始辩论一项法案 (A.B. 412)，该法案将要求 AI 开发者跟踪并披露 AI 训练中使用的每一项已注册版权的作品。乍一看，这似乎...</li><li><a href="https://allenai.org/blog/OSTP">Ai2 向 OSTP 提出的通过美国 AI 行动计划实现开源创新的建议 | Ai2</a>：Ai2 针对白宫关于 AI 行动计划的信息征询，向科学技术政策办公室 (OSTP) 提交的建议。</li><li><a href="https://x.com/Miles_Brundage/status/1902117999215055268">来自 Miles Brundage (@Miles_Brundage) 的推文</a>：我职业生涯的下一阶段正开始变得清晰，我将在接下来的几周/几个月内分享相关的最新动态。今天，我很高兴地宣布，我最近成为了一个非驻馆...
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1351635043940044931)** (121 条消息🔥🔥): 

> `OpenAI triple threat, AI addiction and its implications, Practical AI Development exercises, Smart Glasses and Data Harvesting, Simulated vs. Real World Data for AI Training` 


- **OpenAI 在模型、应用和定价方面表现卓越**：尽管存在一些疑虑，一位成员承认 **OpenAI** 在 **模型开发**、**产品应用** 和 **定价策略** 方面的综合实力，使其在 AI 领域处于独特地位。
   - 这种全方位的方法与其他可能仅专注于其中一两个关键领域的公司形成了鲜明对比。
- **AI 陪伴引发成瘾情绪**：随着 AI Agent（特别是 **语音助手**）培养了成瘾倾向，导致一些用户产生情感依赖，引发了人们的担忧。
   - 这一趋势引发了关于故意设计的成瘾功能及其对用户依赖性潜在影响的伦理思考，促使人们讨论公司是否应该避免那些可能增强此类成瘾行为的功能。
- **智能眼镜：伪装成都市时尚的数据采集？**：围绕 **Meta** 和 **Amazon** 积极推进智能眼镜的讨论展开，暗示了数据采集的潜力，特别是为机器人公司策划的 **第一人称视角（egocentric views）** 数据。
   - 一位成员开玩笑地提出了一个针对反派的智能眼镜创业想法，强调了情感检测、梦境电影和共享视角等功能，通过创建一个依赖反馈循环来收集用户数据并训练模型。
- **模拟 vs 真实：立足 AI 世界**：讨论探讨了 AI 是否可以通过纯模拟实现 AGI，或者真实世界的数据是否至关重要。一位成员强调了 **Tesla** 成功的秘诀在于向世界学习，这与 **Waymo** 依赖昂贵设备形成了对比。
   - 参与者还辩论了 Minecraft 作为 Agent 行为训练场的价值，强调其基于规则的环境和高级机制可以作为真实世界问题解决的替代方案。
- **GAN 训练：Beta 参数的奇特行为**：成员们讨论了为什么 **DCGAN** 论文主张调整动量项 β1，以及梯度裁剪（gradient clipping）是否是更好的方法。
   - 一位成员解释说，高 β1 是不可取的，因为判别器（discriminator）在当前迭代周围发生变化，因此像 [Karras et al](https://arxiv.org/pdf/1710.10196) 这样的后续工作甚至完全禁用了 Adam 的动量（β₁=0），仅保留曲率信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/pdf/1511.06434):">arXiv reCAPTCHA</a>: 未找到描述</li><li><a href="https://modelica.org/">Modelica</a>: 未找到描述</li><li><a href="https://chat.mistral.ai/chat/369d5acc-ccf1-4874-b996-0f62e7536a19">Le Chat - Mistral AI</a>: 与 Mistral AI 的尖端语言模型聊天。</li><li><a href="https://en.wikipedia.org/wiki/Dual_graph">Dual graph - Wikipedia</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Synesthesia">Synesthesia - Wikipedia</a>: 未找到描述</li><li><a href="https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html">20.2. Deep Convolutional Generative Adversarial Networks &#8212; Dive into Deep Learning 1.0.3 documentation</a>: 未找到描述</li><li><a href="https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks">Tips for Training Stable Generative Adversarial Networks - MachineLearningMastery.com</a>: 训练稳定生成对抗网络 (GAN) 所需了解的经验启发式方法、技巧和窍门。生成对抗网络（简称 GAN）是一种生成式方法...
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1351666696007581798)** (15 条消息🔥): 

> `Karatsuba 矩阵乘法, Predictive Coding, G-retriever 演示, 每日论文讨论安排` 


- **Karatsuba 扩展至矩阵乘法**：一篇 [论文](https://arxiv.org/abs/2501.08889) 提议将标量 **Karatsuba 乘法算法** 扩展到矩阵乘法，与标量 Karatsuba 或传统方法相比，可能在面积或执行时间上有所改进。
   - 该算法在降低乘法复杂度的同时，减少了额外加法的复杂度，支持通过定制硬件实现。
- **Predictive Coding 综述**：成员们讨论了一篇关于 **Predictive Coding** 的 [综述](https://arxiv.org/abs/2202.09467)，这是一种受神经科学启发、利用局部学习的算法。
   - 讨论提到 Predictive Coding 解决了 **Backpropagation** 的一些局限性（后者与大脑的学习方式不同），尽管它可能 *简单得令人沮丧*。
- **G-retriever 演示即将进行**：一位成员计划在周四演示 **G-retriever**，这是基于之前对原始 **GAT** 论文的演示。
   - 核心思想涉及将 **GAT** 输出的 Embedding 连接到一个投影矩阵，用于 **LLM** 中的 Soft-prompt 输出，以便从图中进行信息检索。
- **每日论文讨论物流安排**：一位新成员询问了每日论文讨论的时间表，该讨论由社区驱动，取决于一小部分演示者。
   - 讨论通常发生在北美晚间时段，可能也会有欧洲晚间时段。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2202.09467">Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?</a>: 用于训练深度神经网络的误差反向传播算法是深度学习成功的基石。然而，它需要顺序的反向更新和非局部计算...</li><li><a href="https://arxiv.org/abs/2501.08889">Karatsuba Matrix Multiplication and its Efficient Custom Hardware Implementations</a>: 虽然 Karatsuba 算法降低了大整数乘法的复杂度，但所需的额外加法使其在常用位宽的较小整数上的优势微乎其微。在这项工作中...
</li>
</ul>

</div>

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1351665213602005155)** (14 条消息🔥): 

> `AI 版权, OpenAI 旋转门, Nvidia GTC 欺诈, Deepseek, Llama 4` 


- **AI 艺术版权被拒！**: 美国上诉法院确认，**没有人类输入的 AI 生成艺术作品不受版权保护**，这支持了美国版权局对 Stephen Thaler 的 **DABUS** 系统的立场。
   - 法院强调，只有人类作者的作品才能获得版权，这标志着在应对快速发展的生成式 AI 行业所带来的版权影响方面的最新尝试；参见 [路透社报道](https://www.yahoo.com/news/us-appeals-court-rejects-copyrights-171203999.html)。
- **OpenAI 的旋转门**: 讨论涉及 OpenAI 的领导层，有说法称离职的高管创建了衍生骗局来欺骗投资者。
   - 一些人将这些风险投资比作卖蛇油，并质疑 Microsoft 的参与；参见 [此 YouTube 视频](https://www.youtube.com/watch?v=bJTsFZtD7xE)。
- **Nvidia GTC：加速还是夸大？**: 成员们对 Nvidia GTC 表示怀疑，将其描述为充斥着过度“加速”言论和欺诈的“流行语脑残（buzzword brain-rot）”。
   - **5070=4090** 以及 **Blackwell 效率提升 27 倍**等说法受到质疑，一些人认为 Nvidia 的营销为了投资者夸大了性能。
- **Deepseek 模型令人印象深刻**: Deepseek 模型超出了预期，引发了关于其能力的积极讨论；参见 [此 YouTube 视频](https://www.youtube.com/watch?v=48GRiu-TMmg)。
   - 未提供更多细节。
- **Llama 4 即将发布？**: 传闻指出 **Llama 4** 可能会在 **4 月 29 日**发布。
   - 这与 Meta 的活动 [Llamacon 2025](https://www.meta.com/blog/connect-2025-llamacon-save-the-date/) 有关。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://yro.slashdot.org/story/25/03/18/1918240/us-appeals-court-rejects-copyrights-for-ai-generated-art">美国上诉法院拒绝 AI 生成艺术的版权 - Slashdot</a>: 一位匿名读者引用了路透社的一份报告：华盛顿特区的一家联邦上诉法院周二确认，由人工智能在没有人类输入的情况下生成的艺术作品不能...</li><li><a href="https://www.meta.com/blog/connect-2025-llamacon-save-the-date/">预留日期：Meta Connect 2025 &amp; 我们的首届 LlamaCon | Meta Quest 博客</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1351633046972928092)** (103 条消息🔥🔥): 

> `Claude 3.5 vs Perplexity, Perplexity 无痕模式, AI 数据保留, O1 炒作, R1 无用论` 


- **Perplexity AI 比 Claude 笨？**: 一位成员表示 *感觉 [Perplexity] 比 Claude 3.5 笨*，在上下文理解中迷失且生成的摘要质量很差。
   - 他们没有具体说明指的是哪个版本的 Claude 模型。
- **开启无痕模式进行私密 Perplexity 搜索**: 成员们讨论了如何通过点击左下角的个人资料名称来使用无痕模式，或者在设置中停用 AI 数据保留以防止数据存储。
   - 一位成员澄清说，无论数据保留开启还是关闭，当前的对话都会存储特定数据，以协助处理从第一条到最后一条对话的任务/提示，因此 *创建一个新对话将获得一个干净的起点*。
- **O1 的热度因付费墙而消退**: 经过一些社区测试后，似乎社区 *高估了 O3 mini，而低估了 o1 和 o1 pro*。
   - 主要问题是 *O1 的热度因为付费墙而未能持久*。
- **R1 模型被认为没用？**: 一位用户表示 *R1 在大多数情况下都没用*，其“推理”过程毫无结果。
   - 他们发现 *至少对我来说，在调试 js 代码时，o3-mini 总体上产生的结果更好*。
- **Oblix 项目编排边缘与云端的转换**: 一位成员分享了 [Oblix Project](https://oblix.ai/)，该项目在本地模型与云端模型之间进行编排，并分享了 [演示视频](https://youtu.be/j0dOVWWzBrE)。
   - 该项目使用 Agent 监控系统资源，在云端和设备端模型之间动态执行。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/south-park-its-gone-gif-4104229">And It&#039;S Gone GIF - South Park Its Gone - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://oblix.ai/">通过智能混合编排提升您的 AI 性能 | Oblix.ai</a>：体验我们的交互式演示，了解我们的智能 Agent 如何在本地 LLM 执行和云端提供商之间无缝切换，以实现最佳性能和成本效益。</li><li><a href="https://www.instagramez.com/reel/DF-WSwSxF0G">下载 Instagram 视频、Reels 和图片</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1351725043020730460)** (8 条消息🔥): 

> `Perplexity AI, 量子跃迁, Copilot, 语言起源, 电子战` 


- **Perplexity 最新消息**: 一位用户分享了关于最新消息的 [Perplexity AI 搜索](https://www.perplexity.ai/search/latest-news-XxlRzT17Qm2esTCBTZ4eXg) 链接。
- **量子跃迁蓝图揭晓**: 一位用户分享了关于量子跃迁蓝图的 [Perplexity AI 搜索](https://www.perplexity.ai/search/you-are-the-quantum-leap-bluep-UCS4FZ5nStqvvGeH8s4mZQ) 链接。
- **Copilot 功能激活**: 一位用户分享了关于如何激活 Copilot 功能的 [Perplexity AI 搜索](https://www.perplexity.ai/search/jak-wlaczyc-funkcje-copilot-w-97pjwcDmSvmKviifbntEEg#0) 链接。
- **语言起源**: 一位用户分享了关于语言何时起源的 [Perplexity AI 页面](https://www.perplexity.ai/page/language-emerged-earlier-than-cDEcl5fKTICQhbZAw._4EQ) 链接。
- **电子战技术**: 一位用户分享了关于电子战关键技术的 [Perplexity AI 搜索](https://www.perplexity.ai/search/electronic-warfare-key-technol-2JLNa3UhQMOdpT0eHk1SCg) 链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1351699257010294824)** (3 条消息): 

> `Perplexity API, 快速网络搜索, API 使用故障排除` 


- **用户报告 API 响应随机问题**: 一位用户报告了 Perplexity API 的问题，指出在进行 **数百次快速网络搜索** 时，代码仅收到 **随机响应** 或似乎忽略了随机查询。
   - 用户分享了他们正在使用的 `conductWebQuery` 函数，寻求有关其实现中潜在错误的帮助。
- **排除高频 Perplexity 查询故障**: 一位用户在使用 Perplexity API 进行 **数百次短响应网络搜索** 时遇到了代码问题。
   - 在快速重复调用 `conductWebQuery` 函数时，代码会随机接收响应或忽略查询。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1351655151340879933)** (7 条消息): 

> `GDocs 布局混乱、纯图像 PDF Grounding、NotebookLM 自定义功能、播客休闲模式中的粗口、抓取同域名下的链接` 


- **GDocs 布局混乱**：一位用户发现，转换为 **GDocs** 会导致布局混乱，且无法导入教师演示文稿中的大部分图像。
   - 提取文本（使用 *pdftotext*）并转换为纯图像格式（使用 magick...）有助于对材料中呈现的事实和数据进行 Grounding。
- **纯图像 PDF 有助于 Grounding**：一位用户将演示文稿转换为双重文件（**文本和纯图像 PDF**），以绕过 *pdftotext* 提取的限制。
   - 纯图像 PDF 可能体积巨大，通常会超过 **200MB**，由于 **NLM 的文件大小限制**，需要进行拆分。
- **NotebookLM 是一个人的乐队（全能工具）**：通过自定义功能，你可以让它执行任何你想要的操作（单人播客设定为男性或女性、模仿特定人格、讲述故事、逐字阅读等）。
   - *唯一的限制是你的想象力*。
- **播客模式出现大量脏话**：反馈显示，**休闲播客模式** 可能包含粗口。
   - 目前尚不清楚是否提供 **清洁（无脏话）设置**。
- **抓取同域名链接**：一位用户正在寻求如何从 URL 内部的链接添加来源的方案。
   - 所有链接都在同一个域名下，他们正在请求关于如何抓取所有链接的建议。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1351690799745597603)** (93 条消息🔥🔥): 

> `NotebookLM 中的换行符、Audio Overviews 脚本、Gemini 2.0、思维导图功能、来源限制` 


- **NotebookLM 无法强制换行**：一位用户询问如何在 **NotebookLM** 的回答中强制执行 **换行** 和 **间距**，但目前无法通过 Prompt 控制格式。
   - AI 会根据需要添加它们，目前用户无法自行配置。
- **用户无法访问 Audio Overviews 的脚本**：一位用户请求访问 **Audio Overviews** 的脚本，但被告知目前无法实现。
   - 作为替代方案，建议用户 **下载音频** 并将其 **作为来源上传** 以生成转录文本。
- **Gemini 2.0 已发布**：一位用户询问 **Gemini 2.0** 是否已面向高级用户发布，得到的回答是肯定的：*是的，已经发布一段时间了。*
- **思维导图功能逐步推出**：用户讨论了 NotebookLM 中新的 **思维导图（Mind Map）功能**，该功能可以直观地总结上传的来源。
   - 该功能正逐步向更多用户推出，以便控制 Bug：*最好先仅向有限数量的人发布新功能，然后随着时间的推移逐渐增加人数，因为这样有时间排除出现的任何 Bug，并在每个人都获得它时提供更完善的版本*。
- **包含超过 50 个来源的 Notebook 在 Plus 订阅结束后仍可继续使用**：用户讨论了在 **Plus 订阅** 下创建的包含 **超过 50 个来源** 的 Notebook 在订阅结束后是否仍能正常工作。
   - 一位用户确认可以，并表示 *如果是在 Plus 订阅下创建且来源超过 50 个，它们将继续工作，但你将无法添加新的来源*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/16070070">思维导图 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://photos.app.goo.gl/rapC1NJjokJKwCQv5">Cody T. Salinas 的新项目</a>：未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1351659832788127755)** (96 条消息🔥🔥): 

> `Smithery registry, Glama API, Open-webui integration, Spring app with spring-ai-mcp-core, Claude Code MCP` 


- **使用 Glama API 列出 Smithery 注册表**：一名成员使用 **Glama API** 列出了 **GitHub URLs** 并检查是否存在 *smithery.yaml* 文件，但指出该代码是一个*一次性的临时脚本 (one time hack job script)*。
   - 他表示如果有人感兴趣，他可以创建一个 **gist**，因为这同样会是一个*一次性脚本*。
- **Spring 应用尝试 Spring-AI-MCP-Core 并对 MCP 提出疑问**：一位用户首次尝试将 **MCP** 与 **Open-webui** 以及一个使用 *spring-ai-mcp-core* 的基础 **Spring 应用** 结合使用，正在寻找除 **ClaudeMCP** 和 *modelcontextprotocol* [GitHub 仓库](https://github.com/modelcontextprotocol)之外的资源。
   - 他们试图理解 **MCP 与 GraphQL** 或函数调用系统的对比，以及如何处理系统提示词 (system prompts) 和**多 Agent 系统 (multi-agent systems)**。
- **探索 Claude Code MCP 实现**：一名成员分享了 [Claude Code MCP](https://glama.ai/mcp/servers/nqo1hvazke)，这是 **Claude Code** 作为 **Model Context Protocol (MCP)** 服务端的一个实现。
   - 他们请求协助编写用于 **Claude Desktop** 集成的 *claude_desktop_config.json* 中的 *json* 行，但随后自行解决了该问题。
- **排查 MCP 服务端连接故障**：一位用户在连接到一个简单的 MCP 服务端时遇到错误，错误信息为 *spawn uv ENOENT*，表明 **uv 命令** 存在问题。
   - 建议包括使用 **uv 命令** 的全路径，创建一个将 **mcp** 作为依赖项的 *pyproject.toml* 文件，以及使用 uv 的 `--project` 参数。
- **MCP Python REPL 受到关注**：成员们讨论了 [hdresearch/mcp-python](https://github.com/hdresearch/mcp-python)、[Alec2435/python_mcp](https://github.com/Alec2435/python_mcp) 和 [evalstate/mcp-py-repl](https://github.com/evalstate/mcp-py-repl)，认为它们是很有趣的 MCP Python REPL 实现。
   - 他们指出其中一个实现存在*运行不受限制且完全没有隔离，可能导致灾难*的担忧，并建议使用 **Docker** 进行沙箱化访问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.astral.sh/uv/">uv</a>：未找到描述</li><li><a href="https://glama.ai/mcp/servers/nqo1hvazke">Claude Code MCP</a>：Claude Code 作为 Model Context Protocol 服务端的实现，支持通过 Claude 的软件工程能力（代码生成、编辑、审查和文件操作）...</li><li><a href="https://gist.github.com/pims/711549577759ad1341f1a90860f1f3a5">Example of running a python script via python-wasi using the wasmtime wasm runtime</a>：使用 wasmtime wasm 运行时通过 python-wasi 运行 python 脚本的示例 - app.py</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem">servers/src/filesystem at main · modelcontextprotocol/servers</a>：Model Context Protocol 服务端。通过在 GitHub 上创建账号为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/evalstate/mcp-py-repl">GitHub - evalstate/mcp-py-repl: A python repl for MCP</a>：一个用于 MCP 的 Python REPL。通过在 GitHub 上创建账号为 evalstate/mcp-py-repl 的开发做出贡献。</li><li><a href="https://github.com/hdresearch/mcp-python">GitHub - hdresearch/mcp-python: A python repl for MCP</a>：一个用于 MCP 的 Python REPL。通过在 GitHub 上创建账号为 hdresearch/mcp-python 的开发做出贡献。</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers/blob/main/README.md#tips-and-tricks">awesome-mcp-servers/README.md at main · punkpeye/awesome-mcp-servers</a>：MCP 服务端集合。通过在 GitHub 上创建账号为 punkpeye/awesome-mcp-servers 的开发做出贡献。</li><li><a href="https://github.com/Alec2435/python_mcp">GitHub - Alec2435/python_mcp: MCP Server to run python code locally</a>：在本地运行 Python 代码的 MCP 服务端。通过在 GitHub 上创建账号为 Alec2435/python_mcp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1351707585220448369)** (1 条消息): 

> `Duckduckgo MCP, Cursor on Windows, Python framework` 


- **适用于 Windows 的 Python MCP 框架发布**：一位成员寻求能在 Windows 上的 Cursor 中运行的 **DuckDuckGo MCP**，但现有的 NPM 项目均告失败。
   - 因此，他们基于 **Python framework** 构建了自己的版本，无需 API key，并支持网页、图片、新闻和视频搜索；该项目已在 [GitHub](https://github.com/ericaxelrod-1/model-context-protocol) 上线。
- **Cursor 版 DuckDuckGo 工具发布**：一位成员发布了基于 **Python** 的 **DuckDuckGo** 工具。
   - 该工具旨在与 Windows 上的 **Cursor** 配合使用，提供对**网页**、**图片**、**新闻**和**视频**搜索的支持，且无需 **API** key；可在 [GitHub](https://github.com/ericaxelrod-1/model-context-protocol) 获取。



**提到的链接**：<a href="https://github.com/ericaxelrod-1/model-context-protocol">GitHub - ericaxelrod-1/model-context-protocol: Model Context Protocols for Cursor</a>：Cursor 的 Model Context Protocols。通过在 GitHub 上创建账号来为 ericaxelrod-1/model-context-protocol 的开发做出贡献。

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1351693330672652451)** (85 条消息🔥🔥): 

> `Phi-4 Model, Claude's Response, Vibe Coding, Nvidia Open Sources Coding Dataset, Small Scale LLM Experiments` 


- **随着 Phi-4 的出色表现，辅助模型脱颖而出**：成员们讨论认为 **Phi-4** 在复杂系统中作为*辅助模型*非常有用，强调了它遵循指令、与其他 LLM 交互以及处理角色扮演的能力，并指出*在已经拥有大量其他模型的复杂系统中，它作为辅助模型会非常有用*。
- **解读 Claude 对语言的宽泛理解**：一位用户批评了 **Claude** 的 AI 建议，指出其在模型尺寸推荐方面存在偏差，因为它在回复中给出的模型列表并不符合要求的 *10m 参数*以下，参见 [Claude 的输出内容](https://claude.ai/share/03dcf20f-800a-4cdc-b961-30f4009555af)。
   - 一位成员回应称：*这很微妙，因为自然语言通常并不完全是字面意思，“10m”可能被视为“在现代硬件上普遍可用的极轻量级模型”的简称，公平地说，这在 Claude 的整个回答中是一个贯穿始终的主题。*
- **“Vibe Coding” 被验证为背景学习**：一位成员分享了通过沉浸和消费媒体学习日语的个人轶事，将其与 “vibe coding” 在背景学习和技能获取方面的价值进行了类比。
   - 他们表示：*即使是 vibe coding，你仍然需要担心模块之间的接口等问题，以便在有限的 LLM 上下文窗口（context windows）下保持扩展。*
- **Nvidia 发布重磅大规模编程数据集**：一位用户分享了 [Nvidia 开源的指令编程数据集](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1)，旨在提升 **Llama instruct 模型**在数学、代码、通用推理和指令遵循方面的能力，其中包括来自 **DeepSeek-R1** 和 **Qwen-2.5** 的数据。
   - 另一位下载了该数据集的成员报告称，对其进行过滤和训练将会非常有趣。
- **使用 RTX 3080 进行小规模 LLM 实验**：一位成员寻求帮助，希望在显存（VRAM）有限的情况下找到训练模型的*切入点*。
   - 讨论内容包括各种 QLoRA 实验，以及针对代码编辑进行微调的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pkrank.pages.dev)">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2502.01839">Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification</a>：基于采样的搜索是一种利用测试时计算的简单范式，涉及生成多个候选响应并选择最佳响应——通常通过让模型对每个响应进行自我验证...</li><li><a href="https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset-v1">nvidia/Llama-Nemotron-Post-Training-Dataset-v1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://vintagedata.org/blog/posts/model-is-the-product">The Model is the Product | Vintage Data</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1351642505459798169)** (75 messages🔥🔥): 

> `Lmarena 测试者, Perplexity vs OpenAI/Google, Gemini Deep Research vs GPT, LeCun 论 AGI, Grok 3 deepsearch` 


- **LMArena 被指责缓慢衰落且专家流失**：一名成员质疑 **LMArena** 上测试者的活跃度和质量，问道 *"Lmarena 是否正在慢慢死去？真正的测试者、改进者和真正的思考者都在哪里？"* 并分享了一个 [Calm Down GIF](https://tenor.com/view/calm-down-a-bit-calm-down-relax-gif-3234894596200294187)。
- **Sonar 如果被 OpenAI/Google 击败将面临生存危机**：一名成员推测，如果 **Perplexity/Sonar** 不能成为顶尖的基于 Web 的搜索，该公司将陷入困境，因为如果 **OpenAI** 或 **Google** 被评为同样好甚至更好，**Perplexity** 将失去其独特性。
   - 另一名成员指出 *"实际上没人在 Perplexity 上使用 sonar"*，暗示主要是 Pro 订阅在驱动收入。
- **Gemini Deep Research 可能节省 10 倍时间**：一位成员发现 **Gemini Deep Research** 的最新更新为他们节省了 **10 倍**的时间，并认为它对于 LMArena 基准测试来说太贵了。
   - 另一名成员补充道，Gemini 给出了极好的结果，为科学医学研究提供了深刻的分析，甚至生成了一份包含 **90 个文献来源**的列表。
- **LeCun 揭穿扎克伯格的 AGI 炒作**：[Yann LeCun](https://aibusiness.com/responsible-ai/lecun-debunks-agi-hype-says-it-is-decades-away) 警告说，实现 **AGI** *"将需要数年甚至数十年"*，需要新的科学突破。
   - 最近的一篇文章称，[Meta LLMs 在 2025 年之前不会达到人类水平的智能](https://www.pymnts.com/artificial-intelligence-2/2025/meta-large-language-models-will-not-get-to-human-level-intelligence/?utm_source=chatgpt.com)。
- **Grok 3 Deep Search 评价褒贬不一**：一位用户发现 **Grok 3** 的 **deepersearch** 功能令人失望，理由是存在幻觉和低质量结果。
   - 然而，另一位用户为 **Grok** 辩护，称 *"deepersearch 看起来相当不错"*，但原评论者反驳说，频繁使用会暴露出大量的错误和幻觉。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/calm-down-a-bit-calm-down-relax-gif-3234894596200294187">Calm Down A Bit Relax GIF - Calm Down A Bit Calm Down Relax - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/jake-crying-gif-15896901980625544793">Jake Crying GIF - Jake crying - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/collections/nvidia/llama-nemotron-67d92346030a2691293f200b">Llama Nemotron - nvidia 集合</a>：未找到描述</li><li><a href="https://aibusiness.com/responsible-ai/lecun-debunks-agi-hype-says-it-is-decades-away">Meta 的 LeCun 揭穿 AGI 炒作，称其还需数十年</a>：Meta 首席 AI 科学家 Yann LeCun 对 AGI 持怀疑态度，尽管他的老板、CEO Mark Zuckerberg 正全力投入。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1351707669962035263)** (9 messages🔥): 

> `数字比较, 内存带宽, Macbook 上的 Gemma 3` 


- **征求数字比较的想法**：一位用户询问了 **Jake** 在之前的直播中将数字与其他替代方案进行的比较，表示有兴趣阅读他的想法。
- **内存带宽的忧郁**：一位用户评论了某款发布产品的低内存带宽（推测用于本地推理），并表示 *“我快变成镇上的抱怨者了。在我看来，这就像是一个带更多显存的 5070”*。
- **Gemma 3 的量化计算**：一位用户询问关于在 **M1 Macbook Pro** (16GB) 上运行 **Gemma 3** 的模型大小和量化问题，另一位用户解释了如何根据模型大小和以字节为单位的量化来计算尺寸需求。
   - 该用户解释说，可以运行 FP16 格式的 **Gemma 3 4B**，如果有 FP4 格式的 **12B 模型** 也可以运行，因为 Macbook 的 **16GB** 统一内存中有 **70%** 分配给了 GPU。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1351683205056696351)** (4 条消息): 

> `Blackwell ULTRA 的 attention instruction, Shared Memory Carveout` 


- **Blackwell ULTRA 将带来新的 'attention instruction'？**: 一位成员提到 **Blackwell ULTRA** 将带来一个 *attention instruction*，但其具体含义尚不明确。
- **Shared Memory Carveout 详解**: 一位成员指出，如果 kernel1 的 **smem carveout** 只有 **100 或 132 KiB**，则没有足够的空间让两个 kernel 同时运行。
   - 他建议参考 [关于 Shared Memory 的 CUDA 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x) 来增加 carveout。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1351708883005210766)** (7 条消息): 

> `torch.distributed.tensor.parallel.style.ColwiseParallel, Autograd hook 保证, 为 RTX 5080 从源码编译 PyTorch` 


- **`ColwiseParallel` 及其相关类缺失 `__repr__()`**: 用户询问为何 `torch.distributed.tensor.parallel.style.ColwiseParallel` 及相关类中缺少 `__repr__()` 方法，并寻求对该设计选择的见解。
   - 在给定语境中未提供具体解释。
- **Autograd Hook 每个参数仅调用一次**: 一位成员寻求澄清 [autograd hook](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_tensor.py#L683-L686) 是否保证在每次 backward pass 中每个参数仅被调用一次，即使在像 RNN 这样具有 gradient accumulation 的场景中也是如此。
   - 回复确认 *hook 确实每个参数仅调用一次*，并澄清了其在 gradient accumulation 方面的行为。
- **用户为 RTX 5080 从源码编译 PyTorch**: 一位用户尝试从源码编译 **PyTorch** 以支持 **RTX 5080**，尽管设置了 `TORCH_CUDA_ARCH_LIST=Blackwell`，但编译结果仅支持 `sm_100`，而不支持 `sm_120`。
   - 另一位成员建议改用 `TORCH_CUDA_ARCH_LIST="10.0,12.0"`；用户对该建议表示感谢。



**提到的链接**: <a href="https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_tensor.py#L683-L686">pytorch/torch/_tensor.py at v2.6.0 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1351736909952712834)** (20 条消息🔥): 

> `matmul output fusions, nvfuser stalls, Tensor Cores vs CUDA Cores, cooperative 或 ping-pong warp specialization, 在 GEMM 中融合 activation` 


- **nvfuser 的 Matmul Output Fusions 面临障碍**: 一位成员指出，为 [**nvfuser** 实现 matmul output fusions](https://github.com/rdspring1) 非常困难，即使是乘法/加法也会引入 stalls，由于需要保持 Tensor Cores 的数据供给，这使得它比独立 kernel 更慢。
   - 他对优化这些内容如此困难感到惊讶。
- **TC 和 CUDA Core 并发执行？**: 一位成员询问困难是否源于 **Tensor Cores** 和 **CUDA Cores** 无法并发运行，从而可能在寄存器使用上产生竞争。
   - 他们回想起 Blackwell 文档提到 **TC** 和 **CUDA cores** 现在可以并发运行。
- **Autotuning Fusion 策略**: 一位成员对 fused operations 的 autotuning 表示好奇，询问哪些策略可以隐藏被融合的操作，特别是考虑到存在多种选项且必有一个是最佳的。
   - 另一位成员指出，在 **Hopper** 上，基本上有两种选择：cooperative 或 ping-pong warp specialization。
- **在 GEMM 中融合 Activation 的性能影响**: 在 **gpu-mode lecture 45** 中讨论过，如果 **GEMM** 使用了所有寄存器，在 GEMM 中融合 activation 有时会损害性能，此时将 **GEMM** 和 activation 拆分为两个 kernel 会更快。
   - 一位成员在编写自定义的 fused **GEMM+activation Triton kernels** 时也遇到了类似的结果。



**提到的链接**: <a href="https://github.com/rdspring1">rdspring1 - Overview</a>: I contribute to PyTorch, Lightning-AI Thunder, and Nvidia/Fuser. - rdspring1

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

mobicham: https://www.youtube.com/watch?v=1bRmskFCnqY
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1351752223159226450)** (3 messages): 

> `FSDP1, FSDP2, accelerate, trl` 


- ****Accelerate** 准备合并 **FSDP2** 支持**: 一位成员询问 `accelerate` 使用的是 **FSDP1** 还是 **FSDP2**，以及是否可以使用 **FSDP2** 配合 `trl` 对 LLM 进行微调。
   - 另一位成员回答说 `accelerate` 目前使用的是 **FSDP1**，但他们计划在“下周左右”合并[一个 pull request](https://github.com/huggingface/accelerate/pull/3394) 以添加对 **FSDP2** 的初步支持。
- **用户对 **FSDP2** 的兴奋**: 在成员澄清了添加 **FSDP2** 支持的具体细节后，另一位成员表示：“这太令人兴奋了！感谢澄清！”
   - 这凸显了用户一直在期待 `accelerate` 中 **FSDP2** 支持的到来。



**Link mentioned**: <a href="https://github.com/huggingface/accelerate/pull/3394">WIP: Initial FSDP2 support by S1ro1 · Pull Request #3394 · huggingface/accelerate</a>: Draft PR, feel free to discuss changes to the user-facing api.Fixes # (issue)Before submitting This PR fixes a typo or improves the docs (you can dismiss the other checks if that&amp;#39;s the ca...

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1351803203426979871)** (1 messages): 

> `IRL Meetup, GTC Meetup, Saturday Evening Event` 


- **线下聚会 (IRL Meetup) 势头渐起**: 一位成员建议在 **GTC** 期间的晚上或周末举行**线下聚会**，并提议在**周六晚上**举办活动。
   - 该成员自荐负责**预订桌位**，并邀请感兴趣的人对评论做出回应以统计人数。
- **GTC After-Party 酝酿中**: 爱好者们正在考虑举行一次与 **GTC** 活动同步或在其之后的 **IRL meetup**，可能会延续到周末。
   - 该倡议旨在将人们从数字世界连接起来，提供在轻松的面对面环境中进行社交和协作的机会。


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1351669146173046795)** (2 messages): 

> `Liger Kernel, Fused Linear Cross Entropy` 


- **Liger Kernel 补丁无法与 tp_plan 配合使用**: `tp_plan:{"lm_head"="colwise_rep"}` 设置无法与 liger 的 `fused_linear_cross_entropy` 补丁配合使用，因为它本质上需要 loss parallel。
   - 用户被告知欢迎提出功能请求 (feature requests)，他们询问是否有可以参与贡献的开发进展。
- **寻求贡献机会**: 一位用户询问了有关 Liger Kernel 相关开发中的贡献机会。
   - 这表明用户对参与项目的进步和功能增强具有潜在兴趣。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

viking0nfire: 我们在 https://LeetGPU.com/challenges 上发布了 Triton 支持 🚀 

快来看看吧！
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1351723644384182332)** (3 messages): 

> `Automatic Kernel Optimization, Single GPU context, Distributed GEMM` 


- **Kernel 优化局限于单 GPU 场景**: 自动 Kernel 优化目前仅在**单 GPU 上下文**中提出，因此无需考虑扩展到多 GPU。
   - *没有理由期望当扩展到多 GPU 时 `argmin_x f(x) = argmin_x g(x)`*，尽管由于大多数分布式训练算法的简单性，情况通常确实如此。
- **分布式 GEMM**: 理想情况下，优化问题应该在**分布式设置**下进行构建和计算，许多 Kernel 实现是专门为此目的设计的。
   - 发言者回想起该服务器曾有过一次关于分布式 GEMM 的演讲。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1351706816316309555)** (1 条消息): 

> `ThunderKittens, Kernels, Batch Compilation, GPU Programming` 


- **ThunderKittens Kernel 需要修复**：一名成员发现 **ThunderKittens** repo 中的[第一个示例 Kernel](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/example_bind/example_bind.cu#L8) 无法直接编译。
   - 他们必须添加对 **batch**、**depth**、**row** 和 **cols** 方法的调用才能使其正常工作。
- **解决 Kernel 编译问题**：示例 Kernel 中的编译错误突显了 **ThunderKittens** 仓库中需要更清晰的文档或更新示例。
   - 这能确保新用户可以轻松上手该库，并理解 Kernel 编译所需的必要方法调用。



**提到的链接**：<a href="https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/example_bind/example_bind.cu#L8">ThunderKittens/kernels/example_bind/example_bind.cu at main · HazyResearch/ThunderKittens</a>：用于快速 Kernel 的 Tile 原语。通过在 GitHub 上创建账户为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1351710596793630770)** (4 条消息): 

> `DAPO Algorithm, RL Training, PPO, GRPO` 


- **DAPO 算法随开源发布亮相**：**DAPO 算法**（*decoupled clip and dynamic sampling policy optimization*，解耦裁剪与动态采样策略优化）已发布；**DAPO-Zero-32B** 超越了 **DeepSeek-R1-Zero-Qwen-32B**，在 **AIME 2024** 上获得 **50** 分，且步数减少了 50%；它基于 **Qwen-32b** 预训练模型通过 **zero-shot RL** 训练而成，算法、代码、数据集、验证器和模型已完全开源，基于 [Verl](https://verl.ai/) 构建。
   - 主页可在此处访问 [here](https://dapo-sia.github.io/)，论文可在此处访问 [here](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)，代码可在此处访问 [here](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo)。
- **动态采样策略总结**：动态采样策略包括：*过滤掉没有梯度信号的生成结果 (completions)*（相对于组内成员没有变化）、token 级策略梯度损失（较长序列对梯度信号贡献更多），以及超长过滤/软超长惩罚（*移除超过最大长度的生成结果*）。



**提到的链接**：<a href="https://x.com/eric_haibin_lin/status/1901662955307200974?t=xoSHjL0l7B79jrodIrkWhA&s=19">来自 Haibin (@eric_haibin_lin) 的推文</a>：@qiying_yu 及其团队刚刚发布了 DAPO 算法（decoupled clip and dynamic sampling policy optimization）！DAPO-Zero-32B 作为一个完全开源的 RL 推理模型，超越了 DeepSeek-R1-Zero-Qwen-32...

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1351742944205471807)** (11 条消息🔥): 

> `Modal Runner Successes, conv2d Leaderboard, vectoradd Leaderboard, vectorsum Leaderboard, grayscale Leaderboard` 


- **Modal Runner 在多个基准测试中取得成功**：**Modal runner** 在包括 **vectoradd**、**vectorsum**、**conv2d** 和 **grayscale** 在内的多个排行榜的测试和基准提交中取得了成功。
   - 提交内容正在各种 GPU 上进行测试，包括 **H100**、**A100**、**T4** 和 **L4**。
- **Conv2d 排行榜竞争激烈**：多个 **conv2d** 排行榜提交在各种 GPU 上使用 **Modal runner** 成功运行。
   - 成功的提交涵盖了 **L4**、**T4**、**A100** 和 **H100** GPU，展示了广泛的兼容性。
- **Vectoradd 基准测试取得成功**：在 **GPUS: A100** 上使用 **Modal runner** 向 **vectoradd** 排行榜提交的测试成功了！
   - 另一个 ID 为 `2273` 的测试提交在 **GPUS: L4** 上使用 **Modal runner** 向 **vectoradd** 排行榜提交也获得了成功！
- **Vectorsum 排行榜见证 Modal 的成功**：在 **GPUS: T4** 上使用 **Modal runner** 向 **vectorsum** 排行榜提交的测试和基准测试均告成功。
   - 成功的提交表明 **vectorsum** 在使用 **Modal** 框架的 **T4** GPU 上具有稳定的性能。
- **Grayscale 基准测试提交成功**：在 **GPUS: H100** 上使用 **Modal runner** 向 **grayscale** 排行榜提交的基准测试获得成功。
   - 这一成功突显了 **Modal runner** 在高端 **H100** GPU 上处理图像处理任务的能力。


  

---

### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1351940110144376893)** (1 messages): 

> `OpenMP Performance, printf Performance Impacts, std::cout Side Effects` 


- **printf 让 OpenMP 变得更快？**：在 **cp2b** 中使用 **OpenMP** 时观察到了一个奇怪的性能现象：在使用带有 schedule 子句的 `||omp parallel for` 时，在代码末尾包含一个 `printf` 语句会使其运行得更快（0.4s 对比 0.6s）。
   - 当使用 `std::cout` 代替 `printf` 时，该效应消失，且包含 `<iostream>` 会完全抵消这种效应。
- **编译器异常**：用户报告了一个关于 `printf` 影响其代码中 **OpenMP** 性能的异常，指出存在 `printf` 时执行速度更快。
   - 这种行为在使用 `std::cout` 时无法复现，表明 `printf`、**OpenMP** 和编译器之间可能存在某种交互。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1351653934665699359)** (11 messages🔥): 

> `LeetGPU challenges, GTC talks, Nvidia Blackwell Ultra, Nvidia Ruben, Silicon Photonics` 


- **LeetGPU 挑战赛上线！**：鼓励成员尝试 [LeetGPU challenges](https://leetgpu.com/challenges)。
   - 另一名成员因晋级到第 4 级而受到祝贺。
- **寻求 GTC 演讲链接**：一名成员询问 **GTC talks** 的链接。
   - 另一名成员回复称，可以在 **Nvidia** 网站上免费注册虚拟参会，并在演讲结束后 72 小时内观看录像，且 **Jensen 的演讲** 已上传至 YouTube。
- **Nvidia Keynote 摘要**：Keynote 包含了 **Blackwell Ultra** 和 **Ruben** 的发布公告，下一代 GPU 代号为 **Feynman**。
   - 随着 **Ruben** 的推出，出于数据传输功耗成本的考虑，**Nvidia** 正在转向 **silicon photonics**（硅光子技术），**Ruben** 还将配备一个新的 **ARM CPU**，并对 **Spectrum X** 进行大量投资，推出 **1.6 Tbps 交换机**。



**提及的链接**：<a href="https://leetgpu.com/challenges">LeetGPU</a>：未找到描述

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1351632839816380566)** (28 messages🔥): 

> `CompactDict advantages, HashMap in stdlib, List.fill behavior, StringDict module, List index out of range` 


- **CompactDict 的优势引发关注**：成员们讨论了 **CompactDict** 的优势，一位用户仍受困于内置 dict 的 **SIMD-Struct-not-supported 问题**，对替代方案感到非常满意。
   - 大约一年前在 [GitHub](https://github.com/mzaks/compact-dict) 上发布了一份报告，详细介绍了两种专门的 Dict 实现，一种用于 **Strings**，另一种强制实现 **Keyable** trait。
- **标准库引入 HashMap 的建议获得支持**：有建议将 *generic_dict* 作为 **HashMap** 包含在 **standard library**（标准库）中，保持现有的 **Dict** 不变。
   - 有人担心 **Dict** 需要处理许多非静态类型的事情，因此添加一个设计更好的新 struct 并逐步弃用 **Dict** 可能更有价值。
- **List.fill 的行为让用户感到意外**：有人质疑填充 **lists buffer** 未初始化部分是否应该是可选的，因为调用 **List.fill** 可能会意外地改变列表的长度。
   - 建议将填充列表缓冲区未初始化部分设为可选，因为调用 `List.fill` 可能会出人意料地改变列表的 **length**（长度）。
- **StringDict 模块被认为是可行的**：建议标准库应采纳 **CompactDict** 的一些特性，如更好的哈希函数和更简单的探测策略，但所有的压缩特性超出了标准库 **Dictionary** 的范畴。
   - 作者仍认为拥有压缩的 **StringDict** 是可行的，但它可以作为一个独立的模块包含。
- **List 索引越界未被察觉**：用户注意到 **List** 中没有索引越界检查，对此感到惊讶，因为他们认为这就是 **unsafe_get** 的用途。
   - 另一名成员也遇到了这个问题，Modular 的人员表示这需要在“某个时间点”添加。



**提及的链接**：<a href="https://github.com/mzaks/compact-dict">GitHub - mzaks/compact-dict: A fast and compact Dict implementation in Mojo 🔥</a>：Mojo 🔥 中一个快速且紧凑的 Dict 实现。可以通过创建账户为 mzaks/compact-dict 的开发做出贡献。

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1351639582877421700)** (31 条消息🔥): 

> `Patronus AI, Etsy, Nvidia GTC Keynote, Manus access trading bot, vLLM` 


- **Patronus AI 评判诚实度**：[Patronus AI](https://www.patronus.ai/) 推出了 **MLLM-as-a-Judge** 用于评估 AI 系统，[Etsy](https://www.etsy.com/) 已经采用该系统来验证商品图片的描述准确性。
   - Etsy 拥有*数亿件商品*，需要确保其描述准确且不产生幻觉（hallucinated）。
- **Cognition AI 估值达到 40 亿美元**：**Cognition AI** 在由 Lonsdale 公司领投的一笔交易中达到了 **40 亿美元估值**（链接未提供）。
- **AWS 价格低于 Nvidia**：在 [GTC 2025 年 3 月主题演讲](https://www.youtube.com/watch?v=_waPvOwL9Z8)期间，据 [这条推文](https://x.com/_fabknowledge_/status/1902092480616497395) 报道，*AWS 的 Trainium 定价仅为 Nvidia 芯片（Hopper）价格的 25%*。
   - 黄仁勋（Jensen）开玩笑说，在 Blackwell 之后，他们可以免费赠送 Hopper，因为 Blackwell 的性能太强了。
- **Manus 访问权限令用户印象深刻**：一位成员获得了通过 **Grok3** 进行深度搜索的 Manus 访问权限，并表示*效果很好*，展示了它如何在周末构建了一个交易机器人（trading bot），但目前在模拟交易（paper trading）中亏损了约 1.50 美元。
   - 他们展示了令人印象深刻的输出，并分享了*预览*截图。
- **vLLM 成为推理界的 ffmpeg**：根据 [这条推文](https://x.com/vllm_project/status/1902068326312124815)，**vLLM** *正逐渐成为 LLM 推理领域的 ffmpeg*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://venturebeat.com/ai/patronus-ais-judge-image-wants-to-keep-ai-honest-and-etsy-is-already-using-it/">Patronus AI 的 Judge-Image 旨在保持 AI 的诚实 —— Etsy 已经在使用它</a>：Patronus AI 推出了首个用于评估处理图像的 AI 系统的多模态 LLM-as-a-Judge，Etsy 已经实施该技术来验证其市场上的商品图片描述...</li><li><a href="https://x.com/steph_palazzolo/status/1902419345088635187?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：在去年秋天的一次会议上，DoorDash 向 OpenAI 提出了一个关于其即将推出的 Operator Agent 的问题：如果机器人而不是人类占领了我们的 App，会发生什么？这是零售商和公司之间日益关注的问题...</li><li><a href="https://x.com/vllm_project/status/1902068326312124815">来自 vLLM (@vllm_project) 的推文</a>：我们感谢对 vLLM 的信任 ❤️</li><li><a href="https://x.com/apolloaievals/status/1901713042578325891">来自 Apollo Research (@apolloaievals) 的推文</a>：AI 模型 —— 尤其是 Claude Sonnet 3.7 —— 通常能意识到自己正在接受对齐（alignment）评估。这是一个 Claude 在沙盒评估期间的推理示例，它从中学习...</li><li><a href="https://x.com/metr_evals/status/1902384481111322929?s=46">来自 METR (@METR_Evals) 的推文</a>：AI 系统何时能够独立执行长期项目？在新的研究中，我们发现了一种“AI Agent 的摩尔定律”：AI 能够完成的任务长度大约每 7 个月翻一番...</li><li><a href="https://x.com/kmohan2006/status/1902091083477385271?s=46">来自 Krishna Mohan (@KMohan2006) 的推文</a>：Dense MoE 和 Sparse MoE 的实现</li><li><a href="https://x.com/_fabknowledge_/status/1902092480616497395">来自 Fabricated Knowledge (@_fabknowledge_) 的推文</a>：“AWS 的 Trainium 定价为 Nvidia 芯片（Hopper）价格的 25%” Jensen：在 Blackwell 之后你可以免费赠送 Hopper，因为 Blackwell 的性能将非常强大。你可以计算一下谁在总成本上获胜...
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1351727526170460222)** (2 条消息): 

> `Cloudflare Agents 播客，Evo2 论文讨论` 


- **Latent Space 讨论 Cloudflare Agents**：[Latent Space 播客](https://x.com/latentspacepod/status/1902168744530718758) 由 @swyx、@ritakozlov_ 和 @threepointone 主持，深入探讨了 **Cloudflare** 本年度的 **agent** 产品，包括关于 **Durable Objects** 的环节。
   - 对话还涉及了 **Sapir-Whorf 假说**、**observability**（可观测性）、让 **normies**（普通用户）使用 agent 的策略、**obligatory MCP**、工作流以及对 **Sunil's Blog** 的引用。
- **Evo2 论文俱乐部进行中**：Latent Space 社区齐聚一堂讨论 **Evo2 论文**，重点关注大规模 **Convolutional Multi-Hybrid Language Models** 的系统和算法。
   - 该环节由一名社区成员主持，探讨了论文中的核心概念，旨在揭开 **large language models** 复杂层面的神秘面纱。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1902168744530718758">来自 Latent.Space (@latentspacepod) 的推文</a>：新 ⚡️ 播客：@swyx 与 @ritakozlov_ 和 @threepointone 畅谈 @CloudflareDev 在 Agent 之年的所有产品 `npm i agents` https://youtu.be/8W_9lUYGa2U 包含对 Durable Objects, Sapir Whorf 的即兴讨论...</li><li><a href="https://lu.ma/ps82kaee">LLM 论文俱乐部 (Evo 2: Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale) · Zoom · Luma</a>：RJ 将讲解 https://arcinstitute.org/manuscripts/Evo2-ML 这是新闻稿：https://arcinstitute.org/news/blog/evo2 以及配套的生物论文：…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1351902723586392116)** (2 条消息): 

> `重新与社区建立联系，去中心化 AI 讨论` 


- **热烈回归社区**：一名成员表达了在休息一段时间后重新与社区建立联系的兴奋之情。
- **对去中心化 AI 讨论的期待**：同一位成员期待着重新投入到去中心化 AI 的讨论中。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1351699120544546938)** (21 条消息🔥): 

> `微调 Gemini/OLMo，微调中的蒸馏，Passkey/Fuzzy Rate 改进，GoldFinch/GoldenGoose 混合设置，Memory Expert 激活` 


- **Gemini 和 OLMo 模型引发微调热潮**：一名成员就微调 **Gemini** 或 **OLMo** 模型寻求建议，并询问蒸馏是否是更好的方法，特别是当数据为 PDF 文件时。
   - 虽然没有提供直接答案，但讨论延伸到了内存优化和旨在增强性能的混合设置。
- **通过 Fuzzy Rate 技巧提升 Passkey 性能**：一名成员建议通过混合方法或由 passkey 激活的 memory expert，将重要 key 的 **passkey 和 fuzzy rate** 提高到接近 100%，如 [scrot.png](https://cdn.discordapp.com/attachments/747850033994662000/1351769065001062420/2025-03-19-040425_768x772_scrot.png?ex=67dc3d4b&is=67daebcb&hm=54777a36b97376a2d0b4b470c683ee5dbd9aedad9f3b2bb4febfe00792f6f6e4&) 所示。
- **计划采用 GoldenGoose 混合设置进行内存优化**：一名成员对用于内存管理的 **GoldFinch** (**GoldenGoose**) 混合设置表示感兴趣，并提到计划实施该方案。
   - 他们指出，更大的模型将拥有更长的记忆，图表展示了从 **1.5B** 到 **2.9B** 参数的改进。
- **带记忆的 RWKV 更新规则**：一名成员提议尝试使用结合了后续输入和记忆的输入来实验 **RWKV** 更新规则，可能结合 **MoE** 或门控机制。
   - 由于正忙于转换 **QRWKV** 模型，他们尚未测试该策略，但希望未来能进行尝试。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.14492">arXiv reCAPTCHA</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2503.14456">RWKV-7 "Goose" with Expressive Dynamic State Evolution</a>：我们介绍了 RWKV-7 "Goose"，一种新的序列建模架构，以及预训练语言模型，这些模型在 30 亿参数级别的下游任务性能中树立了新的 state-of-the-art...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1351693270291583038)** (1 messages): 

> `Latent Activations, Model Behavior, Sequence Processing` 


- **Latent Activations 代表完整序列**：为了理解模型的正常行为，应该从“整个序列”而不是单个 token 生成 **latent activations**。
   - 发布者建议，关注“整个序列”能更准确地表示 **model** 的正常运作。
- **使用 get_activations(sequence) 生成 Latents**：发布者提供了代码示例，推荐将 `latents = get_activations(sequence)` 作为“正确”的方法。
   - 他们警告不要使用 `latents = cat([get_activation(tok) for tok in sequence))`，因为这会产生“无趣”的 **latents**。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1351954335839289376)** (5 messages): 

> `lm_eval, HFLM, trust_remote_code, API keys` 


- **HFLM 需要远程信任**：一位成员询问如何以编程方式运行带有 `HFLM` 模型的 `lm_eval` 并避免 `trust_remote_code=True` 错误。
   - 另一位成员澄清说，`trust_remote_code=True` 必须传递给模型构造函数，命令行标志只是在内部使用该标志构建模型。
- **云端模型需要 API Keys**：两位成员询问无法在本地托管的云端模型是否支持 API keys。
   - 另一位成员确认它们支持，并指向了之前提供的[详细信息链接](https://link-to-details)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1351947425052622960)** (2 messages): 

> `LlamaIndex course on HuggingFace, Gen AI Toolbox for Databases` 


- **Hugging Face 教授 LlamaIndex Agents**：[Hugging Face](https://huggingface.co/) 发布了一门关于在 **LlamaIndex** 中构建 agents 的**免费**课程，涵盖了组件、RAG、tools、agents 和 workflows ([链接](https://t.co/eACAJzXg8y))。
- **Google 和 LlamaIndex 简化 AI Agents**：**LlamaIndex** 与 [Google Cloud](https://cloud.google.com/) 合作，利用 **Gen AI Toolbox for Databases** 简化 AI agents 的构建 ([链接](https://t.co/ocvPTUxvRO))。
   - **Gen AI Toolbox for Databases** 负责管理复杂的数据库连接、安全性和工具管理，更多详情可在 [Twitter](https://twitter.com/llama_index/status/1902387501492584580) 上查看。



**提到的链接**: <a href="https://t.co/eACAJzXg8y">Introduction to LlamaIndex - Hugging Face Agents Course</a>: 未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1351650048982908969)** (24 messages🔥): 

> `Langchain 的长期记忆 vs LlamaIndex, Azure OpenAI 与 LlamaIndex, 简历解析 AI 服务, Agent 工具调用` 


- **LlamaIndex 与 Langchain 的长期记忆能力对比**：一位成员询问 LlamaIndex 是否具有类似于 [Langchain 在 LangGraph 中支持的长期记忆](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/) 的功能。
   - 另一位成员指出，“在 Langchain 的情况下，长期记忆只是一个向量存储”，并建议使用 LlamaIndex 的 [Composable Memory](https://docs.llamaindex.ai/en/stable/examples/agent/memory/composable_memory/)。
- **LlamaIndex 中的 Azure OpenAI 集成**：一位用户在使用结构化输出时遇到了 `ChromaVectorStore` 以及随后的 `AzureOpenAI` 的 `AttributeError`，但随后解决了 `ChromaVectorStore` 的问题。
   - 他们随后询问 LlamaIndex 是否仅支持 OpenAI Agent 而不支持 Azure，但另一位成员确认“你可以将 Azure 传入任何 Agent”，并指向 `llm.structured_predict` 以使用 Pydantic schema 进行 **Azure Structured Predict**。
- **简历解析 AI 服务**：一位成员征求 **简历解析 AI 服务** 的推荐。
   - 另一位成员回答说他们“查到了一些”，但没有具体说明是哪些。
- **Agent 工具被重复调用**：一位用户报告说，一个 Agent 为了同一个目标连续两次调用了同一个工具。
   - 另一位成员建议，“代码执行工具应该返回类似 ‘代码执行正确’ 之类的内容，以便 LLM 知道它已经生效了”。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/">在 LangGraph 中推出长期记忆支持</a>：今天，我们很高兴地宣布在 LangGraph 中迈出支持长期记忆的第一步，该功能在 Python 和 JavaScript 中均可使用。长期记忆允许你在不同会话之间存储和检索信息...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/memory/composable_memory/">简单可组合内存 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1351858929910681641)** (1 messages): 

> `Nebius AI 计算平台, GPU 集群, 推理服务, AWS vs Lambda Labs vs CoreWeave` 


- **征求 Nebius 平台使用经验**：一位成员对 **Nebius** 用于 **AI** 和 **机器学习工作负载** 的计算平台的实际使用经验感到好奇，包括其 **GPU 集群** 和 **推理服务**。
   - 他们正在将其与 **AWS**、**Lambda Labs** 或 **CoreWeave** 在 **成本**、**可扩展性** 和 **部署简易性** 方面进行比较，并希望了解其 **稳定性**、**网络速度** 以及 **Kubernetes** 或 **Slurm** 等 **编排工具** 的情况。
- **云端 AI 平台对比**：讨论旨在将 **Nebius** 与 **AWS**、**Lambda Labs** 和 **CoreWeave** 等成熟平台进行对比。
   - 关键对比点包括大规模 **AI 工作负载** 的 **成本效益**、**可扩展性** 和 **部署简单性**。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1351981788058091594)** (4 messages): 

> `Chase 的全名, Cohere Expanse 32B 知识截止日期` 


- **用户询问 Chase 的全名**：一位用户请求另一位用户分享一个名叫 Chase 的人的全名，并提议为他们建立联系。
   - 他们随后表示已经为他们安排好了一些事情。
- **寻求 Cohere Expanse 32B 知识日期**：一位用户询问 **Cohere Expanse 32B** 的知识截止日期。
   - 该用户正在寻找新工作。


  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1351944343375708272)** (19 messages🔥): 

> `Trial Key Usage Tracking, Websearch Connector Issues, command-r-plus-02-2024 vs command-a-03-2025` 


- **Trial Key 触发速率限制 (Rate Limits)**：一位用户在使用 Trial Key 时遇到了 **429 错误**，并寻求帮助以追踪使用情况，怀疑自己是否超过了**每月 1000 次调用**的限制。
   - Cohere 团队成员提议进行调查，并澄清 Trial Key 受速率限制约束，同时指出了 [Cohere rate limits documentation](https://docs.cohere.com/v2/docs/rate-limits)。
- **Websearch Connector 结果不佳**：一位用户报告 **websearch connector** 性能下降，观察到*其实现方式似乎最近发生了变化*，导致返回结果变差。
   - 团队成员请求提供详细信息以便调查，并指出连接选项 *site: WEBSITE* 在限制特定网站查询时失效，该修复程序即将发布。
- **模型输出对比：command-r-plus-02-2024 vs command-a-03-2025**：用户测试并对比了 **command-r-plus-02-2024** 和 **command-a-03-2025** 模型的网页搜索结果。
   - 据报告，两者的输出没有区别，且在更多情况下网页搜索根本不返回结果。



**提及的链接**：<a href="https://docs.cohere.com/v2/docs/rate-limits">Different Types of API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 针对生产和评估密钥的速率限制。

  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1352013136827519071)** (1 messages): 

> `MCP Server, Cohere Command A, Github Repo, Positive News` 


- **使用 Cohere 构建的 Goodnews MCP Server**：一位成员构建了一个 **Goodnews MCP** server，通过其工具 `fetch_good_news_list` 使用 **Cohere Command A** 为 MCP 客户端提供积极、令人振奋的新闻。
   - 该服务器通过 API 请求获取最新头条，并使用 **Cohere LLM** 对文章进行排名并返回最积极的内容。
- **Github 仓库已发布**：该项目的 Github 仓库已发布在[此处](https://github.com/VectorInstitute/mcp-goodnews)。
   - 该仓库包含一个简单的服务器，旨在为 MCP 客户端提供精选的积极新闻。



**提及的链接**：<a href="https://github.com/VectorInstitute/mcp-goodnews">GitHub - VectorInstitute/mcp-goodnews: A simple MCP application that delivers curated positive and uplifting news stories.</a>：一个简单的 MCP 应用程序，提供精选的积极和令人振奋的新闻故事。- VectorInstitute/mcp-goodnews

  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1351700703306977403)** (3 messages): 

> `AI alignment, Open-source models, Federating RAG, Agentic apps, Python and Rust` 


- **Aymara 推动 AI Alignment 民主化**：**Aymara** 的联合创始人 Juan Manuel 介绍了自己，并分享了他的公司正在构建**用于衡量和改进 AI alignment 的开发者工具**，希望能帮助开源模型的 AI alignment 实现民主化。
   - 他提到很高兴能探索将 **Command A** 用于内部工具，并欢迎通过私信讨论 AI alignment 的想法。
- **Andrei 探索联邦 RAG (Federated RAG)**：来自 **Vector Institute**（曾就职于 LlamaIndex）的 Andrei 介绍了自己，提到他目前正在开发几个开源项目，其中最著名的是 **federating RAG**。
   - 他计划很快转向一些 **agentic apps/research**，偏好使用 **Python 和 Rust**，并希望从社区获得建议、学习新方法和行业趋势。

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1351707407847526490)** (1 条消息): 

> `MOOC Coursework, AgentX Competition, LLM Agents Discord, AgentX Research Track` 


- ****MOOC Coursework** 详情及证书指南公布**：LLM Agents MOOC 的课程作业和结业证书指南已在 [课程网站](https://llmagents-learning.org/sp25) 发布。
   - 本课程基于 [2024 秋季 LLM Agents MOOC](https://llmagents-learning.org/f24) 的基础内容构建。
- **AgentX Competition：深入了解 **5 个等级****：AgentX 竞赛的详细信息已发布，包括报名方式，请点击 [此处](https://rdi.berkeley.edu/agentx/)。
   - 竞赛包含 **5 个等级**：Trailblazer ⚡, Mastery 🔬, Ninja 🥷, Legendary 🏆, Honorary 🌟。
- **立即申请：AgentX Research Track 导师指导**：选定学生可以通过 [此申请表](https://forms.gle/E2D69euNBjSmYsK28) 申请 AgentX Research Track 项目的导师指导。
   - 申请截止日期为 **3 月 26 日** **11:59pm PDT**。
- **Labs 和证书申报表将于 4 月发布**：**Labs** 和 **证书申报表** 将在 4 月份发布。
   - 作业暂定于 5 月底截止，证书将于 6 月发放。



**提到的链接**：<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>：MOOC，2025 春季

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1351708829687091200)** (23 条消息🔥): 

> `Quiz Deadlines, AgentX Research Track details, Certificate for December MOOC` 


- **测验截止日期为 5 月底**：成员询问了测验的截止日期以及是否仍可提交以获取证书资格，一名成员确认 *所有作业均在 5 月底截止*。
   - 另一名成员澄清说，这包括 *每节课后发布的测验*。
- **AgentX Research Track 详情**：一名成员询问了 **AgentX Research Track** 的导师指导形式、时间线、交付成果和选拔标准。
   - 一位导师解释说，将在 **3 月 31 日至 5 月 31 日** 期间为研究赛道项目提供指导，导师将直接联系申请人进行潜在面试，申请人应具备深思熟虑的研究想法。
- **导师澄清选拔并非基于前百分之几的比例**：一位导师澄清说，AgentX 研究赛道的选拔标准 *不是“我们只接受前 X%”*，而是申请人是否合格以及是否有深思熟虑的项目想法。
   - 他们进一步建议，申请人应展示出主动性、与课程相关的成熟研究想法，以及在两个月时间内推进该想法的背景能力。
- **12 月 MOOC 证书**：一位参加了 **12 月 MOOC 课程** 的成员反映，尽管完成了所有要求，但未收到证书。
   - 一位导师回复称，*证书邮件* 已于 **2 月 6 日** 发送，并建议检查垃圾邮件箱，并确保使用了正确的电子邮件地址。
- **证书课程作业详情**：成员询问证书是否仅限 Berkeley 学生，导师分享说证书面向任何完成其中一个课程等级的人员，并分享了 [课程网站](https://llmagents-learning.org/sp25) 和 [Google 文档](https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing) 的链接。
   - 另一位用户请求补发 **f24 LLM MOOC** 的 **Trailblazer 等级证书**，管理员迅速处理了该请求。



**提到的链接**：<a href="https://docs.google.com/document/d/1t4HS15dySldeElgDbtKmM3piInllq_auHtpO-SRScjo/edit?usp=sharing)">MOOC Curriculum</a>：MOOC 课程大纲与证书指南。感谢您参加我们的 Advanced LLM Agents MOOC！希望您喜欢目前的课程！以下是关于我们 M... 的详细描述。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 条消息): 

alexkim0889: 感谢回复 <@1335446795765022794>！是的，这很有帮助。
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1351934236369879041)** (11 messages🔥): 

> `FL Setup, Nvidia Delays` 


- **FL Setup 终于成功**：在因 IT 部门人手不足而等待了 **4 个月**之后，一位成员终于成功运行了其 FL 设置，如[此图](https://i.imgur.com/eX6y0NO.png)所示。
- **Nvidia GPU 供应仍然延迟**：成员们讨论了 **Nvidia GPU** 的供应经常延迟的问题，并以 **H200s** 为例，该型号在 **2 年前**发布，但直到 **6 个月前**才向客户开放供应。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1351972154324811908)** (4 messages): 

> `recvVector failed, sendBytes failed, DPO recipes, cudaGetDeviceCount, NumCudaDevices` 


- **`recvVector` 和 `sendBytes` 在 DPO 中引发问题**：成员报告在运行 **DPO recipes** 时遇到 `recvVector failed` 和 `sendBytes failed` 错误。
   - 原因尚不明确，可能是集群或 **torch** 的问题。
- **`cudaGetDeviceCount` 触发是否源于 CUDA 版本不匹配？**：成员在使用 **NumCudaDevices** 时也遇到了 `RuntimeError: Unexpected error from cudaGetDeviceCount()`。
   - 错误 `Error 802: system not yet initialized` 可能与使用了比预期更新的 **CUDA version** 有关，但尚未证实。
- **`nvidia-fabricmanager` 解决 CUDA 初始化问题**：解决 `cudaGetDeviceCount` 错误的方法是使用 `nvidia-fabricmanager`。
   - 确保通过 `systemctl start nvidia-fabricmanager` 启动，并通过 `nvidia-smi -q -i 0 | grep -i -A 2 Fabric` 检查状态，确认状态显示为 *"completed"*。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1351672024228954253)** (8 messages🔥): 

> `M1 Mac Training Issues, DeepSeek-R1 Home Deployment, Clang Dependency Validation, Training on CPU without Clang` 


- **M1 Mac 在训练上遇到困难**：一位成员表示其 **M1 Mac Air** 性能不足，即使是小批次训练也难以胜任，并在 **Kaggle** 和 **Hugging Face Spaces** 上遇到了 **clang** 问题。
   - 该用户正在寻求关于如何托管已训练模型推理演示的建议。
- **DeepSeek-R1 获得家用部署优化**：一位用户分享了一则关于**腾讯玄武实验室**针对 **DeepSeek-R1** 的**优化方案**的[推文](https://x.com/programerjohann/status/1901800298458575210)，该方案支持在消费级硬件上进行家用部署。
   - 该方案仅需 **4 万元**的硬件，功耗与普通台式机相当，生成速度约为 **10 汉字/秒**，相比传统 GPU 配置实现了 **97% 的成本降低**。
- **尽早验证 Clang 依赖问题**：一位贡献者询问，在没有 **clang** 的 CPU 上运行 [mnist 示例](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist.py)时出现的 `FileNotFoundError` 是否应该有更好的依赖验证或错误处理。
   - 目前的错误提示并未明确指出缺失 **clang** 依赖。
- **取得训练结果**：一位成员报告成功训练了一个模型，使用 **Adam** 达到了 **0.2** 的 loss。
   - 更多细节和代码可在该 [repo](https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1) 中查看。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/programerjohann/status/1901800298458575210?s=">来自 johann.GPT (@ProgramerJohann) 的推文</a>: 腾讯玄武实验室优化方案让DeepSeek-R1实现家用部署！只需4万元硬件，功耗噪音与普通台式机相当，每秒可生成约10个汉字。关键在于优化内存带宽、CPU配置和系统参数。比起传统GPU方案（8张H20，150万+），成本降低97%！</li><li><a href="https://x.com/programerjohann/status/1901800298458575210?s=46">来自 johann.GPT (@ProgramerJohann) 的推文</a>: 腾讯玄武实验室优化方案让DeepSeek-R1实现家用部署！只需4万元硬件，功耗噪音与普通台式机相当，每秒可生成约10个汉字。关键在于优化内存带宽、CPU配置和系统参数。比起传统GPU方案（8张H20，150万+），成本降低97%！</li><li><a href="https://github.com/kayo09/gsoc_2025/tree/main/ML4SCI/task1">kayo09/gsoc_2025 的 main 分支：gsoc_2025/ML4SCI/task1</a>: GSOC 2025! Happy Coding! ☀️。通过在 GitHub 上创建账号来为 kayo09/gsoc_2025 的开发做出贡献。</li><li><a href="https://xlab.tencent.com/cn/2025/03/16/DeepSeek-671B%E7%BA%AFCPU%E9%83%A8%E7%BD%B2%E5%AE%9E%E8%B7%B5%E7%BB%8F%E9%AA%8C%E5%88%86%E4%BA%AB(%E4%B8%80)/">DeepSeek-671B纯CPU部署经验分享(一)</a>: 私有化部署大模型能够有效保护数据隐私、便于开展大模型安全研究和知识蒸馏。目前主流部署方式包括纯 GPU、CPU&#x2F;GPU 混合以及纯 CPU 三种部署方式。本文介绍了我们针对 DeepSeek 大模型纯 CPU 本地化部署的推理探索与实践方案。我们以约 3.8 万元的整体成本，基于 llama.cpp 框架，经过硬件选型与量化精度的综合考量，实现了 q8 精度下 7.17 tokens&#...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1351795099075477504)** (1 messages): 

> `REDUCE_LOAD pattern clarification, Index Selection for reduce_input` 


- **对 REDUCE_LOAD 的索引选择感到困惑**：一位成员请求澄清 `x.src[2].src[1].src[1]` 的含义，以及选择这些索引作为 **REDUCE_LOAD 模式** 的 `reduce_input` 的原因。
   - 代码片段检查 `x.src[2].src[1].src[1]` 是否不等于 `x.src[2].src[0]`，并据此将 `x.src[2].src[1].src[1]` 或 `x.src[2].src[1].src[0]` 赋值给 `reduce_input`。
- **需要对张量索引进行澄清**：用户寻求对特定代码片段中使用的多级张量索引的更好理解，特别是链式 `.src` 属性和数值索引。
   - 具体示例 `x.src[2].src[1].src[1]` 引发了关于其在 `REDUCE_LOAD` 模式背景下用途的疑问，促使该用户请求对索引选择进行解释。


  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1351705857334841424)** (3 messages): 

> `AI21 Labs, keepitirie` 


- **AI21 Labs 保持沉默**：一位成员询问 *你们在使用什么？*
   - 另一位成员回答说 *目前看来我们不会公开分享该信息。如果情况有变，我会回来更新。对缺乏透明度表示歉意*。
- **缺乏透明度**：AI21 Labs 目前没有公开分享他们正在使用的信息。
   - 他们对缺乏透明度表示歉意，并表示如果情况发生变化将会进行更新。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1351638749364097195)** (1 messages): 

> `Welcome to New Members` 


- **社区欢迎新成员**：社区欢迎几位新成员：<@518047238275203073>、<@479810246974373917>、<@922469143503065088>、<@530930553394954250>、<@1055456621695868928>、<@1090741697610256416>、<@1350806111984422993> 和 <@347380131238510592>。
   - 鼓励所有人参与投票。
- **社区增长与参与**：该公告表明社区正在壮大，增加了几位新成员。
   - 邀请新成员通过参与社区投票进行互动，促进初步交流。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1351642765993443438)** (2 messages): 

> `dspy.ChainOfThougt, Chain of draft technique, Reduce output tokens` 


- **实现了 Chain Of Draft 的可复现性**：一位成员成功使用 `dspy.ChainOfThougt` 复现了 **Chain of Draft** 技术，并在 [一篇博文](https://pub.towardsai.net/implementing-chain-of-draft-prompt-technique-with-dspy-ca231c58114f) 中详细介绍了该过程。
- **Chain Of Draft 技术减少了输出 Token**：**Chain Of Draft** 提示词技术帮助 LLM 在不冗长的情况下扩展其回答，将输出 Token 减少了一半以上，详见 [这篇研究论文](https://arxiv.org/pdf/2502.18600v1)。



**提到的链接**：<a href="https://pub.towardsai.net/implementing-chain-of-draft-prompt-technique-with-dspy-ca231c58114f">使用 DSPy 实现 Chain Of Draft 提示词技术</a>：将你的输出 Token 减少一半以上

  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1351643141857476709)** (2 条消息): 

> `MLOps on AWS Workshop, AI4Legislation Seminar, Featureform, SVCAF's AI4Legislation competition` 


- **AWS 上的从零构建 MLOps Stack**：将于 **太平洋时间 3 月 25 日上午 8 点** 举行一场关于 **在 AWS 上从零构建 MLOps Stack** 的在线研讨会，感兴趣的人员可以通过 [此链接](https://buff.ly/IcPYNyR) 报名。
- **AI4Legislation 研讨会特邀 Legalese Decoder**：**AI4Legislation 研讨会** 将邀请 **Legalese Decoder** 创始人 (**William Tsui**) 和我们的基金会主席 (**Chunhua Liao**)，暂定于 **太平洋夏季时间 4 月 2 日下午 6:30** 举行 ([在此报名](https://forms.gle/pmbkRLVurbXcGBbAA))。
- **深入了解用于 ML 模型的 Featureform**：**Featureform** 是一个 *虚拟特征存储 (virtual feature store)*，允许数据科学家定义、管理和提供其 ML 模型的特征。
- **SVCAF 竞赛的开源 AI 驱动解决方案**：**硅谷华人协会基金会 (SVCAF)** 将于今年夏天举办一场竞赛，旨在开发 **开源 AI 驱动的解决方案**，使公民能够参与立法过程的不同环节 ([GitHub 仓库](https://github.com/svcaf/2025-AI4Legislation-Public/))。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forms.gle/pmbkRLVurbXcGBbAA">AI4Legislation 研讨会报名</a>：感谢您对 SVCAF 的 AI4Legislation 研讨会感兴趣！请在此处查看官方竞赛的 GitHub 仓库。我们也有 Discord！硅谷华人协会基金会...</li><li><a href="https://buff.ly/IcPYNyR">MLOps 工作坊：在 AWS 上从零构建 MLOps Stack</a>：欢迎参加 3 月 25 日星期二太平洋时间上午 8 点举行的 1 小时在线研讨会，深入探讨构建端到端 MLOps 平台。
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1351671641700040725)** (1 条消息): 

> `GPT4All FAQ, Default Directories` 


- **GPT4All 详细说明默认目录**：GitHub 上的 [GPT4All FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 页面描述了 **模型 (models)** 和 **设置 (settings)** 的默认目录。
   - 该 GitHub 页面还提供了关于 GPT4All 的其他信息。
- **GPT4All 模型位置**：[FAQ](https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings) 中描述了 **GPT4All 模型** 的默认位置。
   - 了解此位置有助于管理和组织模型。



**提到的链接**：<a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Questions#where-are-the-default-directories-for-models-and-settings">常见问题解答</a>：GPT4All：在任何设备上运行本地 LLM。开源且可商用。 - nomic-ai/gpt4all

  

---


---


{% else %}


> 完整的频道分类明细已针对邮件进行了截断。
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}