---
companies:
- openai
- huggingface
- thebloke
- nous-research
- mistral-ai
- langchain
- microsoft
- azure
date: '2024-01-26T22:07:42.174546Z'
description: '**OpenAI** 发布了新版 **GPT-4 Turbo**，引发了一项针对 2023 年 11 月版与 2024 年 1 月版在文本摘要能力上的自然实验对比。


  **TheBloke** Discord 社区讨论了 **OpenHermes-2.5-Mistral-7B-4.0bpw** 与 **exllamav2**
  的模型加载故障排除、**RHEL**（红帽企业 Linux）在机器学习中的应用辩论、用于分析 GPT 缺陷的数据集生成，以及在游戏主机上运行 **Llama**
  和 **Mistral** 等大语言模型。此外，还提到了 **LangChain** 在 **Llama2** 微调方面面临的挑战。


  **OpenAI** Discord 频道重点关注了 **GPT-4** 的速度波动、API 与网页端的性能差异、针对 **GPT-3.5** 和 **GPT-4
  Turbo** 的提示词工程，以及 **DALL-E** 在图像文本中出现的拼写错误。讨论内容还包括 *semantic-text-splitter* 等 NLP
  工具，以及在 **Azure** 上协作使用 **GPT-4 Vision** 的相关疑虑。


  **Nous Research AI** Discord 频道则聚焦于扩展上下文窗口，涉及 **Mistral instruct v0.2**、**MistralLite**
  以及实现 16,384 token 上下文的 **LLaMA-2-7B-Chat**，并探讨了如 **SelfExtend** 等无需微调即可扩展上下文的替代方案。最后，AI
  技术的社会影响也受到了审视。'
id: 5daa80d8-1d76-461d-9b2e-facedc00251c
models:
- gpt-4-turbo
- gpt-4
- gpt-3.5
- openhermes-2.5-mistral-7b-4.0bpw
- exllamav2
- llama-2-7b-chat
- mistral-instruct-v0.2
- mistrallite
- llama2
original_slug: ainews-gpt4turbo-ab-test-gpt-4-1106-preview
people: []
title: GPT4Turbo A/B 测试：gpt-4-1106-preview
topics:
- model-loading
- rhel
- dataset-generation
- llm-on-consoles
- fine-tuning
- speed-optimization
- api-performance
- prompt-engineering
- token-limits
- memory-constraints
- text-generation
- nlp-tools
- context-window-extension
- sliding-windows
- rope-theta
- non-finetuning-context-extension
- societal-impact
---

<!-- buttondown-editor-mode: plaintext -->> 2024年1月25日的 AI Discord 动态。我们为您检查了 **20** 个服务器，**297** 个频道和 **5898** 条消息。预计节省阅读时间（按 200wpm 计算）：**557 分钟**。

OpenAI 昨天发布了新的 [GPT4 Turbo 版本](https://openai.com/blog/new-embedding-models-and-api-updates)（[我们的笔记见此](https://twitter.com/swyx/status/1750620187114787243)）。我们正借此机会进行一次摘要生成的自然实验。本版本是由 2023年11月（Dev Day）的“旧版” GPT4T 生成的，请关注下一封包含 2024年1月25日版本的邮件，以进行对比和评论。


--

**目录**

[TOC] 


# 第一部分：Discord 高层摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **模型加载错误排查**：用户 `@sco7116` 在尝试从 Huggingface 加载 *OpenHermes-2.5-Mistral-7B-4.0bpw* 时遇到错误，`@itsme9316` 建议确保使用 *exllamav2* 以保证正确的版本兼容性。错误消息为 "Could not locate pytorch_model-00001-of-00002.bin"。

- **ML 背景下的 RHEL 辩论**：关于在机器学习环境中使用 Red Hat Enterprise Linux (RHEL) 的优缺点进行了活跃讨论。用户 `@.dontriskit` 分享了对基础设施偏好的看法以及在使用 RHEL 时遇到的挑战。

- **强调为理解 GPT 缺陷而生成数据集**：`@kaltcit` 提议生成一个旨在识别 GPT 常见缺点（如循环和话题漂移）特征的数据集，认为这对于系统地理解和解决这些问题至关重要。

- **在游戏机上运行 LLM**：该服务器对 Llama 和 Mistral 等大语言模型在 Nintendo Switch 等非常规硬件上运行的演示非常感兴趣，展示了这些模型在各种平台上的新颖性和潜在的广泛适用性。

- **LangChain 微调困境**：新用户 `@nandavikas` 正在寻求使用 LangChain 对 Llama2 模型进行微调以从 PDF 中提取信息的帮助，此前他已使用 PyTorch 完成了该任务，但目前缺乏来自 LangChain 文档的相关指导。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **GPT-4 对速度的需求**：用户报告 **GPT-4** 存在速度不一致的问题，与网页端相比，API 速度较慢，尤其是在**高峰时段**。对于企业级用户，预留容量被提及为一种潜在解决方案，但在 API 负载较高时问题依然存在。

- **Api-discussions 与 Prompt-engineering 频道交叉讨论**：使用 **GPT-3.5** 进行 *prompt engineering* 的挑战涉及对大文本进行分块以进行语法纠正，突显了 Token 限制和内存约束。建议使用 OpenAPI、Python 脚本和 *Custom Actions*，并推荐 **GPT-4 Turbo** 作为处理大型文档的更优替代方案。

- **DALL-E 的拼写错误困扰**：用户注意到 **DALL-E** 在生成图像内的文本时往往会出现拼写错误，社区讨论建议通过缩短文本输入来缓解该问题。一个 [社区链接](https://community.openai.com/t/does-anyone-experience-issues-with-dall-e3-generating-typos-in-text-within-images/472966) 提供了关于这一持续问题的更多见解。

- **被吹捧为文本转换器的 NLP 工具**：由于 GPT 模型在处理大型文档时表现吃力，像 `@doublefelix` 这样的用户开始寻求外部 NLP 工具，例如 **PyPI** 上的 *semantic-text-splitter*，作为潜在的补救措施。对话强调了在 API 调用中保持历史上下文的重要性，以及利用 **ChatGPT Plus** 作为高性价比解决方案的可能性。

- **协作与合规的概念化**：关于在 **Azure** 上部署 **GPT-4 Vision** 以及团队账户协作的咨询遇到了与异常活动相关的账户问题担忧，这可能是由于使用 VPN/代理或触发了敏感 Prompt 导致的。此外还提到了保存/更新 GPT Bot 的问题，特别是与模仿在世艺术家的政策相关。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **重点关注扩展上下文能力**：目前正在讨论扩展上下文能力的最佳方法，微调（finetuning）被认为是一个可行的解决方案。**Mistral instruct v0.2** 禁用了滑动窗口（sliding windows）以缩放 `rope_theta`，而 **MistralLite** 和主分支模型的配置在上下文窗口管理上正趋于一致。**LLaMA-2-7B-Chat** 表现惊人，仅通过极少的样本和训练步骤就将其上下文窗口扩展到了 16,384，而 **SelfExtend** 为上下文扩展提供了一种无需微调的替代方案。

- **AI 的社会影响与技术谜题**：技术在社会极化中的作用正受到关注，观察到的 Twitter 活跃度波动引发了关于 AI 增速放缓的理论。分享了一个使用共享 notebook 将 LLM 量化为 **GGUF** 格式的资源，助力模型转换过程。

- **寻求 Everyone Coder 和 Hermes Mixtral 基准测试**：已请求针对量化版 **Everyone Coder 33B**（使用 **GGUF** 格式）进行 *human eval* 基准测试，该模型可通过 [Hugging Face 上的 TheBloke](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF) 获取，并由 [Massed Compute](https://massedcompute.com/) 提供支持。人们也有兴趣看到 Hermes 和 Mistral 结合模型（幽默地称为 **Hermes Mixtral**）的基准测试。

- **OpenAI 嵌入模型发布与合成数据创新**：OpenAI 发布了新一代嵌入模型（Embedding Models），在数据隐私和成本降低方面表现显著，详见其[发布公告](https://openai.com/blog/new-embedding-models-and-api-updates)。此外，用于生成高质量内容接地数据（content-grounded data）的 **Genie** 方法预示着长文本问答（Long-Form Question-Answering）和摘要生成方面的潜在进步，如 [arXiv 论文](https://arxiv.org/abs/2401.14367)所述。

- **利用 GPU 加速增强 AI 运营**：机器学习计算正利用 WebGL 进行 GPU 加速，讨论涉及实时词嵌入（word embedding）调整系统。模型训练需要高端主板以支持多 GPU 配置，**Mixtral instruct** 的表现优于多个微调模型，新型 GPU 增强模型的原型也取得了进展。

- **LLM 在代码和网络安全领域备受瞩目**：对于微调 **CodeLlama 34B**，4xA6000 GPU 可能仅够运行 *qlora*，而无法进行全量微调。**T5** 在微调时面临稳定性挑战，而 **WhiteRabbitNeo-13B** 和 **WhiteRabbitNeo-33B-v1** 等 LLM 被推荐用于攻势网络任务。 [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) 是评估 AI 代码生成能力的资源，此外还有关于使用 **Llama Factory** 进行微调时超参数要点的对话。

- **Project Obsidian 脚本与重构**：正在为 **3b Obsidian** 模型寻求用于远程执行的 Python 脚本，并且正在努力重构代码以兼容最新的 **llava repo**。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **挪威语微调中 Base 模型优于 Instruct 模型**：根据 `@henriklied` 在挪威语数据集上的经验，`@le_mess` 建议在训练数据稀缺的语言中进行微调时，应使用 **Base 模型** 而非 Instruct 变体。

- **警惕 Epoch 数量以避免过拟合**：为了防止微调过程中的过拟合，`@le_mess` 建议在 **4 个 Epoch** 时停止，特别是考虑到 `@henriklied` 观察到 10 个 Epoch 后评估损失（eval loss）保持不变。

- **训练中块字符的潜力**：`@suikamelon` 和 `@c.gato` 讨论了在仅有少量数据集的情况下，使用 `▀` 和 `▄` 等独特块字符（block characters）在 Mistral 等模型训练中相对于 ChatML 标签的效果和分词（tokenization）表现。

- **量化与配置优化器讨论**：在量化讨论中，`@dangfutures` 更倾向于 AWQ，而 `@bjoernp` 和 `@noobmaster29` 讨论了使用配置优化器（config optimizer）而非默认 deepspeed 优化器的重要性，参考了 [deepspeed config PR #1208](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208)。

- **资助庆祝与数据集介绍**：`@gahdnah` 在观察到**活跃开发**后撤回了之前的担忧，并对获得的资助表示祝贺。同时，`@dangfutures` 重点介绍了 Hugging Face 上一个基于 Mistral 改进的 Snorkel 模型新数据集。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **LM Studio 在教学环境中的应用**：`@josemanu72` 正在寻找将 LM Studio 作为服务器运行以便从学生桌面连接的方法，`@fabguy` 建议了一个包含 [前端 UI](https://github.com/JShollaj/Awesome-LLM-Web-UI) 的解决方案。
  
- **硬件难题与帮助**：关于利用 GPU 的讨论非常广泛，从 `@xmiruso` 通过重新加载模型解决 GPU 利用率问题，到 `@gitanos` 寻求 GPU 性价比建议（建议选择二手 **Nvidia 3090** 而非 **4060ti 16GB RAM**）。同时，`@mudf00t` 反馈在 Nobara 系统上使用 **RTX 3090** 时遇到显存（VRAM）检测问题，目前尚无即时解决方案。

- **开源模型思辨**：`@vbwyrde` 讨论了 **"Intern" (InternLM)** 的发布及其据称的 200k 上下文窗口（context window）和函数调用（function calling）能力。此外，还有关于 Meta 发布 **Llama2** 等开源模型以及使用 **Solar-10b 进行函数调用**的策略讨论。

- **故障排除与技巧**：多位用户报告了问题，包括 **切换 MoE 模型时的 Bug**，以及最近 LM Studio 更新导致模型加载失败的错误，例如 `@pdg` 需要降级到 **0.2.10 版本**来解决该问题。`@sunglasses.emoji` 报告了 Autogen 中 **空字符串的断开链接**问题，并分享了提升模型性能的建议方案。

- **软件故障**：对话强调了开源模型框架的挣扎，以及模型表现出的古怪行为，例如 **Mistral** 会幻觉出不存在的目录。还有一个有趣的发现：**OpenAI 的模型**无法回想起当前的 API，这影响了训练期间的上下文。



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

**从 GPU 租赁到 Mistral 集成**：讨论了包括 **runpod、vast 和 lambda** 在内的 GPU 租赁选项，还提到 **Kaggle** 提供每周高达 30 小时的免费访问额度。分享了 **Mistral 7B** 的使用案例和集成挑战，寻求有效实现的见解，并引用了 **[Hugging Face 的 Mistral 7B 模型](https://huggingface.co/intfloat/e5-mistral-7b-instruct)**。

**模型微调中的内存问题**：围绕 **Mixtral** 推理时巨大的内存需求展开了讨论，强调在四个 T4 GPU 上需要 26GB 显存，实际使用量可能高于预期。效率辩论对比了用于量化（quantization）的 **exllamav2** 和 **bnb 4 bit**，并推荐通过 [exllamav2 GitHub](https://github.com/turboderp/exllamav2) 高效运行 LLM。

**超越传统指标评估 LLM**：强调了 **BLEU 和 ROUGE** 指标对 LLM 的局限性，建议使用 **Elo 排名** ([arena.lmsys.org](https://arena.lmsys.org/)) 以及 **MMLU** 和 **Alpaca eval** 等基准测试（benchmarks）来更好地衡量性能。提到了归一化 Alpaca eval 市场版本的引入，但未提供更多细节。

**创意展示与随机 RAG 技巧**：展示了一个名为 *SoContextual.com* 的工具，它集成了 AI 进行浏览器搜索（包括 DOM 引用），支持 **MistralAI**，该工具曾在 [Hacker News](https://news.ycombinator.com/item?id=39128480) 上亮相。同时，简要涉及了 RAG 应用的 **提示词优化（prompt optimization）**，推荐使用 **DSPy** 并分享了一份 [提示词指南](https://www.promptingguide.ai/models/mistral-7b)。

**平台困惑与 API 异常**：报告了一个账单页面 Bug，导致月度限额重置为 €150；同时讨论了关于 'max_tokens' 参数和提前停止（early stopping）问题的 **API Bug**，包括一个已发布的 [GitHub issue](https://github.com/mistralai/mistral-src/issues/122)。托管查询确认 Mistral 的 API 位于瑞典的 Azure 上。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **寻求 SAM 模型微调代码**：`@the_alt_man` 正在寻找用于微调 **Meta 的 SAM 模型**的代码库，结果发现原始发布版本中并未包含该代码。讨论中提到了 **`AutoGluon`** 这一与 **Lightning** 集成的工具，但也指出了其在 GPU 使用方面的局限性。

- **联邦学习（Federated Learning）探索**：成员们共同讨论了联邦学习的实用性，特别是没有 **infiniband** 的多节点训练以及模型合并的具体细节。引用了 DiLoCo 的研究，该研究可在 [arXiv](https://arxiv.org/abs/2311.08105) 上找到。

- **深入探讨 LLMs 的代理微调（Proxy Tuning）**：`@digthatdata` 介绍了一种通过代理微调 **Large Language Models (LLMs)** 的方法，这可能会简化微调过程。这种替代方法的详细信息可以在 [arXiv 上的论文](https://arxiv.org/abs/2401.08565)中找到。

- **GPT-NeoX QLoRA 微调难题**：`@kenakafrosty` 寻求使用 **QLoRA** 微调 **GPT-NeoX 20B** 的帮助，面临 Loss 不下降的问题。会议澄清了 NeoX 目前不支持 QLoRA，并建议去 GitHub 寻找 `trl`、`transformers` 和 `peft` 的解决方案。

- **GPT-NeoX 开发中的测试困扰与协作**：**Python、PyTorch 和 CUDA** 的更新导致运行测试时出现问题，引发了关于项目功能测试套件必要性的讨论。目前正在积极修复测试流程、追踪 fork 版本的测试失败，并为项目协作人员提供算力访问权限，例如 [此 GitHub issue](https://github.com/EleutherAI/gpt-neox/issues/1132) 所示。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **数据集消失困扰开发者**：针对 **laion2b-en aesthetics scores** 的 **LAION 数据集**目前无法访问，因为数据集作者要求暂时关闭。建议工程师关注官方公告以获取数据集访问的更新。

- **语音聊天界面即将推出**：演示了一个集成 Whisper 和 WhisperSpeech 与 LLM 的新**语音聊天界面**，承诺降低延迟并使对话更自然。目前正在寻求合作者以进一步改进系统；详情可见 [Hacker News 公告](https://news.ycombinator.com/item?id=39130140)。

- **图像说明（Image Captioning）至关重要**：使用 AI 进行**图像说明**的方法引发了关于清晰 Prompt 重要性的讨论，以减少幻觉（hallucinations），重点是仅对可见内容进行准确描述。

- **竞赛征集创意编程者**：AI Tech Sprint 邀请开发者参与一个专注于临床就诊记录的项目，并有机会赢取奖金。感兴趣的各方应[登记意向](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)。

- **AI 的巨额开支报告**：讨论了训练 **Stable Diffusion** 等 AI 模型所涉及的高昂成本，同时也承认随着时间的推移，技术进步将带来预期的成本降低。

- **Google 错失了 SaaS 班车？**：Google 对广告收入的过度依赖受到了批评，有人建议关注 **SaaS 模式**（如 OpenAI 的模式）可能是另一条路径，可能会产生更深远的财务影响。

- **字节级 Transformer（Byte-Level Transformers）指日可待**：人们对字节级 Transformer 的兴趣日益增加，预计很快会有重大进展，最近相关的 [arXiv 研究](https://arxiv.org/pdf/2401.13660.pdf)证明了这一点。

- **修复与识别的重新构想**：通过 [text-to-image diffusion 技术](https://github.com/YangLing0818/RPG-DiffusionMaster)和 [ID 保持生成系统](https://github.com/InstantID/InstantID)，突显了 **text-to-image diffusion** 和**身份保持（identity preservation）**方面的技术进步，为 AI 生成图像带来了新功能。

- **凭借 SUPIR 登顶**：一篇关于 **SUPIR**（一种利用生成先验和模型缩放的图像修复方法）的论文因其创新方法以及在 Hacker News 热门论文中的提及而受到关注，详见 [arXiv 提交内容](https://arxiv.org/abs/2401.13627)。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Perplexity Pro 功能揭秘**：爱好者们讨论了 [Perplexity Pro 的功能](https://blog.perplexity.ai/faq/what-is-perplexity-pro)，例如无限量的 Copilot 查询以及对 GPT-4 和 Claude 2.1 等 AI 模型的访问权限，揭示了其超越标准版的增强能力。

- **隐私政策备受关注**：对 **Perplexity 数据保留政策** 的担忧促使官方澄清：已删除的线程将在 30 天后清除；然而，隐私政策中关于账户和搜索数据的模糊性引发了用户要求更清晰沟通的呼声，以确保用户放心。

- **API 查询与计费不满**：技术讨论揭示了 Perplexity AI 网站与 API 之间的差异，后者生成的代码质量较低；包括 `@aiagileguy` 在内的用户面临计费问题（如重复扣费），且未能得到快速解决。

- **Perplexity 在教程与教育中的应用**：用户分享了 Perplexity AI 的成功案例和实际用途，例如平滑地完成从 Excel 到 Smartsheet 的过渡，以及在教育场景中辅助解释复杂的天文学概念。
  
- **投票支持 Perplexity 而非巨头**：一段名为“[我使用 Perplexity 的频率超过了 Google 和 ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)”的 YouTube 视频将 Perplexity AI 描述为在内容创作等任务中优于 Google 和 ChatGPT 等主流选择。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **HuggingFace 推出社交帖子探索者**：`@lunarflu` 邀请 **HuggingFace 社区** 加入 [“Posts”功能的早期访问](https://huggingface.co/social-post-explorers)，该功能旨在为 AI 和 ML 讨论提供一个专注的空间，远离 Twitter 或 LinkedIn 等平台的噪音。

- **预训练困境与高性价比策略**：像 [Llama-2-7B 模型](https://huggingface.co/meta-llama/Llama-2-7b) 这样耗费 GPU 资源的模型预训练，让社区开始考虑资源消耗较少的替代方案，如微调或使用 [LoRA/QLoRA 适配器](https://huggingface.co/docs/peft/conceptual_guides/lora)。

- **迫切需要数据集评估标准**：`@rosebei3ngan3g` 强调了缺乏评估大语言模型（LLM）数据集的框架，这与众多的模型评估框架形成了鲜明对比。

- **数据集分析与 Demo 展示的深刻创新**：一个关于 [数据集卡片分析的 GitHub 项目](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis) 和一个多语言文本转语音的 HuggingFace Demo [WhisperSpeech](https://huggingface.co/spaces/Tonic/whisperspeech)，展示了 HuggingFace 生态系统中动态的工作范围。

- **对革命性模型和指标的认可**：Google 的 **Lumiere** 模型结合了 Space-Time UNET 以实现流畅的视频生成能力，在社区中备受关注；同时，大家对 `gradio 4.16` 的新版本也表现出浓厚兴趣，该版本包含对 **Polars Dataframe** 的支持和新的 Gallery 组件，详见 [更新日志](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md)。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **LlamaIndex 举办 LLM 网络研讨会引人关注**：`@jerryjliu0` 宣布了一场关于 **LLMCompiler** 的**网络研讨会**，重点讨论了 **Agent 中的并行函数调用**，并提供了 **LlamaPack** 和 **Notebook** 等资源。相关论文可以在[这里](https://arxiv.org/pdf/2312.04511.pdf)找到，研讨会详情请见[这里](https://lu.ma/lf9iroox)。

- **一系列创新成果揭晓**：`@seldo` 发布的 **Slack Bot 教程**指导如何将组织学习整合到 Bot 中；**Zilliz Cloud Pipeline** 最新集成了 LlamaIndex，详情见[客座博客文章](https://t.co/luDjSgiokt)。LlamaIndex 0.9.38 版本现已支持 **OpenAI 最新的 embedding 模型**，更多细节见其[发布说明](https://t.co/kyIoTUaeuD)；同时 TypeScript 用户迎来了支持相同功能的 **LlamaIndex.TS** 0.1.0 版本。

- **#general 频道关于检索和自定义的讨论升温**：LlamaIndex 目前缺少针对 **TextGenerationInference** 的原生 **LLM** 支持，但与 Langchain 的兼容。此外，还讨论了**复杂的检索场景**以及 OpenAI 更新的 **embedding 模型**的整合。针对如何在上下文不足时提取答案的问题，分享了一个**修改默认 Prompt** 的链接：[用法模式](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts)。

- **Zep 的多功能聊天特性受到关注**：**Zep** 记忆对话和执行实体提取的能力引发了兴趣，`@yoursaviorjesus` 分享了 [Zep 文档](https://docs.getzep.com/)。在澄清 LlamaIndex 的功能时，`@cheesyfishes` 将其描述为类似于 Amazon Kendra，但可以适配**任何 vector store 或语言模型**。

- **知识图谱创新分享**：`@chiajy` 通过一个**哈利·波特书籍演示**展示了一个具有**递归检索和多跳推理**功能的自学习知识图谱。欲深入了解此项工作，请参阅 [Harry Potter and the Self-Learning Knowledge Graph](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **LLM Paper Club 坚持不录音政策**：Latent Space 的 LLM Paper Club 会议**将不进行录音**，以鼓励开放分享，且不提供回放选项。
- **Morpheus-1 让梦境与 AI 相遇**：一条推文发布了关于 **Morpheus-1** 的消息，这是一款旨在诱导清醒梦的多模态生成式超声 Transformer，计划于 2024 年春季发布 Beta 版。其新颖的方法引发了广泛关注。
- **GPT-4 Turbo 和新 Embedding 模型发布**：OpenAI 发布了更新的 **GPT-4 Turbo** 模型和新的 embedding 模型。相关详细说明和公告已分享，强调了改进点以及对 AI 应用的潜在影响。
- **Martian 的 LLM 基准测试上线**：Martian 在 **[Martian's LLM Leaderboard](https://leaderboard.withmartian.com/)** 推出了 Model Router，用于评估不同的 LLM 推理产品，并提供开源文档和工具支持。
- **LLM Paper Club 扩展至亚洲**：LLM Paper Club 已扩展到亚洲，提供对 "Attention Is All You Need" 等开创性论文的讨论。该俱乐部正在征集未来论文的建议和反馈，以优化 **beta** 体验。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **Mixtral 与模型合并（Merging Models）深度解析**：*mergekit* 的作者在 [GitHub comment](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289) 中提供了见解，影响了 **DiscoResearch Mixtral 训练**方法。重点在于 Mixture of Experts (MoE) 训练中 *auxiliary loss* 的正确应用。

- **重新思考数据过滤与模型训练方法**：一篇新论文挑战了预训练数据质量过滤的有效性，指出数据选择应与模型在目标任务上的表现保持一致，详见[此处](https://arxiv.org/abs/2401.12926)。讨论围绕采用 Direct Preference Optimization (DPO) 和 Key Term Optimization (KTO) 等新训练方法展开，关于使用 DPO Trainer 的见解详见 [Hugging Face 的 TRL 文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)。

- **Embedding 开发与使用的进展**：德国 Jina embeddings 模型即将发布，有望增强排名应用。OpenAI 的新 Embedding 模型具有改进的多语言能力，标志着一次飞跃，详见[此处](https://openai.com/blog/new-embedding-models-and-api-updates)。

- **翻译与语言模型微调成果**：**DiscoLM German 7B v1** 已成功使用自定义数据集进行微调，旨在将中古高地德语翻译为现代德语。该微调过程正热切期待基于 **Mixtral-Instruct** 的版本。

- **Embedding 技术即将迎来的效率提升**：即将推出的 Embedding 模型在 MIRACL benchmark 上将超越 OpenAI，在仅 256 维度下即可实现 **12 倍的向量数据库成本节省**，正如 `@Nils_Reimers` 在[这条推文](https://x.com/nils_reimers/status/1750631888094380268?s=46&t=-TRJUfVdW8KeDqen1HJU1Q)中所预告的。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **OpenAI 发布 Embedding 创新**：[OpenAI 的最新公告](https://openai.com/blog/new-embedding-models-and-api-updates)详细介绍了 **GPT-4 Turbo** 和新 **Embedding 模型**的发布、更好的 API 控制工具，以及即将下调的 **GPT-3.5 Turbo** 价格。工程师们将**缩短的 Embedding（shortened embeddings）**视为效率的一次飞跃，并期待将其集成到系统中。
  
- **更新后的 OpenAI API 简化开发者流程**：OpenAI 致力于增强 **API 体验**，包括新的审核模型和旨在完善开发者监督的 **API 使用管理**工具。开发者现在可以参考[更新后的文档指南](https://platform.openai.com/docs/guides/embeddings/)来了解最新的模型和功能。
  
- **便捷性胜过开源**：关于 OpenAI 与开源模型之间的辩论正在展开，像 `@res6969` 这样的专业人士指出使用 OpenAI 实现功能的效率极高，而其他人则主张开源替代方案的可定制性。
  
- **便利性可能优于定制化**：尽管有开源模型可用于个人微调，但像 `@potrock` 这样的成员强调了 OpenAI Embedding 模型提供的简单、开箱即用的便利性。
  
- **在成本效益之间取得平衡**：经济方面的讨论转向了 OpenAI 新的大型 Embedding 模型的**成本效益**，正如 `@shacrw` 和 `@michelcarroll` 所讨论的，在这些更新之后，需要在**存储节省**和 **API 成本**之间进行权衡。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

- **欢迎加入 LangChain 宇宙**：`@quarknova` 目前在 INRIA 实习，展示了在项目中使用 LangChain 的兴趣，并促使其考虑 GitHub 版本与其商业版本的优势对比。

- **定制化 AI 人格现已实现**：`@jstansbe` 探讨了创建如 "Elon Musk AI" 等定制化 AI 实体的可能性，`@ksolo__` 贡献了一个资源，介绍了 finetuning 的概念并分享了一个 [深度学习课程链接](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)。

- **LangChain 助力聊天机器人创建获赞**：`@johnnucleus` 赞扬了 LangChain 社区在协助其使用 LangChain 和 Streamlit 快速开发集成网络搜索功能的聊天机器人方面所提供的有效帮助。

- **LLM 转型为数据合成器**：讨论涉及使用 LLM 生成合成数据以供传统 ML 模型使用，特别提到了利用 LLM 进行 RAG 生成，从而根据上下文和 schema 创建 SQL 查询。

- **在 LangChain 中操作 PARQUET**：`@benjaminbascary` 和 `@johnny2x2` 交流了在 LangChain 中处理 PARQUET 文件的见解，并提供了通过 `pandas` 和 `DataFrameLoader` 功能实现的示例代码。

- **深入探索 LangServe 的能力**：`@veryboldbagel` 分享了使用 LangServe 和 LCEL 创建自定义工具和 Agent 的示例与资源，强调了使用 [LangGraph](https://python.langchain.com/docs/langgraph#agentexecutor) 构建具有更强表达能力的 Agent 的实用性。

- **流式响应缺失的神秘案件**：`@hiranga.g` 在实验 LangServe 的 [agent_with_history](https://github.com/langchain-ai/langserve/blob/main/examples/agent_with_history/server.py) 时遇到了流式响应挑战，指出了在通过 LangServe 使用 `chain.streamLog()` 集成 Agent 时可能存在的 bug。

- **SQL Chain 与大型数据库的博弈**：`@johnny2x2` 讲述了 SQL Chain 在处理大型数据库时的困难，并发现通过在数据库中构建具有描述性名称的 **精选视图 (curated views)** 可以显著提升性能。

- **通过优化改进 SQL 查询管理**：`@johnny2x2` 描述了从使用本地 AI 转向依赖 OpenAI 进行 SQL 查询处理的转变，以在维护数据隐私的同时，在 LangChain 内部实现更高效的查询过程。

- **通过链式工作流提升任务处理能力**：`@johnny2x2` 介绍了一种新方法，描述了在其 **任务处理链 (task processing chain)** 中将每个 SQL 查询作为独立工具使用的转变，从而显著改进了工作流管理。

请注意，任何对用户名的直接引用均已包含在内，因为根据提供的信息，这些引用被认为具有上下文相关性。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **LLM 库更新预告**：@SimonW 宣布了 LLM 项目中 **openai 库** 即将进行的更新，并在 [GitHub 评论](https://github.com/simonw/llm/issues/325#issuecomment-1911533536) 中为测试人员提供了详细信息。

- **LLM 迈向 0.13 版本**：LLM 发布版本中即将到来的 **0.13 里程碑 (Milestone)** 旨在增强命令行对 LLM 的访问能力，相关信息已记录在 [GitHub 里程碑页面](https://github.com/simonw/llm/milestone/8) 上。

- **征集开发者解决 Readline Bug**：正在公开征集关于 LLM 中 readline 问题的协助，该问题会导致方向键显示 ANSI 编码而非移动光标，详见此 [GitHub issue](https://github.com/simonw/llm/issues/376)。

# 第 2 部分：详细的分频道摘要和链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199991901110145084) (1212 条消息🔥🔥🔥): 

- **尝试 Hermes 2.5**：用户 `@sco7116` 寻求帮助测试来自 Huggingface 的名为 *OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2* 的模型，但遇到了错误 "Could not locate pytorch_model-00001-of-00002.bin"。`@itsme9316` 建议他们没有使用 *exllamav2* 加载，应该切换到正确的版本。
- **在 ML 中使用 RHEL 的优缺点**：`@kquant` 报告成功运行了 Linux，并转向使用 Ubuntu 进行合成数据集生成。讨论包含了关于在机器学习中使用 Red Hat Enterprise Linux 的各种观点，`@.dontriskit` 分享了关于首选基础设施以及在 ML 开发环境中使用 RHEL 面临的挑战的见解。
- **分享 DPO 脚本和技术**：`@kaltcit` 提供了关于他们对模型进行数据集剪枝优化（DPO）方法的信息，重点是生成一个捕捉常见 GPT 失败模式（如循环和话题漂移）的数据集。他们认为这样的数据集可能是 GPT 缺陷最全面的集合。
- **对 LLM 预测的着迷**：用户讨论了各种针对 Llama 和 ChatGPT 等模型的微调（finetunes）和合并（merges）。`@itsme9316` 发现一个基于 2000 万 token Discord 消息的 7B 微调模型在质量上超过了几个更大的合并模型，甚至表示他们可能会尝试 5 亿 token 的微调。
- **在 Nintendo Switch 上运行 AI**：用户分享了在令人惊讶的硬件上运行 LLM 的视频和评论。`@kquant` 对在 Nintendo Switch 上运行 Llama 和 Mistral 等模型的可能性表示惊讶，而 `@kalomaze` 则通过展示 LLM 在非常规平台上运行的媒体内容为这一话题做出了贡献。

**提到的链接**：

- [DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/abs/2401.14196)：大语言模型的快速发展彻底改变了软件开发中的代码智能。然而，闭源模型的主导地位限制了广泛的研究和开发...
- [God helmet - Wikipedia](https://en.wikipedia.org/wiki/God_helmet)：未找到描述
- [Marvelous Dc Gotham GIF - Marvelous Dc Gotham Gotham Tv - Discover &amp; Share GIFs](https://tenor. Silview/marvelous-dc-gotham-gotham-tv-hugo-strange-dr-strange-gif-17601265)：点击查看 GIF
- [Tails - Home](https://tails.net/)：未找到描述
- [LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2)：未找到描述
- [no title found](https://neurosity.co/)：未找到描述
- [import sysimport osfrom tqdm import tqdmsys.path.append(os.path.dirname(os - Pastebin.com](https://pastebin.com/wgD8Q5Qs)：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
- [How to Install NVIDIA Drivers on Rocky Linux 9 or 8 - LinuxCapable](https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-rocky-linux/)：学习如何使用命令行终端和 NVIDIA CUDA REPO 安装最新版本的 NVIDIA 驱动程序。
- [How I Won Singapore’s GPT-4 Prompt Engineering Competition](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41)：深入探讨我学到的利用大语言模型（LLM）力量的策略。
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind#screenshots)：一个多模态、由函数调用驱动的 LLM WebUI。
- [GitHub - facebookresearch/audio2photoreal: Code and dataset for photorealistic Codec Avatars driven from audio](https://github.com/facebookresearch/audio2photoreal)：由音频驱动的逼真 Codec Avatars 的代码和数据集。
- [GitHub - facebookresearch/Qinco: Residual Quantization with Implicit Neural Codebooks](https://github.com/facebookresearch/Qinco?tab=readme-ov-file)：带有隐式神经码本的残差量化。
- [Stanford Hypnosis Integrated with Functional Connectivity-targeted Transcranial Stimulation (SHIFT): a preregistered randomized controlled trial - Nature Mental Health](https://www.nature.com/articles/s44220-023-00184-z)：研究人员展示了一项双盲随机对照试验的结果，该试验使用经颅磁刺激对左背外侧前额叶皮层进行个性化刺激，以增加...

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1200016786880471090) (74 条消息🔥🔥): 

- **ExLLamaV2 加载器支持问题**：`@ks_c` 最初质疑 oobabooga 中的 exllamav2 加载器是否支持 `min_p`（尽管它不是 hf loader），但随后确认该功能已合并到 exllama 中。
- **CPU 模式困惑已消除**：在 `@keyboardking` 询问 **exl2** 在纯 CPU 模式下与 **gguf** 相比的实用性后，`@neriss` 告知 `@keyboardking` **exl2** 无法在 CPU 上运行。
- **模型配置比较**：`@dreamgen` 询问了不同模型之间 `rope_theta` 和 `sliding_window` 配置的差异，并分享了 [bagel](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json)、[Mistral instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json) 和 [dolphin](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json) 配置文件链接。`@jondurbin` 回复解释了这些配置是对基础模型的继承，并提到未来可能发生变化。
- **角色扮演模型讨论**：`@shadowplague` 寻求擅长生成*无礼、辱骂、种族歧视和脏话场景*内容的角色扮演模型推荐。`@c.gato` 和 `@kalomaze` 指出可以通过 Prompt 引导现有模型生成此类内容，并建议使用 **Kunoichi DPO v2** 或 **Fett-uccine** 进行角色扮演。
- **用于 RP 的 7B 参数模型**：在讨论最佳的 7B 参数角色扮演模型时，成员们提供了各种建议，如 **HamSter-0.2**、**Kunoichi DPO v2** 和 **Fett-uccine**，同时还讨论了根据 VRAM 容量选择 6-bit 或 8-bit 量化的问题。

**提到的链接**：

- [Epiculous/Fett-uccine-7B-GGUF at main](https://huggingface.co/Epiculous/Fett-uccine-7B-GGUF/tree/main)：未找到描述
- [Release Quadratic Sampling Test Build (koboldcpp) · kalomaze/koboldcpp](https://github.com/kalomaze/koboldcpp/releases/tag/quad-sampling-v1)：上一个想法（Smooth Sampling）的替代方案，采用了不同的缩放机制。其背后的想法是尽可能简化采样并移除尽可能多的额外变量...
- [config.json · mistralai/Mistral-7B-v0.1 at main](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)：未找到描述
- [config.json · jondurbin/bagel-dpo-7b-v0.1 at main](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json)：未找到描述
- [config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json)：未找到描述
- [Kronk Its All Coming Together GIF - Kronk Its All Coming Together - Discover &amp; Share GIFs](https://tenor.com/view/kronk-its-all-coming-together-gif-15058130)：点击查看 GIF
- [config.json · cognitivecomputations/dolphin-2.6-mistral-7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json)：未找到描述

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1200022344538800189) (20 条消息🔥): 

- **聊天机器人学习名称和风格**：`@lordofthegoons` 正寻求训练一个具有一致对话风格并能记住自己名字的聊天机器人，类似于 Samantha 模型。`@amogus2432` 建议使用 10 到 100 个示例进行风格迁移（style transfer），但 `@dirtytigerx` 建议使用更多，并提到 Samantha 模型使用了约 6,000 次多轮对话。
- **寻求金融顾问聊天机器人**：`@VR` 正在寻求关于创建一个金融投资顾问聊天机器人的建议，该机器人需能在 24GB GPU 上运行，并利用 RAG 获取最新的股票价格信息、趋势和专家分析。他们正在考虑是使用 Prompt Tuning 还是对金融文档进行微调（Fine-tuning）。
- **构建独特的聊天机器人人格**：`@lordofthegoons` 旨在通过制作自定义数据集来创建一个具有特定人格（persona）的聊天机器人。他们注意到使用 ChatGPT 生成示例时难以实现多样性，并且由于速率限制（rate limiting）的挑战，正考虑手动创建数据集。
- **财务限制影响数据集构建**：`@dirtytigerx` 指出使用 GPT-4 API 生成数据集的高昂成本以及等待 ChatGPT 速率限制的低效率。他们建议尝试使用本地大语言模型（LLMs）作为更具成本效益的选择。
- **速率限制催生创意解决方案**：`@lordofthegoons` 表示打算在应对速率限制的同时，使用 ChatGPT 手动构建聊天机器人数据集。`@dirtytigerx` 进一步建议，利用 runpod 等服务在本地运行大型 LLMs 可能比面对 OpenAI API 的速率限制更便宜且更高效。

---

### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199997735223447642) (19 messages🔥): 

- **模型合并中的权重优化**：`@sanjiwatsuki` 提出了一个假设，认为将模型权重设置为 **略高于 1.0** 可能是最优的，因为 **TIES 解析过程** 可能会导致一些有效权重丢失。
- **在脚本中探索负权重**：`@kquant` 询问 **负数** 是否会破坏合并脚本。`@sanjiwatsuki` 表示不确定，但推测代码可能会在没有问题的情况下处理负权重。
- **选择性模型同化**：`@kquant` 讨论了选择性合并模型以同化所需特征的可能性，提到了 **DARE** 和 **SLERP** 等方法，这些方法可能结合两个在不同基准测试中具有高评分的模型。
- **SLERP 和过拟合模型的表现**：`@kquant` 注意到一个意外的结果，即使用 **SLERP** 合并了两个过拟合模型，并成功保持了它们的测试排名，这引发了关于模型合并背景下过拟合影响的疑问。
- **合并方法需要进一步澄清**：`@kquant` 提到需要更好地理解 **DARE** 和 **SLERP** 在模型合并背景下的区别，并表示希望进行更多的研究和测试。
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1200086950397366292) (1 messages): 

- **新用户寻求 LangChain 指导**：用户 `@nandavikas` 表示难以用 **LangChain** 复制他们之前使用 **PyTorch** 完成的微调过程。他们正在寻求帮助以微调 **Llama2**，从而从 PDF 中提取特定信息，但在 **LangChain** 文档中找不到相关说明。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1200055671194927236) (35 messages🔥): 

- **GPT-4 文档分析的速度问题**：`@romansh2302` 强调了网页版 GPT-4 与通过 API 访问的 `gpt-4-1106-preview` 模型之间存在显著的速度差异，后者速度较慢。`@lugui` 回复称，在高峰使用期间会出现 **API 速度波动**，并提到了 **请求专用容量** 的可能性，这通常针对公司规模的使用。
  
- **寻找 GPT-4 性能解决方案**：在寻求速度补救措施时，`@romansh2302` 被 `@lugui` 告知速度差异与 **高峰时段** 和 API 负载有关，这种情况不易解决。`@og_tort` 建议考虑将 **GPT-3.5** 作为替代方案；然而，`@romansh2302` 发现它在文档分析方面的效果较差。

- **用户面临账户问题**：`@hellomyfriend1576` 报告称在使用 GPT-4 时收到了 **系统异常活动** 的警告。来自社区成员如 `@lugui` 和 `@og_tort` 的回答认为，使用 VPN 或代理以及潜在的 **违规提示词 (flagged prompts)** 可能是导致该问题的原因。

- **DALL-E 文本生成中的拼写错误引发疑问**：`@alwayspercipient` 注意到 DALL-E 在图像创建中经常包含 **拼写错误**。正如 `@muyfashionista` 所指出的，社区已经讨论过这个问题，他还提供了一个关于该主题的 [社区链接](https://community.openai.com/t/does-anyone-experience-issues-with-dall-e3-generating-typos-in-text-within-images/472966)，并提到使用较短的文本可能会减少错误。

- **对 AI 服务和团队账户的困惑**：`@paras4887` 和 `@leithwhitley` 等用户提出了关于特定用例的问题，例如在 **Azure** 上部署 GPT-4 Vision，以及关于使用共享的 **付费团队 GPT 账户** 进行协作的问题。在提供的消息链中未提供解决方案或明确指导。

**提到的链接**：

[TuringsSolutions/PFAF750 · Hugging Face 数据集](https://huggingface.co/datasets/TuringsSolutions/PFAF750)：未找到描述

  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1200050868888801362) (105 messages🔥🔥): 

- **排查 'Always expand code output' 功能**: `@darthgustav.` 为 `@angry_coder` 澄清了“始终展开代码输出”的含义，在亲自测试该功能后确认，这意味着被折叠的代码块将始终保持展开状态以便阅读。
- **在 GPT 中解压库文件**: `@bambooshoots` 建议将库作为 zip 文件上传，并附带一个 `.py` 文件来解压它，并将 `/mnt/data/` 文件夹添加到系统路径（system path）中，这是过去支持的一种方法。`@darthgustav.` 对潜在的安全问题表示担忧，并停止了对这种环境扩展方式的测试。
- **CustomGPT 编辑需要开启新对话**: 针对 `@elegante94` 的疑问，`@darthgustav.` 确认要看到 CustomGPT 编辑后的效果，必须开启新对话，因为正在进行的对话不会同步更新更改。
- **GPT Prompt 指令中的图片附件**: `@elegante94` 询问在 Prompt 指令中附加图片的有效性，`@darthgustav.` 回复称使用简洁的语言比附加图片更好，因为 DALL-E 会自行发挥创意元素。
- **保存/更新 GPT Bots 的问题**: `@rodney.leonardo` 在保存或更新 GPT Bot 时遇到错误并寻求帮助。`@darthgustav.` 建议先移除知识库（knowledge），保存为私有，然后逐个重新附加文件，并指出由于模仿在世艺术家（这是不被允许的）可能会导致封禁。
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **参与 Prompt Engineering**: `@darthgustav.` 就为自定义工作流创建 API 配置向 `@novumclassicum` 提供了建议，包括在 Python 脚本中使用 OpenAPI 以及在 GPT 中使用 Custom Actions。他们还讨论了尽管存在潜在偏差，但仍确保标准化输出的策略。
- **文档分块（Chunking）难题**: `@doublefelix` 寻求一种将长文本切分为段落以便通过 AI 进行语法纠错的方法，而 `@eskcanta` 推荐了一种使用 Python 处理较小章节的方法。Avi 和 Felix 辩论了该任务的最佳实践，Avi 建议使用 AI 进行语义文本分割（semantic text splitting）。
- **AI 驱动的工作流挑战**: `@doublefelix` 测试了多种方法，指导 GPT-3.5 为布道稿转录文本添加段落标记并解决语法问题，但在执行指令和幻觉（hallucinations）方面遇到了问题，这是由于 Token 限制和 AI 上下文中的记忆漏洞导致的。
- **探索 GPT-4 Turbo 作为解决方案**: `@darthgustav.` 提议利用 GPT-4 Turbo 的 Python Tool 来进行语义分析并生成段落，从而绕过 `@doublefelix` 在使用 GPT-3.5 时面临的分块限制。Darthgustav 还强调了在大文档处理中 Token 消耗过大和上下文丢失的问题。
- **考虑使用 NLP 工具进行文本分割**: 由于对管理 GPT-3.5 局限性的复杂性感到沮丧，`@doublefelix` 决定探索第三方 NLP 工具（如 semantic-text-splitter），以尝试自动化大文档的文本分块过程，同时也承认 GPT-4 Turbo 在此类任务中具有更强的能力。

**提到的链接**:

[如何在重复的 API 调用中保持历史上下文？](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395): 每次我调用 API 时，它都没有先前的上下文，这与 chat.openai.com 的场景不同。有没有办法在会话期间保持模型的状态？ response = openai.Completi...

  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **Prompt Engineering 证明具有挑战性**：用户 `@doublefelix` 一直在尝试通过 Prompt 引导 GPT 3.5 处理大量文本并添加段落分隔。尽管尝试了各种策略（包括将任务分解为更小的块），AI 在管理上下文方面仍有困难，经常忽略部分输入文本或产生幻觉 (hallucinates)。

- **寻找正确的方法**：对话强调了管理 AI 上下文的复杂性以及“文档检索曲线”的概念。`@darthgustav.` 建议 GPT-4（特别是 ChatGPT Plus）可能由于其更大的上下文窗口以及通过检索增强生成 (RAG) 处理附件的能力，能更好地完成该任务。

- **API 与 ChatGPT：高性价比方案之争**：`@doublefelix` 正在探索以最低成本自动处理布道文转录稿的方案。`@darthgustav.` 指出，使用带有自定义指令的 ChatGPT 界面可以避免与 API 方式相关的 Token 成本。

- **自定义 Prompt 的潜力**：`@darthgustav.` 指出了使用明确指令和编码指令的“开放变量”来构建 Prompt 的重要性，这可能允许对 AI 输出进行更细致的控制，并有助于完成将文本分段的任务。

- **探索替代方案并考虑下一步**：对话指出了一个备选计划，包括使用 Custom GPTs 和结合 Python 工具的传统 NLP 方法。`@doublefelix` 计划研究 NLP 软件包，例如在 PyPI 上找到的 semantic-text-splitter，以寻找可行的解决方案。

**提及的链接**：

[如何在重复的 API 调用中保持历史上下文？](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395)：与 chat.openai.com 的场景不同，每次我调用 API 时，它都是在没有先前上下文的情况下开始的。有没有办法在会话期间保持模型的状态？response = openai.Completi...

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1200095821480341595) (7 messages): 

- **寻求最佳上下文扩展方案**：`@cryptossssun` 询问了目前扩展上下文能力的最佳方法。`@_cherrry` 对一篇论文给出了积极回应，建议微调 (finetuning) 是一个可行的解决方案。

- **Mistral Instruct 弃用滑动窗口**：`@dreamgen` 讨论了 **Mistral instruct v0.2** 禁用滑动窗口 (sliding windows) 转而采用缩放 `rope_theta` 的影响，并质疑了滑动窗口方法的有效性。他们分享了一个显示这些更改的 [配置文件](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json)。

- **MistralLite 模仿主分支配置**：`@dreamgen` 还注意到 **amazon/MistralLite** 在上下文窗口管理方面遵循与其主分支对应版本相同的配置策略。

- **LLaMA-2-7B-Chat 上下文扩展的卓越效率**：`@stellaathena` 强调了一项令人印象深刻的成就：LLaMA-2-7B-Chat 模型的上下文窗口仅通过 100 个样本和 6 个训练步骤就扩展到了 16,384。

- **SelfExtend 作为非微调替代方案**：在关于上下文扩展的讨论中，`@leontello` 提到 **SelfExtend** 对于那些不想微调模型的人来说是一个有趣的选项。

**提及的链接**：

[config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json)：未找到描述

  

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1200111343458599076) (5 messages): 

- **技术对社会两极分化的影响**：`@ldj` 思考技术可能会导致进一步的两极分化，即能力最弱的个体在堕落中越陷越深，而能力最强的个体则被推向更高程度的自我提升。
- **Twitter 活跃度波动的谜团**：`@fullstack6209` 观察到 Twitter 新帖子发布频率发生了剧烈变化，询问是否有人注意到这种从每 10 分钟 2-3 条帖子到每分钟约 70 条帖子的转变。
- **Twitter AI 降速理论**：`@fullstack6209` 提出一种建议，认为 Twitter 可能故意放慢了 AI 的速度，以此来解释观察到的帖子频率变化。
- **LLM 量化变得简单**：`@pradeep1148` 分享了一个名为 "[AutoGGUF Quantize LLMs in GGUF format in one click](https://www.youtube.com/watch?v=wlPxEq_Mtkc)" 的 YouTube 视频，提供了一个使用共享 Notebook 将大型语言模型转换为 GGUF 格式的资源。

**提及的链接**：

- [Cat Swimming GIF - Cat Swimming Poopsie - Discover &amp; Share GIFs](https://tenor.com/view/cat-swimming-poopsie-silly-gif-14546589990767279660)：点击查看 GIF
- [AutoGGUF Quantize LLMs in GGUF format in one click.](https://www.youtube.com/watch?v=wlPxEq_Mtkc)：使用 Maxim Labonne 提供的 Notebook 将任何 Hugging Face 上的 LLM 量化为 GGUF 格式 #llms #ml #ai #neuralnetworks #deeplearning #gguf https://colab.research.googl...

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1200251006743744512) (2 messages): 

- **Everyone Coder 33B 的基准测试请求**：`@benxh` 请求对 **Everyone Coder 33B** 的量化版本进行 **HumanEval** 基准测试，该模型使用了 llama.cpp 引入的新 **GGUF** 格式。该模型由 [TheBloke](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF) 在 Hugging Face 上发布，且[量化工作由 Massed Compute 提供支持](https://massedcompute.com/)。
- **呼吁对 Hermes Mixtral 进行评估**：用户 `@teknium` 表示希望看到 Hermes 和 Mistral 组合模型的基准测试，并将其称为 **Hermes Mixtral**，同时带上了一个充满期待的 🙏 表情符号。

**提及的链接**：

[TheBloke/Everyone-Coder-33B-Base-GGUF · Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF)：未找到描述

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1200148361370669126) (2 messages): 

- **OpenAI 发布新 Embedding 模型**：`@tsunemoto` 分享了 [OpenAI 的公告](https://openai.com/blog/new-embedding-models-and-api-updates)，揭晓了新一代 Embedding 模型、GPT-4 Turbo、更新的审核模型以及 GPT-3.5 Turbo 的降价。公告强调了默认的**数据隐私**和改进的 API 管理工具，以及**新 Embedding 模型更低的价格**。

- **Genie：一种高质量合成数据的方法**：`@metaldragon01` 介绍了一篇关于 **Genie** 的论文，这是一种创建高质量基于内容（content-grounded）的数据的新方法，详见 [arXiv 上的发表作品](https://arxiv.org/abs/2401.14367)。据称 Genie 生成的数据非常精炼，在人工评估中被认为既自然又高质量，对于改进长篇问答（Long-Form Question-Answering）和摘要模型具有重要意义。

**提及的链接**：

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)：缺乏用于基于内容的生成任务的高质量数据已被认为是推进这些任务的主要障碍。为了填补这一空白，我们提出了 Genie，一种自动化的新方法...
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在发布新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1200033366272122910) (362 条消息🔥🔥): 

- **介绍 GPU 加速 AI**：用户 `@n8programs` 讨论了利用 WebGL 进行 **machine learning computations**（机器学习计算）。由于 WebGL 不支持原始缓冲区（raw buffers），通常通过将数据打包进纹理（textures）来实现（正如 `@everyoneisgross` 和 `@n8programs` 交流的想法）。这种方法可以实现 GPU 加速，正如 TensorFlow.js 所实践的那样，尽管目前存在一些限制，例如最大向量大小为 16,000 个元素。
- **模型训练的明智建议**：`@intervitens` 建议，为了实现有效的多 GPU 设置，服务器或高端桌面（HEDT）主板是必要的，因为这需要充足的 PCI-e 通道。他建议使用二手 Gen2 EPYC 以平衡性能和经济性。讨论内容还包括矿机风格机箱、支持 4 张双槽显卡的双槽间距主板，以及定制水冷方案。
- **实时词嵌入调整**：用户 `@everyoneisgross` 描述了一个用于快速动态调整 word2vec 模型中词嵌入（word embeddings）的系统，允许根据用户输入增加或减少权重来提供实时反馈。该过程在小语料库上运行速度很快，非常适合基于 LLM（本例中为 **Mistral instruct**）获取的新数据进行扩展或优化。
- **Mixtral Instruct 表现出奇强劲**：`@gabriel_syme` 询问为什么 **Mixtral instruct** 的表现优于许多微调模型。`@intervitens` 回答说，Mixtral instruct 在遵循指令方面特别出色，而且 MoE 模型的微调可能还存在一些尚未解决的问题。
- **探索模型的 GPU 增强**：`@carsonpoole` 分享了在适配 phi2 模型变体方面的进展，该变体采用了修改后的微调方法。他计划在 Hugging Face 上发布权重，并可能开发一个基于 LLaMA 的模型变体。该模型还使用了 chatml 数据和 token 进行微调，并将其整合到模型的知识库中。

**提及的链接**：

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://openreview.net/forum?id=AL1fq05o7H)：基础模型（Foundation models）目前驱动着深度学习中大多数令人兴奋的应用，几乎普遍基于 Transformer 架构及其核心的 attention 模块。许多...
- [Human vs. Machine: Intelligence per Watt](https://meditations.metavert.io/p/human-vs-machine-intelligence-per)：思考机器不会立即在所有领域取胜的可能性
- [Google Colaboratory](https://colab.research.google.com/drive/1-D6ZGE3SZZbIkqhWfxun8CwQWBD5YC2d?usp=sharing)：未找到描述
- [🤗 Transformers](https://huggingface.co/docs/transformers/)：未找到描述
- [Recommendations on new 2 x RTX 3090 setup](https://forums.fast.ai/t/recommendations-on-new-2-x-rtx-3090-setup/78202)：你好，我正在出售旧的 GTX 1080，并用新的 RTX 3090 升级我的深度学习服务器。我还考虑在明年晚些时候再增加一张 RTX 3090。我从多个渠道了解到涡轮式（blower-style）...
- [Growing Living Rat Neurons To Play... DOOM?](https://www.youtube.com/watch?v=bEXefdbQDjw)：前往 https://squarespace.com/thethoughtemporium 使用代码 thethoughtemporium 在首次购买网站或域名时节省 10%...
- [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralConfig)：未找到描述
- [GitHub - cg123/mergekit at mixtral](https://github.com/cg123/mergekit/tree/mixtral)：用于合并预训练大语言模型的工具。- GitHub - cg123/mergekit at mixtral
- [EVGA SuperNOVA 1600 P2, 80+ PLATINUM 1600W, Fully Modular, EVGA ECO Mode, 10 Year Warranty, Includes FREE Power On Self Tester Power Supply 220-P2-1600-X1](https://www.evga.com/products/product.aspx?pn=220-P2-1600-X1)：支持第四代 Intel Core 处理器（C6/C7 空闲模式）。介绍 EVGA SuperNOVA 1600 P2 电源。该电源以 1600W 的连续功率输出和 92% 的效率树立了标杆...
- [Designs Beyond The Reticle Limit](https://semiengineering.com/designs-beyond-the-reticle-limit/)：芯片正面临技术和经济障碍，但这几乎没有减缓设计尺寸和复杂性的进步速度。

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1200051857771475034) (35 条消息🔥): 

- **寻求 CodeLlama 微调规格**：`@ganymede123` 询问了微调 **CodeLlama 34B** 的工作站规格，并考虑使用 4xA6000 GPU。`@teknium` 回复称这仅够运行 *qlora*，并表示全量微调（full fine-tune）几乎需要一套完整的 DGX 设备。
  
- **T5 微调困难**：`@maxpappa` 在对微调版本的 **T5** 进行对齐时面临挑战，注意到输出具有确定性且奖励准确率（reward-accuracies）稳定在 0.5。尽管调整了优化器（optimizers）和调度器（schedulers），`@locutusque` 仍建议避免使用 paged 8bit Adam，`@carsonpoole` 则建议对 encoder 中的 infs 进行数值截断（clamping），以处理明显的数值不稳定性。

- **用于进攻性网络安全和 CTF 的 LLM**：`@useewhynot` 寻求适用于进攻性网络安全或 CTF 挑战的 LLM 推荐。`@kenakafrosty` 和 `@georgejrjrjr` 推荐了 [WhiteRabbitNeo-13B](https://huggingface.co/whiterabbitneo/WhiteRabbitNeo-13B) 和 [WhiteRabbitNeo-33B-v1](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1)，这些模型可在 HuggingFace 上获取。

- **评估 AI 编程器**：`@findmyke` 询问了目前最出色的编程 LLM。`@.ben.com` 链接到了 [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)，该排行榜通过严格的测试评估 AI 编程器，可作为做出明智决策的资源。

- **使用 Llama Factory 进行微调**：`@moconna` 表达了使用 Llama Factory 微调 **Mixtral** 的意图，并征求有关必要超参数（hyperparameters）的建议。随后的讨论中未提供具体的超参数或模板。

**提到的链接**：

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)：未找到描述
- [WhiteRabbitNeo/WhiteRabbitNeo-33B-v1 · Hugging Face](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1)：未找到描述
- [WhiteRabbitNeo/WhiteRabbitNeo-13B-v1 · Hugging Face](https://huggingface.co/whiterabbitneo/WhiteRabbitNeo-13B)：未找到描述
- [WhiteRabbitNeo - A co-pilot for your cybersecurity journey](https://www.whiterabbitneo.com/)：未找到描述

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1200102006929506434) (3 条消息): 

- **寻求用于 3B Obsidian 的 Python 脚本**：`@vic49.` 正在寻找一个简单的 Python 脚本，利用 transformers 库来运行 **3b Obsidian** 模型。他们指定代码应允许远程执行（`remote code = true`）。
- **代码重构进行中**：`@qnguyen3` 确认他们正在进行代码重构，以兼容最新的 **llava repo**，从而增强 **3b Obsidian** 的功能。
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1200042055980822649) (219 条消息🔥🔥): 

- **特定情况下微调首选 Base Model**：`@le_mess` 建议在针对基础训练数据有限的语言进行 Fine-tuning 时，优先选择 Base Model 而非 Instruct 变体。该建议是在 `@henriklied` 分享了其使用 10 万篇挪威语文章数据集进行 Fine-tuning 的方法后提出的。
  
- **训练时长与过拟合担忧**：`@le_mess` 建议在 4 个 epoch 时停止训练以防止 Overfitting，这是针对 `@henriklied` 在 10 个 epoch 的 Fine-tuning 过程中观察到 eval loss 趋于平缓的回应。`@henriklied` 还分享了一个[链接](https://gist.githubusercontent.com/henriklied/3dd25bf3090ddb792ec3b1e702fe321d/raw/a155986e117ea69c384ddd87e0580ec18c1c0cef/gistfile1.txt)，指向 debug flag (`prepare_dataset`) 的输出，用于诊断训练设置。
  
- **训练模型的有效聊天格式**：`@suikamelon` 和 `@c.gato` 讨论了训练语言模型时不同聊天格式的有效性，`@suikamelon` 介绍了使用独特块字符的 "BlockML"，以期提高 token 效率。此外还提到了在训练中集成 ChatML token 的挑战，因为它们出现的频率较低。
  
- **关于使用非常规 Token 进行模型训练的讨论**：`@suikamelon` 报告了使用独特块字符代替 ChatML 标签的初步成功，指出尽管数据集仅约 100 个样本，但 `▀` 和 `▄` 在 Mistral 上提供了可靠的 tokenization。
  
- **关于模型量化与优化器设置**：`@dangfutures` 表达了对使用 AWQ 进行 Quantization 的偏好；`@bjoernp` 询问在配置中设置 Optimizer 是否会覆盖默认的 DeepSpeed Optimizer，`@noobmaster29` 确认由于已从默认 DeepSpeed 配置中移除，应使用配置中的 Optimizer。此外还提到了 `@winglian` 提交的一个相关的 DeepSpeed 配置 PR（[PR #1208](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208)）。

**相关链接**：

- [axolotl/deepspeed_configs/zero3.json at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3.json)：欢迎在 GitHub 上通过创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [Pullrequest GIF - Pullrequest - Discover &amp; Share GIFs](https://tenor.com/view/pullrequest-gif-20256291)：点击查看 GIF
- [more checks and fixes for deepspeed and fsdp by winglian · Pull Request #1208 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208)：未找到描述

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1200172001835352115) (1 条消息): 

- **gahdnah 的快速撤回**：`@gahdnah` 在注意到最新 commit 中的**活跃开发**后撤回了一条消息，这表明该项目区域受到了良好的监控且发展迅速。
- **社区庆祝获得资助**：`@gahdnah` 对获得的一项资助表示兴奋和祝贺，用表情符号庆祝这一好消息。 🎉🎉🎉
  

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1200033196105023572) (47 messages🔥): 

- **不使用 BitsandBytes 加载模型**：`@matanvetzler` 在尝试将 qlora 训练的模型加载到 vLLM 时遇到了 `ValueError`，因为 vLLM 不支持 bitsandbytes 量化。他们得到的建议是：在 vLLM 中以 fp16 加载模型，或者使用 AutoAWQ 进行量化以适应 VRAM 限制。
- **合并 QLoRA 训练的模型**：`@stefangliga` 提供了一个 GitHub 上的 [合并脚本](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base) 链接，用于将 qlora 训练的模型合并回基础模型，这是一个独立于模型服务（serving）机制的过程。他们进一步提到 Tim Dettmers 建议合并量化模型，并链接到一条 [Twitter 帖子](https://twitter.com/Tim_Dettmers/status/1694654191325573456) 以作详细说明。
- **SQL 数据集的困惑**：`@sadaisystems` 对 SQL 数据集在训练几步后出现极低的 loss 表示担忧，怀疑这是否意味着数据集缺乏多样性或模型已过于熟练。`@caseus_` 认为 SQL 的确定性（deterministic）特征可能解释了低 loss，并建议除非数据中有复杂案例，否则可以停止训练。
- **使用 Axolotl 在训练期间进行 Benchmark 评估**：`@caseus_` 告知 `@sadaisystems` 关于 axolotl 中的 `do_bench_eval: true` 选项，可以在训练期间运行小型评估，并指出他们使用来自 [dharma-1](https://huggingface.co/datasets/pharaouk/dharma-1/tree/main) 的数据集作为 Benchmark，这对于相对改进检查非常有用。
- **持续预训练咨询**：`@nickbro0355` 寻求关于如何在 Mistral 上进行持续预训练（Continuous Pretraining）的帮助，`@dangfutures` 表示他们一直在尝试，`@caseus_` 则询问具体的数据集以便提供进一步帮助。

**提到的链接**：

- [qlora/qmerge.py at main · jondurbin/qlora](https://github.com/jondurbin/qlora/blob/main/qmerge.py)：QLoRA：量化 LLM 的高效微调。通过在 GitHub 上创建账户为 jondurbin/qlora 的开发做出贡献。
- [pharaouk/dharma-1 at main](https://huggingface.co/datasets/pharaouk/dharma-1/tree/main)：未找到描述
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base)：尽管提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1200086950594498580) (7 messages): 

- **新数据集发布**：用户 `@dangfutures` 强调了 [Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset) 上一个用于训练 **Snorkel 模型** 的新数据集。该数据集仅利用了 **UltraFeedback** 的 Prompt，没有使用外部 LLM 的响应。

- **Snorkel-Mistral 训练方法论说明**：该数据集的创建过程包括：针对每个 Prompt 使用 **Mistral-7B-Instruct-v0.2** 生成 5 个响应变体，使用 **PairRM** 对其进行重排序，然后应用 **Direct Preference Optimization (DPO)** 进行三次迭代的 LLM 更新。

- **Mistral 获得升级**：用户 `@dangfutures` 表示 **Mistral 7** 已经过微调，这可能是指数据集方法论中提到的改进。

- **提到 ALPACA 的评估指标**：用户 `@dangfutures` 提到了一个与 ALPACA 相关的数字，尽管“34%”的具体背景或含义尚未明确。

- **表现令人印象深刻**：`@dangfutures` 的后续发言指出，尽管最初感知的百分比较低，但其性能被指出 **优于旧版本的 GPT-4**。

- **俏皮的回应**：用户 `_dampf` 分享了一个来自 [Tenor](https://tenor.com/wSUt.gif) 的 GIF，这可以被解释为对前述消息的反应，尽管 GIF 使用的具体背景在对话中并未阐明。

**提到的链接**：

- [snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset)：未找到描述
- [Sure Jennifer Lawrence GIF - Sure Jennifer Lawrence The Mocking Jay - Discover &amp; Share GIFs](https://tenor.com/wSUt.gif)：点击查看 GIF

  

---

### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1200279388797804555) (2 条消息): 

- **寻求 DPO 训练图表方面的建议**：`@noobmaster29` 询问是否有资源可以帮助更好地理解 **DPO 训练图表 (training plots)**。然而，聊天记录中没有提供任何回复或资源。
- **DPO 训练的数据集困境**：`@noobmaster29` 询问了 **DPO 数据集** 的必要组成部分，提到他们已经包含了 *prompt/input* 和 *chosen rejected pair* 列，但在数据集处理过程中遇到了问题。聊天记录中没有提供进一步的说明或排错建议。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=wlPxEq_Mtkc
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199991535681417248) (115 条消息🔥🔥): 

- **寻求教室连接方案**：`@josemanu72` 询问如何在教室内将 LM Studio 作为服务器运行，并让学生从桌面端连接。`@fabguy` 建议使用 [frontend](https://github.com/JShollaj/Awesome-LLM-Web-UI) 和反向代理 (reverse proxy) 设置；随后提到在另一个频道解决了该问题。
- **Ubuntu 上的 GPU 问题**：`@xmiruso` 在 Ubuntu 环境下使用 Geforce Nvidia 3090 时遇到了 LM Studio 无法利用 GPU 的问题。在与 `@fabguy` 等人讨论后，通过卸载并重新加载模型解决了问题，从而提升了处理速度。
- **代理问题阻碍模型搜索**：用户 `@laooopooo_02864` 由于代理问题在执行模型搜索功能时遇到挑战，`@heyitsyorkie` 推测他们可能处于 Hugging Face 被屏蔽的国家。
- **从移动端访问本地模型**：`@cloakedman` 寻求从手机访问其 LLM 模型的方法，`@wildcat_aurora` 提供了一个指向 LM_Chat_TTS_FrontEnd ([front-end](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html)) 的 GitHub 链接，该项目允许与 LM Studio 模型进行交互。
- **讨论最佳实践与错误解决**：不同用户讨论了 LM Studio 中最适合编码的 AI、并行模型运行、针对错误联系支持，并回顾了他们遇到的问题和修复方法，包括 `@fate4real` 分享的关于 AMD 显卡的 GPU 驱动警告。

**提到的链接**：

- [GitHub - FriendofAI/LM_Chat_TTS_FrontEnd.html: LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, making it suitable for a wide range of users interested in exploring voice interactions with AI models.](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html): LM_Chat_TTS_FrontEnd 是一个简单而强大的界面，用于通过文本转语音功能与 LM Studio 模型进行交互。该项目旨在轻量且用户友好，适合对探索 AI 模型语音交互感兴趣的广泛用户。
- [GitHub - JShollaj/awesome-llm-web-ui: A curated list of awesome Large Language Model (LLM) Web User Interfaces.](https://github.com/JShollaj/Awesome-LLM-Web-UI): 一个精选的优秀大语言模型 (LLM) Web 用户界面列表。

  

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199997195924033607) (54 条消息🔥): 

- **C++ Redist 解决模型加载错误**：`@rparada` 在加载包括 **Stable Code**、**Deepseek**、**Codellama** 在内的多个模型时遇到了错误，但在 `@heyitsyorkie` 的建议下，通过更新 C++ Redistributables（可再发行组件）解决了该问题。

- **模型能力评估**：`@heyitsyorkie` 评论称 **Magicoder DS 6.7b** 和 **GPT4** 的性能非常接近，并进一步阐述目前还没有单一的本地多模态开源模型可以与 **GPT4** 抗衡。

- **在 Azure 云上使用 GPT4**：`@mickael6102` 分享了他们公司在 **Azure 云上本地运行 GPT4** 的情况。这引发了与 `@vbwyrde` 关于数据隐私担忧、成本以及微软与 OpenAI 在专有数据的使用和控制权方面关系的讨论。

- **开源模型选项**：`@vbwyrde` 讨论了一个名为 **"Intern" (InternLM)** 的新模型，并提供了一个链接，声称该模型具有卓越的能力，例如 200k 的上下文窗口（context window）和函数调用（function calling）。`@mickael6102` 对此表示感兴趣，并提到正在使用 **Solar-10b 进行函数调用**。

- **辩论 Meta Llama2 的策略**：针对 `@vbwyrde` 的发言，`@.gumdro` 和 `@ptable` 推测了 Meta 提供 **Llama2 等开源模型** 的动机，认为原因包括设定标准以从下游产品开发中获益，以及抢占 OpenAI 等竞争服务的市场空间。

**提到的链接**：

- [internlm (InternLM)](https://huggingface.co/internlm)：未找到描述
- [Mark Zuckerberg Adjust GIF - Mark Zuckerberg Adjust Facebook - Discover &amp; Share GIFs](https://tenor.com/view/mark-zuckerberg-adjust-facebook-smile-on-trial-gif-11618142)：点击查看 GIF

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1200212033254215700) (4 条消息): 

- **切换 MoE 模型时的 Bug 警报**：`@msz_mgs` 报告了一个 **Bug**，即从 **4X MoE** 模型切换到 **2X MoE** 模型时会导致错误且无法更改，必须重启应用。
- **感谢并请求模型详情**：`@yagilb` 确认了 `@msz_mgs` 的 Bug 报告，并请求分享所使用的 **2x moe** 和 **4x moe** 模型的详细信息以便进一步调查。
- **关于 MoE 模型配置的见解**：`@dagbs` 提供了关于设置 **num_experts_used** 配置的技巧，建议对于 **4x 模型**，正确的设置应该是 2 个专家（experts）。
- **最新更新的性能问题**：`@golangorgohome` 对 **0.2.11 版本** 在 32GB RAM 的 Windows 11 上的糟糕表现表示担忧，提到尽管网络连接很快，但搜索图标响应缓慢且搜索时间过长。
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1200033522530914395) (11 条消息🔥): 

- **高性价比 GPU 偏好**：`@gitanos` 询问了 **4060ti 16GB RAM** 的价值，`@heyitsyorkie` 回应建议投资二手的 **3090** 以获得更好的性价比，在英国其价格仅略高一点。
- **e0.211 的兼容性担忧**：用户 `@madan.pandit` 报告了模型停止工作的问题，特别是在使用 e0.211 版本时。另一位用户 `@heyitsyorkie` 表示自己这边没有问题，但指出 *GGML 模型已被弃用，转而支持 llama.cpp*。
- **GGUF 模型的内存错误**：`@madan.pandit` 提到在尝试使用 **GGUF 模型** 时收到内存不足的错误。
- **推荐 M2 Mac Studio 运行 LLM**：`@heyitsyorkie` 建议购买 **顶配 M2 Mac Studio** 是运行 LLM 的完美选择，并称赞了其小巧的体积和美观的设计。
- **对旧款 GPU 的不同看法**：`@docorange88`、`@wildcat_aurora` 和 `@rugg0064` 之间的对话讨论了使用 **P40 或 M40 GPU** 进行机器学习的可行性。共识似乎倾向于 P40 GPU，而通常不推荐 M40 GPU。
  

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1200006286159401030) (46 messages🔥): 

- **Nobara 上的 VRAM 检测问题**：`@mudf00t` 报告称 LM Studio 无法识别 Nobara 系统上 RTX 3090 的 VRAM。`@yagilb` 提供了一个变通方法，但指出该方案是针对不同配置下的 Nvidia 的，不适用于 `@pdg` 在 Mac M2 上的问题。

- **最终版本中模型加载失败**：`@pdg` 在升级到 0.2.11 版本后遇到所有模型均无法加载的问题，而这些模型在旧版本中可以正常工作。通过 `@yagilb` 提供的[此链接](https://releases.lmstudio.ai/mac/arm64/0.2.10/latest/LM+Studio-0.2.10-arm64.dmg)降级到 0.2.10 版本后出现了新的错误，随后用户请求获取更早的 0.2.9 版本的链接。

- **应用下载停滞问题**：`@khalifa007` 遇到了应用下载卡住的问题。`@yagilb` 建议该问题可能与用户的网络连接或防火墙有关，并考虑使用 VPN 可能会有帮助。

- **异常 RAM 错误及临时修复**：`@pdg` 报告了一个显示 RAM 不足的错误，尽管系统有 16 GB 可用空间。他们发现，先以短句开始并让模型先响应，可以避免在提交长文本时出现错误。

- **关于上下文长度设置的见解**：`@mattjcly_55150` 建议 `@pdg` 遇到的错误可能是由于初始上下文长度（context length）设置引起的，并建议调整该设置或上下文溢出策略（context overflow policy），以避免在输入长文本时出错。
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1200325029498458112) (1 messages): 

- **Autogen 中空字符串的链接失效**：`@sunglasses.emoji` 报告称有关空字符串的置顶链接已**失效**，并正在寻求关于在 autogen studio 中创建自定义 Agent 类的帮助。目前尚未提供更多细节或解决方案。
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1200104377801773109) (10 messages🔥): 

- **开源模型框架的使用困境**：`@pefortin` 表达了挫败感，尽管运行着 mixtral8x7B 和 deepseek coder 33B 等中型模型，但 **memGPT**、**crewai** 和 **Open Interpreter** 等开源模型仍无法正确使用它们所拥有的工具和元素。
- **mudf00t 的模型探索**：`@mudf00t` 正在测试各种模型，并强调拥有 **RTX 3090** 可以加载比其他设备大得多的模型。
- **API 健忘症**：`@mudf00t` 幽默地指出，**OpenAI** 的模型（包括通过 API 访问的模型）似乎无法记住当前的 API 本身，从而在训练期间导致上下文问题。
- **微调重点**：`@222gate` 提到停止了与 memGPT 的集成，目前正在研究为特定函数调用（function calls）微调 **mistral** 模型，类似于在 **memgpt** 数据集上看到的努力。
- **Mistral 幻觉出的目录**：`@mudf00t` 分享了一个有趣的案例，**Mistral** 凭空创造了一个包含 node 应用的虚假目录结构，并展示了一个不存在的文件的代码。
  

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1200008042968789032) (163 messages🔥🔥): 

- **GPU 租赁方案讨论**：`@mrdragonfox` 推荐了 **runpod, vast, and lambda** 等按小时租赁 GPU 的服务，随后提到 **Kaggle** 每周提供高达 30 小时的免费 GPU 使用时长。
- **Mistral 消费限额与支持**：`@glorfsf` 提出了修改订阅选项中消费限额的问题，`@mrdragonfox` 澄清该限额默认为 €150。`@mrdragonfox` 还建议联系 **support@mistral.ai** 以寻求修改消费限额的帮助。
- **BART 模型局限性与 LLM 建议**：`@miraimech` 对 Hugging Face 上的 BART 模型在生产环境中的表现表示不满，`@mrdragonfox` 对此建议使用具有更高上下文窗口（context windows）的开源模型。
- **模型讨论与 API 问题**：`@ethux` 和 `@i_am_dom` 讨论了 Mistral 模型的应用以及 GitHub Copilot 所使用的模型版本背后的复杂性，`@mrdragonfox` 澄清了其当前的后端及对 GPT-3.5 的使用情况。
- **Mistral 7B 集成与用例咨询**：`@sophiamyang` 征集 Mistral 模型有趣的用例，而 `@ethux` 和 `@f127467` 分享了他们在模型集成方面的经验和挑战，寻求社区对有效实现方案的见解。

**提到的链接**：

- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/19emcxp/stanford_and_openai_just_released_a_research/): 未找到描述
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): 未找到描述
- [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2): 一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - GitHub - turboderp/exllamav2
- [TuringsSolutions/PFAF750 · Datasets at Hugging Face](https://huggingface.co/datasets/TuringsSolutions/PFAF750): 未找到描述

---

### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1200043535878074391) (9 messages🔥): 

- **微调与推理的显存需求对比**：用户 `@l0gr1thm1k` 为 `@ethux` 澄清，所讨论的显存容量是针对微调（finetuning）而非训练（training）的，并强调重点在于将模型加载到内存中。
- **Mixtral 的显存占用**：在回复 `@ethux` 时，`@l0gr1thm1k` 确认在四张 T4 GPU 上拥有足够的显存来处理 Mixtral 在 4-bit 推理时至少需要的 26GB 显存。
- **实际显存使用报告**：`@l0gr1thm1k` 报告称，仅加载模型时的 GPU 显存占用就超出了预期，这表明实际使用量可能高于之前分享的预估数字。
- **量化效率辩论**：`@mrdragonfox` 建议使用 exllamav2 进行量化，而不是 bnb 4 bit，并对在显存效率场景下使用 accelerate 表示质疑。

---

### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1200202437080916009) (3 messages): 

- **传统指标不适用于 LLM**：`@adrienbufort` 强调 **BLEU 和 ROUGE** 对于评估大语言模型（LLMs）或指令微调后的 LLM 没有帮助，因为这些指标传统上是用于翻译性能评估的。
- **用于类人 LLM 评估的 "Elo"**：`@adrienbufort` 强调了 **"elo"** 系统，这是一种模仿国际象棋排名的系统，非常接近人类对 LLM 评估的偏好，可在 [arena.lmsys.org](https://arena.lmsys.org/) 查看，尽管它需要人工参与。
- **通过 MMLU 和 Alpaca 进行结构化评估**：`@adrienbufort` 指出可以使用多选题，如 **Massive Multitask Language Understanding (MMLU)** 基准测试（[MMLU 论文](https://arxiv.org/pdf/2009.03300.pdf)）来进行明确的 LLM 性能测量，以及使用 **Alpaca eval**（[Alpaca GitHub](https://github.com/tatsu-lab/alpaca_eval)）通过另一个 LLM 来评估回答。
- **标准化 Alpaca Eval 发布公告**：`@akshay_1` 宣布 **标准化版本的 Alpaca eval** 现已在市场上可用。

---

### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1200043856981405820) (1 messages): 

- **别具一格的 AI 浏览器查询**：用户 `@sublimatorniq` 展示了 *SoContextual.com*，这是一个用于 AI 浏览器查询的工具，其中包含 DOM 节点引用。该工具支持 **MistralAI**，并曾在 [Hacker News](https://news.ycombinator.com/item?id=39128480) 上发布。

**提到的链接**：

[无标题](https://news.ycombinator.com/item?id=39128480): 未找到描述

---

### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1200185918250819814) (8 messages🔥): 

- **新人寻求 RAG 指导**：用户 `@xrdg` 加入聊天，询问如何为 RAG 应用构建 prompt 结构的建议。未提供其具体用例的详细信息。
- **使用 DSPy 进行 prompt 优化**：`@akshay_1` 建议使用 **DSPy** 来优化 prompt 结构，引发了与 `@xrdg` 的简短互动。
- **来自危地马拉的问候**：在后续消息中，`@xrdg` 发送了来自 🇬🇹 的问候，但没有提供进一步的讨论点。
- **探索 Mistral prompt 示例**：`@xrdg` 分享了他们一直在使用 **langchain, chroma, 和 Mistral 7B**，并参考了一个 [prompting guide](https://www.promptingguide.ai/models/mistral-7b)。他们提供了一个包含 Mistral 7B 概述和各种相关资源的链接。
- **优化 RAG 技术栈**：`@akshay_1` 建议 `@xrdg` 当前的 RAG 技术栈可以进一步优化，并询问该项目是业余爱好还是用于生产环境，但 `@xrdg` 未提供更多背景信息。

**提到的链接**：

[Prompt Engineering Guide](https://www.promptingguide.ai/models/mistral-7b)：Prompt Engineering 的全面概述

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1200021798826278983) (35 messages🔥): 

- **早停（Early Stopping）难题仍在继续**：用户 `@digitalphotographer` 在其 prompt 中仍面临早停问题，尽管未使用控制 token 或特殊字符。他们之前已向 Mistral 团队提供了带有可复现示例的 notebook，但尚未收到回复。
  
- **报告每月使用限额 Bug**：用户 `@ewanhc`, `@ethux`, 和 `@fersingb` 报告了一个 Bug：在账单页面尝试修改每月使用限额后，限额会重置为 150 欧元，即使初衷是降低限额。他们已通过电子邮件向 Mistral 支持团队报告了此问题。

- **API 托管查询已澄清**：`@loicboutet` 询问了 Mistral API 的托管位置，并获知其托管在瑞典的 Azure 上，该信息是在隐私页面找到的。

- **API 的 "max_tokens" Bug 浮出水面**：`@mrxavierx` 发现并报告了一个 Bug：将 "max_tokens" 设置为 1 会导致 500 内部服务器错误，而不是返回单个 token 响应或正确的验证错误。该问题已记录在 Mistral 的 GitHub 仓库中（[Issue #122](https://github.com/mistralai/mistral-src/issues/122)）。

**提到的链接**：

- [BUG: API /completion endpoint returns 500 (server error) when sending &quot;max_token&quot; = 1 · Issue #122 · mistralai/mistral-src](https://github.com/mistralai/mistral-src/issues/122)：当我使用 API 端点 /completion 时，发现当 "max_tokens" 正文主体字段设置为 1 时存在一个 Bug。它没有返回 1 个 token 的响应或验证错误，而是……
- [无标题](https://console.mistral.ai/billing/)：未找到描述

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1200057056078594111) (29 messages🔥): 

- **寻找 SAM 微调方法**：`@the_alt_man` 询问了用于微调 Meta SAM 模型的代码库，发现原始代码中不包含该功能，并提到使用了采用 Lightning 但仅限于 GPU 使用的 `AutoGluon` 工具箱。

- **讨论联邦学习的可行性**：`@elyxlz` 询问了在没有 infiniband 的情况下进行多节点训练以及模型合并步骤的可行性。`@stellaathena` 指出了关于类孤岛设备训练的实验，而 `@smerkyg` 指向了一项潜在的近期研究，`@_r0n12` 通过 [arXiv.org 链接](https://arxiv.org/abs/2311.08105) 确认其为 DiLoCo 论文。

- **访问 Pile 数据集**：`@sk5544` 寻求如何访问 Pile 数据集的信息，得到了 `@stellaathena` 的指引以及 `@elyxlz` 提供的私信帮助。

- **金融律师的类比描述**：`@catboy_slim_` 提供了一个类比，将金融领域的律师比作战地救护兵，传达了他们在快节奏金融事件中的被动反应地位。

- **项目贡献与数据集创建呼吁**：`@pinconefish` 提供了 ML 专业知识（特别是 CV 领域）以贡献于现有项目，而 `@stellaathena` 和 `@wonkothesensible` 激发了一个想法：创建一个专注于显示 10:10 的模拟时钟数据集，以研究域外泛化（out-of-domain generalization），并指出潜在的模型崩溃和主动学习案例。

**提到的链接**：

[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105)：大型语言模型 (LLM) 已成为机器学习许多应用中的关键组件。然而，训练 LLM 的标准方法需要大量紧密互连的加速器……

  

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1200008066314285086) (125 条消息🔥🔥): 

- **Byt5 的效率受到质疑**：`@main.ai` 提到 Byt5 表明字节级 Transformer 的效率较低，这引发了与 `@the_random_lurker` 之间关于比较 token 和字节序列长度公平性的辩论。
- **论文接收方面的困扰**：`@stellaathena` 对一篇评审分数看似很高但却被莫名拒绝的论文表示困惑，认为这涉及 meta-reviewer 的失误。讨论强调了学术会议中论文申诉流程的困难和缺乏透明度。
- **大语言模型的 Proxy Tuning**：`@digthatdata` 分享了一篇关于 LLM [Proxy Tuning](https://arxiv.org/abs/2401.08565) 的论文链接，这是一种传统微调的高效替代方案，利用较小模型的预测来指导较大的基础模型，展示了显著的性能提升。
- **对 Self-Rewarding LM 论文的批评**：`@thatspysaspy` 批评了一篇关于 Self-Rewarding LM 的论文，原因是其在训练期间使用了 Claude 2 和 Llama-2-chat 等更强的模型，认为这削弱了论文的论点，并可能导致未来研究工作的误导。
- **Chess.com 关于国际象棋 AI 对手的虚构文章**：`@clockrelativity2003` 分享了一篇 [Chess.com 文章](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024)，预测了 2024 年 AI 在国际象棋领域的未来。然而，`@alexanderrgriffing` 认为该文章是由 GPT 编写的，并对其严肃性表示怀疑。

**提到的链接**：

- [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565)：尽管大型预训练语言模型具有通用能力，但它们始终能从进一步的适配中受益，以更好地实现预期的行为。然而，微调这些模型已变得……
- [Transformers and Cortical Waves: Encoders for Pulling In Context Across Time](https://arxiv.org/abs/2401.14267)：ChatGPT 等 Transformer 网络和其他大语言模型（LLMs）的能力引起了全世界的关注。其性能背后的关键计算机制……
- [The Quantum Leap of Checkmate: Chess in the Age of AI The year is 2024](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024)：现在是 2024 年。机器人漫步在街道上，全息图在客厅里闪烁，自动驾驶汽车以资深出租车司机的优雅应对高峰时段。然而，在一个简陋的木板上，一个古老的……
- [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361)：本文介绍了 MoE-Infinity，这是一个具有成本效益的混合专家模型（MoE）推理系统，实现了激活感知的专家卸载。MoE-Infinity 的特点是序列级的专家激活追踪……
- [Evaluating the Medical Knowledge of Open LLMs - Part 1 &mdash; MedARC](https://www.medarc.ai/blog/medarc-llms-eval-part-1)：在这篇 MedARC 博客文章中，我们比较了通用型和医学领域特定的 LLM（如 GPT-4、Mistral 和 Llama），并评估了它们在 MultiMedQA 任务中的表现……
- [MVDream: Multi-view Diffusion for 3D Generation](https://arxiv.org/abs/2308.16512)：我们介绍了 MVDream，这是一种多视图扩散模型，能够根据给定的文本提示生成一致的多视图图像。通过同时学习 2D 和 3D 数据，多视图扩散模型可以……
- [| bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v3)：未找到描述
- [CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition](https://arxiv.org/abs/2310.11830)：多语言语音处理需要理解情感，由于标注数据有限，这项任务变得非常困难。CLARA 最小化了对标注数据的依赖，增强了跨语言的泛化能力。它……
- [Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom](https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1)：虽然 AlphaFold2 (AF2) 和 RoseTTAFold (RF) 通过实现高精度蛋白质结构建模改变了结构生物学，但它们无法模拟共价修饰或相互作用……

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1200153068268961913) (2 条消息): 

- **确认修复集成问题**：`@hailey_schoelkopf` 表示如果针对某个未指明的集成问题有必要，已准备好合并修复程序，并对该行为表示惊讶，希望亲自进行测试。
- **添加 Weights and Biases 支持**：`@hailey_schoelkopf` 分享了由 `@ayulockin` 提交的 [GitHub pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339)，该 PR 为 `lm-evaluation-harness` 添加了对 **Weights and Biases** 的支持。他们正在考虑新创建的 `wandb.py` 文件在项目结构中的最佳位置。

**提到的链接**：

[feat: Add Weights and Biases support by ayulockin · Pull Request #1339 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339)：在 #359 中，@parambharat 曾提议添加对 W&B 日志记录的支持。然而，那是在重大的重构进入之前完成的。作为 lm-evaluation-harness 和 wandb 的用户，我提交了这个 PR ...

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1200163709352427631) (16 条消息🔥): 

- **寻求 QLoRA 微调指导**：`@kenakafrosty` 询问了关于使用 QLoRA 微调 GPT-NeoX 20B 的资源或信息，并提到训练期间 loss 不下降的问题。`@stellaathena` 澄清说 NeoX 库不支持 QLoRA，并建议在 GitHub 上寻求有关 `@kenakafrosty` 正在使用的 `trl`、`transformers` 和 `peft` 的帮助。

- **GPT-NeoX 的 pytest 问题**：`@catboy_slim_` 提到从 pytest 中移除了 `--forked`，并强调需要单独投入精力让 pytest 在该项目中重新正常运行。

- **Fork 运行时测试失败**：`@catboy_slim_` 报告了 Python、PyTorch 和 CUDA 的重大更新，虽然能够运行基础模型，但对无法手动验证每个可能的分支表示担忧，指出测试需要正常工作，并已在 [GitHub 上创建了 issue](https://github.com/EleutherAI/gpt-neox/issues/1132)。

- **关于 Torch 测试框架的讨论**：`@catboy_slim_` 对现有的测试框架能否充分处理 PyTorch 代码表示怀疑，因为开发者很少对这类代码进行测试。

- **项目协作者关于验证和算力访问的讨论**：`@tastybucketofrice` 正在为协作者（包括 `@337128969059172353`）安排算力访问，以便进一步测试他们对项目的更改，并向 `@catboy_slim_` 提供了算力访问权限以协助测试。

**提到的链接**：

[Tests fail when run with pytest --forked · Issue #1132 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/1132)：Bug 描述：当按照 /test/README.md 中的指令使用 pytest --forked 运行测试时，大量测试失败并报错：RuntimeError: Cannot re-initialize CUDA in forked subp...

  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199994496906166282) (59 条消息🔥🔥): 

- **LAION 数据集寻求**：用户 `@ppwwyyxx` 询问了 **laion2b-en aesthetics scores**，因为最初提供的数据集链接已失效。回复称，应数据集作者的要求，该数据集的访问权限已暂时关闭，建议查看公告以获取更新。
- **语音聊天界面 Demo 发布**：`@jpcl_` 发布了一个完整的 **voice chat interface** 新 Demo，结合了 Whisper 和 WhisperSpeech 以及开源 LLM，宣称降低了延迟以实现更自然的对话，并邀请合作以改进系统。他们分享了 [Hacker News 公告](https://news.ycombinator.com/item?id=39130140) 的链接。
- **图像字幕（Image Captioning）策略讨论**：用户 `@pseudoterminalx`、`@thejonasbrothers` 和 `@limiteinductive` 分享了使用 AI 进行图像字幕生成的方法，重点是提供清晰的 prompts 以避免 hallucination（幻觉），并专注于仅描述可见内容。
- **AI Tech Sprint 招募**：`@ninjaa2377` 正在寻找开发者加入 VA 的 AI Tech Sprint 团队，参与一个涉及临床就诊记录（clinical encounter notes）的宏大项目，提供潜在的奖金和声望。感兴趣的开发者请通过 DM 联系并访问 [官方挑战赛网站](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)。
- **盗版美国频道自由运营？**：`@pseudoterminalx` 提到当地有线电视公司在没有后果的情况下使用来自美国的盗版频道，声称其所在国家（未指明）的政府不受外国公司或贿赂的影响。

**提到的链接**：

- [未找到标题](https://news.ycombinator.com/item?id=39130140)：未找到描述
- [Reddit - 深入了解一切](https://www.reddit.com/r/DefendingAIArt/comments/19djc0a/)：未找到描述
- [laion/laion2B-en-aesthetic at main](https://huggingface.co/datasets/laion/laion2B-en-aesthetic/tree/main)：未找到描述
- [欧盟委员会 🇪🇺 在 Instagram 上："Nice try Fluffy, but indeed you got the news right! Today we presented measures to allow European AI start-ups and SMEs to train their model using our High-Performance Computing’s capacity.](https://www.instagram.com/reel/C2e0qkNqqu7/?igsh=MXc2MzB2ZGo3dXUwOQ==)：8.7 万次点赞，666 条评论 - europeancommission 于 2024 年 1 月 24 日发布："Nice try Fluffy, but indeed you got the news right! Today we presented measures to allow Europe..."
- [Challenge.Gov](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data)：Challenge.Gov 是支持由美国联邦政府赞助的奖金挑战和竞赛的官方 GSA 政府网站。在这里，联邦机构向...提供奖金。

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1200069735388287056) (71 条消息🔥🔥): 

- **Google 的变现困境**：用户 `@max_voltage` 讨论了 Google 在过去 15 年中除了广告之外，难以找到具有财务意义的商业模式，指出他们或许应该采取类似于 OpenAI 的 SaaS 模式，而不是将 DeepMind 完全整合进 Google。
- **对 Byte-Level Transformers 持谨慎乐观态度**：`@marianbasti` 引用了一篇 [arXiv 论文](https://arxiv.org/pdf/2401.13660.pdf)，对 Byte-Level Transformers 表达了谨慎的乐观，而 `@thejonasbrothers` 则幽默地指出，进展似乎总是“还有一个月就到”。
- **文本生成图像 Diffusion 和身份保持（Identity Preservation）进展**：`@vrus0188` 分享了两个描述最新进展的 GitHub 链接：[用于文本生成图像 Diffusion 的 RPG-DiffusionMaster](https://github.com/YangLing0818/RPG-DiffusionMaster) 和 [用于身份保持生成的 InstantID](https://github.com/InstantID/InstantID)，`@chad_in_the_house` 确认了其酷炫程度。
- **扩展图像修复 (SUPIR)**：`@thejonasbrothers` 链接到了一篇 [arXiv 投稿](https://arxiv.org/abs/2401.13627)，介绍了 SUPIR 及其在文本提示引导下的图像修复能力，并强调该论文出现在 Hacker News 的热门论文之列。
- **讨论 AI 模型训练的高昂成本**：用户 `@vrus0188`、`@chad_in_the_house`、`@thejonasbrothers` 和 `@limiteinductive` 参与了关于训练 Stable Diffusion 等 AI 模型巨大成本的对话，尽管他们承认随着时间和技术进步，成本预计会降低。


**提到的链接**：

- [The Architecture of a Biologically Plausible Language Organ](https://arxiv.org/abs/2306.15364)：我们展示了一个模拟的生物学上合理的语言器官，由程式化但真实的神经元、突触、大脑区域、可塑性以及简化的感官知觉模型组成。我们通过...展示了...
- [Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild](https://arxiv.org/abs/2401.13627)：我们介绍了 SUPIR (Scaling-UP Image Restoration)，这是一种突破性的图像修复方法，利用了生成先验和模型扩展（Scaling up）的力量。利用多模态技术和...
- [PALP: Prompt Aligned Personalization of Text-to-Image Models](https://arxiv.org/abs/2401.06105)：内容创作者通常旨在利用个人主体创建个性化图像，这超出了传统文本生成图像模型的能力。此外，他们可能希望生成的图像...
- [GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥](https://github.com/InstantID/InstantID)：InstantID：秒级零样本身份保持生成 🔥 - GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥
- [GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (PRG)](https://github.com/YangLing0818/RPG-DiffusionMaster)：掌握文本生成图像 Diffusion：利用多模态 LLMs 进行重标注、规划和生成 (PRG) - GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Pl...
- [Whole brain functional recordings at cellular resolution in zebrafish larvae with 3D scanning multiphoton microscopy - Scientific Reports](https://www.nature.com/articles/s41598-021-90335-y)：未找到描述

  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1200030911715090472) (87 条消息🔥🔥): 

- **Perplexity Pro 功能说明**：针对 `@winnie.zhao` 的询问，`@icelavaman` 和 `@mares1317` 提供了一个[链接](https://blog.perplexity.ai/faq/what-is-perplexity-pro)，详细介绍了 Perplexity Pro 的功能，如无限次 Copilot 查询、上传文件进行内容探索，以及访问包括 GPT-4 和 Claude 2.1 在内的强大 AI 模型。
- **数据保留问题引发关注**：以 `@emisaurus_hex` 和 `@firesonwires` 为首的多位用户对 Perplexity 的数据保留政策表示困惑和担忧。据推测为 Perplexity 专家的 `@icelavaman` 澄清道，已删除的线程会在 30 天后从服务器中移除。
- **关于使用 Perplexity 模型的问题**：`@divyanshu0500`、`@odobostudio`、`@lukas8a` 等用户提出了关于模型 JSON 输出、文件上传限制，以及 Claude 和 GPT-4 在总结 PDF 和学术著作方面的效率等技术问题。
- **了解账户和搜索数据政策**：关于 Perplexity 隐私政策的讨论凸显了一些模糊之处，促使人们建议使用更清晰的政策措辞以避免误解，并确认搜索数据是否确实在账户有效期内一直保留。
- **社区互动与技术支持**：`@danielagmz888` 和 `@icelavaman` 等成员就从申请信用代码到解决疑虑等各种问题提供了帮助。`@reflext` 和 `@sedierta` 之间还就 Pro 订阅成本和各种模型的性能进行了轻松的交流。

**提到的链接**：

- [Perplexity 会收集我的哪些数据？](https://blog.perplexity.ai/faq/what-data-does-perplexity-collect-about-me)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [什么是 Perplexity Pro？](https://blog.perplexity.ai/faq/what-is-perplexity-pro)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Perplexity - AI 助手](https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo?utm_content=chrome-link&ref=chrome&utm_medium=google&utm_source=chrome+store&utm_campaign=chrome-extension)：在浏览时随时提问
- [Perplexity 博客](https://blog.perplexity.ai/faq/how-does-file-upload-work.)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1200104475394850866) (4 条消息): 

- **搜索与 AI 的交汇**：`@jsudaniel` 强调了 CEO 与 Google Search 和 OpenAI 的联系，指出 Perplexity AI 是这些技术的交汇点。他们分享了一个名为“[我使用 Perplexity 的频率超过了 Google 和 ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)”的 YouTube 视频，讨论了使用 Perplexity AI 的好处。

- **Perplexity 降低了 Smartsheet 的学习曲线**：`@nicknalbach` 发现 Perplexity AI 为从 Excel 过渡到 Smartsheet 时遇到的问题提供了高效的解答。Perplexity 帮助他克服了陡峭的学习曲线，而其他资源提供的解决方案往往过于零散。

- **天文学教育的概念辅助**：`@coloradocomplex` 提到使用 Perplexity 来帮助解释天文学课程中的概念，展示了 Perplexity AI 在教育中的实用性。

- **无额外信息**：`@coloradocomplex` 分享了一个链接，但未提供关于该链接内容或目的的背景或额外信息。

**提到的链接**：

[我使用 Perplexity 的频率超过了 Google 和 ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)：该视频的主要看点：“我使用 Perplexity 的频率超过了 ChatGPT、BARD 和 Microsoft Copilots，主要有五个原因，包括它在内容创作中的应用……”

  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1200017158302863361) (5 messages): 

- **代码输出中的 Web 与 API 差异**：`@benhirap` 表示 Perplexity AI 的网页版生成的代码质量比 API 版本好得多。
- **寻求 API 与 Labs 的一致性**：`@stijntratsaert_01927` 询问了 Perplexity AI labs 使用的默认参数，因为他们在通过 API 复现 labs 结果时遇到了困难。
- **账单问题待解决**：`@aiagileguy` 报告了被重复收费的问题，并联系了 support@perplexity.ai 请求退还积分，但在超过 1-2 个工作日后仍未收到解决方案。
- **请求协助处理支持事宜**：针对账单问题，`@aiagileguy` 正在寻求帮助或指引，以加快 Perplexity AI 的退款流程。
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1200176286639870022) (3 messages): 

- **社区亮点 - 社交探索**：`@lunarflu` 赞扬了 **HuggingFace 社区**对 ML 内容的专注，并邀请成员加入该组织以获取 [“Posts”功能的早期访问权限](https://huggingface.co/social-post-explorers)。对于对 AI 和 ML 感兴趣的人来说，它被强调为一个比 Twitter 或 LinkedIn 噪音更小的替代方案。


**提到的链接**：

- [social-post-explorers (Social Post Explorers)](https://huggingface.co/social-post-explorers)：未找到描述
- [Cosmos Arena](https://thenameless.net/cosmos-arena)：未找到描述
- [@gsarti on Hugging Face: &quot;🔍 Today&#39;s pick in Interpretability &amp; Analysis of LMs: From Understanding to…&quot;](https://huggingface.co/posts/gsarti/888341627040205)：未找到描述
- [CheXRay - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/CheXRay)：未找到描述
- [GitHub - TonyAssi/HF-Embed-Images: Generates image embeddings for 🤗 Datasets](https://github.com/TonyAssi/HF-Embed-Images)：为 🤗 Datasets 生成图像嵌入。通过在 GitHub 上创建账号为 TonyAssi/HF-Embed-Images 的开发做出贡献。
- [@Tonic on Hugging Face: &quot;hey there folks , work in progress, but basically celebrating the release of…&quot;](https://huggingface.co/posts/Tonic/220992701457145)：未找到描述
- [not-lain/TunBERT · Hugging Face](https://huggingface.co/not-lain/TunBERT)：未找到描述
- [@mehd-io on Hugging Face: &quot;We just released the first Text2SQL model for DuckDB 🦆🧠
You can try it out…&quot;](https://huggingface.co/posts/mehd-io/779023528910338)：未找到描述
- [@Tonic on Hugging Face: &quot;👋 Hi there folks,

I launched my first competition ! 

Goal : Use AI to…&quot;](https://huggingface.co/posts/Tonic/783827682062088)：未找到描述
- [@gsarti on Hugging Face: &quot;🔍 Today&#39;s pick in Interpretability &amp; Analysis of LMs: Model Editing Can Hurt…&quot;](https://huggingface.co/posts/gsarti/256926950283134)：未找到描述
- [ClovenDoug/small_128_all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/ClovenDoug/small_128_all-MiniLM-L6-v2)：未找到描述
- [Deepfake Detection - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/deepfake-detection)：未找到描述
- [@vicgalle on Hugging Face: &quot;Can you merge models of different sizes? ⚗️

Well, yes, if the models are…&quot;](https://huggingface.co/posts/vicgalle/320544784279721)：未找到描述
- [tenyx/TenyxChat-8x7B-v1 · Hugging Face](https://huggingface.co/tenyx/TenyxChat-8x7B-v1)：未找到描述
- [AI Lineage Explorer: A Step Towards AI Integrity.](https://huggingface.co/blog/backnotprop/integrity-explorer)：未找到描述

  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199992656353304586) (40 messages🔥): 

- **解码训练困境**：`@asprtnl_50418` 强调了预训练模型所需的巨大资源规模，引用了 [Llama-2-7b model](https://huggingface.co/meta-llama/Llama-2-7b)，该模型在 A100 GPU 上需要约 18.4 万个 GPU 小时。他们还提到了其他具有成本效益的方法，如 *fine-tuning* 以及使用 [LoRA/QLoRA adapters](https://huggingface.co/docs/peft/conceptual_guides/lora) 来降低硬件需求。
  
- **训练与评估集划分策略**：用户 `@enka55` 和 `@the_aureo` 讨论了在 LLM 训练中将数据划分为训练集和评估集的挑战，建议包括使用 pandas 的 `train_test_split` 配合 `stratify` 参数，并针对训练数据未涵盖的主题补充 RAG 等知识库。

- **特征提取基础原理解析**：`@vipitis` 澄清了特征提取是指通过 BERT 等 encoder-only 模型进行序列嵌入（sequence embedding），可用于聚类等任务。他们还引导用户参考 [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 以获取相关模型和指标。

- **寻求模型评估见解**：`@green_eye` 对基准测试之外缺乏可获取的模型定性评估表示沮丧，寻求更多详细说明模型优缺点的、更具可读性的人类评论。

- **排除模型加载故障**：`@newincoding` 在加载模型时遇到困难，`@sebastian3079` 诊断这可能是由于硬件限制，建议处理 400 亿参数（40 billion parameters）的模型至少需要 32GB 的 RAM。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1tPiTnqk2tMwYLhehS9qVPkcQ9J0gTVv2?usp=sharing)：未找到描述
- [Supported models and hardware](https://huggingface.co/docs/text-embeddings-inference/supported_models)：未找到描述
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)：未找到描述
- [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b#hardware-and-software))：未找到描述

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1200218914693582849) (1 messages): 

- **寻找数据集评估框架**：用户 `@rosebei3ngan3g` 表示需要专门针对大语言模型的数据集评估框架，指出尽管有很多评估模型本身的框架，但仍缺乏此类工具。他们质疑在没有既定框架的情况下应如何进行数据集评估。
  

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1200037182090530897) (7 条消息): 

- **显微镜下的 HuggingFace Datasets**：用户 `@andysingal` 分享了一个 [GitHub 项目](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis)，专注于对 **HuggingFace** 上的数据集卡片（dataset cards）进行大规模分析。对于任何深入研究 AI 数据集文档的人来说，这个项目都非常有启发性。
- **从零到 ML 英雄**：用户 `@pacificvoltage` 正在通过阅读《Understanding Deep Learning》([udlbook](https://udlbook.github.io/udlbook/)) 的第一章来探索机器学习的基础知识，并对最近 **Machine Learning Street Talk** 采访 Noam Chomsky 时使用的 deepfake 技术感到惊叹，可以在 [YouTube](https://www.youtube.com/watch?v=axuGfh4UR9Q&t=8412s) 上观看。
- **AI 生成文本的双筒望远镜视野**：`@tea3200` 介绍了一篇 [arXiv 论文](https://arxiv.org/abs/2401.12070)，展示了 ***Binoculars***，这是一种新型检测器，声称在不需要任何训练数据或特定模型修改的情况下，区分人类与机器生成文本的准确率超过 90%。
- **SemEval2024 共享任务焦点**：用户 `@vipitis` 提到了 **SemEval2024-task8** 竞赛的一个 [GitHub 共享任务](https://github.com/mbzuai-nlp/SemEval2024-task8)，该任务专注于多领域、多模型和多语言的机器生成文本检测，可能与刚刚分享的 "Binoculars" 方法相关。
- **AI 助力 Flutter 腾飞**：`@akindelemichael` 凭借一个用于在 Flutter 应用中集成 ONNX 模型的新 [package](https://github.com/gtbluesky/onnxruntime_flutter) 引起了关注，这正好契合了 `@osanseviero` 注意到的 Flutter AI 能力增长趋势，包括一个 [用于 HuggingFace Inference API 的 Flutter SDK](https://huggingface.co/posts/shivance/676533662914249)。

**提到的链接**：

- [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070)：检测现代大语言模型生成的文本被认为很困难，因为 LLM 和人类都能表现出广泛的复杂行为。然而，我们发现一种基于对比的评分……
- [@shivance 在 Hugging Face 上表示：“嗨，社区！我非常激动地宣布用于 HuggingFace Inference 的 Flutter SDK……”](https://huggingface.co/posts/shivance/676533662914249)：未找到描述
- [GitHub - mbzuai-nlp/SemEval2024-task8: SemEval2024-task8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection](https://github.com/mbzuai-nlp/SemEval2024-task8)：SemEval2024-task8：多领域、多模型和多语言机器生成文本检测 - GitHub - mbzuai-nlp/SemEval2024-task8...
- [NOAM CHOMSKY - THE GHOST IN THE MACHINE](https://www.youtube.com/watch?v=axuGfh4UR9Q&t=8412s)：Patreon: https://www.patreon.com/mlst Discord: https://discord.gg/ESrGqhf5CB 在这个特别版节目中，我们很高兴地揭晓 Noam Chomsky 教授……
- [GitHub - YoungXinyu1802/HuggingFace-Dataset-Card-Analysis: Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on HuggingFace (ICLR 2024)](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis/tree/master)：导航 AI 中的数据集文档：HuggingFace 数据集卡片的大规模分析 (ICLR 2024) - GitHub - YoungXinyu1802/HuggingFace-Dataset-Card-Analysis...
- [GitHub - gtbluesky/onnxruntime_flutter: A flutter plugin for OnnxRuntime provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile and desktop platforms.](https://github.com/gtbluesky/onnxruntime_flutter)：一个用于 OnnxRuntime 的 Flutter 插件，提供简单、灵活且快速的 Dart API，以便在移动和桌面平台的 Flutter 应用中集成 Onnx 模型。- GitHub - gtbluesky/onnxruntime_flutter...

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1200059722879991848) (12 messages🔥): 

- **HuggingFace 上的 Nemo 项目启动平台**：`@tonic_1` 对启动 **Nemo 模型项目** 表达了热情，并提议在 HuggingFace 上撰写一篇详细的 **blog post**，这一想法得到了积极响应。`@not_lain` 表示赞同，并承诺将*尽快*撰写文章。

- **托管在 HuggingFace 上的 WhisperSpeech 演示**：`@tonic_1` 介绍了 [HuggingFace 上的 **WhisperSpeech** 演示](https://huggingface.co/spaces/Tonic/whisperspeech)，该演示支持多语言文本转语音，并能通过极短的音频输入创建声纹。

- **开发中的 CheXRay 分析工具**：`@tonic_1` 分享了 [CheXRay 的链接](https://huggingface.co/spaces/Tonic/CheXRay)，这是一个正在开发中的胸部 X 光分析工具，展示了医学影像 AI 领域的活跃项目和进展。

- **@lunarflu 发起的社区博客外联**：`@lunarflu` 联系了 `@mateomd_dev`，建议通过社区博客文章来扩大其作品的影响力，并提供了 [HuggingFace 博客板块的链接](https://huggingface.co/blog/community)。`@mateomd_dev` 表现出为 HuggingFace 受众优化其文章的兴趣。

- **即将发布的 wav2vec2-bert 模型公告**：`@yehors` 宣布即将发布基于 Common Voice 10 数据集的 **wav2vec2-bert 模型**，表明该模型已进入最后准备阶段。

**提到的链接**：

- [WhisperSpeech - Tonic 的 Hugging Face Space](https://huggingface.co/spaces/Tonic/whisperspeech)：未找到描述
- [CheXRay - Tonic 的 Hugging Face Space](https://huggingface.co/spaces/Tonic/CheXRay)：未找到描述
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers)：未找到描述
- [Hugging Face – 社区博客](https://huggingface.co/blog/community)：未找到描述

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1200020728574132275) (3 messages): 

- **对 Isamu 的鼓励**：`@lunarflu` 表达了对名为 Isamu 的用户的支持，告诉他们慢慢来，并添加了爱心表情符号。

- **文本转视频模型 Lumiere 提高了标准**：`@fishie22` 讨论了 Google 的新模型 **Lumiere**，解释了其创新的 **Space-Time UNET** 用法，该技术保持了时间一致性，并能以 16fps 的速度生成 80 帧视频。他们提供了研究论文链接：[Google 的 Lumiere 研究](https://arxiv.org/abs/2401.12945)。

- **对 Medium 文章基准测试的正面反馈**：`@starsupernova` 在推特上提到了一篇 Medium 文章，称赞其基准测试“非常棒”，并添加了笑脸表情符号。

**提到的链接**：

[Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945)：我们介绍了 Lumiere —— 一种文本转视频扩散模型，旨在合成展现真实、多样且连贯运动的视频，这是视频合成中的一个关键挑战。为此，我们……

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 

spikespiegel5112: 如何在本地加载 LoRA 模型？
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1200164215164518431) (5 messages): 

- **关于 LMM 架构的简短询问**：`besiktas` 询问了在当前训练的 LMM 中使用 **idefics/flamingo resampler/cross-attention** 的设计选择，而不是采用线性投影或预训练视觉编码器等更简单的方法。
  
- **Gemini Pro Vision AI 介绍**：`ahmed3ibrahim` 讨论了试用 Swift API 的 [Gemini Pro Vision AI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/)，提到了其关键特性，如在单个请求中处理多张图像，并提供全面的 **API 健康报告**。

- **对 CVPR2024 论文的好奇**：`iloveh8` 正在寻找查看 **CVPR2024** 所有论文（包括被拒绝和被接收的）的方法，但未收到回复。

**提到的链接**：

[Gemini Pro Vision AI API 文档 (swift-api-swift-api-default) | RapidAPI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/)：未找到描述

  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1200003453414559744) (15 messages🔥): 

- **TorToiSe 发音引擎获得认可**：`@mr_nilq` 提到 **TorToiSe** 在 TTS 领域提供了最佳质量但速度较慢，并分享了一个快 5 倍的修改版本 [链接](https://github.com/152334H/tortoise-tts-fast)。
- **寻求训练问答 AI 的建议**：用户 `@ysk.dev` 正在考虑训练包含 1 万个问答对的 AI 方案，在 Amazon Lex 和训练 VDS 之间犹豫，并询问运行长回答服务器所需的硬件规格。
- **请求解决 Transformer 导入错误**：用户 `@srovnbh` 在从 `transformers` 包导入 `TFTrainer` 时遇到错误，并收到了确保安装正确版本的建议。
- **关于信任“黑盒”模型的演讲**：`@vipitis` 分享了一个关于评估“黑盒”模型的 [演讲链接](https://talks.cam.ac.uk/talk/index/211336)，质疑在用户无法看到 API 背后运作的情况下对模型的信任度。
- **Windows 兼容性问题导致 Bits and Bytes 报错**：`@kingpoki` 意识到他们的问题原因是 bitsandbytes 缺乏对 Windows 的支持。

**提到的链接**：

[talks.cam : 复制与审计黑盒语言模型。](https://talks.cam.ac.uk/talk/index/211336)：未找到描述

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 

spikespiegel5112: 如何在本地加载 LoRA 模型？
  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1200238081865941012) (1 messages): 

- **Gradio 4.16 发布，功能强大**：`@abidlabs` 宣布发布 **`gradio 4.16`**，其主要特性包括对 **Polars Dataframe** 的原生支持、可作为输入使用的新 Gallery 组件、改进的 Chatbot 低延迟流式传输，以及自定义组件的自动文档。这个“重磅发布”的详细变更日志可以在 [这里](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md) 查看。

**提到的链接**：

[gradio/CHANGELOG.md at main · gradio-app/gradio](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md)：完全使用 Python 构建和分享令人愉悦的机器学习应用。🌟 点赞以支持我们的工作！ - gradio-app/gradio

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1200121061669343343) (1 messages): 

- **LLMCompiler 网络研讨会即将开始**：`@jerryjliu0` 提醒大家，10 分钟后将举行一场关于 **LLMCompiler** 论文作者的**网络研讨会**，该论文详细介绍了一个用于 **Agent 并行函数调用** 的框架。该框架旨在提升性能和效率，可以通过他们的论文 ([LLMCompiler Paper](https://arxiv.org/pdf/2312.04511.pdf)) 进行探索，更多资源如 **LlamaPack** 和 **Notebook** 可在专用链接获取。

**提到的链接**：

[LlamaIndex Webinar: Efficient Parallel Function Calling Agents with LLMCompiler · Zoom · Luma](https://lu.ma/lf9iroox)：LLM 非常擅长推理和采取行动。但之前的 Agent 推理框架（如 ReAct）主要集中在顺序推理，导致了更高的……

  

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1200128299385499748) (7 条消息): 

- **Slack Bot 教程分享**：IFTTT 宣布了一个新的 OSS 仓库，其中包含由 `@seldo` 编写的**分步指南**，用于构建一个可以从对话中学习并回答组织问题的 Slack Bot。该 Bot 基于 @SlackHQ 平台构建。[构建你的 Slack Bot](https://t.co/E8KJNeoXfr)。

- **Zilliz Cloud Pipeline 与 LlamaIndex 集成**：LlamaIndex 强调了他们与 `@zilliz_universe` 的合作，将 Zilliz Cloud Pipeline 集成到 LlamaIndex 中，增强了检索服务和多租户支持。查看[客座博客文章](https://t.co/luDjSgiokt)。

- **LlamaIndex 支持新的 OpenAI Embedding 模型**：LlamaIndex 团队发布了 0.9.38 版本，其中包含对 @OpenAI 最新 Embedding 模型的 **Day 0 支持**。更多详情请参阅[发布说明](https://t.co/kyIoTUaeuD)。

- **LlamaIndex 开箱即用的优质 Prompting**：IFTTT 提到了 LlamaIndex 一个经常被忽视的特性，即它默认会**创建有效的 Prompt**，如果需要也可以进行自定义。更多见解请点击[此处](https://t.co/GUJxx6TO0a)。

- **LlamaIndex 现已支持 TypeScript**：IFTTT 宣布 LlamaIndex.TS 0.1.0 版本已发布，感谢 `@yi_ding` 的快速贡献，将对 @OpenAI 最新 Embedding 的支持扩展到了 TypeScript。TypeScript 爱好者请访问 [LlamaIndex.TS 0.1.0 发布页面](https://t.co/lVVsWAXcdl)。

- **LlamaIndex.TS 发布版包含 Qdrant 引擎**：LlamaIndex.TS 的 0.1.0 版本还增加了对 `@qdrant_engine` 的支持。该更新被强调为 TypeScript 发布版中的一项额外功能。[查看此功能](https://twitter.com/llama_index/status/1750673214840394198)。

**提到的链接**：

- [llama-index](https://t.co/kyIoTUaeuD)：LLM 与你的数据之间的接口
- [使用 LlamaIndex 和 Zilliz Cloud Pipelines 构建可扩展的 RAG 应用](https://t.co/luDjSgiokt)：简介

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1200040486535176283) (38 条消息🔥): 

- **LlamaIndex Text Inference Server 暂无可用 LLM**：`@cheesyfishes` 确认 LlamaIndex 目前没有用于 TextGenerationInference 服务器的 LLM，但提到 Langchain 的模型可以通过 Wrapper 配合使用。
- **使用 `similarity_top_k` 配置 Chat Engine**：针对 `@richard1861` 的提问，`@whitefang_jr` 提供了一段 Python 代码片段，用于在 LlamaIndex 中配置 Chat Engine 的相似性检索数量，使用 `similarity_top_k=5`。
- **特定领域用例中的检索挑战**：`@lancerninja` 和 `@cheesyfishes` 讨论了一个更复杂的检索场景，涉及在执行另一次检索之前使用 LLM 重写问题，旨在提高性能，但担心多步骤会导致响应时间增加。
- **期待与新 OpenAI Embedding 模型的集成**：`@ayfri` 分享了 OpenAI 关于新 Embedding 模型和 API 更新的公告链接。`@cheesyfishes` 做出回应，暗示 LlamaIndex 即将支持这些新功能。
- **在 LlamaIndex 中自定义 Prompt 以实现上下文响应**：`@shri_j` 询问当查询信息不在提供的上下文中时，如何从 OpenAI 获取答案。`@cheesyfishes` 指导其修改默认 Prompt 以实现此类功能，并分享了文档链接。

**提到的链接**：

- [新的 Embedding 模型和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。
- [使用模式 - LlamaIndex 🦙 0.9.38](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts)：未找到描述

  

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1200109478939480164) (5 条消息): 

- **探索 Zep 的功能**：用户 `@yoursaviorjesus` 询问是否有人有使用 **Zep** 的经验，并指出了其聊天历史记忆（chat history memory）和实体提取（entity extraction）等功能。他们提供了 Zep 文档和各种快速入门指南的链接：[Zep Documentation](https://docs.getzep.com/)。

- **询问 LlamaIndex 的本质**：`@zeekg_46676` 询问 **LlamaIndex** 是一个 vector store，还是像使用自然语言搜索的 Amazon Kendra 那样运作。`@cheesyfishes` 澄清说 LlamaIndex 更类似于 Kendra，并且功能多样，能够使用任何 vector store 或语言模型进行各种操作。

- **展示自学习知识图谱**：`@chiajy` 分享了他们在自学习知识图谱 RAG 工作流方面的工作，该工作流具有递归检索（recursive retrieval）、自动创建和多跳推理（multi-hop reasoning）的特点，并以《哈利·波特》书籍演示为例。关于该知识图谱的详细解释及其影响，可以在他们的文章中找到：[Harry Potter and the Self-Learning Knowledge Graph](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning)。

**提到的链接**：

- [Zep - Fast, scalable building blocks for LLM apps](https://docs.getzep.com/)：未找到描述
- [Harry Potter and the Self-Learning Knowledge Graph RAG](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning)：WhyHow.AI 的自学习 RAG 结合知识图谱，为垂直领域 AI 带来准确性和规则——展示了递归检索、记忆以及自动化的上下文感知知识图谱构建。

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1200087852042698923) (36 messages🔥): 

- **无 LLM Paper Club 录制**：`@kbal11` 回复了 `@farrealdori`，告知 LLM Paper Club 的会议不会录制，以便参与者能更自由地分享工作细节，因此没有回放。
- **介绍 Morpheus-1**：`@shivdinho` 分享了一个推文链接，宣布推出 Morpheus-1。它被描述为世界上首个多模态生成式超声 Transformer，旨在诱导和稳定清醒梦（lucid dreams），并强调了其创新性。
- **Go-Go-Labs 编程冲刺**：`@slono` 提供了一个 GitHub 仓库链接，展示了在 4 天内为 `yaml-custom-tags` 实验编写了 5000 行代码，表明项目进展迅速，即将完成。
- **GPT-4 Turbo & Embedding Models 更新**：`@dimfeld` 分享了 OpenAI 关于发布更新版 GPT-4 Turbo 预览模型和新 Embedding Models 的公告，同时 `@swyxio` 链接了来自 Twitter 的相关笔记。
- **Martian 的 LLM 排行榜发布**：`@cute_hamster_07119` 宣布在 `https://leaderboard.withmartian.com/` 发布 Martian 的 Model Router，这是一个帮助评估各种 LLM 推理产品的工具。`@coffeebean6887` 和 `@fanahova` 讨论了该项目的文档和开源方面。

**提及的链接**：

- [🚨🚨 这里的 YAML 真多 🚨🚨](https://noyaml.com/)：未找到描述
- [来自 talrid23 (@talrid23) 的推文](https://x.com/talrid23/status/1750463847226388574?s=46&t=XV1VJkM4nCYVU6fROoKkfw)：JSON 格式正成为 LLM 输出生成的行业标准。然而，它是最优格式吗？🤔 我们认为不是——YAML 输出更短、更简单，从而带来更快的推理...
- [LLM 推理提供商排行榜](https://leaderboard.withmartian.com/)：由 Martian 制作的关于 LLM 推理 API 的实时、公正的基准测试。
- [新的 Embedding Models 和 API 更新](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding Models、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并且很快将下调 GPT-3.5 Turbo 的价格。
- [Reddit - 深入探索一切](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/)：未找到描述
- [KREA 正在构建人类创造力的下一个前沿 ⚡️](https://cerebralvalley.beehiiv.com/p/krea-building-next-frontier-human-creativity)：此外：联合创始人 Diego 谈论拥抱好奇心与混沌...
- [go-go-labs/cmd/experiments/yaml-custom-tags 在 main 分支 · go-go-golems/go-go-labs](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/experiments/yaml-custom-tags)：GO GO 实验实验室。通过在 GitHub 上创建账号为 go-go-golems/go-go-labs 的开发做出贡献。
- [来自 Prophetic (@PropheticAI) 的推文](https://x.com/propheticai/status/1750534355242418300?s=46&t=JE84TqLviekDnEt8MAT-Eg)：介绍 MORPHEUS-1。世界上首个多模态生成式超声 Transformer，旨在诱导和稳定清醒梦。2024 年春季面向 Beta 用户开放。
- [评估方法论 - 提供商排行榜](https://docs.withmartian.com/provider-leaderboard/evaluation-methodology)：未找到描述
- [可复现性 - 提供商排行榜](https://docs.withmartian.com/provider-leaderboard/reproducibility)：未找到描述
- [GitHub - withmartian/provider-dashboard: Martian 的 LLM 推理提供商排行榜的开源后端](https://github.com/withmartian/provider-dashboard)：Martian 的 LLM 推理提供商排行榜的开源后端 - GitHub - withmartian/provider-dashboard

  

---


### Latent Space ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1200014492210319413) (1 messages): 

- **LLM Paper Club 亚洲版启动**：`@ivanleomk` 宣布 **LLM Paper Club 亚洲版**正式启动，首场将讨论 "Attention Is All You Need" 论文。感兴趣的人员可以[注册以获取未来通知](https://lu.ma/llm-paper-asia)并[在此处](https://discord.gg/tPnG5qMu)参加活动。

**提及的链接**：

- [LLM Paper Club (亚洲版!) · Luma](https://lu.ma/llm-paper-asia)：更新：添加了我们将使用的 Discord Stage 链接。Latent.Space x EugeneYan.com LLM Paper Club 的亚洲时区友好版！本周我们将涵盖...
- [Discord - 与朋友和社区聊天的新方式](https://discord.gg/tPnG5qMu)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---

### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1200030161123414027) (8 messages🔥): 

- **亚洲论文俱乐部时间安排**：`@ivanleomk` 感谢大家参加今天的论文俱乐部，并提到下周的讨论可能涵盖 *Self-Rewarding Language Models*。他们对其他论文建议持开放态度，并指出如果没有志愿者，`@796917146000424970` 或他们自己将负责讲解。
- **Beta 测试反馈请求**：`@aimuggle` 对大家的参与表示感谢，并请求提供反馈以改进仍处于 `beta` 阶段的论文俱乐部。
- **关于 Self-Instruction 的澄清**：`@stealthgnome` 询问 "self-instruct" 是否是 "self-reward" 的输入，表明对讨论这些概念之间相互作用的兴趣。
- **即将到来的美国论文俱乐部日程**：`@ivanleomk` 询问了下周美国论文俱乐部的预定论文，`@eugeneyan` 提供了 [Pythia paper](https://arxiv.org/abs/2304.01373) 作为讨论主题，并列出了作者及其 arXiv 链接。
- **对 Pythia 论文信息的感谢**：`@ivanleomk` 对 `@eugeneyan` 为即将进行的 Pythia 论文讨论提供的详细信息表示感谢。

**提到的链接**：

[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)：大型语言模型 (LLMs) 在训练过程中是如何发展和演变的？这些模式如何随着模型规模的变化而改变？为了回答这些问题，我们推出了 \textit{Pythia}，一套包含 16 个模型的系列...

  

---



### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1199993210039177247) (2 messages): 

- **Mixtral 训练的 Mergekit 指南**：`@philipmay` 分享了来自 **mergekit** 作者的 [GitHub issue comment](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289)，这可能会为 **DiscoResearch mixtral training** 提供参考，文中质疑了在使用 "hidden" 或 "random" 等选项合并模型后的 finetuning 过程。
- **MoE 训练的关键辅助损失 (Auxiliary Loss)**：`@bjoernp` 承认分享的 **mergekit** 信息可能很有帮助，并强调正确处理 auxiliary loss 对于 **MoE (Mixture of Experts) training** 至关重要。

**提到的链接**：

[Mixtral branch: What option should I choose when I want to do some finetuning after the merge? · Issue #116 · cg123/mergekit](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289)："hidden" 和 "random" 的参数描述并没有准确解释当我之后想要进行 finetune 时该怎么做。在合并后进行 finetune 是否有用（可能）...

  

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1200000600281194536) (23 条消息🔥): 

- **高质量数据过滤可能并非王道**：`@bjoernp` 分享了一篇来自 arXiv 的引人入胜的论文，该论文挑战了过滤预训练数据质量的标准做法，表明“质量”过滤并不总是与模型性能的提升正相关。研究建议选择能够最大化模型在目标任务上性能的数据，从而避免人为挑选数据质量概念带来的偏见。[在此阅读摘要](https://arxiv.org/abs/2401.12926)。

- **实验 LLM 的偏好信号**：用户 `@hammadkhan` 建议进行一项涉及有监督微调 (SFT) 的实验，其中将提示词的补全内容从正面改为负面，这可能会影响语言模型的学习。

- **KTO：一种不同的模型训练方法**：`@bjoernp` 提到 Key Term Optimization (KTO) 可用于训练模型。它被比作 Direct Preference Optimization (DPO)，但使用二进制信号，将补全内容关联为理想或不理想。

- **在数据集上使用 KTO 的指南**：在详细解释中，`@hammadkhan` 概述了如何最大化 KTO loss 以提高模型生成效用，并将其与需要基于偏好的成对数据的 DPO 进行了对比。Hugging Face 的 TRL 文档和 Rafailov 等人 (2023) 的论文为 DPO Trainer 和预期的数据集格式提供了进一步的背景信息。[查看 TRL 文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)。

- **用于持续模型更新的二进制标签**：`@hammadkhan` 提到了 ContextualAI 建议的 Kahneman-Tversky Optimisation (KTO)，它使用二进制的“好”或“坏”标签进行模型更新，简化了生产环境中的标注过程。

- **OpenAI 发布 GPT4-Turbo 并降低 GPT3.5 价格**：`@rasdani` 强调了 @OfficialLoganK 关于 OpenAI 发布 GPT-4 Turbo 的公告，以及对 GPT-3.5 Turbo 的更新，包括大幅降价，以及新的 API 功能，如作用域 API Key 和嵌入维度规范。[更多详情请见 OpenAI 博客](https://openai.com/blog/new-embedding-models-and-api-updates)。

**提到的链接**：

- [DsDm: Model-Aware Dataset Selection with Datamodels](https://arxiv.org/abs/2401.12926)：在为大规模模型选择训练数据时，标准做法是过滤符合人类数据质量观念的示例。这种过滤会产生定性上干净的数据点，但...
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)：未找到描述
- [来自 Logan.GPT (@OfficialLoganK) 的推文](https://x.com/officiallogank/status/1750589278709780780?s=46&t=1jtkL4JPu-DUOdo8JC668g)：给 @OpenAIDevs 的好消息，我们正在发布：- Embedding V3 模型（small & large）- 更新的 GPT-4 Turbo 预览版 - 更新的 GPT-3.5 Turbo（下周发布 + 输入 Token 降价 50% / 25% 价格...）

---

### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1200004770333741117) (12 条消息🔥): 

- **新的德国 Jina 模型发布公告**：`@sebastian.bodza` 告知了即将发布的 **德国 Jina 模型** "jinaai/jina-embeddings-v2-base-de" 在 *hf* 上的消息。这可能对 Ranking 任务有所帮助。

- **探索使用 Mixtral 进行问题生成**：`@sebastian.bodza` 在 GitHub 上分享了 [问题生成示例](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md)，并提到在该任务中使用了 4 bit **gptq** 格式的 **Mixtral** 配合 **vllm**。

- **社区协作努力**：`@bjoernp` 对 `@sebastian.bodza` 的工作表示感兴趣并愿意提供支持，特别是在生成正样本和困难负样本（hard negative examples）方面。

- **新的 OpenAI Embedding 模型发布**：`@bjoernp` 指出了 OpenAI 发布了新的 Embedding 模型，这些模型在多语言能力上有所提升。该帖子包含一个包含更多信息的链接：[在此阅读更多内容](https://openai.com/blog/new-embedding-models-and-api-updates)。

- **使用 Genie 自动生成高质量数据**：`@bjoernp` 分享了一篇关于 Genie 方法的 [研究链接](https://arxiv.org/abs/2401.14367)，该方法用于自动创建高质量的基于内容的数据，其中可能包含有用的过滤机制。

**提到的链接**：

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)：针对基于内容生成的任务，高质量数据的缺乏被认为是推进这些任务的主要障碍。为了填补这一空白，我们提出了 Genie，一种新型的自动生成方法...
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在推出新一代 Embedding 模型、新的 GPT-4 Turbo 和审核模型、新的 API 使用管理工具，并即将降低 GPT-3.5 Turbo 的价格。
- [GitHub: Let’s build from here](https://github.com)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...
- [Embedding_Training/README.md at main · SebastianBodza/Embedding_Training](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md)：通过在 GitHub 上创建账户，为 SebastianBodza/Embedding_Training 的开发做出贡献。

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199995853339902003) (6 条消息): 

- **DiscoLM German 微调成功**：`@thomasrenkert` 报告了使用 *unsloth* 成功 **微调 DiscoLM German 7B v1** 的消息，并期待基于 **Mixtral-Instruct** 的 **DiscoLM German** 版本。
- **中古高地德语翻译数据**：在回答 `@hammadkhan` 的询问时，`@thomasrenkert` 澄清微调是在一个用于将中古高地德语翻译为现代德语的 **自定义数据集** 上进行的。
- **Bjoernp 认可 DiscoLM 更新**：`@bjoernp` 通过一条简短的确认消息赞扬了 `@thomasrenkert` 的微调成果。
- **宣布令人印象深刻的 Embeddings 效率**：`@hammadkhan` 分享了 `@Nils_Reimers` 的一条推文，内容是关于即将推出的 **embeddings**，其在 MIRACL 基准测试中仅用 256 维度就显著优于 OpenAI，有望节省 **12 倍的向量数据库成本**。

**提到的链接**：

[Nils Reimers (@Nils_Reimers) 的推文](https://x.com/nils_reimers/status/1750631888094380268?s=46&t=-TRJUfVdW8KeDqen1HJU1Q)：@OttoZastrow @jerryjliu0 是的，embeddings 是我们的重点关注领域，即将有惊人的发布。例如，OpenAI 在 MIRACL 上使用 3072 维度获得 54.3 分，而我们即将推出的 256 维度模型...

  

---

### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1200152287843197129) (2 messages): 

- **OpenAI 发布下一代 Embedding 模型**：用户 `@potrock` 分享了 [OpenAI 的公告](https://openai.com/blog/new-embedding-models-and-api-updates)，内容涉及新 Embedding 模型的发布、GPT-4 Turbo 和 Moderation 模型、API 使用管理工具，以及即将下调的 GPT-3.5 Turbo 价格。这些增强功能旨在优化开发者对 API Key 的控制，并提供 API 使用情况的洞察。
- **新功能文档已上线**：伴随公告，OpenAI 更新了其 [文档](https://platform.openai.com/docs/guides/embeddings/)，以指导用户使用新的 Embedding 模型以及更新后的 GPT 和 Moderation 模型。该文档是开发者使用这些 API 的关键资源。
- **消息发布位置有误**：`@shacrw` 注意到消息发布位置不对，建议该公告应在不同的频道分享，以便针对新更新进行集中讨论。正确的频道已通过链接指出。

**提及的链接**：

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在发布新一代 Embedding 模型、新的 GPT-4 Turbo 和 Moderation 模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---


### LLM Perf Enthusiasts AI ▷ #[announcements](https://discord.com/channels/1168579740391710851/1168760950803931136/) (1 messages): 

mat_mto: 谢谢 Jeff！非常喜欢你目前所做的一切工作。
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1200152253890301993) (16 messages🔥): 

- **OpenAI 发布新模型并降价**：`@potrock` 分享了一篇 [博客文章](https://openai.com/blog/new-embedding-models-and-api-updates)，宣布了新的 Embedding 模型、GPT-4 Turbo 和 Moderation 模型的更新、新增的 API 管理工具，以及即将推出的更低价格的 GPT-3.5 Turbo。
- **Embedding 效率的胜利**：`@potrock` 强调了新缩短版 Embedding 的优势，而 `@res6969` 表示渴望升级其系统以包含更新后的模型，并提到鉴于这些改进，转向开源 Embedding 变得不再必要。
- **OpenAI：交付功能的简单解决方案**：`@res6969` 反思了使用 OpenAI 快速实现功能的便捷性，相比之下管理独立的开源模型则较为繁琐。
- **便捷性与社区模型之间的抉择**：虽然 `@potrock` 承认了 OpenAI 解决方案的便捷性，但也指出有许多优秀的开源 Embedding 模型可供选择，并允许个人进行 Fine-tuning。
- **模型选择中的经济权衡**：`@shacrw` 和 `@michelcarroll` 讨论了使用 OpenAI 最新的、维度可缩短的大型 Embedding 模型的成本优势，提到了存储节省和具有竞争力的 API 成本，这可能会降低整体支出。

**提及的链接**：

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)：我们正在发布新一代 Embedding 模型、新的 GPT-4 Turbo 和 Moderation 模型、新的 API 使用管理工具，并且很快将降低 GPT-3.5 Turbo 的价格。

  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1200045647257153567) (12 messages🔥): 

- **欢迎来到 AI 银河系**：`@quarknova` 是来自 ENS 并在 INRIA 实习的新人，表达了在项目中使用 LangChain 的兴趣，并向社区征求建议，考虑使用 GitHub 版本而非商业版本。

- **打造 AI 个性**：`@jstansbe` 询问是否可以在不依赖外部 AI API 的情况下创建自定义 AI 个性（如 "Elon Musk AI"）。`@ksolo__` 提供了资源，指出该过程被称为 finetuning，并提供了一个[深度学习课程链接](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)。

- **致敬 LangChain 的效率**：`@johnnucleus` 赞扬了 LangChain 社区，称其能够使用 LangChain 和 Streamlit 快速创建一个具备网页搜索能力的聊天机器人，并对其效率和简洁性感到惊讶。

- **使用 LLM 生成合成数据**：`@rajib2189` 正在探索使用 Large Language Models (LLM) 合成数据来训练传统的机器学习模型，而 `@johnny2x2` 分享了他如何将 LLM 用于 RAG 生成，从而根据上下文和 schema 生成 SQL 查询。

- **在 LangChain 中处理 PARQUET**：`@benjaminbascary` 寻求在 LangChain 中操作 PARQUET 文件的帮助，随后 `@johnny2x2` 提供了一个代码片段，展示了如何使用 `pandas` 加载并利用 LangChain 的 `DataFrameLoader` 将 PARQUET 文件作为文档源导入和使用。

**提到的链接**：

[Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)：未找到描述

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1200090590503780512) (3 messages): 

- **推广 LangServe Agent 示例**：用户 `@veryboldbagel` 分享了 **LangServe** agent 示例的链接，包括一个未列在 [LangServe 主示例部分](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples)的示例，以及一个针对 [可配置 agent 执行器](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor) 的特定示例。
- **使用 LCEL 构建自定义 Agent**：`@veryboldbagel` 澄清说，对于定义自定义工具，现成的 **OpenAI tools agent** 就足够了，并进一步指导了如何使用 **LCEL** 构建自定义 agent，推荐使用 [LangGraph](https://python.langchain.com/docs/langgraph#agentexecutor) 来定义具有更强表达能力的自定义 agent 运行时。
- **LangServe 中的流式响应问题**：`@hiranga.g` 报告了在使用 [agent_with_history](https://github.com/langchain-ai/langserve/blob/main/examples/agent_with_history/server.py) 示例并尝试使用 **langchain.js** 的 `RemoteRunnable` 时，无法接收到 **stream response** 的问题；还提到了在 LangServe 中使用 Agent 时的 bug，建议 `chain.streamLog()` 可能是一种变通方法，但并未达到预期效果。

**提到的链接**：

- [🦜🕸️LangGraph | 🦜️🔗 Langchain](https://python.langchain.com/docs/langgraph#agentexecutor.)：⚡ 以图的形式构建语言 agent ⚡
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples)：LangServe 🦜️🏓。通过在 GitHub 上创建账号为 langchain-ai/langserve 的开发做出贡献。
- [langserve/examples/configurable_agent_executor at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor)：LangServe 🦜️🏓。通过在 GitHub 上创建账号为 langchain-ai/langserve 的开发做出贡献。

  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1200076122575085578) (2 messages): 

- **探索 SQL chain 的局限性**：`@johnny2x2` 分享了使用 **LangChain** 处理一家制造公司订单延迟的 **SQL 查询** 的见解。他们发现 **SQL Chain** 在处理大型数据库时表现不佳，但在数据库中创建具有描述性名称的**精选视图 (curated views)** 可以提高性能。
- **优化带来更好的查询管理**：通过在自定义多向量检索器中嵌入返回查询的问题，`@johnny2x2` 最初发现本地 AI 运行示例过于频繁——通过使用 OpenAI 处理 SQL 查询，同时使用本地 LLM 保持**数据隐私**，这一挑战得到了缓解。
- **通过面向工具的查询增强链式工作流**：`@johnny2x2` 现在放弃了使用本地 AI 进行 SQL 代码生成，转而采用一种新策略，即每个查询在他们的**任务处理链**中充当一个工具，从而在涉及生成任务、使用 SQL 工具处理任务以及评估任务循环信息的流程中获得了更好的结果。
  

---



### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1200325637450235934) (3 messages): 

- **即将发布的 LLM 版本升级预览**：`@simonw` 宣布计划发布 LLM 的更新，其中涉及对 **openai 库** 的重大升级。邀请测试人员参与，详情见 [GitHub 评论](https://github.com/simonw/llm/issues/325#issuecomment-1911533536)。

- **期待 0.13 里程碑**：关于即将发布的 LLM 版本（标记为 **0.13 Milestone**）的更多信息，可以在专门的 [GitHub 里程碑页面](https://github.com/simonw/llm/milestone/8) 找到。

- **请求解决 Readline 问题**：`@simonw` 正在寻求帮助解决 LLM 中的 readline 问题，即方向键产生 ANSI 编码而不是光标导航，如该 [GitHub issue](https://github.com/simonw/llm/issues/376) 所述。

**提到的链接**：

- [0.13 Milestone · simonw/llm](https://github.com/simonw/llm/milestone/8)：从命令行访问大语言模型 - 0.13 Milestone · simonw/llm
- [llm chat - readline problems still present · Issue #376 · simonw/llm](https://github.com/simonw/llm/issues/376)：当我打开 llm chat 时，我希望使用左右方向键可以移动光标，但屏幕上却打印出了讨厌的 ANSI 编码。$ llm chat Chatting with gpt-4 Type &#39;exit...
- [Upgrade for compatibility with OpenAI 1.0 library · Issue #325 · simonw/llm](https://github.com/simonw/llm/issues/325#issuecomment-1911533536)：当前：成功安装 openai-1.0.1 $ llm -m gpt-4-turbo &#39;hi&#39; 错误：module &#39;openai&#39; has no attribute &#39;ChatCompletion&#39;

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=wlPxEq_Mtkc
  

---


### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 

arielnlee: 有人在研究 bakklava-2 吗？！