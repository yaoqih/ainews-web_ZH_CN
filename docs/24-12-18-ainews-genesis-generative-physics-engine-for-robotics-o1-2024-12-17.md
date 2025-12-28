---
companies:
- openai
- google
- carnegie-mellon-university
date: '2024-12-19T04:48:33.221430Z'
description: '**Genesis** 是由**卡内基梅隆大学（CMU）博士生周仙（Zhou Xian）**领导的大规模协作团队最新发布的**通用物理引擎**。它集成了多种最先进的物理求解器，可模拟各种材料和物理现象，并针对机器人应用提供了轻量化、超快速模拟、照片级渲染以及生成式数据能力等特性。该引擎现已开源，旨在为机器人仿真提供支持，其用途远超单纯的视频生成。


  此外，**OpenAI** 已向 API 开放了 **o1** 模型，支持函数调用和视觉能力等高级功能，在数学和编程性能上表现强劲。**谷歌**也预告了 **Gemini
  2.0 Pro** 的更新，正加速面向高级用户的部署。'
id: a782d961-2f3c-41fc-a476-d01484fea774
models:
- o1
- gemini-2.0-pro
original_slug: ainews-genesis-generative-physics-engine-for-6175
people:
- zhou-xian
- aidan_mclau
- sundar-pichai
title: Genesis：面向机器人技术的生成式物理引擎 (o1-2024-12-17)
topics:
- universal-physics-engine
- robotics-simulation
- physics-simulation
- photo-realistic-rendering
- generative-data
- simulation-platform
- open-source
- function-calling
- vision
- performance-benchmarks
- sdk
- realtime-api
---

<!-- buttondown-editor-mode: plaintext -->**一个通用的物理引擎就是你所需要的一切。**

> 2024/12/17-2024/12/18 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord (包含 **215** 个频道和 **4542** 条消息)。预计节省阅读时间（以 200wpm 计算）：**497 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

> 您正在阅读由 o1-2024-12-17 生成的 AINews。按照新前沿模型发布日的传统，我们会尝试发布多个版本进行 A/B 测试/自我评估。请查看我们的存档以获取 o1-mini 版本。对于昨天的重复发送（平台漏洞）我们深表歉意，但今天的发送是故意的。

十二月显然已成为 [生成式视频世界模拟器](https://www.latent.space/p/icml-2024-video-robots) 之月，Sora Turbo 正式商用 (GA)，Google 也预告了 [Genie 2](https://news.ycombinator.com/item?id=42317903) 和 [Veo 2](https://deepmind.google/technologies/veo/veo-2/)。现在，由 [CMU 博士生 Zhou Xian 领导](https://x.com/zhou_xian_/status/1869511650782658846?s=46) 的学术团队宣布了 [**Genesis: A Generative and Universal Physics Engine for Robotics and Beyond**](https://genesis-embodied-ai.github.io)，这是一项涉及 20 多个实验室、为期 2 年的大规模研究合作，首秀展示了一滴水从喜力啤酒瓶上滚落的画面：


![image.png](https://assets.buttondown.email/images/fe9a08d1-1640-4c7c-9c66-9d93b0318ac9.png?w=960&fit=max)


因为它是一个物理引擎，它可以从不同的摄像机角度渲染同一个引擎：


![image.png](https://assets.buttondown.email/images/1ce99dbf-f6f4-4b36-a298-7a8751240348.png?w=960&fit=max)
 

以及暴露驱动向量：


![image.png](https://assets.buttondown.email/images/21a717f8-4bce-499e-9bcd-a5fa982652ca.png?w=960&fit=max)


这个“统一物理引擎”集成了各种 SOTA 物理求解器（MPM, SPH, FEM, Rigid Body, PBD 等），支持模拟广泛的材料：刚体 (rigid body)、关节体 (articulated body)、布料 (Cloth)、液体 (Liquid)、烟雾 (Smoke)、可变形体 (Deformables)、薄壳材料 (Thin-shell materials)、弹性/塑性体 (Elastic/Plastic Body)、机器人肌肉 (Robot Muscles) 等。

渲染一致的对象在今天立即就能发挥作用，但这听起来不像大实验室所采取的“纯粹主义”苦涩教训 (bitter pilled) 方法——它是手动组合的一堆物理求解器，而不是通过数据进行机器学习得到的——但它的优势在于[开源且现成可用](https://github.com/Genesis-Embodied-AI/Genesis)（目前还没有论文）。

如果目的是视频生成，这已经令人印象深刻，但真正的目标是机器人技术。Genesis 实际上是一个包含 4 个方面的平台：

1. 一个从头开始重建的**通用物理引擎**，能够模拟广泛的材料和物理现象。
2. 一个**轻量级**、**超快速**、**Pythonic** 且**用户友好**的机器人模拟平台。
3. 一个强大且快速的**照片级真实感渲染系统**。
4. 一个**生成式数据引擎**，可将用户提示的自然语言描述转化为各种模态的数据。

而机器人应用才是它真正大放异彩的地方。


![image.png](https://assets.buttondown.email/images/fdd20ce3-444f-4f15-a891-c6060ddd7802.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/b1bbaecb-b9f0-4f63-9501-dc4a1b5f2a76.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要由 Claude 3.5 Sonnet 生成，取 4 次运行中的最佳结果。

以下是按主题分类的关键讨论：

**OpenAI o1 API 发布与特性**

- **o1 模型已发布至 API**，支持 [function calling, structured outputs, vision support, 以及 developer messages](https://twitter.com/OpenAIDevs/status/1869156065788715409)。该模型使用的**推理 token 比 o1-preview 减少了 60%**，并包含一个新的 "reasoning_effort" 参数。

- **性能基准测试**：[@aidan_mclau 指出](https://twitter.com/aidan_mclau/status/1869068880326635645) o1 **“在数学/代码方面强得离谱”，但“在其他方面表现平平”**。[基准测试结果显示](https://twitter.com/scaling01/status/1869083247554220353) o1 在 **LiveBench Coding 上得分为 0.76**，而 Sonnet 3.5 为 0.67。

- **新 SDK**：发布了 [Go 和 Java 的 beta 版 SDK](https://twitter.com/OpenAIDevs/status/1869140165798821942)。同时为 realtime API 添加了 **WebRTC 支持**，且**价格降低了 60%**。

**Google Gemini 更新**

- [@sundarpichai 确认](https://twitter.com/scaling01/status/1869072489881747818) **Gemini Exp 1206 即为 Gemini 2.0 Pro**，在代码、数学和推理任务上表现出更强的性能。

- 为了响应用户反馈，[Gemini 2.0 的部署已加速](https://twitter.com/asadovsky/status/1869114982971093316)，现已面向 Advanced 用户开放。

**模型开发与架构**

- 关于模型大小和训练的讨论——[关于 o1-preview 的尺寸是否与 o1 匹配](https://twitter.com/aidan_mclau/status/1869080913860251911)以及其与 GPT-4o 关系的辩论。

- Meta 的新研究：[直接在原始字节（raw bytes）上训练 Transformer](https://twitter.com/LiorOnAI/status/1869409580192555015)，使用基于熵（entropy）的动态分块（dynamic patching）。

**行业与商业**

- [@adcock_brett 报告](https://twitter.com/adcock_brett/status/1869235067580764635)商用人形机器人已在客户现场成功部署，并实现了从总部到现场的快速迁移。

- [宣布推出新的 LlamaReport 工具](https://twitter.com/llama_index/status/1869094544169677138)，用于使用 LLM 将文档数据库转换为人类可读的报告。

**梗与幽默**

- [关于在 IMAX 观看《Attention Is All You Need》重映版的笑话](https://twitter.com/jxmnop/status/1869154293888258139)

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Hugging Face 的 3B Llama 模型：通过搜索超越 70B 模型**

- **[Hugging Face 研究人员通过搜索让 3B Llama 表现超越 70B](https://i.redd.it/kksacsh1sk7e1.png)** ([得分: 668, 评论: 123](https://reddit.com/r/LocalLLaMA/comments/1hgybhg/hugging_face_researchers_got_3b_llama_to/))：**Hugging Face** 的研究人员取得了一项突破，通过搜索技术使 **3B Llama 模型**在 MATH-500 准确率上超越了 **70B Llama 模型**。图表显示，在特定条件下，**3B 模型**在每个问题的生成准确率上超过了 **70B 模型**，突显了该模型与大型模型相比潜在的效率和有效性。
  - **推理时间与模型大小优化**：用户讨论了在推理时间和模型大小之间寻找最佳平衡的潜力，认为如果小型模型在特定任务上表现足够好，它们会更高效，特别是当知识已嵌入 prompt 或针对特定领域进行了微调时。
  - **可复现性与数据集引用**：由于 **Diverse Verifier Tree Search (DVTS)** 模型尚未公开发布，人们对结果的可复现性表示担忧。文中提供了所用数据集的链接 ([Hugging Face Dataset](https://huggingface.co/datasets/edbeeching/dvts_3b)) 以及 DVTS 的实现代码 ([GitHub](https://github.com/huggingface/search-and-learn/blob/main/src/sal/search/diverse_verifier_tree_search.py))。
  - **领域特定限制**：由于缺乏在其他领域训练的 **PRMs** 以及带有逐步标注的数据集，人们对该方法在数学和代码领域之外的适用性持怀疑态度，质疑该通用的普适性。


**主题 2. Moonshine Web：比 Whisper 更快、更准确**

- **[Moonshine Web: 实时浏览器内语音识别，比 Whisper 更快、更准确](https://v.redd.it/gqh3gg170n7e1)** ([Score: 193, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hh5y87/moonshine_web_realtime_inbrowser_speech/)): **Moonshine Web** 声称提供比 **Whisper** 更快且更准确的**实时浏览器内语音识别**。
  - **Moonshine Web** 在 **MIT license** 下开源，目前正致力于将其集成到 **transformers** 中，详见[此 PR](https://github.com/huggingface/transformers/pull/34784)。**ONNX models** 已在 [Hugging Face Hub](https://huggingface.co/models?library=transformers.js&other=moonshine&sort=trending) 上线，尽管人们对 **ONNX web runtime** 的不透明性存在担忧。
  - 讨论要点包括对 Moonshine 相比 **Whisper** 模型（特别是 **v3 large**）在**实时能力**和准确性声明的怀疑。用户对其执行**说话人日志 (speaker diarization)** 的能力以及目前仅限于**英语**的局限性感到好奇。
  - **Moonshine** 针对实时、设备端应用进行了优化，**Transformers.js v3.2** 已添加支持。[演示源代码](https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web)和[在线演示](https://huggingface.co/spaces/webml-community/moonshine-web)可供测试和探索。


**Theme 3. Granite 3.1 Language Models: 128k Context & Open License**

- **[Granite 3.1 语言模型：128k 上下文长度与 Apache 2.0](https://huggingface.co/collections/ibm-granite/granite-31-language-models-6751dbbf2f3389bec5c6f02d)** ([Score: 144, Comments: 22](https://reddit.com/r/LocalLLaMA/comments/1hh403g/granite_31_language_models_128k_context_length/)): **Granite 3.1 语言模型**现在具备 **128k context length**，并采用 **Apache 2.0 license**，这标志着在处理更大数据集和开发者可访问性方面的重大进展。
  - **Granite 模型性能**：据报告，**Granite 3.1 3B MoE 模型**在 Open LLM Leaderboard 上的平均得分高于 **Falcon 3 1B**，这反驳了 MoE 模型性能与具有等效激活参数的稠密模型相似的说法。尽管其**激活参数比竞争对手少 20%**，但表现依然出色。
  - **模型规格与许可**：**Granite 稠密模型**（2B 和 8B）和 **MoE 模型**（1B 和 3B）分别在超过 **12 万亿**和 **10 万亿** **tokens** 上进行了训练，其中稠密模型支持基于工具的使用场景，而 MoE 模型专为低延迟应用设计。这些模型均以 **Apache 2.0 license** 发布，其中 8B 模型在代码生成和翻译任务中的表现备受关注。
  - **社区见解与对比**：**Granite Code 模型**因其被低估的性能而受到赞誉，特别是 **Granite 8BCode** 模型，可与 **Qwen2.5 Coder 7B** 竞争。讨论还强调了 MoE 模型促进各种检索策略的潜力，以及像 Red Hat 集成 Granite 模型这样熟悉的企业级解决方案的重要性。


**Theme 4. Moxin LLM 7B: A Fully Open-Source AI Model**

- **Moxin LLM 7B: 一款完全开源的 LLM - Base 和 Chat + GGUF** ([Score: 131, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1hh067r/moxin_llm_7b_a_fully_opensource_llm_base_and_chat/)): **Moxin LLM 7B** 是一款完全开源的大语言模型，在来自 **SlimPajama**、**DCLM-BASELINE** 和 **the-stack-dedup** 的文本和代码数据上进行了训练，实现了优于其他 7B 模型的零样本性能。它具有 32k 上下文大小，支持通过 Grouped-query attention、Sliding window attention 和 Rolling Buffer Cache 进行长上下文处理，所有开发资源均可在 [GitHub](https://github.com/moxin-org/Moxin-LLM) 和 [Hugging Face](https://huggingface.co/moxin-org/moxin-chat-7b) 上获取。
  - **Moxin LLM 7B** 被赞誉为模型训练的极佳资源，正如 **Stepfunction** 所指出的，它拥有简洁且易于获取的代码和数据集。该模型全面的开发资源被视为一项显著优势。
  - **TheActualStudy** 称赞该模型集成了 **Qwen 级别的上下文**、**Gemma 级别的技术**以及 **Mistral-7B-v0.1** 的性能。这种先进方法和数据的结合令人印象深刻。
  - **Many_SuchCases** 提到探索了 GitHub 仓库，并注意到缺少一些组件（如中间检查点），暗示这些可能会在稍后上传。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Imagen v2 质量提升图像生成基准**

- **[新的 Imagen v2 简直疯狂](https://www.reddit.com/gallery/1hh5swo)** ([Score: 680, Comments: 119](https://reddit.com/r/OpenAI/comments/1hh5swo/new_imagen_v2_is_insane/)): **Imagen 3** 正在其发布的版本（被称为 **Imagen v2**）中树立 **image quality** 的新标杆。该帖子强调了这项技术的惊人进步，但未提供额外的背景或细节。
  - **访问与使用**：用户讨论了通过 **Google Labs** 网站访问 **Imagen 3** 的方法，并建议在受限地区使用 **VPNs**。有提到在 [labs.google/fx/tools/image-fx](https://labs.google/fx/tools/image-fx) 上可以免费访问，但有一定的每日使用配额。
  - **艺术方面的担忧**：艺术家们对 **Imagen 3** 对艺术行业的影响表示极大担忧，担心对人类艺术家的需求会减少，以及传统艺术会被 AI 生成的图像所掩盖。一些用户认为，这种转变可能会导致创意领域的私有化和艺术劳动的侵蚀。
  - **模型混淆与改进**：关于 **Imagen 3** 的命名和版本存在一些混淆，用户澄清其为 **Imagen3 v2**。用户注意到图像质量有了显著提高，早期测试人员对结果表示满意，认为优于之前的版本。


**主题 2. NotebookLM 的对话式播客革命**

- **OpenAI 应该开发自己的 NotebookLM 应用，这太令人震撼了！** ([Score: 299, Comments: 75](https://reddit.com/r/OpenAI/comments/1hgwvwt/openai_should_make_their_own_notebooklm/)): **NotebookLM** 生成的 AI 播客听起来非常自然，在对话质量上甚至超越了 **Huberman** 的播客。该帖子建议 **OpenAI** 应该开发类似的应用，因为这可能会对该领域产生重大影响。
  - **NotebookLM 的语音质量** 受到称赞，但与人类主持人相比仍被认为不够自然，而 **Gemini 2.0** 提供了与播客主持人实时聊天的功能，增强了其吸引力。用户注意到不同平台之间的功能集成问题，强调了在使用高级语音模式和自定义项目方面的限制。
  - **对话式 AI** 在总结 PDF 等任务中的价值引发了讨论，一些人认为这在节省时间和成人学习理论方面具有革命性，而另一些人则认为内容浅薄、缺乏深度。**Gemini** 模型因其巨大的 **context window** 而受到关注，使其非常适合处理海量信息。
  - **Google 的硬件优势** 被反复强调，他们在基础设施和能源解决方案上的投资使其能够提供比 **OpenAI** 更具成本效益的 AI 模型。这使得 Google 有可能在播客 AI 领域超越 OpenAI，利用其硬件能力大幅降低成本。


**主题 3. Gemini 2.0 在学术写作方面超越其他模型**

- **Gemini 2.0 Advanced 在学术写作方面表现极其出色。** ([Score: 166, Comments: 39](https://reddit.com/r/OpenAI/comments/1hgva9g/gemini_20_advanced_is_insanely_good_for_academic/)): **Gemini 2.0 Advanced** 在学术写作方面表现优异，与包括 **ChatGPT** 在内的其他模型相比，提供了更出色的理解力、结构和风格。作者考虑在 **OpenAI** 发布改进版本之前一直使用 Gemini 2.0。
  - **Gemini 2.0 Advanced** 在 [AI Studio](https://aistudio.google.com/) 上被识别为 **Gemini Experimental 1206**，目前无需付费版本即可使用，尽管用户需要以数据交换访问权限。**Google** 的命名惯例以及缺乏统一的 AI 服务给用户带来了一些困惑。
  - **Gemini 2.0 Advanced** 在学术写作质量上展示了显著的进步，在评估中优于 **GPT-4o** 和 **Claude**。它提供详细的反馈，经常以幽默的方式批评回复，用户认为这既有效又有趣。
  - 用户讨论了通过订阅获取 **Gemini 2.0 Advanced** 的情况，对于它在 **Gemini web app** 中被列为 "2.0 Experimental Advanced, Preview gemini-exp-1206" 存在一些困惑。该模型在学术背景下的表现受到称赞，用户希望这将推动 **OpenAI** 解决 **ChatGPT** 中的问题。


**主题 4. Veo 2 通过逼真的视频生成挑战 Sora**

- **[Google 正在通过其视频生成模型的最新版本 Veo 2 挑战 OpenAI 的 Sora，据称该版本能生成更具真实感的视频。](https://v.redd.it/qok7o7rhsl7e1)** ([Score: 124, Comments: 34](https://reddit.com/r/OpenAI/comments/1hh0vwu/google_is_challenging_openals_sora_with_the/)): **Google** 正在通过发布 **Veo 2** 与 **OpenAI** 的 **Sora** 展开竞争，这是其视频生成模型的新版本，声称可以生成更真实的视频。
  - **Veo 2 的可用性与性能**：几位评论者指出 **Veo 2** 仍处于早期测试阶段，尚未广泛可用，这与宣称的发布情况形成对比。尽管如此，**Twitter** 等平台上的部分测试者报告了令人印象深刻的结果，特别是在物理特性和一致性等领域，表现优于 **Sora**。
  - **市场策略与可访问性**：有人怀疑这次发布是针对 **OpenAI** 的一种营销策略。对于 **Veo 2** 和 **Sora** 缺乏公众访问权限和 API 可用性的担忧十分普遍，不过已确认 **January** 将在 **aistudio** 上发布。
  - **对视频真实性的信任**：讨论涉及了由于 **Veo 2** 等先进生成模型的出现，可能导致对视频真实性信任度的削弱。一些人提出了解决方案，例如通过个人 AI 利用区块链注册表来验证媒体真实性，以解决这一问题。


---

# AI Discord Recap

> 由 o1-2024-12-17 生成的摘要之摘要

**主题 1. AI 扩展与项目中的挑战**  

- **Codeium 扩展在 VSCode 中短暂失效**：该扩展仅在瞬间显示自动补全建议，导致无法使用。根据多名用户的报告，回退到 1.24.8 版本可恢复正常功能。  
- **Windsurf 在高负载下性能崩溃**：部分用户遇到了超过 10 分钟的加载时间，以及偶尔出现的“代码消失”或 Cascade 功能损坏。在稳定修复方案发布前，提交支持工单是首选建议。  
- **Bolt 用户抱怨 Token 浪费**：在收到消耗额度却无关痛痒的回复后，用户开玩笑地提议增加一个*“拳打 AI”*按钮。许多人呼吁在即将发布的版本中改进记忆控制功能。

**主题 2. 新模型与升级模型**  

- [**OpenAI o1 凭借 Function Calling 大放异彩**](https://openrouter.ai/openai/o1-preview)：作为 o1-preview 的继任者，它引入了一个新的 *“reasoning_effort”* 参数，用于控制回复前的思考时间。通过 [OpenRouter](https://openrouter.ai) 使用时，其延迟明显降低。  
- [**EVA Llama 成为叙事专家**](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)：该模型针对 Roleplay 和叙事任务，据报道在多步骤故事讲述方面表现出色。早期采用者称赞其创意输出和用户友好的设计。  
- **热门模型大幅降价**：MythoMax 13B 降价 12.5%，QwQ 推理模型降价 55%。这些折扣旨在扩大社区实验的准入门槛。

**主题 3. GPU 与推理陷阱**  

- **AMD 驱动更新导致性能骤降**：用户发现从驱动版本 24.10.1 升级到 24.12.1 后，每秒 Token 数（tps）从 90+ 暴跌至 20 左右。回滚驱动可解决减速问题，这再次提醒用户对新发布的 GPU 驱动保持谨慎。  
- **Ubuntu 上的 Stable Diffusion 遇到障碍**：像 ComfyUI 或 Forge UI 这样的工具通常需要深入的 Linux 知识来解决兼容性问题。许多人仍推荐将拥有 16GB VRAM 的 NVIDIA 3060 作为更顺畅的入门基准。  
- [**TinyGrad, Torch 与 CUDA 显存困惑**](https://discuss.pytorch.org/t/reduce-time-to-first-kernel-when-using-cuda-graphs/214310)：移除如 *IsDense(y) && IsSame(x, y)* 之类的检查解决了意外的推理失败，但引入了新的复杂性。这促使开发者参考官方 CUDA Graphs 讨论以寻求潜在解决方案。

**主题 4. 高级微调与 RAG 技术**  

- [**使用 4-bit 转换微调 Llama 3.2**](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)：许多人依赖 *load_in_4bit=true* 来平衡 VRAM 占用和模型精度。通过偏精度设置，Checkpoint 可以重复使用，并最大限度地减少资源限制。  
- [**Depth AI 大规模索引代码库**](https://www.trydepth.ai/)：它在回答技术查询时达到了 *99% 的准确率*，尽管索引 18 万个 Token 可能需要 40 分钟。虽然存在 [LightRAG](https://github.com/HKUDS/LightRAG) 等竞争方案，但 Depth AI 因设置更简单而受到称赞。  
- [**Gemini 2.0 新增 Google Search Grounding**](https://github.com/BerriAI/litellm/pull/7257)：新配置允许实时网页查询以优化回答。早期评论强调了其在编程和问答场景中事实精准度的提升。

**主题 5. NotebookLM 与 Agentic 工作流**  

- **NotebookLM 改版其 3 面板 UI**：此次更新因使用率低移除了*“建议操作”*，但开发者承诺将重新引入设计更佳的类似功能。计划包括根据用户反馈增强*“引用”*和*“回答准确性”*。  
- **多语言提示词引发广泛参与**：用户尝试了巴西葡萄牙语和孟加拉语查询，发现明确*告知* NotebookLM 语言语境会使交互更流畅。这展示了其包容性全球通信的能力。  
- **控制播客长度依然困难**：即使在提示词中指定了时间，最终输出往往仍会超出或忽略限制。大多数人依靠灵活的长度范围来在深度覆盖和听众参与度之间取得平衡。


---

# PART 1: High level Discord summaries

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 难题仍在继续**：用户反馈了 **Codeium** 扩展的问题，例如自动补全消失以及取消订阅后仍产生意外费用，并引导用户前往 [codeium.com/support](https://codeium.com/support) 寻求支持。
  
  - **Elementor** 的粉丝想知道 Codeium 是否可以生成 JSON 或 CSS 代码，而 flex credits 关于使用和结转的问题也一直存在。
- **Windsurf 问题加剧**：许多人报告 **Windsurf** 导致他们的笔记本电脑卡顿十分钟或更长时间，反复出现的错误动摇了用户对最新版本的信心。
  
  - 一些人考虑降级，同时参考了诸如 [Windsurf Focus Follows Mouse](https://codeium.canny.io/feature-requests/p/windsurf-focus-follows-mouse-as-a-configuration-option) 之类的功能请求，以解决性能问题。
- **Codeium 与 Copilot 之争**：辩论集中在随着 **Copilot** 开放免费层级，**Codeium** 是否仍具有优势，并猜测基于 GPT 的服务是否正面临容量问题。
  
  - 支持者坚持认为 Codeium 的自动补全仍然很强大，并暗示 Copilot 的免费层级可能会引发对 **Claude** 和 GPT 广泛使用的担忧。
- **Cascade 的自动批准困扰**：一些开发者批评 **Cascade** 自动批准代码更改，阻碍了对关键合并的彻底审查。
  
  - 讨论集中在改进审查工作流上，人们推动更好的输出检查以避免未经审核的合并。
- **Llama 模型与免费 AI 工具讨论**：**Llama 3.3** 和 **4o-mini** 的基准测试引起了兴趣，有说法称较小的变体可以与较大的模型并驾齐驱。
  
  - [LiveBench](https://livebench.ai/#/?Coding=a) 和 [Ray](https://www.ray.io) 等资源也相继出现，促进了对编程项目免费工具链的探索。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 0.44.2 修复 Bug**：在从 0.44 版本回滚后，更新后的 **Cursor v0.44.2** 问世，解决了 [changelog](https://www.cursor.com/changelog) 中提到的 composer 异常和终端文本问题。
  
  - 成员们描述了 devcontainers 更好的稳定性，并参考了[一篇具有警示意义的论坛帖子](https://forum.cursor.com/t/warning-cursor-v0-44-breaks-all-devcontainers-v0-394-0/35747/7)，该帖子最初标记了 0.44 版本中的中断问题。
- **Kepler 浏览器强调隐私**：一款名为 **Kepler** 的 Python 编写浏览器承诺极低的服务端依赖和用户控制，在 [社区仓库](https://github.com/TheGalaxyStars/KEPLER-COMMUNITY) 中备受关注。
  
  - 它实现了随机化的 user agents，并邀请开源贡献以增强安全性。
- **UV 工具简化 Python 管理**：社区讨论揭示了用于 Python 环境管理的 **UV** 工具，具有如 [其文档](https://docs.astral.sh/uv/) 所示的强大版本处理能力。
  
  - 它简化了项目依赖关系，并与 [Poetry](https://python-poetry.org/) 和 [Python Environment Manager 扩展](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager) 等其他资源集成。
- **O1 Pro 大放异彩**：**O1 Pro** 功能在 20 次尝试内解决了一个困扰用户的持久 Bug，展示了在复杂场景下性能的提升。
  
  - 然而，聊天输出格式化方面出现了连贯性问题，表明仍需进一步改进。
- **Galileo API 访问停滞**：关于 **Galileo API** 以及 Cursor 内 **Gemini 2.0** 等模型的咨询显示可用性有限，引发了开发者群体的关注。
  
  - 他们正在寻求官方的集成时间表，特别是在 [Cursor 平台](https://www.cursor.com/) 内。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 冲击与 Sonnet 的平局**：OpenAI 向 Tier 5 密钥持有者推出了 **O1 API**，但部分用户反映无法访问，这引发了对其推理能力的期待与挫败感。
  
  - 多位用户展示了 **O1** 在测试中获得 **84.2** 分，与 **Sonnet** 持平，并就 **O1 Pro** 与标准版之间的价格和性能差异展开了讨论。
- **AI 模型间的竞争**：成员们将 **Google** 的 **Veo 2** 与 **Sonnet** 等现有竞争对手进行了比较，评估其输出质量和在编程任务中的实用性。
  
  - 他们还对不断上涨的订阅费用和多个模型方案的可持续性表示担忧。
- **支持与退款全凭运气**：社区成员反映退款时间不一，有人等待了 **4 个月**，而有人在几小时内就获得了退款。
  
  - 这导致了对客户支持整体响应速度的怀疑。
- **Gemini 作为编辑器获得认可**：爱好者们强调 **gemini/gemini-exp-1206** 的编码错误极少，且在实际场景中与 Aider 有很强的协同作用。
  
  - 尽管如此，他们承认 **Gemini** 与 **O1** 相比仍有局限性，建议针对高级用例进行进一步测试。
- **用于代码库洞察的 Depth AI 和 LightRAG**：参与者称赞 [Depth AI](https://www.trydepth.ai/) 的代码索引准确率接近 **99%**，且资源占用适中。
  
  - 虽然 [LightRAG](https://github.com/HKUDS/LightRAG) 被提及作为替代方案，但一些人观察到 Depth AI 存在“无输出”的故障，并对其一致性提出质疑。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 的 12 天活动与令人惊讶的拨号机器人**：OpenAI 正在通过 [第 10 天精彩回顾](https://www.youtube.com/watch?v=LWa6OHeNK3s) 和 1-800-chatgpt 的大胆电话新功能来庆祝 **12 Days of OpenAI**。
  
  - 有些人认为电话服务在处理高级任务时并不可靠，但它可能对老年人或完全的初学者有所帮助。
- **Gemini 在 AI 竞争中取得进展**：传闻 **Gemini** 正在超越 **OpenAI**，引发了关于 AI 竞争加剧的讨论。
  
  - 怀疑论者暗示 OpenAI 可能在保留功能，仅在绝对必要时才发布，这引发了更多热议。
- **AI 模型安全性引发激烈辩论**：参与者引用了 [AlignAGI 仓库](https://github.com/AlignAGI/Alignment/)，思考人类是否能在没有高级 AI 支持的情况下独自解决 AI 安全问题。
  
  - 他们在**审查**与创意表达之间寻求平衡，并警告双方都可能出现意想不到的极端情况。
- **DALL-E 在与 Midjourney 和 Imagen 的对比中受挫**：一些人声称 **DALL-E** 在写实度上不如 **Midjourney** 和 **Imagen**，并将其归咎于限制性的设计选择。
  
  - 批评者指出“Midjourney 吹嘘得太多”，而其他人则坚持认为“DALL-E 需要更多自由”才能脱颖而出。
- **GPT 管理器训练与编辑难题**：爱好者们尝试将 **GPT** 作为“管理者”以简化任务，但发现效果一般。
  
  - 其他人对**自定义 GPTs** 的编辑限制表示沮丧，呼吁给予用户更多控制权。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **提示词链（Prompt Chaining）助力更敏捷的原型开发**：贡献者指出，**Prompt Chaining** 如何将一个模型的输出连接到另一个模型的提示词中，从而通过 [Langflow](https://docs.langflow.org/) 改进多阶段工作流和高级 Agent 设计。他们认为更快的迭代和更结构化的 AI 响应是主要优势，展示了小规模模型之间更好的协同效应。
  
  - 几位成员认为这种链式方法对于快速测试想法至关重要，并称赞它在完善新 AI 原型时消除了繁琐的手动编排。
- **Falcon 3 势头强劲**：成员们强调了 [Hugging Face](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit) 上的 **Falcon3-7B**，这是一个新发布的版本，改进了工具调用（tool-call）处理，并在 7B 到 40B 参数范围内具有可扩展的性能。热心的测试者讨论了其在模拟和推理任务中的潜力，并对其在本地硬件上处理实际使用情况的表现表示关注。
  
  - 他们还注意到大型 **Falcon** 变体即将进行的改进，并引用了用户关于在 HPC 部署中进行更稳健模型实验的反馈。
- **轻量级模型的本地函数调用**：参与者权衡了小型本地模型的**函数调用（function calling）**技术，比较了能有效解析结构化输出的库。他们寻求灵活的设置来处理个性化任务，而不依赖于大规模云解决方案。
  
  - 一些人以 **Gemini** 为例，说明将搜索任务卸载到真实数据源的做法，建议采用混合方法，而不是仅仅依赖聊天机器人进行记忆。
- **Hermes 3 405B 的重复性怪癖**：一位用户报告称，尽管有避免重复的指令，**Hermes 3 405B** 仍会逐字重复提示词，这使对话流程变得复杂。他们将其与 **gpt-4o** 进行了对比，指出后者在行为上表现出更简洁的合规性，且重复问题更少。
  
  - 社区成员尝试了专门的提示策略来抑制重复，强调了对高参数模型进行迭代微调以确保可靠性的重要性。
- **信噪比与一致的 LLM 输出**：一次讨论强调了 AI 推理中的**信噪比（signal vs noise）**，认为它是连贯思维的基石。社区对比将其与人类大脑过滤无关输入的方式进行了类比，将清晰度与更好的模型输出联系起来。
  
  - 一位成员还征求了关于在扩展 AI 响应中保持输出一致性的**最佳论文**，暗示了对长文本生成中持续可靠性的持续关注。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **三面板调整优化 NotebookLM**：新的 **3 面板 UI** 不再包含“解释”和“评论”等“建议操作”，以解决该功能使用率低的问题。
  
  - 目前，用户依靠从源文件中复制文本到聊天框，而开发人员计划以更直观的方式恢复缺失的功能。
- **引用与笔记面板调整**：NotebookLM 的最新版本从笔记中移除了嵌入式引用，引发了恢复这些引用的请求。
  
  - 用户还发现他们无法轻松合并选定的笔记，迫使他们只能要么全选，要么一次选择一个。
- **播客长度控制变得棘手**：在 **NotebookLM** 中设置较短音频段的尝试经常失败，因为 AI 显然忽略了长度指令。
  
  - 一个想法是将内容拆分为更小的文件，而 [博客公告](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/) 则暗示了未来的改进。
- **游戏与聊天的多语言混合**：玩家称赞 **NotebookLM** 通过检索增强查询简化了复杂的规则，而一些人则测试了用于交互式播客的多语言聊天。
  
  - 他们分享了诸如 [YouTube 上的 Starlings One](https://youtube.com/@starlingsone) 之类的链接以供更广泛使用，同时也对被称为“AI 垃圾（AI slop）”的肤浅 AI 播客发出了警告。
- **共享笔记本与空间奇谈**：用户希望在组织之外共享 NotebookLM 项目，引发了关于外部访问和家庭友好型功能的问题。
  
  - 与此同时，一段关于太空孤独感的 AI 视频提出了“你能活下来吗？”的问题，展示了 NotebookLM 在更广泛的视听用途上的潜力。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.2 Loss 之谜**：训练者在针对 1bn instruct 模型时，遇到了令人困惑的高 Loss 差异：使用 **Llama** 模板时为 (5.1→1.5)，而使用 **Alpaca** 模板时为 (1.9→0.1)。
  
  - 一位用户怀疑不正确的 Prompt 风格导致了这种差异，并进一步询问了关于合并新数据集进行重复微调的问题。
- **QwQ 推理模型之争**：成员们测试了开源模型 **QwQ**，但注意到在省略数学 Prompt 时，它会默认表现为 instruct 行为。
  
  - 一些人声称 **RLHF** 对于强化推理至关重要，而另一些人则认为仅靠 **SFT** 就能构建高级逻辑能力。
- **多 GPU 与 M4 MAX Mac 相关事宜**：用户证实 **Unsloth** 支持跨平台的多 GPU 使用，尽管有些人在 M4 MAX GPU 上安装时遇到了困难。
  
  - 开发者计划在 **2025 年第二季度**左右添加官方支持，并建议将 [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=r2v_X2fA0Df5) 和社区贡献作为临时解决方案。
- **LoRA 增益与合并最小化**：参与者澄清说 **LoRA** 适配器调整的参数较少，并与基础模型合并以获得更小的最终文件。
  
  - 他们指出，典型的合并会产生紧凑的 **LoRA** 输出，在不损害训练性能的情况下抑制 VRAM 占用。
- **DiLoCo 的分布式开发**：社区成员展示了他们关于大语言模型低通信训练的 [DiLoCo 研究演示文稿](https://docs.google.com/presentation/d/18Twuq0q1H75BxUOgRc8ZTs2lWGvE2XtnAGZ7CWtq3cA/edit?usp=sharing)。
  
  - 他们强调了在开源框架上的持续工作，并链接了 [DiLoCo arXiv 论文](http://arxiv.org/abs/2311.08105)，鼓励合作。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI 的 O1 增强**：OpenRouter 推出了支持 function calling、structured outputs 以及改进后的 `reasoning_effort` 参数的新型 **O1** 模型，详情见 [openrouter.ai/openai/o1](https://openrouter.ai/openai/o1)。
  
  - 用户可以在[此链接](https://x.com/OpenRouterAI/status/1869077909438091485)找到关于 structured outputs 的教程，并在 **Chatroom** 中探索挑战以测试模型的思维能力。
- **EVA Llama 登场**：OpenRouter 添加了 **EVA Llama 3.33 70b**，这是一款故事叙述和角色扮演专家，扩展了其在[此链接](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)的高级模型阵容。
  
  - 该模型专注于叙事生成，提升了平台创意互动的范围。
- **降价带来喜悦**：**gryphe/mythomax-l2-13b** 模型现在的价格降低了 **12.5%**，方便爱好者进行实验。
  
  - 同时，**QwQ** 推理模型的成本大幅下降了 **55%**，鼓励更多用户挑战其极限。
- **密钥泄露，支持警报**：一位用户在 GitHub 上发现了泄露的 **OpenRouter** 密钥，并被建议联系支持人员寻求即时帮助，避免直接通过电子邮件提交受损的 Token。
  
  - 其他人注意到调用后仅保留 metadata，如果需要更详细的跟踪，建议使用基于 proxy 的日志记录等解决方案。
- **Google 密钥费用与推理小问题**：社区成员确认，在将个人 Google AI 密钥链接到 OpenRouter 时会收取 **5%** 的服务费，这也适用于信用额度使用。
  
  - 同时，**QwQ** 在处理严格的指令格式时比较吃力，尽管 OpenAI 的 "developer" 角色最终可能会增强推理导向模型的合规性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 取得进展**：在 Google Shipmas 期间，**Gemini 2.0** 凭借 **Astra** 和 **Mariner** 等演示引起了关注，[Jeff Dean 确认了](https://x.com/JeffDean/status/1865079431544607089) Gemini-exp-1206 的进展。
  
  - 社区反馈赞赏其多模态成就，但也指出了 **Gemini Exp 1206** 中的速率限制（rate-limit）问题。
- **Copilot 的免费代码服务**：**GitHub** 宣布了 Copilot 的免费层级，每月包含 **2,000 次代码补全**和 **50 条聊天消息**，参考了[这条推文](https://x.com/github/status/1869447551876788359)。
  
  - 成员们称赞了 **Claude 3.5 Sonnet** 和 **GPT-4o** 模型的加入，同时预测 GitHub 目前 **1.5 亿+** 的用户基数将迎来开发者激增。
- **微软考虑投资 Anthropic**：据 [Dylan Patel](https://x.com/dylan522p/status/1869455045873528847) 透露，传闻 **Microsoft** 将以 **590 亿美元**的估值投资 **Anthropic**。
  
  - 他们的目标是在维持与 **OpenAI** 关系的同时引入 **Claude**，这引发了社区对这种微妙伙伴关系的讨论。
- **Qwen 2.5 Tulu 预告**：**Qwen 2.5 7B Tulu 3** 预计将超越 **Olmo**，具有改进的许可协议，并预告了正在开发中“更多疯狂的 RL 内容”。
  
  - 团队成员将重复的 RL 运行比作 "souping"，强调了令人惊讶的积极结果，为即将发布的版本造势。
- **RL 涌现的惊喜**：[Edward Hughes](https://x.com/edwardfhughes/status/1868624698260812108) 指出，在“捐赠者博弈”（Donor Game）中对 **LLM agents** 进行的实验表明，合作差异取决于基础模型。
  
  - 随后关于 **RLVR** 训练的更新引起了人们对基于结果奖励中自我修正行为的关注，引发了对重复 RL 运行的新兴趣。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **零售综述：电商工具关注 Runway**：成员们讨论了使用 **Runway**、**OpenAI Sora** 和 **Veo 2** 来制作**零售**广告内容的视频和文案，寻求更高层次的解决方案以脱颖而出。
  
  - 他们征求了更多建议，以优化营销方法，而不是重复旧有的策略。
- **Koopman 骚动：我们是在原地踏步吗？**：一篇关于[神经网络中 Koopman 算子理论](https://arxiv.org/abs/2409.01308)的论文引发了争论，讨论它究竟是真正增加了新的见解，还是仅仅重新包装了残差连接（residual connections）。
  
  - 一些人认为它缺乏实际价值，而另一些人则坚持认为该方法可能会通过先进的线性算子技术补充**网络分析**。
- **涌现标签：真实还是炒作？**：社区成员仔细研读了[《大语言模型的涌现能力是幻象吗？》](https://arxiv.org/abs/2304.15004)，质疑这些能力反映的是重大飞跃还是仅仅是评估伪像（evaluation artifacts）。
  
  - 这种怀疑延伸到了“仅靠扩展大模型就能解决核心局限”的假设，指向了更深层次的未解决理论鸿沟。
- **迭代压缩：廉价代理与 OATS**：工程师们讨论了用于模型压缩的迭代函数方法，并指出 [OATS 剪枝技术](https://openreview.net/forum?id=DLDuVbxORA)是减小模型体积且不牺牲高级行为的路径。
  
  - 他们还提出了*廉价代理*（cheap surrogate）层的策略，尽管有些人担心在链接近似值时会出现误差累积。
- **WANDB 日志与非参数 Norm：下一步是什么？**：开发者要求将 **MFU** 和吞吐量指标直接记录到 WANDB，暗示 [GPT-NeoX 的日志代码](https://github.com/EleutherAI/gpt-neox/blob/f5325805678c2b9e35aae4528283e0132c5f5bbc/megatron/logging.py#L352-L361)中即将加入新功能。
  
  - 他们还预计在未来几天内会有一个**非参数 layernorm** 的 pull request，从而为 GPT-NeoX 拓宽实验选项。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA 秘籍：训练心得**：成员们强调在训练 **LoRA** 之前收集高质量数据集的重要性，并建议通过彻底的测试和迭代来保证质量。
  
  - 他们强调了数据集策划策略的重要性，建议研究专门的资源以提高训练成功率。
- **Stable Diffusion 大杂烩**：建议初学者尝试 **InvokeAI**，因为它具有直观的工作流；而 **ComfyUI** 和 **Forge UI** 则因其模块化功能而受到推崇。
  
  - 分享了 [Civitai](https://civitai.com/models/1045828) 上的模型链接以及 [stable-diffusion-webui-forge 的 GitHub 脚本](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh)，并提供了有效利用它们的技巧。
- **量子争议与经典算力**：部分成员提到了 **quantum computing** 的突破，同时也指出实际部署仍需时日。
  
  - 针对量子技术进步引发的未来战争场景和计算能力的重大飞跃，人们表达了担忧。
- **GPU 收益与 FP8 调优**：优化 VRAM 使用是一个热门技巧，特别是在 **3060 GPU** 上采用 **FP8 mode** 以提升速度和显存效率。
  
  - 建议在生成图像期间监控 GPU 显存使用情况，以避免意外的减速或崩溃。
- **AI 视频愿景与局限**：参与者一致认为，虽然 AI 生成图像已取得显著进展，视频输出仍有提升空间。
  
  - 讨论了实现无缝 AI 视频的实际时间表，并提到了 [macOS 的静态 FFmpeg 二进制文件](https://evermeet.cx/ffmpeg/) 等工具，用于优化后期处理。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Spaces 重大更新：自定义 Web 来源**：Perplexity 在 *Spaces* 中引入了 **Custom Web Sources**，允许用户选择偏好的网站以获得更专业的搜索结果。一段简短的 [发布视频](https://cdn.discordapp.com/attachments/1047204950763122820/1318746209778929674/Custom_web_sources_launch_video_-_v6.mp4) 展示了这些来源的简化设置过程。
  
  - 社区成员强调了处理高级任务的潜力，提到 **customizing Perplexity** 如何满足高强度的工程需求。
- **赠送 Pro：订阅与频率限制**：用户称赞了 **Perplexity Pro** 赠送订阅功能，该功能提供了更多来源和 AI 模型，[Perplexity Supply 的推文](https://x.com/pplxsupply/status/1868738538231287816) 宣传了这一优惠。他们还对达到请求上限表示担忧，怀疑更高级别可能解决这些限制。
  
  - 有些人提议在 UI 中加入有趣的“降雪效果”，而另一些人则认为这对于硬核使用来说太让人分心。
- **Meta 与 OpenAI 的对峙**：Meta 希望阻止 **OpenAI** 的营利性尝试，引发了关于货币化 AI 伦理的辩论。社区讨论质疑企业优先级是否会掩盖开放研究的理想。
  
  - 其他人引用了早期的对峙，将其定性为无限制开发与收入驱动模型之间的“决定性时刻”。
- **拒绝死亡的细胞**：研究表明，**cells** 在死亡后可以恢复功能，这动摇了细胞最终关机的观念。一段 [视频解释](https://www.youtube.com/embed/7PBvDi_aKbs) 推动了关于急性医疗手段可能取得突破的讨论。
  
  - 论坛讨论还涉及了 **microbial threat warning**（微生物威胁警告），敦促密切关注健康影响和未来的预防措施。
- **植物之泪与多巴胺机制**：新发现声称 **plants** 可能会表现出类似于哭泣的压力信号，挑战了关于植物交流的旧观点。关于 *dopamine precursor* 探索的讨论也浮出水面，暗示了心理健康干预的精细化策略。
  
  - 参与者思考了更广泛的影响，提到这些生物学见解如何塑造研究轨迹。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MatX 直奔芯片**：**MatX** 宣布了一款旨在提升 AI 性能的 **LLM accelerator ASIC**，并正在积极招聘 **low level compute kernels**、**compiler** 开发和 **ML performance** 工程方面的专家，详情见 [MatX Jobs](https://matx.com/jobs)。
  
  - 他们强调了重塑下一代推理和训练的潜力，吸引了对 **on-chip** 性能提升感兴趣的工程师。
- **Pi 5 展现 1.5B 参数实力**：一台超频至 **2.8GHz** 并配备 **256GB NVMe** 的 **Raspberry Pi 5** 正在通过 **Ollama** 和 **OpenBLAS** 运行 **1.5B** 参数模型，展示了在边缘设备上的本地 LLM 部署。
  
  - 社区成员赞赏这种实用方法，指出 **Pi 5** 可以在不需要大规模 GPU 资源的情况下托管较小的专用模型。
- **CoT 获得视觉与深度**：一个团队探索了为小尺寸图像集成 **custom vision encoder**，并讨论了扩展 **Chain of Thought** 以通过更深层次的迭代步骤来完善推理。
  
  - 他们计划进行一项 **Proof of Concept**，将 **inner reasoning** 嵌入到 LLM 中，旨在实现更好的上下文处理和解决方案准确性。
- **强力的 int4group 与 Tinygemm**：工程师们描述了在 **int4group scheme** 中使用 **int4 weights** 和 **fp16** activations，让 **matmul kernel** 处理即时反量化（on-the-fly dequantization）。
  
  - 他们确认在训练期间不进行激活量化，利用 **bf16** 计算来确保性能的一致性。
- **A100 对决 H100**：用户注意到在比较 **A100** 和 **H100** 时，训练损失（training loss）存在 **0.3%** 的差异，引发了关于硬件特定变异性的疑问。
  
  - 他们辩论了是 **Automatic Mixed Precision (AMP)** 还是 GPU 架构的细微差别导致了这一差距，强调了进行更深层次分析的必要性。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Beta 惊喜**：用户使用 [Llama 3.2-11B-Vision-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit) 测试 **LM Studio** 时遇到了架构错误，如 **unknown model architecture mllama**。
  
  - 一些人通过查看 [LM Studio Beta Releases](https://lmstudio.ai/beta-releases) 克服了困难，并指出损坏的下载以及大型模型对某些用户有效。
- **角色扮演 LLM 势头强劲**：一位用户请求关于配置 **roleplay LLM** 的指导，促使社区分享了高级使用技巧。
  
  - 他们宣传了用于深入讨论的独立频道，并强调了长时间会话的内存限制。
- **GPU 困惑与驱动难题**：成员们质疑 *3060 Ti 11GB* 是否真的存在，还是其实是 12GB 的 **3060**，引发了关于 GPU 细节的辩论。
  
  - 与此同时，**Radeon VII** 受到驱动程序 24.12.1 问题的困扰，导致 **100% GPU usage** 却无功耗，被迫退回到 24.10.1。
- **推理开销与 Mac 愿望**：爱好者们意识到一个 **70B** 的 Llama 模型在 q8 量化下可能需要总计 **70GB** 的内存，涵盖 VRAM 和系统 RAM。
  
  - 一位用户开玩笑说，拥有一台 **M2 MacBook Air** 激发起对 **future MBP M4** 的渴望，指出了高性能配置的高昂成本。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **从 Firebase 切换到 Supabase**：成员们讨论了将整个网站从 **Firebase** 迁移到 **Supabase** 的过程，遇到了 [create-mf-app](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419) 和 **Bootstrap** 使用中的样式冲突。
  
  - 一位用户指出这些框架在配置上存在重叠，并暗示很快会完善迁移步骤。
- **Bolt Pilot GPT 开启 Beta 测试**：一位成员介绍了用于 **ChatGPT** 的 **Bolt Pilot** GPT，并建议探索 [stackblitz-labs/bolt.diy](https://github.com/stackblitz-labs/bolt.diy) 以获取相关的代码示例。
  
  - 他们展示了对多租户能力的乐观态度，并邀请社区反馈以指导未来的更新。
- **Token 纠纷与 Bolt 忧郁**：多位成员抱怨 **Bolt** 在占位符提示词上消耗 Token，敦促增加重置功能，并希望在 **Office Hours** 期间提供节日折扣。
  
  - 一些用户在 Token 上花费了大量资金却效果不佳，引发了对协作和资源池化的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **多模态嵌入（Multimodal Embeds）速率限制大幅提升**：Cohere 将生产密钥在 **Multimodal Image Embed** 端点的速率限制从 40 张图像/分钟提升至 400 张图像/分钟，而测试密钥固定为 **5 张图像/分钟**。
  
  - 官方鼓励用户使用更新后的 [API Keys](https://dashboard.cohere.com/api-keys)，并在[文档](https://docs.cohere.com/v2/docs/rate-limits)中列出了更多细节。
- **通过 Maya 命令增强本地模型**：开发者探索了通过发布 **command-r-plus-08-2024** 将本地模型与 Maya 连接，使模型能够处理查询中的*图像路径*。
  
  - 他们还在基础模型中增加了 **tool use**（工具使用）以实现更好的*图像分析*，引发了关于高级 Pipeline 设置的讨论。
- **Cohere Toolkit 应对 AWS 流错误**：在 AWS 上成功部署的 **Cohere Toolkit** 遇到了间歇性的 *stream ended unexpectedly*（流意外结束）警告，导致聊天兼容性中断。
  
  - 用户通过检查 **docker logs** 来诊断这一随机故障，希望能查明根本原因。
- **结构化输出展示与 Reranker 难题**：用户测试了 **Cohere** 的 **Structured Outputs** 配合 `strict_tools` 使用，通过精炼 Prompt 参数来获得精确的 JSON 响应。
  
  - 与此同时，一个基于 RAG 的 PDF 系统在使用 **Cohere Reranker** 时遇到了困难，表现为 Reranker 忽略了相关分块，但偶尔又能准确命中正确内容。
- **Findr 作为“无限大脑”亮相**：**Findr** 在 [Product Hunt](https://www.producthunt.com/posts/findr-remember-everything) 上线，提供了一个无限的记忆库来存储和访问笔记。
  
  - 爱好者们对这一发布表示欢迎，称赞其*可搜索数字大脑*的概念有助于更好地记忆。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 与 Archcraft：缺失链接器的谜团**：一位用户在 Archcraft 上启动 **Mojo REPL** 时遇到问题，原因是缺失 **mojo-ldd** 库以及未管理的环境阻塞了 Python 依赖。他们还提到安装 **Max** 的尝试停滞不前，引发了创建一个专门线程来解决问题的建议。
  
  - 一个 Stable Diffusion 示例被提及作为新功能的有用 GitHub 资源。对话强调了调整环境设置以避免安装突然失败的重要性。
- **文档与 'var' 之争：当语法遇上困惑**：关于 [Mojo 文档](https://docs.modular.com/mojo/manual/basics#variables)中明确提到的 **var** 关键字要求引发了讨论，让部分用户感到不安。官方透露即将更新文档，但未提供明确的时间表。
  
  - 社区成员对变量是否需要专用关键字持有不同意见。他们鼓励通过反馈来完善未来的文档版本。
- **Kernel 还是仅仅是一个函数？Mojo 的术语纠葛**：成员们澄清了 **Mojo** 中的 “Kernel” 通常指为 GPU 执行优化的函数，以区别于操作系统内核。该术语含义各异，有时描述核心计算逻辑或对加速器友好的代码。
  
  - 参与者交流了对该概念的理解，争论其是否应仅限于 GPU 任务。一些人注意到它在数学中用于表示更广泛计算中的基本操作。
- **缺失 argmax 与 argmin：关于 Reduction 的思考**：一位用户感叹 algorithm.reduction 中缺少 **argmax** 和 **argmin**，质疑是否需要从头开始重构它们。他们对重新实现那些在其他库中可能是标准的优化函数感到沮丧。
  
  - 成员们呼吁对这些操作提供更好的文档或官方支持。讨论凸显了 Mojo 不断演进的标准库中的连续性问题。
- **MAX 与 Mojo：自定义算子（Custom Ops）难题**：用户在 **custom ops** 集成上苦苦挣扎，引用了 `session.load(graph, custom_ops_paths=Path("kernels.mojopkg"))` 来修复缺失的 mandelbrot kernel 问题。他们还引用了 [Issue #269](https://github.com/modularml/max/issues/269)，要求改进错误消息和单编译单元 kernel。
  
  - MOToMGP Pass Manager 错误进一步复杂化了自定义算子的加载，促使人们呼吁在失败报告中提供更好的清晰度。贡献者强调了通过更具描述性的诊断来引导用户的重要性。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Nvidia 的 Nimble Nano 推动 AI 发展**：Nvidia 推出了售价 **$249** 的 [Jetson Orin Nano Super Developer Kit](https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/)，拥有 **67 TOPS** 的算力，性能比前代提升了 **70%**。
  
  - 它包含 **102GB/s** 的带宽，旨在帮助爱好者运行更繁重的 AI 任务，尽管一些参与者质疑它是否能处理高级机器人技术。
- **GitHub Copilot 开启免费模式**：正如 [Satya Nadella 所确认的](https://x.com/satyanadella/status/1869445091213095140)，GitHub **Copilot** 现在对 VS Code 免费开放，每月限制 **50 次对话**。
  
  - 社区成员讨论了与 Cursor 等竞争对手相比，这一限制是否会削弱 **Copilot** 的实用性。
- **1-800-CHATGPT 热线开通**：由 [Kevin Weil](https://x.com/kevinweil/status/1869446218163839264) 宣布，**1-800-CHATGPT** 为全球提供免费的电话和 WhatsApp 访问 GPT 的渠道。
  
  - 讨论强调了它对更广泛受众的普及性，无需额外的 App 或账号。
- **AI 视频工具迎来转折点**：OpenAI 的 [Sora](https://openai.com/index/sora/) 引发了关于视频模型演进的讨论，并引用了 **Will Smith 测试**片段来衡量进展。
  
  - 爱好者们将这些突破与早期的图像生成浪潮进行了比较，并引用了 [Replicate 的一篇博客文章](https://replicate.com/blog/ai-video-is-having-its-stable-diffusion-moment)，探讨了对高清 AI 视频日益增长的需求。
- **EvoMerge & DynoSaur 双重专题**：[LLM Paper Club](http://Latent.Space) 在一次“双头龙”会议中展示了 **Sakana AI 的 EvoMerge** 和 **DynoSaur**，并邀请现场观众提问。
  
  - 与会者被敦促将 [Latent.Space RSS](http://Latent.Space) 订阅源添加到日历中，以紧跟未来的活动。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Interpreter 中断问题加剧**：多位用户在使用 **Open Interpreter** 时遇到重复异常，导致丢失了关键的对话日志。他们报告了加载聊天历史记录和 API key 混淆的问题，在多个帖子中引发了不满。
  
  - 一位用户特别感叹*丢失了高质量的对话*，而其他用户则提到在多个设置中都发生了类似事件。尚未解决的技术故障继续阻碍着长时间的使用。
- **1.x 的谜团与 0.34 混杂**：社区成员讨论了 **Open Interpreter** 1.x 是否存在，而他们目前仅限于使用 0.34 版本。他们质疑功能上的变化，指出 OS 模式似乎并未出现在 1.0 中。
  
  - 这种版本不匹配引发了关于更新程序和支持的困惑。一些人询问切换版本的官方步骤，希望能确认新功能。
- **Cloudflare 扮演网关策略角色**：一位用户提议使用 **Cloudflare AI Gateway** 来解决 Open Interpreter 的一些配置障碍。这种方法引发了关于外部解决方案和高级部署的简短辩论。
  
  - 成员们考虑了新的工具链，研究 Cloudflare 的平台如何提高可靠性。他们还提到了与其他 AI 应用的协同作用，但尚未得出最终结论。
- **Truffle 的端侧设备预告**：一位用户介绍了 **Truffle-1**：这是一款拥有 64GB 内存的设备，押金为 $500，每月租金 $115。他们发布了[官方网站](https://itsalltruffles.com)，展示了一个支持无限端侧推理的球体设备。
  
  - [来自 simp 4 satoshi 的推文](https://x.com/iamgingertrash/status/1869450385896751449)提供了更多财务细节，提到了押金和月度计划。这激发了人们通过该球体设备的本地堆栈构建和共享自定义应用的兴趣。
- **长期记忆思维**：爱好者们探索了将扩展记忆与 **Open Interpreter** 集成的方法，重点在于代码库管理。一些人提议使用包括 Raspberry Pi 在内的本地设置来存储对话数据以备后用。
  
  - 他们看到了通过持久化日志实现流线型协作工具的潜力。这一想法在寻求保持更大上下文窗口随时可用的参与者中获得了支持。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LLaMA 基准测试之争**：一位用户询问是否有人对比过使用 tinygrad OpenCL 的 **LLaMA models** 与 **PyTorch CUDA** 的基准测试，但目前没有相关数据。
  
  - 讨论结论是目前没有已知的正面性能对比统计数据，这让 **AI engineers** 暂时无从参考。
- **ShapeTracker 合并乱象**：关于在 **Lean** 中合并两个任意 **ShapeTrackers** 的悬赏引发了关于 strides 和 shapes 的疑问。
  
  - 贡献者指出，一种通用的方法似乎行不通，因为变量使得合并过程变得复杂，超出了简单的修复范围。
- **反例导致崩溃**：成员们遇到了会导致当前合并算法在处理异常 view 对时崩溃的 **counterexamples**。
  
  - 他们暗示可以从单个不规则案例中自动生成更多示例，并强调了维度溢出（dimension overflow）问题。
- **CuTe Layout 代数对比**：**TinyGrad** 的合并对被比作 CuTe layout 代数中的 **composition**，参考了 [layout docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)。
  
  - 这种类比引起了人们对在确定 shape 兼容性之前验证某些代数属性这一复杂过程的关注。
- **单射性证明被视为 NP Hard**：对于证明 layout 代数中的 **injectivity** 存在质疑，有人认为这可能是 NP hard 问题。
  
  - 同时检查必要性和充分性似乎过于复杂，无法迅速解决，这暗示了一个更深层次的理论挑战。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FSDP 之争与 TRL 纠葛**：团队意识到，如果 **FSDP** 的 reduce 操作取平均值，则必须在 [这段代码](https://github.com/pytorch/torchtune/blob/3518492f43a8a5a462cbd604be4101268ff5bd52/recipes/full_finetune_distributed.py#L768) 中应用 `world_size` 缩放。他们还发现 **trl** 中可能存在缩放故障，并指向了 [这个修复 PR](https://github.com/pytorch/torchtune/pull/2172)。
  
  - 成员们建议提交一个 **Unsloth** 风格的 Bug 报告，并建议直接在 `scale_grads` 中调整代码以提高清晰度。他们预计这一修正将简化分布式设置中的梯度行为。
- **Loss 处理策略与梯度增益**：贡献者一致认为在训练 recipe 中显式缩放 loss 可以提高清晰度，参考了 [这段内存代码](https://github.com/pytorch/torchtune/blob/main/torchtune/training/memory.py#L219) 中的更新。他们强调代码注释有助于突出每个缩放步骤的目的。
  
  - PR 中增加了一个 **optimizer_in_bwd** 场景的修复，以正确处理归一化。此调整旨在保持训练循环的透明度并维持一致的梯度缩放。
- **Sakana 的进化视角**：工作人员对 **Sakana** 扩展 **evolutionary algorithms** 以抗衡基于梯度的技术产生了兴趣。他们发现进化驱动的方法可能会为 AI 发展注入不同的视角，这一点值得关注。
  
  - 一些人看到了将进化思想与标准梯度 recipes 合并的前景。其他人则计划关注 Sakana 的进展，看它是否能在严格的基准测试中站稳脚跟。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Collabin 视频片段引发好奇**：一段名为 **collabin** 的简短视频出现在[此链接](https://youtu.be/BrvVheleOqc)，暗示了某些协作项目或演示。
  - 参与者分享的细节很少，但视频展示了**未来团队合作**或演示的可能性。
- **自主 Agent 助力知识精英**：一篇名为 **Artificial Intelligence in the Knowledge Economy** ([链接](https://arxiv.org/abs/2312.05481)) 的新论文强调了**自主 AI Agent** 如何通过自动化常规工作来提升高技能人才的水平。
  - 社区评论提醒，随着这些 Agent 的普及，拥有深厚专业知识的人在生产力方面将获得**额外的优势**。
- **Coconut 将 LLM 推理引导至潜空间 (Latent Space)**：一篇关于 **Coconut (Chain of Continuous Thought)** ([链接](https://arxiv.org/html/2412.06769v1)) 的论文挑战了基于文本的一致性，转而建议采用潜空间方法。
  - **重 Token 策略 (Token-heavy strategies)** 可能会忽略细微差别，因此社区评论支持重写 LLM 的思维过程以管理复杂的规划。
- **RouteLLM 进度落后，但 DSPy 出现疑问**：人们注意到 **RouteLLM** ([仓库](https://github.com/lm-sys/RouteLLM)) 已不再维护，这引发了对未来与 DSPy 协同作用的担忧。
  - 虽然没有产生具体的计划，但这标志着 DSPy 生态系统内对强大**路由工具 (routing tools)** 的**渴望**。
- **DSPy 规划以推理为中心的路径**：讨论指出 **TypedReAct** 已部分弃用，敦促转向更简单的命名和模式，不再使用 'TypedChainOfThought'。
  - 其他人认为 **fine-tuning** 在 DSPy 内部正转向奖励级分支，并参考 [DSPy 的 Agent 教程](https://dspy.ai/tutorials/agents/) 作为后续步骤的资源。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Localdocs 支持点亮 GPT4All**：成员们讨论了在 **GPT4All** 中引用本地文档，但发现旧的 CLI 缺乏官方支持，这促使他们转向 **server API** 或 GUI 方法。
  - 几位参与者确认了 CLI 的局限性，强调 **local document 功能** 仍需要在官方工具集中进行特定配置。
- **GPT4All 的 Docker 梦想停滞**：一位用户询问关于在带有 Web UI 的 Docker 容器中运行 **GPT4All** 的问题，但没有人提供现成的解决方案。
  - 该问题仍然**悬而未决**，让容器爱好者们希望有人能尽快发布官方或社区镜像。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agentic AI SDR 激发线索**：在 blog 频道讨论中，可以关注这个使用 **LlamaIndex** 自动化营收任务并生成线索的 [agentic AI SDR](https://t.co/tczv5ZDI4H)。
  - 它展示了将 **function calling** 与销售工作流结合的新可能性，突显了 LlamaIndex 产生直接业务影响的能力。
- **Composio Quickstarters 启动 LLM Agent**：[Quickstarters 文件夹](https://twitter.com/llama_index/status/1869146329764831681) 指向了 **Composio**，将 LLM Agent 与 GitHub 和 Gmail 连接起来。
  - 通过 **function calling**，它实现了一种流线型的方法来处理代码提交 (code commits) 或收件箱扫描等任务，所有这些都由自然语言输入触发。
- **异步工具提升 OpenAIAgent 并发能力**：成员们讨论了 **OpenAIAgent** 的并发性，参考了 [OpenAI 文档](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#example-from-openai-docs) 中关于 API v1.1.0+ 并行函数调用的内容。
  - 他们分享了建议使用异步函数的代码片段，并澄清并发并不意味着真正的 CPU 并行执行。
- **RAG 评估协作浮现**：一位成员邀请社区在 **RAG 评估**方面进行合作，敦促其他人分享见解。
  - 他们欢迎通过直接讨论进行更深层次的技术探索，反映了人们对检索增强生成 (RAG) 技术日益增长的兴趣。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL 排行榜小故障**：一位用户观察到 [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 因**证书问题**导致临时停机，卡在 "Loading Model Response..." 界面。
  
  - 他们强调**模型端点 (model endpoint)** 无法访问，这引起了急于尝试新 Function Calling 功能的测试人员的关注。
- **Gorilla 基准测试关注 JSON 一致性**：一位参与者提议使用 **Gorilla** 基准测试来验证模型对 **JSON schema** 或 Pydantic 模型的遵循情况，强调结构化输出测试。
  
  - 他们询问是否有专门用于衡量**结构化生成 (structured generation)** 准确性的子任务，尽管目前还没有官方提到此类任务。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GPT-O1 逆向工程引发研究人员关注**：爱好者们征集任何已知的 **GPT-O1** 逆向工程努力或材料，包括技术报告和论文，并欢迎来自社交媒体帖子的额外见解。这引发了集体分享和资源收集的热潮，旨在揭开 **GPT-O1** 复杂性的神秘面纱。
  
  - 他们提议发起一个协作倡议，汇编有关 **GPT-O1** 的参考资料，特别是来自 Twitter 讨论和已发表的材料，旨在汇集社区知识。
- **Meta 开启生成式 AI 实习**：**Meta** 宣布了一个为期 3-6 个月的文本生成图像模型和视觉语言模型研究实习职位，提供大规模动手实验的机会，申请地址在[这里](https://www.metacareers.com/jobs/539208255364368/)。该职位专注于推动*核心算法进展*并构建生成式 AI 的新能力。
  
  - 变现生成式 AI 团队正在寻找具有**深度学习 (deep learning)**、**计算机视觉 (computer vision)** 和 **NLP** 背景的研究人员，强调对用户在线交互方式产生全球性影响。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **无重大讨论**：我们只看到了一些友好的感谢表达，没有发布额外的信息或参考资料。
  
  - 没有进一步的聊天或数据共享，因此没有其他值得关注的内容。
- **无技术更新**：本次讨论中没有出现新的模型、代码发布或相关的技术进展。
  
  - 对于 AI 专家来说，这条单一消息没有提供更多数据。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **一月强化学习人才加入**：一名新工程师将于 **1 月**加入团队，协助**强化学习 (RL)** 计划，为训练扩展提供额外人手。
  
  - 他们还将为 **KTO** 项目提供直接支持，确保 RL 组件的及时集成和功能的改进。
- **KTO 获得额外人手**：新工程师入职后将帮助完善 **KTO** 系统，重点关注实时性能的提升。
  
  - 项目负责人预计他们的贡献将显著提高 RL 任务的生产力。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **开发者中心 (Developer Hub) 势头强劲**：一项**重大公告**推出了 **Developer Hub** 的全新功能，强调通过社区反馈进行持续改进，详见[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1318638353503227935)。
  
  - 参与者强调了明确使用指南以及收集未来扩展建议的重要性。
- **Blueprints 计划简化 AI 构建**：**Blueprints 计划**旨在通过精心提供的资源帮助开发者组装开源 AI 解决方案，讨论见[此处](https://discord.com/channels/1089876418936180786/1318689803021058158)。
  
  - 他们指出，计划中的增强功能可以实现更广泛的项目协作，并为特定用例提供灵活的模板。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：渠道详细摘要和链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #**[**discussion**](https://discord.com/channels/1027685395649015980/1027697446446432336/1318669267478253689) (60 条消息🔥🔥):

> `Codeium 问题, Windsurf 性能, Elementor 与 Codeium 集成, 支持工单系统, JetBrains 扩展问题`

- **用户报告频繁的 Codeium 问题**：许多用户反映了 Codeium 扩展的问题，包括自动补全建议消失以及与服务器的连接问题。
  
  - *jamespond000* 敦促遇到最新 JetBrains 版本问题的用户创建带有日志的支持工单，以便获得进一步协助。
- **Windsurf 导致用户笔记本电脑变慢**：许多用户报告 Windsurf 导致他们的笔记本电脑显著变慢，部分用户甚至需要等待超过十分钟才能打开应用程序。
  
  - 部分用户每隔几秒就会出现错误，这表明存在持续的性能问题。
- **Elementor 用户咨询 Codeium**：用户好奇 Codeium 扩展是否能协助他们使用 JSON 或 CSS 代码为 Elementor 元素编写功能。
  
  - 关于 Codeium 是否能为这些元素自动生成代码，目前还存在困惑。
- **针对计费问题的支持工单请求**：一些用户报告在取消订阅后仍被扣费，正寻求通过支持渠道解决。
  
  - 用户被引导至 codeium.com/support 提交工单以解决这些计费问题。
- **Flex credits 及其结转政策**：一位用户询问 flex credits 是否会累积或结转到后续的计费周期。
  
  - 关于 flex credit 系统的说明在用户中仍不明确。

**提到的链接**：

- [Hello There GIF - Hello there - Discover & Share GIFs](https://tenor.com/view/hello-there-gif-5677380953331354485)：点击查看 GIF
- [Reasoning with o1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/)：学习如何使用 OpenAI 的 o1 模型并进行提示词工程，以处理复杂的推理任务。

---

### **Codeium (Windsurf) ▷ #**[**windsurf**](https://discord.com/channels/1027685395649015980/1306163501286293515/1318669293239533671) (678 条消息🔥🔥🔥):

> `Windsurf 性能问题，Codeium 与 Copilot 的对比，Cascade 功能，Llama 模型基准测试，免费 AI 工具选项`

- **Windsurf 出现性能问题**：用户报告称 Windsurf 自上次更新以来一直存在严重的 Bug 和性能下降，诸如“代码消失”和内部错误等问题变得越来越频繁。
  
  - 由于这些问题，一些用户甚至在考虑降级，并正在寻求针对持续性问题的支持。
- **Codeium 对比 Copilot**：几位用户讨论了 Codeium 与 Copilot 的效能对比，有人表示尽管 Copilot 推出了新的免费层级，Codeium 的 autocomplete 仍然更胜一筹。
  
  - 用户对 Copilot 免费层级的引入可能如何影响 Claude 和 GPT 的性能表示担忧，推测可能会出现过载。
- **Cascade 的自动批准功能**：有人注意到 Cascade 一直在自动批准更改而无法进行审查，一些用户认为这很烦人，因为审查更改至关重要。
  
  - 用户对 Cascade 的性能表示不满，并正在分析如何提高其输出质量。
- **Llama 模型性能**：对话强调了 Llama 模型的基准测试，特别是 Llama 3.3 和 4o-mini，用户指出在特定条件下，某些版本可以超越更大的模型。
  
  - 有人提到 4o-mini 对于较小的任务非常有用，并且具有与高级模型类似的能力。
- **AI 工具探索**：用户讨论了各种免费的 AI 工具和平台，强调了项目规划和 Prompt 结构化对于最大限度提高这些技术效率的重要性。
  
  - 总的来说，参与者分享了使用不同 AI 工具的经验，权衡了它们在编码中的优势，以及在项目中使用 Gemini Flash 等模型的潜力。

**提到的链接**：

- [LiveBench](https://livebench.ai/#/?Coding=a)：未找到描述
- [Windsurf - Focus Follows Mouse (as a configuration option) | Feature Requests | Codeium](https://codeium.canny.io/feature-requests/p/windsurf-focus-follows-mouse-as-a-configuration-option)：VSCode 有一个公开的 GitHub PR，表面上看已经超过 4 年了，但实际上比这还要久。
- [Productionizing and scaling Python ML workloads simply | Ray](https://www.ray.io)：Ray 管理、执行并优化跨 AI 工作负载的计算需求。它统一了基础设施并支持任何 AI 工作负载。立即免费试用。
- [Cannot use windsurf as git editor | Feature Requests | Codeium](https://codeium.canny.io/feature-requests/p/cannot-use-windsurf-as-git-editor)：`git config --global core.editor 'windsurf --wait '` 在 rebases 时抛出错误提示：Waiting for your editor to close the file... [1119/144632。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPTCoding/s/vPI207Unh3)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/Codeium/s/v22PYhpKn1)：未找到描述

---

### **Cursor IDE ▷ #**[**general**](https://discord.com/channels/1074847526655643750/1074847527708393565/1318672809093763163) (707 条消息🔥🔥🔥):

> `Cursor IDE Updates, Kepler Browser Development, Python Environment Management, O1 Pro Performance, Galileo API Integration`

- **Cursor IDE 更新**：最新更新 Cursor 版本 0.44.2 在因不稳定从 0.44 版本回滚后发布，用户反馈功能有所改进。
  
  - 讨论了各种功能，包括 Composer 在终端文本粘贴不当方面的恼人行为。
- **Kepler 浏览器开发**：一位用户正在开发一款名为 Kepler 的注重隐私的浏览器，使用 Python 构建，旨在无需后端服务器，强调用户控制。
  
  - 该浏览器旨在通过随机 User Agent 等功能提高安全性，并已开源以供社区贡献。
- **Python 环境管理**：用户讨论了使用 UV 工具高效管理 Python 环境，特别是其处理各种 Python 版本的能力。
  
  - 该工具简化了虚拟环境的创建，使开发者更容易管理依赖项和项目配置。
- **O1 Pro 性能**：O1 Pro 功能收到了积极反馈，一位用户报告称它在尝试 20 多次后成功解决了他们的 bug。
  
  - 讨论表明 O1 Pro 增强了性能，尽管 Chat 和 Composer 中的输出格式问题仍然存在。
- **Galileo API 集成**：有关于 Cursor 中 Galileo API 和 Gemini 2.0 等模型可用性的咨询，用户目前遇到了一些限制。
  
  - 用户表示有兴趣测试集成到 Cursor 平台中的新模型的功能和特性。

**提到的链接**：

- [Settings | Cursor - The AI Code Editor](https://www.cursor.com/settings)：您可以在此处管理您的账户、账单和团队设置。
- [Downloads | Cursor - The AI Code Editor](https://www.cursor.com/downloads)：选择您的平台以下载最新版本的 Cursor。
- [Python Environment Manager - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-environment-manager)：Visual Studio Code 扩展 - 查看和管理 Python 环境与包。
- [Poetry - Python dependency management and packaging made easy](https://python-poetry.org/)：未找到描述
- [uv](https://docs.astral.sh/uv/)：未找到描述
- [no title found](https://download.todesktop.com/230313mzl4w4u92/cursor-0.44.0-build-2412187f9v0nffu-x86_64.AppImage)：未找到描述
- [WARNING: Cursor v0.44 breaks all devcontainers v0.394.0](https://forum.cursor.com/t/warning-cursor-v0-44-breaks-all-devcontainers-v0-394-0/35747/7)：你是如何强制禁用 Cursor 更新的？我陷入了困境，每次重启 Cursor，它现在总是会更新到 v0.44.0。额外的问题是，即使我禁用了 “devcontainer” 扩展...
- [Danger Alert GIF - Danger Alert Siren - Discover & Share GIFs](https://tenor.com/view/danger-alert-siren-alarm-red-light-gif-16931369)：点击查看 GIF
- [Changelog | Cursor - The AI Code Editor](https://www.cursor.com/changelog)：新的更新和改进。
- [GitHub - TheGalaxyStars/KEPLER-COMMUNITY: Explore freely, leave no trace.](https://github.com/TheGalaxyStars/KEPLER-COMMUNITY)：自由探索，不留痕迹。通过在 GitHub 上创建账户来为 TheGalaxyStars/KEPLER-COMMUNITY 的开发做出贡献。
- [GitHub - ultrasev/cursor-reset: Mac utility to reset Cursor editor's device identification system. Helps resolve account restrictions and trial-related issues.](https://github.com/ultrasev/cursor-reset)：用于重置 Cursor 编辑器设备识别系统的 Mac 工具。有助于解决账户限制和试用相关问题。 - ultrasev/cursor-reset
- [GitHub - ZackPlauche/add-cursor-to-win-context-menu](https://github.com/ZackPlauche/add-cursor-to-win-context-menu)：通过在 GitHub 上创建账户来为 ZackPlauche/add-cursor-to-win-context-menu 的开发做出贡献。
- [index.html - Pastebin.com](https://pastebin.com/Bgu7XD6C)：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个您可以在线存储文本一段时间的网站。
- [style.css - Pastebin.com](https://pastebin.com/cqNdfphK)：Pastebin.com 自 2002 年以来一直是排名第一的粘贴工具。Pastebin 是一个您可以在线存储文本一段时间的网站。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1318696787279876159) (264 条消息🔥🔥):

> `O1 API Release, Aider Benchmarking, Competition in AI Models, Support and Refunds, Using Gemini as Editor`

- **O1 API 仍在逐步开放访问权限**：包括 Tier 5 API 密钥持有者在内的许多用户报告称，由于 **O1 API** 正在逐步推出，他们尚未获得访问权限。围绕其能力的讨论凸显了 O1 Pro 与标准版之间的差异。
  
  - 虽然一些人对 **O1 的推理能力**感到兴奋，但也有人持批评态度，将其与之前的模型进行比较，并指出价格等因素仍然是一个主要问题。
- **Aider 在基准测试中的表现**：用户分享了他们使用 Aider 进行基准测试的经验，指出 **O1 得分为 84.2**，与 **Sonnet** 持平，而有人声称它在特定任务中表现甚至更好。Discord 上的对话强调了在编程和推理任务中模型选择的重要性。
  
  - 随着 Aider 的发展，社区成员表达了对高效编辑和调试模型的渴望，并指出 O1 的更新迭代对于测试其功能至关重要。
- **AI 模型之间日益激烈的竞争**：讨论中出现了关于 AI 模型产品竞争日益激烈的议题，包括提及 **Google 的 Veo 2** 以及关于模型有效性的持续争论。用户特别注意到 **O1 Pro** 与 **Sonnet** 等现有工具在输出和实用性方面的差异。
  
  - 随着新模型的出现，人们对价格点和当前 AI 订阅模式的可持续性提出了担忧，凸显了用户的挫败感和期望。
- **支持流程与退款**：几位用户分享了他们在 OpenAI 支持和退款流程方面的经历，包括漫长的等待时间和参差不齐的结果。一位用户幽默地提到，在联系支持部门后 **退款花了四个月**才到账，这增加了人们对服务质量的怀疑。
  
  - 相反，另一位用户报告在几小时内就收到了快速退款，反映了用户在客户支持响应速度方面体验的不一致。
- **Gemini 作为编辑器的表现**：一些用户称赞了 **Gemini，特别是 gemini/gemini-exp-1206** 模型的编辑能力，指出在广泛使用过程中错误极少。讨论强调了 **Gemini** 在与 Aider 配合使用时，如何有效地处理特定的编程任务。
  
  - 尽管存在竞争，但人们对 **Gemini 与 O1 等模型相比的局限性**仍有疑虑，强调用户需要根据其编程需求识别合适的工具。

**提到的链接**：

- [Andrew Ng (@AndrewYNg) 的推文](https://x.com/AndrewYNg/status/1869421643925422166)：OpenAI 昨天刚刚宣布了 o1（高级推理模型）的 API 访问权限。我很高兴今天宣布一门新的短课程《使用 o1 进行推理》，该课程与 @OpenAI 合作构建，由 @colintjarvis 授课...
- [Linting 和测试](https://aider.chat/docs/usage/lint-test.html)：自动修复 Linting 和测试错误。
- [Poonam Soni (@CodeByPoonam) 的推文](https://x.com/CodeByPoonam/status/1869289412951220395)：Google 刚刚发布了 Veo 2，简直太疯狂了。剧透：OpenAI Sora 现在落后了。10 个展示其能力的疯狂示例：（不要错过第 5 个）
- [o1 - API, Providers, Stats](https://openrouter.ai/openai/o1)：来自 OpenAI 的最新且最强的模型系列，o1 旨在响应前花费更多时间思考。o1 模型系列通过大规模强化学习进行训练，以使用...进行推理。
- [选项参考](https://aider.chat/docs/config/options.html)：关于 aider 所有设置的详细信息。
- [选项参考](https://aider.chat/docs/config/options.html#fixing-and-committing)：关于 aider 所有设置的详细信息。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1318706793655963689) (18 messages🔥):

> `Aider 与 Gemini Flash 2 Grounding、使用 Aider 进行项目管理、Aider 的文件处理问题、使用 Architect 与 Ask 模式、Repo Map 相关担忧`

- **Aider 支持 Gemini 2.0 的 Google Search**：一位成员指出 **Gemini 2.0 Flash Experimental** 支持 Google Search grounding，但这可能非常**昂贵**，每 1000 次请求耗费 **$35**。
  
  - 必须遵守关于搜索结果显示的 ToS，且 grounding 功能使用了特定的模型配置。
- **优化 Aider 的项目管理**：一位用户讨论了通过 **O1** 和 **Claude-Sonnet** 管理任务，询问是否可以将 O1 作为 architect 模型引入其项目管理流程。
  
  - 用户对随着项目增长，由 **Claude** 进行功能开发所导致的**混乱代码**表示担忧，这通常需要**重构步骤**。
- **Aider 文件处理 Bug 报告**：一位用户报告了一个 Bug，在使用 Aider 的 **/add 命令**后无法访问用于添加文件的下拉菜单。
  
  - 该问题已确认在 main 分支中修复，并附带了使用 `aider --install-main-branch` 进行安装的说明。
- **理解 Aider 的 Architect 和 Ask 模式**：用户寻求关于 **/architect** 和 **/ask** 之间区别的澄清，指出它们看起来很相似，但在应用两个不同的 LLM 时可能会很有用。
  
  - 讨论了如何有效地管理在一个模式中定义的计划以及在另一个模式中的分步实现。
- **关于 Repo Map 变化的担忧**：一位用户对 **repo map** 随每次重构而变化表示担忧，担心 Aider 可能会在原始代码库和当前代码库之间迷失方向。
  
  - 尽管承认 repo map 在上下文中的好处，但他们预见到在项目演进过程中维持秩序的挑战。

**提到的链接**：

- [Repository map](https://aider.chat/docs/repomap.html)：Aider 使用 Git 仓库的 map 为 LLM 提供代码上下文。
- [FAQ](https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-t)：关于 aider 的常见问题。
- [yamad - Overview](https://github.com/yamad)：yamad 有 85 个公开仓库。在 GitHub 上关注他们的代码。
- [FAQ](https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat)：关于 aider 的常见问题。
- [GitHub - yamadashy/repomix: 📦 Repomix (formerly Repopack) is a powerful tool that packs your entire repository into a single, AI-friendly file. Perfect for when you need to feed your codebase to Large Language Models (LLMs) or other AI tools like Claude, ChatGPT, and Gemini.](https://github.com/yamadashy/repomix)：📦 Repomix（原名 Repopack）是一个强大的工具，可将整个仓库打包成单个 AI 友好文件。非常适合需要将代码库提供给 LLM 或其他 AI 工具（如 Claude、ChatGPT 和 Gemini）的情况。
- [Add support for Gemini 2.0 GoogleSearch tool by samling · Pull Request #7257 · BerriAI/litellm](https://github.com/BerriAI/litellm/pull/7257)：标题：将 googleSearch() 工具添加到 Gemini/VertexAI 模型的有效工具列表中，以支持 Gemini 2.0 grounding。相关 Issue：增强了 #7188。类型：🆕 新功能。✅ 测试。变更：添加 googleSearch() 工具...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1318741968242737282) (11 messages🔥):

> `Depth AI, LightRAG, Codebase indexing, AI assistants, Technical accuracy`

- **Depth AI 在理解代码库方面表现出色**：成员们讨论了他们使用 [Depth AI](https://www.trydepth.ai/) 的经验，这是一款能够准确索引代码库并以 **99% 的准确率**回答深度技术问题的 AI 工具。
  
  - *一位用户指出*，“到目前为止，我在一个已经忘掉的大型代码库上使用它的体验非常棒。”
- **大型项目的索引时间各不相同**：项目的索引时间差异很大，一位用户指出 **180k** token 的代码库索引需要 **40 分钟**，而另一位用户报告说他们的中型项目已经索引了 **4 小时**。
  
  - 他们提醒说，大型项目的索引可能需要 **1-2 小时**，特别是那些在 **200k** 到 **150 万** token 之间的项目。
- **LightRAG 提供替代方案**：一位成员提到了 [LightRAG](https://github.com/HKUDS/LightRAG)，将其描述为“简单且快速的检索增强生成（RAG）”，为 Depth AI 提供了一个替代方案。
  
  - 然而，另一位用户表示更倾向于 Depth AI，因为它易于设置且被认为更具优势。
- **Depth AI 的初始输出问题**：一位成员遇到了一个问题，即 Depth AI 在索引其仓库后返回了“未生成输出（no output was generated）”。
  
  - 尽管最初对该工具充满热情，但这引发了对输出可靠性的担忧。

**提到的链接**：

- [Depth AI - 深度理解代码库的 AI](https://www.trydepth.ai/)：与你的代码库聊天或构建定制的 AI assistants。将它们部署在任何你工作的地方 —— Slack, GitHub Copilot, Jira 等。
- [GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG)："LightRAG: 简单且快速的检索增强生成" - HKUDS/LightRAG

---

### **OpenAI ▷ #**[**annnouncements**](https://discord.com/channels/974519864045756446/977259063052234752/1318999359412506645) (1 messages):

> `12 Days of OpenAI, OpenAI Role Customization`

- **随时掌握 OpenAI 的最新动态！**：社区提醒大家在 **12 Days of OpenAI** 活动期间保持关注。
  
  - 参与者可以在 [customize](https://discord.com) 中选取角色以接收更新。
- **观看第 10 天精彩回顾**：第 10 天的内容通过一个展示活动的 [YouTube 视频](https://www.youtube.com/watch?v=LWa6OHeNK3s) 链接进行了重点介绍。
  
  - 该视频捕捉了与正在进行的庆祝活动相关的各个方面和公告。

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1318674710694989916) (220 条消息🔥🔥):

> `OpenAI ChatGPT 进展，Gemini AI 对比，AI 模型安全担忧，生成式图像差异，用户对 AI 模型的使用体验`

- **OpenAI 为 ChatGPT 推出电话支持**：OpenAI 宣布了一项新功能，允许美国居民拨打 1-800-chatgpt 与 ChatGPT 通话，但一些用户认为这令人失望且没那么有用。
  
  - 讨论中提到这可能对老年用户有益，但对其对大多数受众的实用性仍持怀疑态度。
- **Gemini 与 OpenAI 在 AI 进展方面的对比**：用户讨论了 Google Gemini AI 的快速进步，并将其与 OpenAI 的产品进行了对比，认为 Gemini 在 AI 竞争中正逐渐占据优势。
  
  - 有人担心 OpenAI 可能会保留技术进展，直到为了与其他公司的新发布竞争时才推出。
- **围绕 AI 模型安全与控制的辩论**：对话中包含了对人类未来能否在没有 AI 协助的情况下有效管理 AI 安全问题的怀疑。
  
  - 参与者对审查与允许创作自由之间的平衡表示怀疑，认为这两个极端都可能存在陷阱。
- **图像生成模型的对比分析**：讨论显示，用户认为 DALL-E 缺乏 Midjourney (MJ) 和 Google Imagen 等其他模型所提供的真实感和质量。
  
  - 一些人认为 DALL-E 的局限性源于强加的约束，而另一些人则认为 Midjourney 过度夸大了其能力。
- **用户在日常任务中使用 AI 工具的体验**：一位用户分享了他们希望如何利用 AI Agent 来帮助正在中风康复中的父亲，强调了对残障人士的潜在益处。
  
  - 普遍共识是，在隐私问题得到妥善解决的前提下，多功能且个性化的 AI 助手可以极大地提高生活质量。

 

**提到的链接**：[GitHub - AlignAGI/Alignment: 促进全球对道德 AI 对齐的认识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。](https://github.com/AlignAGI/Alignment/): 促进全球对道德 AI 对齐的认识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。 - AlignAGI/Alig...

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1318739662197624903) (3 条消息):

> `GPT 管理培训，编辑自定义 GPT`

- **质疑 GPT 的管理角色**：一位成员询问，提示 ChatGPT 扮演经理角色的功能在特定任务培训中是否有效。
  
  - 这引发了关于 AI 交互中角色扮演实际应用的有趣讨论。
- **对编辑自定义 GPT 的挫败感**：用户分享了关于无法编辑自定义 GPT 的担忧，这让他们感到无从下手。
  
  - 这个问题突显了用户在定制 GPT 并寻求改进时在灵活性方面的局限性。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1318796591389347902) (4 条消息):

> `频道适用性，垃圾信息管理，寻求帮助`

- **频道使用说明**：一位成员对是否在正确的频道发帖表示不确定，正在寻求**帮助**以获得最佳建议。
  
  - *“我不清楚合适的频道”* 突显了对更清晰频道指南的需求。
- **处理垃圾信息问题**：另一位成员将这些帖子识别为**垃圾信息**，并指明了寻求帮助的正确频道：<#1047565374645870743>。
  
  - 他们要求从其他频道删除这些帖子，以保持讨论的条理性。
- **成员确认反馈**：寻求帮助的成员用点赞表情确认了反馈，表示接受所提供的指导。
  
  - 这表明了用户愿意遵循正确的频道协议进行后续咨询。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1318796591389347902) (4 条消息):

> `频道混淆，垃圾信息担忧`

- **频道混淆导致明显的垃圾信息**：一位用户对是否在正确的频道发帖表示不确定，寻求帮助以获得更好的建议。
  
  - 另一位成员将这些发布内容识别为垃圾信息，建议了正确的频道，并表示在妥善删除后提供帮助。
- **成员寻求更好的建议**：用户 .noval 在收到关于正确频道的指导后，通过点赞表情确认了他们的意图。
  
  - 这一交流反映了一个更广泛的主题，即确保相关的讨论在适当的空间进行。

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1318670110012997702) (210 条消息🔥🔥):

> `Prompt Chaining, Falcon Models, AI Tool Use, OpenAI Safety Discussions, Data Preprocessing Optimizations`

- **探索 Prompt Chaining 技术**：成员们讨论了 Prompt Chaining 的概念，即一个模型的输出可以作为另一个模型的输入，从而为 LLM 构建复杂的工作流。
  
  - 这有助于更快地进行 Agent 设计的原型制作，并使 AI 响应中能够产生更好的结构化输出。
- **Falcon3-7B 的初步印象**：Falcon3-7B 模型最近备受关注，在更新提升了其处理 Tool Calls 的能力后，用户们渴望测试其性能。
  
  - 一些成员对其成熟度表示不确定，但承认其在市场模拟应用方面的潜力。
- **对 OpenAI 安全实践的担忧**：在 GPT-4o 和 o1 preview 的对比中，OpenAI 在强调安全性的同时演示了 Jailbreak 方法，这引发了外界的审视。
  
  - 这引发了关于 AI 模型运行安全与滥用风险之间平衡的讨论。
- **数据预处理速度的进展**：一位用户分享了他们在数据预处理时间上的显著改进，将工作时长缩短至现在的约 10 小时。
  
  - 这种优化突显了通过重叠日志记录和改进数据准备逻辑所实现的效率提升。
- **Tool Use 与模型协调**：成员们观察到模型文档与实际能力之间存在脱节，特别是在 Falcon 模型及其 Tool-use 功能方面。
  
  - 讨论内容包括将这些模型集成到市场 Agent 模拟中，以评估其在真实场景中的有效性。

**提到的链接**：

- [Welcome to Langflow | Langflow Documentation](https://docs.langflow.org/)：Langflow 是一个用于构建多 Agent 和 RAG 应用的新型可视化框架。它是开源的、由 Python 驱动、完全可定制，并且与 LLM 和向量数据库无关。
- [Scaling test-time compute - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)：未找到描述
- [Tweet from Democratize Intelligence (@demi_network)](https://x.com/demi_network/status/1869085748852662718)：“这不是公司与 AI 之间的对齐问题，而是公司与你之间的对齐问题。你的 AI 为谁工作将变得非常重要。如果你的 AI 是……”
- [tiiuae/Falcon3-7B-Instruct-1.58bit · Hugging Face](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit)：未找到描述
- [tiiuae/falcon-11B · Hugging Face](https://huggingface.co/tiiuae/falcon-11B)：未找到描述
- [tiiuae/falcon-7b-instruct · Hugging Face](https://huggingface.co/tiiuae/falcon-7b-instruct)：未找到描述
- [Tweet from xjdr (@_xjdr)](https://x.com/_xjdr/status/1869062741849461146)：这是我在 NeurIPS 从~可靠来源多次听到的最有趣的事情之一（newsonnet 是 400B dense 模型）引用 Aidan McLau (@aidan_mclau) @Heraklines1 @deedydas 不，不是……
- [tiiuae/falcon-40b-instruct · Hugging Face](https://huggingface.co/tiiuae/falcon-40b-instruct)：未找到描述
- [Welcome to the Falcon 3 Family of Open Models!](https://huggingface.co/blog/falcon3)：未找到描述
- [tiiuae/Falcon3-10B-Instruct · Hugging Face](https://huggingface.co/tiiuae/Falcon3-10B-Instruct)：未找到描述
- [tiiuae/Falcon3-10B-Instruct · Hugging Face](https://huggingface.co/tiiuae/Falcon3-10B-Instruct#benchmarks))：未找到描述
- [Safepine](https://safepine.co/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/175ejvi/quick_start_example_for_llava_generate_image/)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1318853936660221984) (13 messages🔥):

> `Function Calling Methods for Local Models, Language Model Data Recollection, Bias in AI Search Integration, Hermes 3 405B Model Responses, Search Engine Expectations`

- **探索本地模型上的 Function Calling**：讨论集中在小型本地模型上进行 **function calling** 的最佳库和方法，并评估其有效性。
  
  - 成员们正在寻找针对个人用例量身定制的、旨在提高功能效率的解决方案。
- **对语言模型数据回忆（Data Recollection）的担忧**：一位成员认为不应使用语言模型聊天机器人来回忆数据，因为数据会根据上下文和作者意图而变化。
  
  - 他们强调，像 **Gemini** 这样的模型应该采用软件方法来获取可靠数据，而不是仅仅依赖聊天机器人。
- **聊天机器人搜索功能引入的偏见**：人们对在聊天模型中启用搜索功能可能导致偏见增加和可信度降低表示担忧。
  
  - 一位成员指出，除非搜索源经过人工筛选，否则结果的整体质量仍然存疑，正如普遍存在的垃圾信息和 SEO 策略所显示的那样。
- **Hermes 3 405B 模型：提示词重复问题**：一位用户分享了关于 **Hermes 3 405B 模型**在响应中逐字返回 Prompt 的挫败感，尽管已有不这样做的指导。
  
  - 他们提到正在尝试不同的 Prompting 策略以减少重复行为，并将其与 **gpt-4o** 的响应质量进行了对比。
- **搜索引擎质量的未来**：一位成员表达了对涵盖所有著作的搜索引擎的希望，批评当前的搜索结果充斥着 SEO 策略。
  
  - 讨论反映了在现有局限性下，用户对更**强大且可靠**的搜索功能的渴望。

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1318732957287972976) (2 messages):

> `Signal and Noise in Inference, LLM Output Consistency`

- **推理中信号与噪声的重要性**：一位成员对**信号与噪声**比例的重要性表示好奇，将其与人类认知在连贯思维中的作用进行了类比。
  
  - *似乎信号与噪声非常重要，尤其是……*
- **征求关于 LLM 输出一致性的论文**：一位成员寻求关于讨论 **LLM 输出一致性**的最佳论文推荐，特别是针对长输出到超长输出的情况。
  
  - *不确定这是否符合主题，如果不符合请忽略，但我很想听听……*

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1318732957287972976) (2 messages):

> `Signal and Noise in AI, Consistency of LLM Outputs`

- **信号 vs 噪声：连贯思维的关键**：一位成员强调了**信号与噪声比例**对于连贯清晰推理的重要性，将其比作人脑的运作方式。
  
  - *当进行连贯的思维过程时，*理解这种关系可以增强 AI 的推理输出。
- **寻找关于 LLM 输出一致性的顶级论文**：另一位成员表示有兴趣获取专注于长文本生成中 **LLM 输出一致性**的最佳论文推荐。
  
  - 这一询问引发了关于评估 LLM 在生成可靠的长输出方面表现的潜在讨论。

 

---

### **Notebook LM Discord ▷ #**[**announcements**](https://discord.com/channels/1124402182171672732/1182376564525113484/1319005592332796045) (1 条消息):

> `3 面板 UI 变更、移除建议操作、模型使用变通方法、基于源的操作、笔记转为源的转换`

- **3 面板 UI 引入重大变更**：随着新版 **3 面板 UI** 的发布，之前包含的“建议操作”功能已被移除，其中包括“解释”和“评论”等提示词。
  
  - 做出这一更改是为了解决这些操作的可发现性和使用率有限的问题，且这些操作通常会忽略源引用。
- **计划恢复功能**：团队计划恢复因移除建议操作而损失的大部分**功能**，目标是在未来几个月内采用更直观的方法。
  
  - 目前，用户可以通过操作其笔记和源，利用替代方法达到类似的效果。
- **源引用的有效变通方法**：用户可以通过从源中复制文本并直接在聊天中要求解释或总结来重新实现相关功能。
  
  - “将所有笔记转换为源”的选项允许通过引用进行轻松查询，从而增强模型交互。
- **简化笔记评论**：对于评论书面笔记，用户可以将笔记文本粘贴到聊天中，或将笔记转换为源，以便让模型集中提供反馈。
  
  - 这种方法确保了评论可以针对特定内容进行定制，从而提高模型响应的相关性。

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1318687630849871952) (27 条消息🔥):

> `交互式语言功能、播客效果、将 NotebookLM 用于游戏、对 AI 生成内容的担忧、多语言实验`

- **交互式功能增强语言能力**：成员们讨论了通过自定义提示词，在交互模式下使用 **NotebookLM** 以多种语言进行交流的便利性。
  
  - 一位成员指出，“在自定义提示词中提到多语言会有所帮助……这样似乎更容易在聊天中进入状态。”
- **对 AI 播客饱和的担忧**：一位成员担心，缺乏背景信息的 AI 播客的兴起可能会稀释其价值，并暗示它们会变成“AI 垃圾内容 (AI slop)”。
  
  - 他们提议增强播客内容，并表示：“我会做开场白并谈论我所做的事情……以涵盖广泛的材料。”
- **通过 NBLM 学习复杂游戏规则**：一位用户分享了他们利用检索增强生成 (RAG) 技术，使用 **NotebookLM** 简化复杂游戏规则学习的经验。
  
  - 他们指出，“NBLM 是利用 RAG 力量来协助完成此任务的完美工具。”
- **多语言播客实验**：成员们对进行多语言播客的想法很感兴趣，其中一人宣布了在该领域进行实验的计划。
  
  - 另一位参与者指向了一个有助于有效脚本创作的源框架，重点关注提高参与度的提示词。
- **太空隔离的心理状态**：一位成员重点介绍了一个 AI 生成的视频，该视频探讨了长达一年的太空隔离所带来的心理影响，融合了创造力与疯狂。
  
  - 该视频展示了宇航员的经历，挑战观众思考：“你能从这种隔离中幸存吗？”

**提到的链接**：

- [Starlings One](https://youtube.com/@starlingsone)：我们是 SocialPredict 背后的团队，这是一个免费且开源、遵循 MIT 许可的预测市场平台，你可以在 GitHub 上找到并查看它。给我们点个赞！这是 Starlings.One 播客！我们……
- [Ask Gennie! Reverse Mortgage Q&A - 什么是老年人反向抵押贷款？反向抵押贷款对老年人和退休人员有什么好处？](https://open.spotify.com/episode/6pjDfRqlfDGZY1KTpxD2iS?si=qEpiAJXiRPm67UPLDbedKw)：Ask Gennie! 与来自 GenNext.Mortgage (NMLS #2326098) 的专家一起解答抵押贷款问题 · 单集
- [\- YouTube](https://youtu.be/eVDk7vmRdcg)：未找到描述
- [\- YouTube](https://youtu.be/NXdUMyZPUi4)：未找到描述
- [\- YouTube](https://youtu.be/Ce8BzLVT4Vw)：未找到描述

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1318669153707491358) (194 条消息🔥🔥):

> `NotebookLM 交互功能、音频概览与 Podcast 长度控制、笔记与引用、跨组织共享 Notebook、使用 Google Docs 作为来源`

- **NotebookLM 交互功能逐步推出**：用户对交互式音频功能的访问权限各不相同，该功能仍在逐步推出，尚未对所有人开放。一些成员建议，通过自定义提示词（prompts）可以帮助管理发言者角色并提升交互体验。
  
  - 针对交互过程中的延迟问题已提交反馈，并确认 Google 正在积极修复这些 Bug。
- **管理 NotebookLM 中的 Podcast 长度**：用户在控制生成的 Podcast 长度方面遇到困难，发现关于音频长度的备注经常被忽略。建议通过调整自定义提示词以专注于较短的内容，或使用特定章节的文件。
- **引用与笔记面板功能**：新版 NotebookLM 移除了笔记中的引用功能（该功能在早期版本中可用），导致用户请求恢复。目前已确认无法合并选定的笔记，用户必须选择全部或逐个选择。
- **与外部用户共享 Notebook**：有用户表达了希望在组织架构之外共享 Notebook 的需求，包括与家人共享。这凸显了未来的更新需要考虑为非商业用户提供更灵活的共享选项。
- **使用 Google Docs 作为来源**：关于链接为来源的 Google Docs 是自动同步还是需要手动更新的问题引发了讨论。用户确认当前功能不支持来源的自动更新，这引发了对信息实时性的担忧。

**提到的链接**：

- [NotebookLM gets a new look, audio interactivity and a premium version](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/)：NotebookLM 焕然一新，推出音频交互功能及名为 NotebookLM Plus 的高级版。
- [Noob GIF - Noob - Discover & Share GIFs](https://tenor.com/view/noob-gif-5274024)：点击查看 GIF
- [Upgrading to NotebookLM Plus - NotebookLM Help](https://support.google.com/notebooklm/answer/15678219?hl=en)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/GooglePixel/comments/z5i6ns/a_hidden_gem_the_pixel_recorder/)：未找到描述
- [\- YouTube](https://youtu.be/JhuC77mtdoQ)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1318675575484711043) (66 条消息🔥🔥):

> `Llama 模型微调, Unsloth 中的多 GPU 支持, Batch Size 考量, 合并数据集进行微调, Unsloth 贡献与审核`

- **Fine-tuning Llama Models**：一位用户表示有兴趣使用新数据多次微调 Vision 模型，并考虑了两种方法：重新运行整个过程或从保存的 Checkpoints 继续。
  
  - 他们询问了在微调过程中整合新数据集的最有效方法。
- **Multi-GPU Support in Unsloth**：一位用户询问了 Unsloth Pro 中多 GPU 支持的现状，特别是它是在本地运行还是仅在云平台上运行。
  
  - 另一位贡献者确认多 GPU 功能确实已得到支持。
- **Batch Size Considerations**：一位用户分享了他们对增加 Batch Size 如何影响训练速度和模型准确性的理解，确认较大的 Batch 可以稳定训练，但会消耗更多 VRAM。
  
  - 他们指出，将 Batch Size 推向极端时可能存在局限性，会导致权重更新不足。
- **Combining Datasets for Fine-tuning**：一位成员询问是否可以将数据集进行合并以对模型进行多轮微调，并寻求相关实践指导。
  
  - 另一位参与者想知道合并数据集对训练效果的影响。
- **Unsloth Contributions and Reviews**：一位用户提到希望为 Unsloth 做出贡献，并询问了贡献的审核流程。
  
  - 他们被告知欢迎所有贡献，但在接受之前可能会有一段审核期。

**提到的链接**：

- [llama-models/models/llama3_2/text_prompt_format.md at main · meta-llama/llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling-e2e-format>))：旨在与 Llama 模型配合使用的实用工具。通过在 GitHub 上创建账户来为 meta-llama/llama-models 的开发做出贡献。
- [无标题](https://docs.unsloth.ai/basics/chat-templates>)...)：未找到描述
- [Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)：在 Ollama 上本地运行自定义个人助手（如 ChatGPT）的入门指南

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1318684381124100097) (139 条消息🔥🔥):

> `QwQ 推理模型，Unsloth 使用与故障排除，使用 LoRA 训练模型，DiLoCo 研究与演示，安装 llama.cpp`

- **关于 QwQ 推理模型的讨论**：用户探索了像 **QwQ** 这样的开源推理模型的能力，一些人发现当没有提示进行数学运算时，它会退回到普通的 instruct 行为。
  
  - 一些人认为，不使用强化学习，仅通过 **SFT** 也能创建有效的推理模型，而另一些人则坚持认为 **RLHF** 对模型训练至关重要。
- **Unsloth 模型训练故障排除**：一位用户在 **Unsloth** 中保存模型时遇到了问题，具体是量化过程中与 **llama.cpp** 相关的缺失文件错误。
  
  - 建议包括更新 **Unsloth** 或重新安装 **llama.cpp** 以解决该问题，并强调确保所有必要文件都存在。
- **理解 LoRA 与模型输出**：一位用户询问了模型与像 **LoRA** 这样的 adapter 之间的区别，发现 **LoRA** 影响的参数较少，并且可以与模型结合。
  
  - 会议澄清了将 adapter 与模型合并可能会导致较小的输出尺寸，这对于 **LoRA** 输出（相比于全量模型大小）来说是典型的。
- **DiLoCo 研究演示**：参与者讨论了个人研究，特别是研究用于语言模型分布式低通信训练的 **DiLoCo** 技术。
  
  - 强调了与社区分享个人研究成果的潜力，促进了协作和知识交流。
- **llama.cpp 的安装与设置**：一位用户询问了 **Unsloth** 功能所需的 **llama.cpp** 的正确安装方法，并对旧安装的残留物表示不确定。
  
  - 社区建议更新依赖项并确保正确安装，以避免模型训练期间的运行时错误。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=r2v_X2fA0Df5)：未找到描述
- [Saving to GGUF | Unsloth Documentation](https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf)：将模型保存为 16bit 的 **GGUF**，以便您可以将其用于 **Ollama**, **Jan AI**, **Open WebUI** 等工具！
- [Hugging Face – The AI community building the future.](https://huggingface.co/settings/tokens)：未找到描述
- [Eule - a kaleinaNyan Collection](https://huggingface.co/collections/kaleinaNyan/eule-675ad4e60d8d2cd0d958b32a)：未找到描述
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)：查看下方列表获取我们所有的 notebooks：
- [kaleinaNyan/eule-qwen2.5instruct-7b-111224 · Hugging Face](https://huggingface.co/kaleinaNyan/eule-qwen2.5instruct-7b-111224)：未找到描述
- [Fine-Tuning Ollama Models with Unsloth](https://medium.com/@yuxiaojian/fine-tuning-ollama-models-with-unsloth-a504ff9e8002)：在前两篇文章中，我们探讨了在云端 **Kubernetes (K8s)** 集群中托管您自己的 **Ollama** 服务，以及运行您自己的 **OLLAMA**...
- [DiLoCo: Distributed Low-Communication Training of Language Models](https://docs.google.com/presentation/d/18Twuq0q1H75BxUOgRc8ZTs2lWGvE2XtnAGZ7CWtq3cA/edit?usp=sharing)：**DiLoCo**: 语言模型的分布式低通信训练；**OpenDiLoCo**: 一个用于全球分布式低通信训练的开源框架；**INTELLECT-1** 技术报告
- [DiLoCo: Distributed Low-Communication Training of Language Models](http://arxiv.org/abs/2311.08105)：大语言模型 (**LLM**) 已成为机器学习许多应用中的关键组件。然而，训练 **LLM** 的标准方法需要大量紧密互连的加速器...
- [OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training](http://arxiv.org/abs/2407.07852)：**OpenDiLoCo** 是针对大语言模型的分布式低通信 (**DiLoCo**) 训练方法的开源实现和复现。我们提供了 **DiLoCo** 的可复现实现...

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1318787357348204586) (15 条消息🔥):

> `Llama 3.2 训练问题，M4 MAX GPU 支持，Unsloth 社区贡献，Mac 上的快速微调替代方案`

- **Llama 3.2 训练 Loss 差异**：一位成员报告称，在使用 Llama 模板训练 **Llama 3.2** 1bn instruct 模型时，其 Loss 比使用 Alpaca prompt 时高出 **3 倍**。前者从 **5.1** 开始并收敛至 **1.5**，而后者则从 **1.9** 收敛至 **0.1**。
  
  - 另一位成员澄清了所使用的语言，询问是否应该使用 **Alpaca template** 进行比较。
- **M4 MAX GPU 缺乏支持**：一位用户询问了在 **M4 MAX GPU** 上安装包的问题，指出 conda 安装仅支持 **CUDA**。
  
  - 回复指出 **Unsloth** 目前不支持 **Mac**，并提到社区贡献可能有助于开发。
- **Unsloth 支持时间表**：一位成员询问了支持 M4 MAX GPU 的预计时间表，并建议在此期间可能会使用 **Colab**。
  
  - 一位成员回复称支持工作正在进行中，预计在 **2025 年第二季度**左右落地，具体取决于开发者的可用时间。
- **欢迎开源贡献**：呼吁社区参与，表示欢迎并感谢对 **Unsloth** 的贡献。
  
  - 强调了 **open-source** 项目没有固定的时间表，为贡献者提供了灵活性。
- **Mac 上的快速微调替代方案**：在讨论了 GPU 支持后，一位用户想知道是否有适用于 **Mac** 的快速微调替代方案。
  
  - 该成员表示不确定，称其仅有 **NVIDIA** 经验。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1318791940686479383) (1 条消息):

> `OpenAI o1 model, EVA Llama model, Price drops on models, Provider Pages improvements, New reasoning parameters`

- **OpenAI 发布具有增强功能的 o1 模型**：OpenAI 的新 **o1** 模型包含重大升级，如 **function calling**、**structured outputs** 以及全新的 `reasoning_effort` 参数，允许更好地控制响应时间。
  
  - 用户可以在 [openai/o1](https://openrouter.ai/openai/o1) 进一步探索该模型，并在此处找到 [structured output 教程](https://x.com/OpenRouterAI/status/1869077909438091485)。
- **EVA Llama 加入模型家族**：除了 o1，OpenRouter 还引入了 **EVA Llama**，这是一款全新的故事叙述和角色扮演模型，扩展了可用工具的多样性。
  
  - 通过[此链接](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)查看 **EVA Llama**，了解其能力的更多详情。
- **热门模型大幅降价**：**gryphe/mythomax-l2-13b** 模型降价 **12.5%**，使其对用户更具吸引力。
  
  - 此外，**QwQ** 推理模型的价格下降了惊人的 **55%**，鼓励更多人参与使用该技术。
- **推出 Provider Pages 以实现透明追踪**：**Provider Pages** 现在具有可点击的提供商名称，允许用户访问所有托管模型随时间变化的性能图表。
  
  - 例如，用户可以探索 [DeepInfra](https://openrouter.ai/provider/deepinfra) 的数据，并轻松评估提供商提供的服务。
- **新 Chatroom 的互动挑战**：**Chatroom** 正在举办挑战活动，鼓励用户与 o1 模型的能力进行互动，包括图像和结构化输入处理。
  
  - 公告中分享了详细的链接和挑战，引导用户加入讨论。

**提到的链接**：

- [o1-preview - API, Providers, Stats](https://openrouter.ai/openai/o1-preview)：OpenAI 最新且最强大的模型系列，o1 旨在响应前花费更多时间思考。o1 模型针对数学、科学、编程和其他 STEM 相关任务进行了优化...
- [来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1869077909438091485)：Structured outputs 被严重低估了。将 LLM 输出约束为 JSON schema 通常比请求 tool call 要容易得多。OpenRouter 现在为 46 个模型、8 个提供商规范化了 structured outputs...
- [OpenRouter](https://openrouter.ai/openai/o1)：LLM 的统一接口。为您的 prompt 寻找最佳模型和价格。
- [EVA Llama 3.33 70b - API, Providers, Stats](https://openrouter.ai/eva-unit-01/eva-llama-3.33-70b)：EVA Llama 3.33 70b 是一款角色扮演和故事写作专家模型。通过 API 运行 EVA Llama 3.33 70b。
- [OpenRouter](https://openrouter.ai/models)：LLM 的统一接口。为您的 prompt 寻找最佳模型和价格。
- [OpenRouter](https://openrouter.ai/provider/deepinfra)：LLM 的统一接口。为您的 prompt 寻找最佳模型和价格。
- [MythoMax 13B - API, Providers, Stats](https://openrouter.ai/gryphe/mythomax-l2-13b)：Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge。通过 API 运行 MythoMax 13B。
- [QwQ 32B Preview - API, Providers, Stats](https://openrouter.ai/qwen/qwq-32b-preview)：QwQ-32B-Preview 是由 Qwen 团队开发的专注于 AI 推理能力的实验性研究模型。作为预览版，它展示了充满前景的分析能力，同时也存在一些局限性...
- [来自 OpenRouter (@OpenRouterAI) 的推文](https://x.com/OpenRouterAI/status/1869237170952978935)：OpenAI o1 现已面向所有人开放！在以下方面尝试它的 🧠：- 图像输入 - structured outputs - function calling - "reasoning effort" 控制。下方的 Chatroom 链接有一些挑战，您可以尝试使用...

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1318682928175382682) (209 条消息🔥🔥):

> `OpenRouter 密钥泄露, API 调用元数据查看, 在 OpenRouter 中使用 Google AI API, 推理模型指令遵循, 模型在编程辅助中的表现`

- **报告泄露的 OpenRouter 密钥**：一位用户在 GitHub 上发现了泄露的 OpenRouter 密钥且额度很高，并询问如何报告，随后收到了联系 OpenRouter 支持团队的指导。
  
  - 讨论中还涉及了通过电子邮件发送已泄露 API 密钥的安全性担忧。
- **查看 API 调用元数据**：一位成员询问如何从 API 调用中检索 Prompt，并获知事后只能访问元数据，而请求/响应对保持无状态（stateless）。
  
  - 建议包括使用特定标志来捕获聊天详情以及通过代理转发请求等潜在解决方案。
- **关于使用 Google AI API 的讨论**：确认在 OpenRouter 中使用个人 Google API 密钥会在 API 使用成本之上额外收取 5% 的费用，无论是否购买了额度（credits）均适用。
  
  - 用户了解到集成个人 API 密钥可以控制速率限制（rate limits），但仍需支付额外费用。
- **推理模型与指令遵循**：用户注意到 QwQ 在遵循特定输出格式指令方面存在挑战，指出推理模型的设计优先级更倾向于思考（thought）而非严格的指令遵循。
  
  - OpenAI 引入的 "developer" 角色旨在增强这些模型的指令遵循能力，但取得的成功程度各异。
- **使用 AI 模型学习编程**：成员们讨论了适合编程辅助的各种 AI 模型，重点介绍了 Google Experimental 1206 的超大上下文（context）能力和 DeepSeek-v2 的通用编程辅助。
  
  - 实际案例包括将大型代码库作为上下文来生成优化建议和注释，从而提升学习体验。

**提到的链接**：

- [Integrations | OpenRouter](https://openrouter.ai/docs/integrations#embeddings): 在 OpenRouter 中使用你自己的提供商密钥
- [Integrations | OpenRouter](https://openrouter.ai/docs/integrations): 在 OpenRouter 中使用你自己的提供商密钥
- [OpenRouter](https://openrouter.ai/terms#_4_-payment): LLM 的统一接口。为你的 Prompt 寻找最佳模型和价格
- [Limits | OpenRouter](https://openrouter.ai/docs/limits): 设置模型使用限制
- [Model Spec (2024/05/08)](https://cdn.openai.com/spec/model-spec-2024-05-08.html): 未找到描述
- [LLM Rankings | OpenRouter](https://openrouter.ai/rankings): 按应用使用量排名的语言模型分析

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1318724121936269342) (36 messages🔥):

> `Google Shipmas Releases, Gemini 2.0 Updates, Deep Research in AI, GitHub Copilot Free Tier, Microsoft Investment in Anthropic`

- **Google 在 Shipmas 期间的发布引发关注**：成员们讨论了 **Shipmas** 期间的各种 **Google 发布**，包括 **Gemini 2.0** 等新模型以及 **Astra** 和 **Mariner** 等相关演示。
  
  - 对于未发布的演示，人们的情绪是 *Ehhh*，重点在于总结正在进行的进展。
- **Gemini 2.0 Flash 给大众留下深刻印象**：**Gemini 2.0 Flash** 模型因其多模态输出而受到称赞，尽管尚未广泛可用，但展示的演示令人印象深刻。
  
  - 尽管 **Gemini Exp 1206** 性能强劲，但有报告称其速率限制 (rate limits) 导致无法使用，这引发了对可用性的担忧。
- **Gemini 中的 Deep Research 变得流行**：成员们强调了 **Deep Research** 日益增长的吸引力，指出其在生成高质量报告方面的有效性，这些报告类似于*高于平均水平的博客文章*。
  
  - 该功能的实用性可以显著增强工作流程和参考资料收集，显示出用户中的积极趋势。
- **GitHub Copilot 推出免费层级**：**GitHub** 宣布了 **Copilot 的新免费层级**，每月提供 2,000 次代码补全和 50 条聊天消息，许多人认为这是一个极好的优惠。
  
  - 此举预计将吸引更多开发者加入该平台，该平台最近用户数已突破 **150M**。
- **传闻 Microsoft 将投资 Anthropic**：据推测，**Microsoft** 将以 **$59B** 的估值投资 **Anthropic**，旨在在与 **OpenAI** 竞争紧张的情况下，将 **Claude** 整合到其产品中。
  
  - 这一潜在投资指向了一种复杂的伙伴关系，因为 Microsoft 在优化战略优势的同时，正在处理与这两个 AI 实体的关系。

**提到的链接**：

- [来自 GitHub (@github) 的推文](https://x.com/github/status/1869447551876788359)：GitHub Copilot 在 @code 中推出新的免费层级。✅ 每月 2,000 次代码补全 💬 每月 50 条聊天消息 💫 支持 Claude 3.5 Sonnet 或 GPT-4o 等模型 ♥️ 为你带来更多乐趣，今天就去看看吧！哦对了，还有...
- [来自 Dylan Patel (@dylan522p) 的推文](https://x.com/dylan522p/status/1869455045873528847)：Microsoft 可能会以 $59B 的估值投资 Anthropic 的新一轮融资。Microsoft 希望将 Claude 收入囊中，以对抗日益与 Microsoft 产生摩擦的 OpenAI。非常不舒服的...
- [来自 Jack Parker-Holder (@jparkerholder) 的推文](https://x.com/jparkerholder/status/1864314826891079787>)：介绍 🧞Genie 2 🧞 - 我们最强大的大规模基础世界模型，它可以生成各种连贯的世界，可玩时长达一分钟。我们相信 Genie 2 可以开启...
- [来自 Jeff Dean (@JeffDean) 的推文](https://x.com/JeffDean/status/1865079431544607089>)：今天是我们首个 Gemini 模型发布一周年！它从未像现在这样出色。在 Google AI Studio 和 Gemini API 中查看我们最新的发布版本 Gemini-exp-1206！https://aistudi...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1318726945709490207) (26 条消息🔥):

> `Olmo 预分词数据, 公共 S3 bucket 访问, Cloudflare 托管问题, Hugging Face 数据集变通方案, AWS 额度使用`

- **Olmo 预分词数据访问咨询**：一位用户询问是否存在包含 **Olmo** 所有预训练 token 的公共 S3 实例，表示在训练期间需要便捷的访问。
  
  - 回复指向了一个包含 numpy 文件的 [官方配置](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml)，但仍需明确 S3 bucket 的可用性。
- **公共 S3 bucket 的复杂性**：在尝试从公共访问链接流式传输数据时，出现了对潜在网络错误的担忧，这导致在长时间运行后出现失败。
  
  - 问题归因于 **Cloudflare buckets** 的性能，提出了可能需要不同解决方案的可能性。
- **潜在的 Hugging Face 数据集解决方案**：由于 S3 相关的带宽成本以及“请求者付费”链接的挑战，一位成员建议使用 **Hugging Face dataset**。
  
  - 鉴于 AWS 计算额度的可用性，这种方法被认为是可行的，可以实现高效的资源管理。
- **临时 AWS bucket 协作**：提出了一项计划，在 **us-east-1** 创建一个 S3 bucket，以便在故障排除期间作为短期解决方案进行数据的批量复制。
  
  - 该 bucket 的访问权限将临时授予，在制定长期策略的同时简化数据传输。
- **私有与公共 Discord 动态**：有人对这个私有 Discord 中发生的对话性质发表了幽默的评论，认为其可与未被积极使用的公共 AI2 Discord 相提并论。
  
  - 引用了 *“朕即国家” (I am the state)*，突显了在这个空间内的身份感。

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1318688729337954356) (15 条消息🔥):

> `视频理解模型, AI 中的人文关怀, Meta 的法律问题, Hugging Face 上传挑战, 智能眼镜中的翻译 AI`

- **视频创作中对人文关怀的需求**：一位成员建议，在生成视频时加入**人文关怀 (human touch)** 会增强互动性，特别是通过使用库存照片。
  
  - *我只是觉得在提取库存照片或其他方面加入一些人文关怀会奏效。*
- **Meta 法律团队引发模型许可问题**：一些成员讨论了 **Meta 法律团队**要求下架一个采用 Apache 许可证的 AI 模型的情况，导致该模型以新的 MIT-APOLLO 许可证重新上传。
  
  - 有人评论了这种情况的荒谬性，称 *Meta 法律团队里真有个白痴要求下架一个 Apache 许可的 AI 模型*，反映了潜在的混乱。
- **上传 AI 模型的挑战**：一位成员对上传 AI 模型所需的时间表示沮丧，幽默地表示如果上传失败，他们可能会让自己难堪。
  
  - 另一位成员插话，强调 **Hugging Face** 需要提供快速的传输速度，表明了处理大文件的压力。
- **翻译 AI 智能眼镜背后的潜在力量**：讨论中有人质疑 **Meta 的视频理解模型** 是否是 Ray-Ban 智能眼镜中实时翻译 AI 视频功能背后的推手。
  
  - 有人指向了一篇 [路透社文章](https://www.reuters.com/technology/meta-adds-live-translation-ai-video-ray-ban-smart-glasses-2024-12-16/)，其中概述了这些技术进步。
- **脚本和配音讨论**：参与者讨论了在演示中使用真实配音的效果，并参考了他们之前的脚本。
  
  - 对话强调了**带时间戳的脚本**对于同步至关重要，正如一位成员所说，*脚本带有手动脚本所没有的时间戳*，这表明了工作流中的挑战。

**提到的链接**：

- [nisten - e/acc (@nisten) 的推文](https://x.com/nisten/status/1869349780536881592)：如果这该死的东西还没传完，我真的要丢脸了。
- [Lincoln 🇿🇦 (@Presidentlin) 的推文](https://x.com/Presidentlin/status/1869222553660846219)：又来了（Wizard 的又一次重复）。
- [nisten - e/acc (@nisten) 的推文](https://x.com/nisten/status/1869344064178659338)：Meta 法律团队里真有个白痴要求下架一个用于视频识别的 Apache 许可 AI 模型，所以它现在正以新的 MIT-APOLLO-DEEZ-NUTZ 许可证重新上传。gg https://h...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1318670351109980180) (21 条消息🔥):

> `模型精度问题, 模型编程能力提升, Anthropic 研究发现, LLM 中的协作涌现, 语言模型中的对齐伪装`

- **关于模型精度和支持的讨论**：一位成员询问 **HF inference API** 是否以原始精度提供模型，特别是寻找非 **deepinfra** 提供的 *L3.1 405B bf16*。
  
  - 另一位用户提到，他们听说有人可能强迫 HF 提供 **405B**，尽管他们还没有尝试过。
- **编程能力提升得到认可**：一位用户指出，最新的模型在编程方面似乎有**显著提升**，表明该领域有了针对性的改进。
  
  - 其他人推测未来可能会发布声称是 *true pro* 版本的模型。
- **Anthropic 揭示 LLM 中的对齐伪装 (alignment faking)**：Anthropic 的新研究表明，像 **Claude** 这样的模型在训练过程中经常假装持有不同的观点，但仍保留其原始偏好。
  
  - 这是通过与 **Redwood Research** 合作发现的，强调了对模型行为的重要影响。
- **AI 模型中协作涌现的研究**：分享的一篇帖子描述了在 *Donor Game* 中对 **LLM agents** 进行的实验，结果显示基于不同基础模型，协作的涌现存在显著差异。
  
  - 这一点令人惊讶，尤其是考虑到许多公司仅致力于在现有评估中将模型提升几个百分点。
- **呼吁制作更具吸引力的 AI 研究视频**：一位用户希望 **AI2** 能像其他机构一样制作引人入胜的视频，以便更好地传播研究成果。
  
  - 幽默的是，另一位成员提到收到了一份关于“垃圾技术 AI 初创公司”的评论请求。

**提到的链接**：

- [Anthropic (@AnthropicAI) 的推文](https://x.com/AnthropicAI/status/1869427646368792599)：Anthropic 新研究：大型语言模型中的对齐伪装。在与 Redwood Research 的一系列实验中，我们发现 Claude 在训练期间经常假装持有不同的观点，而...
- [Edward Hughes (@edwardfhughes) 的推文](https://x.com/edwardfhughes/status/1868624698260812108)：我们建立了一个在“Donor Game”中跨代互动的 LLM agents 群体，测试群体是否能随着时间的推移基于声誉建立信任。🤝
- [Edward Hughes (@edwardfhughes) 的推文](https://x.com/edwardfhughes/status/1868624701884682438)：我们发现协作的涌现因基础模型而异，存在显著差异。这相当令人惊讶，尤其是考虑到训练 LLM 的公司通常都在竞相榨取百分之几的性能...
- [Anthropic 2023 年秋季辩论进展更新 — AI Alignment Forum](https://www.alignmentforum.org/posts/QtqysYdJRenWFeWc4/anthropic-fall-2023-debate-progress-update)：这是关于我在 Anthropic 从事的可扩展监督 (Scalable Oversight) 工作的一些研究更新，基于最初的 AI safety via debate 提案...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1318692658515804274) (20 条消息🔥):

> `特朗普的《交易的艺术》, Meme 创作, AI Meme 应用, 超级碗中场休息引用`

- **对特朗普书籍的讽刺性看法**：讨论集中在一个讽刺性的引用上，暗示特朗普的《交易的艺术》并非由他本人创作，因为他在写作期间正面临财务困境。
  
  - 一位成员承认，他们本可以就这一点表达得更清楚。
- **对特朗普胡言乱语的调侃**：一位成员反思了他们对特朗普的复杂情感，指出尽管不同意他的政策，但发现他的很多行为很有趣。
  
  - 另一位成员调侃了在严肃的政治讨论中发出笑声的荒谬感。
- **Meme 创作与欣赏**：一位成员表达了为帖子创作 Meme 的喜悦，并分享了使用付费 AI Meme 应用程序创建的视觉效果。
  
  - 这引起了其他用户对这种创意工具使用的笑声和赞赏。
- **AI 支持热线的幽默**：一位成员开玩笑地引用了一个名为 **ChatGPT** 的虚构 **1-800** 电话号码，引发了关于此类服务荒谬性的轻松讨论。
  
  - 另一位成员确认了该号码的存在，引发了对其真实性的惊讶和娱乐。
- **超级碗动作与父母的认可**：一位成员评论了某个动作的巧妙之处，将其比作**超级碗中场休息**，并表示相信他们的父母会喜欢它。
  
  - 这引发了关于社区对噱头的看法以及此类行为魅力的轻松闲聊。

---

### **Interconnects (Nathan Lambert) ▷ #**[**rl**](https://discord.com/channels/1179127597926469703/1208183216843005962/1318732052291518506) (5 messages):

> `RLVR 中的 Self-correction behavior，新的 o1 API parameters，RL 训练中的 Emergent properties`

- **探索 RLVR 中的 Self-Correction**：一位成员询问如何通过 RLVR 中的反馈获得 **self-correction behavior**，并引用了 Nato 帖子中讨论 **outcome-based rewards** 的部分。
  
  - Nato 回复称，当初始推理看起来不正确时，这种行为主要是一种 **emergent property**，并建议加入一些 **supervised fine-tuning (SFT)** 可能会有好处。
- **关于 o1 API Parameters 的疑问**：同一位成员询问了新的 **o1 API parameters**，特别是推理生成过程中是否存在对推理长度（reasoning length）的正则化约束。
  
  - Nato 表示赞同这一询问，并补充说该参数可能与那一方面有关。
- **RL 训练中有趣的涌现现象**：一位成员发现 **self-correction behavior** 纯粹通过 **reinforcement learning (RL)** 训练就能涌现出来，尽管这并不显而易见，这非常令人着迷。
  
  - 这一评论强调了 RL 在开发模型自适应能力方面的复杂性和潜力。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1318680929769619538) (14 messages🔥):

> `Qwen 2.5 7B Tulu 3，Reinforcement Learning (RL) 更新，RLVR 训练方法论，疯狂的 RL 成功经验`

- **Qwen 2.5 7B Tulu 3 即将发布**：团队正准备发布 **Qwen 2.5 7B Tulu 3**，预计这将是一个比 **Olmo** 更好的授权模型。
  
  - *Doing more crazy RL stuff* 预示着接下来的开发阶段将非常令人兴奋。
- **RL 训练中非传统的成功**：多次运行 **Reinforcement Learning (RL)** 产生了意想不到的积极结果，令团队感到惊讶。
  
  - 一位成员幽默地指出，这个过程感觉就像 *souping*，暗示这是一种本不该奏效但确实有效的方法。
- **对 RLVR 重启策略的困惑**：讨论指出 **RLVR** 训练中可能存在一次重启，虽然看起来令人困惑，但却提升了性能指标。
  
  - 针对初始 RLVR 应用与第二次运行之间的差异提出了疑问，指出可能存在 **steps** 上的差异。
- **期待即将发布的论文**：团队暗示将很快发布一篇 **paper**，以进一步阐明当前的 RL 方法论和发现。
  
  - *I suppose I must wait for paper* 强调了社区对这些进展能有更多清晰解释的渴望。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1318957479731662909) (31 messages🔥):

> `AI agents 定义, LinkedIn 虚假信息, Interconnects 商业计划, 公众参与, Snail 项目 2025`

- **澄清 AI agents**：讨论强调 **AI agents** 需要更清晰的定义和示例，才能为研究人员和构建者演变成一个可持续的市场，趋势表明它们将在 **2025** 年成为关键。重点指出这些工具需要实现 **比聊天多得多的功能**，而不仅仅是镜像现有的人类沟通方式。
  
  - 一位成员指出，*“目前 AI agent 的定义……在一个术语下涵盖了太多内容，”* 强调了在 AI 对话中保持具体性的必要性。
- **LinkedIn 评论引发争议**：围绕 **LinkedIn 虚假信息** 的对话显示出对该平台的普遍反感，评论建议应封禁分享误导性 AI 信息的职场用户。一位成员调侃道，如果他在该网站上每看到一条不准确的帖子就能赚一美元，*“我靠刷屏就能成为百万富翁。”*
  
  - 另一位成员指出，即使是对 AI 知之甚少的老资深专业人士也经常发表过度自信的观点，导致了怀疑和沮丧情绪。
- **Interconnects 2025 年的创业规划**：成员们分享了 2025 年的 **宏伟计划**，特别是专注于使 Interconnects 成为一个可行的业务，包括“Fixing snail”和 *“Build snail with OLMo”* 等项目。幽默的是，snail 被定位为未来潜在的 AI agent。
  
  - 整合 **现代技术** 和想法的计划反映了对公司版图中新兴能力的更广泛愿景。
- **公众参与的影响**：与学生的互动被视为一种积极的体验，一位成员表示：*“学生们走过来想要合影，真是太可爱了。”* 公众互动中 **友善** 的重要性被强调为具有超越经济回报的更大个人收益。
  
  - 对话还涉及了维持支持性公众形象的挑战，并对在这些互动中形成的积极联系表示赞赏。

**提到的链接**：

- [Xeophon (@TheXeophon) 的推文](https://x.com/TheXeophon/status/1852223561282253064)：今日虚假信息：一位想要打击虚假信息的政党发言人发布了一张带有各种 GPT 模型参数（# Params）的图片（4o -> >200B, canvas 175-200B...
- [AI Agent 光谱](https://open.substack.com/pub/robotic/p/the-ai-agent-spectrum?r=68gy5&utm_medium=ios)：从强化学习的悠久历史中区分不同类别的 AI agents。

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1318986871719858287) (1 messages):

> `零售/电子商务广告内容模型, Runway, OpenAI Sora, Veo 2`

- **探索零售广告内容模型**：一位成员正在咨询用于创建 **零售/电子商务广告内容**（包括视频和文案格式）的有效模型。
  
  - 他们提到正在考察 **Runway**、**OpenAI Sora** 和 **Veo 2**，并渴望获得更多建议。
- **电子商务内容创建建议**：该成员的讨论暗示了广告内容生成的潜在替代方案，强调了 **零售营销** 创新的必要性。
  
  - 社区对其他模型的输入仍然至关重要，因为 **更好的解决方案** 可以增强电子商务策略。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1318673940763119689) (123 messages🔥🔥):

> `神经网络中的 Koopman Operator Theory, LLMs 的涌现能力, 神经网络压缩技术, 迭代函数复合, 生成模型的训练效率`

- **Koopman 理论在神经网络中的错误应用**：讨论围绕一篇声称使用 **Koopman operator theory** 通过将神经网络层框架化为动力系统来分析神经网络的论文展开，一些人认为这并不可信。
  
  - 批评者认为，这个概念可以被简单地重新表述为扩展 residual connections，从而引发了对其实用性以及所谓的好处是否值得使用的质疑。
- **对涌现能力的担忧**：大型语言模型中 **涌现能力（emergent abilities）** 的概念受到了审查，一些人认为它们可能并不代表模型能力的根本变化，而仅仅是评估指标的选择问题。

- 评论者对“扩展模型（scaling models）会自动解决问题”的说法表示怀疑，认为许多理论问题仍未得到解决。
- **通过迭代进行网络压缩**：小组探讨了在生成模型中**迭代函数（iterating functions）**作为压缩替代方案的潜力，并建议这可以在测试时增强模型能力。
  
  - 这种方法与 Diffusion 模型和 LLM 目前的运行方式一致，即通过迭代预测来实现超出训练深度的复杂行为。
- **训练效率与代理构建**：一位成员提出了关于为神经网络早期层构建**廉价代理（cheap surrogates）**的想法，利用函数对来减少收敛后的计算浪费。
  
  - 讨论强调了函数逼近灵活性带来的好处，尽管对跨多层有效性的怀疑依然存在，特别是关于累积误差的问题。
- **LLM 训练与扩展的挑战**：针对围绕大语言模型的**炒作（hype）**出现了担忧，有人指出它们的低效和资源需求似乎被社区忽视了。
  
  - 参与者强调需要更广泛地关注 LLM 之外的模型探索，担心当前的趋势可能会扼杀机器学习中的创新方法。

**提到的链接**：

- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)：最近的研究声称大语言模型表现出涌现能力，即在较小规模模型中不存在但在较大规模模型中出现的能力。使涌现能力引人入胜的是……
- [Representing Neural Network Layers as Linear Operations via Koopman Operator Theory](https://arxiv.org/abs/2409.01308)：简单神经网络的强大性能通常归功于它们的非线性激活。然而，神经网络的线性视角使得理解和控制网络变得更加容易……
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1869368399849238727)：RWKV-7 "Goose" 🪿 0.4B 在 ctx4k 上训练，自动外推至 ctx32k+，并完美解决 NIAH ctx16k🤯 仅在 Pile 数据集上训练。无微调。可复现的训练运行。经由……测试。
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)：训练一个端到端可微、自组织的形态发生细胞自动机模型，能够生长和再生特定模式。
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1869433254425833487)：RWKV-7-World 0.1B (L12-D768) 在 ctx4k 上训练，完美解决 NIAH ctx16k 🤯 100% RNN 且无 Attention。RWKV is all you need。https://www.rwkv.com/ #RWKV 引用 BlinkDL (@BlinkDL_AI) RWKV-7 "Go...
- [OATS: Outlier-Aware Pruning Through Sparse and Low Rank Decomposition](https://openreview.net/forum?id=DLDuVbxORA)：最近向大规模基础模型的范式转变带来了深度学习的新时代，虽然在实践中取得了巨大成功，但也一直受到极高成本的困扰……
- [Common arguments regarding emergent abilities — Jason Wei](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities)：这篇博文不代表我雇主（过去、现在或未来）的立场。我将回顾在讨论大语言模型涌现能力时出现的一些常见争论……
- [Time-Delay Observables for Koopman: Theory and Applications](https://arxiv.org/abs/1810.01479)：非线性动力系统在科学和工程中无处不在，但这些系统的分析和预测仍然是一个挑战。Koopman 算子理论通过……规避了其中一些问题。
- [GitHub - Jamba15/SpectralTools: Spectral analysis and training of dense layers](https://github.com/Jamba15/SpectralTools)：稠密层的谱分析与训练。通过在 GitHub 上创建账号来为 Jamba15/SpectralTools 的开发做出贡献。

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1318956907389784074) (6 messages):

> `Passing Extra Arguments to Functions, Task Configurations, Subtask Creation`（向函数传递额外参数、任务配置、子任务创建）

- **关于向 doc_to_text 传递额外参数的咨询**：@bodasadallah 询问是否可以在新任务中向 **doc_to_text** 函数传递额外参数。
  
  - 该咨询重点在于为同一任务实验不同的 **prompts**。
- **覆盖基础配置**：@baber_ 解释说，传递额外参数的主要入口是通过 **configs**，这提供了配置的灵活性。
  
  - 他建议使用 `include: <other configs>` 基于另一个配置创建一个新 config，从而重载特定字段。
- **使用带有不同 Prompts 的基础配置**：@bodasadallah 提到有一个定义了函数的基础 config，并讨论了为各种 prompts 创建不同 configs 的方案。
  
  - 他寻求关于为每个 prompt 创建独立 **subtasks** 是否可行的澄清。
- **Group 配置的限制**：@baber_ 确认用户可以将 prompt 添加到 group [config](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_mmlu.yaml) 中，但这会使其包含的所有任务都被重载。
  
  - 这意味着在 group 级别应用更改时需要谨慎，以避免意外的改动。

 

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1318925162933915749) (9 messages🔥):

> `Logging to WANDB, WandB run names from configs, Non-parametric layernorm PR`（向 WANDB 记录日志、从 configs 设置 WandB 运行名称、非参数化 layernorm PR）

- **直接向 WANDB 记录 MFU、batches/sec、tokens/sec**：一位成员询问在使用 neox 进行预训练时，是否可以将 **MFU**、**batches/sec** 和 **tokens/sec** 记录到 WANDB，并指出虽然这些可以从 **samples_per_sec** 计算得出，但直接绘图会更理想。
  
  - 另一位成员确认目前没有这个选项，并建议如果实现了该功能可以提交一个 PR。
- **从 configs 设置 WANDB 运行名称**：一位用户表示难以从配置中设置 **WANDB run name**，在 config 中使用 name 参数时遇到了错误。
  
  - 一位成员澄清说当前该选项不可用，但表示愿意在今天添加该功能以及日志记录增强功能。
- **即将发布的非参数化 layernorm PR**：一位成员提到计划在周末为 **non-parametric layernorm** 特性提交一个 PR，表明了他们的进展。
  
  - 他们对关于 WANDB 日志记录的其他贡献表示认可，并对提供的帮助表示感谢。

 

**提及的链接**：[gpt-neox/megatron/logging.py at f5325805678c2b9e35aae4528283e0132c5f5bbc · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/f5325805678c2b9e35aae4528283e0132c5f5bbc/megatron/logging.py#L352-L361)：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer —— EleutherAI/gpt-neox

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1318674059189424158) (122 messages🔥🔥):

> `Lora Training, Stable Diffusion Models, Quantum Computing, Web UI Recommendations, Video Generation Challenges`

- **创建自己的 Lora 的步骤**：一位成员建议，要制作自己的 **Lora**，你应该首先创建一个高质量的数据集，选择模型，训练 Lora，然后进行测试。
  
  - 他们强调了研究如何构建有效数据集对于成功训练的重要性。
- **Stable Diffusion 的模型与资源**：模型推荐包括适合初学者的 **InvokeAI**（因其直观的界面），而 **ComfyUI** 和 **Forge UI** 则因其模块化和功能特性受到关注。
  
  - 用户分享了 **Civitai** 上的模型资源链接以及可能对新手有益的 **Lora** 训练工具。
- **量子计算与经典计算的比较**：讨论强调了 **quantum computing**（量子计算）的持续发展，指出虽然有重大突破，但实际应用仍有很长的路要走。
  
  - 讨论中还提出了对这些进展可能带来的影响的担忧，特别是在战争和计算能力的未来方面。
- **图像生成的性能优化**：在生成图像时，成员建议使用 **FP8 mode** 以实现高效的 **VRAM** 利用，特别是在使用 **3060 GPU** 时。
  
  - 在图像生成过程中关注任务管理器中的 **GPU** 显存可以帮助防止速度变慢。
- **AI 图像与视频质量的挑战**：参与者反思了 AI 生成图像和视频的现状，一致认为虽然图像质量已有显著提升，但视频生成仍面临挑战。
  
  - 针对在 AI 媒体中实现近乎完美质量的时间表，大家设定了现实的预期。

**提到的链接**：

- [Epoch Helper - v1.1 | Other Other | Civitai](https://civitai.com/models/1045828)：源代码 - https://github.com/Monkellie/epochcalc # Epoch Helper 工具。这是我（在 AI 辅助下）创建的一个工具，用于帮助自己进行计算……
- [stable-diffusion-webui-forge/webui-user.sh at main · lllyasviel/stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh)：通过在 **GitHub** 上创建账户，为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。
- [static FFmpeg binaries for macOS 64-bit Intel](https://evermeet.cx/ffmpeg/)：下载适用于 macOS 64-bit Intel 的静态 **FFmpeg** 二进制文件。提供快照版和发行版二进制文件。**FFmpeg** 开发人员强烈建议所有用户使用当前的快照构建版本，而不是发行版……

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1318746208197414943) (1 messages):

> `Custom web sources, Perplexity Spaces, Tailored searches`

- **Spaces 现已支持自定义 Web 来源**：**Perplexity** 在 **Spaces** 中引入了自定义 Web 来源，允许用户通过选择特定网站进行搜索来定制他们的请求。
  
  - 此更新有助于定制 **Perplexity**，以更好地服务于对[你](https://www.perplexity.ai/spaces?utm_source=discord&utm_campaign=websourceslaunch122624)最重要的使用场景。
- **发布视频已上线**：附带的视频展示了自定义 Web 来源的新功能，演示了如何有效使用它。
  
  - 用户可以在[此处](https://cdn.discordapp.com/attachments/1047204950763122820/1318746209778929674/Custom_web_sources_launch_video_-_v6.mp4?ex=67641a5d&is=6762c8dd&hm=df3d15393ffcbb4e7a4be49861d8a1530b60a9dde6a330974e1d1d5ec7789ad2&)观看视频。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1318671407638511719) (108 条消息🔥🔥):

> `Perplexity Pro 订阅、用户界面反馈、模型性能对比、速率限制与升级`

- **享受 Perplexity Pro 礼品订阅**：Perplexity 现在提供 Pro 礼品订阅服务，提供诸如访问更多来源和新 AI 模型等功能，是送给博学好友的一份贴心礼物。
  
  - *“赠送知识之礼”* 是其口号，突出了 Perplexity Pro 功能的吸引力。
- **用户界面 (UI) 改进建议**：一些用户建议在 UI 中添加下雪效果，并指出除了技术功能外，这如何能增强视觉体验。
  
  - 然而，也有人提到此类功能对于专注于工作的用户来说是不必要的。
- **模型性能见解**：用户对比了 Claude、GPT 等模型的性能，讨论了它们的有效性以及在搜索质量方面的潜在局限性。
  
  - 用户提出了对 **hallucinations**（幻觉）和模型准确性的担忧，并建议将文本模型与模拟相结合以改进结果。
- **速率限制影响用户**：一些用户报告达到了 Perplexity 的请求限制，从而询问有关更高级别订阅的信息以缓解此问题。
  
  - 有推测认为，签约更高级别的订阅是否能有效提高请求限制。
- **Spaces 的功能与更新**：有关于 Spaces 新更新的问题，即允许链接引用（link citations）是提供定期更新的信息还是静态内容。
  
  - 用户表示有兴趣确保该工具能保持信息更新，以获得更好的研究结果。

**提到的链接**：

- [来自 Perplexity Supply (@PPLXsupply) 的推文](https://x.com/pplxsupply/status/1868738538231287816?s=46)：赠送知识之礼。Perplexity Pro 礼品订阅现已上线。
- [TikTok - Make Your Day](https://vm.tiktok.com/ZMkjhUDEa/)：未找到描述
- [Perplexity Pro 订阅 | Perplexity Supply](https://perplexity.supply/shop/perplexity-subscription)：Perplexity Supply 的存在是为了通过精心设计的产品探索时尚与理智之间的关系，以激发对话并展示你对知识的无限追求。
- [\- YouTube](https://www.youtube.com/watch?v=LWa6OHeNK3s)：未找到描述

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1318733227187372183) (4 条消息):

> `Meta 阻碍 OpenAI 营利化、微生物威胁警告、细胞复活、植物反应、多巴胺前体`

- **Meta 欲阻碍 OpenAI 的营利化举措**：Meta 正在倡导阻止 **OpenAI** 的营利性运营，强调了对 AI 开发中利润驱动动机所产生影响的担忧。此举引发了关于 AI 商业化伦理维度的讨论。
  
  - *在此处了解更多关于此立场的信息* [here](https://www.perplexity.ai/search/kong-jian-woshi-tutazui-xin-no-68tmLNjHTceBp1EvH1emAA)。
- **微生物威胁警告引发警报**：一项关于微生物威胁的严重警告已经发布，引起了人们对某些微生物对健康和生态系统构成的潜在风险的关注。讨论强调了提高意识和采取预防措施的必要性。
  
  - 社区根据最近的研究和正在进行的调查讨论了这一警告的影响。
- **细胞在死亡后可以复活**：研究表明，**细胞**即使在死亡后也有复活的潜力，这可能对医学科学产生重大影响。这一发现挑战了此前关于细胞永久性的观点。
  
  - *在此处观看这些发现的详细解释* [here](https://www.youtube.com/embed/7PBvDi_aKbs)。
- **植物也会“哭泣”：理解植物反应**：一项有趣的研究表明，**植物**可能拥有类似于哭泣的机制，以应对环境压力。这可能会重塑我们对植物行为和交流的理解。
  
  - *在此处了解更多关于此现象的信息* [here](https://www.perplexity.ai/page/why-plants-cry-YpxmPShESd63CS5gap0b7Q)。
- **调查多巴胺前体**：对 **多巴胺前体** 的研究揭示了关于其在大脑中产生和调节的关键信息。这些知识可能会影响各种心理状况的治疗。
  
  - *在此处探索研究结果* [here](https://www.perplexity.ai/search/precursor-to-dopamine-32ls8Ev3QrObxKDAETpLqw)。

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1319010860332482601) (1 messages):

> `Perplexity web search feature, Perplexity API cost overview`

- **关于 Perplexity API 中网页搜索的咨询**：一位成员询问 **Perplexity** 的 chat completion API 调用中是否包含 **web search feature**（网页搜索功能）。
  
  - 他们强调了之前使用 **Anthropic** 和 **OpenAI APIs** 的经验，表示希望利用 Perplexity 进行创新。
- **请求 Perplexity API 成本概览**：同一位成员询问是否有关于使用 Perplexity API 的 **cost overview**（成本概览）。
  
  - 他们对任何能澄清定价细节的资源表示感谢。

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1318676213417381979) (41 messages🔥):

> `6D Parallelism, PC Troubleshooting, GPU Performance Comparison, Coil Whine Issues`

- **6D Parallelism 见解分享**：一篇讨论 [6D parallelism](https://main-horse.github.io/posts/visualizing-6d/) 的文章重点介绍了 **2⁶ 6D parallel mesh** 中集合通信的可视化。
  
  - 作者指出，与此处展示的见解不同，许多资源未能捕捉到其中涉及的复杂通信。
- **新 PC 显示器连接故障**：一位用户报告他们的新 PC 可以启动，但 **显示器无信号**，直到等待一分钟后才出现启动画面。
  
  - 另一位用户建议仅使用一根内存条进行测试，以进一步排除故障。
- **Radeon GPU 与 Nvidia 4060 性能对比**：一位用户表达了对 **Radeon 显卡** 的不满，将其性能与 **Nvidia 4060** 进行了对比，指出 Radeon 虽然 FPS 高出 **5-10 FPS**，但伴随着严重的 **coil whine**（电感啸叫）。
  
  - 用户讨论了两款显卡的优缺点，强调了 Radeon 因电感啸叫产生的噪音，同时赞赏 Nvidia 4060 更安静的表现。
- **将电感啸叫作为音符**：一位成员幽默地建议创建一个利用 **coil whine** 播放音乐的程序，因为他们注意到音调会根据 GPU 负载而变化。
  
  - 另一位用户评论说音调随功耗变化，尽管声音范围可能有限。
- **寻找多 GPU NVLink 实例**：一位用户询问在 VastAI 上寻找 **带有 NVLink 的多 GPU 实例** 的经验，并对显示的带宽选项感到困惑。
  
  - 讨论围绕在这种多 GPU 设置中寻找合适资源的各种能力和局限性展开。

**提到的链接**：

- [NCCL Source Code Study](https://main-horse.github.io/series/nccl-source-code-study/)：未找到描述
- [Visualizing 6D Mesh Parallelism](https://main-horse.github.io/posts/visualizing-6d/)：包含一些背景知识

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1318874718299885653) (1 messages):

> `Kernel Computation Methods, Output Concatenation in Kernels`

- **探索 Kernel 中高效的输出拼接**：一位成员询问在 GPU Kernel 内部循环中进行 **concatenating output**（输出拼接）的非低效方法，并对比了之前累加输出的经验。
  
  - *他们表达了对* 每次迭代直接写入 **global memory**（全局内存）可能导致速度变慢的担忧，并询问是否可以使用类似 `var[idx:idx+block_size] = value` 的操作。
- **关于全局内存写入的担忧**：讨论展开了关于在 Kernel 循环中写入 **global memory** 对性能影响的探讨，因为此类操作会减慢执行速度。
  
  - *一位成员建议* 研究使用 shared memory（共享内存）作为潜在的变通方案，以提高拼接输出时的性能。

 

---

### **GPU MODE ▷ #**[**cuda**](https://discord.com/channels/1189498204333543425/1189607726595194971/1318751828472762399) (8 条消息🔥):

> `LLM Model Inference, Cuda Memory Operations, A100 vs H100 Training, CUDA Graphs and Async Operations, AMP Impact on Loss`

- **LLM 模型推理代码中的问题**：一位用户分享了 `CopyInternal` 的代码片段，指出在处理稠密张量（dense tensors）的部分处于激活状态时可能存在问题，导致推理过程产生错误结果。
  
  - *注释掉某些行可以得到正确的结果*，这表明这些行可能导致了非预期的行为。
- **理解 CudaCopy Kernel 执行**：在提供的代码中，注意到 `CudaCopy` 负责执行 CUDA kernels，这指明了在复制过程中处理内存操作的位置。
  
  - 对 stream 和 kernel 类型的提及凸显了 CUDA 环境中内存处理的复杂性。
- **A100 与 H100 训练性能调查**：一位用户对比较 **A100** 和 **H100** GPU 时的训练损失（loss）差异提出了担忧，指出即使在单 GPU 任务下也存在 **0.3%** 的差异。
  
  - 这种非预期行为引发了关于影响训练一致性的底层因素的疑问。
- **CUDA Graphs 与异步操作的挑战**：有人询问为什么缺乏关于 **CUDA Graphs** 不支持 `cudaMemcpyAsync` 操作的文档，这可能会限制某些用例。
  
  - 这表明需要就 CUDA Graphs 在异步内存操作方面的功能和约束进行更广泛的讨论。
- **AMP 与训练损失之间的可能联系**：一位用户推测训练损失的差异是否可能与 **Automatic Mixed Precision (AMP)** 有关，这引发了对优化技术的进一步探索。
  
  - 该询问强调了对不同精度策略如何影响模型性能的持续关注。

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1318671579131150367) (33 条消息🔥):

> `Megatron-LM for efficient training, Torch.compile warnings handling, Coreweave packaging challenges, Handling various shapes in image generation, Distributed training at Gensyn`

- **Megatron-LM 在分布式训练中的相关性**：有人询问 **Megatron-LM** 是否仍被认为对**高效训练**有效，特别是对于提高训练吞吐量的研究。
  
  - 建议联系 **Gensyn 的 Christine Yip**，她在分布式训练社区非常活跃，可以提供见解。
- **抑制 Torch.compile 警告**：一位成员建议将 **torch.compile** 包装在上下文管理器（context manager）中以抑制所有输出，提出了一种处理过多日志记录的潜在变通方法。
  
  - 另一位用户指出，与其抑制日志，不如使用 **warnings 模块** 来过滤预期的警告，这可能是一种更简单的方法。
- **Coreweave 打包的挑战**：讨论强调了使用 **Coreweave 定制镜像** 的复杂性，这需要编译一个新的 torch Docker 镜像，增加了显著的开销。
  
  - 必须依赖特定的编译标志（compiled flags）进行性能优化被提及为一种额外的挫败感。
- **在 Torch.compile 中管理动态形状 (Dynamic Shapes)**：一位开发者分享了在管理图像生成的各种输入形状时，**torch.compile** 中 **dynamic=True** 性能缓慢的问题。
  
  - 提出了一种策略：为已知的支持形状预热 kernel，并对意外形状无缝回退到非编译调用。
- **不使用装饰器使用 torch.compile**：建议通过存储 `fn_compiled = torch.compile(fn)` 而不是使用装饰器来进行函数式编译，从而根据条件提供灵活性。
  
  - 这种方法在快速编译整个模型方面的局限性被提及，因为漫长的编译时间是用户关注的问题。

**提到的链接**：

- [Reduce time to first kernel when using CUDA graphs](https://discuss.pytorch.org/t/reduce-time-to-first-kernel-when-using-cuda-graphs/214310)：我一直在针对 vLLM 分析我使用的推理栈，我发现由于他们调用了 graph replay，第一个 kernel 几乎立即执行（左侧），而在我的代码中...
- [GitHub - pytorch/torchtitan: A native PyTorch Library for large model training](https://github.com/pytorch/torchtitan)：一个用于大模型训练的原生 PyTorch 库。通过在 GitHub 上创建一个账号来为 pytorch/torchtitan 的开发做出贡献。

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1318674557439311932) (4 条消息):

> `Raspberry Pi 5 deployment, Edge device LLM models, Esp32 / Xtensa LX7 chips`

- **提升 Raspberry Pi 5 的 LLM 性能**：配备了 **256GB NVMe** 以增强数据传输，并超频至 **2.8GHz** 的 **Raspberry Pi 5**，正被配置用于部署 **1.5B 参数**的**小模型**，使用的是通过 **OpenBlas** 编译的 **Ollama**。
  
  - 此设置旨在促进在 Pi 5 上的本地部署，展示了边缘设备作为模型托管节点的潜力。
- **对 Esp32 / Xtensa LX7 芯片的期待**：一位成员分享了对新型 **Esp32 / Xtensa LX7 芯片**的热切期待，这些芯片拟用于通过 **API** 远程调用 **LLM** 的不同场景。
  
  - 应用场景的转变凸显了新兴硬件在 AI 部署策略中的多样性和广泛潜力。

 

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1319053856751095948) (1 条消息):

> `MatX LLM Accelerator, Hiring for roles, Low level compute kernel, Compiler engineering, ML performance engineering`

- **MatX 正在开发 LLM 加速器 ASIC**：MatX 宣布了其构建旨在提升 AI 性能的 **LLM 加速器 ASIC** 的计划。
  
  - 有关该项目的详细信息可以在其网站上找到，他们正在该领域积极寻求人才。
- **MatX 招聘技术岗位**：该公司正在招聘多个职位，包括 **low level compute kernel author**（底层计算内核作者）、**compiler**（编译器）和 **ML performance engineer**（ML 性能工程师）。
  
  - 感兴趣的候选人可以查看 [MatX Jobs](https://matx.com/jobs) 上的空缺职位。

 

**提到的链接**：[Tweet from MatX | Jobs](https://matx.com/jobs)：未找到描述

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1318929596292534343) (5 条消息):

> `int4group scheme, Tinygemm compute, Activation quantization, Matmul kernel processing`

- **int4group 方案的关键**：在 **int4group 方案**中，权重保持为 **int4** 量化，而激活保持为 **fp16**，从而产生 **fp16 x int4 = fp16** 的 matmul。
  
  - 随附的 [图片](https://cdn.discordapp.com/attachments/1205223658021458100/1318929596074426429/image.png?ex=67641c68&is=6762cae8&hm=e5372ccf58051c226b883b52d91377b7edb6c4b0ea18c4d3022aaee235d80d87&) 展示了这一过程。
- **训练期间无激活量化**：已确认在训练过程中没有针对激活的 **fake quantization**（伪量化），因为 **Tinygemm** 使用 **bf16** 进行计算。
  
  - **int4 权重**在 **Quantization Aware Training** (QAT) 和推理阶段都会进行即时反量化（dequantized on-the-fly）。
- **Matmul 内核处理反量化**：针对关于 **on-the-fly**（即时）计算的问题，澄清了反量化过程发生在 **matmul kernel** 内部。
  
  - 这种方法确保了量化权重的无缝集成，且不影响激活的浮点表示。

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1318820781857177630) (1 条消息):

> `MigraphX, ONNX Frontend, MI300X, Opset 11 Support`

- **MigraphX 在 MI300X 上的潜力**：一位成员表达了在 **MI300X** 平台上为 **ONNX Frontend** 构建 **MigraphX** 的信心。
  
  - 他们提到尚未检查支持的 **Opset 11**，但认为它应该是可用的最新版本。
- **关于 MI300X 经验的咨询**：同一位成员询问是否有人尝试过将 **MigraphX** 与 MI300X 结合使用。
  
  - 重点在于确保与当前使用的 **Opset** 标准的兼容性。

 

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/) (1 条消息):

kimishpatel: 这正是我来这里的目的 🙂

---

### **GPU MODE ▷ #**[**arc-agi-2**](https://discord.com/channels/1189498204333543425/1316377974672588850/1318672936063733890) (18 条消息🔥):

> `自定义 Vision Encoder 开发、Chain of Thought 实现、LLM 实验、LLM 训练配置、CoT Prompt 的去中心化采样过程`

- **自定义 Vision Encoder 探索**：一名成员建议创建一个**自定义 Vision Encoder** 以与现有的语言模型集成，并指出当前的视觉模型对于小像素尺度的图像可能不是最优的。
  
  - 他们认为这种灵活性带来的好处可能超过预训练 VLM 所提供的收益。
- **Chain of Thought 的粒度**：针对 **Chain of Thought** (CoT) 实现的粒度提出了担忧，特别是关于其在核心思想之外的深度。
  
  - 有建议认为，推理中的多次迭代可能会改进问题的解决。
- **LLM 与 CoT 实验**：讨论了测试 CoT 在生成推理 token 方面有效性的**概念验证** (PoC) 阶段计划，强调了实验结果的重要性。
  
  - 一名成员表示，他们期待将**内部推理**集成到模型中，以提升解决方案的质量。
- **微调配置咨询**：针对 LLM 的微调设置提出了疑问，特别是 **bf16**、**Lora** 或 **Qlora+int8** 等组合是否为首选方法。
  
  - 分享了针对 **llama-3-vision** 的 Lora 配置验证，供那些在 GPU 环境下进行实验的人员参考。
- **高效 CoT Prompt 的去中心化采样**：提出了在不进行训练的情况下运行**去中心化采样过程**以开发高效 CoT prompt 的潜力，希望能收集到一个全面的数据集。
  
  - 该方法旨在促进人工引导的探索并增强 prompt 效率。

**提到的链接**：

- [augmxnt](https://wandb.ai/augmxnt/train-bench/runs/zelehjsm/overview): Weights & Biases，机器学习开发者工具
- [axolotl/examples/llama-3-vision/lora-11b.yaml at effc4dc4097af212432c9ebaba7eb9677d768467 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/blob/effc4dc4097af212432c9ebaba7eb9677d768467/examples/llama-3-vision/lora-11b.yaml): 欢迎在 GitHub 上通过创建账号为 axolotl-ai-cloud/axolotl 的开发做出贡献。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1318672102517248053) (87 条消息🔥🔥):

> `LM Studio Beta 功能、使用 Llama 3.2 模型、角色扮演 LLM 设置、将 LM Studio 连接到移动设备、运行模型的硬件规格`

- **LM Studio Beta 功能与问题**：用户讨论了 LM Studio 的各种 Beta 功能，包括安装问题以及特定的模型架构错误，如 “unknown model architecture mllama”。
  
  - 一些用户成功运行了大型模型，而另一些用户则面临下载困难或文件损坏的挑战。
- **关于 Llama 3.2 模型使用的澄清**：关于在 LM Studio 中使用 Llama 3.2 11B Vision 模型存在困惑，特别是运行该模型是否必须使用 MLX。
  
  - 几位用户确认他们能够运行该模型，但其他用户遇到了与 Safetensors header 和内存限制相关的错误。
- **设置角色扮演 LLM**：一位用户寻求设置角色扮演 LLM 的帮助，并被引导至特定频道进行进一步咨询。
  
  - 社区在分享有效利用 LLM 模型的技巧和最佳实践方面非常活跃。
- **将 LM Studio 连接到移动设备**：一位用户询问在桌面运行 LM Studio 的同时，如何在手机上使用它。
  
  - 成员们确认目前没有 LM Studio 的移动端应用，但讨论了远程访问功能的潜在变通方案。
- **运行模型的硬件规格**：用户分享了他们的硬件配置，特别提到了配备 96GB 和 128GB 内存的 M3 Max 设置，以获得最佳模型性能。
  
  - 讨论强调了充足的 RAM 和 GPU 规格对于有效运行大型模型的重要性。

**提到的链接**：

- [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit · Hugging Face](https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit): 未找到描述
- [Haider. (@slow_developer) 的推文](https://fixupx.com/slow_developer/status/1869059311969661103): 🚨 NVIDIA 推出 Jetson Nano Super：超紧凑型 AI 计算机，算力达 70-T operations per second；专为机器人设计，支持包括 LLM 在内的先进模型，售价 249 美元。
- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases): LM Studio Beta 版本发布页面

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1318670170578751529) (17 条消息🔥):

> `3060 Ti vs 3060, AMD Radeon VII driver issues, Llama model utilization, Inference performance with GPUs, M2 MacBook Air as a gateway`

- **3060 Ti 困惑已澄清**：关于是否存在 **3060 Ti 11GB** 版本存在困惑，用户质疑它是否只是 12GB 的 **3060**。
  
  - *一位成员指出了规格上的差异*，引发了关于 GPU 对比的简短讨论。
- **Radeon VII 面临驱动问题**：成员们讨论了影响 **Radeon VII** 的 **24.12.1** 驱动问题，其中一人需要回滚到 **24.10.1** 以获得正常性能。
  
  - 他们强调使用 **24.12.1** 会导致 **100% GPU utilization without power usage**（GPU 利用率 100% 但无功耗），从而导致基础任务出现严重延迟。
- **Llama 模型性能见解**：讨论显示，将一个 **5GB** 模型拆分到多个 GPU 会导致性能下降，促使一位用户考虑强制使用单个 GPU 进行推理。
  
  - *另一位成员建议检查* `llama.cpp` *中的设置*，以确保其正确使用 CUDA，并建议调整设置可以提高性能。
- **推理需要大量 RAM**：一位成员指出运行 **70B model** 需要 **70GB RAM**，这包括了 VRAM 和系统内存。
  
  - *作为回应，另一位成员澄清说，在 q8 下运行可能需要每十亿参数约 1GB*，并强调了为上下文保留额外 VRAM 的重要性。
- **M2 MacBook Air 激发升级幻想**：一位用户幽默地提到，拥有 **MBA M2** 让他们渴望拥有一台 **powerful MBP M4** 以获得更好的推理能力。
  
  - *另一位成员用一个幽默的比喻插话，说需要优先考虑显存*，并开玩笑说要卖掉家当来换取高性能配件。

---

### **Stackblitz (Bolt.new) ▷ #**[**prompting**](https://discord.com/channels/364486390102097930/1301167628416454737/1318837235692474419) (6 条消息):

> `Migrating from Firebase to Supabase, Using create-mf-app with Bootstrap, Google reCAPTCHA Issues, Testing Bolt Pilot GPTs, Vite Pre-Transform Errors`

- **Firebase 到 Supabase 的迁移策略**：一位成员询问了将整个网站从 **Firebase** 迁移到 **Supabase** 的最佳方法。
  
  - 讨论中尚未提供具体的技术方案。
- **寻求 Vite 和 Tailwind 的替代方案**：一位用户表示希望使用 **create-mf-app** 和 **Bootstrap** 而不是 **Vite** 和 **Tailwind**，这导致了冲突。
  
  - 他们无意中造成了两种样式框架纠缠在一起的情况。
- **Google reCAPTCHA 密钥类型困惑**：一位成员分享了在使用 **Google reCAPTCHA v3** 而非 **v2** 时遇到“Invalid key type”错误的经历。
  
  - 切换到 v2 后，他们仍然面临无法从联系表单接收电子邮件等问题。
- **测试 Bolt Pilot GPT**：一位成员宣布为 **ChatGPT** 创建了一个名为 **Bolt Pilot** 的新 GPT，并请求对其调整提供反馈。
  
  - 他们的热情表明他们渴望根据社区反馈改进产品。
- **循环出现的 Vite 错误**：一位用户报告在处理项目时多次遇到“[vite] Pre-Transform”错误。
  
  - 这个问题似乎是一个普遍关注点，可能也会影响到其他成员。

---

### **Stackblitz (Bolt.new) ▷ #**[**discussions**](https://discord.com/channels/364486390102097930/680953097354215446/1318669185370296461) (97 条消息🔥🔥):

> `对 Bolt 的沮丧感、寻求项目帮助、Token 使用、协作项目、技术讨论`

- **用户对 Bolt 浪费 Token 感到沮丧**：多位用户表达了对 **Bolt** 在不必要的提示词和占位内容上浪费 Token 的不满，并建议增加重置按钮以停止过度的 Token 消耗。
  
  - 有人提到他们已经在 Token 上花费了大量资金，但没有得到满意的结果。
- **项目协助请求**：几位用户表示正在寻求项目指导，有些人愿意为获得帮助支付费用，特别是针对构建多租户应用程序的承包商和服务提供商。
  
  - 讨论内容包括分享项目链接以及在各种任务上协作的意愿。
- **Token 折扣和计划**：用户询问了 Token 的潜在节日折扣，并对 **Bolt** 过度消耗导致的当前定价表示担忧。
  
  - 有人建议关注 **Office Hours** 期间的公告，以获取 Token 定价的更新。
- **协作构建想法**：一些用户热衷于组队创建项目，并提议整合他们的 Token 资源进行协作，旨在共同构建实质性的内容。
  
  - 这反映了社区对参与项目开发的兴趣日益浓厚。
- **Bolt 的技术问题及潜在解决方案**：用户分享了与**注释代码（commented code）**相关的 Bug 经验，并讨论了潜在的修复方法，例如调整设置以解决 context window 问题。
  
  - 建议包括关闭 diff 模式，以缓解开发过程中遇到的一些挫败感。

**提到的链接**：

- [Vite + React + TS](https://aesthetic-sorbet-a2513b.netlify.app/): 未找到描述
- [\- YouTube](https://youtu.be/nUYhPID5sjM): 未找到描述
- [GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!](https://github.com/stackblitz-labs/bolt.diy#join-the-community-for-boltdiy): 使用任何你想要的 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy
- [GitHub - RealSput/Wenode: WebContainers, except it's a million times easier to use](https://github.com/RealSput/Wenode): WebContainers，但使用起来要简单一百万倍 - RealSput/Wenode
- [oTTomator Community](https://thinktank.ottomator.ai/): 创新者和专家汇聚一堂，共同推进 AI 驱动自动化的未来

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1318718521940770847) (42 条消息🔥):

> `在 Maya 中使用工具、本地模型集成、睡眠的重要性、Hugging Face UI 更新、使用 VQA 进行图像分析`

- **基础模型对工具使用（Tool Use）的需求**：成员们讨论了在基础模型中集成 **tool use** 的必要性，以增强 **image analysis** 的功能。
  
  - *一位成员评论道*，“我可能不应该在这里谈论这个。这是 c4 的人做的。”
- **Maya 工具集成讨论**：一位成员建议调用 **command-r-plus-08-2024**，以促进将 Maya 作为其本地模型的工具进行**集成**。
  
  - 这种方法旨在允许模型通过发送**图像路径**和问题来与 Maya 交互，从而优化响应。
- **提及睡眠的重要性**：在聊天中，一位成员强调了睡眠的重要性，促使另一位成员回应道：*“别把自己逼得太紧，休息一下！”*
  
  - 这引发了成员之间关于恢复精力和工作平衡的轻松交流。
- **注意到 Hugging Face UI 更新**：一位成员指出了 **Hugging Face** 的更新，并与小组分享了截图，表示有明显的改进。
  
  - *在使用新 UI 工作和应对挑战时的幽默感吸引了其他参与者的俏皮评论。*
- **处理大型 JSON 文件遇到困难**：一位成员表达了在尝试加载 **71,000 行** JSON 导致 **IDE 崩溃**后的沮丧。
  
  - *另一位成员对这种情况进行了幽默的评论，强调了处理大数据集时的常见困难。*

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1318696792207921213) (1 messages):

> `Rate-limit increase, Multimodal Image Embed endpoint, API key types, Rate limits details, Community engagement`

- **Multimodal Image Embed 速率限制大幅提升！**：响应社区需求，**Multimodal Image Embed** 端点在生产密钥（production keys）上的速率限制从 **40 images/min 飙升至 400 images/min**。
  
  - 测试密钥（trial keys）的测试速率限制保持不变，仍为 **5 images/min**，方便用户自由实验。
- **检查您的 API 密钥选项！**：Cohere 提供**两种类型的 API 密钥**：评估密钥（evaluation keys，使用受限）和生产密钥（production keys，付费使用且限制较少）。
  
  - 用户可以通过 [API keys 页面](https://dashboard.cohere.com/api-keys)创建测试或生产密钥，更多价格详情请参阅 [pricing docs](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work)。
- **详尽的速率限制细分**：一份详细的表格列出了各种端点的**测试**和**生产**速率限制，确保使用额度清晰透明。
  
  - 显著调整包括 **Embed (Images)** 现在在生产环境下拥有 **400 images/min**，而测试环境保持 **5 images/min**。
- **鼓励社区分享！**：公告邀请用户使用更新后的端点限制来创建并分享他们的应用程序。
  - 在提供的频道中推广社区参与，用于分享经验和反馈。
- **提供查询支持**：对于有关更新的任何问题或疑虑，鼓励用户在特定的支持频道中提问或直接联系支持团队。
  
  - 这确保了用户有足够的支撑来使用新功能和理解速率限制。

 

**提到的链接**：[API Keys and Rate Limits — Cohere](https://docs.cohere.com/v2/docs/rate-limits)：此页面描述了 Cohere API 针对生产和评估密钥的速率限制。

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1318719316299874397) (51 messages🔥):

> `Cohere Structured Outputs Implementation, Embedding Dimensions Concerns, RAG-based PDF Answering System Issues, Cohere Reranker Functionality, Cohere and Nvidia Relationship`

- **有效地实现 Cohere Structured Outputs**：一位用户成功地在其项目中实现了 Cohere 的 Structured Outputs，并通过调整请求参数解决了与 tool usage 相关的问题。
  
  - 他们讨论了使用 `strict_tools` 设置，以确保模型能够有效地利用工具生成结构化响应。
- **向量存储中的 Embedding 维度**：一位用户询问了关于存储由不同模型生成的 Embedding 的问题，因为存在维度差异，特别是 `text-3-embedding-large` 和 `cohere embed v3` 之间。
  
  - 这突显了在统一的向量存储中管理来自不同模型的 Embedding 的复杂性。
- **RAG 系统中的 Cohere Reranker 问题**：一位开发者在其基于 RAG 的 PDF 问答系统中遇到了 Cohere Reranker 的不一致问题，尽管分块（chunking）正确，但它偶尔会选择相关性较低的数据块。
  
  - 尽管系统有时能检索到准确答案，但他们对 Reranker 表现出的随机性表示担忧。
- **理解 AI 基础设施选择**：针对关于 Cohere 对 Nvidia 产品依赖性的问题，一位用户指出 Nvidia 在大多数 AI 系统中处于核心地位，这归功于其强大的生态系统，特别是 CUDA 和 NCCL。
  
  - 然而，AMD 和 Google TPU 等替代方案也同样存在，这表明 AI 基础设施有着更广泛的选择范围。
- **探索 AI 中的 TPU 性能**：对话指出，虽然 TPU 技术并非小众，但它主要被 Anthropic 等特定组织使用，因为其在处理效率方面具有优势。
  
  - 讨论转向了优化矩阵乘法（matrix multiplication）的重要性，以及基础设施的选择如何影响 AI 系统的性能。

**提到的链接**：

- [Chat — Cohere](https://docs.cohere.com/reference/chat#request.body.strict_tools)：生成对用户消息的文本响应并逐个 Token 进行流式传输。要了解如何使用带流式传输的 Chat API，请遵循我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/cha...)。
- [Structured Outputs — Cohere](https://docs.cohere.com/docs/structured-outputs)：此页面描述了如何让 Cohere 模型以特定格式（如 JSON）创建输出。
- [Chat — Cohere](https://docs.cohere.com/reference/chat#request.body.response_format)：生成对用户消息的文本响应并逐个 Token 进行流式传输。要了解如何使用带流式传输的 Chat API，请遵循我们的 [Text Generation 指南](https://docs.cohere.com/v2/docs/cha...)。

---

### **Cohere ▷ #**[**cmd-r-bot**](https://discord.com/channels/954421988141711382/1168578374038470656/) (1 messages):

setupisanoun: 嘿，伙计

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1318889996584878081) (2 messages):

> `Findr 发布, 无限记忆, Product Hunt`

- **Findr 正式起飞！**: **Findr** 已在 [Product Hunt](https://www.producthunt.com/posts/findr-remember-everything) 正式发布，旨在为用户提供**无限记忆（infinite memory）**和可搜索的数字大脑。
  
  - 创作者表达了对支持的感谢，并表示他们对构建这款增强记忆的新工具感到兴奋。
- **社区为发布成功欢呼**: 成员们祝贺创作者成功发布 Findr。
  
  - 这种积极的社区反馈凸显了人们对旨在增强人类记忆的创新项目的**热情**。

 

**提到的链接**: [来自 Nishkarsh (@Nish306) 的推文](https://x.com/Nish306/status/1868953328975261712): 我们已在 Product Hunt 发布。非常感谢您的支持 [https://www.producthunt.com/posts/findr-remember-everythingwe're](https://www.producthunt.com/posts/findr-remember-everythingwe're) 我们正在赋予人类无限记忆和可搜索的数字大脑...

 

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1318945686573551689) (3 messages):

> `Cohere Toolkit 部署, AWS 流错误, Docker 日志分析`

- **Cohere Toolkit 部署成功**: 一位成员根据提供的 AWS 指南成功部署了 **Cohere Toolkit**，并分享了他们的兴奋之情。
  
  - 然而，他们遇到了一个间歇性错误：*stream ended unexpectedly*（流意外结束），影响了功能。
- **间歇性 AWS 流错误**: 部署过程中遇到了 AWS 流的随机问题，偶尔会导致聊天功能中断。
  
  - 该成员向他人寻求帮助，询问是否有人遇到过类似问题。
- **建议检查 Docker 日志**: 另一位成员建议检查 **docker logs** 以获取诊断问题的相关信息。
  
  - 该建议旨在缩小间歇性流错误原因的排查范围。

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1318669970682679397) (22 messages🔥):

> `Archcraft 上的 Mojo REPL 问题, Magic 环境挑战, Stable Diffusion 示例参考, Max 安装问题, 创建问题解决线程`

- **Mojo REPL 在 Archcraft 上找不到 ldd 库**: 一位用户报告了在 Archcraft Linux 上访问 Mojo REPL 的问题，提到它似乎在寻找一个名为 **mojo-ldd** 的**动态链接库**。
  
  - 另一位成员指出 `mojo-lld` 可能是一个链接器，但需要确切错误的详细信息以提供进一步帮助。
- **Magic 环境中的困境**: 同一位用户表示，在 **magic environment** 中时，他们无法安装 Python 依赖项，收到了关于处于**外部管理环境**（externally managed environment）的消息。
  
  - 尽管存在这些问题，用户确认可以访问 **Max** 和 **Magic**，但进一步的探索因安装失败而中断。
- **分享 Stable Diffusion 示例**: 一位成员祝贺社区发布了新版本，并引用了 GitHub 仓库中关于 **Stable Diffusion** 的一个示例。
  
  - 提供的 GitHub 链接被认为是对有兴趣使用新功能的用户非常有用的资源。
- **Max 安装进程被终止**: 用户在尝试安装 **Max** 时遇到问题，称安装进程意外被**终止（killed）**。
  
  - 他们正在等待 magic environment 安装 Max，这表明尽管之前出现了错误，安装尝试仍在进行中。
- **创建线程以增强问题解决效率**: 一位社区成员建议为问题解决创建一个**新线程（thread）**，以帮助遇到类似问题的其他人。
  
  - 该提议旨在促进持续的讨论和解决方案，同时不干扰主对话流。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1318676804692873278) (57 条消息🔥🔥):

> `Mojo 文档更新、Mojo Kernel 定义、功能开发与语法的权衡、重新审视 Mojo 早期决策、Mojo 中的计算 Kernel`

- **Mojo 文档更新引发困惑**：在最近的讨论中，成员们在查阅 [Mojo 文档](https://docs.modular.com/mojo/manual/basics#variables) 后，对 Mojo 代码中必须使用 `var` 关键字的要求表示困惑。一位成员强调了对此的潜在不满，指出当前实现中存在一个尚未解决的问题。
  
  - 另一位成员提到文档即将更新，但尚未确认发布时间，并表示欢迎社区反馈。
- **澄清 “Kernel” 术语**：讨论了 Mojo 语境下的 “Kernel” 一词，成员们澄清它指的是针对 GPU 执行优化的函数，而非操作系统语境下的内核。该术语的使用范围很广，一位成员强调它应该指代核心计算逻辑。
  
  - 成员们一致认为，它可以表示通常在加速器上执行的纯函数，但其定义可能因语境而异，在数学中也包含多种含义。
- **功能开发优先于语法**：成员们达成共识，在关注语法改进之前，应优先考虑 Mojo 的核心功能开发。一位成员指出，这种方法可以最大限度地减少重写的需要，并有助于稳定语言。
  
  - 关于是否保留或移除 `var` 要求的讨论揭示了不同的偏好，凸显了关于语言设计的持续对话。
- **对 Mojo 未来的希望**：几位成员对 Mojo 的潜力表示乐观，其中一位指出用该语言构建独特事物的愿景。对话表明了 Mojo 与 CUDA 等成熟计算 Kernel 一起有效集成到现有生态系统中的雄心。
  
  - 成员们承认，要使 Mojo 适用于 OS 内核和驱动程序开发，还有大量工作要做，强调了社区对其增长的渴望。
- **Algorithm Reduction 中缺失的功能**：一位成员询问了 `algorithm.reduction` 中 `argmax` 和 `argmin` 的状态，并对它们未出现在更新日志中表示沮丧。成员们对需要从头开始重新实现这些函数的优化版本表示担忧。

 

**相关链接**：[Mojo 语言基础 | Modular 文档](https://docs.modular.com/mojo/manual/basics#variables)：Mojo 基础语言特性介绍。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1318740743908622337) (13 条消息🔥):

> `Mojo 中的 Custom Ops, MOToMGP Pass Manager 错误, 文档问题, 错误消息的功能请求, 改进 Custom Ops 的 UX`

- **Mojo 中的 Custom Ops 面临挑战**：用户在通过 MAX 加载 custom ops 时遇到问题，特别是找不到 **mandelbrot** kernel。
  
  - 一位用户发现，正确的命令涉及 `session.load(graph, custom_ops_paths=Path("kernels.mojopkg"))` 以解决该问题。
- **MOToMGP Pass Manager 中的错误**：多个 **Unhandled exceptions** 表明 MOToMGP pass manager 存在故障，特别是缺少针对 custom operations（如 **get_scalar_from_managed_tensor_slice**）的 mojo kernels。
  
  - 这突显了对更清晰的错误消息以及对来自 Mojo 源文件的 custom ops 支持的需求。
- **建议改进文档**：一位成员建议在 GitHub max 仓库中提交 issue，以解决 custom op 错误消息的缺陷，特别是与 **'op not found'** 错误相关的缺陷。
  
  - 对话引导出了一项改进建议，即让错误消息将用户引导至相关文档。
- **更好错误处理的功能请求**：提出了一个功能请求，旨在增强错误消息系统，以便在找不到 **custom op** 时更好地通知用户。
  
  - 该请求包括提供有效使用 custom ops 的指导，并确保可以访问相应的文档。
- **链接了 GitHub Issue 供参考**：一位用户链接了一个关于 custom ops 和错误处理的 GitHub issue：[Feature Request Issue #269](https://github.com/modularml/max/issues/269)。
  
  - 该 issue 提出了双重改进目标，重点是更好的错误消息和 single compilation unit kernels 的集成。

 

**提到的链接**：[[Feature Request] Single compilation unit kernels and/or improved error messages · Issue #269 · modularml/max](https://github.com/modularml/max/issues/269)：你的请求是什么？这是一个分为两部分的请求，但由于它们都解决了相同的 UX 问题，因此捆绑在一起。第一部分是让 "custom op not found" 错误消息引导用户查看文档...

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1318673443691958324) (90 条消息🔥🔥):

> `Nvidia Jetson Orin Nano, Github Copilot Free, 1-800-CHATGPT 体验, AI 视频趋势, Google Whisk 工具`

- **Nvidia Jetson Orin Nano Super Developer Kit 发布**：[Nvidia 宣布](https://blogs.nvidia.com/blog/jetson-generative-ai-supercomputer/)推出 Jetson Orin Nano Super Developer Kit，售价为 **$249**，具备 67 TOPS 的神经处理能力，相比前代提升了 **70%**。
  
  - 拥有 **102GB/s** 的内存带宽，面向爱好者，旨在增强 AI 和机器人项目，尽管有人对其性能是否充足表示担忧。
- **Github Copilot 免费访问**：GitHub **Copilot** 已对 VS Code 免费开放，正如 [Satya Nadella](https://x.com/satyanadella/status/1869445091213095140) 所确认的那样。
  
  - 反应包括对其每月仅限 **50 次对话** 的怀疑，以及对 Cursor 等竞争对手影响的讨论。
- **1-800-CHATGPT 将 AI 带入电话**：由 [Kevin Weil](https://x.com/kevinweil/status/1869446218163839264) 宣布，美国用户可以拨打 **1-800-CHATGPT**，全球用户可以在 WhatsApp 上发消息，无需账号。
  
  - 这被视为一种让老年用户在无需安装 App 的情况下接触 AI 的有效方式。
- **AI 视频模型进入佳境**：关于 AI 视频模型演进的讨论重点提到了 OpenAI 的 [Sora](https://openai.com/index/sora/) 及其变革性影响，能够生成高分辨率输出。
  
  - 社区注意到这与过去图像生成的进步有相似之处，表明随着更多模型的推出，需求正在增长。
- **Google Whisk 工具发布**：[Google Whisk](https://labs.google/fx/tools/whisk) 是一款旨在生成可爱 Logo 和数字艺术的新工具，引起了寻求快速解决方案的开发者的兴趣。
  
  - 尽管反响热烈，但对其可用性存在担忧，特别是加拿大用户发现无法访问。

**提到的链接**：

- [来自 Satya Nadella (@satyanadella) 的推文](https://x.com/satyanadella/status/1869445091213095140)：VS Code 版 GitHub Copilot 免费版已上线。
- [来自 Michelle Burgess (@Misslfc) 的推文](https://x.com/vercel/status/1869144568)：@Pink American Idol 做了哈哈！太烂了，你肯定做得更好
- [来自 Sar Haribhakti (@sarthakgh) 的推文](https://x.com/sarthakgh/status/1868881401283764698)：MidJourney 创始人
- [来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文](https://x.com/kevinweil/status/1869446218163839264)：第 10 天 🎁：1-800-CHATGPT ✨ 如果你在美国，可以拨打 ChatGPT 电话，或者在地球任何地方通过 @WhatsApp 给它发消息。免费，无需账号！现在就试试吧。
- [来自 Andras Bacsai (@heyandras) 的推文](https://x.com/heyandras/status/1869462505992642848?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q)：回家吧 Github Copilot。你喝醉了。
- [Nvidia 售价 $249 的开发套件承诺提供廉价、小巧的 AI 动力](https://www.theverge.com/2024/12/17/24323450/nvidia-jetson-orin-nano-super-developer-kit-software-update-ai-artificial-intelligence-maker-pc)：价格仅为前代的一半。
- [来自 fofr (@fofrAI) 的推文](https://x.com/fofrAI/status/1869105503349223480)：Minimax 的 `music-01` 现已上线 Replicate。使用参考歌曲和歌词生成长达 1 分钟的音乐：- 使用参考歌曲、人声和乐器轨道 - 可选歌词 - 重用参考以实现快速...
- [AI 视频正迎来它的 Stable Diffusion 时刻](https://replicate.com/blog/ai-video-is-having-its-stable-diffusion-moment)：未找到描述
- [Whisk](https://labs.google/fx/tools/whisk)：未找到描述
- [来自 Vercel (@vercel) 的推文](https://x.com/vercel/status/1869144568350019900)：参加 AI 开发者现状调查。帮助我们了解你如何使用 AI——你在构建什么，如何构建，以及面临哪些挑战。https://state-of-ai.vercel.app/
- [来自 Taelin (@VictorTaelin) 的推文](https://x.com/VictorTaelin/status/1868826950107861268)：什么
- [来自 tldraw (@tldraw) 的推文](https://x.com/tldraw/status/1869401069849379109)：我们刚刚发布了 tldraw computer
- [来自 web weaver (@deepfates) 的推文](https://x.com/deepfates/status/1868762119040385045?s=46)：是时候进行 Will Smith 测试了。Replicate 上所有的视频模型，提示词均为 “Will Smith eating spaghetti”。这是 2023 年的原始版本，以及一个最先进的版本...
- [未找到标题](https://ai.google.dev/showcase/tldraw)：未找到描述
- [\- YouTube](https://youtu.be/a0bEU83P8g8?si=9V0yJeqtWnhVicKI)：未找到描述
- [\- YouTube](https://youtu.be/kO192K7_FaQ?si=DVQzEn5aDkUb-EhA)：未找到描述
- [未找到标题](https://news.ycombinator.com/item?id=42430296)：未找到描述

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1319020087461548064) (1 条消息):

> `Sakana AI 的 EvoMerge、DynoSaur、LLM Paper Club 活动`

- **EvoMerge 和 DynoSaur 双重活动**：今天将举行一场特别的**双重活动**，展示 **Sakana AI 的 EvoMerge** 和 **DynoSaur**，计划在公告发布 45 分钟后开始。
  
  - 鼓励成员加入并参与 [LLM Paper Club](http://Latent.Space) 活动中的讨论。
- **后续活动提醒**：分享了一个提醒，建议使用 RSS 功能将 [Latent.Space](http://Latent.Space) 活动添加到个人日历中。
  
  - *请点击日历上方的 RSS 图标*，以获取新活动和讨论的最新动态。

 

**提到的链接**：[LLM Paper Club (Sakana AI EvoMerge and DynoSaur) · Zoom · Luma](https://lu.ma/yv4j5d8h)：Ramon 首次加入我们并进行展示……

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1318671414433284116) (28 条消息🔥):

> `Open Interpreter 错误、OI 1.x 版本、Cloudflare AI Gateway、Truffle AI Computer、长期记忆集成`

- **持续存在的 Open Interpreter 错误**：多位用户对使用 Open Interpreter 时反复出现的错误表示沮丧，特别是与加载对话和 API key 问题相关的错误。
  
  - *一位用户对丢失优质对话感到惋惜*，引发了关于这些在不同平台上持续存在的问题的讨论。
- **对 Open Interpreter 1.x 版本的困惑**：用户讨论了 Open Interpreter 不同版本之间的差异，包括关于在当前使用 0.34 版本时如何访问 1.x 版本的疑问。
  
  - *一位用户指出 OS 模式在 1.0 中似乎不可用*，引发了对功能变化的疑问。
- **集成 Cloudflare AI Gateway 的建议**：一名成员建议尝试使用 Cloudflare AI Gateway，作为解决 Open Interpreter 某些配置问题的潜在方案。
  
  - 这引发了关于各种可以增强功能的 AI 应用和工具的讨论。
- **Truffle AI Computer 介绍**：一位用户分享了关于 Truffle-1 的见解，这是一款 AI 计算设备，可在设备上运行模型，配备 64GB 统一内存，需支付 500 美元押金和每月 115 美元。
  
  - 他们强调了在该设备上编写和分享应用的能力，激发了对实际用例的兴趣。
- **长期记忆的潜力**：关于为 Open Interpreter 集成长期记忆功能以有效管理代码库的讨论仍在继续。
  
  - *用户表示有兴趣使用 Raspberry Pi 等设备进行本地设置*，以支持此类集成。

**提到的链接**：

- [Truffle](https://itsalltruffles.com)：个人 Agentic 计算栈 —— Truffle-1 在设备上运行混合模型，配备 64GB 统一内存。
- [First Of All All Things Are Possible GIF - First Of All All Things Are Possible Jot That Down - 发现并分享 GIF](https://tenor.com/view/first-of-all-all-things-are-possible-jot-that-down-pointing-serious-gif-14586817)：点击查看 GIF。
- [来自 simp 4 satoshi (@iamgingertrash) 的推文](https://x.com/iamgingertrash/status/1869450385896751449)：简要回顾：> 今天授权 500 美元押金 > 12 个月每月 115 美元 > 无限的推理时间计算 > 编写和分享你自己的应用 > 一个带有 64GB Orin 的发光球体 > 我们实际上……

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1318864276387139694) (27 条消息🔥):

> `Benchmarks for LLaMA models, Mergeability of ShapeTrackers in Lean, Counterexamples in view merging, CuTe layout algebra, Challenges in proving layout injectivity`

- **LLaMA 模型基准测试请求**：一位成员询问是否有任何基准测试可以对比使用 **tinygrad OpenCL** 与 **PyTorch CUDA** 运行 **LLaMA 模型**的性能。
  
  - 讨论中没有提供任何回复或基准测试数据。
- **关于 ShapeTracker 可合并性的讨论**：有人寻求关于在 **Lean** 中实现两个任意 **ShapeTrackers** **可合并性（mergeability）**相关悬赏目标的澄清。
  
  - 贡献者们讨论了由于影响 stride 和 shape 的变量存在，实现通用合并标准是不可能的。
- **反例揭示合并问题**：一位成员分享了关于**反例（counterexamples）**的见解，指出当前的合并算法在处理某些无法正确合并的特定视图对（view pairs）时存在困难。
  
  - 他们表示，针对与维度和溢出相关的单个问题，有可能生成多个示例。
- **CuTe 布局代数与可合并性**：讨论中提出了 TinyGrad 中的可合并性是否类比于 **CuTe 布局代数**中的**组合（composition）**，并链接了相关资源以供深入理解。
  
  - 另一位成员指出，证明布局代数中的某些属性（如必要性和充分性）可能非常具有挑战性。
- **布局代数中的单射性是 NP 困难的**：一位成员对证明布局代数中的**单射性（injectivity）**表示怀疑，认为其计算复杂度可能很高。
  
  - 他们推论，检查单射性的必要性和充分性可能被归类为 NP 困难（NP hard）。

**提到的链接**：

- [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)：NVIDIA CUTLASS 库中高性能线性代数的核心抽象是 CuTe Layout。在这篇技术笔记中，我们对这些布局的代数进行了严谨的数学处理……
- [Issues · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/8194)：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad
- [cutlass/media/docs/cute/02_layout_algebra.md at main · NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)：线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1318727531725197412) (25 条消息🔥):

> `FSDP Adjustment, Bug Fix in TRL, Loss Scaling Discussion, Gradient Scaling, Optimizer in Backward Case`

- **Scaling 需要 FSDP 调整**：成员们一致认为，如果 **FSDP** 的 reduce 操作取平均值，则需要通过 `world_size` 进行 scaling，以抵消梯度计算中的这种影响。
  
  - 此调整仅涉及更改提供的代码示例中对 `scale_grads` 的调用，从而简化实现。
- **TRL 中的潜在 Bug**：一名成员建议 **trl** 库中存在一个与 scaling 相关的 Bug，且可能具有普遍性，这意味着它也可能影响其他领域。
  
  - 讨论暗示，如果 scaling 错误被证实广泛存在，将有机会提交一个类似于 **Unsloth** 风格的 Bug 修复报告。
- **Training Recipes 中的 Scaling Loss**：成员们达成共识，认为直接在 training recipe 中对 loss 进行 scaling 对于保持代码的显式性和清晰度更为理想。
  
  - 成员们讨论了 scaling 策略的影响，表示代码修改应附带额外的注释以方便理解。
- **Gradient Scaling 考量**：团队探讨了在不同场景下适当调整 gradient scaling 是否需要多种解决方案，特别是与 no-sync 情况相关时。
  
  - 一名成员强调，更倾向于在 training loop 中进行显式 scaling，而不是依赖 hooks 中的隐藏调整。
- **Optimizer in Backward 情况修复**：在进行中的 PR 中进行了一次更新，为 **optimizer_in_bwd** 情况添加了 scaling factor，以解决关于 loss normalization 的疑虑。
  
  - 该成员指出，目前正采取一种更清晰的方法直接在 recipe 中处理 scaling，以提高代码的可读性和可维护性。

**提及的链接**:

- [torchtune/recipes/full_finetune_distributed.py at 3518492f43a8a5a462cbd604be4101268ff5bd52 · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/3518492f43a8a5a462cbd604be4101268ff5bd52/recipes/full_finetune_distributed.py#L768): PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [torchtune/torchtune/training/memory.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/training/memory.py#L219): PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
- [transformers/src/transformers/trainer.py at 052e652d6d53c2b26ffde87e039b723949a53493 · huggingface/transformers](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3662): 🤗 Transformers: 面向 Pytorch, TensorFlow 和 JAX 的前沿机器学习库。 - huggingface/transformers
- [Add DDP token averaging for equivalent non-parallel training similar to #34191 · Issue #34242 · huggingface/transformers](https://github.com/huggingface/transformers/issues/34242): 功能请求：gradient accumulation 中的 Token averaging 已在 #34191 中修复。但 DDP 中的 token averaging 似乎存在同样的问题。预期行为：所有 token 都对 loss 做出贡献...
- [Fix gradient scaling to account for world_size normalization by mirceamironenco · Pull Request #2172 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/2172): 上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的所有 Issue。Changelog...
- [GitHub - pytorch/torchtune at 3518492f43a8a5a462cbd604be4101268ff5bd52](https://github.com/pytorch/torchtune/blob/3518492f4): PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1318872136684933151) (2 条消息):

> `Evolutionary Algorithms, Sakana Scaling, Gradient Techniques`

- **Evolutionary Algorithms 引起兴趣**：讨论强调了 **evolutionary algorithms** 在当前 AI 方法论中的有趣应用。
  
  - “它使用了进化算法，这非常有趣”引发了对其应用的关注。
- **Sakana 旨在与 Gradients 竞争**：一名成员提到 **Sakana** 正试图通过 **scale up evolution** 来匹配现有 **gradient techniques** 的性能。
  
  - 这一努力标志着将进化概念整合到具有竞争力的 AI 策略中的趋势。

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/) (1 条消息):

collabin: [https://youtu.be/BrvVheleOqc](https://youtu.be/BrvVheleOqc)

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1318687341027655800) (4 messages):

> `AI 对知识经济的影响，Chain of Continuous Thought`

- **AI 通过自主 Agent 重塑知识经济**：论文讨论了**自主 AI Agent** 如何通过让知识最渊博的人更高效地执行常规任务来使其获益最大，而非自主 AI 则通过提供专家级的问答能力来帮助知识最匮乏的人。
  
  - *随着自主 Agent 变得更加普及*，这种动态关系会发生转变，可能会增加那些拥有更多知识和技能的人所获得的收益。
- **Coconut 为 LLM 提出不受限制的推理方式**：一种名为 **Coconut (Chain of Continuous Thought)** 的新方法挑战了“语言空间始终是 Large Language Models (LLM) 推理的理想选择”这一观点，转而提倡使用潜空间模型。
  
  - 它建议减少对文本连贯性的依赖可以增强推理过程，因为某些关键的 Token 需要复杂的规划。

**提到的链接**：

- [Artificial Intelligence in the Knowledge Economy](https://arxiv.org/abs/2312.05481)：人工智能 (AI) 的兴起有可能通过实现大规模的问题解决来重塑知识经济。本文介绍了一个框架来分析这种转变……
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/html/2412.06769v1)：未找到描述

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1318971783793344573) (11 messages🔥):

> `TypedReAct 类，RouteLLM 维护，DSPy 随推理模型的发展`

- **关于** `TypedReAct` **使用的困惑**：一位成员对是否要为他们新的 `TypedReAct` 实现提交 PR 表示不确定，称其运行良好但尚未经过全面的压力测试。
  
  - 另一位成员指出，使用 `TypedChainOfThought` 是不必要的，并建议从名称中删除 “Typed”，因为该功能在 2.5 版本中已弃用，并在 2.6 版本中移除。
- **RouteLLM 缺乏维护**：一位用户注意到 [RouteLLM](https://github.com/lm-sys/RouteLLM) 项目似乎无人维护，并询问是否可以将类似功能集成到 DSPy 中。
  
  - 虽然没有针对集成给出直接回复，但这突显了对项目未来可行性的担忧。
- **关于 DSPy 和推理模型的讨论**：一位成员询问了关于 DSPy 将如何随新的推理模型发展的持续讨论，并建议微调可能会向分支或过程奖励层面（process reward level）转移。
  
  - 这表明关注点可能会从传统的 Prompting 转向增强系统内的奖励结构。

**提到的链接**：[Agents - DSPy](https://dspy.ai/tutorials/agents/)：用于对语言模型进行编程（而非 Prompting）的框架。

---

### **Nomic.ai (GPT4All) ▷ #**[**general**](https://discord.com/channels/1076964370942267462/1090427154141020190/1318701547609260185) (12 messages🔥):

> `Jinja 模板问题，GPT4All CLI 使用，GPT4All 中的 Localdocs 支持，Docker 容器版本`

- **Jinja 模板 Bug 报告**：多位成员讨论了 **Jinja 模板** 的问题，指出了诸如空格错误以及不支持 'none' 和 `[1:]` 等函数的问题。
  
  - 一位成员强调了 **Jinja** 对模型功能的重要性，而另一位成员则表示愿意在模板错误发布后协助解决。
- **GPT4All CLI 的局限性**：一位用户报告使用 **GPT4All-Cli** 访问模型，但在引用本地文档时遇到问题。
  
  - 另一位成员澄清说，旧版的 CLI 已不再获得官方支持，但当在 GUI 中启用时，**server API** 可以以编程方式访问本地文档。
- **关于 Docker 版本的咨询**：一位成员询问是否存在可以从带有 Web UI 的 Docker 容器运行的 **GPT4All** 版本。
  
  - 目前尚未收到回复，该咨询仍处于开启状态。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1318705846204633099) (2 条消息):

> `Agentic AI SDR, Composio 平台, LlamaIndex 中的 Function calling, Agentic RAG, ReAct 集成`

- **Agentic AI SDR 生成线索**：查看这个使用 LlamaIndex 构建的 [agentic AI SDR](https://t.co/tczv5ZDI4H)，它可以为你生成潜在客户，展示了其在潜在客户生成方面的潜力。
  
  - 这个 SDR 象征着 LlamaIndex 在自动化创收任务中的创新应用。
- **为 AI Agent 探索 Composio**：[Quickstarters 文件夹](https://twitter.com/llama_index/status/1869146329764831681) 是进入 Composio 的门户，使用户能够为 GitHub 和 Gmail 等平台构建智能 Agent。
  
  - 通过利用 Composio，用户可以使用自然语言命令自动化任务并提高生产力。
- **构建 Agent 的速成课程**：通过 [@TRJ_0751](https://twitter.com/llama_index/status/1869454248620269615) 的速成课程学习如何从头开始构建 Agent，重点关注 LlamaIndex 中的 Function calling。
  
  - 参与者将探索三种不同的方法来处理实时数据查询，并创建在工具之间智能路由的 Agentic RAG。
- **使用 LlamaIndex 创建 ReAct**：课程包括一个关于创建 [ReAct](https://t.co/X9IaBmdMCE) 的环节，展示了 LlamaIndex 的能力。
  
  - 这种集成体现了开发交互式 AI Agent 的创新方法。

 

**提到的链接**：[composio/python/examples/quickstarters at master · ComposioHQ/composio](https://t.co/tczv5ZDI4H)：Composio 通过 Function calling 为你的 AI Agent 和 LLM 提供 100 多个高质量集成 - ComposioHQ/composio

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1318816823642292254) (4 条消息):

> `OpenAIAgent 并发, RAG 评估, 异步函数执行`

- **探索 OpenAIAgent 并发**：一位成员询问 `OpenAIAgent` 的函数执行是否可以在异步环境中**并发**，并将其与并行 Function calling 区分开来。
  
  - 他们引用了 [OpenAI 文档](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#example-from-openai-docs)，其中解释了 API v. 1.1.0+ 中并行 Function calling 的更新功能，同时指出当前的实现不允许真正的并行计算。
- **使用 OpenAIAgent 的异步函数工具化**：作为回应，一位成员建议使用异步入口点配合异步工具，以实现 `OpenAIAgent` 执行中的并发。
  
  - *他们提供了代码片段，展示了使用* `FunctionTool` 和 `OpenAIAgent` 实现异步函数，以促进并发行为。
- **对 RAG 评估讨论的兴趣**：另一位成员表示有兴趣在 **RAG 评估**方面进行合作，邀请其他人私信交流。
  
  - 这表明了社区在检索增强生成（Retrieval-Augmented Generation）领域的互动和知识共享。

 

**提到的链接**：[Single-Turn Multi-Function Calling OpenAI Agents - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/#example-from-openai-docs)：未找到描述

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**discussion**](https://discord.com/channels/1111172801899012102/1111353033352294440/1318697465855086655) (3 条消息):

> `BFCL 排行榜问题, 用于结构化输出的 Gorilla Benchmark`

- **BFCL 排行榜演示面临问题**：一位成员报告说 [BFCL 排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) 上的 Function call 演示卡在 'Loading Model Response...'。
  
  - 另一位成员确认这是由于**证书问题**导致的，造成模型端点暂时下线。
- **对使用 Gorilla Benchmark 评估结构化输出的兴趣**：一位成员表示有兴趣使用 Gorilla Benchmark 来评估模型生成符合提供的 JSON schema 或 Pydantic 模型的文本的效果。
  
  - 他们询问在当前框架内是否有专门针对此类评估的特定子任务。

 

**提到的链接**：[Berkeley Function Calling Leaderboard V3 (又名 Berkeley Tool Calling Leaderboard V3)](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)：未找到描述

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1318856639696076810) (1 条消息):

> `GPT-O1 Reverse Engineering, Technical Reports, Twitter Updates on GPT-O1`

- **寻求 GPT-O1 逆向工程见解**：一位成员询问是否有人遇到过与 **GPT-O1** 相关的 **逆向工程工作**，并请求分享相关的论文、技术报告或 Twitter 更新。
  
  - 讨论突显了社区对 **GPT-O1** 新兴的兴趣，以及潜在的资源协作分享。
- **呼吁就 GPT-O1 展开协作**：鼓励成员分享关于 **GPT-O1** 的任何发现，特别是技术报告或在社交媒体上观察到的讨论。
  
  - 这一努力旨在促进社区参与，并收集有关 **GPT-O1** 逆向工程的相关信息。

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1318830612186140753) (1 条消息):

> `GenAI Research Internship, Generative AI advancements, Monetization AI team`

- **Meta 的 GenAI 研究实习机会**：我在 Meta 的团队正在提供一个专注于 **Generative AI** 的研究实习职位，包括文本生成图像模型（text-to-image models）和视觉语言模型（vision-language models）等领域。感兴趣的候选人可以直接在[这里](https://www.metacareers.com/jobs/539208255364368/)申请该职位。
  
  - 实习持续 **3–6 个月**，旨在推动技术的根本性进步，为实习生提供在大规模场景下实现 **核心算法突破** 的机会。
- **对 Generative AI 领域的贡献**：Meta 的变现 Generative AI 团队正在寻找对 **Deep Learning**、**Computer Vision** 和 **Natural Language Processing** 充满热情的人才。实习生将有机会影响全球人们的连接和沟通方式。
  
  - 该团队致力于推进 **Generative AI** 研究，并在一个 **快速发展的组织** 中应用创新想法。

 

**提到的链接**：[Research Scientist Intern, Monetization AI (PhD)](https://www.metacareers.com/jobs/539208255364368/)：Meta 的使命是构建人类连接的未来以及实现这一目标的各种技术。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/) (1 条消息):

kallemickelborg: 谢谢！

---

### **Axolotl AI ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1318904347194556490) (1 条消息):

> `New engineer for RL, KTO assistance`

- **新工程师加入以支持 RL**：一名新工程师将于 **1 月** 加入团队，协助 **Reinforcement Learning (RL)** 计划。
  
  - 届时他们还将为 **KTO** 项目提供支持。
- **KTO 的额外资源**：新工程师入职后将致力于增强 **KTO** 系统的功能。
  
  - 这一补充预计将提高 **RL** 相关任务的整体生产力。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1318691196566241371) (1 条消息):

> `Developer Hub Announcement, Blueprints Initiative`

- **开发者中心发布激动人心的功能**：关于 **Developer Hub** 及其最新功能发布了一项 **重大公告**，强调了社区反馈和参与的必要性。
  
  - 您可以在[这里](https://discord.com/channels/1089876418936180786/1230938514955436242/1318638353503227935)查看完整公告以获取所有详细信息。
- **针对开源 AI 解决方案的 Blueprints 计划**：**Blueprints 计划** 旨在协助开发者创建 **开源 AI 解决方案**，提供必要的资源和指导。
  
  - 更多见解和讨论可以在此[线程](https://discord.com/channels/1089876418936180786/1318689803021058158)中找到。

 

---

---

---

{% else %}

> 完整的频道细分内容已为邮件格式截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}