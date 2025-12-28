---
companies:
- openai
- cohere
- financial-times
date: '2024-05-01T01:41:25.208668Z'
description: '**OpenAI** 已向所有 ChatGPT Plus 用户推出了**记忆功能**，并与**《金融时报》**达成合作，获得其内容授权用于
  AI 训练。由于付费训练数据授权以及 **GPT-4 使用限制可能缩减**，引发了关于 **OpenAI 盈利能力**的讨论。有用户反映，在记忆功能更新后，ChatGPT
  出现了数据清理方面的问题。


  相关的教程和项目包括构建由大语言模型（LLM）驱动的 AI 语音助手和界面智能体。在 **Stable Diffusion** 领域，用户正在寻找可媲美 PonyXL
  的写实 **SDXL 模型**；同时，**Hi-diffusion** 和 **Virtuoso Nodes v1.1** 等新插件为 ComfyUI 增强了高级图像生成及类
  Photoshop 的功能。**Cohere** 的研究发现，在 LLM 评判任务中，多智能体的表现优于单智能体，这凸显了多智能体系统的技术进展。'
id: ad06ca0b-4375-46a3-a673-3d829bbb1f66
models:
- gpt-4
- gpt-3.5
- sdxl
- ponyxl
original_slug: ainews-to-be-named-4408
people: []
title: 大语言模型作为陪审团 (LLMs-as-Juries)
topics:
- memory
- training-data
- model-usage-limits
- data-cleansing
- ai-voice-assistants
- interface-agents
- image-generation
- model-extensions
- multi-agent-systems
---

 

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

以下是更新后的摘要，采用了要求的格式并降低了 AGI 相关帖子的权重：

**OpenAI 新闻**

- **Memory 功能现已面向所有 ChatGPT Plus 用户开放**：OpenAI [在 Twitter 上宣布](https://twitter.com/OpenAI/status/1784992796669096181)，Memory 功能现已推送到所有 ChatGPT Plus 订阅用户。
- **OpenAI 与《金融时报》（Financial Times）合作，将 AI 应用于新闻领域**：OpenAI 已[签署协议，获得《金融时报》的内容授权](https://www.reuters.com/technology/financial-times-openai-sign-content-licensing-partnership-2024-04-29/)以训练其 AI 模型。官方[分享了一张图片](https://i.redd.it/s09mjga1jgxc1.jpeg)宣布这一合作伙伴关系，旨在开发新闻领域的 AI 体验。
- **对 OpenAI 在支付训练数据费用后的盈利能力的担忧**：在 /r/OpenAI 中，一则[帖子质疑](https://www.reddit.com/r/OpenAI/comments/1cfxd42/how_is_openai_going_to_be_profitable_if_they_have/) OpenAI 在开始支付内容授权费用后的盈利能力，推测本地开源模型可能会削弱其业务。
- **GPT-4 使用限制可能有所降低**：一位 [/r/OpenAI 的用户注意到](https://www.reddit.com/r/OpenAI/comments/1cfxzvl/has_openai_reduced_the_number_of_questions/) GPT-4 的使用限制已从每 3 小时 40 条消息降至每小时约 20 个问题。
- **Memory 更新后 ChatGPT 出现的问题**：在 /r/OpenAI 中，一位用户[发现 ChatGPT 在 Memory 更新后](https://www.reddit.com/r/OpenAI/comments/1cg8zsd/chatgpt_laziness_data_cleansing_and_analysis_is/)在数据清洗和分析任务上表现挣扎，会出现错误和不完整的输出。

**OpenAI API 项目与讨论**

- **使用 OpenAI 构建 AI 语音助手的教程**：/r/OpenAI 分享了一篇[博客文章](https://www.reddit.com/r/OpenAI/comments/1cgh184/how_i_build_an_ai_voice_assistant_with_openai/)，介绍如何结合 OpenAI API 和 Web Speech API 构建 AI 语音助手。
- **AI 驱动的副业项目讨论**：在 /r/OpenAI 中，一则[帖子邀请他人分享](https://www.reddit.com/r/OpenAI/comments/1cg5mm7/whats_your_ai_backed_side_project/)他们的 AI 驱动副业项目。发帖者展示了一个使用 GPT-4 制作的需求分析工具，以及一个使用 GPT-3.5 制作的互动德语导师。
- **由 LLM 驱动的界面 Agent（Interface agents）**：/r/OpenAI 的一则[帖子讨论了“界面 Agent”](https://www.reddit.com/r/OpenAI/comments/1cg3f2z/p_interface_agents_building_llmenabled_agents/)——即能够与浏览器和应用程序等用户界面进行交互并控制它们的 AI。内容涵盖了关键组件、工具、挑战和用例。
- **在 GPT-4 生成的图像中调整元素大小的困难**：在 /r/OpenAI 中，一位[用户寻求建议](https://www.reddit.com/r/OpenAI/comments/1cga0zy/best_way_to_tell_gpt4_to_shrink_something_in_a/)，如何指示 GPT-4 缩小生成图像中的某个元素，因为该模型在一致地调整物体大小方面表现不佳。

**Stable Diffusion 模型与扩展**

- **寻找与 PonyXL 媲美的写实 SDXL 模型**：在 /r/StableDiffusion 中，一位[用户询问](https://www.reddit.com/r/StableDiffusion/comments/1cfv7ga/any_realistic_sdxl_model_as_good_as_ponyxl/)是否有在质量和摄影风格提示词对齐（prompt alignment）方面能与 PonyXL 相提并论的写实 SDXL 模型。
- **ComfyUI 的 Hi-diffusion 扩展**：一位 /r/StableDiffusion 用户[发现 Hi-diffusion 在 ComfyUI 中表现出色](https://www.reddit.com/r/StableDiffusion/comments/1cg2394/hidiffusion_is_very_impressive_now_the_comfyui/)，配合 SD1.5 模型可生成细节丰富的 2K 图像，性能优于 Khoya deep shrink。目前已有相关扩展，但仍需改进。
- **Virtuoso Nodes v1.1 为 ComfyUI 引入 Photoshop 功能**：[Virtuoso Nodes 的 1.1 版本](https://www.reddit.com/r/StableDiffusion/comments/1cgexi9/virtuoso_nodes_release_v11_with_new_photoshop/)已为 ComfyUI 发布，新增了 8 个节点，可模拟 Photoshop 的核心功能，如混合模式、可选颜色、色彩平衡等。
- **在 Fooocus 中简化 Pony XL 提示词的样式 (Styles)**：一位 /r/StableDiffusion 用户[为 Fooocus 创建了样式](https://www.reddit.com/r/StableDiffusion/comments/1cglyq4/styles_for_fooocus_to_shorten_your_pony_xl/)来处理 Pony XL 提示词中的质量标签，从而实现更简洁、更专注于内容的提示词。
- **动漫风格阴影 LoRA 发布**：一款[动漫风格阴影 LoRA](https://huggingface.co/2vXpSwA7/iroiro-lora/blob/main/test3/sdxl-shadow_01.safetensors) 发布，建议配合 Anystyle 和其他 ControlNet 使用。文中提供了该 LoRA 文件的 Hugging Face 链接。

**Stable Diffusion 帮助与讨论**

- **避免生成图像中出现显式内容**：在 /r/StableDiffusion 中，一位用户在[生成的图像中 80% 都出现了生殖器元素](https://www.reddit.com/r/StableDiffusion/comments/1cgjrds/80_of_my_generated_pics_have_dicks_coming_out_of/)，因此寻求负面提示词（negative prompt）建议，以便生成“常规色情内容”。
- **使用 AI 图像和动态文本创建短视频剪辑**：/r/StableDiffusion 的一篇[帖子询问了相关 API](https://www.reddit.com/r/StableDiffusion/comments/1cfwxct/how_to_create_short_videos_by_using_ai_images_and/)，旨在生成带有动态文本叠加的 AI 图像，从而制作短视频剪辑。
- **尽管游戏性能提升，新款 Nvidia GPU 在 AI 任务中可能变慢**：有[警告指出](https://www.reddit.com/r/StableDiffusion/comments/1cg0gz6/be_careful_when_buying_new_nvidia_card_or_laptop/)，像 4070 笔记本版这样的新款 Nvidia GPU 使用的显存总线（memory bus）比旧款更窄，导致其在 AI 工作负载中的速度变慢。
- **社区图像打标项目提案**：/r/StableDiffusion 的一篇[帖子建议发起社区协作](https://www.reddit.com/r/StableDiffusion/comments/1cgbivm/community_effort_for_best_image_tagging/)，对图像进行全面打标，以创建一个具有一致说明文字（caption）的图像数据集，用于训练更好的模型。
- **使用 VAE 进行图像压缩**：在 /r/StableDiffusion [分享的实验](https://www.reddit.com/r/StableDiffusion/comments/1cgdyjc/vae_as_image_compression/)表明，在某些情况下，使用 VAE latents 进行图像压缩的性能可与 JPEG 媲美。将生成的图像保存为 latents 是无损的，且体积远小于 PNG。
- **从头像生成全身照**：在 /r/StableDiffusion 中，一位[用户询问](https://www.reddit.com/r/StableDiffusion/comments/1cg3a4z/help_me_with_this_will_pay/)是否可以在不大幅改变面部的情况下，使用 SD Forge 从头像图像生成全身照。
- **Audrey Hepburn 的 Textual Inversion 模型**：一位 /r/StableDiffusion 用户[制作了 Audrey Hepburn 的 Textual Inversion 模型](https://www.reddit.com/r/StableDiffusion/comments/1cft1gp/give_you_a_slightly_different_audrey_hepburn/)，可以生成相似但各具特色的面部，并分享了示例图像和 Civitai 链接。

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，从 4 次运行中择优。我们正在尝试使用 Haiku 进行聚类和流程工程 (flow engineering)。

**LLM 与 AI 模型**

- **Llama 3 性能**：[@abacaj](https://twitter.com/abacaj/status/1785147493728039111) 指出，无需训练的 Llama 3 模型即可获得 **32k 上下文且质量卓越**，超越了规模大得多的模型。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1784889182558539917) 提到 Llama 3 捕捉到了极其细微的数据关系，甚至利用了 BF16 精度中最小的小数位，这使得它相比 Llama 2 **对量化损耗（quantization degradation）更敏感**。
- **Llama 3 基准测试**：[@abacaj](https://twitter.com/abacaj/status/1785295341736043007) 报告称 Llama 3 70B 在**某项基准测试中位列第三，取代了 Haiku**。[@abacaj](https://twitter.com/abacaj/status/1785153286976237765) 分享了该模型在**代码片段基准测试**中的补全结果，该测试要求模型根据描述查找函数。
- **Llama 3 变体**：[@mervenoyann](https://twitter.com/mervenoyann/status/1785320444918211022) 注意到**基于 Llama 3 和 Phi-3 的新型类 LLaVA 模型**通过了 baklava 基准测试。[@AIatMeta](https://twitter.com/AIatMeta/status/1785042326416658580) 提到了 Meditron，这是一个由 @ICepfl 和 @YaleMed 研究人员为低资源医疗环境构建的 LLM 套件，它在 MedQA 和 MedMCQA 等基准测试中，使用 Llama 3 **在同参数级别中表现优于大多数开源模型**。
- **GPT-2 Chatbot**：关于 gpt2-chatbot 模型的身份存在诸多猜测，[@sama](https://twitter.com/sama/status/1785107943664566556) 提到他对 gpt2 情有独钟。一些理论认为它可能是 GPT-4.5/5 的预览版或衍生模型，但大多数人认为它**不太可能是最新的 OAI 模型**。 
- **Phi-3 及其他模型**：[@danielhanchen](https://twitter.com/danielhanchen/status/1785040680106234225) 发布了一个 **Phi-3 notebook，其微调速度比 HF+FA2 快 2 倍，且显存（VRAM）占用减少 50%**。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785220060803453160) 分享了一篇论文，认为 Transformer 通过在其前向传播（forward pass）中**对根据上下文数据构建的损失函数执行梯度下降**来实现上下文学习（in-context learning）。

**Prompt Engineering 与评估**

- **Prompt Engineering 技术**：[@cwolferesearch](https://twitter.com/cwolferesearch/status/1784992130777137362) 将最近的 Prompt Engineering 研究归类为**推理、工具使用、上下文窗口和更好的写作**。技术包括 zero-shot CoT 提示、基于复杂度选择示例、优化推理过程（rationales）、任务分解、使用 API、优化上下文窗口以及迭代提示。
- **LLM 作为陪审团**：[@cohere](https://twitter.com/cohere/status/1785284142789242932) 发布了一篇论文，探讨在评估中**用多个 LLM 陪审团（juries）取代单一 LLM 裁判（judge）**。这种使用多样化 LLM 集合的 “PoLL” 方法在各数据集上的**表现优于单一裁判，且成本比 GPT-4 低 7-8 倍**。
- **评估 LLM**：[@_lewtun](https://twitter.com/_lewtun/status/1785246966626029596) 询问了除了 @lmsysorg 的工作之外，还有哪些研究探讨了哪些提示能使 LLM 裁判与人类对成对排名（pairwise rankings）的偏好最相关。[@_philschmid](https://twitter.com/_philschmid/status/1785273493375922221) 总结了 @cohere 提出的用于 LLM 评估的 **PoLL (Panel of LLM) 方法**，作为单一大型模型裁判的替代方案。

**应用与用例**

- **财务计算**：[@llama_index](https://twitter.com/llama_index/status/1785325832317415641) 分享了一个全栈教程，用于构建财务助手，该助手可以使用 @llama_index 中的 LlamaParse、RAG、Opus 和数学公式，**针对非结构化财务报告计算百分比演变、CAGR（复合年增长率）和 P/E（市盈率）**。
- **SQL 查询生成**：[@virattt](https://twitter.com/virattt/status/1785059112478257413) 使用 @cohere cmd r+ 在约 1 秒内**从财务查询中提取股票代码（ticker）和年份元数据**，然后使用元数据过滤向量数据库（vector db），将结果输入 GPT-4，并以约 3 秒的总延迟回答用户查询。
- **多 Agent RAG**：[@LangChainAI](https://twitter.com/LangChainAI/status/1785066609847291986) 宣布了一个 YouTube 工作坊，探索“多 Agent”应用，这些应用利用规划（planning）、反思（reflection）、工具使用以及他们的 LangGraph 库，**结合独立 Agent 来解决复杂问题**。
- **机器人与具身智能（Embodied AI）**：[@DrJimFan](https://twitter.com/DrJimFan/status/1785292766387302897) 主张**机器人是 LLM 之后的下一个前沿领域**，分享了 MIT AI Lab 1971 年强调机器人的提案，并对现状进行了反思。[@_akhaliq](https://twitter.com/_akhaliq/status/1785139220534730771) 分享了一篇关于 Ag2Manip 的论文，该论文利用 Agent 无关的视觉和动作表示，**改进了操作任务的模仿学习**。

**框架、工具与平台**

- **LangChain 教程**：[@LangChainAI](https://twitter.com/LangChainAI/status/1784970647875330251) 分享了一个 **4 小时的课程，旨在理解 LangChain 如何与各种技术协作**来构建 6 个项目。[@llama_index](https://twitter.com/llama_index/status/1784962053641478454) 提供了一个使用 LlamaParse, AWS Bedrock 和 @llama_index 的 **高级 RAG 参考架构**。
- **Diffusers 库**：[@RisingSayak](https://twitter.com/RisingSayak/status/1785162074844197174) 解释了 Diffusers 库如何 **支持自定义 Pipeline 和组件**，在保持 `DiffusionPipeline` 类优势的同时，为构建 Diffusion 模型提供了灵活性。
- **Amazon Bedrock**：[@cohere](https://twitter.com/cohere/status/1785015769971220720) 宣布他们的 **Command R 模型系列现已在 Amazon Bedrock 上线**，用于企业级工作负载。[@llama_index](https://twitter.com/llama_index/status/1785105949818237227) 展示了如何使用 LlamaParse 在 AWS/Bedrock 生态系统中进行高级解析，并 **利用 Bedrock Knowledge Base 构建 RAG**。
- **DeepSpeed 支持**：[@StasBekman](https://twitter.com/StasBekman/status/1785091895733154116) 指出，一个合并到 `main@accelerate` 的 PR 使得 FSDP 在加载 fp16 模型时，通过自动将可训练参数上采样到 fp32，从而达到 **与 DeepSpeed 相同的收敛速度**。

**迷因、幽默及其他**

- **ASCII 艺术**：几条推文嘲讽了 LLM 的 ASCII 艺术能力，[@ylecun](https://twitter.com/ylecun/status/1785109502565531699) 指出 **AI 炒作已经变得与讽刺作品无异**。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1785325820166185399) 分享了一个使用 emoji 绘制 Katamari Damacy 关卡地图的 Prompt，这极大地考验了 "GPT2" 的指令遵循能力。
- **Anthropic Slack**：[@alexalbert__](https://twitter.com/alexalbert__/status/1785369914204938326) 分享了他从 Anthropic 内部 Slack 频道中挑选的 **10 个最爱内容**，自发布以来，员工们一直在那里发布酷炫的 Claude 交互和迷因。
- **对 Rabbit 的失望**：几位用户对 Rabbit AI 设备表示失望，指出其 **功能与预期相比非常有限**。[@agihippo](https://twitter.com/agihippo/status/1785359480294936882) 质疑 Rabbit r1 有什么功能是手机做不到的。

---

# AI Discord 总结

> 总结的总结之总结

**1) 微调与优化大语言模型**

- **LLaMA-3 微调中的挑战**：工程师们面临着模型 **不生成 EOS tokens** 以及 **不同位格式间的 Embedding 层兼容性**等问题。然而，一位成员通过利用 **[LLaMA-3 特定的 Prompt 策略](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553)** 进行微调取得了成功。

- **LLaMA-3 对量化敏感**：讨论强调，与 LLaMA-2 相比，**[LLaMA-3 在量化过程中经历了更多的性能退化](https://x.com/rohanpaul_ai/status/1784972618472317180)**，这可能是因为它从 15T tokens 的训练中捕捉到了更细微的关系。

- **Perplexity 微调挑战**：为 **Perplexity（困惑度）微调 LLaMA-3** 可能无法超越 Base 模型的性能，Tokenizer 被怀疑是潜在原因。

**2) 扩展上下文长度与能力**

- **Llama-3 创下上下文长度新高**：**[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)** 的发布将上下文长度从 8k 扩展到了超过 1048k tokens，展示了最前沿的长上下文处理能力。

- **Llama 3 通过 SigLIP 获得视觉能力**：一项突破性进展利用 SigLIP 为 **[Llama 3 集成了视觉能力](https://huggingface.co/qresearch/llama-3-vision-alpha-hf)**，尽管存在量化限制，但仍可直接在 Transformers 中使用。

- **使用 PoSE 将上下文扩展到 256k**：**Llama 3 8B** 的上下文长度已通过 **[PoSE 扩展到 256k tokens](https://huggingface.co/winglian/llama-3-8b-256k-PoSE)**，但在“大海捞针”（needle in haystack）场景下仍面临推理挑战。

**3) LLM 基准测试与评估**

- **Llama 3 在德语 NLG 中表现优于 GPT-4**：在 **[ScanEval German NLG 基准测试](https://scandeval.com/german-nlg/)** 中，**Llama 3** 的表现超过了 **GPT-4**，显示出其强大的语言生成能力。

- **神秘的 GPT2-Chatbot 引发猜测**：一个具有 GPT-4 级别能力的神秘 **[GPT2-chatbot](https://chat.lmsys.org/)** 出现，引发了关于它是 **GPT-4.5** 的早期预览还是原始 GPT-2 的微调版本的争论。

- **质疑代码生成排行榜的实用性**：一篇 **[博客文章](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)** 质疑了 AI 排行榜在代码生成方面的有效性，理由是像 LLM debugger 这样排名靠前的模型运行成本极高。

**4) 利用 LLM 驱动的 NPC 变革游戏体验**

- **LLM 驱动的 NPC 和推理栈**：**[LLM 驱动的 NPC 模型](https://github.com/GigaxGames/gigax)** 的发布旨在增强动作空间并简化 API 调用，包括单次 LLM 调用功能以及在 Hugging Face 上的开放权重。

- **克服游戏中的 LLM 挑战**：开发者面临着诸如 **NPC 打破第四面墙**、长 Prompt 中细节丢失以及运行时速度优化等问题，并提出了 **输出压缩**、**减少模型调用** 以及利用 **更小模型** 等解决方案。

- **NPC 场景下微调 LLM 的见解**：开发者计划通过即将发布的博客文章分享他们在 **为动态 NPC 行为微调 LLM 过程中的挣扎与胜利**，为游戏应用提供新的策略。


**5) 杂项**

- **CUDA 优化技术**：CUDA 开发者讨论了各种优化策略，包括使用 `Packed128` 自定义结构体来优化内存访问模式，使用位移替代整数除法（[Compiler Explorer 链接](https://godbolt.org/z/9K9Gf1v6P)），以及比较 **CUTLASS vs CuBLAS** 在矩阵乘法中的性能。引入了 **Effort Engine** 算法，该算法允许在 LLM 推理期间调整计算量，从而在 Apple Silicon 上实现与标准矩阵乘法相当的速度（[kolinko.github.io/effort](https://kolinko.github.io/effort), [GitHub](https://github.com/kolinko/effort)）。

- **LLaMA-3 上下文长度扩展与微调**：使用 **PoSE** 技术将 **LLaMA-3 8B** 模型的上下文长度扩展到了超过 1M tokens（[huggingface.co/winglian/llama-3-8b-256k-PoSE](https://huggingface.co/winglian/llama-3-8b-256k-PoSE)），引发了关于其检索性能和计算需求的讨论。微调 LLaMA-3 带来了诸如 **量化退化**、**EOS token 生成** 以及跨位格式的 **embedding 层兼容性** 等挑战。一个 [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553) 分享了潜在的突破，展示了使用特定模型 Prompt 策略的成功微调。

- **Civitai 变现引发的抵制**：Stable Diffusion 社区成员对 **Civitai 的变现策略** 表示不满，特别是 **Buzz 捐赠系统**，被 Tower13Studios 等人称为“坑人”（[The Angola Effect](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9)）。讨论还强调了与饱和的 SFW 市场相比，**NSFW AI 生成艺术委托** 的潜在盈利能力。

- **Perplexity AI 性能问题**：用户报告在日本黄金周期间，各种 Perplexity AI 模型出现显著变慢和性能不佳的情况，特别是 **日语搜索** 导致输出无意义。用户对 **过期的 Pro 订阅优惠券** 和取消 **7 天免费试用** 表示沮丧。技术问题包括影响登录的 **邮件链接延迟** 以及取决于应用版本的 **iOS 语音功能** 不一致。

- **去中心化 AI 训练倡议**：Prime Intellect 提出了一种使用 **H100 GPU 集群** 的去中心化训练方法，使开源 AI 能够与专有模型竞争（[博客文章](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)）。该倡议旨在通过利用全球分布的 GPU 资源来解决计算基础设施的限制。

---



# 第 1 部分：高层级 Discord 摘要

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 难题**：工程师们讨论了 **Triton blocks** 的限制，发现 4096 个元素的 block 是可行的，但 8192 个元素的 block 则不行，这暗示了与预期的 **CUDA** 限制存在差异。

- **CUDA 认知与协作**：讨论了各种 **CUDA** 主题，包括 **CUTLASS vs. CuBLAS 性能**、**CUDA checkpointing** 以及用位移代替整数除法。分享了一个 [Compiler Explorer](https://godbolt.org/z/9K9Gf1v6P) 链接以辅助实验。
  
- **追踪 PyTorch 特性**：成员们研究了 PyTorch `linear` 函数和矩阵乘法 kernel 启动的行为，观察到了双重 kernel 启动现象，以及由于转置导致的性能差异的错误预期。

- **使用 Effort Engine 优化 LLM 推理**：讨论围绕 **Effort Engine** 算法展开，该算法允许在 **LLM** 推理期间调整计算量（computational effort），据称在较低计算量下，其速度可与 Apple Silicon 上的标准矩阵乘法相媲美。实现和详情见 [kolinko.github.io/effort](https://kolinko.github.io/effort) 和 [GitHub](https://github.com/kolinko/effort)。

- **InstaDeep 的机器学习人才招聘**：**InstaDeep** 正在寻找在**高性能 ML 工程、自定义 CUDA kernels 和分布式训练**方面具有专业知识的 **Machine Learning Engineers**。候选人可以在 [InstaDeep Careers](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/) 查看机会。 

- **Llama-3 迈向更长上下文**：[Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) 的发布为 **LLM** 的上下文长度能力设定了新基准。

- **ROCm 助力 Flash Attention 2**：**ROCm** 频道的对话集中在将 NVIDIA 的 Flash Attention 2 适配到 **ROCm**，重点是与 **ROCm 6.x** 版本的兼容性，并提供了相关仓库链接 [ROCm/flash-attention on GitHub](https://github.com/ROCm/flash-attention)。

- **CUDA 秘密会议聚焦 “Packed128” 创新**：**llmdotc** 频道是一个热点，讨论集中在优化 `Packed128` 数据结构和 **BF16 混合精度策略**，同时也涉及了 **NVTX** 上下文的细微用法以及 **Modal** 等不同基准测试工具集的效用。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **融合 Checkpoints 以避免过拟合**：一位成员寻求关于 checkpoint 合并以避免过拟合的指导，并被引导至 Unsloth [finetuning checkpoint wiki](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint)。对于精细的训练方案，建议使用 *warmup steps* 和 *resuming from checkpoints* 等技术。

- **WSL2 中的量化困境**：用户报告在 WSL2 中将模型转换为 F16 时出现 **RuntimeError: Unsloth: Quantization failed**。尽管尝试重新构建 `llama.cpp` 并重新量化，错误仍然存在。

- **Phi-3：备受关注的模型**：即将发布的 **Phi-3** 引起了兴趣，工程师们在辩论是采用 3.8b 版本，还是等待更强大的 7b 或 14b 变体。

- **OOM 对策与性能数据混淆**：交流了通过清理缓存处理 Google Colab 上 Out of Memory (OOM) 错误的技巧。同时，对于量化后的 **Llama 2** 和 **Llama 3** 报告的性能指标出现了混淆，暗示 Bits Per Word (BPW) 和 Perplexity (PPL) 之间可能存在数据放错位置的情况。

- **扩展的可能性**：**Llama 3 8B** 通过 **[PoSE](https://huggingface.co/papers/2309.10400)** 将上下文长度增加到 256k token，达到了新的潜力，展示在 [winglian/llama-3-8b-256k-PoSE](https://huggingface.co/winglian/llama-3-8b-256k-PoSE)。社区对 Winglian 表示赞赏，尽管一些人对非官方上下文扩展模型的行为表示怀疑。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Groq 给 Discord 机器人的礼物**：一位用户分享了一个 [YouTube 视频](https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB)，重点介绍了 *免费* 的 Groq API，它能让 LLAMA-3 模型达到每秒 300 tokens 的惊人速度，由于其零成本设置，非常适合小型服务器的 Discord 机器人。
- **规格大比拼**：用户建议在排查 **Ubuntu GPU 上的 LM Studio** 问题时，在特定频道发布系统规格（specs），辩论了 GPU 与 **inference 任务** 的兼容性，并讨论了 LM Studio 中可能不正确的 VRAM 容量显示，这引发了对 **GPU offloading 效率** 的担忧。
- **模型狂热**：社区热议从 Huggingface 以外的来源下载 GGUF 模型的替代方法，创建 *iQuants* 和 *imatrices* 的时间和资源需求，并分享了优化 **Goliath 120B Longlora** 模型以创建其 *iQuant* 版本的悬赏。
- **低配机器上的模型混乱**：用户正在努力解决 Phi-3 模型的 **提示词泄漏（leaking prompts）**、基于 Hugging Face 模型的 *local training* 咨询，以及 Llama3m 在生成 token 时硬盘发出的意外噪音等问题。一些人断定，较旧的硬件勉强可以应付 **7b Q4 模型**，但再大的就无能为力了。
- **ROCm 沉思**：爱好者们剖析了 ROCm 版本，思考 **beta 0.2.20** 对 AMD 功能的好处，解决了关于兼容性的困惑——特别是 RX 6600 对当前 **HIP SDK** 的支持——并讨论了 ROCm 在不同操作系统（如 **Ubuntu 与 Windows**）上功能的差异。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**滚开，Civitai**：公会中的 AI 创作者对 Civitai 的变现策略感到不满，特别是 Buzz 捐赠系统，被 Tower13Studios 等成员贴上了 **“坑人（rip-off）”** 的标签。不满情绪集中在价值没有公平地回馈给创作者（[安哥拉效应](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9)）。

**寻找 AI 艺术金矿**：关于 AI 生成艺术经济学的一场热烈讨论展开了，共识指向 NSFW 委托（包括 furry 和 vtuber 内容），认为这比拥挤的 SFW 市场更有利可图。

**实时渲染竞赛**：成员们积极分享用于加速 Stable Diffusion (SDXL) 模型的 Python 脚本技术，着眼于 Discord 机器人等动态领域的应用，旨在提高实时应用的图像生成速度。

**对 Collider 的期待与日俱增**：社区正热切期待 Stable Diffusion 的下一个迭代版本，代号为 “Collider”，关于发布日期和潜在进步的猜测激发了用户的热切期待。

**技术故障排除讨论**：公会成员就一系列技术挑战交换了见解和解决方案，从创建 LoRAs 和 IPAdapters 到在低配硬件上运行 AI 模型，展示了推动模型实现和优化边界的集体努力。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **日本黄金周期间的故障**：在日本黄金周期间，用户观察到 **Opus、Sonar Large 32K** 和 **GPT-4 Turbo** 等工具的性能明显下降，特别是在日语搜索方面出现了特定问题，导致输出被用户视为“无意义的垃圾内容”。为了解决这一问题，建议对这些模型进行严密监控和优化。

- **对 Pro 订阅和试用风险的挫败感**：**Pro 订阅**用户反映优惠券在到期日失效，而与 **Nothing Phone 2(a)** 相关的优惠活动因欺诈活动而提前终止。此外，网站取消了 7 天免费试用，引发了用户的失望，强调了其作为用户转化工具的价值。

- **Perplexity AI 的技术动荡**：社区正在应对 **Email 链接延迟**问题，这导致了登录困难，尤其是对于非 Gmail 服务。此外，**iOS 语音功能**的差异被发现取决于所使用的 **App 版本**，反映了用户体验的不一致。

- **API 途径探索**：工程师们在 **pplx-api** 频道询问了关于通过 API 访问 **Source URL** 的问题（此前在路线图文档中提及），并讨论了使用 **Claude 3** 是否需要遵守 Perplexity 条款下 **Anthropic 的政治用途**限制。

- **杂项查询与见解浮现**：**#[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234586871569449121)** 频道的一篇帖子重点介绍了 Lenny 关于产品增长和构建概念的通讯，同时还涉及了关于 WhatsApp 自动回复功能和 Vimeo API 的咨询。这些讨论（尤其是关于 API 的讨论）突显了工程师们对在系统/流程中集成和利用各种功能的关注。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**大胆的去中心化举措**：Prime Intellect 发起的去中心化 AI 训练倡议，利用 **H100 GPU 集群**，承诺通过全球化分布式训练来突破界限。正如其[去中心化训练博客](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)中所讨论的，这种开源方法可能会解决当前的计算基础设施瓶颈。

**LLama-3 引发的检索革命**：**LLama-3 8B** 的上下文长度扩展到超过 1040K tokens，引发了关于其检索性能是否名副其实的讨论。怀疑论者依然存在，强调持续改进和训练的必要性，并引用了一篇关于 [IN2 训练的 ArXiv 论文](https://arxiv.org/abs/2404.16811)作为支持。

**解决 PDF 挑战**：为了解决 AI 模型中的 PDF 解析挑战（尤其是表格解析），社区讨论了变通方案和工具，如 [OpenAI 的 File Search](https://platform.openai.com/docs/assistants/tools/file-search)，以实现更好的多模态功能，处理约 1 万个文件。

**世界模拟器展示 AI 的角色扮演实力**：与 AI 驱动的世界模拟（World Sims）的互动展示了 **LLama 3 70B** 和 **Claude 3** 的能力，涵盖了从历史人物到商业和歌唱事业模拟器。OpenAI 在 [HuggingChat](https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8) 上的聊天以及指向 [Snow Singer Simulator](https://hf.co/chat/assistant/6626e4869232378718adc5f2) 等小众模拟器的链接，反映了可以实现的样性与深度。

**利用数据集进行多语言密集检索**：HuggingFace 上一个著名的 [Wikipedia RAG 数据集](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7)标志着提升 AI 语言检索能力的兴起。其中包含的 Halal 和 Kosher 数据点指向了创建多样化和包容性 AI 资源的趋势。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的内存安全和并发性引发讨论**：尽管 **Mojo** 的潜力备受关注，但由于 **borrow checking** 被禁用，目前尚未实现类似 **Golang** 的并发和类似 **Rust** 的内存安全。然而，正在探索使用 actor model 并发性的可能性，这可能会提高 Mojo 的运行时效率。

- **不同系统上的 Mojo 安装策略**：用户在 **Mac M1** 上使用 **Python 3.12.3** 安装 **Mojo** 时面临挑战，建议使用 **Conda 环境**。此外，虽然原生 **Windows 支持** 尚在开发中，但 **Windows 上的 WSL** 是目前的权宜之计，并通过 **LLVM** 暗示了交叉编译能力。

- **社区对 Mojo 生态系统的贡献**：多个社区驱动的项目正在增强 Mojo 生态系统，从 GitHub 上的 Mojo 论坛到针对长字符串优化了 **20% 性能** 的 atof-simd 项目。随着成员分享项目并呼吁共同应对 1brc 等挑战，协作和知识共享的热情显而易见。

- **Nightly 编译版本引发关于 SIMD 和 Source Location 的讨论**：**Mojo 编译器** 的新 **nightly** 版本引发了关于 **SIMD** 转换为 **EqualityComparable** 的讨论，以及需要显式的 `reduce_and` 或 `reduce_or` 来替代隐式转换为 `Bool`。将 `__source_location()` 移至 `__call_location()` 引起了关于语言内正确用法的交流。

- **性能和基准测试成为焦点**：从优化基于 SIMD 的纠错码到分享 1brc 项目中的显著速度提升，性能话题引发了关于 **LLVM/MLIR 优化** 的讨论。有人呼吁组建 "team-mojo" 进行社区挑战攻关，强调了在 Mojo 与其他语言的基准测试对比中取得进展的共同兴趣。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Snowflake 的 MoE 模型取得突破**：Snowflake 推出了一个[具有 408B 参数的庞大 Dense + Hybrid MoE 模型](https://x.com/reach_vb/status/1783129119435210836)，拥有 4K 上下文窗口，完全采用 Apache 2.0 许可证，其在复杂任务上的表现令人兴奋。

**Gradio 分享服务器出现故障**：Gradio 承认其[分享服务器存在问题](https://status.gradio.app/)，影响了 Colab 集成，目前正在积极解决中，其状态页面提供更新。

**CVPR 2023 激发竞争精神**：CVPR 2023 [宣布了竞赛活动](https://huggingface.co/spaces/BVRA/SnakeCLEF2024)，如 SnakeCLEF、FungiCLEF 和 PlantCLEF，奖金超过 12 万美元，将于 2024 年 6 月 17 日至 21 日举行。

**MIT 深度学习课程上线**：MIT 更新了其 2024 年深度学习导论课程，并在 [YouTube 上提供了完整的讲座视频](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2)。

**聊天机器人领域的 NLP 难题**：在 NLP 社区中，尽管在意图识别和分类方面存在困难，但人们正努力使用 Rasa 框架微调聊天机器人，并计划通过自定义 NER 模型和公司特定意图来增强性能。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Alex Atallah 指出与 Syrax 的合作**：Alex Atallah 已开始与 **Syrax** 进行实验，并通过提议建立群聊来扩展支持以进行协作，这标志着合作伙伴关系的开始，得到了 Mart02 的热情回应。

- **面向普通用户的前端**：社区探索了在没有高级技术要求的情况下，在共享主机上部署多用户前端的解决方案。**LibreChat** 被建议作为一个可行的平台，并提到 Vercel 的免费层托管是解决托管和成本障碍的一种手段。

- **LLM 大比拼**：围绕 *Llama-3 8B*、*Dolphin 2.9* 和 *Mixtral-8x22B* 等多个大语言模型展开了激烈的辩论，涉及上下文窗口大小以及与对话风格和数据集相关的审查问题。

- **训练“放飞自我”的 AI**：一个有趣的实验涉及使用毒性数据集训练模型，以培养更“放飞自我（unhinged）”的人格。讨论深入探讨了长上下文下的模型局限性，一致认为虽然像 *Llama 3 8B* 这样的模型可以处理长上下文，但超过阈值后性能可能会下降。

- **OpenRouter 上的高性价比实验**：对话集中在 **OpenRouter** 上寻找高效且实惠的模型。值得注意的是，像 *GPT-3.5* 这样能够很好地平衡价格和性能、并提供类人输出的模型，让人们感到惊喜并获得了认可。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**AWS 架构走向学术化**：**LlamaIndex** 展示了一种基于 AWS 的高级架构，用于构建复杂的 RAG 系统，旨在进行解析和推理。详细信息可以在其 [代码仓库](https://t.co/sfQOvhHHg5) 中获取。

**文档机器人（Documentation Bot）在黑客松中获胜**：黑客松冠军 **Team CLAB** 开发了一个令人印象深刻的文档机器人，利用了 **LlamaIndex** 和 **Nomic embeddings**；请在这一篇 [博客文章](https://t.co/2UMqrHwO56) 中查看黑客松总结。

**金融助手获得提升**：构建能够解释非结构化数据并执行复杂计算的金融助手得到了极大改进。该方法在 [最近的一篇文章](https://t.co/6cTNxUBJcr) 中进行了深入探讨。

**通过语义缓存（Semantic Caching）加速 RAG**：与 @Redisinc 的合作展示了通过使用 **语义缓存** 来加速查询，从而显著提升 RAG 应用的性能。合作详情可以在 [这里](https://t.co/oGxFrZLMRn) 找到。

**GPT-1：被铭记的开拓者**：分享了对 GPT-1 及其对 LLM 发展贡献的回顾，讨论了 positional embeddings 等特性，这些特性为 Mistral-7B 等现代模型铺平了道路。这篇充满怀旧色彩的 [博客文章](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms) 重新审视了 GPT-1 的架构和影响。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**参与新的社区项目**：成员们正在寻求机会为提供计算资源的社区 AI 项目做出贡献，以解决那些缺乏个人 GPU 基础设施的人员面临的问题。

**揭开 AI 记忆的奥秘**：讨论了 AI 记忆过程的复杂性，特别关注了压缩记忆（compressive memory）中的 "clear-ing"、orthogonal keys 和 delta rule。尽管 infini-attention 在理论上很有前景，但人们对其是否被过度炒作表现出讨论兴趣。

**苹果与超级计算机的对比**：关于 *mixtral 8x22B* 和 *llama 3 70B* 等模型之间性能差异的辩论非常活跃，其中 *llama* 虽然参数更多，但层数较少，这可能会影响其速度和批处理（batching）效率。

**LLMs：窥探黑箱内部**：社区正在思考大语言模型（Large Language Models）的“黑箱”性质，讨论涌现能力（emergent abilities）和数据泄露。有人将涌现能力与预训练损失（pretraining loss）联系起来，挑战了将算力（compute）作为性能唯一指标的观点。

**位深（Bit Depth）困惑**：一位用户报告了在 **llama3-70b** 和 **llamma3-8b** 等模型上使用 **8bit** 编码时遇到的问题，经历了输出质量的显著下降，这表明存在一个需要解决的跨模型编码挑战。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **GDPR 投诉挑战 AI 生成的生日信息**：一位欧盟隐私倡导者在 AI 模型错误估计其生日后提交了 [GDPR 投诉](https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/)，引发了关于 AI 在欧洲运营潜在影响的讨论。
- **神秘的 GPT-5 推测**：在有关新 GPT-5 模型发布的传言中，社区对不一致的测试结果以及缺乏官方沟通或排行榜认可展开了辩论，质疑该框架在产生幻觉（hallucinations）方面的回避性。
- **Llama3 70B 性能缓慢备受关注**：AI 工程师正在排查 [Llama3 70B](https://rentry.co/GPT2) 模型在双 3090 设备上每秒仅 13 tokens 的异常缓慢生成率，深入研究可能的硬件和配置优化。
- **Exllama 库超越竞争对手**：用户因 **Exllama** 在语言模型任务中的快速表现而推崇它，并建议利用 [TabbyAPI](https://dct.openempathic.ai/) 仓库进行更简单的集成，称其为优于其他库的选择。
- **OpenCLIP 的研究突破**：成功将 **OpenCLIP** 应用于心脏超声分析的研究已发表，强调了严格的修订过程以及向新型非 zero-shot 技术的转变，研究报告见 [此处](https://doi.org/10.1038/s41591-024-02959-y)；同时 *r/StableDiffusion* 已恢复上线，并在 Reddit 最近 API 更改的背景下讨论了一个相关的 CLIP 训练仓库，详见 [此 Reddit 讨论](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/)。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**升级版 ChatGPT Plus 的记忆功能**：ChatGPT Plus 现在允许用户命令 AI 记住特定的上下文，该功能可以在设置中开启或关闭；目前该功能尚未在欧洲或韩国推出。此外，免费版和 Plus 用户都获得了增强的数据控制功能，包括在对话结束后立即丢弃对话的“Temporary Chat”选项。

**AI 的好奇心与相机技巧**：讨论内容从通过迷宫挑战定义 AI 的好奇心和感知力，转向了使用 DragGAN 以新角度修改照片的优点。同时，Llama-3 8B 模型亮相，展示了其长上下文（long-context）能力，可在 [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) 获取，但社区仍在纠结先进 AI 技术的获取门槛以及模型间协作的愿景。

**GPT-4：更大且可能更慢？**：社区深入探讨了 GPT-4 的属性，指出其体积明显大于 3.5 版本，并对规模是否会影响处理速度表示担忧。同时，批量删除存档聊天的可能性也是关注的话题。

**Prompt Engineering 的竞争优势**：Prompt Engineering 引起了关注，有人建议通过竞赛来磨练技能，并利用 GPT Builder 进行“meta prompting”以优化 AI 输出。小组一致认为，正面提示优于列出禁止事项，并致力于优化 AI 文本生成中西班牙语的地区差异。

**跨频道的优质 Prompt 主题**：AI 讨论和 API 频道都探讨了 Prompt Engineering，元提示技术成为焦点，这表明 Prompt 策略正向更高效的方向转变，可能会减少对竞赛的需求。处理多语言输出的复杂性也成为共同的挑战，强调的是适配而非禁止。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**LLaMA 3 在量化方面的挑战**：观察到 **LLaMA 3** 在量化过程中存在显著的性能下降，比其前代产品更严重，这可能是由于其在 15T tokens 上进行的广泛训练捕捉到了非常细微的数据关系。社区内的一篇评论称一项关于量化敏感性的研究“毫无价值”，认为问题可能更多地与模型训练方法而非规模有关；该评论引用了 [arXiv 上的研究](https://arxiv.org/abs/2311.16452)。

**赶上 Zero 列车**：协会讨论了 **Huggingface** 的 **ZeroGPU**，这是一项提供免费访问 Nvidia A100 等多 GPU 资源的测试功能，一些成员对错过早期访问表示遗憾。一位成员[分享了访问权限](https://huggingface.co/zero-gpu-explorers)，并欢迎在平台上进行测试的建议。

**微调技巧**：建议不要直接对 `meta-llama/Meta-Llama-3-70B-Instruct` 进行微调，建议成员从 8B 等较小模型开始以磨练微调技能。协会阐明了如何将微调数据集从 OpenAI 格式转换为 ShareGPT 格式，并提供了用于数据集转换的 Python 代码指导。

**教程传播**：分享了一个关于使用 dstack 微调 Axolotl 的实用[教程](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md)，展示了社区协作改进实践的能力。成员们表达了感谢，并指出该教程易于使用。

**Axolotl 的适配**：在讨论 Axolotl 内部 *command-r* 的微调及相关格式适配时，一位成员分享了一个与此主题相关的[未测试的 pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547)，同时也指出其合并尚不成熟。此外，关于 phi-3 格式的支持以及 *sample packing* 功能的实现状态仍存在不确定性，表明需要进一步的澄清或开发。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Memary：自主 Agent 的长期记忆**：GitHub 上的 [Memary](https://github.com/kingjulio8238/memary) 项目引入了一种实现自主 Agent 长期记忆的新方法，该方法使用文档相似度搜索，而非传统的知识图谱。

- **GPT-2 Chatbot 之谜**：关于 [GPT2-chatbot](https://chat.lmsys.org/) 的激烈辩论正在展开，该机器人展示了令人惊讶的高级能力，引发了人们猜测它可能是 OpenAI GPT-2 的一个微调版本。

- **去中心化训练能否与科技巨头竞争？**：[Prime Intellect 的博客文章](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training) 讨论了去中心化训练作为开源人工智能的一种可行途径，以此与拥有大量 GPU 资源的跨国公司开发的专有模型进行竞争。

- **通过模块化上下文和记忆重新定义 LLM**：讨论中出现了一种范式转移，建议转向设计具有模块化共享上下文和记忆能力的自主 Agent 来进行推理和规划，从而摆脱对独立大型语言模型（LLM）的依赖。

- **为有抱负的 AI 爱好者提供的教育资源**：对于那些寻求学习 AI 基础知识的人，社区成员推荐了一些资源，包括神经网络教程（如 [YouTube](https://youtu.be/aircAruvnKk?feature=shared) 上的视频）和 *Learn Prompting* 等课程，提供了 AI 工程和 Prompt Engineering 基础知识的概览。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**带有视觉功能的 OS 启动**：一位用户在尝试**为 Moondream 启动带有本地视觉模型的 OS 模式**时遇到了挑战，并收到了乱码输出，但讨论尚未产生解决方案或直接建议。

**集成成果**：提到了一项令人兴奋的集成，将 **OpenInterpreter** 的输出集成到 **MagicLLight** 中，并期待未来发布包含 `stream_out` 函数钩子和 `external_input` 的代码及 Pull Request。

**硬件故障帮助**：有人提出了关于在树莓派 Zero 等廉价硬件上运行 **OpenInterpreter** 的疑问，并请求协助**调试启动问题**。社区成员表示在提供更多细节后将帮助进行故障排除。

**按钮编程**：一位个人修复了 **pin 25** 上的外部按钮问题，并分享了 [代码片段](https://discord.com/channels/openinterpreter/01)，同时也得到了社区对该修复方案有效性的确认。

**技术讨论中的音量提升**：关于技术类 YouTuber 是否真正掌握 AI 技术存在不同意见，同时在增加扬声器音量的方案上给出了建议，包括使用 **M5Unified** 或[外部放大器](https://www.amazon.com/dp/B01DKAI51M)。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **深入了解 Tinygrad 的内部运作**：[tinygrad GitHub 仓库](https://github.com/tinygrad/tinygrad/tree/master)被推荐给那些对 **tinygrad** 感兴趣的人，这是一个为 PyTorch 和 micrograd 爱好者准备的教育性项目。另一位社区成员询问了图形可视化问题，随后有人建议使用 `GRAPH=1` 环境变量来生成图表，以解决反向传播（backward pass）问题 [#3572](https://github.com/tinygrad/tinygrad/issues/3572)。

- **学习资源的发现**：社区通过 [MicroGrad](https://github.com/unknownusername504/MicroGrad) 和 [MiniTorch](https://minitorch.github.io/) 等资源探索使用 TinyGrad 学习 AI，其中 MiniTorch 被特别指出对于理解深度学习系统非常有用。"[tinygrad 快速入门指南](https://tinygrad.github.io/tinygrad/quickstart/)" 被强调为初学者的起点。

- **走符号化路线**：在 TinyGrad 中实现符号化均值（symbolic mean）运算引发了关于 LazyBuffer 与数据类型交互，以及在 `sum` 和 `mean` 等操作中变量缓存实用性的讨论。一个 [Pull Request](https://github.com/tinygrad/tinygrad/pull/1552) 展示了符号化代码执行，而进一步的 GitHub 对比视图则处理了带变量的符号化均值开发：[tinygrad symbolic-mean-var-pull](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull) 以及 [gh 的 GitHub 更改](https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840)。

- **寻找均值方案的悬赏任务**：社区正在寻求有关 *"Mean of symbolic shape"* 和 *"Symbolic arrange"* 悬赏挑战的指导。讨论集中在 TinyGrad 环境中这些问题的实现细微差别和实际方法。

- **好奇心汇集**：一个关于成员如何发现该 Discord 服务器的随机问题引发了一连串推测，受访者承认他们不记得是如何遇到的，为频道对话增添了一抹神秘色彩。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R 中的单站点限制**：**API Command R+** 的 `web_search` 工具每次仅允许搜索一个网站，讨论的解决方法涉及**为每个站点进行单独的 API 调用**。
  
- **功能请求热潮**：工程师们渴望 **Command-R** 的改进，重点在于 **Connectors**，包括多网站搜索和额外的参数控制；要熟悉当前功能，请参考 [Cohere Chat 文档](https://docs.cohere.com/reference/chat)。

- **多步 Connector 功能目前受限**：已确认在 **Command-R** 中尚无法通过 **Connectors** 实现**多步工具使用（multi-step tool use）**。

- **Generate 选项消失**：有关从仪表板中消失的用于微调（fine-tuning）模型的“Generate”选项的查询不断增加，其未来的存在状态仍存疑。

- **寻求战略性 Embedding**：讨论围绕保持 Embedding 数据新鲜度的高性价比策略展开，重点是仅对修改的部分进行重新索引（reindexing）。

- **提及北欧网络**：成员们强调了在**瑞典**使用 **Cohere** 的业务，以及通过 **Omegapoint** 公司建立的现有联系，业务横跨瑞典和挪威。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Gemini 经验需求与可观测性工具寻求**：**general** 频道的用户正在寻求 **Gemini 1.0 或 1.5 模型** 的专业知识，并讨论可用的 Large Language Model (LLM) 可观测性工具，重点关注与 **LlamaIndex** 兼容的自托管、开源选项。同时，有人推动在连接 OpenAI 模型时增强 SQL 安全性，并就将 **autoawq** 与 **LangGraph** 集成以使用 **exllamav2 kernels** 进行高速 AI Agent 推理进行了技术讨论。

- **异步冒险与 Google Drive 技巧**：在 **langserve** 频道中，一位用户因 **AzureSearchVectorStoreRetriever** 缺乏异步支持而面临挑战，正在考虑是推动异步功能还是自己编写异步包装器。另外，讨论转向了使用 Google Drive 库的细微差别以及将 drive key 设置为环境变量的重要性。

- **作品展示盛会与插件揭秘**：在 **share-your-work** 中，有一段充满洞察力的回顾，探讨了 **GPT-1** 在启动当前 LLM 进展中的作用，以及几个 LangChain 使用案例，包括 **YouTube** 上的 "D-ID Airbnb Use Case" 和 "Pizza Bot"。**LM Studio 的 VectorDB 插件**也亮相了，旨在增强服务器模式下的 ChromaDB 向量数据库，而 **QuickVid** 则发布了，用于提供 YouTube 视频摘要和事实核查。

- **RAG Agent 走向多语言与私有化**：Tutorials 频道正在为有兴趣使用 **LangChain, Mistral Large** 和 **Llamaindex** 构建 RAG 助手的法语使用者分享资源。另一份指南演示了通过整合个人知识库来增强 **llama3** 的性能，以创建 Agentic RAG，揭示了更本地化和数据丰富的 AI 能力的潜力。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**警报：非法垃圾信息席卷频道**：多个频道的众多消息推广了涉及 "18+ Teen Girls and OnlyFans leaks" 的露骨内容，并附带了 [Discord 邀请链接](https://discord.gg/CYNumE8ABr)。所有消息性质相似，使用表情符号和 `@everyone` 来吸引注意力，公然违反了 Discord 的社区准则。

**需要立即采取审核行动**：重复的帖子表明这是一次协调一致的垃圾邮件攻击，需要立即进行审核干预。每条消息都无一例外地链接到一个外部 Discord 服务器，可能诱导用户进入剥削性环境。

**倡导工程师保持警惕**：鼓励成员举报此类帖子以维持职业礼仪。这些内容违反了法律和道德界限，不符合公会的宗旨或标准。

**Discord 服务器安全面临风险**：这些消息的泛滥凸显了对服务器安全和成员安全的担忧。垃圾邮件表明服务器完整性可能受损，强调了采取强大反垃圾邮件措施的必要性。

**敦促社区无视可疑链接**：敦促工程师和成员避免参与或点击未经请求的链接。这些做法有助于保护个人信息和社区的信誉，同时遵守法律和道德准则。

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **游戏开发者准备迎接游戏化**：Rosebud AI 的 **Game Jam** 邀请创作者使用 **Phaser JS** 制作基于浏览器的 2D 游戏，奖金池为 500 美元。此外，一场 **AIxGames Meetup** 定于周四在旧金山举行，旨在汇聚 AI 和游戏领域的专业人士 [在此预约](https://partiful.com/e/TwvC5qxskuPGqiliMj5f)。

- **LLM 带来的 NPC 革命**：一位开发者推出了由 LLM 驱动的 NPC 模型和推理栈，可在 [GitHub 上的 GigaxGames](https://github.com/GigaxGames/gigax) 获取，该项目承诺提供 LLM 单次调用功能，并在 [Huggingface Hub](https://huggingface.co/Gigax) 上提供开放权重模型，尽管目前 API 访问链接存在故障。

- **应对游戏 NPC 的现实挑战**：开发者正在尝试通过**输出压缩**、减少模型调用和使用更小的模型来提高 NPC 的运行性能，并努力解决 NPC “打破第四面墙”的问题。其中，**Claude 3** 模型在共情交互方面表现出潜力，有助于提升游戏体验。

- **关于 NPC 使用 LLM 的博客预告**：即将发布的一篇博客文章记录了在为动态 NPC 行为微调 LLM 过程中的挣扎与胜利，指出了可能在社区内分享的新策略。

- **在 Windows 上使用 Convex 的困扰**：**Convex local** 设置在 Windows 上运行不佳，导致用户遇到障碍。虽然已经提出了 **WSL** 或 **Docker** 等潜在解决方案，但据报道，兼容 Windows 的 Convex 版本即将推出。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**HaystackDB 中的二进制探索**：社区对 [HaystackDB](https://github.com/carsonpo/haystackdb) 中使用 **2-bit embeddings** 的潜力感到好奇，同时 **Binary Quantized (BQ)** 索引因其在更精简、更快速的相似性搜索方面的潜力而成为关注焦点。

**LLaMA-3 微调的坎坷之路**：工程师们在 **LLaMA-3 微调**过程中遇到了困难，面临从模型忽略 **EOS token generation** 到不同位格式下的嵌入层兼容性等一系列问题。

**对困惑度 (Perplexity) 的困惑**：社区讨论了针对**困惑度微调 LLaMA-3** 的问题，认为其性能可能不会超过基础模型，这可能是由于分词器（tokenizer）相关的复杂性导致的。

**LLaMA-3 改进的曙光**：一位用户通过特定模型的 Prompt 策略成功微调了 **LLaMA-3**，并提交了一个 GitHub [Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553) 供集体审查，这带来了新的希望。

**闲聊杂事不予总结**：**#off-topic** 频道中仅有一个孤立的链接，未对集体知识库贡献任何技术讨论。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 的 AI 人才招募**：Mozilla AI 正在积极招聘多个职位，为有兴趣为其倡议做出贡献的人提供工作机会。有意加入团队的人员可以通过提供的 [链接](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823) 了解更多信息并申请。

- **LM-buddy：语言模型评估工具**：开源评估工具 Lm-buddy 的发布将有助于改进对 LLM 的评估。鼓励贡献者和用户通过给出的 [链接](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378) 参与该项目。

- **Prometheus 在司法角色中对 LLM 进行基准测试**：Prometheus 项目展示了本地大语言模型（LLM）充当仲裁者的潜力，这一新颖概念引发了讨论。感兴趣的各方可以通过 [链接](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599) 加入关于此应用的对话。

- **对 LLaMA 的深入代码分析请求**：一位工程师指出，llama.cpp/llamafile 中的 Token 生成是瓶颈，矩阵-向量乘法消耗了 LLaMA2 推理时间的 95%。这引发了关于循环展开（loop unrolling）是否使 llama.cpp 的性能比其他实现高出 30% 的推测。

- **LLaMA 的混淆与兼容性轶事**：Discord 讨论了关于 LLaMA 参数的一些有趣混淆和匿名误解。此外，还分享了关于集成 Plush-for-comfyUI 的挑战，以及 LLaMA3 在 M1 Macbook Air 上的兼容性问题，并承诺在解决当前的 LLaMA3 问题后将优先测试 M1。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI Maverick 分享的 OLMo 深度解析**：发布了 Hanna Hajishirzi 关于“OLMo: Findings of Training an Open LM”的详细演讲，展示了她在 [Open-Source Generative AI Workshop](https://youtu.be/qFZbu2P1vZ8) 的工作。她在介绍 OLMo, Dolma, Tulu 等实质性内容时语速极快，对学生来说可能难以消化，这反映了她的专业素养以及这些项目背后广泛的研究工作。

- **基于 LM 系统的 RL 揭秘**：John Schulman 关于基于语言模型系统的强化学习（Reinforcement Learning）讨论的核心要点被封装在一个 GitHub [Gist](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) 中，为工程师们提供了他研究方法和发现的压缩综合。

- **AI 排行榜的局限性被指出**：Sayash Kapoor 和 Benedikt Stroebl 发表的一篇 [博客文章](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful) 挑战了 AI 排行榜在代码生成方面的有效性，强调了 LLM debugger (LDB) 尽管排名靠前但运行成本极高，质疑了在面临巨额开支时此类基准测试的实用性。

- **SnailBot**：提到了与 SnailBot 相关的更新或新闻，但缺乏进一步的信息或上下文来进行实质性总结。

- **注意**：根据 Discord 频道提供的片段，没有其他值得总结的内容，这表明这些消息可能是更大背景或后续讨论的一部分，但未被包含在内。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Gamma 寻找 AI 奇才**：**Gamma** 正在招聘一名 **AI engineer**，以推动 AI 驱动的演示文稿和网站设计的创新，重点关注 prompt engineering、指标和模型 fine-tuning；详情见 [Gamma Careers](https://careers.gamma.app/ai-engineer)。尽管需要在 **旧金山** 实地办公，但该职位对具有强大 Large Language Model (LLM) 技能的人开放，即使他们缺乏丰富的工程经验。

- **处于增长快车道的 AI 驱动型企业**：**Gamma** 拥有超过 **1000 万用户**和 **1000 万美元以上融资**，目前正在寻找一名 AI engineer 来帮助维持其增长，同时在其**盈利**且紧凑的 16 人团队中享受混合办公文化。

- **GPT-4.5 的推测案例**：**@phill__1** 的一条推文暗示 gpt2-chatbot 拥有“疯狂的领域知识”，引发了人们对其可能代表 **GPT-4.5** 版本能力的猜测 [phill__1 的观察](https://x.com/phill__1/status/1784964135920235000)。

- **Chatbot 引起社区轰动**：工程师社区对 gpt2-chatbot 可能是无意中窥见的 **GPT-4.5** 实力这一想法议论纷纷，一位成员简洁地评价其“很好”。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **代码生成的语法消除方案**：一位用户讨论了在语言模型中加入**自定义语法（custom grammar）**的概念，以便在代码生成过程中优先识别语义错误而非语法错误。

- **Datasette 的数据化下拉菜单**：交流了关于改进 **Datasette UX** 的建议，包括一个带有下拉菜单的首页设计，使用户能够根据所选参数（如国家选择）生成汇总表。

- **直接数据交付的 UX 魔法**：成员们提出了增强 **Datasette** UX 的解决方案，包括动态更新 URL 或构建根据用户选择调整的主页查询，以简化对相关数据的访问。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **加载异常之谜**：一段对话强调了一个进程**在本地机器上加载只需 3 秒**，但在通过作业提交运行时面临延迟，这暗示问题可能与存储无关，而可能是特定环境的开销。
- **Llama 在语言基准测试中击败 GPT-4**：如 [ScandEval 排行榜](https://scandeval.com/german-nlg/) 所示，**Llama 3** 在 **ScanEval 德语 NLG 基准测试**中表现优于 **GPT-4**。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细总结和链接

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1234899266837938176)** (1 条消息): 

- **澄清 Triton Block 大小限制**：一位成员询问了 **Triton block** 的最大尺寸，指出虽然他们可以创建具有 4096 个元素的 block，但无法对 8192 个元素执行相同操作，这表明与预期的 CUDA 限制存在差异。

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1234454843696087122)** (8 messages🔥): 

- **寻找 Flash Attention 代码**：一位用户询问如何下载 Thomas Viehmann 演示的 Flash Attention 代码的 lecture12；聊天中未提供该查询的解决方案。
- **理解 CUDA Reductions**：一位成员解决了关于 CUDA 中行向（row-wise）与列向（column-wise）Reductions 的困惑，意识到性能差异是由（非）合并内存访问（coalesced memory accesses）引起的，并澄清了自己的问题。
- **Kernel 代码中的整数除法**：进行了一场关于用位移替换整数除法的优化讨论；建议指出，当除数为 2 的幂时，nvcc 或 ptxas 可能会优化除法，并提供了一个 [compiler explorer 链接](https://godbolt.org/z/9K9Gf1v6P) 以供进一步实验。
- **CUDA Checkpointing 资源分享**：分享了一个用于 CUDA 检查点和恢复工具的外部 GitHub 资源 [NVIDIA/cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)，未进行进一步讨论。
- **比较 CUTLASS 和 CuBLAS 性能**：一位成员对 CuBLAS 和 CUTLASS 的矩阵乘法性能进行了基准测试，报告称 CUTLASS 在独立分析器中优于 CuBLAS，但集成到 Python 中后性能提升消失了，详情分享在 [Thonking AI 关于矩阵乘法的文章](https://www.thonking.ai/p/strangely-matrix-multiplications) 中。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>：智者讨论每瓦特浮点运算次数（flops per watt）。</li><li><a href="https://github.com/NVIDIA/cuda-checkpoint">GitHub - NVIDIA/cuda-checkpoint: CUDA checkpoint and restore utility</a>：CUDA 检查点和恢复工具。通过在 GitHub 上创建账号为 NVIDIA/cuda-checkpoint 的开发做出贡献。</li><li><a href="https://godbolt.org/z/9K9Gf1v6P">Compiler Explorer - CUDA C++ (NVCC 11.7.0)</a>：#include &amp;lt;algorithm&amp;gt; #include &amp;lt;cassert&amp;gt; #include &amp;lt;cstdio&amp;gt; #include &amp;lt;cstdlib&amp;gt;  __global__ void sgemmVectorize(int M, int N, int K, float alpha, f...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1234490936143249428)** (4 messages): 

- **关于双 Kernel 启动的好奇**：一位成员询问为什么在 PyTorch 的矩阵乘法期间，分析器有时会显示两次 Kernel 启动。
- **关于 PyTorch `linear` 函数的澄清**：另一位成员澄清说，PyTorch 中的 `linear` 默认确实对输入包含转置操作，这可能不会导致性能差异。
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1234626145421365259)** (2 messages): 

- **为 LLM 引入 Effort Engine**：分享了 Effort Engine 算法，该算法能够在 LLM 推理过程中动态调整计算量。根据 [kolinko.github.io/effort](https://kolinko.github.io/effort) 上的详细信息，在 **50% effort** 时，它的速度可与 Apple Silicon 上的标准矩阵乘法媲美；在 **25% effort** 时，速度提高了一倍，且质量损失极小。

- **Effort Engine 的模型推理方法**：这种新技术允许选择性地加载重要权重，在不显著降低质量的情况下潜在地提高速度。它已针对 **Mistral** 实现，经过一些转换和预计算后应与其他模型兼容，实现代码可在 [GitHub](https://github.com/kolinko/effort) 上获得。

- **仅限 FP16 的实现及改进空间**：Effort Engine 目前仅适用于 **FP16 实现**，虽然乘法速度很快，但在 Softmax 和 Attention 累加操作等其他领域仍需改进。

- **探讨 Effort Engine 的潜在局限性**：一位成员强调，虽然 Effort Engine 的方法具有创新性，但它可能与激活稀疏性（activation sparsity）方法存在共同的局限性，特别是在 Batch Size 大于 1 的批处理计算中，由于激活强度不一致（misaligned activation magnitudes）导致的问题。

**提到的链接**：<a href="https://kolinko.github.io/effort/">Effort Engine</a>：一种可能用于 LLM 推理的新算法。在推理过程中平滑且实时地调整你想要进行的计算量。

  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1234455593343783014)** (1 messages):

- **InstaDeep 正在招聘 ML 工程师**：InstaDeep Research 正在积极寻找对 **高性能 ML 工程** 及其现实应用充满热情的 **Machine Learning Engineers**。在构建自定义 CUDA kernels、最先进的模型架构、量化（quantisation）和分布式训练（distributed training）方面表现出色的候选人可以 [联系获取机会](https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/)。

- **加入协作创新者**：InstaDeep 提供了一个充满刺激、协作的环境，致力于现实生活中的决策和技术产品，并鼓励渴望产生变革性影响的优秀人才申请。公司强调在 **Bio AI** 和 **Decision Making AI** 领域的创新和实际应用。

- **寻求实习生和多职位申请者**：对实习或在 InstaDeep 寻求多个工作机会感兴趣的人员可以 [探索实习机会](https://www.instadeep.com/internships) 并申请多个职位（前提是具备相关技能），但建议申请不要超过两个，以避免申请被拒绝。

- **建议的重新申请指南**：建议之前申请过但未被录用的人员在重新申请前等待一段时间，特别是如果他们在过去六个月内申请过，这表明需要一段时间来考虑申请人概况或公司需求的变化。

**提到的链接**：<a href="https://www.instadeep.com/job-offer/92900fa3-5501-4506-a63f-cebee958fc6f/">Job Offer | InstaDeep - 企业级决策 AI</a>：未找到描述

  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1234509189091426334)** (2 条消息): 

- **无进度更新**：一名成员确认目前 **没有新的进展** 需要报告。
- **视频中的性能分析技术**：聊天中分享了一个名为 [“Lecture 16: On Hands Profiling”的 YouTube 视频](https://youtu.be/SKV6kDk1s94)，提供了学习性能分析（profiling）技术的资源，尽管没有提供具体描述。

**提到的链接**：<a href="https://youtu.be/SKV6kDk1s94">Lecture 16: On Hands Profiling</a>：未找到描述

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1234630522106282065)** (1 条消息): 

- **Llama-3 刷新上下文长度纪录**：Gradient 发布了 [Llama-3 8B Gradient Instruct 1048k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)，将上下文长度从 8k 扩展到超过 1048k。这一成就表明，最先进的语言模型只需极少的训练调整即可适应长上下文。

**提到的链接**：<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1234635788642287696)** (1 条消息): 

- **CUTLASS：整数的舞蹈**：一位成员观察到，[CUTLASS](https://developer.nvidia.com/cutlass) 尽管是一个线性代数库，但在调用高级线性代数例程之前，主要处理整数操作和索引操作。这一特性合理化了其作为 **header-only library**（仅头文件库）的性质，无需复杂的链接。
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1234443683856650250)** (721 条消息🔥🔥🔥): 

- **CUDA 编程讨论与 Packed128 类型**：关于使用 `Packed128` 自定义结构体来优化内存访问模式（包括 **读取和写入**）进行了详细辩论。特别关注了 `Packed128` 的正确构造和利用，以及是否在 kernel 内部对 **floatX** 和 **BF16** 使用显式类型转换。 

- **混合精度策略担忧**：人们担心在整个模型中使用 BF16 的影响，以及 **随机舍入（stochastic rounding）** 是否会影响训练收敛。计划比较 **llm.c** 的 BF16 方法与标准 PyTorch 混合精度实现之间的 loss 指标。

- **性能分析与调试**：一名成员添加了 **NVTX** 上下文，以便使用 NSight Compute 进行更好的性能分析，从而实现更准确的 GPU 计时。一名成员观察到 **AdamW** kernel 可能需要在 FP32 原子操作（atomics）和 scratch 存储使用方面进行优化。

- **基准测试的工具与基础设施**：成员们讨论了像 Modal 这样在标准化规格上运行基准测试（benchmarks）的外部平台的潜在效用，特别是 **Modal** 在 **nvprof** 和 **nsys** 等性能分析工具方面的优势和局限性。

- **PR 评审已准备好合并及 CI 建议**：频道中有几个 PR 已准备好合并，主要涉及针对各种 kernel 的 f128 和 Packed128 优化。此外，还强调了保持分支**文档更新**、**-Wall 编译**以及通过 **CI 检查**确保 Python 和 C 实现结果一致的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/">Nvidia&#8217;s H100：有趣的 L2 和海量带宽</a>：GPU 最初是纯粹用于图形渲染的设备，但其高度并行的特性使其对某些计算任务也具有吸引力。随着过去几年 GPU 计算场景的增长……</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/associate_access_property.html">cuda::associate_access_property</a>：CUDA C++ 核心库</li><li><a href="https://arxiv.org/abs/2310.18313">FP8-LM：训练 FP8 大语言模型 (LLMs)</a>：在本文中，我们探索了用于高效训练大语言模型 (LLMs) 的 FP8 低比特数据格式。我们的核心见解是，LLM 训练中的大多数变量，如梯度和优化器状态……</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html">cuda::memcpy_async</a>：CUDA C++ 核心库</li><li><a href="https://www.thonking.ai/p/strangely-matrix-multiplications">奇怪的是，当给定“可预测”数据时，GPU 上的矩阵乘法运行得更快！[简短版]</a>：伟大的思想讨论 flops per watt。</li><li><a href="https://developer.nvidia.com/nccl/nccl2-download-survey">登录</a>：未找到描述</li><li><a href="https://godbolt.org/z/hME5EqYrr">Compiler Explorer - CUDA C++ (NVCC 12.2.1)</a>：#include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt;  t...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/layernorm_backward.cu">llm.c/dev/cuda/layernorm_backward.cu (master 分支) · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L553">llm.c/train_gpt2.cu (master 分支) · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/issues/246">WikiText 103 评估 · Issue #246 · karpathy/llm.c</a>：我看到一些仓库使用 WikiText-103 作为评估类 GPT 模型的数据集，例如：https://github.com/tysam-code/hlb-gpt/tree/main。添加预处理脚本以下载、预处理和分词……</li><li><a href="https://github.com/karpathy/llm.c/blob/9464f4272ef646ab9ce0667264f8816a5b4875f1/train_gpt2.cu#L734">llm.c/train_gpt2.cu (提交号 9464f42) · karpathy/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://godbolt.org/z/1hs47YzvY">Compiler Explorer - CUDA C++ (NVCC 12.3.1)</a>：#include &amp;lt;cuda_fp16.h&amp;gt;   template&amp;lt;class ElementType&amp;gt; struct alignas(16) Packed128 {     __device__ __forceinline__ Packed128() = default;     __device__ __forceinline__ exp...</li><li><a href="https://github.com/karpathy/llm.c/pull/311">由 leloykun 提交的在 Modal 上运行基准测试的脚本 · Pull Request #311 · karpathy/llm.c</a>：此 PR 添加了一个在 Modal 平台上运行基准测试的脚本。这对于本地无法使用昂贵 GPU 的开发者很有用。要在 attention 前向传播上运行基准测试……</li><li><a href="https://github.com/graphcore-research/out-of-the-box-fp8-training/tree/main">GitHub - graphcore-research/out-of-the-box-fp8-training：unit_scaling 库的演示，展示了如何轻松地调整模型以进行 FP8 训练。</a>：unit_scaling 库的演示，展示了如何轻松地调整模型以进行 FP8 训练。 - graphcore-research/out-of-the-box-fp8-training</li><li><a href="https://github.com/NVIDIA/cudnn-frontend">GitHub - NVIDIA/cudnn-frontend：cudnn_frontend 为 cudnn 后端 API 提供了一个 C++ 封装以及如何使用它的示例</a>：cudnn_frontend 为 cudnn 后端 API 提供了一个 C++ 封装以及如何使用它的示例 - NVIDIA/cudnn-frontend</li><li><a href="https://github.com/karpathy/llm.c/commit/3fb7252924e342739ba47b5144a785470e839081">第一轮变更。现在即使 dtype 设置为 float16 或 bfloat16，我们也始终以 fp32 写入…… · karpathy/llm.c@3fb7252</a>：……接下来，当设置了 dtype 时，我们实际上希望以较低精度写入。</li><li><a href="https://github.com/karpathy/llm.c/pull/313/files">由 ngc92 修复了潜在错误并泛化了 gelu 前向传播 · Pull Request #313 · karpathy/llm.c</a>：这增加了一个用于将 size_t 安全转换为 int 的辅助函数（可能也想在 utils.h 中包含它）。该宏随后用于将 size_t 类型的 block_size * x128::size 转换回常规……</li><li><a href="https://github.com/karpathy/llm.c/pull/298">由 karpathy 提交的 Feature/packed128 · Pull Request #298 · karpathy/llm.c</a>：未找到描述</li>

<li><a href="https://github.com/karpathy/llm.c/pull/303">由 ChrisDryden 提交的 Pull Request #303 · karpathy/llm.c：更新 adamw 以使用 packed 数据类型</a>：运行前总平均迭代时间：38.547570 ms；运行后总平均迭代时间：37.901735 ms。Kernel 开发文件规范：在当前的测试套件中几乎察觉不到：Bef...</li><li><a href="https://github.com/karpathy/llm.c/pull/273">由 PeterZhizhin 提交的 Pull Request #273 · karpathy/llm.c：添加 NSight Compute 范围，使用 CUDA events 进行计时</a>：CUDA events 允许更精确的计时（由 GPU 测量）。nvtxRangePush/nvtxRangePop 为 NSight Systems 添加了简单的堆栈跟踪：示例运行命令：nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://github.com/karpathy/llm.c/pull/293">由 ngc92 提交的 Pull Request #293 · karpathy/llm.c：另一个 gelu 实现</a>：更复杂的 Packet128 以实现更整洁的 kernels</li><li><a href="https://github.com/karpathy/llm.c/pull/272">由 ademeure 提交的 Pull Request #272 · karpathy/llm.c：默认全 BF16 包括 layernorms（最小化 BF16 atomics 数量）</a>：我添加了 4 个不同新版本的 layernorm_backward_kernel，性能最好的是：Kernel 4（使用 atomicCAS，无 scratch，但多次舍入，因此数值精度可能较差）；Kernel 6...</li><li><a href="https://github.com/karpathy/llm.c/pull/275#issuecomment-2083693720">由 ChrisDryden 提交的 Pull Request #275 · karpathy/llm.c：移除 Atomic Adds 并添加 memory coalescion</a>：此 PR 基于 GELU memory coalescion PR，本质上是重写了 backwards encoder，使用 shared memory 代替 atomic adds，并使用 Packed 结构体进行 coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/275#issuecomment-2083658642">由 ChrisDryden 提交的 Pull Request #275 · karpathy/llm.c：移除 Atomic Adds 并添加 memory coalescion</a>：此 PR 基于 GELU memory coalescion PR，本质上是重写了 backwards encoder，使用 shared memory 代替 atomic adds，并使用 Packed 结构体进行 coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/275">由 ChrisDryden 提交的 Pull Request #275 · karpathy/llm.c：移除 Atomic Adds 并添加 memory coalescion</a>：此 PR 基于 GELU memory coalescion PR，本质上是重写了 backwards encoder，使用 shared memory 代替 atomic adds，并使用 Packed 结构体进行 coale...</li><li><a href="https://github.com/karpathy/llm.c/pull/306">由 JaneIllario 提交的 Pull Request #306 · karpathy/llm.c：为 Gelu backwards 进行 Packing</a>：更新 gelu backwards kernel 以进行 128 位 packing，并创建 gelu backward cuda 文件。之前的 kernel：block_size 32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size 64 | time 0.0760...</li><li><a href="https://github.com/karpath">karpath - 概览</a>：GitHub 是 karpath 构建软件的地方。</li><li><a href="https://github.com/karpathy/llm.c/pull/295">由 ademeure 提交的 Pull Request #295 · karpathy/llm.c：移除 FloatN 并通过 BF16 LayerNorms 简化 adam/reduce</a>：MULTI_GPU 路径未经测试，但其他部分似乎运行良好。我保留了每个 tensor 的 "param_sizeof"，因为它在 test_gpt2.cu 等文件中被使用，代码量不多且可能有用...</li><li><a href="https://github.com/karpathy/llm.c/pull/60">由 leloykun 提交的 Pull Request #60 · karpathy/llm.c：通过实现 Flash Attention 2 kernel 加速 `attention_forward_kernel2`</a>：通过将实现替换为极简的 Flash Attention 2 kernel 来加速 attention_forward_kernel2 kernel，详见 https://github.com/leloykun/flash-hyperbolic-attention...</li><li><a href="https://github.com/leloykun/flash-hyperbolic-attention-minimal/blob/main/flash_attention_2.cu">leloykun/flash-hyperbolic-attention-minimal 项目 main 分支下的 flash_attention_2.cu</a>：约 [...] 行 CUDA 代码实现的 Flash Hyperbolic Attention - leloykun/flash-hyperbolic-attention-minimal</li><li><a href="https://github.com/karpathy/llm.c/pull/285">由 kilianhae 提交的 Pull Request #285 · karpathy/llm.c：Flashattention</a>：更快的 Flash Attention 实现。在 src/attention_forward 中添加了 attention_forward6：一个不带任何依赖项编写的快速 flash attention 前向传递。我们假设...</li><li><a href="https://github.com/karpathy/llm.c/blob/9464f4272ef646ab9ce0667264f8816a5b4875f1/train_gpt2.cu#L1233">llm.c/train_gpt2.cu (版本 9464f42) · karpathy/llm.c</a>：使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2022">llm.c/train_gpt2.cu (master 分支) · karpathy/llm.c</a>：使用简单的原生 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2024">llm.c/train_gpt2.cu at</a>

master · karpathy/llm.c</a>: 使用简单、纯粹的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/301">Added packing for gelu forwards kernel by ChrisDryden · Pull Request #301 · karpathy/llm.c</a>: 此 PR 使用提供的示例实现了 Gelu 前向核函数的 packing。核函数开发文件也进行了更新，以展示更改 floatX 数据类型的影响。更改前：...</li><li><a href="https://github.com/karpathy/llm.c/pull/299">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: 更新 residual_forward 以使用 128 位 packed 输入，配合 floatX。之前的核函数：block_size 32 | 时间 0.1498 ms | 带宽 503.99 GB/s block_size 64 | 时间 0.0760 ms | 带宽 993.32 GB/s b...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1234617660747157535)** (8 messages🔥): 

- **关于 ROCm 6.x 的 Flash Attention 2 的咨询**：一位成员询问是否有人在为 **ROCm 6.x** 构建 Flash Attention 2，并指出他们已成功为 ROCm 5.6 和 Torch 2.2 构建，但对更新的技术栈感兴趣。
- **Torch Nightly 的构建困扰**：成员们讨论了为当前版本（如 Torch 2.3）构建的困难，其中一人表示希望使用 **Torch nightly** 但遇到了问题。
- **官方 Fork 版本滞后**：提到 AMD 硬件的 Flash Attention 官方 Fork 版本已经过时，仍停留在 Flash Attention 2.0 版本，且未移植最近的开发成果。
- **Backward Pass 更新确认**：当被问及 AMD Flash Attention 是否增加了 backward pass 时，一位成员确认确实已经添加。
- **Flash Attention GitHub 仓库**：分享了 [ROCm/flash-attention 在 GitHub 上的仓库](https://github.com/ROCm/flash-attention) 链接，该仓库是快速且内存高效的精确 Attention 的资源。

**提及的链接**：<a href="https://github.com/ROCm/flash-attention">GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention</a>: 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号来为 ROCm/flash-attention 的开发做出贡献。

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1234428342305030204)** (487 messages🔥🔥🔥): 

- **WSL2 上 llama3 的转换问题**：一位用户报告了在 WSL2 中将模型转换为 F16 时的错误，提示 `RuntimeError: Unsloth: Quantization failed`。即使尝试重新构建 `llama.cpp` 并重新进行 Quantization，问题依然存在。
- **模型 Checkpoint 合并查询**：一位成员询问如何合并特定的 Checkpoint 以避免最新 epoch 导致的过拟合。另一位成员提供了指向 Unsloth [wiki 关于 Checkpointing 更多信息](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint) 的链接，进一步的讨论建议了诸如 *warmup steps* 和训练函数中的 *resuming from a checkpoint* 选项等方法。
- **对 Phi-3 的期待**：成员们讨论了 Phi-3 可能的发布，期待尝试 3.8b 版本。对话涵盖了从发布时间线的推测到是否等待 7b 或 14b 等更大版本的考虑。
- **训练技巧与故障排除**：多位用户讨论了他们在训练 *Gemma*、*LLaMA-3* 和 *Mistral* 等模型时的经验和策略。技巧包括保存 Checkpoint 的重要性以及调整 *max steps* 和 *batch sizes* 等训练参数。
- **Unsloth 工具更新**：重点强调了使用新版本更新 Unsloth 安装，讨论了仓库中的更新，并对平台正在开发的 multi-GPU 支持进行了推测。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/dudeman6790/status/1785060925206097976">来自 RomboDawg (@dudeman6790) 的推文</a>：目前正在使用 OpenCodeInterpreter 数据集中完整的 230,000+ 行代码数据训练 Llama-3-8b-instruct。我想知道我们能在 humaneval 上把那个 .622 提高多少 🤔🤔 大家为我的 jun 祈祷吧...</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharin">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIk">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1NvkBmkHfucGO3Ve9s1NKZvMNlw5p83ym?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1cIlNmJS-mvO60iRqxYVFUfD0D9g_B7x0?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/dudeman6790/status/1784414430781931961">来自 RomboDawg (@dudeman6790) 的推文</a>：如果你不想手动复制代码，这里有一个完整的 colab 笔记本。再次感谢 @Teknium1 的建议 https://colab.research.google.com/drive/1bX4BsjLcdNJnoAf7lGXmWOgaY8yekg8p?usp=shar...</li><li><a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1">DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/here-we-go-joker-heath-ledger-the-dark-knight-and-here-we-go-gif-17775369">Here We Go Joker GIF - Here We Go Joker Heath Ledger - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/weird-minion-gif-23757545">Weird Minion GIF - Weird Minion - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/wheel-of-fortune-wheel-wof-game-show-celebrity-wheel-of-fortune-gif-23489251">Wheel Of Fortune Wheel GIF - Wheel Of Fortune Wheel Wof - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading">Load</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k">mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-1048k-GGUF/tree/main">crusoeai/Llama-3-8B-Instruct-Gradient-1048k at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-fro">主页</a>：微调 Llama 3, Mistral & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/botbot-ai/CabraLlama3-8b/tree/main?show_tensors=model.safetensors.index.json">botbot-ai/CabraLlama3-8b at main</a>：未找到描述</li><li><a href="https://huggingface.co/arthrod/cicerocabra/tree/main?show_tensors=model.safetensors.index.json">arthrod/cicerocabra at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/400">[已修复] NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs · Issue #400 · unslothai/unsloth</a>：我是尝试使用 unsloth 的初学者。我运行了免费的 Llama 3 (8B) 笔记本，然后遇到了以下错误：在第一个安装步骤中我也遇到了以下错误：ERROR: pip's dep...</li><li><a href="https://github.com/M-Chimiste/unsloth_finetuning">GitHub - M-Chimiste/unsloth_finetuning</a>：通过在 GitHub 上创建账号来为 M-Chimiste/unsloth_finetuning 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">主页</a>：微调 Llama 3, Mistral & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral & Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral & Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/pull/30079">winglian 提交的 schedulefree 优化器 · Pull Request #30079 · huggingface/transformers</a>：此 PR 的作用是什么？集成了 Meta 的 https://github.com/facebookresearch/schedule_free 用于 adamw & sgd https://twitter.com/aaron_defazio/status/1776320004465582331 在提交之前，这 ...</li><li><a href="https://download.pytorch.org/whl/cu121">未找到标题</a>：未找到描述</li><li><a href="https://github.com/huggingface/datasets/issues/6753">导入 d 时出现类型错误</a>

atasets on Kaggle · Issue #6753 · huggingface/datasets</a>: 描述 Bug。当尝试运行 `import datasets; print(datasets.__version__)` 时，它生成以下错误 `TypeError: expected string or bytes-like object`。看起来它找不到 val...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: 可定制且优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: 继续 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 添加了对 BPE 预分词的支持。总结：到目前为止的状态是，对于所有基于 BPE 的模型，llama.cpp 应用了一个默认的预...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1234459978820227147)** (48 messages🔥): 

- **处理 Colab 中的 Out of Memory**: 一位成员分享了在 Google Colab 中应对 **Out of Memory (OOM)** 错误的技巧，通过运行一段使用 `torch` 和 `gc` 模块清理缓存并进行垃圾回收的 Python 代码片段。*其他成员对这个技巧表示赞赏，并计划在未来采用*。

- **对 Llama 模型性能数据的困惑**: 讨论了量化 Llama 模型（特别是 **Llama 2** 和 **Llama 3**）时的困惑度（perplexity）差异。似乎在实际数据方面存在沟通误解，成员们指出 Bits Per Word (BPW) 和 Perplexity (PPL) 列可能存在交换或错误。

- **Phi-3 现已支持**: 分享了关于 **Phi-3** 已被支持的更新，成员们对在项目中使用它表示兴奋。本应分享一个 *Colab notebook* 的链接，但显然没有提供。

- **Phi-3 集成问题**: 成员们讨论了在 Unsloth notebook 中尝试使用 **Phi-3** 模型时遇到的问题，弹出的错误消息提示需要自定义脚本。*讨论集中在排除故障并确保使用正确的 notebook*。

- **Llama 3 许可证问题**: 一位成员提出了关于 **Llama 3 许可证条件** 的问题，想知道根据许可证，所有衍生模型是否都应带有特定前缀并显示致谢。还有人对 Huggingface 模型可能违反许可证的情况表示担忧。

**提到的链接**: <a href="https://en.wikipedia.org/wiki/Out_of_memory">Out of memory - Wikipedia</a>: 未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1234461140344508418)** (230 messages🔥🔥): 

- **微调期间 Loss 的澄清**: 一位成员询问使用 Unsloth 进行微调期间显示的 loss 是测试 loss 还是训练 loss。给出的建议是向 trainer 传递一个验证数据集，具体是使用带有 `train_dataset` 和 `eval_dataset` 的 `SFTTrainer` 进行验证。

- **SFTTrainer 中不支持早停**: 有人指出 `SFTTrainer` 不支持基于验证 loss 的早停（Early Stopping）。用户被告知一个名为 'trainer' 的更高级类可能会提供此功能。

- **UnslothAI 在 GGUF 转换和 Xformers 方面的问题**: 多位用户报告了 GGUF 转换的问题，特别是 Phi-3 模型，出现了词表大小（vocab size）版本不匹配的情况。此外，最近对 xformers 的更新破坏了兼容性，现在需要 PyTorch 2.3；一位成员通过将版本固定为 `xformers<0.0.26` 提供了临时解决方案。

- **Llama 3 训练模型“胡言乱语”**: 一位成员表示担心，他们的微调 Llama-3 模型在使用 Ollama 进行推理时会不停地说话，怀疑是 `EOS_TOKEN` 的问题。另一位用户建议，问题可能是 Ollama 没有识别出训练期间设置的正确 `EOS_TOKEN`。

- **在 Unsloth 中使用多个 GPU 会产生警告**: 一位用户询问如何在 Unsloth 中使用多个 GPU，并分享了一个关于检测到多个 CUDA 设备但仅允许单个设备的错误。相关消息显示系统将 `CUDA_VISIBLE_DEVICES` 覆盖为第一个设备。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#local-and-remote-files">Load</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/package_reference/peft_model#peft.get_peft_model.peft_config">Models</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 使用 Unsloth 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral &amp; Gemma LLM - unslothai/unsloth</li><li><a href="https://github.com/ollama/ollama/issues/3759">llama3-instruct models not stopping at stop token · Issue #3759 · ollama/ollama</a>: 问题是什么？我正在通过 OpenAI 兼容端点使用 llama3:70b。生成时，我得到了如下输出：请提供上述命令的输出。让我们继续...</li><li><a href="https://github.com/vllm-project/vllm/issues/4180">[Usage]: Llama 3 8B Instruct Inference · Issue #4180 · vllm-project/vllm</a>: 您当前的环境：在 2 个 L4 GPU 上使用最新版本的 vLLM。您想如何使用 vLLM：我正尝试利用 vLLM 部署 meta-llama/Meta-Llama-3-8B-Instruct 模型并使用 OpenAI...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1234474052270428191)** (7 messages): 

- **Llama 3 8B 的海量上下文扩展**：正如 [Hugging Face](https://huggingface.co/winglian/llama-3-8b-256k-PoSE) 上展示的那样，通过使用 **[PoSE](https://huggingface.co/papers/2309.10400)**，**Llama 3 8B** 的上下文长度已从 8k 显著扩展到 256k。虽然由于推理挑战尚未在 "needle in haystack" 场景中进行测试，但该模型已通过 75M tokens 的持续预训练数据进行了增强。
- **社区赞赏 Winglian**：聊天成员赞扬了 Winglian 对社区的贡献，特别是与 **Llama 3 8B 256K** 开发相关的贡献。
- **从 128k 到 256k**：一位成员对从 128k 上下文进展到 **256k 上下文模型**表示惊讶。
- **开源的力量**：由于在上下文扩展模型中观察到一些奇怪的行为，有人对非官方版本表示怀疑，但仍然强调了 **开源** 贡献的潜力。

**提到的链接**：<a href="https://huggingface.co/winglian/llama-3-8b-256k-PoSE">winglian/llama-3-8b-256k-PoSE · Hugging Face</a>: 未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1234453305980096563)** (25 messages🔥): 

- **Unsloth 与 Recurrent Gemma 2b 集成咨询**：一位社区成员表示有兴趣将 **Recurrent Gemma** 与 **Unsloth** 集成以提高性能。然而，Unsloth 团队承认 Gemma 2b 的基础模型存在一个已知的 bug，目前的工作重点是 **Phi 3**，这意味着集成可能不会立即进行。

- **Gemma 2b VRAM 消耗问题**：据报告，Gemma 2b 有时会超过 VRAM 限制，但尚不清楚这是一个普遍问题还是个别事件。Unsloth 团队已知晓并表示他们需要解决这个问题。

- **尽管有 VRAM 开销，Gemma 2b 仍可运行**：虽然存在 VRAM 消耗问题，但 Gemma 2b 模型仍然可以运行。目前只有一位用户报告了此问题，这表明它可能不是一个普遍问题。

- **提供了 Gemma 2b VRAM 问题的参考**：Unsloth 团队引导用户查看 Discord 消息链接以获取 VRAM 问题的参考，尽管提供的文本消息中未正确包含该链接。
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1234439098459230241)** (135 messages🔥🔥): 

- **关于在 Ubuntu GPU 上运行 LM Studio 的咨询**：成员们寻求在 Ubuntu GPU 上运行 LM Studio 的建议，并建议在特定频道发布详细的系统规格。还提到了某些 GPU 与推理任务的兼容性问题。
  
- **针对 Llama3 的 Groq API**：一位成员分享了一个 [YouTube 链接](https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB)，内容是关于 Groq 提供的免费 API，该 API 可访问 LLAMA-3 模型，据报道其速度可达每秒 300 tokens，并因其速度和成本（免费）而被称赞非常适合小型服务器 Discord 机器人。

- **LM Studio 本地训练咨询**：LLM 新手询问了基于现有 Hugging Face 模型训练本地模型的问题，讨论表明这是硬件密集型且耗时的。一位成员声称在一个微型数据集上微调 phi-3 4k 模型花费了近 8 小时。

- **GPU Offload 困惑**：提到了关于利用 GPU 提升 LM Studio 性能的咨询，一位成员表示他们的 Intel Titan A770 在 LM Studio 的 GPU offloading 中没有作用，其他人则讨论了禁用 “GPU Offload” 以解决错误的有效性。

- **在 LM Studio 中将 KV Cache 保存到磁盘**：成员们对 LM Studio 是否允许将 Key-Value (KV) caches 保存到磁盘并在以后重用感兴趣，类似于 llama.cpp 中的功能，以避免为查询重新处理大型数据输入，目前尚未提供明确的解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/mods-discord-mod-moderator-moderation-clash-of-clans-gif-24080525">Mods Discord Mod GIF - Mods Discord Mod Moderator - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/ySwJT3Z1MFI?si=qFfek8gTGXVJWoxB">在 Groq Playground 和 API 上免费体验极速 LLAMA-3</a>：了解如何在 Groq API 上开始使用 LLAMA-3，这是目前市场上任何 API 中最快的推理速度。了解如何使用 Gro...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : 由 ggerganov 添加 Flash Attention · Pull Request #5021 · ggerganov/llama.cpp</a>：ref #3365 为 ggml 和 llama.cpp 中的 Flash Attention 支持设置所需内容。提议的算子执行：// new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale); // fused scale ...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : 改进 BPE 预处理 + LLaMA 3 和 Deepseek 支持 · Pull Request #6920 · ggerganov/llama.cpp</a>：继续 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 添加了对 BPE 预分词的支持。摘要：到目前为止的状态是，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预处理...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1234440283932856351)** (149 条消息🔥🔥): 

- **寻找替代模型下载**：由于 Huggingface 的问题，用户讨论了下载 GGUF 模型的替代来源。一个建议的解决方法涉及制作 `imatrices`，这需要*非常长的时间*且是*计算密集型*的。

- **iQuants 和 iMatrices 的复杂性**：讨论了为模型创建 iQuants 的过程。大家认识到 iQuant 的创建可能很费力，imatrices 表明了模型中权重的重要性，并有助于更有效的压缩。

- **模型优化的协作努力**：一位用户提供 Humblebundle Steam 游戏作为奖励，以寻求帮助制作 Goliath 120B Longlora 模型的 iQuant 版本，并预期公开分享输出结果。

- **Phi 3 问题浮现**：多位用户报告并讨论了 Phi-3 模型的问题，包括提示词泄露和输出偏差，并提到了可供下载的更新版本 - [new 4k instruct](https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF)。

- **寻求无审查模型**：互动涉及了某些无审查模型在低配置硬件上使用的可用性和适用性，建议在 8GB RAM 配置下使用 *Everything 7b q4* 和 *wizard-vicuna-uncensored* 模型。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Snowflake/snowflake-arctic-instruct?_fsi=v2MrQoFW">Snowflake/snowflake-arctic-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF">AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/commit/c9b8888921fe528fe4be053258f48b952281bb1b">fix(root): Replaces system by user to improve generation experience. · microsoft/Phi-3-mini-128k-instruct at c9b8888</a>: 未找到描述</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-1048k-GGUF/tree/main">crusoeai/Llama-3-8B-Instruct-Gradient-1048k at main</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/lla">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/AUTOMATIC1111">AUTOMATIC1111 - Overview</a>: AUTOMATIC1111 有 41 个代码仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ceh5cp/gpt2chatbot_at_lmsys_chatbot_arena/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/shorts/fgG8E6bNwjo">Neuro Challenges Vedal</a>: 当 Vedal 挑战 Neuro 时，Neuro 会不停地在聊天框刷屏。►Twitch: http://www.twitch.tv/vedal987►Twitter: https://twitter.com/Vedal987#neurosama #vtuber #vedal
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1234538781273489408)** (31 条消息🔥): 

- **神秘的最小化和切换板块崩溃**：一名用户在应用程序从最小化恢复到全屏或在程序内切换板块时遇到随机崩溃。用户运行环境为 Windows 10 Pro，配置为高端 PC，包括 Ryzen 7 5800X, RTX 3090, 以及 64GB DDR4 RAM。

- **疑似低 RAM 的 Linux 系统**：多名 Linux 用户报告空闲 RAM 仅剩几 KB，对于报告拥有 64GB 或更多内存的系统来说，这极不寻常。这一持续存在的问题引起了社区成员的怀疑和推测。

- **Llama 异常的 HDD 活动**：
    - 一名用户注意到，在运行 Llama3m 并开启部分 GPU offload 时，尽管拥有 96GB RAM 且模型存储在 HDD 上，但每生成一个 token，HDD 都会发出特定的“咯哒”声。
    - 用户讨论了模型推理期间 HDD 使用率过高的潜在原因；可能性包括 RAM 使用过度导致交换到 pagefile 或日志写入进程。

- **GPU 并非元凶**：社区成员讨论了这种噪音是否可能是 LLM 高负载运行时的 GPU coil whine，并分享了识别硬盘声音的经验和链接，确认噪音并非由冷却系统引起。

- **故障排除继续**：关于模型运行期间奇怪的 HDD 行为的对话仍在继续，讨论了诸如 GPU offloading、context size 以及 Lexi-Llama-3-8B 模型的特殊性等细节。提醒用户将 bug 报告和求助问题保留在指定频道内。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=rJM8rHfsgjk">Hard Drive Sounds</a>: 这是我收藏的所有 HDD 硬盘声音的对比。硬盘按从旧到新的时间顺序播放。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1234495899623886911)** (74 条消息🔥🔥):

```html
<ul>
  <li><strong>聚合 GPU 上的经验</strong>：讨论指出，采用 <strong>Q4 量化</strong>的 <strong>Llama 70B</strong> 可以适配两块 RTX 3090 GPU，但由于 PCIe 总线限制，增加更多 GPU 可能会导致速度变慢。提到对于运行和微调大多数模型，两块 RTX 3090 是性价比最高的配置。</li>
  <li><strong>旧款 GPU 仍有用武之地</strong>：一位成员在 GTX 1070 上成功测试了 <em>dolphin-Llama3-8b</em> 和 <em>Llava-Phi3</em>，这表明较旧且性能较弱的 GPU 仍有潜力运行较小的模型，用于机器人项目的角色扮演等特定应用。</li>
  <li><strong>能效与运行成本</strong>：一位用户计算了在笔记本电脑上生成 1M tokens 的成本，并将其与使用 GPT-3.5 Turbo 进行了比较，发现与其设置相比，在本地运行模型比使用 API 服务更贵且更慢。</li>
  <li><strong>探索模型性能与准确性</strong>：用户之间讨论了 <em>Llama3</em> 等新型 LLM 与 GPT-4 等成熟服务相比的准确性和效率，一些人对量化版或更小、压缩程度更高的模型版本的准确性和信息质量表示怀疑。</li>
  <li><strong>寻找合适的本地模型</strong>：建议用户尝试各种模型以找到最适合其硬件的模型，建议范围从 <em>CMDR+</em>（对某些 GPU 来说可能太大）到 <em>Llama3</em> 和 <em>Wizard V2</em>，后者在普通配置上可能提供不错的性能。</li>
</ul>
```

---

**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1234783013846515752)** (5 条消息):

- **硬件难题**：一位用户在硬件上安装了 Ubuntu 并尝试运行 Linux beta 版本，但发现其 **LLM 未被接受**。他们询问该问题是否由于其硬件规格引起。
- **配置未达标**：另一位成员回应称，该用户的硬件（包括 i5-4570 和 16GB RAM）**可能不足以**运行大多数模型，大概只能有效处理 **7b Q4 模型**。
- **计划体面退出**：用户感谢了及时的反馈，并表示计划 **卸载该软件**，提到他们无法承担升级更好硬件的费用。
- **Tokenizer 问题工单**：有人请求获取 **llama.cpp 的最新 commit**，以解决 llama tokenizer 的问题，该问题正等待更新。

**提到的链接**：<a href="https://www.canadacomputers.com/product_info.php?cPath=7_4528_4570&item_id=230804">Dell Treasure Box (Black) Desktop i5-4570, 16GB, 512GB SSD, DVD, Win10</a>：戴尔 RGB Treasure Box OptiPlex SFF（翻新机）家用台式机 Intel Core i5-4570（最高 3.6GHz），16GB，512GB SSD，DVD，Windows 10 Professional (EN/FR)（黑色）

---

**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1234815876772134932)** (4 条消息):

- **寻求模型加载问题的故障排除**：一位成员表达了解决 **模型加载问题** 的紧迫性，但未提供有关问题性质的更多细节。
- **Discord 礼仪提醒**：另一位成员建议不要在无关频道重复发送问题，建议将疑问留在指定的支持频道 (*<#1111440136287297637>*)。

---

**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 条消息):

ahakobyan.：我们也能知道吗？

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1234647462166401115)** (19 条消息🔥):

- **ROCm 版本查询**：用户探讨了 **0.2.20 与 0.2.21 版本** 在 GPU offloading 方面的差异，有人质疑安装 **0.2.20 beta** 是否在 AMD 功能支持上更有优势，或者新版本是否已经包含了必要的支持。
- **VRAM 差异引起关注**：一位用户报告 **LM Studio** 显示其 **7900xtx** 的 VRAM 容量不正确，猜测可能包含了来自 Smart Access Memory (SAM) / resizable BAR 的共享内存，导致 GPU offload 预估不准确。
- **理解 GPU 和 IGPU 配置**：在讨论中，一位用户提到系统中有一个 **IGPU**，同时使用的是 **7800x3d**，其 VRAM 低于 LM Studio 显示的数值，这表明可用显存的显示可能存在偏差。
- **ROCm 兼容性困惑**：多位用户讨论了某些 AMD GPU（特别是 **RX 6600**）是否受 ROCm 支持，并澄清虽然某些旧版本可能通过 OpenCL 运行，但 RX6600 并不受 LM Studio 所使用的 **HIP SDK** 支持。
- **开发环境规范**：关于 **ROCm** 在 Windows 上的兼容性性质存在不确定性，一位用户声称在 **Ubuntu** 上成功使用 **ROCm** 运行图像生成模型，暗示 ROCm 在不同操作系统上的支持存在差异。
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1234429669860970498)** (400 messages🔥🔥): 

- **Civitai 与变现困境**：成员们对 AI 模型开发中的俱乐部和潜在付费墙表示担忧，特别是对 Civitai 的变现举措表示抵制，例如对创作者没有实际金钱收益的 Buzz 捐赠，被 [Tower13Studios](https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9) 描述为 **“坑人（rip-off）”**。
- **追求 AI 驱动的成功**：讨论显示出对通过 SFW (Safe For Work) AI 艺术赚钱的怀疑，原因是市场过度饱和。NSFW (Not Safe For Work) 作品，尤其是 furry 和 vtuber 约稿，被反复提及为 AI 生成内容中更赚钱的一面。
- **AI 图像生成速度加快**：使用 SDXL 模型和 Python 脚本快速生成图像是一个热门话题，成员们分享了代码并寻求关于提高实时应用（如 Discord 机器人）速度限制的建议。
- **准备迎接 Collider**：Stable Diffusion 的新发布引发了热切的询问，以及围绕发布日期和相比旧版本潜在改进的推测，用户分享了对该模型的期待和希望。
- **技术查询与故障排除**：用户就模型训练的各种技术层面寻求建议，例如创建 LoRAs 和 IPAdapters，以及如何克服在性能较低的硬件上运行 AI 模型时遇到的瓶颈，其他成员偶尔会提供解决方案。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dreamstudio.ai/terms-of-service">DreamStudio</a>: 未找到描述</li><li><a href="https://tenor.com/view/dj-khaled-tayomaki-sakigifs-dancing-jamming-gif-22144912">Dj Khaled Tayomaki GIF - Dj Khaled Tayomaki Sakigifs - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://civitai.com/models/428813">Mythos - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: V1 版本不知为何有 3.55GB 大.... 我想我成功做了一个稳定的 fp8 prune（剪枝）？？？？我真的不知道为什么它是 3.55GB... V2 是正常的 6GB 模式...</li><li><a href="https://civitai.com/articles/5069">Towards Pony Diffusion V7 | Civitai</a>: 大家好，我很高兴能分享我们即将推出的 V7 的进展更新，以及对 V6 的回顾分析。V6 所获得的认可...</li><li><a href="https://tenor.com/vD6Ib9MNmkI.gif">Melxts2008 Emoji GIF - Melxts2008 Emoji Smile - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/0862863bc00165b9ba0607595f304f93ca995887/tests/distributed/test_embedded_client.py#L32">ComfyUI/tests/distributed/test_embedded_client.py at 0862863bc00165b9ba0607595f304f93ca995887 · hiddenswitch/ComfyUI</a>: 一个强大且模块化的 Stable Diffusion GUI，具有图形/节点界面。 - hiddenswitch/ComfyUI</li><li><a href="https://warpcast.com/~/invite-page/404899?id=fd0fd839">Warpcast</a>: 未找到描述</li><li><a href="https://warpcast.com/~/channel/aigc">Warpcast</a>: 未找到描述</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/dreambooth">diffusers/examples/dreambooth at main · huggingface/diffusers</a>: 🤗 Diffusers：用于 PyTorch 和 FLAX 中图像和音频生成的尖端 Diffusion 模型。 - huggingface/diffusers</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cdm434/sd3_is_amazing_much_better_than_all_other/#lightbox">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://youtu.be/nLT32AR5c68?si=bV9wXlRzb_oLutW9">The Angola Effect | Horrifying death traps in the cradle of evolution</a>: 🧟‍♂️🎧 恐怖片爱好者？快来关注并收听 RUN, FOOL! —— 我们来自 Ballen Studios 的最新节目。每周二更新 - https://smarturl.it/RunFoolTime St...</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/master/script_examples/basic_api_example.py">ComfyUI/script_examples/basic_api_example.py at master · hiddenswitch/ComfyUI</a>: 一个强大且模块化的 Stable Diffusion GUI，具有图形/节点界面。 - hiddenswitch/ComfyUI
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1234429101729644615)** (322 条消息🔥🔥): 

- **Perplexity 性能骤降**：用户报告了各种模型的显著变慢和性能不佳，包括 **Japanese searches**（日语搜索），Perplexity 将查询**翻译**成英语导致产生“无意义的垃圾内容”。像 **Opus**、**Sonar Large 32K** 和 **GPT-4 Turbo** 这样的模型变得迟钝，使得平台无法使用，并阻碍了日本黄金周期间的任务。

- **Pro 订阅困惑**：用户遇到 **Pro 订阅优惠券**在到期日显示已过期的问题，与 **Nothing Phone 2(a)** 相关的优惠因欺诈行为被提前暂停。建议通过 [support@perplexity.ai](mailto:support@perplexity.ai) 联系客户支持以寻求解决。

- **免费试用取消**：提到 **7 天免费试用**因滥用已从网站移除，这引起了用户的失望，因为这被视为向新用户介绍 **Perplexity Pro** 的有效方式。

- **登录循环**：用户因**电子邮件链接延迟**而遇到登录困难，特别是对于排名比 Gmail 等服务“更低”的电子邮件，影响了 **Pro 账户访问**。

- **语音功能差异**：注意到 **iOS** 上的**语音功能**存在差异；一些用户只有之前已有的功能，而其他用户则可以使用发布视频中展示的更新版本。发现这可能取决于所使用的 **App 版本**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/Gradient_AI_/status/1785030931407143040?t=U4_FdN9hNDaE9y432-lssQ&s=19">来自 Gradient (@Gradient_AI_) 的推文</a>：我们一直在闭门研发 🔥 很高兴在 @huggingface 上发布首个上下文长度超过 1M 的 @AIatMeta Llama-3 8B 模型 —— 这是继我们发布的 160K 上下文长度模型之后的...</li><li><a href="https://flashcardfy.lol">Flashcardfy - 带有个性化反馈的 AI 闪存卡生成器</a>：通过提供个性化反馈的 AI 生成闪存卡，学习得更快、更聪明。</li><li><a href="https://chat.reka.ai/">Reka Playground</a>：探索由 Reka 构建的最新多模态语言模型。
</li>
</ul>

</div>
  

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234586871569449121)** (13 条消息🔥): 

- **深入探讨 WhatsApp 的自动回复功能**：一条消息分享了 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/whatsapp-auto-reply-JlOlDYw1Qyuik7pDTuJMuw)，探讨了 WhatsApp 中的自动回复功能。
- **揭示 'Topic 3' 的本质**：一个链接将用户引导至 [关于 Topic 3 的 Perplexity AI 搜索](https://www.perplexity.ai/search/Topic-3-One-n3JNQZT4T.ij7MosuLX5OA)，但未提供进一步的上下文或描述。
- **关于 Surroind 的研究信息**：该消息包含一个 [Perplexity AI 链接](https://www.perplexity.ai/search/research-info-surroind-oAy5SMejT4S72Fyxei7MYw#0)，推测与 "Surroind" 的研究信息有关，具体细节未说明。
- **来自 Lenny's Newsletter 关于未指定主题的见解**：用户分享了一个 [newsletter 链接](https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web)，其中包含来自 Lenny's Newsletter 的见解，重点介绍了 Lenny 对产品构建、增长驱动和职业加速等问题的处理方式。
- **关于 Vimeo API 的查询**：一位用户发布了关于 Vimeo API 的 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/Vimeo-API-kZ3X_KA2TUqmkwXzSe9ymA)，未给出查询的具体细节。

*注意：部分消息包含 Perplexity AI 搜索结果链接但未提供上下文；因此，无法总结这些主题讨论的具体内容或性质。*

**提到的链接**：<a href="https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web">How Perplexity builds product</a>：联合创始人兼产品负责人 Johnny Ho 解释了他如何像黏菌（slime mold）一样组织团队，如何利用 AI 构建他们的 AI 公司，以及更多内容。

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1234574679038230599)** (7 条消息): 

- **通过 API 寻求源 URL 访问权限**：一位用户询问 API 中是否提供 **source URLs**，并提到这在之前的路线图文档中有所列出。该功能的访问权限通过 [表单链接](https://perplexity.typeform.com/to/j50rnNiB) 提供的申请流程授予。

- **引用访问权限仍然受限**：一位成员对被拒绝访问 **source URL feature** 表示失望；在他们申请时，访问权限仅限于获得融资的初创公司。

- **关于 make.com 模型可用性的查询**：一位用户询问为什么 **Llama 3** 模型和 **Mixtral 8x22b** 未列在 make.com 的集成服务选项中。

- **请求 API 引用格式**：一位成员询问是否可以通过 **API requests** 获取引用（例如 [1]），特别是希望获得 **RAG-like knowledge over the web**（基于网络的类 RAG 知识）。

- **Perplexity 与 Anthropic 使用政策澄清**：一位用户提出了关于使用政策的平衡问题，试图了解在 **Perplexity** 的条款下使用 **Claude 3** 是否仍需遵守 **Anthropic** 的 **political usage** 限制。

**提到的链接**：<a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>：使用 Typeform 将数据收集转化为一种体验。创建精美的在线表单、调查、测验等等。免费试用。

---

**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 条消息): 

kainan_e: 已封禁（是垃圾邮件机器人）

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1234510416768667719)** (3 条消息): 

- **承诺与现实**：一位成员嘲讽了一条关于 **"pioneering the future"**（开拓未来）的过度炒作消息，结果发现那只是另一个候补名单（waitlist）公告。
- **寻找 MLOps 悬赏任务**：有人提问在哪里可以找到最好的 **MLOps bounties**，并建议需要一个类似于 Fiverr 的专注于 AI 的平台。
- **寻找程序员市场**：针对 MLOps 悬赏的查询，另一位成员质疑是否连标准编程悬赏的专用市场是否存在。

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1234469824021659729)** (6 条消息): 

- **去中心化 AI 训练**：Prime Intellect 提出了一种开源解决方案，以对抗部署 *H100 GPU clusters* 的闭源同行。他们的平台旨在通过在全球集群中实现分布式训练来克服传统计算基础设施的限制，详见其关于 [去中心化训练的博客文章](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training)。

- **通过 IN2 训练改进 LLM**：一种名为**信息密集型 (IN2) 训练 (information-intensive training)** 的新训练方案，通过对长上下文提供显式监督，解决了大语言模型的 "lost-in-the-middle"（迷失在中间）挑战。这些细节以及研究链接可以在 [arXiv 论文](https://arxiv.org/abs/2404.16811)中找到。

- **回归起源：GPT-1**：一篇博客文章回顾了原始的 GPT-1 模型，识别了其持久的关联性以及与当代模型的相似之处。正如 [amgadhasan 的 substack](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms) 所解释的，它讨论了这一早期模型如何为最新的 LLM 发展奠定基础。

- **通过协同分析理解 LLM**：一段推荐的 YouTube 视频提供了关于语言模型稳定性、拐点和连贯性分析的见解。**Synapse 的分析**可以在[这里](https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse)观看。

- **GitHub 上的 Agent 长期记忆项目**：memary 仓库展示了使用 neo4j 进行记忆存储的自主 Agent 实现长期记忆的有趣可能性。其实现和性能可以在 [GitHub](https://github.com/kingjulio8238/memary) 上探索。

- **GPT-2 Chatbot 下线**：在一次突发事件中，据 @itsandrewgao 推文称，gpt2-chatbot 被报告已下线，尽管半小时前它还处于活跃状态，该消息由 @shaunralston 发现。这一情况在 [Twitter](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw) 上引起了关注。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Andrew Gao (@itsandrewgao) 的推文</a>：gpt2-chatbot 刚刚下线了，我半小时前还在用它！感谢 @shaunralston 的发现 #gpt2 @openai</li><li><a href="https://arxiv.org/abs/2404.16811">让你的 LLM 充分利用上下文</a>：虽然许多当代的语言模型 (LLMs) 可以处理冗长的输入，但它们仍然难以充分利用长上下文中的信息，即所谓的 lost-in-the-middle 挑战。我们……</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary：自主 Agent 的长期记忆。</a>：自主 Agent 的长期记忆。通过在 GitHub 上创建账户来为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">重温 GPT-1：点燃 LLM 之火的火星</a>：全面回顾 GPT-1 对现代 LLM 发展的贡献</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">去中心化训练的最新进展</a>：本文探讨了各种新型的去中心化训练方法，以及它们如何实现在全球分布的 GPU 上进行有效的 AI 模型训练。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1234472373114372176)** (231 条消息🔥🔥): 

- **关于通过 OpenAI API 处理 PDF 的问题**：一位成员询问了通过 API 上传 PDF 的事宜，特别是寻找多模态功能。会议澄清了可以使用 [OpenAI API 中的文件搜索工具 (file search tool)](https://platform.openai.com/docs/assistants/tools/file-search)，它可以处理约 1 万个独立文件。

- **PDF 解析的挑战与解决方案**：讨论了关于 AI 模型准确解析 PDF 表格的担忧。一种建议的变通方法是独立分离并上传 PDF 中的文本和图像，[这是由于 **assistants** 平台的局限性](https://platform.openai.com/docs/assistants/whats-new/agents)。

- **模型集成实验**：一位成员分享了他们尝试结合 **Hermes 2 Pro** 和 **BakLLaVA-1** 来创建一个[带有 LLaMA 权重的简单多模态 GPT-4 模型](https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B)的尝试，这不需要微调，只需合并与 **mistral-7b-v0.1** 相关的权重。

- **GPT2-Chatbot 谜团引发社区关注**：围绕一个被称为“gpt2-chatbot”的神秘模型有很多传闻；推测范围从它是 **GPT-4.5** 的早期版本到知识截止日期为 2023 年 11 月的高级模型。尽管尝试辨别其能力，但该模型在[进一步详细测试发生之前已被移除](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw)。

- **Llama 3 通过 SigLIP 获得视觉能力**：讨论了一个突破，一名成员使用 SigLIP 为 Llama 3 实现了[视觉能力](https://huggingface.co/qresearch/llama-3-vision-alpha-hf)，使其在没有 bitsandbytes 量化支持的情况下也能直接在 Transformers 中使用。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 Andrew Gao (@itsandrewgao) 的推文</a>：gpt2-chatbot 刚刚下线了，我半小时前还在用它！感谢 @shaunralston 的发现 #gpt2 @openai</li><li><a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>：未找到描述</li><li><a href="https://google-research.github.io/seanet/audiopalm/examples/">AudioPaLM</a>：未找到描述</li><li><a href="https://x.com/hingeloss/">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/lmsysorg/status/1785394860754866234?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 lmsys.org (@lmsysorg) 的推文</a>：感谢社区难以置信的热情！我们真的没预料到。有几件事需要澄清：- 根据我们的政策，我们已经与几个模型开发团队合作...</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Q (@qtnx_) 的推文</a>：llama-3-vision-alpha 现在可以使用 @huggingface transformers 运行了</li><li><a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json">llava_instruct_150k.json · liuhaotian/LLaVA-Instruct-150K at main</a>：未找到描述</li><li><a href="https://x.com/ylecun/status/1785100806695325804?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Yann LeCun (@ylecun) 的推文</a>：有人可能会认为，到目前为止，人们应该意识到检索常见谜题的解决方案并不需要任何推理能力。 ↘️ 引用 Colin Fraser | @colin-fraser.net on bsky (@colin_...</li><li><a href="https://huggingface.co/a-normal-username/Mixtral-8x22B-OpenHermes-2.5">a-normal-username/Mixtral-8x22B-OpenHermes-2.5 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha-hf">qresearch/llama-3-vision-alpha-hf · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>：[NeurIPS'23 Oral] 视觉指令微调 (LLaVA) 旨在实现 GPT-4V 级别的能力及更高水平。- haotian-liu/LLaVA</li><li><a href="https://github.com/nestordemeure/stop_word/tree/main">GitHub - nestordemeure/stop_word: Huggingface transformers 停止准则，当遇到给定的停止词时停止生成。</a>：Huggingface transformers 停止准则，当遇到给定的停止词时停止生成。- nestordemeure/stop_word</li><li><a href="https://github.com/tincans-ai/gazelle">GitHub - tincans-ai/gazelle: 联合语音-语言模型 - 直接响应音频！</a>：联合语音-语言模型 - 直接响应音频！- tincans-ai/gazelle</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=st">来自 Q (@qtnx_) 的推文</a>：llama-3-vision-alpha 现在可以使用 @huggingface transformers 运行了</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">“我希望 Llama3 结合我的私有知识发挥 10 倍效能” - 使用 llama3 的本地 Agentic RAG</a>：高级 RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama：由 ggerganov 改进 BPE 预处理 + LLaMA 3 和 Deepseek 支持 · Pull Request #6920 · ggerganov/llama.cpp</a>：延续 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 增加了 BPE 预分词支持。摘要：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预处理...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1234577224812990635)** (19 条消息🔥):

- **LLM 训练中混合任务的共识**：一位成员建议在 LLM 训练期间混合任务更为理想，以避免与“在微调基础上再次微调”相关的性能退化。另一位成员补充说，在通用微调之上进行特定微调有时对非常专业化的任务有益。
- **对 LLama-3 8B Gradient Instruct 的主张持怀疑态度**：重点包括一个将 LLama-3 8B 上下文长度扩展到 >1040K 的模型链接，一名成员对其检索性能的主张表示怀疑，并指出可能需要根据相关的 [ArXiv 论文](https://arxiv.org/abs/2404.16811)进行进一步训练。
- **对算力需求的好奇**：关于 **LLama-3 8B Gradient Instruct** 令人印象深刻的上下文长度扩展的讨论引发了对所需计算资源的询问，回复称其需要 **512 个 L40s**。另一位成员评论说，许多应用不需要完整的 1M token 上下文窗口，但会从改进的检索性能中受益。
- **GitHub Pull Request 修复 Llama**：分享了一个更新，包括一个 [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/6920) 链接，该 PR 解决了 llama.cpp 中 LLaMA 模型支持的问题，表明改进了 BPE 预处理并支持 LLaMa 3。
- **关于 Tokenization 和 Quantization 的疑问**：关于 LLaMA 模型中 tokenizer 问题以及 GGUFs 是否需要重新量化（requantized）的对话导致了不确定性，一名成员表示 pull request 的描述对解决方案并不清晰。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>：延续了 @dragnil1 在 #6252 中的工作。此 PR 为 llama.cpp 增加了对 BPE 预分词的支持。摘要：到目前为止，对于所有基于 BPE 的模型，llama.cpp 都应用了默认的预处理...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1234865912696537130)** (6 条消息): 

- **扩展语言检索视野**：一位用户强调了一个用于**多语言稠密检索（multilingual dense retrieval）**的 [Wikipedia RAG 数据集](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7)，并链接到一篇关于利用 LLMs 合成多种语言训练数据的论文。
- **包含饮食数据**：提到的数据集包含重点关注 **Halal & Kosher**（清真与犹太洁食）的信息，表明其尝试提供多样化和包容性的数据。
- **模型选择的幕后**：一位成员表示有兴趣查看上述数据集讨论中使用了哪些模型，但未做进一步阐述。
- **开发琐事**：传达了正在参与编码活动，但未提供有关工作性质的细节。
- **将 Pydantic 集成到 Cynde**：分享了使用新工具 [Pydantic Logfire](https://pydantic.dev/logfire) 的兴奋之情，并考虑将其与 AI 工具 **Cynde** 集成。它提供了一种更简单的方式来理解应用程序，并能高效地跟踪 Pydantic 模型验证。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pydantic.dev/logfire">Pydantic Logfire | Uncomplicated observability</a>：Logfire 是一个新型的可观测性平台，其构建理念与 Pydantic 相同——即最强大的工具也可以易于使用。</li><li><a href="https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7">🦢SWIM-IR Dataset - a nthakur Collection</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1234429520203747328)** (35 条消息🔥): 

- **World Sim 将角色扮演提升到新高度**：用户透露，在 **llama 3 70b** 上运行的 *worldsim* 提示词虽然有些生硬，但非常吸引人。当启用网络搜索功能时，注意到了一些问题，会导致沟通中断。

- **与 AI 建立联系？比你想象的更有可能！**：在 **Claude 3** 上运行的 **Nous Research World Sim** 因其对话和适应性而获得赞誉。一位用户描述了一次极具说服力的互动体验，其细微差别足以媲美人类沟通。

- **实验性世界等待探索**：一位用户讨论了在**原始 WorldSim** 和自定义模拟中尝试使用 **70B 和 8B 模型**，在各种场景中遇到了历史人物有趣的涌现行为（emergent behaviors）。

- **多样化模拟器发布**：聊天中分享了指向新型 AI 驱动模拟器的链接，包括**商业**和**歌手模拟器**，展示了该技术在模拟复杂系统和个人职业方面的灵活性。

- **对 World Sim 访问的期待升温**：社区氛围活跃，用户们热切期待测试或重新参与 World Sim。讨论提到了周末可能进行公开测试的可能性，尽管目前尚未确定。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8">HuggingChat</a>：未找到描述</li><li><a href="https://huggingface.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>：在 HuggingChat 中使用 Super World Sim 助手</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>：在 HuggingChat 中使用 Snow Singer Simulator 助手</li><li><a href="https://hf.co/chat/assistant/662d91081ca01a81e3c21715">CompSim - HuggingChat</a>：在 HuggingChat 中使用 CompSim 助手</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>：在 HuggingChat 中使用 Snow World Simulator 助手
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1234626943333175307)** (28 messages🔥): 

- **揭秘 Mojo 的并发和所有权特性**：一名成员澄清说，**Mojo** 目前尚不具备**类似 Golang 的并发**或**类似 Rust 的内存安全**，因为在早期阶段**借用检查（borrow checking）是禁用的**。建议查看 GitHub 仓库以了解功能请求和路线图。
- **Mojo 尚不支持原生 Windows**：关于 Mojo 与 Windows 兼容性的讨论强调，原生支持尚未推出，但在 **Windows 的 WSL** 中构建是一个选项。有人推测未来会利用 LLVM 实现交叉编译功能。
- **探索 Mojo 替代编程语言的未来**：一名成员推测，鉴于 Mojo 充满前景的早期阶段发展，它最终可能会取代 Rust 和 Go 等语言。
- **讨论 Mojo 的 Actor 模型并发**：关于未来在 Mojo 中使用 **Actor 模型**风格并发的共识正在形成，这可以提供一种细粒度且可选的运行时方法，而不会产生巨大的开销。
- **Mojo Playground 的编译器怪癖曝光**：用户分享了使用 Mojo Playground 的经验，指出了对未识别声明（如 `ui64`）和位宽整数支持的困惑和错误。示例显示了在代码中尝试使用未知声明时的错误消息。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/engine/reference/cli/input-data-schema#data-types:~:text=ui64%3A%20unsigned%20integer%20with%20bitwidth%2064.">Input data schema | Modular Docs</a>：以下 YAML schema 允许你指定所需的输入形状</li><li><a href="https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416)">Proposal For An Actor System Based On Mojo by reid-spencer · Pull Request #1445 · modularml/mojo</a>：这目前是一个正在进行中的工作。没有代码更改，只是在提案部分写了一个提案。这在 2023 年 6 月的一次对话中得到了 Chris Lattner 的预先批准。我将继续...</li><li><a href="https://youtu.be/SEwTjZvy8vw)">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>：2023 LLVM 开发者大会 https://llvm.org/devmtg/2023-10------Mojo 🔥：一种用于异构计算的系统编程语言。演讲者：Abdul Dakkak, Chr...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1234600906893426840)** (4 messages): 

- **Modular 发布推文链接**：**Modular 的 Twitter 账号**分享了几条推文。推文的具体内容未在聊天中讨论。推文链接：[推文 1](https://twitter.com/Modular/status/1785036097292292472), [推文 2](https://twitter.com/Modular/status/1785036111804575967), [推文 3](https://twitter.com/Modular/status/1785036126224548005), [推文 4](https://twitter.com/Modular/status/1785131461345157140)。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1234433929331740702)** (2 messages):

- **Mojo 与 Python 3.12.3 的安装困扰**：一位用户报告了在 Python **3.12.3** 环境下安装 **Mojo** 的困难，另一位用户建议使用 Conda 虚拟环境在 Mac M1 上运行最新的 **Mojo** 和 **Mojo nightly** 版本。
- **Mojo 作为 Python 的超集**：**Mojo** 的目标是成为 *Python 的超集*，这意味着它应该与现有的 Python 程序和 Python 软件包生态系统兼容；然而，需要强调的是，Mojo 仍处于早期开发阶段，许多 Python 特性尚未实现。
- **桥接 Mojo 与 Python**：由于 Mojo 使用标准的 Python 解释器 CPython，用户可以从 Mojo 代码中[导入 Python 模块](https://docs.modular.com/mojo/manual/python/#import-a-python-module)、调用函数并与 Python 对象交互，从而能够无需修改地使用现有的 Python 代码。
- **使用 Conda 设置 Mojo**：建议通过 [Conda 环境](https://www.modular.com/blog/using-mojo-with-python) 来配置 **Mojo** 与 Python，以避免在同一系统上安装多个 Python 解释器时常见的路径和库冲突。

**提到的链接**：<a href="https://docs.modular.com/mojo/manual/python/">Python integration | Modular Docs</a>：共同使用 Python 和 Mojo。

---

**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1234434178922184714)** (153 条消息🔥🔥): 

- **Mojo 激发 Esolang 创作灵感**：一位成员受到启发，正在用 Mojo 为其设计的一种 esoteric language (eso lang) 编写解析器，该语言类似于 BrainF*** 但语法有所改进。他们遇到了 `None` 未实现 `__is__` 方法的问题，引发了关于 Mojo 中 `None` 和 optional types 正确用法的讨论。
  
- **Mojo 语法引起个人共鸣**：一位成员进行了一项实验，尝试结合其接触过的所有编程语言中偏好的特性，发现结果与 Mojo 的语法非常相似。这展示了 Mojo 凭借其直观的设计选择对用户的吸引力。

- **对 Mojo 新进展的热情**：在中断一段时间后，一位成员回到了 Mojo 社区，并对新特性以及 Mojo 已经开源的事实表示惊喜。这有助于提升对 Mojo 项目的关注度和参与度。

- **对 Mojo 测量宏（Measurement Macros）的兴趣**：借鉴 Julia 的 `@time` 宏，一位成员表示希望在 Mojo 中看到类似的功能，以便测量代码执行的时间和资源分配。另一位成员暗示此类功能可能会作为内置 decorators 添加。

- **关于 Windows 兼容性的疑问**：关于 Mojo 在 Windows 上可用时间表的咨询表明，社区成员渴望跨平台支持。去年 10 月给出的预期是“很快”，这让一些成员期待进度的更新。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/Matmul">Mojo 中的矩阵乘法 | Modular 文档</a>：学习如何利用 Mojo 的各种函数来编写高性能的 matmul。</li><li><a href="https://github.com/search?q=repo%3Amodularml%2Fmojo+%22None%22&type=code&p=0)">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://mojodojo.dev/mojo-team-answers.html#unsafe-code">Mojo 团队回答 | Mojo Dojo</a>：未找到描述</li><li><a href="https://rosettacode.org/wiki/99_Bottles_of_Beer/EsoLang">99 Bottles of Beer/EsoLang</a>：未找到描述</li><li><a href="https://github.com/karpathy/minbpe">GitHub - karpathy/minbpe：用于 LLM 分词中常用的字节对编码（BPE）算法的极简、整洁代码。</a>：用于 LLM 分词中常用的字节对编码（BPE）算法的极简、整洁代码。 - karpathy/minbpe</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">让我们构建 GPT Tokenizer</a>：Tokenizer 是大语言模型（LLMs）中一个必要且普遍存在的组件，它在字符串和 token（文本块）之间进行转换。Tokenizer...</li><li><a href="https://youtu.be/kgUXfDpAmGQ?si=VmrPUT7YLBmzMq8I">C++ 作为优化汇编器 - 性能演讲 - Levo DeLellis - CppNorth 2023</a>：https://www.cppnorth.ca --- C++ 作为优化汇编器 - 性能演讲 - Levo DeLellis - CppNorth 2023。你是否厌倦了抽象、模板和...</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/620">[功能请求] 原生 Windows 支持 · Issue #620 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？对 Windows 的原生支持。什么时候可用？...</li><li><a href="https://github.com/modularml/mojo/issues/620#issuecomment-2082106584">[功能请求] 原生 Windows 支持 · Issue #620 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？对 Windows 的原生支持。什么时候可用？...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1234494559527108669)** (4 条消息): 

- **Mojo 开发者社区焕发生机**：一个名为 *用Mojo写一个Mojo社区* 的 Mojo 社区项目已在 GitHub 上分享。该项目可在 [shadowqcom/mojo_dev](https://github.com/shadowqcom/mojo_dev) 查看。
- **atol-simd 提速**：[atol-simd 项目](https://github.com/VMois/mojo-atol-simd) 报告称，对于 15-16 个字符的字符串，其性能比 stdlib atol 提升了 **20%**，尽管对于较短的字符串，stdlib 仍然稍快一些。仓库中包含了基准测试。
- **发出协作邀请**：一位社区成员表示有兴趣为 atol-simd 项目做出贡献，并邀请协作机会。
- **SIMD 项目共享向量化模式**：在关于 SIMD 库的讨论中，提到了另一个项目 [mojo-fast-base64](https://github.com/mzaks/mojo-fast-base64)，强调了在输入不适合向量化时回退到标量处理的常见模式。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/shadowqcom/mojo_dev">GitHub - shadowqcom/mojo_dev：用Mojo写一个Mojo社区！</a>：用Mojo写一个Mojo社区！。通过在 GitHub 上创建账户为 shadowqcom/mojo_dev 的开发做出贡献。</li><li><a href="https://github.com/mzaks/mojo-fast-base64">GitHub - mzaks/mojo-fast-base64</a>：通过在 GitHub 上创建账户为 mzaks/mojo-fast-base64 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1234485565181657214)** (40 条消息🔥):

- **纠错编码优化探索**：一场围绕 [mocodes GitHub 仓库](https://github.com/alainrollejr/mocodes) 中基于 SIMD 函数性能改进的持续讨论。成员们就 LLVM/MLIR 优化技术的潜力以及一个看似简单的函数生成的惊人汇编代码量交换了意见。
- **强大的 Mojo 基准测试**：一位成员分享了他们在 1brc（十亿行挑战）项目中的进展，实现了令人印象深刻的迭代速度，并提供了代码仓库供协作。对话涉及了在性能测试中使用 nightly 构建版本与稳定版本的优劣。
- **Nightly 版本中的 Bug 追踪**：一位成员提出了 `FileHandle.read_bytes()` 导致内存问题的问题，随后被确认为 GitHub 上已报告的已知问题。
- **Team Mojo 集合！**：有人提议组建一个 "team-mojo" 来应对 1brc 挑战，旨在将其打造为社区的展示案例和教程。这与另一项建议相呼应，即解决 Mojo 与其他语言对比的基准测试问题，这一领域尚未得到充分探索。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/MoSafi2/BlazeSeq/blob/main/blazeseq/iostream.mojo">BlazeSeq/blazeseq/iostream.mojo at main · MoSafi2/BlazeSeq</a>：通过在 GitHub 上创建账号来为 MoSafi2/BlazeSeq 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479)">"Mojo 比 Python 快 68,000 倍"这类博客很棒，但能否与其他语言也进行出色的对比？ · modularml/mojo · Discussion #843</a>：Mojo 比 Python 快 35,000 倍、68,000 倍……这令人印象深刻且非常酷，但对于那些尚未关注 Mojo 的非 Python 用户和反 Python 人士来说……</li><li><a href="https://github.com/alainrollejr/mocodes">GitHub - alainrollejr/mocodes: 使用 Mojo 进行纠错（编）解码</a>：使用 Mojo 进行纠错（编）解码。通过在 GitHub 上创建账号来为 alainrollejr/mocodes 的开发做出贡献。</li><li><a href="https://github.com/MoSafi2/1brc-mojo/tree/dev">GitHub - MoSafi2/1brc-mojo at dev</a>：使用 Mojo 语言完成的十亿行挑战 (1brc)。通过在 GitHub 上创建账号来为 MoSafi2/1brc-mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2051">[stdlib] 使用 `FileHandle.read_bytes()` 时不要复制元素 · Issue #2051 · modularml/mojo</a>：我正在用 Mojo 进行十亿行挑战，尝试使用 read_bytes() 读取 10 亿行（约 13GB 文件），结果很快就耗尽了内存。使用 read() 则不会发生这种情况。alias input_f...</li><li><a href="https://github.com/VMois/1brc-mojo">GitHub - VMois/1brc-mojo: 使用 Mojo 语言完成的十亿行挑战 (1brc)</a>：使用 Mojo 语言完成的十亿行挑战 (1brc)。通过在 GitHub 上创建账号来为 VMois/1brc-mojo 的开发做出贡献。</li><li><a href="https://github.com/VMois/mojo-atol-simd">GitHub - VMois/mojo-atol-simd: 在 Mojo 中使用 SIMD 将字符串转换为整数（目前支持最多 16 个字符）</a>：在 Mojo 中使用 SIMD 将字符串转换为整数（目前支持最多 16 个字符） - VMois/mojo-atol-simd
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1234682806752247818)** (2 条消息): 

- **仓库更新带来准确的速度结果**：在从仓库拉取最新更新后，一位成员观察到了准确的速度提升报告。然而，他们也注意到在基准测试期间其 CPU 未达到最高频率，且与 PyTorch 和 TensorFlow 相比，MAX 在较低的 CPU 时钟频率下表现更好。

- **ModularBot 升级**：ModularBot 庆祝其达到 **1 级**，标志着其在 Discord 环境中运行的一个里程碑。
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1234618988965789747)** (51 条消息🔥): 

- **EqualityComparable SIMD 讨论**：讨论了一个 [拉取请求 (PR)](https://github.com/modularml/mojo/pull/2412)，该变更使 `SIMD` 符合 `EqualityComparable` 协议而不改变原始行为。然而，这可能会导致现有代码中大小大于 1 的 `SIMD` 隐式转换为 `Bool` 时出现问题。

- **SIMD 转 Scalar：显式优于隐式**：关于 `SIMD` 的讨论强调了在从 `SIMD` 转换为 `Scalar` 时需要显式使用 `reduce_and` 或 `reduce_or`。有人认为 `SIMD.__bool__()` 由于当前的实现方式会导致 Bug 和困惑。

- **Mojo 编译器 Nightly 版本发布警报**：宣布了新的 Nightly Mojo 编译器版本，鼓励用户使用 `modular update nightly/mojo` 进行更新。可以通过 [GitHub 上的 diff](https://github.com/modularml/mojo/pull/2449/files) 和 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 查看更改内容。

- **关于 SIMD 和布尔转换的辩论**：针对 `bool(SIMD[type, size])` 的恰当行为展开了辩论，即它应该返回 `SIMD[bool, size]` 还是保持标量布尔表示。一些人认为保持将 `bool` 作为逻辑接口的能力非常重要，这可能会影响 `if` 和三元表达式等操作。

- **Source Location 函数在 Nightly 版本中被移动**：关于 `__source_location()` 的讨论显示，在 Nightly 版本中它可能已被 `__call_location()` 取代。经过一番讨论后，分享了 [示例用法](https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo)，以澄清在新版本编译器中如何导入和使用该函数。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D">context:global __source_… - Sourcegraph</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo">modularml/mojo Nightly 分支下的 mojo/stdlib/src/testing/testing.mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2412">[stdlib] SIMD 对 EqualityComparable 的一致性，由 helehex 提交 · Pull Request #2412 · modularml/mojo</a>：这允许 SIMD 符合 EqualityComparable，且不丢失任何原始行为。它使用第 4 条重载解析规则赋予新方法较低的优先级，同时仍保持一致性...</li><li><a href="https://github.com/modularml/mojo/pull/2449/files">[stdlib] 根据 2024-04-29 nightly/mojo 更新 stdlib，由 JoeLoser 提交 · Pull Request #2449 · modularml/mojo</a>：这使用与今天的 Nightly 版本（mojo 2024.4.2923）相对应的内部提交更新了 stdlib。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">modularml/mojo Nightly 分支下的 mojo/docs/changelog.md</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1234762736504672346)** (2 条消息): 

- **CVPR 2023 宣布高额奖金竞赛**：HF 竞赛平台上宣布了 CVPR 2023 会议的三项新竞赛：[SnakeCLEF](https://huggingface.co/spaces/BVRA/SnakeCLEF2024)、[FungiCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024) 和 [PlantCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024)，总奖金超过 12 万美元。活动将于 2024 年 6 月 17 日至 21 日举行。
- **第 100 期 Hugging News**：庆祝 *Hugging News* 第 100 期，重点介绍了 **Transformers v4.40.0**、**Gradio 4.28.0**、**Datasets v2.19.0**、**Optimum v1.19.0** 的发布，以及多项社区互动更新，包括在 HuggingFace 上提及（mention）他人的功能。显著亮点包括 [Phi-3 在浏览器中运行](https://x.com/fleetwood___/status/1783195985893863578) 以及 [Common Voice 17 在 Hub 上线](https://x.com/reach_vb/status/1785039538185703909)。
- **在 Kaggle 上运行 AutoTrain UI**：在一个分享的 notebook 中，向用户展示了如何在 Kaggle Notebooks 后端运行 AutoTrain UI，进一步增强了机器学习项目的可访问性。该指南可在 [此 Kaggle notebook](https://www.kaggle.com/code/abhishek/autotrain-ui) 复制并使用。
- **Snowflake 发布巨型 MoE 模型**：Snowflake 发布了一个新的 [408B 参数 Dense + Hybrid MoE 模型](https://x.com/reach_vb/status/1783129119435210836)，拥有 4K 上下文窗口并完全采用 Apache 2.0 许可，因其在复杂任务上的出色表现而引发关注。
- **社区增长与产品公告**：公告强调了在 HuggingFace Hub 上为 [记者建立的新社区](https://x.com/BrigitteTousi/status/1783573043815596426)，以及社区驱动内容的集成，例如如何使用 **Diffusers 中的自定义流水线**，并号召参与 **ML 论文研读小组**。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/fleetwood___/status/1783195985893863578)">来自 Fleetwood (@fleetwood___) 的推文</a>：🚨 Phi-3 在浏览器中运行 🚨 速度达到约 20 tok/s 🏎️ 仅需 3 行 JS 代码。仍有一些小问题需要解决，即将集成到 Ratchet 0.4.0 中。</li><li><a href="https://x.com/abhi1thakur/status/1785279012232736991)">来自 abhishek (@abhi1thakur) 的推文</a>：我能在 Kaggle 上运行 AutoTrain UI 吗？是的，你可以！！！查看我最新的 notebook，复制它，填入你的 token，即可享受在 Kaggle Notebooks 后端运行的 AutoTrain UI 🚀 notebook 链接：https://www...</li><li><a href="https://x.com/reach_vb/status/1785039538185703909)!">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：冲！！Common Voice 17 - 现已在 Hub 上发布！🔥 包含 124 种语言的 31,000 小时音频（及转录）。*开启声音 🎶* CV 17 中新增了 847 小时数据，以及 493 小时的...</li><li><a href="https://x.com/BrigitteTousi/status/1783573043815596426):">来自 Brigitte 🤗 (@BrigitteTousi) 的推文</a>：🔊 呼叫所有记者！我们很高兴与 @fdaudens 一起宣布在 @huggingface Hub 上建立一个新社区：Journalists on Hugging Face。📰🤗 https://huggingface.co/JournalistsonHF 1/</li><li><a href="https://x.com/reach_vb/status/1783129119435210836)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：Snowflake 发布了一个 408B Dense + Hybrid MoE 🔥 > 17B 激活参数 > 128 个专家 > 在 3.5T tokens 上训练 > 使用 top-2 gating > 完全采用 Apache 2.0 许可（附带数据方案...）</li><li><a href="https://x.com/RisingSayak/status/1785162074844197174)">来自 Sayak Paul (@RisingSayak) 的推文</a>：Diffusers 中的自定义 Pipeline 和组件 🎸 想要在 Diffusers 中使用自定义 Pipeline 和其他组件（schedulers, unets, text encoders 等）？觉得不够灵活？这个 🧶（推文串）就是为你准备的...</li><li><a href="https://x.com/lunarflu1/status/1785359306847666431)">来自 lunarflu (@lunarflu1) 的推文</a>：你现在可以在 @huggingface 上提及（mention）别人了！
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1234451724257984574)** (208 条消息🔥🔥): 

- **寻求 LLM 可观测性工具**：一位成员请求关于 LLM 可观测性工具的建议，特别是对兼容 LlamaIndex 且倾向于自托管开源选项的工具感兴趣。
- **huggingchat 的 API 交互协助**：有人寻求通过 API 调用与 [Hugging Face Chat](https://huggingface.co/chat/) 进行通信的帮助，表示需要指导。
- **为 Gradio 专家提供悬赏**：一位成员对 Gradio 的问题感到沮丧，提供 200 美元悬赏寻求高质量帮助，随后被引导至 Gradio 专用频道寻求支持。
- **弹珠台 AI 视觉模型讨论**：围绕开发一个识别弹珠台游戏和分数的 AI 模型展开了详细对话，讨论了复杂性、工具、图像分类的必要性，以及复用 llava 等现有模型作为部分解决方案的可行性。
- **LLM 的电脑配置**：一位用户正在寻找针对 LLM 的 DDR5 和 CPU 性能资源，考虑为其新电脑配置高规格硬件。其他成员分享了与 AI 工作硬件选择相关的建议和个人经验。
- **Zero GPU Explorer 会员查询与笑话**：聊天中显示出对 Zero GPU Explorers 会员资格和订阅状态的困惑，同时一些成员幽默地尝试使用 AI 相关的搭讪词来“撩” Hugging Face 的开发者。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://apply.workable.com/huggingface/?lng=en">Hugging Face</a>：在 Hugging Face，我们致力于为每个人推进和民主化 ML。在此过程中，我们为技术向善的发展做出贡献。</li><li><a href="https://x.com/noaroggendorff/status/1785095305408422234">来自 Noa Roggendorff (@noaroggendorff) 的推文</a>：懂的都懂</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/26">zero-gpu-explorers/README · 邀请申请一直在等待。审批需要多长时间？</a>：未找到描述</li><li><a href="https://huggingface.co/amazon/chronos-t5-small">amazon/chronos-t5-small · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/tasks/image_classification">图像分类</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/25">zero-gpu-explorers/README · 更新 README.md</a>：未找到描述</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">“我希望 Llama3 结合我的私有知识实现 10 倍性能” - 本地 Agentic RAG 与 llama3</a>：高级 RAG 101 - 使用 llama3 构建 agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file">GitHub - amazon-science/chronos-forecasting: Chronos: 用于概率时间序列预测的预训练（语言）模型</a>：Chronos: 用于概率时间序列预测的预训练（语言）模型 - amazon-science/chronos-forecasting</li><li><a href="https://huggingface.co/blog/personal-copilot">Personal Copilot: 训练你自己的编程助手</a>：未找到描述</li><li><a href="https://github.com/pacman100/LLM-Workshop/blob/main/personal_copilot/training/train.py">LLM-Workshop/personal_copilot/training/train.py at main · pacman100/LLM-Workshop</a>：Sourab Mangrulkar 的 LLM 工作坊。通过在 GitHub 上创建账号为 pacman100/LLM-Workshop 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/trl/v0.8.6/en/sft_trainer#trl.trainer.ConstantLengthDataset">有监督微调训练器 (Supervised Fine-tuning Trainer)</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1234517512213889147)** (2 条消息): 

- **学习热情**：一名成员表达了在频道中分享和接收信息的兴奋之情，标志着一个积极且协作的学习环境。
- **寻求微调指导**：提出了关于创建用于微调大语言模型 (LLM) 的指令数据集的最佳实践咨询，表明了对为模型增强而定制数据集准备的兴趣。
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1234513287299731578)** (9 条消息🔥): 

- **深入探索深度学习**：MIT 深度学习导论课程现已更新至 2024 年，提供了深度学习概念的基础理解。[讲座视频](https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2) 已在 YouTube 上线，供任何对该领域感兴趣的人学习。
  
- **文本生成图像模型评估**：即将举行一场关于文本生成图像模型评估的演讲，演讲者将讨论文本与图像的一致性以及模型的鲁棒性。

- **Stallman 歌唱自由**：一段 YouTube 视频展示了 Richard Stallman 在厄瓜多尔的一次活动中演唱“自由软件之歌”。这个奇特的时刻可以在 [这里](https://www.youtube.com/watch?v=9sJUDx7iEJw) 找到。

- **社区计算机视觉课程发布**：Hugging Face 发布了一门面向所有人的社区驱动计算机视觉课程，包括如何加入学习者社区、提交作品以及认证信息。通过他们的 [欢迎页面](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) 开始学习。

- **AI 安全基准受到关注**：一篇 LinkedIn 帖子宣布了 LLM Safety LeaderBoard，这是一个衡量 AI 安全、保障和负责任 AI 实践的新平台。点击 [这里](https://www.linkedin.com/posts/divyanshuusingh_safetyleaderboard-aisecurity-responsibleai-activity-7190907558071558145-qeVK) 了解更多关于该排行榜的信息。

- **通过 GenAI 发现 5 个 AI 工具**：一篇名为“GenAI 冒险：每个人都应该尝试的 5 个有趣的 AI 工具”的 Medium 文章介绍了一份精选的 AI 工具列表。读者可以在 [Medium](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af) 上探索这些工具。

- **构建直观的 RAG 应用**：一篇文章指导如何使用 Groq、Langchain 和 Datastax 创建具有强大功能的 webloader RAG 应用。感兴趣的读者可以在 [Medium](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8) 上深入研究这些集成。

- **使用机器学习简化数据库查询**：一种创新方法正在开发中，允许用户利用 RAG 和 Gemini，在仅需极少 SQL 知识的情况下查询“人员数据库”。有关该项目的更多详情可以在 [Datai Alliance](https://www.dataialliance.org) 找到。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>：未找到描述</li><li><a href="https://www.dataialliance.org">blog</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=9sJUDx7iEJw">Richard Stallman Free software Song</a>：Richard Stallman 在厄瓜多尔演唱自由软件之歌，由 Julian Coccia 录制。</li><li><a href="https://www.youtube.com/watch?v=ErnWZxJovaM&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=2">MIT Introduction to Deep Learning | 6.S191</a>：MIT 深度学习导论 6.S191：第 1 讲 *2024 新版* 深度学习基础。讲师：Alexander Amini。包含所有课程、幻灯片和实验材料...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1234430834967318530)** (13 条消息🔥): 

- **模型发布困境**：一则帖子提到在五个待发布模型中选择其一的困境，包括邀请大家就下一个应推出的模型提供建议或偏好，并提供了一个 [LinkedIn 帖子链接](https://www.linkedin.com/posts/bineric_llm-ai-europe-activity-7190590676055506944-QW9f) 以获取更多背景信息。
- **来自 LifePal 的问候**：介绍了一款名为 LifePal 的新型 AI 驱动应用，它作为平衡生活的个性化指南，并声称与 Apple Vision Pro 无缝集成。它被描述为生活副驾驶（life co-pilot），并展示了其可感知的优势和功能，同时附带了 [Apple Store 链接](https://apps.apple.com/se/app/lifepal-ai-chat-assistant/id6471972439)。
- **ChatGPT 的挪威语有待改进**：一名成员指出 ChatGPT 的挪威语翻译表现欠佳，因此需要通过带有当地俚语的检索增强生成（RAG）进行重新处理，并提到了一款专为挪威语理解和生成设计的替代方案 [NorskGPT-Mistral](https://huggingface.co/bineric/NorskGPT-Mistral-7b)。
- **招募高级研究助手和搜索引擎的 Beta 测试人员**：提供了一个招募高级研究助手和搜索引擎工具 Beta 测试人员的机会，提供 2 个月的免费高级服务，包含 GPT-4 Turbo、Mistral Large 等多种模型。感兴趣的人员可前往 [Rubik's AI](https://rubiks.ai) 并使用促销代码获取免费高级优惠。
- **Hugging Face 上的创新 Inpainting SDXL**：分享了一个名为 SDXL 的 Inpainting 工具的独特版本，允许在之前的生成结果上进行迭代 Inpainting 并保留版本历史。鼓励用户提供反馈并分享示例，该 [Inpainting 工具可以在 Hugging Face 上找到](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">Inpainting SDXL Sketch Pad - a Hugging Face Space by tonyassi</a>：未找到描述</li><li><a href="https://huggingface.co/bineric/NorskGPT-Mistral-7b">bineric/NorskGPT-Mistral-7b · Hugging Face</a>：未找到描述</li><li><a href="https://apps.apple.com/se/app/lifepal-ai-chat-assistant/id6471972439">‎LifePal AI Chat &amp; Assistant</a>：‎探索 LifePal：您的生产力 AI 伴侣。您准备好释放全部潜力，过上更健康、更快乐的生活了吗？LifePal 将引导您开启成为更好的自己的旅程...</li><li><a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">GitHub - Lama-West/PnPR-GCN_ACM_SAC_24</a>：通过在 GitHub 上创建账户来为 Lama-West/PnPR-GCN_ACM_SAC_24 的开发做出贡献。</li><li><a href="https://vimeo.com/940824094?share=copy">Vinner - Nybygg i og rundt Bergen</a>：非常感谢 Sn&oslash;hetta</li><li><a href="https://github.com/GDSC-FSC/gemini-node-1">GitHub - GDSC-FSC/gemini-node-1</a>：通过在 GitHub 上创建账户来为 GDSC-FSC/gemini-node-1 的开发做出贡献。</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>：未找到描述
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1234684966013767731)** (12 messages🔥): 

- **图与 LLMs 阅读准备**：一位成员宣布计划回顾[关于大语言模型 (LLMs) 及其与图交互的论文](https://arxiv.org/abs/2404.14928)，重点关注复杂关系的表示，并讨论下周六进行演示的可能性。
- **周六会议的额外论文综述**：同一位成员还考虑回顾两篇综述论文，一篇关于[应用于图的 LLMs](https://arxiv.org/abs/2312.02783)，另一篇关于[基础模型 (foundation models)](https://arxiv.org/abs/2310.11829)，并建议这些主题也可能被纳入，但指出需要避免在未来的阅读小组中过于分散精力。
- **探索基于分数的模型蒸馏**：一位聊天参与者询问了关于蒸馏基于分数 (score-based) 的模型的资源，特别是那些与经典的 SDE solver 模型相比能减少生成步骤的模型。
- **蒸馏论文和社区指南**：一份回复引导上述询问者前往 Laion 和 Eleuther 服务器，那里聚集了模型蒸馏方面的专家，并推荐了领先的研究员 Gothos，同时提到了 rectified flow 和 LCM Lora 领域的相关论文。
- **论文阅读活动创建**：小组内初步安排了一场活动，允许讨论时间调整，并鼓励成员参与即将举行的关于 LLMs 与图交互的阅读和演示。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>：图在表示社交网络、知识图谱和分子发现等各个领域的复杂关系中发挥着重要作用。随着深度学习的出现，图神经网络...</li><li><a href="https://discord.gg/hugging-face-879548962464493619?event=1234913780048203856">加入 Hugging Face Discord 服务器！</a>：我们正致力于民主化优秀的机器学习 🤗 验证以链接您的 Hub 和 Discord 账号！| 77552 名成员</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>：大语言模型 (LLMs)，如 GPT4 和 LLaMA，凭借其强大的文本编码/解码能力和新发现的涌现能力，正在自然语言处理领域取得重大进展...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>：基础模型已成为各种人工智能应用中的关键组件，并在自然语言处理和其他几个领域展示了显著的成功...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1234548112270426135)** (15 messages🔥): 

- **平衡准确率与效率**：一位成员讨论了在以原始分辨率处理边界框 (bounding boxes) 时，计算效率与模型准确率之间的权衡。另一位成员建议使用模糊等图像预处理技术来优化 VRAM 占用。
- **图像分割模型探索**：在寻求图像分割进阶指导时，OneFormer、MaskFormer、Segformer 被提及作为该成员已研究过的一系列模型。
- **寻找 CNN 学习伙伴**：一位成员表示有兴趣寻找学习和研究卷积神经网络 (CNNs) 的学习伙伴。
- **传统轮廓算法与现代预处理的结合**：在讨论 YOLO 架构时，一位成员建议回顾 YOLO/CNN 之前的图像分割和轮廓查找算法，并提到预处理和下采样仍然可以产生良好的效果。分享了 OpenCV 关于形态学操作和图像处理的文档链接：[形态学操作](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html)，[图像处理目录](https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html)。
- **CNN 项目中 PyTorch 与 TensorFlow 的选择**：对话涉及是学习 PyTorch 还是坚持使用 TensorFlow，强调了 PyTorch 在社区和学术界的势头，以及 TensorFlow 来自 Google 的强大 DevOps 支持。再次确认了使用 TensorFlow 创建涉及目标检测和图像分割项目的灵活性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html">OpenCV: Image Processing in OpenCV</a>: 未找到描述</li><li><a href="https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html">OpenCV: Morphological Transformations</a>: 未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1234572716359548999)** (3 messages): 

- **寻求 NLU/NLP 指导**：一位新成员正在使用 *Rasa framework* 开发聊天机器人，但在意图识别方面遇到问题，即通用的销售查询被错误地分类为特定公司的销售意图。
- **致力于增强意图识别**：他们正在考虑创建一个自定义 NER 模型，将特定关键词识别为意图（销售、购买等），并使用数据库中的公司名称作为 *NER-company*，以提高聊天机器人的性能。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1234554096149725184)** (4 messages): 

- **Hyper-SD 和 IP-Adapter 的写实性挑战**：一位用户分享了在使用 **Hyper-SD** 配合 **IP-Adapter** 时无法获得写实结果的问题。他们提供了一个 [discussion link](https://github.com/huggingface/diffusers/discussions/7818) 指向 GitHub，并在那里详细阐述了该问题。
- **对不同模型间结果不一致感到惊讶**：一位用户在从 **Seaart 切换到 A1111** 后感到困惑，发现尽管使用了相同的设置和 seed，图像的颜色和阴影质量却发生了变化。他们询问是否存在后端差异，以及是否可能在两个模型上实现统一的结果。
- **DeepFloyd 的不可预测行为**：据一位用户称，**DeepFloyd** 在调整 step count、sampler 和 CFG 时表现出奇怪的模式。他们将其与 **Ambigram** 研究模型进行了对比，并分享了对不同设置（特别是 **DPM++ 2M** scheduler）性能的见解。


**Link mentioned**: <a href="https://github.com/huggingface/diffusers/discussions/7818">Not getting good realistic results with Hyper-SD + IP-Adapter · huggingface/diffusers · Discussion #7818</a>: 大家好，（也许 @asomoza 知道这个？）Hyper-SD 能很好地配合 IP-Adapter 工作吗？我正在按照仓库中的说明在 Diffusers 中测试 Hyper-SD。我原以为会得到更好的结...

  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1234862689357009087)** (1 messages): 

- **Gradio Share Server 故障**：Gradio 的 Share Server 遇到了问题，可能会影响 Colab 上的分享和使用。他们正在积极调查并解决该问题；更新信息可在其 [status page](https://status.gradio.app/) 查看。
- **随时检查 Gradio 的运行状况**：如需了解 Gradio 在过去 90 天（包括过去 24 小时、一周和一个月）的运行状态概览，请参考其 [calendar view](https://status.gradio.app/#)。
- **过去一周运行良好**：过去 7 天内没有状态更新，表明没有发生新事件。历史状态更新可以在 [status update history](https://status.gradio.app/#) 页面查看。

**Link mentioned**: <a href="https://status.gradio.app/">Gradio Status</a>: 未找到描述

  

---



**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1234571459699933314)** (3 messages): 

- **OpenRouter 探索 Syrax**：Alex Atallah 表示开始尝试使用 **Syrax** 并向团队提供支持，提议组织一个群聊。
- **热情接受合作**：Mart02 确认并感谢了 Alex 的联络，通过接受好友请求标志着合作努力的开始。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1234433355626319872)** (240 messages🔥🔥): 

- **寻求非技术性部署的前端**：一位成员询问是否有一种多用户前端可以部署在共享主机上，而无需 Docker 或 Node.js。*LibreChat* 被推荐为最合适的选项，但另一位成员提到了托管挑战和成本问题，随后有人建议将 Vercel 的免费层托管作为潜在解决方案。

- **LLM 的对比与期待**：用户对各种大语言模型进行了激烈的讨论，包括 *Llama-3 8B*、*Dolphin 2.9* 和 *Mixtral-8x22B*。用户分享了对模型能力的见解，例如 context window 大小，以及模型根据其对话风格和数据集被审查的可能性。

- **模型训练探险**：一位用户分享了他们尝试通过使用自己的有毒数据集（toxic dataset）来训练模型变得更加“不受限（unhinged）”的历程。用户对不同模型的行为进行了对比，并讨论了 LLM 是否能有效处理大上下文，达成的共识是：虽然像 *Llama 3 8B* 这样的模型可以处理长上下文，但其性能在超过一定限度后可能会下降。

- **经济型模型实验与发现**：成员们讨论了 OpenRouter 平台上性价比高的模型选项。*Mixtral-8x7B-Instruct* 被认为是价格与性能之间合理的平衡点，一位用户对 *GPT-3.5* 输出质量的提升表示惊讶，认为其写作风格更像人类。

- **OR 修正消息顺序的功能**：有一个关于 *Claude 3* 如何处理 assistant/user 消息顺序的问题。确认了 **OpenRouter** 会自动修正顺序以确保模型正常工作，并鼓励用户报告可能遇到的任何顺序问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cws-docs.pages.dev/en/">首页 | ChatGPT Web Share 文档</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/jondurbin/cinematika-7b-v0.1">jondurbin/cinematika-7b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/psyonic-cetacean-20B-AWQ">TheBloke/psyonic-cetacean-20B-AWQ · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/maywell/Llama-3-8B-Instruct-1M">maywell/Llama-3-8B-Instruct-1M · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/erhartford/status/1784315764079796541?s=46&t=2a7uDiV3mox9o-E5jIFbLQ">来自 Eric Hartford (@erhartford) 的推文</a>：dolphin-2.9-llama3-8b-256k 已发布。它是应用了 @winglian 出色的 256k 上下文适配器的 dolphin-2.9-llama3-8b。我今天会完成模型卡片。</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9-mixtral-8x22b">cognitivecomputations/dolphin-2.9-mixtral-8x22b · Hugging Face</a>：未找到描述</li><li><a href="https://rentry.org/GPT2/#main-alternative-theory">gpt2-chatbot</a>：此页面正在完善中。随着收集到更多信息，其结论可能会发生变化。截至 2023-04-30 的新闻：gpt2-chatbot 极有可能运行在由某公司运营或关联的服务器上...</li><li><a href="https://www.clay.com/">Clay - 规模化个性化外联</a>：结合 50 多个数据提供商、实时抓取和 AI，发送 1 对 1 个性化营销活动，预订更多会议。</li><li><a href="https://huggingface.co/datasets/jondurbin/cinematika-v0.1">jondurbin/cinematika-v0.1 · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://openrouter.ai/models/openrouter/cinematika-7b">openrouter 的 Cinematika 7B (alpha) | OpenRouter</a>：该模型正在开发中。查看 [OpenRouter Discord](https://discord.gg/fVyRaUDgxW) 获取更新。</li><li><a href="https://www.cyon.ch/hosting/managed-server">托管服务器：您自己的服务器，总部位于瑞士</a>：未找到描述
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1234521268070645790)** (4 条消息): 

- **高级 RAG 参考架构发布**：**LlamaIndex** 团队发布了一个在 **AWS 生态系统**中构建高级 RAG（检索增强生成）系统的参考架构。该资源提供了关于高级解析和智能体推理（agentic reasoning）的指导，可通过共享的 [代码仓库](https://t.co/sfQOvhHHg5) 获取。

- **黑客松获胜者开发文档机器人**：最近一次黑客松的获胜者 **Team CLAB** 开发了一个全栈文档机器人，该机器人集成了用于解析和编排的 **LlamaIndex** 以及 **Nomic embeddings**。关于该项目和黑客松的更多详情可以在链接的 [博客文章](https://t.co/2UMqrHwO56) 中找到。

- **使用 Agentic RAG 创建金融助手**：一项新进展允许直接基于非结构化财务报告构建能够处理复杂计算（如百分比演变和 **CAGR**）的金融助手。[最近的一篇文章](https://t.co/6cTNxUBJcr)解释了如何在不需要人工数据转换步骤的情况下实现这一目标。

- **使用 Semantic Caching 构建高效 RAG**：通过与 @Redisinc、@tchutch94 和 @seldo 的合作，展示了如何构建包含 **Semantic Caching** 的高性能 RAG 应用，以加速频繁发起的查询。正如[合作文章](https://t.co/oGxFrZLMRn)中所讨论的，这项创新旨在提高质量、效率和成本效益。

**提到的链接**：<a href="https://t.co/oGxFrZLMRn">未找到标题</a>：未找到描述

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1234440203788222516)** (159 条消息🔥🔥): 

- **对 Assistant Agent V2 的期待**：成员们正在询问 **LlamaIndex OpenAI Assistant Agent V2** 的更新或发布情况，以便利用新版 OpenAI Assistant V2 的功能。目前，该版本尚无具体的更新或 Pull Request。
  
- **更新 Pinecone 索引查询**：关于在 Pinecone 中更新索引部分的说明文档尚不完善。虽然成员建议使用 `pinecone_index.update` 等方法，但 LlamaIndex 知识库中并未提供使用 `SimpleDirectoryReader` 的直接示例。

- **LLM Observability 工具偏好**：关于 **Arize Phoenix** 和 **Langfuze** 哪种 LLM Observability 工具更好的讨论。一位成员表示这两种工具都能提供详细的见解，但没有表现出明显的偏好。

- **LlamaIndex YouTube 资源**：用户在寻找 LlamaIndex 网络研讨会的录像，一位成员建议查看 **[LlamaIndex YouTube 频道](https://www.youtube.com/@LlamaIndex)**，以及 X space 和 LinkedIn 等其他平台以获取最新的网络研讨会。

- **AzureOpenAI 的异步调用**：一位成员提出了关于 LlamaIndex 中 **AzureOpenAI 异步调用**的问题，并收到了关于使用 `acomplete`、`astream_complete`、`achat` 和 `astream_chat` 异步方法的指导。强调了使用异步方法的好处，例如通过并行执行和非阻塞任务带来的速度提升。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://imgur.com/a/9uLmSxD">摘要与资源</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来提振你的精神...</li><li><a href="https://www.youtube.com/@LlamaIndex">LlamaIndex</a>：LlamaIndex 的官方 YouTube 频道 —— 为你的 LLM 应用提供的资料框架。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/vector_stores/TypesenseDemo#query-index>).">Typesense Vector Store - LlamaIndex</a>：未找到描述</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;我希望 Llama3 结合我的私有知识发挥 10 倍效能&quot; - 使用 llama3 构建本地 Agentic RAG</a>：高级 RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/customization#i-want-to-retrieve-more-context-when-i-query>).">常见问题解答 (FAQ) - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/zby/answerbot/blob/main/answerbot/replay_client.py">answerbot/answerbot/replay_client.py at main · zby/answerbot</a>：使用 LLMs、搜索 (RAG) 和其他工具回答问题 - 示例代码 - zby/answerbot</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/output_parsing/function_program/">用于结构化提取的 Function Calling 程序 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/retriever#get-started>).">Retriever - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/zby/LLMEasyTools">GitHub - zby/LLMEasyTools: 用于 LLM agents 的工具。</a>：用于 LLM agents 的工具。通过在 GitHub 上创建账户来为 zby/LLMEasyTools 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/openai#async>).">OpenAI - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/metaphor#llama_index.tools.metaphor.MetaphorToolSpec.retrieve_documents>):">Metaphor - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/vectara_auto_retriever#running-over-some-sample-data>).">从 Vectara 索引进行自动检索 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llamabot">GitHub - run-llama/llamabot</a>：通过在 GitHub 上创建账户来为 run-llama/llamabot 的开发做出贡献。</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/chat_engines/context#llama_index.core.chat_engine.ContextChatEngine>)">Context - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#query-pipeline-with-asyncparallel-execution>),">具有异步/并行执行能力的查询管道 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#try-out-queries>).">具有异步/并行执行能力的查询管道 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/parallel_execution_ingestion_pipeline#in-summary>),">并行化摄取管道 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1234543867987230760)** (1 条消息): 

- **回顾 GPT-1**：一位成员分享了一篇[博文](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms)，探讨了来自 OpenAI 的原始 GPT-1 模型，强调了它对 Mistral-7B 等当前 LLMs 的持久影响。该博文深入研究了 GPT-1 的架构，包括 **positional embeddings 和 Conv1D 的使用**，并展示了 Alec Radford 关于这一突破性 NLP 技术的推文截图。

**提到的链接**：<a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">重温 GPT-1：点燃 LLMs 之火的火花</a>：全面回顾 GPT-1 对现代 LLMs 发展的贡献

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1234434408405139507)** (25 条消息🔥): 

- **寻找寻求志愿者的社区项目**：一位成员询问了寻找需要志愿者的社区项目的资源，特别是那些提供计算预算的项目，因为该成员缺乏个人 GPU 资源。

- **理解 AI 中的正交键 (Orthogonal Keys)**：针对 AI 键 (keys) 和状态 (states) 背景下的“清除 (clear-ing)”过程提供了一个细致的解释，以正交键及其在方程中的表现为例，解释了模型中的内存更新。

- **Infini-Attention 与 Compressive Memory 的复杂性**：围绕 Infini-Attention 的概念及其被认为过度炒作的现象展开了对话，提到了 2021 年 Compressive Memory 中的 delta rule，并对其目前的测试结果表示怀疑。讨论中还请求并提供了一篇相关的研究论文。

- **性能对比令社区困惑**：成员们讨论了 fireworks.ai 上 *mixtral 8x22B* 性能慢于 *llama 3 70B* 的原因，涉及 Batching、利用率以及与 MoEs 相关的速度问题，并指出 *mixtral* 虽然参数更多但层数较少。

- **斯坦福 CS25 Transformers 社交活动邀请**：发布了在 EVGR Pub & Beer Garden 举行的 **Stanford CS25** Transformers 社交活动公告，提供了活动详情、RSVP 呼吁以及校园内相关讲座的信息。邀请 Discord 社区参加关于 Transformers 的线下讲座或通过 Zoom 加入，并提供了 RSVP 表单和活动详情的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://kolinko.github.io/effort/">Effort Engine</a>：一种可能用于 LLM 推理的新算法。可以平滑地——且实时地——调整推理过程中想要进行的计算量。</li><li><a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>：我们展示了线性化自注意力机制与 90 年代初的快速权重控制器（fast weight controllers）在形式上的等价性，其中“慢速”神经网络通过梯度下降学习来编写“快速”网络...</li><li><a href="https://cs25.stanford.edu)">未找到标题</a>：未找到描述</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09).">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于在移动设备、桌面设备和会议室系统上进行视频和音频会议、聊天和网络研讨会。Zoom ...</li><li><a href="https://discord.gg/2vE7gbsjzA)">Discord | 你的聊天和聚会场所</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。与你的朋友和社区交谈、聊天、聚会并保持联系。</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1cef7gc/tradingview_premium_pack_crack_2024/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1234512372647989329)** (105 条消息🔥🔥): 

- **解决长上下文挑战**：[信息密集型 (IN2) 训练提案](https://arxiv.org/abs/2404.16811) 旨在提高 Large Language Model (LLM) 对长上下文的使用能力。它涉及一个合成数据集，要求模型整合长文本中不同片段的信息，以克服“lost-in-the-middle”问题。
  
- **涌现能力与预训练损失相关**：一篇 [Twitter 帖子](https://x.com/_jasonwei/status/1784990066609414556?s=46&t=OICM4zGqs0OOATmLPoNFyw) 讨论了模型中的涌现能力（emergent abilities）可能与预训练损失（pretraining loss）相关的发现。与算力相比，预训练损失通过考虑数据集质量和架构因素，能更好地反映模型性能。

- **剖析模型偏见**：讨论强调了将特定偏见（如数字偏好）追溯到模型权重变化的难度。由于偏见可能在持续训练过程中产生，成员们指出可能需要实施工具来分析这些变化以进行验证。

- **关于 LLM 作为黑盒的辩论**：对话围绕 LLM 是否应被视为黑盒展开，因为我们对其内部机制的理解有限。有人认为，虽然我们了解 LLM 的某些方面，但其推理不可信，因为解释是事后的（post-hoc），可能无法反映真实的内部过程。

- **LLM 中的数据泄漏检测**：一条消息链接到了一篇论文，该论文介绍了一种检测流水线，用于识别 LLM 基准测试中潜在的数据泄漏，强调了训练集和测试集误用的问题 ([PDF](https://arxiv.org/pdf/2404.18824))。研究结果旨在促进 AI 领域的公平比较和健康发展。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://videogigagan.github.io/">VideoGigaGAN</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.14662">NExT: Teaching Large Language Models to Reason about Code Execution</a>: 人类开发者的一个基本技能是理解和推理程序执行的能力。例如，程序员可以在脑海中用自然语言模拟代码执行来调试...</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>: 虽然许多现代 Large Language Models (LLMs) 可以处理长输入，但它们仍然难以充分利用长上下文中的信息，这被称为 lost-in-the-middle 挑战。我们...</li><li><a href="https://x.com/_jasonwei/status/1784990066609414556?s=46&t=OICM4zGqs0OOATmLPoNFyw">Jason Wei (@_jasonwei) 的推文</a>: 很喜欢这篇将 emergent abilities 与 x 轴上的 pretraining loss 关联起来的论文，这实际上也是 @OriolVinyalsML 几年前提出的建议：https://arxiv.org/abs/2403.15796 ...</li><li><a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: 随着预训练数据使用的扩大，基准测试数据集泄露现象日益突出，而不透明的训练过程以及经常未披露的包含情况加剧了这一问题...</li><li><a href="https://arxiv.org/abs/2403.18506">Faster Convergence for Transformer Fine-tuning with Line Search Methods</a>: 最近的研究表明，线搜索方法在各种数据集和架构上大大提高了传统随机梯度下降方法的性能 [1], [2]。在这项工作中，我们建议...</li><li><a href="https://arxiv.org/abs/2404.12388">VideoGigaGAN: Towards Detail-rich Video Super-Resolution</a>: 视频超分辨率 (VSR) 方法在升采样视频中表现出令人印象深刻的时间一致性。然而，这些方法往往比图像领域的同类方法产生更模糊的结果，因为...</li><li><a href="https://arxiv.org/abs/2404.16717">Embracing Diversity: Interpretable Zero-shot classification beyond one vector per class</a>: 视觉语言模型实现了无需任何重新训练的开放世界物体分类。虽然这种 Zero-shot 范式标志着重大进步，但即使是当今最好的模型也表现出...</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1">Sequential predictive learning is a unifying theory for hippocampal representation and replay</a>: 哺乳动物的海马体包含一个认知地图，代表动物在环境中的位置，并生成离线 "replay" 用于回忆、规划和形成长...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1234570912951697500)** (3 messages): 

- **针对不同 Prompt 的自定义函数**：一位成员讨论了在单个任务中根据模型传递不同 Prompt 的可能性，建议使用自定义 `!function` 来实现。
- **BitsAndBytes 8bit 的异常**：一位用户观察到，使用 **BitsAndBytes 4bit** 编码在 **llama3-70b** 上运行良好，但切换到 **8bit** 编码后效果很差，将输出描述为“完全是垃圾”。
- **llama3-8b 的 8bit 编码问题**：同一位成员注意到在 **llama3-8b** 上使用 **8bit** 编码时也存在类似问题，表明 8bit 在不同模型中存在一致性问题。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1234495993056329738)** (113 messages🔥🔥):

- **AI Birthday Bungle Sparks GDPR War**: 一名欧盟隐私活动人士在 AI 模型错误猜测其生日后，针对 AI 模型提交了 [GDPR 投诉](https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/)。他认为这一错误可能导致 AI 模型在欧盟被禁。
- **New GPT Surprise Rumors Circulate**: 关于 GPT-5 模型秘密发布的传闻正在流传，推测基于其性能表现以及在测试中拒绝产生 Hallucination 的行为，但由于没有正式进入 Leaderboard 且测试响应存在矛盾，目前仍存在诸多困惑。
- **Performance Queries for Llama3 70B**: 针对 [Llama3 70B 模型](https://rentry.co/GPT2) 在双 3090 配置下仅 13 tokens per second 的低生成速度提出了质疑，引发了关于潜在硬件优化和模型配置调整的讨论。
- **Exllama: The Underrated Speedster**: 用户讨论了 Exllama 在 LLM 任务中优于其他库的性能表现，建议使用 [TabbyAPI](https://dct.openempathic.ai/) 仓库以简化安装。
- **Debates Over LMSYS’s Leaderboard Transparency**: 成员们对 LMSYS Leaderboard 的客观性表示怀疑，对科学评估与商业企业之间潜在的利益冲突表示担忧，并呼吁提高透明度以及增加按 Open Weights 过滤的功能。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-03-01-policy/">LMSYS Chatbot Arena: Live and Community-Driven LLM Evaluation | LMSYS Org</a>: &lt;h2&gt;&lt;a id=&quot;our-mission&quot; class=&quot;anchor&quot; href=&quot;#our-mission&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&...</li><li><a href="https://www.politico.eu/article/chatgpts-hallucinations-get-eu-privacy-complaint/">ChatGPT&#8217;s hallucinations draw EU privacy complaint</a>: 活动人士要求监管机构针对 ChatGPT 对其出生日期的胡乱猜测展开调查。</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1234583301562437682)** (12 messages🔥): 

- **OpenCLIP Fine-Tuned for Cardiac Ultrasound**: 一位成员分享了他们关于针对心脏超声微调 OpenCLIP 的研究成果，[点击此处查看](https://doi.org/10.1038/s41591-024-02959-y)。尽管面临诸多挑战和广泛的修订过程，他们对项目的完成感到欣慰。
- **Echoes of Exhaustion**: 该成员还表达了他们想要结束这个高强度项目的意愿，幽默地提到了项目中使用的“粗糙” Zero-shot 技术，以及在项目开始时对 Multimodal AI 世界的陌生。
- **Stable Diffusion Community Reopens**: 分享了一个用于独立于 U-Net 训练 CLIP 的 GitHub 仓库链接，同时传来了 /r/StableDiffusion 在抗议 Reddit 开放 API 变更后重新开放的消息。更多详情和讨论论坛可见于 [此 Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://doi.org/10.1038/s41591-024-02959-y">Vision–language foundation model for echocardiogram interpretation - Nature Medicine</a>: 一个视觉-语言基础模型，在超过 100 万个超声心动图视频-文本对的数据集上进行训练，能够评估各种心脏结构和功能参数...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1234551748413358170)** (2 messages): 

- **ChatGPT Plus Integrates Memory Feature**: **Memory** 功能现已面向所有 ChatGPT Plus 用户开放，允许用户通过开启新对话来告诉 ChatGPT 需要记住的内容。该功能可在设置中开启或关闭，目前尚未在欧洲或韩国推出。

- **Enhanced Data Control for Users**: ChatGPT Free 和 Plus 用户现在即使选择了不贡献数据用于模型改进，也可以访问其聊天记录。此外，新的 **Temporary Chat** 功能允许进行不会保存在用户聊天记录中的对话。
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1234440550707630160)** (81 messages🔥🔥):

- **探索 AI 的好奇心与感知力**：一位用户详细介绍了他们进行的好奇心测试，涉及 ChatGPT 处理一个包含迷宫的 zip 文件。随后进行了一些关于如何衡量 AI 好奇心潜力及其与感知力关系的讨论，但关于这些概念的共识仍然难以达成。
- **DragGAN 引发关注**：一位成员发现了 DragGAN，这是一个可以操纵照片以改变角度和姿势的工具，引发了关于 AI 在没有完整模型的情况下从新视角重建图像能力的讨论。
- **Llama-3 8B 扩展上下文能力**：Llama-3 8B Instruct Gradient-1048k 的发布引人注目，展示了最先进的语言模型如何处理长上下文信息；该模型可在 [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k) 获取。
- **关于先进 AI 工具可访问性的辩论**：关于 OpenAI 免费开放 DALL-E 等新功能政策的讨论浮出水面，一些用户质疑为什么更先进的工具不也免费，并思考 OpenAI 提供学生折扣的可能性。
- **LLM 之间潜在的协作**：一位用户询问了让 ChatGPT 和 Claude Opus 两个语言模型协作撰写论文的可能性，引发了关于使用第三方服务来管理多模型交互的建议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>：未找到描述</li><li><a href="https://dontasktoask.com/">别问能不能问，直接问</a>：未找到描述</li><li><a href="https://vcai.mpi-inf.mpg.de/projects/DragGAN/">Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1234492334725533807)** (11 条消息🔥): 

- **模型性能中规模至关重要**：强调了 **GPT-4** 与其前代模型的对比，*GPT-4* 被认为 **“比 3.5 大得多”**。

- **GPT-4 的速度预期受到挑战**：一位成员质疑了对 **GPT-4** 速度会更快的预期，考虑到它比之前的模型规模更大。

- **寻求 AI 安全项目协助**：一位名为 **abhibetter** 的成员就 AI 在安全项目中的应用寻求帮助，但未提供有关具体问题或疑问的细节。

- **探索 GPT-2 性能**：成员 **namenot223_69478** 询问是否有人在 **chatlmsys** 上实验过 **GPT-2**，另一位成员引导其前往另一个频道进行深入讨论。

- **处理批量删除聊天存档**：**silensu** 正在寻求关于如何处理意外存档大量聊天的建议，并询问 *批量删除* 的可能性。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1234436383544971264)** (15 条消息🔥): 

- **提议举办百万美元 Prompt 竞赛**：一位成员建议组织设有巨额现金奖励的 **Prompt Engineering 竞赛**，以激发社区内的学习和最佳实践分享。他们设想了付费和免费的“游乐场”竞赛，创建一个奖励积极协作和 Prompt 创作实际成果的游戏化环境。

- **Meta Prompting 为其铺平道路**：在关于改进 Prompt 创作的讨论中，有人指出 **GPT Builder** 所采用的 “Meta Prompting” 是一种有效的方法，即 AI 根据用户指令调整上下文和对话以优化结果。

- **AI 中 Negative Prompting 的挑战**：用户讨论了在指导 AI 时 Negative Prompting（负面提示）的低效性，解释说与正面示例和指令相比，强调 **禁用词** 可能会导致输出不一致且效果较差。

- **处理 AI 任务中的本地化语言**：一位用户努力尝试让 AI 生成的文本适应地区语言变体，特别是阿根廷西班牙语，其中某些词汇具有不同的含义。讨论了重构项目框架和为地区词汇提供特定替换等选项，以便在存在大量禁用词列表的情况下更好地定制输出。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1234436383544971264)** (15 条消息🔥): 

- **通过竞赛进行 Prompt Engineering**：一位成员提议举办 Prompt 竞赛以提高 Prompt Engineering 技能。竞赛范围将涵盖从 *No-code* 挑战（AI 处理数据以提取信息）到交互式任务（如导航基于文本的游戏），并包括社区讨论和知识共享。

- **Meta-Prompting 胜过竞赛**：一位参与者建议使用 *meta-prompting*，这是一种由 AI 协助编写更好 Prompt 的方法，有可能取代对竞赛的需求。这表明用户正趋向于通过 **GPT Builder** 来简化 Prompt 编写过程。

- **GPT Builder 和 Meta Prompting 的实际应用**：讨论强调 **GPT Builder** 基于 *meta prompting* 运行，AI 根据用户请求调整上下文和对话，这暗示了存在用于优化 Prompt 策略的文档。

- **正向 Prompt 优于负向**：在解决不希望出现的语言生成问题时，建议在 Prompt 中使用*正向指令和示例*，而不是指定禁用词。建议包括创建强化首选术语的 Prompt，并解释在特定方言中的用法。

- **应对多语言细微差别**：面对多语言挑战，一位用户表达了在为西班牙语变体构建 Prompt 时的困难，因为词汇在不同地区可能有不同的含义。优化 AI 语言输出的策略包括重新描述项目，或明确将禁用词与其理想的替代词配对。
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1234518684274262038)** (25 messages🔥): 

- **LLaMA 3 对量化敏感**：讨论强调 **LLaMA 3** 比 LLaMA 2 经历更多的[量化降级（degradation from quantization）](https://x.com/rohanpaul_ai/status/1784972618472317180)，这可能是由于它在创纪录的 15T token 上进行训练，使其能够捕捉到极其细微的数据关系。
- **LLaMA 3 Tokenization 问题**：提到了 **llama-3** 不生成句子开头（BOS）token 的问题，但通过手动在 chat template 中添加 BOS 得到了解决。
- **量化敏感性研究评论**：社区讨论了一项关于**量化敏感性**的研究，认为这与模型训练方法有关，而不仅仅是模型大小，一位成员形容相关的 [arXiv 论文](https://arxiv.org/abs/2311.16452)“毫无价值”。
- **Llama-3 扩展上下文长度**：提到了 **Llama-3 8B Gradient Instruct 1048k 模型**，该模型显著扩展了模型的上下文长度，由 Gradient 开发并由 Crusoe Energy 提供算力赞助，详情见 [huggingface.co](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k)。
- **BOS 需要调整模板**：在遇到 LLaMA-3 模型的 BOS token 生成问题时，注意到仅更改 tokenizer 是不够的，需要在 chat template 中包含 BOS 才能显示。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.16452">Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine</a>: 诸如 GPT-4 之类的通用基础模型在广泛的领域和任务中展示了惊人的能力。然而，普遍存在一种假设，即它们无法与专业微调相匹配...</li><li><a href="https://x.com/rohanpaul_ai/status/1784972618472317180">Rohan Paul (@rohanpaul_ai) 的推文</a>: 量化对 LLaMA 3 的伤害比对 LLaMA 2 更大。llama cpp 仓库中的这个 PR 对此进行了深入调查。（Perplexity 衡量模型预测下一个 token 的能力，数值越低越好...）</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1234556675000897687)** (7 messages): 

- **探索 Huggingface 的 ZeroGPU**：一位成员提到他们已获得 [Huggingface Zero 项目](https://huggingface.co/zero-gpu-explorers)的访问权限，邀请大家建议在这个新平台上进行哪些测试。
- **ZeroGPU 提供免费多 GPU 访问**：他们分享了关于 **ZeroGPU** 的信息，这是 **Huggingface** 上的一个测试功能，提供**免费 GPU 访问**以及在**多 GPU**（使用 _Nvidia A100_）上运行 Spaces 的能力。ZeroGPU 通过根据需要高效分配和释放资源来优化 GPU 利用率。
- **错失良机**：几位成员对没有早点报名 ZeroGPU 项目以利用 **[PRO 订阅者](https://huggingface.co/subscribe/pro)的早期访问权限**表示遗憾。

**提及的链接**: <a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: 未找到描述

  

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1234752355518386236)** (11 messages🔥): 

- **Llama-3-70B 微调问题**：一位成员建议，对 `meta-llama/Meta-Llama-3-70B-Instruct` 进行微调可能会降低其性能，因为它已经经过了指令微调。建议在尝试复杂的 70B 模型之前，先从 8B 模型开始。

- **数据集格式转换指南**：成员们建议了一种将微调数据集从 OpenAI 格式转换为 ShareGPT 格式的简单方法：将 "messages" 替换为 "conversations"，"role" 替换为 "from"，"content" 替换为 "value"，"user" 替换为 "human"，"assistant" 替换为 "gpt"。

- **推荐的微调学习路径**：一位资深的社区成员建议初学者在尝试微调像 70B 这样的大型模型之前，先微调较小的模型（如 8B）。

- **轻松实现数据集转换**：提供了 Python 代码，通过使用角色映射字典和列表推导式，方便地将给定格式的数据转换为 ShareGPT 所需的格式。

**提到的链接**：<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt.load_role)">Axolotl - Conversation</a>：未找到描述。

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

gbourdin: 添加到我的书签。感谢分享！
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1234879220686258296)** (2 messages): 

- **Axolotl 微调变得更简单**：一位成员分享了一个[教程](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md)，指导用户如何使用 `dstack` 微调 `axolotl`。`dstack` 是一个开源编排器，可与任何云端或本地机器池配合使用。该教程由一位 `axolotl` 用户贡献。
- **社区认可**：另一位成员对该教程表示赞赏，提到它看起来非常易于遵循。

**提到的链接**：<a href="https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md">dstack/examples/fine-tuning/axolotl/README.md at master · dstackai/dstack</a>：一个用于在任何云或数据中心运行 AI 工作负载的开源容器编排引擎。https://discord.gg/u8SmfwPpMd - dstackai/dstack

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1234456215904587827)** (10 messages🔥): 

- **LoRA 与 QLoRA 的区别说明**：**LoRA** 和 **QLoRA** 的主要区别在于，LoRA 专注于通过低秩矩阵进行模型适配，而 QLoRA 将其与量化（quantization）结合，以进一步优化部署。*LoRA 高效地适配预训练模型；QLoRA 则更进一步，适用于资源受限的环境。*

- **将 Axolotl 数据集修剪至特定百分比**：在 Axolotl 配置中将数据集修剪到特定百分比并不是内置功能，需要进行预处理或修改数据集加载脚本。可以在数据集加载期间通过子采样逻辑修改 `DPODataset` 的使用。

- **GPU 数量与 Micro Batch Size 的等效性**：有人询问使用 **4x GPU & Micro Batch Size 4** 是否在最终输出上等同于 **8x GPU & Micro Batch Size 2**。频道讨论中未给出明确答案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c42603f2-ce0e-4806-aa15-b77ac3002f7d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=650c6038-10b5-46b9-aacc-ce5f8e81ff17)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快地理解代码。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1234798037612625921)** (39 messages🔥): 

- **Command-R 模型微调**：成员们讨论了在 Axolotl 中微调 *command-r* 模型。一位用户分享了一个[未经测试的 Pull Request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547)，用于将 *command-r* 添加到 Axolotl，但指出该请求尚未经过测试，目前不建议合并。

- **Command-R 的格式适配**：当询问如何使用 *command-r* 的指令格式（instruct format）时，建议使用 `input_output` 格式并预先准备好正确的 Token。关于实现不常用格式的更全面指南，请参阅 [input_output 文档](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html)。

- **Sample Packing 功能的不确定性**：关于 *sample packing* 功能的实现存在困惑，该功能旨在为 Axolotl 将小样本打包成大样本。虽然一些用户需要此功能，但它似乎需要在一个未经测试的 Pull Request 中概述的修改。

- **对 runpod Template 缺乏经验**：一位用户表示由于不熟悉 *runpod template*，在集成补丁更改时感到不确定。讨论中没有提供明确的解决方案。

- **对 phi-3 格式的支持尚不明确**：一位用户询问 Axolotl 是否支持 phi-3 格式，但根据当前文档，Bot 的回复暗示不支持 phi-3。文档列出了包括 phi 在内的各种模型对不同功能的兼容性，但未特别提到 phi-3。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">Feat: 由 NanoCode012 添加 cohere (commandr) · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>：描述、动机与背景、如何测试？未经测试！截图（如果适用）、更改类型、社交账号（可选）</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=83b91c9b-bb5c-4485-894c-0b878d17f7e2)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L77L100)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>：尽管提出 Axolotl 相关问题。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=1f87fb72-80ec-4321-b37b-d7574206e8af)">OpenAccess-AI-Collective/axolotl | Phorm AI 代码搜索</a>：更快速地理解代码。
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1234538847430246513)** (80 条消息🔥🔥): 

- **探索自主 Agent 的记忆**：讨论涉及一个名为 [Memary](https://github.com/kingjulio8238/memary) 的 GitHub 项目，该项目旨在为自主 Agent 提供长期记忆。对话澄清了虽然可能使用知识图谱，但 Memary 主要通过对文档进行相似性搜索来运行。
  
- **关于神秘 GPT-2 Chatbot 的辩论**：围绕 lmsys 上具有 GPT-4 级别能力的令人费解的 [GPT2-chatbot](https://chat.lmsys.org/) 展开了讨论。尽管有各种分析和推测，该模型的真实来源或性质仍不清楚，一种可能性是 OpenAI 原始 GPT-2 的微调版本。

- **开源 AI 直面大厂**：来自 [Prime Intellect](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training) 的一篇博客文章强调了开源 AI 开发在与使用大型互连 GPU 集群的闭源对手竞争时面临的挑战。文章详细阐述了去中心化训练作为开源进步的潜在解决方案。

- **关于 Agent 和 LLM 角色的讨论**：针对自主 Agent 与大语言模型 (LLM) 的混淆进行了深入讨论。对话展示了框架向使用“模块”构建并发共享上下文/记忆以进行推理和规划的转变，而不是期望 LLM 作为独立的自主单元运行。

- **学习 AI 基础与技能**：一位用户询问了从零开始学习 AI 的方法，寻求在不致力于特定领域的情况下理解基本概念。其他成员提供了资源，包括关于神经网络的 YouTube 教程、AI 工程入门课程以及 Prompt Engineering 指南。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/AlexReibman/status/1784844434682560721">来自 Alex Reibman 🖇️ (@AlexReibman) 的推文</a>：OSWorld：在真实计算机环境中针对开放式任务的多模态 Agent 基准测试。自从 OpenInterpreter 出现以来，我们一直在思考，如果你给 Agent 一个...，它们能有多高效。</li><li><a href="https://www.latent.space/p/aie-2023-workshops">AI Engineering 101 和 201 工作坊</a>：来自 2023 年 AI Engineer 峰会</li><li><a href="https://x.com/lmsysorg/status/1785078213712208291?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">来自 lmsys.org (@lmsysorg) 的推文</a>：嗨 @simonw，非常感谢！我们非常看重您的反馈。澄清一下，根据我们的政策，我们已与多家模型开发商合作，将他们的新模型引入我们的平台进行社区...</li><li><a href="https://learnprompting.org/docs/intro">Learn Prompting：您的 AI 沟通指南</a>：Learn Prompting 是互联网上规模最大、最全面的 Prompt Engineering 课程，拥有超过 60 个内容模块，被翻译成 9 种语言，并拥有一个繁荣的社区。</li><li><a href="https://rentry.co/GPT2">GPT-2?</a>：背景：https://chat.lmsys.org 为 LLM（以及一些 MLLM）提供盲测用户基准。最近可用的模型之一是 GPT2-chatbot，它展示了远超...的能力。</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">去中心化训练（Decentralized Training）的最新技术</a>：这篇文章探讨了各种新颖的去中心化训练方法，以及它们如何实现在全球分布的 GPU 上进行有效的 AI 模型训练。</li><li><a href="https://roadmap.sh/prompt-engineering">Prompt Engineering 路线图 - roadmap.sh</a>：学习 Prompt Engineering 的分步指南。我们还在路线图项目中附带了资源和简短说明，让您可以在一个地方获取想要学习的所有内容。</li><li><a href="https://x.com/karan4d/status/1785000251096437161?s=46&t=">来自 mephistoooOOHHHHHHSHI- (@karan4d) 的推文</a>：好的，它肯定使用了 GPT-4 的 Tokenizer，所以我敢打赌它也是 4.5。始终使用异常 Token 进行指纹识别。</li><li><a href="https://x.com/lmsysorg/status/1785078213712208291">来自 lmsys.org (@lmsysorg) 的推文</a>：嗨 @simonw，非常感谢！我们非常看重您的反馈。澄清一下，根据我们的政策，我们已与多家模型开发商合作，将他们的新模型引入我们的平台进行社区...</li><li><a href="https://x.com/albfresco/status/1784964830887104999?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 albs — 3/staccs (@albfresco) 的推文</a>：我的猜测是，这个神秘的 'gpt2-chatbot' 实际上是 OpenAI 2019 年的 GPT-2，并使用现代助手数据集进行了微调。如果是这样，那意味着他们最初的预训练仍然...</li><li><a href="https://x.com/karan4d/status/1785000251096437161?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 mephistoooOOHHHHHHSHI- (@karan4d) 的推文</a>：好的，它肯定使用了 GPT-4 的 Tokenizer，所以我敢打赌它也是 4.5。始终使用异常 Token 进行指纹识别。</li><li><a href="https://x.com/markatgradient/status/1785032103429865748?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mark Huang (@markatgradient) 的推文</a>：1M 上下文长度的 Llama-3 8B 模型。无需多言。已在 HF 上线 @ClementDelangue 抄送：@winglian @mattshumer_ ↘️ 引用 Gradient (@Gradient_AI_)：我们一直在努力研发 🔥 很高兴...</li><li><a href="https://x.com/MKBHD/status/1785102259740667960">来自 Marques Brownlee (@MKBHD) 的推文</a>：新视频 - Rabbit R1：几乎无法评价 https://youtu.be/ddTV12hErTc 这是多年来一个令人恼火的趋势的顶峰：交付几乎未完成的产品以赢得一场“竞赛”...</li><li><a href="https://github.com/xlang-ai/OSWorld">GitHub - xlang-ai/OSWorld：OSWorld：在真实计算机环境中针对开放式任务的多模态 Agent 基准测试</a>：OSWorld：在真实计算机环境中针对开放式任务的多模态 Agent 基准测试 - xlang-ai/OSWorld</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary：自主 Agent 的长期记忆。</a>：自主 Agent 的长期记忆。通过在 GitHub 上创建账号来为 kingjulio8238/memary 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=1hDK7gZbJqQ&t=25s">第 8 集 — ColBERT + ColBERTv2：以合理的推理成本实现后期交互（late interaction）</a>：Andrew Yates（阿姆斯特丹大学助理教授）和 Sergi Castella（Zeta Alpha 分析师）讨论了两篇引入 Co... 的具有影响力的论文。</li><li><a href="https://youtu.be/aircAruvnKk?feature=shared),">神经网络到底是什么？| 第一章，深度学习</a>：什么是神经元，为什么有层，背后的数学原理是什么？帮助资助未来的项目：https://www.patreon.com/3blue1brown 编写/交互...</li><li><a href="https://x.com/jessechenglyu/status/1785342519045394465?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Jesse Cheng Lyu 的推文</a></li>

eet from Jesse Lyu (@jessechenglyu)</a>: 立即将你的 r1 更新到最新版本 - 我们解决了目前发现的大多数问题，更多修复/改进即将到来！待机电池寿命现在提升了高达 5 倍。 ↘️ 引用 rabbit inc. (@rabb...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1234527192797417594)** (21 messages🔥): 

- **关于使用本地视觉模型启动 OS 模式的问题**：一位成员询问**如何使用本地视觉模型启动 OS 模式**以尝试 **Moondream**，但反馈使用命令 `interpreter --os --local` 时会出现乱码。
- **关于模型功能的讨论**：另一位用户提到在**几个月前**使用过 `llava`，并确认可以**通过 OpenInterpreter 获取图像描述而无需执行自定义代码**。
- **OpenInterpreter 的集成更新**：一位成员宣布他们成功将所有 OpenInterpreter 输出集成到 **MagicLLight** 中，并计划为 `stream_out` 函数钩子和 `external_input` 向 OpenInterpreter 提交 Pull Request。MagicLLight 和 AAA+ 的代码预计在清理后发布。
- **在廉价硬件上运行 OpenInterpreter**：有人质疑在 **BeepyBerry-Raspberry Pi Zero** 上流畅运行 **OpenInterpreter** 的可行性，并附带了相关的 [YouTube 视频](https://youtube.com/shorts/E7WQZdJKsbM?si=1XMj0aTtN83cZ5aY) 链接。
- **寻求启动失败的调试协助**：一位用户就**调试启动失败**寻求帮助，表示错误信息很模糊。他们被引导分享错误信息，以便社区协助排查。

**相关链接**: <a href="https://discord.gg/SdwpMQaW?event=1232436050165764096">Discord - 与朋友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1234538284089344000)** (20 messages🔥): 

- **按钮代码成功运行**：一位成员通过更新 `ButtonChecker` 代码并将按钮连接到**引脚 25**，成功解决了外部按钮无反应的问题，并提供了[修改后的代码片段](https://discord.com/channels/openinterpreter/01)。另一位社区成员确认其工作正常。
- **扬声器连接稳定性**：在另一个硬件相关的修复中，建议使用热熔胶固定扬声器导线，以减少连接引脚时的压力。
- **提高扬声器音量的咨询**：有人提出了如何增加扬声器音量的问题，建议尝试 **M5Unified** 或可能使用[外部放大器](https://www.amazon.com/dp/B01DKAI51M)。
- **关于 YouTuber 评论的辩论**：讨论了 YouTuber 对 **AI pins** 和 **R1** 等 AI 产品的评论相关性，质疑像 **MKBHD** 和 **Dave2d** 这样的科技评论者是否完全理解 AI 领域，因为这与评论手机或笔记本电脑等消费电子产品不同。
- **配合 OS 模式的 01 Light 硬件**：一位成员寻求在当前版本的 01 Light 硬件上运行 OS 模式的帮助，提到已成功连接到 Mac，但无法访问屏幕。
<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://www.amazon.com/dp/B01DKAI51M">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee">Rabbit R1: 几乎无法评价</a>: 盒子里的 AI。但是个不同的盒子。在 https://dbrand.com/rabbit 获取 dbrand 皮肤和屏幕保护贴。MKBHD 周边: http://shop.MKBHD.com 我现在使用的科技产品...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1234533862407671860)** (10 messages🔥):

- **Tinygrad 咨询**：一位用户询问什么是 **tinygrad**，另一位成员提供了 [tinygrad GitHub 仓库](https://github.com/tinygrad/tinygrad/tree/master)的链接，并将其定义为一个喜欢 PyTorch 和 micrograd 的人会爱上的项目。
- **Discord 发现之谜**：一位成员对另一位成员是如何偶然发现这个 Discord 服务器表示好奇，后者给出了不确定的回答，表示不清楚自己是如何发现的。
- **寻求 Bounty 指导**：一位用户就涉及 *"Mean of symbolic shape"* 和 *"Symbolic arrange"* 的两个 Bounty 寻求帮助，并正在寻找相关参考资料以理解并解决它们。
- **反向传播优化问题**：一位成员正在调查与包含 2 个 reduce 操作的反向传播相关的 issue [#3572](https://github.com/tinygrad/tinygrad/issues/3572)，并询问如何生成图表（graph diagrams）来阐明该问题。
- **Tinygrad 的图表生成**：针对有关生成图表以解决反向传播问题的查询，一位成员提到使用 `GRAPH=1`，建议通过环境变量来辅助完成此任务。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/4362">geohot 提交的 tensor variable · Pull Request #4362 · tinygrad/tinygrad</a>：未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/tree/master">GitHub - tinygrad/tinygrad: 喜欢 pytorch？喜欢 micrograd？你会爱上 tinygrad！❤️</a>：喜欢 pytorch？喜欢 micrograd？你会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1234463379603722291)** (29 条消息🔥): 

- **探索 TinyGrad 学习资源**：成员们讨论了学习使用 TinyGrad 进行 AI 开发的资源；分享了 [MicroGrad GitHub 仓库](https://github.com/unknownusername504/MicroGrad)和 [MiniTorch](https://minitorch.github.io/) 的链接，其中 MiniTorch 被强调为理解深度学习系统的教学工具。
- **分享 TinyGrad 快速入门指南**：一位用户向任何想要学习 AI（特别是使用 TinyGrad）的人推荐了 "[tinygrad 快速入门指南](https://tinygrad.github.io/tinygrad/quickstart/)"，因为它提供了 TinyGrad 为模型开发提供的高层 API 的基本概览。
- **TinyGrad 中的 Symbolic Mean Bounty 挑战**：讨论围绕在 TinyGrad 中实现符号平均值（symbolic mean）操作展开，并考虑了 LazyBuffer 处理 Variable 类型数据的需求以及是否应该分配内存。
- **TinyGrad 符号执行的 Pull Request**：分享了一个[之前的 pull request](https://github.com/tinygrad/tinygrad/pull/1552) 链接，以说明 TinyGrad 中符号代码生成和执行的机制，暗示了变量缓存如何对 `sum` 和 `mean` 等操作有用。
- **开发带有变量的 Symbolic Mean**：对话继续围绕符号平均值的开发展开，重点是需要以符号方式表示 Tensor 长度，以及 `Const` 支持输入缓冲区中变量的潜力。作为解决此挑战的一部分，分享了 GitHub 上 master 分支与功能分支的对比链接 [tinygrad symbolic-mean-var-pull](https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull)，以及进一步的 [gh 提交的 GitHub 更改](https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tinygrad.github.io/tinygrad/quickstart/">Quickstart - tinygrad docs</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull">Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！ ❤️ - Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840">Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！ ❤️ - Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</li><li><a href="https://github.com/unknownusername504/MicroGrad">GitHub - unknownusername504/MicroGrad</a>: 通过在 GitHub 上创建账号，为 unknownusername504/MicroGrad 的开发做出贡献。</li><li><a href="https://minitorch.github.io/">MiniTorch</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5">rename Scalar to ConstType and cast_scalar to as_const (#3946) · tinygrad/tinygrad@77589bc</a>: 前置清理工作，使 const 参数与 dtype 具有相同的 python 类型</li><li><a href="https://github.com/tinygrad/tinygrad/pull/1552">symbolic codegen and exec by chenyuxyz · Pull Request #1552 · tinygrad/tinygrad</a>: #1353 的一部分，通过 codegen 和 exec 为符号化输入实现 realize。合并后的 var_vals 直接传入 kernel 函数。我已为 CLANG, GPU, METAL 实现了后端。glob...
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1234452605997023242)** (34 条消息🔥): 

- **Command-R 中的单 URL 限制**: 在关于 **API Command R+** 中 **web-search 工具**的讨论中，成员们澄清目前该工具的 `site` 选项只能使用一个网站，并建议变通方法是**为每个网站分别运行一次 API 调用**。
- **缺少多步 Connectors**: **Cohere** 确认目前在 **Command-R** 的多步工具使用中无法使用 **Connectors**。
- **对未来 Command-R 功能的期望**: 一位成员建议了针对 **Command-R** 的增强功能，重点是 **Connectors**，例如在 `web_search` 中使用多个网站，向自定义 Connectors 发送额外参数以实现更精细的控制，以及启用 `use_rerank` 选项进行自动重排序。文中分享了一个有用的文档链接：[Cohere Chat Documentation](https://docs.cohere.com/reference/chat)。
- **关于模型可用性的疑问**: 有人询问微调模型的 "Generate" 选项是否可用，因为发现它在仪表板中缺失，引发了关于它是否会回归的猜测。
- **高效 Embedding 的策略**: 一位成员询问了关于**保持数据更新**以进行高效 Embedding 的策略，涉及如何以低成本方式仅对已更新的数据块进行重新索引。

**提到的链接**: <a href="https://docs.cohere.com/reference/chat">Chat API Reference - Cohere Docs</a>: 未找到描述

  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1234628219492241449)** (2 条消息): 

- **瑞典的问候**: 一位来自瑞典斯德哥尔摩的成员提到在他们的公司中**使用 Cohere**。
- **北欧协作**: 另一位成员强调了他们通过公司 Omegapoint 与**挪威和瑞典**的联系。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1234429137763176458)** (12 条消息🔥): 

- **Gemini 模型探索**: 一位成员正在寻找具有 **Gemini 1.0 或 1.5 模型**经验的人，以便通过私信讨论具体细节。
- **寻求 LLM 可观测性工具**: 有人请求推荐大语言模型（LLM）可观测性工具。该成员正在考虑 **Arize Phoenix** 或 **Langfuze**，并倾向于选择与 **LlamaIndex** 兼容的自托管开源选项。
- **OpenAI 与 SQL 安全**: 一位成员询问如何在**不使用 LangChain 的情况下将 OpenAI 直接连接到 SQL 服务器**，并在此过程中优先考虑安全性。
- **结合 LangGraph 与 autoawq**: 讨论了将 **autoawq** 与 **LangGraph** 集成，并配合 **exllamav2 kernels** 使用，以实现驱动 AI Agent 的高推理速度。
- **PDF 内容提取挑战**: 一位刚接触 LangChain 和 AI 编程的新成员正在寻求建议，以改进拆分跨越 PDF 多页的单个表格时的效果，并提到使用 **unstructure** 进行 AI 驱动的 PDF 内容提取时效果不尽如人意。
  

---

**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1234549931969216563)** (2 条消息): 

- **AzureSearchVectorStoreRetriever 异步问题**：一位用户提到由于 **AzureSearchVectorStoreRetriever** 不支持异步操作而遇到错误，并询问可能的解决方案。讨论的选项包括请求 langserve 实现该功能，或者为同步检索函数创建一个异步包装器 (async wrapper)。

- **使用 Google Drive 库**：另一位用户建议在某个函数中使用 Google Drive 库，并提到需要将 drive key 设置为环境变量。有人指出这些库在过去曾被移除后又重新添加。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1234542917406822560)** (8 条消息🔥): 

- **与 GPT-1 一起重温往昔**：一位博主重新审视了 **原始 GPT-1 模型**，深入探讨了它如何为当前的 LLM 奠定基础，并指出了它与 **Mistral-7B** 等模型的相似之处。该博客讨论了 Transformer block 中的位置嵌入 (positional embeddings) 和 Conv1D，详见 [Revisiting GPT-1: The Spark That Ignited LLMs](https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms)。

- **在 Airbnb 场景中展示 LangChain**：一段名为 **"D-ID Airbnb Use Case: A RAG Agent Demo using Ollama and Langchain with code on Github"** 的演示视频展示了一个创新的房产网站 **实时虚拟人问答 (Live Avatar Q&A)**，该功能由 LangChain 驱动，包含 150 个问答对。在 [YouTube](https://youtu.be/N_GcPLJCQQY) 上查看演示。

- **用披萨机器人提供答案**：LangChain 的另一个使用案例是在视频中展示的 **Pizza Bot**，它具有实时虚拟人界面。在 [YouTube](https://youtu.be/6Qa2qdlN2pU) 上查看这个移动端友好型应用的实际运行情况。

- **用于代码维护的无代码自动化**：一个名为 **Autonoma** 的无代码平台发布，旨在自动化代码改进任务，如输入验证、错误处理和测试。该平台目前已开放免费演示，并支持与 GitHub 集成。通过 [Autonoma Free Demo](https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain) 测试这些 Agent。

- **为 LM Studio 推出 VectorDB 插件**：分享了一个名为 **VectorDB** 插件的 GitHub 仓库，该插件可以创建一个 ChromaDB 向量数据库，以便在服务器模式下与 LM Studio 配合使用。仓库地址为 [VectorDB Plugin for LM Studio on GitHub](https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio)。

- **QuickVid：AI 驱动的 YouTube 摘要工具**：QuickVid 已上线，这是一款能为 YouTube 视频提供快速摘要和事实核查的新工具。访问 [QuickVid](https://quickvid.vercel.app/) 体验简洁、信息丰富的摘要，提升你的 YouTube 体验。

- **创建 Webloader RAG 应用教程**：一篇 Medium 文章详细介绍了如何使用 **Groq, Langchain, 和 Datastax** 构建强大的 Webloader RAG 应用程序，为你的应用赋能。指南详见 [Building Powerful Webloader RAG Applications with Groq, Langchain, and Datastax](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/revisiting-gpt-1-the-spark-that-ignited-llms">Revisiting GPT-1: The spark that ignited the fire of LLMs</a>: 深入探讨 GPT-1 对现代 LLM 发展的贡献</li><li><a href="https://quickvid.vercel.app/">QuickVid</a>: 暂无描述</li><li><a href="https://gitgud.autonoma.app?utm_source=discord&utm_medium=chat&utm_campaign=discord-langchain&utm_id=discord-langchain>)">GitGud</a>: 暂无描述</li><li><a href="https://youtu.be/N_GcPLJCQQY">D-ID Airbnb Use Case:  A RAG Agent Demo using Ollama and Langchain with code on Github</a>: 演示如何为商业场景构建实用的实时虚拟人助手... 我将制作一个详细的代码审查视频，以便你可以尝试... ...</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!</a>: 创建 ChromaDB 向量数据库以配合在服务器模式下运行的 LM Studio 的插件！ - BBC-Esq/VectorDB-Plugin-for-LM-Studio
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1234782249166049310)** (2 条消息):

- **来自巴黎的问候**：一位成员分享了一个名为 ["Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement"](https://youtu.be/ol2QMp64lgo) 的 YouTube 视频，演示了如何使用 **LangChain**、**Mistral Large** 和 **Llamaindex** 创建一个高级 RAG 助手。该视频面向法语社区，应用的源代码可在 **GitHub** 上的视频描述中找到。

- **DIY Llama3 RAG 助手**：另一位成员在名为 ["I want Llama3 to perform 10x with my private knowledge" - Local Agentic RAG w/ llama3"](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P) 的 YouTube 视频中，介绍了一个关于如何使用私有知识训练 **llama3** 以构建 Agentic RAG 的教程。该视频旨在指导观众如何利用自己的数据提升 **llama3** 的性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>：高级 RAG 101 - 使用 llama3 构建 Agentic RAG。获取关于 AI 如何重新定义初创公司 GTM 策略的免费 HubSpot 报告：https://clickhubspot.com/4hx🔗 链接- F...</li><li><a href="https://youtu.be/ol2QMp64lgo">Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement</a>：在这段新视频中，我向大家展示了一个基于 Agent 开发的 RAG 助手，使用了 Mistral、Langchain 和 LlamaIndex。代码 ...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1234580644273717471)** (2 条消息): 

- **不当内容警报**：一条承诺提供包含 18+ 青少年女孩的 Onlyfans **免费泄露内容**的帖子包含了一个 Discord 链接。该消息还包含表情符号和 `@everyone` 标签以引起广泛关注。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1234580388391944193)** (3 条消息): 

- **不当内容警报**：发布了一条包含露骨内容链接的消息，可能违反了 Discord 的社区准则。该消息推广了涉及未成年人的免费内容访问，这是非法且有问题的。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1234580548698247390)** (2 条消息): 

所提供的消息与 AI 协作、研究或 "looking-for-collabs" 频道的相关主题无关，且看起来是垃圾信息。因此，根据此消息没有合适的摘要内容。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1234580564871221329)** (2 条消息): 

- **不当内容警报**：发布了一条推广**成人内容**和所谓“OnlyFans 泄露”的消息，并提供了 Discord 邀请链接。此类内容显然不适合该频道，并可能违反社区准则。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

  

---


**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1234767716267855884)** (1 条消息): 

- **垃圾信息警报**：发布了一条推广**成人内容**及此类材料的垃圾消息，包括一个 Discord 邀请链接。这可能与频道的关注点无关，可能需要采取管理操作。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>：查看 Discord 上的 e-girl paradise 🍑🍒 // +18 社区 - 与其他 11801 名成员一起聚会，享受免费的语音和文字聊天。

  

---

**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1234767861927645225)** (1 messages): 

- **Inappropriate Content Alert**: A user posted a message containing explicit content and an invitation link, promoting access to what appears to be private or sensitive media involving underage individuals. The message includes emojis and a Discord invite URL.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1234580949870710797)** (2 messages): 

- **Inappropriate Content Alert**: A message promoting **18+ content** and **OnlyFans leaks** was posted, including an invitation link and emojis suggesting adult material. The content of the message is against Discord's community guidelines.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1234768131247964212)** (1 messages): 

- **Inappropriate Content Alert**: A Discord user posted a message promoting **adult content**, including a mention of *'18+ Teen Girls and onlyfans leaks for free'*, along with an invitation link to another server. The user utilized emojis and tagged **@everyone** to draw attention.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1234581080389062810)** (2 messages): 

- **Inappropriate Content Warning**: A message was posted promoting **18+ Teen Girls and OnlyFans leaks** with a Discord invite link. This type of content is likely against the platform's rules and may warrant moderation action.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1234581103633891358)** (2 messages): 

- **Inappropriate Content Alert**: The message suggests sharing of leaked content from OnlyFans involving teen girls, accompanied by a Discord invite link. This post raises serious concerns regarding legality and ethics.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1234581301672149132)** (2 messages): 

- **Inappropriate Content Alert**: A message was posted that promoted **adult content** including "18+ Teen Girls and onlyfans leaks". The post included an emoji of a peach and the underage sign, along with a Discord invitation link.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1234581174794453042)** (2 messages): 

- **Inappropriate Content Alert**: A message was posted promoting **18+ Teen Girls and OnlyFans leaks** with a Discord invite link. The content appears to be explicit and not suitable for this professional setting.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1234581352272363562)** (2 messages): 



- **不当内容警示**：一位用户发布了一条推广**成人内容**的消息，包括“18+ Teen Girls”和“OnlyFans 泄露”，并附带了一个 Discord 邀请链接（**未点击或验证**）。该消息使用了表情符号并标记了 `@everyone` 以吸引注意力。

**提到的链接**：<a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里聊天、聚会，并与你的朋友和社区保持紧密联系。

---

**AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1234713529202769981)** (1 条消息): 

- **对应对机制犯罪化的担忧**：一位成员对将某些未指明的活动犯罪化表示了强烈担忧，这些活动可能是那些遭受严重个人和法律挫折的男性最后的应对机制。人们担心，由于感到被社会边缘化，此类措施可能会迫使这些个体采取极端行动。

---

**AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1234598116523642941)** (2 条消息): 

- **与 Rosebud AI 合作的 Game Jam 盛会**：Rosebud AI 宣布与 Week of AI 合作举办一场 **Game Jam**，邀请参与者围绕“教育与 AI”的主题，使用 **Phaser JS** 开发 2D 网页游戏。**500 美元的奖金池**等你来拿，你可以点击[这里](https://twitter.com/Rosebud_AI/status/1785034624256618617)了解如何参加。

- **旧金山 AIxGames 见面会**：一场 AIxGames 见面会定于本周四在旧金山举行，旨在联系游戏领域中使用 AI 的从业者。活动共有 160 个名额，你可以在[这里](https://partiful.com/e/TwvC5qxskuPGqiliMj5f)预约（RSVP）并查看地点，Demo 演示的申请可以通过[此表单](https://forms.gle/6hiqnws3tg6EY7348)进行。

**提到的链接**：<a href="https://partiful.com/e/TwvC5qxskuPGqiliMj5f">RSVP to AIxGames Meetup | Partiful</a>：AI 已经在改变游戏格局，并且可能会带来更多改变。我们希望尽可能多地聚集在 AI 与游戏交汇领域工作的人。无论是关于...

---

**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1234530269331980359)** (8 条消息🔥): 

- **利用 LLM 彻底改变 NPC 交互**：一位用户宣布发布了由 LLM 驱动的 NPC 模型和推理栈（inference stack），以增强动作空间（action spaces）并简化 API 调用，详见 [GitHub 上的 GigaxGames](https://github.com/GigaxGames/gigax)。该解决方案包括针对复杂 NPC 动作的“单次 LLM 调用”功能，在 [Hugging Face Hub](https://huggingface.co/Gigax) 上提供开放权重，并提供 API 访问（链接似乎已失效）。

- **克服游戏开发中的 LLM 挑战**：为了追求游戏功能的运行时速度，他们面临了多个问题，例如 NPC 在执行 `speak` 命令时打破“第四面墙”，以及在大型 Prompt 中丢失细节。该用户建议，通过**输出压缩**、最小化模型调用以及利用更小的模型，可以显著提升 NPC 的性能。

- **期待对 LLM 增强型 NPC 的深入探讨**：该用户表示打算**写一篇博客文章**，分享在为改进 NPC 行为而进行 LLM 微调（fine-tuning）过程中遇到的困难和见解。

- **窥见同行的 NPC 开发之旅**：另一位用户表示，他们的项目在现有模型上也遇到了挑战，并指出 **Claude 3** 的表现更好，这可能归功于其“共情式”的训练背景。他们目前正在探索一种涉及小型 Prompt 函数调用（functional calling）的策略，并对这种方法的结果很感兴趣。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/GigaxGames/gigax">GitHub - GigaxGames/gigax: LLM-powered NPCs running on your hardware</a>：在你的硬件上运行由 LLM 驱动的 NPC。通过在 GitHub 上创建账号为 GigaxGames/gigax 的开发做出贡献。</li><li><a href="https://tally.so/r/w7d2Rz)">Form - Tally</a>：由 Tally 制作，这是创建表单最简单的方式。
</li>
</ul>

</div>

---

**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1234844604638167094)** (13 条消息🔥):

- **轻松完成本地设置**：一位成员确认他们成功在本地运行了设置，并表示整个过程非常简单直接。
- **赞赏成员贡献**：一位成员对另一位社区成员的出色工作表示了感谢。
- **在 Windows 上遇到障碍**：一位成员在 Windows 上克隆仓库时遇到问题，卡在 *'Checking for index or schema changes...'* 阶段。经澄清，**Convex local 不支持 Windows**。
- **用于日志和开发的替代命令**：建议使用 `just convex dev` 进行独立的开发同步，使用 `just convex logs` 来监控日志；这些命令提供了**定制日志**和**详细输出（verbose output）**的选项。
- **Windows 兼容性变通方案**：成员们讨论了针对 **Convex local** 缺乏 Windows 支持的解决方法，例如使用 **WSL (Windows Subsystem for Linux)** 或 **Docker**，并提到 Windows 编译支持正在开发中。
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1234486969397149837)** (15 messages🔥): 

- **探索 HaystackDB Embeddings**：一位用户引用了 [GitHub 上的 HaystackDB](https://github.com/carsonpo/haystackdb)，询问其是否使用了 **2-bit embeddings**。
- **理解二进制量化索引**：澄清了**二进制量化（Binary Quantized, BQ）**索引旨在为相似性搜索创建更小的索引，从而实现更高效的存储和搜索机制。
- **微调 LLaMA-3 的挑战**：成员们表达了在微调 **LLaMA-3** 时遇到的困难，指出模型不生成 **EOS token**，以及 Embedding 层在以不同位格式加载时存在挑战等问题。
- **困惑度微调难题**：讨论表明，针对 **LLaMA-3 的困惑度（Perplexity）微调**可能不会产生优于原始模型的结果，并有建议认为 Tokenizer 可能是导致问题的原因之一。
- **LLaMA-3 微调的潜在突破**：一位小组内部成员分享了通过使用 LLaMA-3 特定的 Prompt 格式成功微调 **LLaMA-3** 的经验，并链接到了相关的 GitHub [Pull Request 以获取更多信息](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/carsonpo/haystackdb">GitHub - carsonpo/haystackdb</a>：通过在 GitHub 上创建账号来为 carsonpo/haystackdb 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553">feat: Add LLaMA-3 instruct prompt strategies for fine-tuning by 0-hero · Pull Request #1553 · OpenAccess-AI-Collective/axolotl</a>：描述：此功能基于并包含了以下 PR 中的更改：#1542 #1539。在合并此项之前，需要先合并来自 @TJ-Solergibert 的 Fastchat PR lm-sys/FastChat#3257。动机...
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

oleegg: https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1234890920575631360)** (1 messages): 

- **Mozilla AI 正在火热招聘**：Mozilla AI 宣布了多个开放职位，正在寻找新的人才。查看机会并考虑在[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823)申请。

- **使用 Lm-buddy 评估模型**：介绍了一个名为 Lm-buddy 的开源工具，旨在帮助更有效地评估语言模型。可以通过[此处](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378)提供的链接探索该工具并做出贡献。

- **Prometheus 将本地 LLM 用于评测**：一个名为 Prometheus 的项目展示了使用本地大语言模型（LLM）担任裁判角色。这一创新应用可以在[此处](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599)链接的专用频道中进一步讨论和深入研究。
  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1234502618420613130)** (13 messages🔥):

- **AI Token 生成速度咨询**：一位成员咨询了 llama.cpp/llamafile 中 Token 生成的效率，指出他们在 llama2 的推理实现中，95% 的时间花在矩阵-向量乘法（matrix-vector multiplications）上。他们想知道 llama.cpp 中的循环展开（loop unrolling）是否是其性能快 30% 的原因，因为他们在反汇编中观察到了循环和向量化。
- **LLaMA 命名混淆**：一位用户在消息参数上遇到了一个幽默的混淆，将自己设置为 "Z" 后忘记了，导致消息出现时看起来像是 LLaMA 在自言自语，引发了一些困惑。
- **匿名入侵引发困惑**：另一位用户讲述了一个不寻常的事件，有人以 "kimkardashian" 的名字加入聊天，造成了奇怪的局面。然而，这种异常现象在随后的运行中无法复现。
- **技术集成困难**：一位用户在将 LLaMA 与 Plush-for-comfyUI 节点集成时遇到困难。尽管该节点可以与其他 OpenAI endpoints 配合使用，但无法在 llamafile 上正常运行。
- **LLaMA3 兼容性与支持沟通**：已确认在 M1 Macbook Air 上使用 llamafile 运行 LLaMA3:8b 存在问题，而在 Ollama 上运行则没有问题。已承诺在解决 LLaMA3 的其他现有问题后，将优先进行 M1 兼容性测试。
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1234545676260474942)** (1 messages): 

由于提供的消息似乎是唯一的，或者是没有额外上下文或其它消息的单条消息的一部分，因此无法进行总结。请提供来自 "ideas-and-feedback" 频道的完整消息集，以便我创建适当的总结。
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1234547539186024519)** (4 messages): 

- **与 Hanna Hajishirzi 一起探索 OLMo**：分享了来自 [AI2](https://homes.cs.washington.edu/~hannaneh/) 的 Hanna Hajishirzi 最近在康奈尔科技学院开源生成式 AI 研讨会上所做的关于 "OLMo: Findings of Training an Open LM" 的[演讲](https://youtu.be/qFZbu2P1vZ8)。演讲幻灯片可以在[这里](https://drive.google.com/file/d...)查看。
- **信息流的强度**：一位成员透露 Hanna Hajishirzi 是他们的经理，工作节奏极快，暗示了她讲座的深度和复杂性。
- **OLMo 演示虽然信息量巨大但令人印象深刻**：另一位成员认为 Hanna 25 分钟演讲的内容（涵盖 OLMo, Dolma, Tulu 等主题）非常庞大且有点令人应接不暇，但承认她令人印象深刻的履历以及这些信息对学生的价值。

**提到的链接**：<a href="https://youtu.be/qFZbu2P1vZ8">Hanna Hajishirzi (AI2) - OLMo: Findings of Training an Open LM</a>：康奈尔科技学院开源生成式 AI 研讨会的演讲。演讲者：https://homes.cs.washington.edu/~hannaneh/ 幻灯片 - https://drive.google.com/file/d...

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1234622923449569322)** (2 messages): 

- **通过 Gist 获取 John Schulman 的见解**：一份 GitHub [Gist](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) 提供了宝贵的见解，总结了 **John Schulman** 关于基于语言模型的系统的强化学习（reinforcement learning）的演讲。

- **质疑 AI 排行榜的效用**：Sayash Kapoor 和 Benedikt Stroebl 的一篇[博客文章](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful)声称，目前没有准确的方法来确定用于代码生成的最佳 AI。他们强调，虽然 LLM 调试器（**LDB**）在 HumanEval 代码生成排行榜上名列前茅，但由于依赖运行昂贵的语言模型（如 GPT-4），它是一个成本高昂的 Agent。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful">AI leaderboards are no longer useful. It&#x27;s time to switch to Pareto curves.</a>：花费 2,000 美元能告诉我们关于评估 AI Agent 的什么信息</li><li><a href="https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81">rl-for-llms.md</a>：GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot 新闻：<@&1216534966205284433>
  

---



**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/1234606317595791490)** (1 messages):

- **知名 AI 驱动公司 Gamma 招聘 AI 工程师**：**Gamma** 在 a16z 的前 100 名消费级 AI 应用中排名第 16，目前正在寻找一名 **AI engineer**，旨在通过 AI 创新演示文稿和网站设计。该职位职责包括 **prompt engineering**、**metrics/evaluations**、**fine-tuning**，以及利用前沿模型开发新功能。职位详情见 [Gamma Careers](https://careers.gamma.app/ai-engineer)。

- **挑战 Large Language Models 的极限**：如果候选人在最大化 **Large Language Models (LLMs)** 潜力方面拥有实践专长，即使没有丰富的工程经验也会被考虑。该职位位于 **San Francisco**，需要线下协作。

- **Gamma 令人瞩目的 AI 驱动增长与文化**：Gamma 拥有超过 **1000 万用户**（自然增长），**已实现盈利且获得 1000 万美元以上融资**，团队规模仅为 **16 人的精干团队**，并推行在 **San Francisco** 混合办公的办公室文化。

- **大规模创意内容生成**：Gamma 的目标是**简化内容创作**，目前每天生成超过 100 万张图片并处理数百万次 LLM 请求。他们旨在消除创作**极具吸引力的演示文稿和网站**过程中的复杂性。

**提到的链接**：<a href="https://careers.gamma.app/ai-engineer">AI Engineer</a>：AI Engineer San Francisco 点击此处申请

---

**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1234583399029805107)** (3 条消息): 

- **关于 GPT-4.5 泄露的猜测**：@phill__1 的一条推文引发了讨论，他认为 gpt2-chatbot 感觉像是 **GPT-4.5**，并拥有“惊人的领域知识”。推文链接：[phill__1 的观察](https://x.com/phill__1/status/1784964135920235000)。
- **社区对潜在泄露议论纷纷**：频道成员表示相信 gpt2-chatbot 可能是 **GPT-4.5** 的一次意外预览。
- **对神秘 Bot 的简短赞赏**：一位成员分享了简短的评价，仅表示：“它很棒”。

**提到的链接**：<a href="https://x.com/phill__1/status/1784964135920235000">Phil (@phill__1) 的推文</a>：无论 gpt2-chatbot 是什么，它绝对感觉像 gpt4.5。它拥有我从未见过的惊人领域知识。

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1234505496761991198)** (3 条消息): 

- **代码生成对话的自定义 Grammar**：一位用户表示有兴趣传递自定义 **grammar**（可能作为模型特定选项），以便在代码生成中专注于语义错误而非语法错误。

- **Datasette 用户体验脑暴**：寻求关于 Datasette 首页 **UX** 设计的想法，允许用户从下拉菜单中选择选项，例如选择一个国家来生成摘要表。

- **通过下拉选择直接访问数据**：一位成员提出了两种 **UX** 方案：一种是在事件发生时更新 **URL**，将用户引导至相关数据；另一种是允许用户通过根据选择更新 **canned queries** 来“构建”主页。

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1234775513499963463)** (1 条消息): 

- **本地机器快速加载**：讨论围绕一个观察展开：某个进程*在机器上运行时 3 秒即可加载*，但通过*提交 job* 进行同样操作时似乎存在问题。这表明在 **job** 提交语境下，存储可能不是导致问题的原因。

---

**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/)** (1 条消息): 

le_mess: **Llama 3** 在 **ScandEval** 上似乎击败了 **GPT-4**
https://scandeval.com/german-nlg/

---