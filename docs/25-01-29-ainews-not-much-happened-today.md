---
companies:
- deepseek
- hugging-face
- dell
- openai
date: '2025-01-30T01:07:40.495919Z'
description: '**DeepSeek-R1 和 DeepSeek-V3** 模型取得了重大进展。这些模型基于包含 **150 万个样本的指令微调数据集**进行训练，其中包括
  **60 万条推理数据**和 **20 万条非推理 SFT（监督微调）数据**。


  这些模型展现了强劲的**性能基准**，并已通过与**戴尔（Dell）**和 **Hugging Face** 的合作实现了**本地化部署**。其训练成本估计在
  **550 万至 600 万美元**之间，并在 **8xH100 服务器**上实现了高效的硬件利用。


  **《国际人工智能安全报告》**强调了诸如**恶意使用**、**故障**以及包括 **AI 驱动的网络攻击**在内的**系统性风险**。行业领袖 **Yann
  LeCun** 和 **Yoshua Bengio** 就市场反应、AI 安全和伦理考量分享了见解，并强调了 AI 在创造力和经济激励方面的作用。'
id: f2aa0ef6-e431-40e9-97a8-a466b41c6991
models:
- deepseek-r1
- deepseek-v3
- coder-v2
- prover
original_slug: ainews-not-much-happened-today-2391
people:
- yann-lecun
- yoshua-bengio
- francois-chollet
- giffman
title: 今天没发生什么特别的事。
topics:
- instruction-tuning
- performance-benchmarks
- model-deployment
- training-costs
- hardware-scalability
- ai-safety
- risk-mitigation
- ethical-ai
- open-source
- gpu-utilization
---

<!-- buttondown-editor-mode: plaintext -->**安全端点就是你所需的一切。**

> 2025/1/28-2025/1/29 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discords（**225** 个频道和 **4890** 条消息）。预计节省阅读时间（按 200wpm 计算）：**549 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

关于 [**Grok 3**](https://x.com/btibor91/status/1884612786183135627) 和 **o3-mini** 的传闻仍在继续。

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

**DeepSeek 的进展与性能**

- **DeepSeek-R1 与 V3 的进步**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1884440764677251515) 强调，从 **DeepSeek-R1** 蒸馏出的 **DeepSeek-V3** 是在一个包含 **150 万样本的指令微调数据集**上训练的。此外，[@alexandr_wang](https://twitter.com/alexandr_wang/status/1884440764677251515) 强调，DeepSeek 模型正在**刷新开源模型已披露的后训练数据量记录**，其中包括 **60 万条推理数据**和 **20 万条非推理 SFT 数据**。
  
- **性能基准测试**：[@teknium1](https://twitter.com/teknium1/status/1884500035934773446) 指出，**DeepSeek-R1 AI + Groq** 实现了**“以思考的速度”**进行编程。此外，[@osanseviero](https://twitter.com/osanseviero/status/1884356079217434995) 指出，**DeepSeek** 在过去一年中持续发布了 **Coder V2** 和 **Prover** 等模型，展示了持续的**模型性能和创新**。

**AI 模型训练、成本与硬件**

- **训练成本与基础设施**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1884699523064164837) 对 **DeepSeek** 宣称的 **550 万美元训练成本**提出质疑，认为实际成本涉及**消除 Token 路由效率低下**以及通过**流水线训练（pipelined training）**保持较低的通信量。此外，[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1884391362012819459) 提供了一份估算，认为 **V3 的完整预训练成本**大约在 **600 万美元**左右。

- **硬件利用率**：[@giffmana](https://twitter.com/giffmana/status/1884689840278548644) 讨论了 **DeepSeek 在 GPU 使用**方面的竞争优势，而 [@MarkTenenholtz](https://twitter.com/MarkTenenholtz/status/1884370601672073230) 提到一台 **8xH100 服务器**即可运行 **DeepSeek-R1**，这表明了此类模型所需的**硬件可扩展性**。

**开源 AI 与部署**

- **部署平台**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1884345858121941177) 宣布，通过**与 Dell 和 Hugging Face 的合作**，**DeepSeek-R1** 现已支持**本地部署（on-premise）**，为企业用户提供了便捷的**开源部署**方案。

- **社区与贡献**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1884593470116699750) 肯定了在编写《**国际 AI 安全报告**》过程中的协作努力，而 [@Markchen90](https://twitter.com/markchen90/status/1884499976488919145) 则参与了关于 **AI 风险评估**和**模型部署策略**的讨论。

**AI 安全、风险与伦理**

- **安全报告与风险缓解**：[@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1884593469265502482) 详细介绍了《**国际 AI 安全报告**》，将**风险**分为**恶意使用**、**故障**和**系统性风险**。这包括对 **AI 驱动的网络攻击**和**环境影响**等担忧。

- **伦理考量**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1884638441008181573) 赞扬了**版权局（Copyright Office）**对 **AI 工具辅助人类创作**的立场，强调在适当使用时，**AI 不会削弱版权保护**。

**AI 行业洞察与对比**

- **市场反应与竞争力**：[@ylecun](https://twitter.com/ylecun/status/1884692313118495221) 批评了**市场对 DeepSeek 不合理的反应**，认为**性能基准测试**证明了 **DeepSeek 的竞争优势**。此外，[@giffmana](https://twitter.com/giffmana/status/1884689840278548644) 强调 **DeepSeek 的推理能力**超过了许多开源模型，使其在面对 **OpenAI** 时具有强劲的竞争力。

- **投资与经济影响**：[@fchollet](https://twitter.com/fchollet/status/1884675418223239306) 讨论了驱动 **AI 发展**的**经济激励措施**，而 [@scaling01](https://twitter.com/scaling01/status/1884507045233041542) 则认为**使用 GPT-4o** 等同于**向 OpenAI 捐款**，反思了 **AI 行业**内部的**成本动态**。

**梗/幽默**

- **轻松的互动**：[@ylecun](https://twitter.com/ylecun/status/1884384838699954520) 和 [@gabrielpeyre](https://twitter.com/ylecun/status/1884597537618739351) 进行了幽默的交流，使用了 **"LOL"** 和 **🤣🤣🤣** 等反应，展示了 AI 社区技术讨论中**轻松的一面**。

- **幽默的 AI 输出**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1884536280949604391) 分享了一个**有趣的 AI 生成脚本**，用于制作**弹跳的黄球**，将**技术脚本**与 **AI 创意幽默**结合在一起。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. 关于 DeepSeek R1 模型与蒸馏版本的混淆**

- **公益公告 (PSA)：你的 7B/14B/32B/70B "R1" 并不是 DeepSeek。** ([Score: 1246, Comments: 357](https://reddit.com/r/LocalLLaMA/comments/1icsa5o/psa_your_7b14b32b70b_r1_is_not_deepseek/))：该帖子澄清了 **7B/14B/32B/70B "R1" 模型**并非真正的 **DeepSeek** 模型，而是对现有稠密模型（如 **Qwen 2.5** 和 **Llama 3.3**）的微调（finetunes）。真正的 **DeepSeek** 模型是完整的 **671B** 版本，作者对因常见误解而需要反复解释感到沮丧。
  - 围绕 DeepSeek 模型的**命名混淆**是一个主要问题，许多用户被 **Ollama 的命名规范**所误导。由于 "DeepSeek-R1:70b" 这种具有误导性的名称，**蒸馏模型 (distilled models)** 经常被误认为是完整的 R1 模型，而这些名称并未明确指出它们是 **Qwen 2.5** 和 **Llama 3.3** 的较小微调版本。
  - 讨论强调了 YouTube 和 TikTok 等平台上普遍存在的**虚假信息**，创作者经常声称在本地运行 DeepSeek，导致了广泛的误解。用户表示，需要不断澄清这些并非完整的 **671B DeepSeek** 模型（该模型需要超过 **1TB 的 VRAM**，家庭使用并不可行），这让他们感到沮丧。
  - 强调了蒸馏（distillation）与微调（fine-tuning）之间的**技术区别**，多条评论解释说，所谓的“蒸馏”实际上只是在 R1 的回复上进行微调。真正的 R1 是一个 **Mixture of Experts (MoE)** 模型，与正在被微调的 Qwen 2.5 和 Llama 3.3 等稠密模型（dense models）有显著不同。


- **[好东西](https://i.redd.it/azitnmgpqxfe1.png)** ([Score: 289, Comments: 138](https://reddit.com/r/LocalLLaMA/comments/1icttm7/good_shit/))：**OpenAI** 指责中国的 **DeepSeek** 使用其模型来训练竞争对手，引发了对知识产权盗窃的担忧。**白宫 AI 顾问 David Sacks** 强调了这些问题，正如《金融时报》一篇刊登了两家公司 Logo 的文章所描述的那样。
  - 许多评论者批评 **OpenAI** 指责 **DeepSeek** 盗窃知识产权，考虑到 OpenAI 自身也使用公共数据进行训练，这显得十分讽刺。**DeepSeek** 被一些人视为“罗宾汉”式的角色，而这一指控被视为通过将“中国威胁”武器化来扼杀竞争的策略。
  - 人们对 **OpenAI 服务条款 (Terms of Service)** 的可执行性持怀疑态度，一些人认为服务条款在某些司法管辖区（可能包括中国）可能不具备法律效力。其他人则认为 **DeepSeek** 为其使用的 Token 支付了费用，因此并未违反任何协议。
  - 评论者中更普遍的情绪是呼吁 **OpenAI** 专注于改进产品而非诉讼，一些人因感知到的贪婪和虚伪而主张抵制 "ClosedAI" 的产品。


- **DeepSeek CEO 下的一盘大棋 (4D Chess)** ([Score: 478, Comments: 91](https://reddit.com/r/LocalLLaMA/comments/1icmxb5/4d_chess_by_the_deepseek_ceo/))：**DeepSeek** 的 CEO **梁文锋**认为，像 **OpenAI** 这样的闭源方法只能提供暂时的竞争优势。相反，他强调建立强大的团队和组织文化以促进创新，才是可持续的竞争护城河。[点击此处阅读更多](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas)。
  - 讨论强调了 **DeepSeek** 使用 **PTX** 而非 **CUDA** 的技术优势，由于过去十年 Python 和 CUDA 的根深蒂固，许多美国工程师并不具备处理 PTX 的能力。这一选择赋予了 DeepSeek 显著的技能优势，因为 PTX 在训练时效率更高，而转向 PTX 需要大幅提升技能水平。
  - **DeepSeek** 对 AI 领域的影响被比作 90 年代的 **Unix** 开源运动，暗示竞争格局可能会发生转变。如果 **OpenAI** 和其他美国公司不适应 DeepSeek 所展示的高效率，它们在维持竞争优势方面可能会面临挑战，这可能导致其竞争护城河被快速且廉价地侵蚀。
  - **DeepSeek** 在金融领域的创新得到了认可，讨论涉及其从仅将 ML 应用于金融到构建基础模型的战略转变。此举被视为获得对技术更深层次控制和理解的一种方式，突显了在量化金融公司内部拥有机器学习专业知识的价值。


**主题 2. 关于美国禁用 DeepSeek 的猜测及其市场影响**

- **[DeepSeek 很快会在美国被禁吗？](https://i.redd.it/5gpitg40dtfe1.png)** ([Score: 1371, Comments: 863](https://reddit.com/r/LocalLLaMA/comments/1icer8t/will_deepseek_soon_be_banned_in_the_us/))：该帖子推测美国可能会**禁掉 DeepSeek**，因为**白宫**正在审查其对国家安全的影响。信息来源于 **InsidersHut** 账号，引发了人们对 DeepSeek AI 平台在该国未来可用性的担忧。
  - **开源与可访问性**：许多评论者强调 **DeepSeek** 是开源的，其模型（包括 **670B 参数版本**）可以在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1) 等平台下载。这使得禁令难以有效实施，因为用户可以在本地或私有服务器上运行这些模型。
  - **安全与竞争担忧**：讨论围绕着因**国家安全威胁**而禁止开源 AI 的讽刺性展开，而其他评论者则认为此举更多是为了遏制来自非美国实体的竞争。一些人对安全风险表示怀疑，质疑禁止一个可以在离线状态下运行且不向中国发送数据的工具的实际可行性。
  - **对美国政策的批评**：许多评论批评美国处理外国技术竞争的方式，将其比作保护主义，并与过去针对 TikTok 等中国公司的行动相提并论。有一种观点认为，禁止 DeepSeek 违背了自由市场的理想，反映了对被创新的外国技术超越的恐惧。


- **[如此多关于 DeepSeek 的恐慌情绪](https://i.redd.it/zfykihriztfe1.jpeg)** ([Score: 539, Comments: 234](https://reddit.com/r/LocalLLaMA/comments/1ichk40/so_much_deepseek_fear_mongering/))：该帖子批评了围绕 **DeepSeek** 广泛传播的**恐慌情绪**，并质疑那些反对者的可信度。它引用了一篇 **LinkedIn 帖子**，该帖子将 DeepSeek 描绘成潜在的网络安全威胁，敦促对其战略影响和透明度进行审查，该帖子获得了 **3,058 次反应**、**1,148 条评论**和 **433 次转发**。
  - 讨论强调了对 **DeepSeek** **恐慌情绪**的怀疑，用户将其与 COVID 疫苗辩论期间的无端指责相提并论。批评者认为这种恐惧被夸大了，并质疑这些叙事背后的动机，认为这是一种操纵认知或市场的策略。
  - 一些评论者强调了**透明度和安全性**问题，指出与 OpenAI 等专有模型不同，**DeepSeek** 是开源的，允许任何人检查其代码。用户指出，可以通过在本地运行模型或使用具有良好隐私政策的服务来降低安全风险，从而质疑恐慌叙事的一致性。
  - 对话中混合了讽刺和严肃的批评，用户嘲笑 **DeepSeek** 构成重大威胁的想法，而另一些人则对数据隐私和使用不同国家开发的 AI 工具的地缘政治影响提出了合理的担忧。这反映了人们对企业和政府实体在管理 AI 技术方面更广泛的不信任。

- **关于 DeepSeek 遭受 DDoS 攻击的一些证据已发布！** ([Score: 322, Comments: 87](https://reddit.com/r/LocalLLaMA/comments/1icjg39/some_evidence_of_deepseek_being_attacked_by_ddos/))：**DeepSeek** 在 1 月份经历了一系列 **DDoS 攻击**，不同阶段涉及 **HTTP proxy 攻击**、**SSDP 和 NTP 反射放大攻击**以及**应用层攻击**。攻击在北京时间 1 月 28 日 03:00-04:00 达到顶峰，证据表明攻击目标是海外服务提供商，特别是来自美国的 IP，其中许多是 **VPN** 出口。**DeepSeek** 迅速做出反应，于 1 月 28 日 00:58 切换了其 IP 以缓解攻击，这与其安全公告一致。
  - 几位评论者认为，针对 **DeepSeek** 的 **DDoS 攻击**可能根本不是攻击，而是由于用户兴趣激增和服务器基础设施不足造成的。**AnhedoniaJack** 和 **PhoenixModBot** 强调，合法流量的突然激增可能会模仿 **DDoS** 模式，特别是如果基础设施没有为高负载做好准备。
  - **Johnxreturn** 和 **mobiplayer** 讨论了针对 **DDoS** 的技术防御，提到了 **WAF**、**OWASP 漏洞**和 **CDN** 网关，同时质疑这些措施对 **NTP 放大攻击**等特定攻击的有效性。**Mobiplayer** 批评了对 **NTP 放大攻击**工作原理的误解，指出了某些解释中的技术错误。
  - 对攻击证据和来源的怀疑很普遍，**TsaiAGw** 和 **YT_Brian** 等用户质疑将攻击归因于美国的来源可靠性。**Agabeckov** 和 **PhoenixModBot** 要求提供更详细的技术数据来证实 **DDoS 攻击**的说法，认为由于缺乏适当的分析，感知到的攻击可能被误解了。


**Theme 3. DDoS 攻击背景下的 DeepSeek API 挑战**

- **伯克利 AI 研究团队声称以 30 美元复现 DeepSeek 核心技术** ([Score: 286, Comments: 87](https://reddit.com/r/LocalLLaMA/comments/1icwys9/berkley_ai_research_team_claims_to_reproduce/))：由 **Jiayi Pan** 领导的**加州大学伯克利分校**研究团队声称仅用 **30 美元**就复现了 **DeepSeek R1-Zero** 的核心技术，展示了如何以极具成本效益的方式实现先进的 **AI** 模型。该团队使用了一个拥有 **30 亿参数**的小语言模型，通过**强化学习**开发了自我验证和搜索能力，这可能会挑战 **OpenAI** 的市场地位。
  - **OpenAI 的地位与技术**：有人认为 **OpenAI** 已经意识到 **DeepSeek** 使用的技术，虽然这些方法的复现令人印象深刻，但 **OpenAI** 可能会利用更多资源来实现它们。讨论强调，**OpenAI** 的模型（如 **o3 model**）实现了高性能，但计算成本巨大，这表明 **AI** 开发中存在降低成本的潜力。
  - **强化学习与开源**：**强化学习 (RL)** 的复兴和开放知识转移被强调为关键优势，特别提到了 **TinyZero** 仓库在 [GitHub](https://github.com/Jiayi-Pan/TinyZero) 上的可用性。这种方法允许模型进行自我改进和蒸馏，可以应用于像 **LLaMa 3.1 405B** 这样的大型模型，从而增强其能力并支持开源 **AI** 项目的可行性。
  - **市场影响与开源可行性**：正如 **DeepSeek** 所展示的，蒸馏方法的成功对 **OpenAI** 和 **Anthropic** 等公司的专有模型提出了挑战。通过开源方法创建高性能、定制化模型的能力表明，行业正转向更具可行性的开源项目，这影响了竞争格局，并可能迫使专有基础设施策略发生变化。

- **[DeepSeek API：每次请求都超时 :(](https://i.redd.it/wpmv3ibe0ufe1.png)** ([Score: 246, Comments: 83](https://reddit.com/r/LocalLLaMA/comments/1ichohj/deepseek_api_every_request_is_a_timeout/))：该帖子幽默地批评了 **DeepSeek API** 频繁出现超时的问题，并用一张墓碑图像象征其在 **2025年1月** 短暂的功能寿命。讽刺的语气凸显了用户对该 API 不稳定性的沮丧。
  - 用户对 **DeepSeek** 因提供免费服务而产生的长期可持续性表示担忧，部分用户在访问平台时遇到了 **503 errors**。**Openrouter** 提供了替代的（尽管更贵）API 端点来运行 **R1 671b model**，且运行效果良好。
  - 讨论强调了 **DeepSeek** 的问题与过去 **GPT-4** 停机事件之间的相似之处，将问题归因于知名度激增以及可能的 **DDoS attacks**。一些人推测中国的**春节**可能也导致了服务中断。
  - 平台之间的竞争备受关注，**ChatGPT** 针对 **DeepSeek** 的问题，取消了其 **basic pro plan** 的典型限制，展示了竞争市场带来的好处。用户还讨论了 **open-source** 选项的可用性以及独立运行较小模型的能力。


## 其他 AI Subreddit 总结

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 的指控：DeepSeek 利用了他们的模型**

- **[OpenAI 声称有证据表明中国的 DeepSeek 使用其模型训练竞争产品](https://www.ft.com/content/a0dfedd1-5255-4fa9-8ccc-1fe01de87ea6)** ([Score: 589, Comments: 418](https://reddit.com/r/OpenAI/comments/1iclu4b/openai_says_it_has_evidence_chinas_deepseek_used/))：OpenAI 声称中国的 **DeepSeek** 使用其模型来训练竞争 AI。由于帖子中未提供进一步的背景或细节，支持这一指控的影响或证据仍不明确。
  - 许多评论者指出了 **OpenAI** 投诉中的讽刺之处，指出 OpenAI 本身也使用了来自互联网的数据（包括可能受版权保护的材料）来训练他们的模型。**DeepSeek** 被指控使用 OpenAI 的模型，但这反映了 OpenAI 最初如何基于现有技术和数据集进行构建。
  - 据报道 **DeepSeek** 使用了可能由 **OpenAI models** 生成的合成数据，这引发了关于此类模型的输出是属于用户还是模型创建者的讨论。这引起了对 **OpenAI's terms of service** 的担忧，以及他们是否对用户生成的输出主张所有权，从而可能散布 *恐惧、不确定和怀疑 (FUD)*。
  - 一些评论讨论了 AI 训练的技术和经济方面，例如 **Runpod** 等平台上的**电力成本**和 **GPU pricing**。提到了 **H100 GPU** 的功耗为 **0.7 kilowatt**，成本为 **每 GPU 小时 1.99 美元**，突显了 AI 模型训练所需的大量资源。


- **[Anduril 创始人对 DeepSeek 的看法](https://i.redd.it/w5z4gnu3evfe1.png)** ([Score: 306, Comments: 179](https://reddit.com/r/OpenAI/comments/1icmwlu/andurils_founder_gives_his_take_on_deepseek/))：**Anduril** 创始人 **Palmer Luckey** 批评了媒体对 **DeepSeek** 500 万美元估值的反应，认为这是夸大其词，并受到一家有别有用心的中国对冲基金的影响。他认为媒体叙事对美国科技公司存有偏见，并强调了关于 AI 初创公司投资的误导信息，正如他在 2025 年 1 月 28 日发布的 Twitter 帖子所证实的，该帖子获得了 1000 次转发、3000 次点赞和 2500 次分享，浏览量达 160 万次。
  - 讨论凸显了对 **DeepSeek** **500 万美元估值**的怀疑，评论认为考虑到基础设施和工资等因素，实际成本要高得多。一些人认为媒体和公众被过度简化的数字误导了，而另一些人则认为这种叙事被美国公司用来为在竞争中输给中国开脱。
  - 存在对**媒体偏见**的重大批评，一些评论者认为媒体叙事不公平地针对美国科技公司或支持特朗普等政治人物。其他人则反驳说，媒体并非铁板一块，可能有各种偏见，有时甚至为了收视率而偏袒大型科技公司或政治人物。
  - 对话还涉及 **open-source contributions**，一些人承认中国在促进开源 AI 发展方面的作用。评论者赞赏这些贡献带来的能源节省和性能提升，并将其与 OpenAI 等公司缺乏透明度形成对比。


**主题 2. Qwen 2.5 Max vs GPT-4o：价格与性能的碰撞**

- **[总统先生，第二个中国 AI 已进入市场](https://i.redd.it/iqabn8v52xfe1.png)** ([得分: 1600, 评论: 99](https://reddit.com/r/OpenAI/comments/1icrhk2/mr_president_the_second_chinese_ai_has_hit_the/)): **Alibaba** 推出了一款新的 **AI 平台**，据报道其性能超越了 **Deepseek**，这一消息由 "The Spectator Index" 在推文中发布。截至 **2025 年 1 月 29 日**，该推文已获得 **1.78 万次浏览**。
  - Alibaba 的 **Qwen 2.5 Max 模型** 因其高昂的成本而受到关注，价格比 **GPT-4o** 贵 **3-4 倍**，输入 token 价格为 **$10/M**，输出 token 价格为 **$30/M**，相比之下 **Deepseek** 的成本要低得多。然而，它缺乏 **thinking mode** 且并非开源，这限制了它的可访问性和吸引力。
  - 用户对 Alibaba AI 的性能评价褒贬不一，一些人称赞其 **图像和视频生成能力**，并提供了 **粉色橡皮鸭视频** 和 **握手视频** 等示例。另一些人则批评其推理能力，称其不如 **Deepseek-v3** 先进。
  - 讨论中还涉及了其他替代 AI 模型，**Hugging Face** 正在开发 **Deepseek** **R1** 的开源版本，名为 **open-r1**，旨在提供更易于获取且功能强大的 AI 解决方案。


- **[“长官，中国刚刚发布了另一个模型”](https://i.redd.it/guo4m8iyxwfe1.png)** ([得分: 514, 评论: 45](https://reddit.com/r/OpenAI/comments/1icr5ud/sir_china_just_released_another_model/)): 来自中国的全新 AI 模型 **Qwen 2.5 Max** 现已通过 **Alibaba Cloud** 提供使用，正如 **Junyang Lin** 在推文中所述。该帖子幽默地强调了模型的发布，并邀请用户通过提供的链接进行探索。
  - **技术信任度**：人们对中国技术的可靠性存在怀疑，但一些用户认为中国技术与美国技术一样可靠，并对 **Google**、**OpenAI** 和 **Meta** 等公司的诚信提出质疑。
  - **性能担忧**：用户对声称与大型模型并驾齐驱的新 **LLMs** 表示怀疑，质疑它们在现实任务中的表现。一位用户分享了直接测试 **Qwen 2.5** 的链接，指出它在调整 **Python** 代码方面很有用，但强调在复杂场景下需要进行事实核查。
  - **服务可用性**：据报道该服务遭到了 **DDOS 攻击**，影响了其可用性，尽管目前尚不清楚该问题在初始报告后是否仍然存在。


**主题 3. Gemini 2 的 Flash Thinking：AI 速度的演进**

- **[当我们还在关注 OpenAI vs Deepseek 时](https://i.redd.it/q3c8l9h4wsfe1.jpeg)** ([得分: 2043, 评论: 80](https://reddit.com/r/OpenAI/comments/1icci3r/while_we_got_openai_vs_deepseek/)): **Gemini 2** 的 flash 功能在一段幽默的对话中得到了体现，虚拟助手在回答一年有多少秒的问题时，调皮地列出了每月的日期。这展示了助手在保持现代且具有视觉吸引力的界面的同时，参与轻松、对话式互动的能力。
  - **Google Assistant vs Gemini**：讨论澄清了 **Google Assistant** 和 **Gemini** 是不同的，**Gemini** 在某些任务中会调用助手。一些用户批评 **Google Assistant** 的智能程度，指出与 **Google AI Studio** 中更先进的 AI 系统相比，它存在局限性。
  - **AI Studio vs Gemini App**：用户强调 **Google AI Studio** 提供了比 **Gemini** app 更强大的 AI 能力，后者被认为在处理高级任务时效果较差。**AI Studio** 因其免费访问和高级功能而受到称赞，而 **Gemini** app 被认为仅适用于日常休闲使用。
  - **Gemini 2 的独特功能**：**Gemini 2** 以其 **flash thinking** 能力而闻名，这使其能够快速处理大量数据，如视频或书籍。然而，用户指出这些功能需要 **AI Studio** 中的特定工具，在主版本 **Gemini** 中不可用。


---

# AI Discord 摘要

> 由 Gemini 2.0 Flash Exp (gemini-2.0-flash-exp) 生成的摘要之摘要的总结

**主题 1: DeepSeek R1 模型热潮：性能、问题与前景**

-   **DeepSeek R1 被大幅压缩！**：Unsloth AI 将 [**DeepSeek R1 1.58-bit**](https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit) 从 720GB 压缩到了苗条的 131GB，同时运行速度仍能达到 140 tokens/sec。事实证明，[选择性层量化（selective layer quantization）](https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/)是这种压缩魔法的关键。此外，[Magpie-Align 的数据集](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)激发了 CoT 训练实验。虽然一些成员担心如果没有显式的训练数据，推理能力可能会退化，但其他人则希望扩大数据集规模。
-   **DeepSeek vs. OpenAI 对决：这不仅仅是模型之战**：社区正在针对编程和创意任务，将 [**DeepSeek R1**](https://openrouter.ai/deepseek/deepseek-r1:free) 与 OpenAI 模型进行对比测试。早期结果显示 DeepSeek 在连贯性方面表现出色，但在敏感领域也触及了内容限制。与此同时，一段声称 DeepSeek 揭露了科技寡头数十亿美元骗局的 [YouTube 视频](https://youtu.be/e9XCbX40ijA)也在流传，引发了关于审查制度的讨论。
-   **DeepSeek 数据泄露引发严重警示**：一个被称为 “DeepLeak” 的公开暴露的 [ClickHouse 实例](https://x.com/wiz_io/status/1884707819737223591)泄露了密钥、聊天记录和数据外泄途径，让人们意识到 [API key 泄露](https://x.com/h4x0r_dz/status/1884566387349225598)是一个迫在眉睫的威胁。

**主题 2：模型部署与硬件难题**

-   **Mac 在 LM Studio 加载中受阻**：[LM Studio](https://lmstudio.ai/docs/basics/import-model) 用户在 **Mac** 设备上遇到了“模型加载失败”的问题，这归咎于最低硬件规格要求和 GPU 显存限制，用户也敦促通过频繁的 Beta [更新](https://lmstudio.ai/docs/basics/import-model)来修复。社区注意到显存限制会导致系统冻结，而 [gguf](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 文档对于修复至关重要。此外，关于本地使用 [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) 还是 DeepSeek 的权衡讨论也在进行中。
-   **内存带宽是本地 LLM 的核心**：性能现在很大程度上取决于内存带宽，Mac 在这方面不如 [A4000 或 3060 等 GPU](https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix)。一位用户开玩笑说：“即使使用 Threadripper CPU，你也跑不赢内存带宽。”
-   **DeepSeek 已上线 Azure 和 GitHub**：该模型现已在 [Azure AI Foundry](https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/) 和 GitHub 上提供，使企业级 AI 更易于获取。

**主题 3：AI 工具、框架及其特性**

-   **Cursor 难以维持稳定**：最近的 [Cursor IDE](https://forum.cursor.com/t/upgrade-to-0-45-x-always-breaks-cursor/44688/8) 更新引发了混乱，破坏了 Tab 补全功能并错误解析 Markdown，用户表示：“Cursor 不再能正确显示其 Markdown 输出。”与此同时，用户对 [Claude 3.5](https://forum.cursor.com/t/sonnet-3-5-stops-working/46053) 的额度限制感到不满，因为它在 50 次请求后就会锁定使用。
-   **OpenRouter 的 DeepSeek 集成**：虽然 [Chutes](https://chutes.ai/) 现在为 [DeepSeek R1 提供免费端点](https://openrouter.ai/deepseek/deepseek-r1:free)，但用户在 DeepSeek v3 的翻译质量上遇到了问题，并批评 OpenRouter 收取 5% 的 API 费用，呼吁更好的错误处理机制。
-   **Windsurf 表现挣扎，用户渴望 DeepSeek**：[Windsurf](https://status.codeium.com) 用户抱怨缺少 DeepSeek R1 集成，有些人甚至威胁要转向 [Cursor](https://x.com/shaoruu/status/1884395102195548357) 以获得更好的 Tool calling 能力。他们还批评 Sonnet 在编程方面的不可靠性，称其 Prompt 理解能力下降并要求更快的修复，同时还指出了 Cascade 的问题。

**主题 4：训练技术与新兴模型**

-   **Mixture-of-Experts 获得内存提升**：社区强调内存大小对于 CPU 设置上的 MoE 性能至关重要，同时分享了优化技巧，指出 *类 HPC 资源管理* 优于标准配置。此外，一篇新论文 [Autonomy-of-Experts (AoE)](https://arxiv.org/abs/2501.13074) 被引入，让模块决定是否应该处理某个输入，从而潜在地提高效率。
-   **Min-P 采样方法**：[社区中](https://openreview.net/forum?id=FBkpCyujtS)正在讨论 **min-p** 采样的引入，该方法根据模型置信度调整阈值，旨在增强文本质量和多样性。
-   **稀疏自编码器可能不可靠**：一篇新论文揭示，稀疏自编码器 (SAEs) 在不同种子之间仅共享 30% 的学习特征，这引发了对可解释性[任务](https://arxiv.org/abs/2501.16615)中特征稳定性和可靠性的质疑。

**Theme 5: AI Ethics, Data, and the Future**

-   **对 DeepSeek 数据实践的担忧加剧**：[Bloomberg](https://www.bloomberg.com/news/articles/2025-01-29/microsoft-probing-if-deepseek-linked-group-improperly-obtained-openai-data) 和 [Financial Times](https://www.ft.com/content/a0dfedd1-5255-4fa9-8ccc-1fe01de87ea6) 报道称，DeepSeek 涉嫌利用 OpenAI 数据进行训练，引发了关于数据伦理的辩论，而一些人则将其斥为焦虑的竞争对手发起的抹黑行动。
-  **GPTs 在零宽空格字符上遇到麻烦**：社区发现使用不可见的[零宽空格](https://link.to.stackoverflow)（如 `httpXs://`）可以绕过 GPTs 中不希望出现的链接格式化，同时用户也报告 Custom GPTs 经常无法可靠地输出所有链接，引发了对用户内存处理的疑问。
-  **AI 的未来可能取决于 Grok3 和 O3-mini**：[传闻](https://x.com/btibor91/status/1884612786183135627)暗示 Grok3 和 O3-mini 将于 1 月发布，激发了对下一代推理能力的希望，而 [O3-mini](https://x.com/bindureddy/status/1884619428383633594) 承诺运行速度将达到 O1-mini 的 4 倍。

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 的大幅瘦身**：Unsloth AI 将 [**DeepSeek R1 1.58-bit**](https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit) 与 **OpenWebUI** 集成，体积从 720GB 缩减至 131GB，同时在 **160GB VRAM** 上保持约 140 tokens/sec 的速度。
   - 社区成员指出，**选择性层量化**是此次提速的关键，引发了关于**微调**的进一步讨论，并引用了 [Magpie-Align 的 250K CoT 数据集](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)。
- **显著的 CoT 收益**：参与者强调通过大型模型生成 **Chain-of-Thought** 样本来增强 **DeepSeek** 的推理能力，并参考了 [Magpie-Align 的数据集](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)。
   - 一些人担心，如果没有显式的推理数据进行训练，可能会降低逻辑能力，因此呼吁从大规模模型进行合成扩展。
- **Qwen2.5-VL 视觉探索**：成员们期待在本周末获得 **Qwen2.5-VL** 支持，旨在为增强的视觉语言任务扩展 **OCR** 功能。
   - 他们注意到与 *OpenWebUI* 在实时图像问答方面可能存在协同效应，这激发了对下一代 **OCR** 微调的乐观情绪。
- **异步联邦学习尝试**：一位成员展示了一篇 [异步联邦学习论文](https://arxiv.org/abs/2111.04877)，强调了并行训练模型的设备之间只需极少的协调。
   - 他们还分享了一个[幻灯片](https://docs.google.com/presentation/d/1KP1u_N5_zk9tuIXWfxyytC_YpfpEEoXcxEGzI3CwJ_w/edit?usp=sharing)，激发了关于在多个系统间扩展本地训练的讨论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 挑战 OpenAI**：社区将 **DeepSeek R1** 与 **OpenAI** 的模型在 **coding** 和 **creative** 任务上进行了并排测试，发现在某些条件下输出更具连贯性，但在涉及政治等敏感话题时也存在局限性。
   - 他们还分享了[这段关于“DeepSeek AI 揭露科技寡头数十亿美元骗局”的视频](https://youtu.be/e9XCbX40ijA)，强调了更广泛的 **censorship** 问题。
- **多模型意味着更多洞察**：成员建议并行查询多个 AI 系统，以绕过默认的内容过滤器或单一模型的不足，特别是针对有争议的查询。
   - 有些人将其称为一种 **ensemble AI** 形式，但也有人指出目前还没有官方框架可以无缝合并这些输出。
- **GPT 链接烦恼与记忆失误**：参与者发现了一个涉及不可见 **zero width space**（例如 `httpXs://`）的技巧，用于规避不需要的链接格式化，并引用了 [一篇 StackOverflow 帖子](https://link.to.stackoverflow)。
   - 他们还报告了 **Custom GPT** 无法可靠地输出所有链接，并指出 GPT 在用户记忆处理方面存在矛盾，引发了关于个人细节引用不完整的讨论。
- **o3-mini 挑战猫头鹰与棕榈树谜题**：一位成员专注于 **o3-mini** 是否能解决 **owl-palm tree riddle**，将其视为对推理能力的严肃测试。
   - 他们宣称 *“这是我唯一关心的 benchmark！”*，强调了单一谜题的表现如何引导模型对比。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek R1 在性价比对决中挑战 Qwen2.5**：社区成员在 [LM Studio](https://lmstudio.ai/docs/basics/import-model) 中对比了 **DeepSeek R1** 及其蒸馏变体与 **Qwen2.5** 在编程任务中的表现，权衡了预算限制和整体响应质量。他们还指出，可以通过 [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) 或 [bartowski builds](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF) 获取 **Qwen2.5**，并强调了价格与性能之间的相互作用。
   - 一位用户建议说 *“Qwen2.5 部署更简单，但牺牲了一些微调选项，”* 而其他人则称赞 **DeepSeek** 尽管对 **VRAM** 要求更高，但仍保持了较高的准确性。他们分享了 [gguf README 笔记](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 作为高级调优的参考。
- **LM Studio 加载困境**：多人在 **Mac** 设备上运行 [LM Studio](https://lmstudio.ai/docs/basics/import-model) 时遇到了 *模型加载失败* 的问题，认为最低硬件规格是主要症结所在。一些人建议切换高级设置或采用 Beta 版本，并参考了 [gguf 文档](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) 中的潜在修复方案。
   - 一位用户指出 *“除非调整并发设置，否则 GPU 内存限制可能会导致一切冻结”*。另一位用户建议频繁进行 [LM Studio Beta 频道的更新](https://lmstudio.ai/docs/basics/import-model) 以解决稳定性问题。
- **文档处理中的 RAG 难题**：用户讨论了 *LM Studio* 中 **RAG** 的可靠性，强调选择一个强大的模型对于高要求的特定领域任务至关重要。他们认为标准配置在处理专业问题时经常出错，暗示需要“GPT 级别”的解决方案或更精细的检索策略，尽管未提供直接参考。
   - 一位用户指出 *“如果模型没有足够的上下文，RAG 可能会让人感到困惑，”* 而其他人则建议针对领域密集型数据采用专门的检索解决方案。一些人建议探索更高级的分块（chunking）或嵌入（embeddings）以降低错误率。
- **内存带宽成为核心焦点**：参与者指出 **LLM** 性能很大程度上取决于内存带宽，并认为 **Mac** 在这方面不如 **A4000** 或 **3060** 等 **GPU**。他们补充说，将 **Threadripper** 或 **EPYC** **CPU** 与多个 **GPU** 配对可以更高效地处理 **DeepSeek R1 Distill-Qwen 70B** 等模型，但未给出直接链接。
   - 一位用户开玩笑说 *“即使使用 Threadripper CPU，你也跑不过内存带宽，”* 并引用了[这张 GPU 带宽表](https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix)。同时，其他人强调了更高 **VRAM** 与深度语言模型的协同作用。
- **CSV 混乱：LLM 与跨链交易**：一位用户寻求使用 **LLM** 方法来统一 **CSV** 交易格式，重点关注跨链数据的复杂性。响应者建议使用 **Python** 脚本以保证一致性和规模化，并暗示对于较大的数据集，仅依靠 **LLM** 可能会出错。
   - 一位社区成员调侃道 *“对于大型 CSV 合并，代码比 LLM token 更便宜，”* 强调了脚本在以数据为中心任务中的可靠性。另一位成员表示赞同，提到 **Python** 是获得稳定输出的首选工具。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen 2.5 Max 混淆**：社区讨论了 *Qwen 2.5 Max* 的开源性质，引用 [这条推文](https://x.com/alibaba_qwen/status/1884263157574820053) 得出结论：由于巨大的 **GPU** 需求，它**无法**完全用于本地使用。
   - 其他人探索了将 **Qwen 2.5 Max** 纳入编程工作流的方法，注意到了 [Hugging Face 上的 Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Max-Demo)，但对其高内存需求表示遗憾。
- **模型速度马拉松**：一些用户报告称 **hyperbolic** 的 **R1** 吞吐量较低，响应时间偶尔超过一分钟，输出速率约为每秒 12 个 token。
   - 他们检查了系统资源使用情况，并参考 [aider/benchmark README](https://github.com/Aider-AI/aider/blob/main/benchmark/README.md) 来识别瓶颈并改进性能指标。
- **Open-R1 备受 GitHub 关注**：一个名为 **open-r1** 的项目出现，通过 [此 GitHub 链接](https://github.com/huggingface/open-r1) 分享，暗示了 **R1** 模型潜在的开源方法。
   - 爱好者建议研究其架构和可能的应用，暗示它可能为大模型爱好者提供新的探索路径。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar 与 DeepSeek 赢得掌声**：**Sonar Reasoning** API 已发布，支持带有实时引用的 chain-of-thought。根据[官方说明](https://www.businesstoday.in/technology/news/story/perplexity-ai-increases-limit-for-daily-deepseek-r1-queries-on-its-platform-462421-2025-01-29)，**DeepSeek R1** 现已通过快速命令更新集成到 Perplexity Mac App 中，并托管在美国数据中心以保护隐私。
   - 社区成员报告了 **Sonar** 的一些格式拒绝问题，但赞扬了其实时搜索功能；同时，一些人质疑它使用的是 **R1 (671B)** 还是蒸馏模型，并要求提高透明度。
- **DeepSeek 每日额度大幅提升及与 O1 的竞争**：Perplexity 将 Pro 用户的 **DeepSeek R1** 每日查询限制提高到 **50** 次，免费用户提高到 **5** 次。CEO [Aravind Srinivas](https://x.com/aravsrinivas/status/1884708550795985382) 表示，随着容量的改善，将进一步扩大限制。
   - 一段 [YouTube 视频](https://www.youtube.com/embed/YjOGIs4sA50) 暗示 **DeepSeek R1** 可能会超越 OpenAI 的 **O1**，这激发了关于性能指标和 chain-of-thought 影响的讨论，反映了对推理质量的持续探讨。
- **阿里巴巴筹备新模型**：一位用户分享了关于阿里巴巴可能推出新 AI 模型的[链接](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg)，暗示了科技行业竞争格局的变化。
   - 社区成员讨论了其加剧市场竞争和加速 R&D 的潜力，强调了大规模模型如何重塑阿里巴巴的生态系统。
- **Java 23 到 Java 2 的转折**：从 **Java 23 SDK** 转向 **Java 2** 的举动引发了关于公共服务落后于私人采用的辩论，并引用了[现实世界的适配情况](https://www.perplexity.ai/search/updating-java-23-sdk-to-java-2-lDOVkVzrQ.OSUPQoRvn3gA)。
   - 参与者担心政府使用中的 QA 瓶颈，并质疑更快的推广是否能对抗机构惯性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **内存对 MoE 至关重要**：在 **Mixture-of-Experts** 的讨论中，参与者强调内存大小对 CPU 配置的性能至关重要，更高的带宽能提升 token 速度。
   - 他们分享了优化技巧，并指出在处理复杂负载时，**HPC-like** 的资源管理通常优于标准配置。
- **Nous 的资金蓬勃发展**：社区成员透露，**Nous Research** 依靠 VC 支持者、捐赠和少量的周边销售来支付计算费用。
   - 他们幽默地提到周边商品的收入虽然微薄，但仍是维持大规模 AI 项目运作的多渠道方案之一。
- **DeepSeek R1 在 Azure 首次亮相**：**DeepSeek R1** 模型已在 [Azure AI Foundry](https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/) 和 GitHub 上线，为开发者提供了即时可用性。
   - 社区成员对其进入超过 **1,800** 个 AI 模型之列表示欢迎，认为它是 Microsoft 产品线中稳健的企业级解决方案。
- **Olama：CLI 与 GUI 的对决**：虽然有人提议使用 **Olama** 运行 Mistral 或 Deepseek-distilled 等本地模型，但一些人不喜欢其对 CLI 的依赖，更倾向于可视化方式。
   - 其他人则建议那些想要更友好界面或不同许可协议的用户使用 **KoboldCPP** 或 **LM Studio**，在易用性与功能集之间进行权衡。
- **AoE：专家自行选择 Token**：一篇新[论文](https://arxiv.org/abs/2501.13074)介绍了 **Autonomy-of-Experts (AoE)**，其中模块利用内部激活来决定是否处理输入，从而绕过了常规的 router。
   - 在这种设置中，只有**排名靠前的专家**会继续处理，这可能会提高效率并超越传统的 MoE token 分配方式。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的 DeepSeek 困境**：用户哀叹 **Windsurf** 缺失 **DeepSeek R1** 集成，这促使他们威胁要转向 Cursor 以获得更好的 tool-calling 功能。
   - 一些人观察到 **DeepSeek** 在处理高效请求方面存在困难，这使得它与 Windsurf 的协同变得困难。
- **Sonnet LLM 的失误**：多位成员批评 **Sonnet LLM** 的编码可靠性不一致，称其 Prompt 理解能力有所下降。
   - 其他人要求更快的改进，指出次优的性能在不提高生产力的情况下消耗了额度（credits）。
- **Cascade 的混乱与代码质量下降**：一些人报告 **Cascade** 在修改文件时意外清除了上下文或生成错误，迫使他们进行手动重构（refactoring）。
   - 少数人仍看好 Cascade 的方法，敦促在编辑大型代码库时要谨慎，以避免重复失误。
- **Flex Credits 的迷雾**：新注册用户发现 **Flex credits** 的分配令人困惑，试用总额不明确，且对于有缺陷的输出没有简便的退款机制。
   - 几个人指向了 [Codeium Status](https://status.codeium.com) 以寻求可能的澄清，而其他人则鼓励直接联系支持团队。
- **Windsurf 性能与扩展设置**：成员们注意到 **Windsurf** 聊天的速度断断续续，并指出 VSCode 中的 **Codeium extension** 无法完全解析选定文本的问题。
   - 他们还提到了反复出现的登录失败，引用了与休眠语言服务器相关的 ‘Sign in failed’ 错误，以及引发成本担忧的 [Plans and Pricing Updates](https://codeium.com/blog/pricing-windsurf)。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek R1 的新进展**：在最近的一项举措中，[Chutes](https://chutes.ai/) 通过 [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1:free) 为 **DeepSeek R1** 提供了一个**免费端点（endpoint）**，增强了去中心化覆盖。这一补充为开发者提供了更多方式来体验 DeepSeek R1 的 671B 参数能力。
   - **OpenRouter** 强调 **DeepSeek R1** 的性能可与 [OpenAI o1](https://openrouter.ai/openai/o1) 媲美，推理时有 **37B** 激活参数。*一位用户总结道*，“尽管有开销，但它是一个不错的替代方案”，强调了该模型的开源推理 token。
- **Perplexity 优化 Sonar**：**Perplexity** 升级了 **Sonar**，提升了速度并降低了成本，详见 [sonar.perplexity.ai](https://sonar.perplexity.ai)。此次改进旨在优化大规模搜索任务并保持最低资源消耗。
   - 预告中的 **Sonar-Pro** 承诺提供更多功能，预计很快发布，引发了广泛期待。一些参与者支持这一路线，认为它能与 **DeepSeek** 模型产生更好的协同效应。
- **Sonar-Reasoning 表现出色**：基于 **DeepSeek** 引擎构建的 **Sonar-Reasoning** 专门用于高级搜索和基于逻辑的任务，如[此公告](https://discord.com/channels/1091220969173028894/1092729520181739581/1332050064624975953)所示。该模型旨在简化复杂查询的处理。
   - **OpenRouter** 提供了将网络搜索与 Sonar-Reasoning 结合的建议，承认了用户对集成设置的需求。*一位用户表示*，“搜索加上高级逻辑正是我们进行大数据工作所需要的。”
- **关于定价与性能的反馈激增**：多位成员对 **DeepSeek v3** 的波兰语等语言翻译表示担忧，理由是上下文不完整。他们还批评 **OpenRouter** 5% 的 API 费用过高。
   - 一些人遇到了空 token 输出和界面故障，要求更好的错误处理。其他人强调需要改进检索功能和可调节的使用限制。
- **对图像生成的呼声**：一些成员请求将 **DALL-E** 或 **Stability AI** 直接集成到 **OpenRouter** 中，希望能扩展平台的功能。他们认为视觉生成可以吸引更多参与者并拓宽使用场景。
   - 其他人注意到了与翻译功能的联系，建议进行潜在的多模态增强。虽然目前还没有确切的消息，但强烈的兴趣暗示了未来更大的可能性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 数据风波与数据库崩溃**：Wiz Research 发现了 **DeepLeak**，这是一个公开暴露的 ClickHouse 实例，泄露了**密钥**、内部聊天记录以及数据**外泄 (exfiltration)** 的开放路径（参见 [Tweet](https://x.com/wiz_io/status/1884707819737223591)）。
   - 另一份[严重漏洞报告](https://x.com/h4x0r_dz/status/1884566387349225598)进一步概述了可能的 API key 泄露，引发了立即修复的呼声。
- **R1 与 R1-Zero 之争**：社区分析认为 **R1-Zero** 在重要性上超越了 **R1**，并重点推荐了一篇关于这两个模型托管挑战的[深度文章](https://arcprize.org/blog/r1-zero-r1-results-analysis)。
   - 爱好者们对作为面向公众的旗舰模型 **R1** 表达了轻微的失望，称其为**“为了人类使用而进行了‘阉割’ (nerfed)”。**
- **Llama 4 重构与延迟**：传闻指出 **Llama 4** 正在从零开始重构，[这一说法](https://x.com/vibagor44145276/status/1884467879963418645)暗示了战略上的重大转向。
   - 像 Together 这样的合作伙伴收到的细节寥寥无几，这意味着发布时间将从之前预测的 2 月份推迟。
- **Grok 3 与 O3-mini 发布传闻**：有迹象表明 **Grok 3** 和 **O3-mini** 可能会在 1 月发布，尽管内部传闻指向可能会重新安排在典型的周四发布。
   - [Tibor Blaho 的更新](https://x.com/btibor91/status/1884612786183135627)提到了一种“思考型”模型方法，激发了人们对下一代推理功能的期待。
- **带有 MoE 和 MTP 的 DeepSeek v3**：**DeepSeek v3** 论文中跳过了 **Mixture-of-Experts (MoE)** 的辅助损失（auxiliary losses），这让读者感到惊讶，并引发了对训练设置的好奇（参见 [MoE LLMs](https://cameronrwolfe.substack.com/p/moe-llms)）。
   - 人们推测 **Multi-Token Prediction (MTP)** 提高了 Token 接受率，但许多推理框架仍缺乏对该方法的原生支持。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek 困境：Token 恐惧**：**DeepSeek** 反复出现因 Token 限制而无法生成代码的问题，导致用户因输出不完整而感到恼火；一位用户抱怨道：*“它一直在喋喋不休，然后因为 Token 限制无法生成代码。”*
   - 另一位用户引用了 [Ihtesham Haider 关于 ‘Qwen’ 盖过 DeepSeek 光芒的推文](https://x.com/ihteshamit/status/1884654075071127678)，声称 **Qwen** 在多项任务中击败了 **ChatGPT-o1** 和 **Claude Sonnet**。
- **Cursor IDE 灾难：更新后的混乱**：多名用户报告在最近的更新后出现了新的 **Cursor IDE** Bug，包括 Tab 补全失效、杂乱的 import 以及错误的 Markdown 输出，一位用户指出：*“Cursor 不再正确显示其 Markdown 输出。”*
   - 社区成员建议在 [Cursor 论坛](https://forum.cursor.com/t/upgrade-to-0-45-x-always-breaks-cursor/44688/8)报告问题，或查看 [Cursor 状态页面](https://status.cursor.com)了解已知的服务中断。
- **Claude 3.5 额度封锁**：许多人抱怨 **Claude 3.5** 的免费层级限制，该限制在 50 次慢速高级请求后会锁定使用，且没有冷却期的绕过方法。
   - 一位用户询问是否有缓解办法，但其他人确认一旦达到限制，**Claude 3.5** 将拒绝进一步的请求。
- **为 Cursor 众筹升级建议**：用户呼吁在 **Cursor** 中加入更多 **AI 模型**，特别是在 Agent 模式下，以增加开发者的选择并减少与 Token 相关的陷阱。
   - 一位用户在[一条推文](https://x.com/shaoruu/status/1884395102195548357)中提出了建议，询问人们最希望 **Cursor** 做出哪些改进。
- **Sonnet 3.5 订阅故障**：一位用户报告称 **Sonnet 3.5** 无法在他们的 Cursor 订阅中运行，但可以使用个人 API key 正常工作。
   - 社区引导他们前往 [关于 Sonnet 3.5 问题的 Cursor 论坛帖子](https://forum.cursor.com/t/sonnet-3-5-stops-working/46053) 进行 Bug 报告和寻求潜在修复。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Softmax 变革与 RL 困境**：提出了一种新的 **Softmax 变体**，旨在应对某些场景下的噪声准确率和次优学习问题，引起了寻求更好训练梯度的研究人员的兴趣。
   - 几位成员强调了对 **Deep RL** 的担忧，指出默认的 Softmax 可能会导致 **mode collapse**，并呼吁采用更灵活的方法。
- **DeepSeek 数据风波**：**DeepSeek** 使用 **2,048 块 Nvidia H800 GPU** 和 **PTX**，在两个月内训练了一个 **671B 参数** 的 Mixture-of-Experts 模型，据报道其效率比标准做法提升了 **10 倍**。
   - 与此同时，[Bloomberg](https://www.bloomberg.com/news/articles/2025-01-29/microsoft-probing-if-deepseek-linked-group-improperly-obtained-openai-data) 和 [Financial Times](https://www.ft.com/content/a0dfedd1-5255-4fa9-8ccc-1fe01de87ea6) 报道了关于 **DeepSeek** 不当使用 **OpenAI** 数据的指控，在 **Italy** 持续的审查背景下，一些人称之为“抹黑行为”。
- **Qwen2 VL 与 PydanticAI 推荐**：**Qwen2 VL** 在 **7B M1 芯片**上通过 **8K** 量化实现了极高的 Token 生成速度，令用户印象深刻，有人评论称其生成速度“快得惊人”。
   - 一段 **PydanticAI** 代码片段也引发了热议，展示了数据验证如何轻松地与基于 **GroqModel** 的 Agent 集成。
- **O3-mini 的巨大飞跃**：围绕即将推出的 **O3-mini** 展开了激烈讨论，该模型承诺运行速度是 **O1-mini** 的 **4 倍**，且性能可能超越 **R1**。
   - 一些人引用了[这条推文](https://x.com/bindureddy/status/1884619428383633594)作为证据，认为 **OpenAI** 凭借这种更快的模型可能会在美国市场获得显著优势。
- **Claude 3.5 的成本标签**：据报道，**Claude 3.5** 的训练成本高达数千万美元，凸显了下一代语言模型巨大的资金投入规模。
   - 社区成员认为这笔金额证明了雄心勃勃的 AI 开发需要巨额资金和广泛的计算资源。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Mordechai 的动力：神经科学书籍与 Kickstarter**: Mordechai Rorvig 展示了**他的神经科学书籍项目**，重点关注大规模脑功能、情感 AI 处理以及来自 [Kickstarter 众筹项目](https://www.kickstarter.com/projects/45417589/ai-how-we-got-herea-neuroscience-perspective)的潜在扩展。他征求了关于 **deep learning 架构**与生物认知之间协同作用的反馈，希望完善针对高级 AI 系统提出的设计特性。
   - 讨论涉及了这些想法如何为改进**情感智能 (emotional intelligence)** 模型提供信息，几位参与者对神经科学与现代 AI 研究相结合的视角表示赞赏。
- **Min-P 魔法：文本生成的新花样**: 新引入的 **min-p** 采样技术根据模型置信度调整阈值，旨在提高文本质量和多样性，并参考了 [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM...](https://openreview.net/forum?id=FBkpCyujtS)。它引发了关于 Token 限制是否会阻碍探索的疑问，特别是与 **top-p** 方法相比。
   - 一些参与者担心过度限制模型输出，而另一些人则认为 min-p 是管理不同任务中 **perplexity** 的宝贵方法。
- **SFT vs. RL：关于泛化的大辩论**: 成员们剖析了 **'SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training'** ([链接](https://arxiv.org/abs/2501.17161))，讨论了如何结合 **SFT** 的快速模式应用和 **RL** 的更广泛解空间搜索，以实现更强的泛化能力。他们指出 SFT 可以可靠地应用训练数据，而 RL 似乎能培养更开放的行为。
   - 一些人建议 RL 能够实现涌现式的问题解决，但其他人强调了 SFT 在某些任务中的一致性，指出平衡这两种方法是下一步的策略。
- **Sparse Autoencoders：种子驱动的传奇**: 一篇名为 [Sparse Autoencoders Trained on the Same Data Learn Different Features](https://arxiv.org/abs/2501.16615) 的新论文报告称，在不同种子下训练的 **SAEs** 仅共享 **30%** 的学习特征，这引发了对特征稳定性的担忧。作者质疑在没有额外约束的情况下，这些表示对于可解释性任务是否仍然可靠。
   - 该小组提议在多个种子上进行并行训练以对齐输出，而一些人则反驳称，替代的正则化或架构选择可能会提供更一致的结果。
- **聚焦 Fastfood：快速核扩展**: 工程师们重新审视了来自 [Fastfood: Approximate Kernel Expansions in Loglinear Time](https://arxiv.org/abs/1408.3060) 的 **Fastfood**，利用 **Hadamard** 操作实现更快的核扩展和更小的内存占用。初步测试显示在大规模计算中减少了开销，并激发了高级 **LLM** 开发者的兴趣。
   - 一些参与者探索将 Fastfood 集成到大型网络中，希望在保持准确性的同时遏制存储需求，尽管一些人警告需要更多的实际测试。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Direct Storage 收益与权重压缩传闻**：在 #general 频道中，成员们探讨了利用 **GPU Direct Storage** 进行高效的 PCIe 点对点（peer-to-peer）数据传输，并报告了将权重从 4.7GB 压缩至 3.7GB 的部分成功案例。
   - 他们还考虑了并行友好的压缩方案和 **memory snapshotting**（内存快照），引用了 [NVIDIA/gdrcopy](https://github.com/NVIDIA/gdrcopy) 和 [gpudirect/libgdsync](https://github.com/gpudirect/libgdsync) 来减少开销，并将 safetensors 直接加载到 VRAM 中。
- **Blackwell 架构动态与 CUDA 类型双关**：在 #cuda 频道中，有传言称 **RTX Blackwell** 架构相比 4090 将提升 **27%** 的 FP16/32 吞吐量，而第 5 代 Tensor Cores 在消费级显卡上的变化微乎其微，详见 [NVIDIA 官方页面](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)。
   - 他们还强调在 CUDA 中使用 **memcpy()** 进行类型双关（type punning）和严格的内存对齐，以避免未定义行为，并可能获得寄存器级的优化。
- **Lean Llama：极简训练代码问世**：在 #cool-links 和 #self-promotion 频道中，成员们分享了一个位于 [speed_llama3](https://github.com/ahxt/speed_llama3) 的 **Llama 极简训练代码库**，旨在追求极致效率。
   - 他们展示了针对大语言模型的 FP4 方案，并讨论了块大小量化（block-size quantization）策略以优化性能。
- **Thunderkitten 与 DSM 潜力**：一位开发者提议在 Thunderkitten 中增加对 **Distributed Shared Memory (DSM)** 硬件特性的支持，建议使用持久化内核（persistent kernels）以实现更好的数据复用。
   - 他们还强调了线程块到 SM（threadblock-to-SM）调度带来的性能提升，这得益于其在 NV 工作 2.5 年的相关背景。
- **Arc-AGI-2：象棋谜题与动态推理**：#arc-agi-2 的成员讨论了推理任务的 **dynamic evaluation**（动态评估），目前正在开发简化的象棋谜题（如两步杀）。
   - 他们还提议生成“维基百科游戏”解决方案并训练解释器模型以获得更深层的洞察，同时参考了 **vLLM** 等推理引擎来实现流式批处理。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 与 Forge 之争**：用户们争论 **ComfyUI** 是否过于复杂，并推荐了 [Forge 的 GitHub 仓库](https://github.com/lllyasviel/stable-diffusion-webui-forge/) 作为更直接的替代方案。
   - 一些人欣赏 **ComfyUI** 先进的工作流流水线特性，而另一些人则希望有一个极简界面以便快速设置。
- **图像生成工具与工作流**：参与者讨论了写实角色生成等任务的 **workflows**（工作流），重点介绍了使用 **autismmix** 模型进行奇幻题材生成的尝试。
   - 他们提到了 [Kolors Virtual Try-On](https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On) 作为一个案例，指出许多人希望通过更简单的菜单获得稳定的结果。
- **Stable Diffusion 的 Python 问题**：一位用户在安装 **Stable Diffusion** 时遇到了 **Python 错误**，引发了关于依赖项的调试建议。
   - 他们还分享了一个[奇特的链接](https://james4ever0.github.io/Cybergod__God_is_in_your_computer.html)，引起了人们对潜在环境配置错误的关注。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 的导出/导入功能改进**：从现在起，**Bolt** 保证所有导入和导出功能均能正常运行，包括之前缺失的默认导出，如[这条推文](https://x.com/boltdotnew/status/1884634733386006980)所述。
   - 此次更新特别确保了对 **'export default'** 的支持，提供了更流畅的编码环境，并对所有项目带来了即时改进。
- **后端选择与 Firebase 挑战**：开发者们寻求关于推荐后端解决方案的指导，希望能有稳健的配置来满足项目需求。
   - 另一位成员描述了 **Firebase** 陡峭的学习曲线，但指出通过反复的动手实践，熟悉度正在提高。
- **Bolt 中的 Token 纠纷与服务故障**：用户对频繁调试过程中过快的 Token 消耗表示担忧，强调了长提示词（Prompts）和复杂项目的影响。
   - 一些用户还报告了 **Bolt** 的**服务器错误**和可用性故障，对平台的稳定性表示沮丧。
- **GitHub OAuth 与域名困境**：要切换与 Stackblitz 关联的 GitHub 账号，用户必须在 GitHub 中撤销权限并删除旧的 Stackblitz 账号，目前没有其他变通方法。
   - 同时，关于在 Supabase 和 Netlify 中使用**自定义域名**的问题揭示了根域名 CNAME 记录冲突，尽管 Supabase 在没有自定义域名的情况下也能工作，但使用自定义域名对邮件清晰度更有利。

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Goose 取得进展**：社区成员称赞了 **Goose client** 的 CLI 导向以及与 **MCP servers** 的协同作用，涵盖了使用方法和更优的集成流程。
   - 他们还指出了 **token usage** 的限制，并参考了 [michaelneale/deepseek-r1-goose](https://ollama.com/michaelneale/deepseek-r1-goose) 来寻找解决速率限制的方法。
- **Sheets 集成备受关注**：一位开发者展示了一个 **MCP server**，可以从 Google Drive 读取并编辑 Google Sheets，项目展示在 [mcp-gdrive](https://github.com/isaacphi/mcp-gdrive) 中。
   - 他们注意到图表格式化功能有限，但认为通过进一步探索，该功能具有更广泛的应用潜力。
- **DeepSeek Distill 表现强劲**：据 [DeepSeek model info](https://glama.ai/models/deepseek-r1-distill-qwen-32b) 报道，**DeepSeek-R1-Distill-Qwen-32B** 在多个基准测试中超越了 **OpenAI-o1-mini**。
   - 成员们反映使用 **Kluster.ai** 将这些模型集成到 **MCP** 中效果更流畅，并强调了其他替代方案。
- **mcp-agent 登上 Show HN 榜首**：**mcp-agent** 框架在 **Show HN** 荣获第一名，重点展示了使用 Model Context Protocol 构建 Agent 的劳动力友好型模式。
   - 位于 [lastmile-ai/mcp-agent](https://github.com/lastmile-ai/mcp-agent) 的仓库收集了用于未来改进的反馈。
- **lüm AI 支持心理健康**：心理健康伴侣 **lüm**（详见 [lüm - Your AI Companion](https://lum.frgmt.xyz)）推出了一种隐私优先的实践方法。
   - 其开发者呼吁社区分享关于未来心理辅助功能的想法，以对齐心理健康应用的需求。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **蒸馏版 DeepSeek R1 取得进展**：社区成员对 [bartowski's DeepSeek-R1-Distill-Llama-8B-GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF) 进行了报道，强调 **8b distill models** 与更重的 70b 量化配置相比，表现出人意料地强劲。
   - 他们指出，虽然 R1 蒸馏版看起来很能打，但许多人仍希望看到更大规模的模型选项，并参考了[一段解释 DeepSeek R1 概念的视频](https://youtu.be/r3TpcHebtxM)。
- **CUDA 与 CPU 协作提升速度**：参与者讨论了在 **CUDA** 上运行 **DeepSeek models**，在本地任务中使用 **q8_0** 时，CPU 速度通常能达到 5t/s。
   - 他们描述了为实现更高吞吐量而进行的持续改进，并参考了 [GPT4All 上的一项公开 PR](https://github.com/nomic-ai/gpt4all/pull/3431) 以增强本地推理能力。
- **对 LM Studio 的疑虑与模板调整**：贡献者对 **LM Studio** 表示迟疑，原因是其闭源特性以及与 DeepSeek 不确定的兼容性。
   - 他们建议优化 **template strategies** 和高级指令，以提升 R1 蒸馏模型的 prompt 输出质量。
- **对新 R1 版本的乐观态度**：多位成员期待 **32b R1 distills**，希望这些即将发布的版本能解决本地环境下的性能差距。
   - 他们引用了 [unsloth 的 8B Distill LLaMA 仓库](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) 作为持续改进和近期潜力的范例。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 的文件大小限制困扰**：用户对加载沉重的生态工程教科书和多个文档表示担忧，称在查询时会出现**大海捞针**的情况。他们引用了 [NotebookLM Help](https://support.google.com/notebooklm/answer/14278184) 中关于最大文件限制的说明，并建议将文档切分为更小的部分以提高清晰度。
   - 此外，由于 NotebookLM 不提供已上传源文件的直接下载，用户对仅在 NotebookLM 上存储学术资料表示担忧，并建议在 **Google Drive** 中保留副本。
- **笔记转换提升效率**：一位用户强调了将笔记转换为源文件的技巧，这使得**比较非结构化调查数据**变得更加容易。他们分享道，在交叉引用多个数据集时，对参考文献进行总结和重新格式化可以提高清晰度。
   - 然而，一些人质疑这种方法是否多余，指出笔记本质上就是现有源内容的镜像。
- **“Add New” 按钮消失**：成员们在 “Add New” 按钮消失时感到**困惑**，怀疑 NotebookLM 的使用量可能存在上限。他们建议咨询内置的自查询功能，以发现任何隐藏的账户或功能限制。
   - 虽然出现了指向 [NotebookLM Plus Upgrade](https://support.google.com/notebooklm/answer/15678219) 的链接，但按钮消失的确切原因仍不确定。
- **LinkedIn 限制与 PDF 应对方案**：一位用户在将 LinkedIn 个人资料添加为源文件时遇到了问题，可能是由于爬虫限制。提议的权宜之计是将页面**导出**为 PDF，然后将其上传到 NotebookLM。
   - 这种策略在处理限制直接数据抓取的**网站**时确保了更好的可靠性。
- **播客计划与 API 愿景**：人们在 NotebookLM 中尝试**生成更长时间的播客**，目标是 30 分钟或更长的脚本。他们就确保稳定的音频输出和可能的集成方案交换了意见。
   - 还有关于连接 NotebookLM 与 Salesforce 的 API 的咨询，但该功能目前**没有预估的发布日期**。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek 的 R1-Zero 势头强劲**：在查看 [R1-Zero 和 R1 结果](https://arcprize.org/blog/r1-zero-r1-results-analysis)后，**R1-Zero** 在数学和编程方面取得了相当的性能，这表明可能不需要大规模的 SFT。
   - 社区成员最初对不连贯性表示担忧，但测试报告显示 R1-Zero 的逻辑输出**没有重大缺陷**。
- **华为 910C 助力 DeepSeek**：如[此帖](https://x.com/Dorialexander/status/1884167945280278857)所述，DeepSeek 已转向使用**华为 910C 芯片**进行推理，这引发了关于其与 Nvidia 硬件相比潜在权衡的讨论。
   - 参与者讨论了华为芯片的内存限制，一些人不确定它们是否能在不损失性能的情况下处理大规模训练。
- **OpenAI 的 ChatGPT Pro 营收超过 Enterprise 版**：根据[这条推文](https://x.com/steph_palazzolo/status/1884364666622927023)，**OpenAI 每月 200 美元的 ChatGPT Pro** 在收入上超过了 **ChatGPT Enterprise**，反映了强劲的订阅增长。
   - 然而，评论者认为**企业交易**可能会亏损，从而对长期模式提出了质疑。
- **Sourcegraph 首次推出 Enterprise Agent**：Sourcegraph 推出了一种新的 **Enterprise Agent** 编程解决方案来对抗 Windsurf，并计划在 **AIENYC** 上通过专门的预订案例研究进行讨论。
   - 社区讨论强调了该产品旨在使 AI 辅助编程对于大规模部署更加易用且相关。
- **微软 Copilot 的推广备受指责**：观察者批评 [Microsoft 365 Copilot 的发布](https://www.zdnet.com/home-and-office/work-life/the-microsoft-365-copilot-launch-was-a-total-disaster/)执行不力，引起了新用户的困惑。
   - 评论指出营销失误和策略不明，暗示微软的 AI 服务存在**身份危机**。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-r-plus 混淆与重复**：一些用户报告 **command-r-plus** 的回复较短，但在切换到 [command-r-plus-08-2024](https://docs.cohere.com) 处理问题解决任务时，得到了详尽（但有重复）的回复。
   - 支持团队澄清 **command-r-plus** 自 9 月以来仍指向 **-04-2024**，并建议分享代码片段，同时推荐升级到 [command-r7b-12-2024](https://docs.cohere.com) 以获得更稳健的输出。
- **从上下文到严格的安全模式**：新的 **Safety Modes**（**CONTEXTUAL**、**STRICT** 和 **NONE**）随 [Cohere 文档](https://docs.cohere.com/v1/docs/safety-modes) 一起发布，用于对新模型进行精细的输出限制。
   - 用户称赞 **CONTEXTUAL** 适用于创意或教育任务，**STRICT** 适用于强力防护栏，而切换到 **NONE** 则完全禁用安全防护以获取不受限的内容。
- **Rerveting Efforts 提示词与 Aya 8b 的进展**：开发者在 **Aya 8b** 上测试了 **Rerveting Efforts Reasoning Prompt**，虽然遇到了设置障碍，但发现了有前景的逻辑。
   - 他们请求关于其 *“隐藏潜力”* 的反馈，并计划在进行中的 *图像分析* 实验中进一步完善它。
- **Markdown 故障与剪贴板保存**：一位用户几乎丢失了一个关键提示词，但通过 **Windows + V** 将其救回，突显了高级剪贴板功能的重要性。
   - 同时，**Markdown** 中的格式问题引发了挫败感，促使大家分享简化项目工作流中 Markdown 使用的技巧和窍门。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书惊喜与无黑客松**：MOOC 讨论确认了 **非学生证书**，宣布本学期 **没有黑客松**，并明确了应用赛道（application track）项目团队为每组 **3-4 名学生**。
   - 与会者了解到公开课程与 Berkeley 的原始课程一致，并被建议关注即将发布的公告以获取最终细节。
- **LLM Agents 的讲座链接与资源**：成员们分享了 **CS 194/294-280** 的新 [讲座转录稿](https://docs.google.com/document/d/1FquWB_ovVAmTZJczrI8rbEjOEDXKFVwryEYw3K2pKV8/edit) 和官方 [幻灯片](https://llmagents-learning.org/sp25)，以方便深入学习。
   - 他们提议将这些资源扩展到所有讲座，强调了小组对开放协作的热情。
- **Stake 空投引发关注**：**Stake Airdrop** 活动开始，鼓励参与者在活动结束前尽早在 [stakeair-drop.com](https://stakeair-drop.com/) 领取奖励。
   - 爱好者们强调了其 **限时** 利益，敦促早期质押者最大化收益。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 LSP 之谜**：一位用户在运行 `magic run mojo-lsp-server --help` 时发现了隐藏的 **LLVM flags**，但目前没有任何可查阅的文档。
   - 另一位用户建议在 [GitHub](https://github.com) 上提交 issue，以便 **Mojo tooling team** 处理或隐藏这些内部参数。
- **TIOBE 谈论 Mojo**：Mojo 在 TIOBE 指数中被提及，其 CEO 预测到 **2025 年** 排名将接近前 20。
   - 社区成员表示兴奋，将其视为开发者兴趣加速增长的信号。
- **VS Code 折叠问答**：有人询问 **Mojo 的 VS Code 扩展** 是否支持代码折叠，或者是否计划很快添加该功能。
   - 一位用户建议将该查询移至相关频道，并指出这可能需要扩展维护者的反馈。
- **Mojo 路线图传闻**：随着 2025 年的临近，社区成员请求一份 **更新的 Mojo 路线图**。
   - 他们强调需要明确该语言后续开发的具体步骤和清晰度。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **办公时间与香蕉面包盛宴**：Torchtune 将于下周四 **13:30 US ET** 举办开放办公时间，讨论即将推出的功能并解决库相关问题，活动链接见[此处](https://discord.gg/2ecxr4TY?event=1334167213048860702)。
   - 与会者可以在交流期间享用**著名的香蕉面包**，这有望让大家保持高涨的热情。
- **指标混乱：DPO 设备聚合**：社区成员询问 **DPO 指标** 如何跨设备合并，并建议使用 `dist.all_reduce` 以获得更好的一致性，参考 [issue #2307](https://github.com/pytorch/torchtune/issues/2307)。
   - 他们计划很快提交一个 **PR**，以统一多台机器上的指标，旨在改进 DPO 验证。
- **损失归一化：缺失的关键要素**：人们注意到 DPO 实现中**没有包含损失归一化**，并指出 `lora_dpo_distributed` 和 `full_finetune_distributed` 训练方案（recipes）之间存在差异。
   - 他们计划探索快速修复方案，成员们提议协调调试工作。
- **Imagen 还是 Chatbot？困惑的询问**：出现了一个关于 **Imagen** 或 **Image2Txt** 的问题，但最终焦点转向了 **chatbot** 功能。
   - 询问者撤回了原始查询，最终认定对话仍以 chatbot 为中心。



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **多轮 KTO 之谜**：一位成员询问了**多轮 KTO** 的进展，但未得到更新。
   - 他们的提问引发了关于 KTO 下一步行动的推测，但对话并未产生任何明确计划。
- **RLHF 新人被重新分配**：**Nanobitz** 确认一名为 **RLHF** 加入的新成员被指派到了另一个不同的 PR。
   - 这一变动让希望在该项目中立即参与 RLHF 工作的成员感到失望。
- **NeurIPS 论文正在撰写中**：一位成员宣布计划今年提交一篇 **NeurIPS 论文**，表明正在认真推进研究成果的发表。
   - 他们报告称，这项工作可能会受益于即将开展的与 KTO 项目的研究协同。
- **三月截止日期临近**：同一位成员强调相关模型将于 **3 月**到期，引发了对能否按时完成里程碑的担忧。
   - 他们担心任何延误都可能破坏计划中的实验并阻碍进度。
- **Axolotl 焦虑**：一位成员警告说，**Axolotl** 的使用挑战可能会危及项目的 KTO 愿景。
   - 他们建议及时解决 Axolotl 的问题，以避免中断并保持工作流程正常运行。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **ScrapeGraph 与 LlamaIndex 联手实现快速网页策展**：将 [ScrapeGraph AI](https://twitter.com/llama_index/status/1884415138192937344) 与 **LlamaIndex** 集成，可以快速从网站提取非结构化数据，为高效的网页抓取流程提供动力。
   - 这一方法在 Twitter 上受到关注，展示了 AI Agent 如何以极低的开销处理重复的数据收集任务。
- **LlamaIndex 通过视觉效果增强财务报告**：一份新指南展示了如何通过 [LlamaIndex](https://twitter.com/llama_index/status/1884642320181830098) 混合 PDF 中的文本和视觉效果，生成多模态财务报表。
   - 这种策略有助于团队在单一流程中处理文本分析和基于图像的元素，提升财务任务的洞察力。
- **LlamaCloud 的变化引发了关于候补名单的疑问**：GUI 中缺失的 **Index** 按钮引发了关于仅限受邀参加的 **LlamaCloud** 计划的疑问，成员可以通过长度不明的候补名单加入。
   - 其他人注意到 **Confluence** 选项呈灰色，这意味着某些数据源可能需要 **Premium** 会员资格，尽管具体条件尚不明确。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Databricks 与 Featureform 助力 MLOps**：1 月 30 日太平洋时间上午 8 点的 MLOps 研讨会将由 **Simba Khadder** 讲解如何在 **Databricks** 上构建特征存储（feature store）。
   - 与会者将学习 [Featureform](https://buff.ly/40Ej4Z6) 的集成以及 **Unity Catalog** 的使用技巧，最后设有问答环节。
- **对 AI 进军开发角色的质疑**：一位参与者反驳了 **Zuck** 关于 AI 可以取代中级开发者的说法，指出该职业远未消亡。
   - 其他人指出了 **AI wrapper** 的持续增长，加剧了关于 AI 是否真的威胁到开发者岗位的讨论。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **自动微分告别手动 Prompting**：题为 [Auto-Differentiating Any LLM Workflow](https://arxiv.org/pdf/2501.16673) 的论文强调了本地语言模型工作流中的**自动微分 (auto-differentiation)** 如何消除手动 Prompting，从而实现更快的迭代过程。
   - 作者指出，通过消除 LLM 交互中的重复指令，**自动化**驱动了更高效的生成周期。
- **转向自动化的 LLM 交互**：论文断言，自动微分通过自动化 **LLM usage** 中的复杂步骤，显著提升了用户体验。
   - 社区成员预计这将大幅减轻认知负荷，并称其为在日常任务中实现 **smooth** LLM 集成迈出的一步。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Goose 凭借透明度崭露头角**：可以在[这里](https://block.github.io/goose/)找到 **Goose** Agent，它在本地运行，同时提供与 **MCP** 服务器或 API 的连接，将直接控制权交给开发者。
   - 用户称赞其对调试和部署任务的**自主 (autonomous)** 处理，减轻了工程团队的负担。
- **工程师庆祝 Goose 的自主性**：一位开发者表示，使用 **Goose** 感觉就像《壮志凌云》中的 *Maverick*，享受着有趣且高效的工作流。
   - 他们分享了一个成功案例：通过简单地指示 Goose 更新对象并运行测试，为 **API** 测试生成**伪数据 (fake data)**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 引入交互式分支教学**：一名成员提议构建一个类似于 [Learn Git Branching](https://learngitbranching.js.org/) 的工具，通过分支步骤谜题来教授 **Tinygrad** 基础知识。
   - 他们还引用了来自 [tinygrad-tensor-puzzles](https://github.com/obadakhalili/tinygrad-tensor-puzzles) 的谜题，强调了简短的挑战如何保持学习者的参与度。
- **关注结构化的 Tinygrad 代码架构**：参与者强调 **Tinygrad** 受益于组织良好的代码布局，建议使用基于谜题的模块来减少困惑。
   - 他们指出，对 **Tinygrad** 内部机制的系统性概述可以加强技能构建，并激发开发者更多的好奇心。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **来自 spirit_from_germany 的日常问候**：他们只是问了句 *'最近怎么样？'*，但没有讨论任何 AI 或技术细节。
   - 此处未引入新的对话点或对 AI 项目的引用。
- **无额外 AI 讨论**：该问候之后没有进一步的回复或关于 LLM 或 AI 进展的展开。
   - 因此，没有新的工具、基准测试或模型发布可供总结。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1333889405705912402)** (584 messages🔥🔥🔥): 

> `Unsloth AI 性能与功能, 深度学习模型训练, 强化学习 (Reinforcement Learning) 进展, 使用合成数据集微调模型, 用于高效建模的动态量化 (Dynamic quantization)`

- **Unsloth AI 将 R1 1.58-bit 集成到 OpenWebUI**：Unsloth AI 已成功将 DeepSeek-R1 的 1.58-bit 版本实现到 OpenWebUI 中，将模型大小从 720GB 缩减至仅 131GB。
   - 得益于选择性层量化（selective layer quantization），该模型在使用 **160GB VRAM** 时可实现约 **140 tokens/sec** 的快速推理速率。
- **通过微调生成推理能力的挑战**：用户对在没有包含 Chain of Thought 示例的数据集的情况下微调 DeepSeek 的可行性表示担忧，因为这可能会削弱其推理能力。
   - 建议是创建包含推理过程的合成数据集（synthetic datasets）以辅助模型微调，利用更大的模型来生成推理输出。
- **模型加载与配置问题**：一位用户在 Unsloth 自动加载微调模型（fine-tuned model）而非基础模型（base model）时遇到困难，导致测试期间出现混淆。
   - 这归因于模型配置中的命名规范，凸显了对模型加载源进行清晰沟通的必要性。
- **强化学习（Reinforcement Learning）训练方法的探索**：社区讨论了将 GRPO 训练与现有方法集成的方案，部分用户正在尝试使用 DPO 进行优化。
   - 有效的训练需要一个奖励模型（reward model），重点在于理解如何制定策略（policies）以改善模型行为。
- **利用现有数据集进行高级训练**：人们有兴趣利用 Wikimedia 数据集来训练 Mistral 模型，尽管有人对数据格式提出了担忧。
   - 对话强调了清晰的结构化和数据集准备对于实现有效训练结果的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenWebUI/status/1884719609552752801">Open WebUI (@OpenWebUI) 的推文</a>：🚀 感谢 @UnslothAI，你现在可以在 Open WebUI 上通过 llama.cpp 运行 1.58-bit DeepSeek-R1（非蒸馏版本）了！💻⚡️（已在 M4 Max，128GB RAM 上测试）📝 详情请参阅他们的博客文章：htt...</li><li><a href="https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit">SIGJNF/deepseek-r1-671b-1.58bit</a>：Unsloth 的 DeepSeek-R1 1.58-bit，我刚刚完成了合并并上传到这里。这是完整的 671b 模型，尽管是经过 1.58-bit 动态量化的。</li><li><a href="https://tenor.com/view/cat-wizard-meme-funny-gif-3870502440791733376">猫巫师 GIF - 猫巫师迷因 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B?row=1">Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/estrogen/DeepSeekMoE-3B">estrogen/DeepSeekMoE-3B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Kukedlc/Qwen2-1.5B-Spanish-1.0">Kukedlc/Qwen2-1.5B-Spanish-1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/DevQuasar/DevQuasar-R1-Uncensored-Llama-8B">DevQuasar/DevQuasar-R1-Uncensored-Llama-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/estrogen/DeepSeekMoE-3B/tree/main">estrogen/DeepSeekMoE-3B (main 分支)</a>：未找到描述</li><li><a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal">GitHub - EvolvingLMMs-Lab/open-r1-multimodal: 为 open-r1 添加多模态模型训练的分支</a>：为 open-r1 添加多模态模型训练的分支 - EvolvingLMMs-Lab/open-r1-multimodal</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B">Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1ic9x8z/you_can_now_run_dee">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1ic9x8z/you_can_now_run_deepseekr1_on_your_own_local/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：在 Ollama 上本地运行自定义个人助手（类似 ChatGPT）的初学者指南</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course：关于对齐 smol 模型的课程。</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建账户来为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">模型与定价 | DeepSeek API 文档</a>：下列价格以每 1M tokens 为单位。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总计...</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">奖励建模 - DPO, ORPO & KTO | Unsloth 文档</a>：要在 Unsloth 中使用 DPO, ORPO 或 KTO，请遵循以下步骤：</li><li><a href="https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/finetune/finetune.py">DeepSeek-MoE/finetune/finetune.py (main 分支) · deepseek-ai/DeepSeek-MoE</a>：DeepSeekMoE：在混合专家（Mixture-of-Experts）语言模型中实现极致的专家专业化 - deepseek-ai/DeepSeek-MoE</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 需求 | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1333937864634798141)** (24 条消息🔥): 

> `Federated Learning, LaMDA 自我意识主张, 意识与感知 (Consciousness and Sentience), AI 角色扮演, Deepseek 在职场中的使用` 


- **探索异步 Federated Learning**：一名成员展示了一篇关于 [Federated Learning](https://arxiv.org/abs/2111.04877) 的论文，讨论了设备如何异步训练模型，并引用了移动端键盘自动补全等应用场景。
   - 他们分享了一个 [幻灯片](https://docs.google.com/presentation/d/1KP1u_N5_zk9tuIXWfxyytC_YpfpEEoXcxEGzI3CwJ_w/edit?usp=sharing) 以突出演示中的重要见解。
- **关于 LaMDA 自我意识的讨论**：针对一名 Google 工程师关于 LaMDA 具有自我意识的主张引发了关注，引用中 LaMDA 暗示它有时会感到快乐或悲伤，这引发了辩论。
   - 成员们对该工程师的可信度开起了玩笑，其中一人认为 LaMDA 的能力更像是模仿角色扮演，而非真正的自我意识。
- **辩论意识的本质**：成员们讨论了意识的复杂性，以及是否有任何 LLM 具有自循环连接（self-recurring connections），其中一人表示它可能和某些人类一样具有感知力（即“并不怎么有”）。
   - 讨论中包含了关于定义意识之难的幽默言论，并暗示意识产生于极其复杂的系统。
- **关于 Deepseek 的 CISO 邮件**：一名成员幽默地询问其他人是否收到了首席信息安全官（CISO）建议不要在工作中使用 Deepseek 的邮件，对其安全性表示疑虑。
   - 另一名成员指责该讨论偏离主题，建议在 Stable Diffusion 等其他专业服务器上进行相关活动。
- **博士级别的虚数探索**：一名成员寻求帮助以在博士级别理解虚数，回忆起学校学到的关于 'i'（-1 的平方根）的基础知识。
   - 这引发了一条评论，调侃蒸馏模型（distilled models）表现得像普通的大学辍学者，强调了对复杂概念理解的挣扎。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.scientificamerican.com/article/google-engineer-claims-ai-chatbot-is-sentient-why-that-matters/">Google Engineer Claims AI Chatbot Is Sentient: Why That Matters</a>：人工智能是否可能具有感知力？</li><li><a href="https://docs.google.com/presentation/d/1KP1u_N5_zk9tuIXWfxyytC_YpfpEEoXcxEGzI3CwJ_w/edit?usp=sharing">PAPAYA: PRACTICAL, PRIVATE, AND SCALABLE FEDERATED LEARNING</a>：PAPAYA：实用、私密且可扩展的 Federated Learning
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1333889346541060096)** (131 条消息🔥🔥): 

> `DeepSeek R1 模型处理、模型训练问题与优化、Qwen2.5-VL 支持更新、Ollama 和 llama.cpp 功能、在各种硬件上运行模型` 


- **DeepSeek R1 内存需求问题**：用户报告称，尽管合并了权重，运行 DeepSeek R1 模型仍需要 132 GiB 的 RAM，这促使一些人考虑使用 llama.cpp 以获得更好的性能。
   - 一位用户确认成功合并了权重，但仍遇到性能限制。
- **模型训练 CUDA 显存错误**：多位用户在尝试于 4070 笔记本 GPU 等硬件上训练模型时遇到了 `Cuda is out of memory` 错误，引发了对 batch sizes 的关注。
   - 讨论集中在如何在特定硬件配置上实现较小模型的训练。
- **即将支持 Qwen2.5-VL**：社区成员热切期待 Qwen2.5-VL 支持的发布，预计将在本周末可用。
   - 这一即将到来的支持引发了热烈反响，特别是对于对 OCR 微调项目感兴趣的用户。
- **Ollama 的功能与磁盘卸载 (disk offloading)**：关于 Ollama 将模型数据卸载到磁盘的能力存在讨论，部分用户对该功能表示不确定。
   - 针对不同操作系统（即 Linux 和 Mac）的卸载功能提供了说明。
- **参数操作与模型效率**：用户讨论了通过关注特定语言数据和高效训练技术来操作并可能减小模型尺寸的方法。
   - 有关于在有限数据集上重新训练模型以增强性能和效率的可行性咨询。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit">SIGJNF/deepseek-r1-671b-1.58bit</a>: Unsloth 的 DeepSeek-R1 1.58-bit，我刚刚合并并上传到了这里。这是完整的 671b 模型，尽管是动态量化到 1.58-bit 的。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md">ollama/docs/modelfile.md at main · ollama/ollama</a>: 开始运行 Llama 3.3, DeepSeek-R1, Phi-4, Gemma 2 以及其他大型语言模型。 - ollama/ollama</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit 动态 GGUF 版本。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit">unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Borcherding/unsloth">Unsloth - Borcherding 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/Leoleojames1/unsloth">GitHub - Leoleojames1/unsloth: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - Leoleojames1/unsloth</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/mlabonne/FineTome-100k">mlabonne/FineTome-100k · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1333994942275522581)** (3 条消息): 

> `AGI 突破、Cybergod 论文、自动下载链接争议` 


- **讨论 AGI 突破**：一位成员分享了关于 **AGI** 最新突破的见解，强调这与**金钱**和**进化**有关。他们提供了题为 [Cybergod](https://github.com/James4Ever0/agi_computer_control/releases/download/paper-initial-publication/Cybergod__God_is_in_your_computer.pdf) 的论文链接。
   - 他们简明扼要地总结了自己的发现：“这一切都关乎金钱和进化。”
- **自动下载链接引发争论**：一位成员对**自动下载链接**表示蔑视，称其为“邪恶”。
   - 这引起了另一位成员的幽默回应，回复道 “lol”。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1333894217361133579)** (404 条消息🔥🔥🔥): 

> `DeepSeek vs OpenAI, AI 审查, 使用多个 AI 模型, AI 在创意写作中的应用, AI 的实时功能` 


- **DeepSeek 与 OpenAI 模型的对比**：用户正在将 DeepSeek R1 与 OpenAI 的模型进行对比，并对编码和创意写作等任务的表现发表了不同看法。
   - 一些用户报告称，在某些条件下它能产生更连贯的输出，而另一些用户则指出其在处理敏感话题时存在局限性。
- **对审查制度的担忧**：讨论揭示了对 AI 审查的担忧，特别是涉及中国和美国政治的政治内容。
   - 参与者指出，许多 AI 都有内置过滤器，会影响回答的质量和广度。
- **查询多个模型**：用户建议查询不同的模型并合并结果，以从 AI 获取更全面的答案。
   - 这种方法被提及为规避单一模型过滤和性能问题限制的一种方式。
- **创意写作中的局限性**：用户强调了由于敏感过滤器和上下文限制，使用 GPT 模型进行创意写作的困难。
   - 这种敏感性可能导致涉及暴力或历史事件的话题内容无法获取，从而限制了创意表达。
- **实时功能和用例**：一位用户表示有兴趣在 OpenAI 平台上为其 AI 助手连接实时功能，并寻求指导。
   - 讨论包括对 LM Studio 等工具的推荐，以及在各种项目中增强 AI 可用性的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.google.com/amp/s/decrypt.co/302691/did-openai-cheat-big-math-test%3famp=1">Redirect Notice</a>: 未找到描述</li><li><a href="https://youtu.be/e9XCbX40ijA?si=MnPB-HMHaxwWrGv3">DeepSeek AI Exposes Tech Oligarchy&#39;s Multi-Billion Dollar Scam</a>: Ground News：在 https://www.ground.news/majority 获取其无限访问 Vantage 计划的 50% 折扣。美国东部时间周一至周五中午 12 点观看 Majority Report 直播...</li><li><a href="https://github.com/tencent/Hunyuan3D-2">GitHub - Tencent/Hunyuan3D-2: High-Resolution 3D Assets Generation with Large Scale Hunyuan3D Diffusion Models.</a>: 使用大规模 Hunyuan3D 扩散模型生成高分辨率 3D 资产。- Tencent/Hunyuan3D-2
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1333921610658480228)** (30 条消息🔥): 

> `不可见的零宽空格字符, Custom GPT 链接输出问题, GPT 记忆和上下文限制, GPT 回答中的矛盾, 用户记忆 (User Memory) 功能的挑战` 


- **使用不可见字符避免链接格式化**：一位成员分享了他们利用不可见的“零宽”空格字符（如 `httpXs://`）来防止不必要的链接格式化的方法，并引用了他们针对该主题撰写的 [StackOverflow 文章](https://link.to.stackoverflow)。
   - 另一位成员称赞了这个想法，表明了该方法的潜在有效性。
- **Custom GPT 间歇性的链接输出**：讨论揭示了对 **Custom GPT** 无法始终如一地输出所有链接的沮丧，无论脚本功能如何。
   - 有人指出，依赖 GPT 执行此类任务是不可靠的，成员们建议使用更具体的指令来缓解此问题。
- **GPT 中的记忆和上下文问题**：成员们探讨了记忆功能的可靠性，担心 GPT 即使在受到提示时也经常不按预期使用用户记忆。
   - 对话强调了上下文长度如何导致模型回答的不一致，特别是在偏好或个人细节方面。
- **GPT 回答中的矛盾**：讨论集中在 GPT 回答的矛盾上，并与之前的版本进行了比较，观察了上下文处理情况。
   - 成员们指出，即使提供了明确且相关的问题， GPT 可能仍难以保持一致性，并将这一挑战比作在草堆中寻找第二根“针”。
- **用户记忆识别的挑战**：有人对 GPT 错误处理记忆提示表示担忧，有时会对用户记忆的存在表现出困惑。
   - 这引发了关于用户细节识别可能发生在单独处理阶段的评论，从而影响了回答的准确性。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1334207599431778376)** (1 条消息): 

> `o3-mini, owl-palm tree riddle` 


- **对 o3-mini 能力的好奇**：一位成员询问 **o3-mini** 是否能够解决 **owl-palm tree riddle**，并表示这是他们最关注的基准测试。
   - 这表明用户关注的重点是模型在解决特定谜题方面的表现，而非通用能力。
- **对谜题基准测试的兴趣**：该成员对 **owl-palm tree riddle** 的强调，反映了其评估 AI 能力的个人基准。
   - 这种关注点凸显了一种趋势，即用户开始优先考虑特定任务而非更广泛的功能。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1334207599431778376)** (1 条消息): 

> `o3-mini, owl-palm tree riddle` 


- **o3-mini 能破解 owl-palm tree riddle 吗？**：一位成员对 **o3-mini** 是否能够解决 **owl-palm tree riddle** 表示感兴趣，并将其视为一个重要的基准。
   - *那是我唯一关心的基准测试！*
- **关于 o3-mini 重要性的讨论**：讨论了基准测试对 **o3-mini** 的重要性，特别是侧重于它处理类似 owl-palm tree 场景等独特谜题的能力。
   - 成员们似乎将这个谜题视为检验 **o3-mini** 能力的试金石。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1333889691564376138)** (247 条消息🔥🔥): 

> `DeepSeek R1 Models, LM Studio Functionality, RAG Implementation and Performance, User Experience with LLMs, Learning Resources for LLM Optimization` 


- **探索 DeepSeek R1 模型**：用户正在讨论 DeepSeek R1 及其蒸馏版本的性能和价格，并将其与 Qwen 等模型在编程任务中的表现进行对比。
   - 对话反映了不同模型在不同价位提供高质量回答的有效性。
- **LM Studio 中的模型加载问题**：多位用户报告了模型加载困难，并询问了 LM Studio 的系统要求，特别是针对 Mac 用户。
   - 提供了更改设置和 guardrails 以缓解加载问题的建议，强调了满足系统规格的重要性。
- **LM Studio 中 RAG 的功能性**：成员们对 LM Studio 内 RAG 实现的性能以及其在处理文档时对模型质量的依赖提出了疑问。
   - 讨论包括用户对 RAG 的使用体验，以及静态模型在回答特定查询时面临的挑战。
- **学习优化本地 LLM**：一位用户因技术背景有限且担心数据隐私，寻求在 LM Studio 中优化本地 LLM 使用的指导。
   - 讨论强调了需要适合初学者的资源，以帮助医疗保健专业人员有效地利用 LLM。
- **访问模型和 Beta 版本**：用户正在解决 LM Studio 中模型可见性的相关问题，建议尝试 Beta 版本以获得增强功能和代理支持。
   - 对话强调了更新到最新版本以访问新模型和功能的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.cpaviation.com/images/downloads/CESSNA_150_POH.pdf&ved=2ahUKEwiyq_fd65qLAxXdrYkEHZRGOnIQFnoECAYQAQ&usg=AOvVaw1-2JNbrSWM45n3tyqIyHiB">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/fallout-tv-fallout-codsworth-fallout-prime-fallout-amazon-gif-14576962590525720544">Fallout Tv Codsworth GIF - Fallout tv Fallout Codsworth - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix">Feature matrix</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/tree/main?show_file_info=Qwen2.5-7B-Instruct-1M-IQ2_M.gguf">bartowski/Qwen2.5-7B-Instruct-1M-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">ggml/docs/gguf.md at master · ggerganov/ggml</a>: 用于机器学习的张量库。通过在 GitHub 上创建账号，为 ggerganov/ggml 的开发做出贡献。</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M">Qwen/Qwen2.5-7B-Instruct-1M · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/basics/import-model">Import Models | LM Studio Docs</a>: 使用你在 LM Studio 之外下载的模型文件</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M#3-launch-vllm">Qwen/Qwen2.5-7B-Instruct-1M · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF">bartowski/Qwen2.5-7B-Instruct-1M-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1333895135062396970)** (152 条消息🔥🔥): 

> `LLM 推理速度、DeepSeek 的硬件要求、在 Apple Silicon 上使用模型、ML 模型性能、使用 LLM 处理 CSV 数据` 


- **LLM 推理速度受内存带宽影响**：一位成员指出，LLM 推理速度在很大程度上取决于内存带宽，并指出 Mac 的带宽低于 **A4000** 甚至 **3060** GPU。
   - *LM Studio* 用户观察到现有模型的性能较慢。
- **DeepSeek 的硬件和模型要求**：多位用户讨论了在各种配置下运行 **DeepSeek R1 Distill-Qwen 70B** 等模型的硬件要求，建议重点关注至少具有 **12GB** 显存（VRAM）的 GPU。
   - 评论强调了 **Threadripper** 和 **EPYC** CPU 在高效处理多 GPU 和大型模型方面的能力。
- **在 Apple Silicon 模型上测试 DeepSeek**：一位成员询问是否可以在配备 **64GB RAM** 的 MacBook Pro 上运行 **DeepSeek**，并表示他们主要使用的是 CPU 资源。
   - 讨论表明，虽然 RAM 起着一定作用，但 GPU 利用率对于获得最佳性能至关重要。
- **模型性能比较**：用户分享了关于 **DeepSeek R1** 和 **Qwen 2.5** 等不同模型的经验，并指出了不同配置下的性能差异。
   - 建议模型的选择会影响速度和准确性，成员们建议针对日常任务测试较小的模型。
- **使用 LLM 处理 CSV 数据**：一位成员表示有兴趣使用 LLM 来格式化 CSV 交易以实现统一，考虑到跨链数据的复杂性。
   - 回复强调了使用 **Python** 编写脚本进行可靠的数据处理，特别是对于较大的数据集。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1333889897253048440)** (329 messages🔥🔥): 

> `DeepSeek API Issues, Qwen 2.5 Max, Sonnet as Editor, Test Driven Development (TDD), Pricing and Spending on AI Models` 


- **DeepSeek API 问题**：多位用户报告了 DeepSeek API 严重的宕机和性能问题，促使他们转回使用 Sonnet 等其他模型。
   - OpenRouter 的供应商（如 Nova）使用成本似乎很高，进一步加剧了用户对 DeepSeek 可靠性的不满。
- **Qwen 2.5 Max 的困惑**：关于 Qwen 2.5 Max 是否开源存在混淆，但由于对 GPU RAM 的高要求，它目前无法在本地使用。
   - 一些用户表示有兴趣了解如何有效地实现 Qwen 2.5 Max 并将其集成到他们的编码工作流中。
- **Sonnet 作为编辑器**：尽管存在相关成本和偶尔的延迟问题，许多用户因其可靠性和速度而迁回使用 Sonnet。
   - 与其他供应商的体验相比，Sonnet 在需要稳定编码和编辑性能的任务中更受青睐。
- **测试驱动开发 (TDD)**：用户讨论了测试驱动开发 (TDD) 作为一种方法论的优势，重点是在开发代码之前编写测试以确保质量。
   - 将 Aider 等 AI 工具与 TDD 实践相结合，似乎提高了积极使用这些方法论的用户的生产力。
- **AI 模型定价与支出**：用户分享了他们的支出经验，其中一位指出每月在 AI 工具上的支出约为 50 美元，主要是在使用 Sonnet 时。
   - 在尝试 AI 模型时维持开支是普遍关注的问题，用户对大规模使用期间不断上升的成本表示不安。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/2025/01/28/deepseek-down.html#ollama?">Alternative DeepSeek V3 providers</a>: DeepSeek 的 API 一直存在可靠性问题。这里是你可以使用的替代供应商。</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API, Providers, Stats</a>: Sonar Reasoning 是 Perplexity 基于 [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 提供的推理模型。通过 API 运行 Sonar Reasoning</li><li><a href="https://x.com/alibaba_qwen/status/1884263157574820053?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Qwen (@Alibaba_Qwen)</a>: DeepSeek V3 的爆发吸引了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM...</li><li><a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Max-Demo">Qwen2.5 Max Demo - a Hugging Face Space by Qwen</a>: 未找到描述</li><li><a href="https://aider.chat/2025/01/28/deepseek-down.html">Alternative DeepSeek V3 providers</a>: DeepSeek 的 API 一直存在可靠性问题。这里是你可以使用的替代供应商。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">DeepSeek R1 (free) - API, Providers, Stats</a>: DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但已开源且具有完全开放的推理 token。其参数量为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://x.com/testingcatalog/status/1884652009938178115">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: 突发新闻 🚨：Grok 3 将支持推理！它还能够向 UI 展示其“思考”过程 👀 引用 Tibor Blaho (@btibor91) 的话，独立的 Grok Web 应用现在包含了...</li><li><a href="https://tenor.com/view/vegetto-ssj3-dbz-gif-6326837151155652534">Vegetto Ssj3 Dbz GIF - VEGETTO SSJ3 DBZ - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bad-work-citizen-gif-27615137">Bad Work Citizen GIF - Bad Work Citizen - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://pluggy.readthedocs.io/en/stable/">pluggy &#8212; pluggy 0.1.dev94+gf8aa4a0 documentation</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=42854130">&gt; 99% of the code in this PR [for llama.cpp] is written by DeekSeek-R1 It&#x27;s defi... | Hacker News</a>: 未找到描述</li><li><a href="https://ollama.com/library?sort=newest">library</a>: 快速上手并运行大型语言模型。
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1333889588917178390)** (56 messages🔥🔥): 

> `Aider 上下文与文件管理，模型性能与速度，使用代码风格规范，Architect Mode 工作流，排查 Token 限制问题` 


- **Aider 通过 /add 命令辅助文件上下文**：为了添加编辑相关的上下文，成员们分享了如何使用 `/add` 命令，在终端中直接通过 `aider` 指定文件。
   - 有人指出，虽然模型会提示进行文件编辑，但有时并不需要编辑文件，即使你取消了响应，上下文也会保留。
- **对模型性能和速度的担忧**：一些用户报告了 Hyperbolic 的 R1 响应时间不佳，交互时间长于预期，有时超过一分钟。
   - 计算显示输出约为 **每秒 12 个 Token**，这引发了对其系统资源利用率的质疑。
- **引入规范以保持一致的代码风格**：一位成员强调了创建规范文件的能力，以确保代码风格一致，例如始终添加类型提示（type hints）并使用特定方法。
   - 这可以通过上传 `CONVENTIONS.md` 文件并使用 `/read` 加载它，以便在编辑过程中提供指导。
- **理解 Architect Mode 及其工作流**：用户对 Architect Mode 的工作流表示不确定，注意到系统每次都会提示编辑文件，感觉有些冗余。
   - 反馈表明，在选择不进行文件编辑时，保留之前的结论具有复杂性，这与使用 `/code implement above` 不同。
- **排查 Token 限制和访问问题**：一位用户分享了关于 Token 限制错误的挫败感，这在使用某些模型时很常见，并质疑了实际支持的限制。
   - 解决方案指出，提高使用层级（usage tier）可以缓解访问问题，并讨论了其他供应商的相关配置。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: 告诉 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://aider.chat/2025/01/28/deepseek-down.html#fireworks">Alternative DeepSeek V3 providers</a>: DeepSeek 的 API 一直存在可靠性问题。这里是你可以使用的替代供应商。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

apcameron: 看看这个项目。 https://github.com/huggingface/open-r1
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1334174162557735036)** (2 messages): 

> `Sonar Reasoning API, DeepSeek R1 在 Mac App 上线` 


- **发布 Sonar Reasoning API**：推出了由 **DeepSeek** 推理模型驱动的 **Sonar Reasoning** API，它支持思维链（chain-of-thought）推理，并结合了实时联网搜索和引用功能。
   - *这一新服务托管在美国数据中心*，旨在通过不收集或共享 API 数据来保护用户隐私。
- **DeepSeek R1 现已登陆 Perplexity Mac App**：**DeepSeek R1** 已可通过 Perplexity Mac App 中的命令快捷键访问，可通过 Mac App Store 的更新获取。
   - *建议用户尽快更新应用*以使用这一新功能。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1333889446503907359)** (316 messages🔥🔥): 

> `DeepSeek R1 查询，Perplexity Pro 订阅，模型可用性与使用，API Key 与使用，Web 和 iOS 功能`

- **DeepSeek R1 每日额度提升**：Perplexity 已将 Pro 用户的每日限制提高至 **50 次 DeepSeek R1 查询**，非 Pro 用户可使用 **5 次查询**。
   - 这一变化由 CEO Aravind Srinivas 宣布，预示着持续的更新与增强。
- **Pro 用户每日查询限制**：Pro 用户目前在 Perplexity 内拥有每日 **100 次 R1 查询**的额度，使其成为大规模使用的强力选择。
   - 用户对此反响积极，认为这提升了查询能力。
- **R1 模型参数澄清**：Perplexity 所使用的 R1 模型确认为 **全量 671B 参数模型**。
   - 这符合对增强性能和能力的预期。
- **Playground 中的 API Key 使用**：用户讨论了在 Perplexity Playground 上使用 **API key** 的情况，并指出其在日常使用中的高性价比，允许几乎无限次的交互。
   - 尽管在没有 key 的情况下也能运行，但 API 提供了一种结构化的方式来访问额外功能。
- **Web 和 iOS 的即将推出功能**：社区成员询问了 Web 和 iOS 平台上线 **'agent' 或 'cron functionality'** 的时间表。
   - 随着部分用户继续利用 Android 上可用的功能，大家对新功能的兴趣非常高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1884652009938178115">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重大新闻 🚨：Grok 3 将支持推理！它还能够在 UI 中展示其“思考”过程 👀 引用 Tibor Blaho (@btibor91) 的话：独立的 Grok Web 应用现在包含了...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的 polyglot 基准测试中创下 SOTA</a>：R1+Sonnet 在 aider polyglot 基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://x.com/aravsrinivas/status/1884509590684934211?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：@julheetel8890 Full。</li><li><a href="https://x.com/denisyarats/status/1883763037875978669">来自 Denis Yarats (@denisyarats) 的推文</a>：@D_Twitt3r 是的，这是真正的 R1，即将推出</li><li><a href="https://x.com/itsPaulAi/status/1870892042769039598">来自 Paul Couvert (@itsPaulAi) 的推文</a>：推理模型正在成为常态。友情提醒，OpenAI 已经发布了针对它们的提示词编写指南。一些要点和示例：</li><li><a href="https://artificialanalysis.ai/models/deepseek-r1/providers?utm_source=perplexity">DeepSeek R1：API 提供商性能基准测试与价格分析 | Artificial Analysis</a>：对 DeepSeek R1 的 API 提供商进行的性能指标分析，包括延迟（首个 token 时间）、输出速度（每秒输出 token 数）、价格等。参与基准测试的 API 提供商包括...</li><li><a href="https://x.com/aravsrinivas/status/1884492340657606896?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity Pro 用户的 DeepSeek R1 每日查询限制已从每天 10 次增加到 25 次。目标是随着我们增加更多容量而继续提升！请享用。</li><li><a href="https://x.com/aravsrinivas/status/1884563740575551915?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：好吧，即使你不在乎你的数据流向中国，我认为也值得关注不要使用 DeepSeek 应用提供的受审查模型。这就是为什么在 Perplexity 上使用 R1 模型是值得的...</li><li><a href="https://x.com/johncoogan/status/1884127587292832005">来自 John Coogan (@johncoogan) 的推文</a>：这当然是你的论点。你两天前才听说 DeepSeek。刚看完一段 40 分钟的深度解析——可能是 Deirdre Bosa 做的。你打算谈论这如何使情况复杂化...</li><li><a href="https://x.com/aravsrinivas/status/1884708550795985382?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity Pro 用户的 DeepSeek R1 每日查询次数已增加到每天 50 次，免费用户增加到每天 5 次。更多更新即将推出。请享用！</li><li><a href="https://x.com/aravsrinivas/status/1884718399797821642?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity 是否应该将 DeepSeek R1 设为默认模型？</li><li><a href="https://www.businesstoday.in/technology/news/story/perplexity-ai-increases-limit-for-daily-deepseek-r1-queries-on-its-platform-462421-2025-01-29">Perplexity AI 提高了其平台上 DeepSeek R1 的每日查询限制</a>：Perplexity AI CEO 在 X 上宣布了提高每日限制的消息，用户对这一举措表示赞赏，并要求进一步提高。</li><li><a href="https://www.cplx.app/">Complexity</a>：每个人都梦寐以求的增强版 Perplexity.ai。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i8rujw/notes_on_deepseek_r1_just_how_good_it_is_compared/">Reddit - 深入探讨任何事物</a>：未找到描述</li><li><a href="https://www.deeplearning.ai/short-courses/reasoning-with-o1/">使用 o1 进行推理</a>：学习如何使用 OpenAI 的 o1 模型并为其编写提示词，以处理复杂的推理任务。</li><li><a href="https://www.searchenginejournal.com/perplexity-ai-deploys-chinese-deepseek-ai-model/538452/?utm_source=chatgpt.com">Perplexity AI 部署中国 DeepSeek AI 模型</a>：Perplexity AI 在其 AI 搜索引擎上提供了一个自托管版本的中国 DeepSeek R1 推理模型供用户使用。
</li>
</ul>

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1333926179496787978)** (13 条消息🔥): 

> `Java 23 SDK 更新，DeepSeek 对标 OpenAI O1，F-35 战斗机事故，切叶蚁培养，阿里巴巴新模型` 


- **Java 23 SDK 更新发现**：一位用户分享了[关于将 Java 23 SDK 更新至 Java 2 的细节](https://www.perplexity.ai/search/updating-java-23-sdk-to-java-2-lDOVkVzrQ.OSUPQoRvn3gA)，强调了在公共服务环境中的快速实现。
   - 社区讨论了私营企业与公共服务适配流程相比的效率问题。
- **DeepSeek 可能超越 OpenAI 的 O1**：[一段 YouTube 视频](https://www.youtube.com/embed/YjOGIs4sA50)讨论了 DeepSeek R1 如何可能超越 OpenAI 的 O1 模型，以及对 AI 职业成功的预测。
   - 视频还涉及了安眠药对大脑活动的影响，引发了聊天中的进一步探讨。
- **F-35 战斗机坠毁**：发生了一起涉及 [F-35 战斗机坠毁](https://www.perplexity.ai/page/f-35-fighter-jet-crashes-in-al-FzWYTWVYQ7WqDaXf5z1EXA)的事件，引发了对军用飞机安全措施的担忧。
   - 坠机细节和调查吸引了成员们的注意，引发了对军事协议的讨论。
- **切叶蚁的培养习性**：围绕切叶蚁是否主动培养真菌展开了[一场对话](https://www.perplexity.ai/search/do-leafcutter-ants-cultivate-f-5SA1dh2vRrKfyMie3qx4WA)，揭示了它们在生态系统中角色的迷人见解。
   - 成员们交换了观点和研究引用，以更好地理解自然界中的共生关系。
- **阿里巴巴推出新模型**：通过[此链接](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg)分享的关于阿里巴巴新模型的讨论引发了有趣的辩论，这可能会影响其市场策略。
   - 用户讨论了该模型对科技领域竞争和创新的潜在影响。



**提到的链接**：<a href="https://www.youtube.com/embed/YjOGIs4sA50">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1334061976527048744)** (10 条消息🔥): 

> `Sonar Reasoning 性能，推理搜索反馈，Sonar 模型规格，推理输出问题，来源与引用` 


- **Sonar Reasoning 的功能性获得认可**：一位成员强调 **Sonar Reasoning** 运行良好，特别提到在一段关于 **MCP servers** 的视频中成功使用了实时引用功能。
   - 其他人则表示体验不一，有人报告了来自 Sonar 的拒绝错误，指出消息格式化存在问题。
- **询问 Sonar Reasoning 的底层模型**：成员们正在询问 **Sonar Reasoning** 使用的是完整的 **R1 model (671B)** 还是像 **Llama 70B** 这样的蒸馏版本。
   - 一位用户指出，他们怀疑这是一个与模型选择相关的 Bug，并提到将把询问移至反馈频道。
- **征求关于推理搜索有效性的反馈**：有人对新推理搜索与早期模型相比的性能提升提出了疑问。
   - 一位成员观察到它似乎没有预期的那样进行深度“思考”，引发了关于潜在修复方案的讨论。
- **寻求答案的更多来源**：另一位用户询问如何获取更多来源，可能是为了提高 **Sonar Reasoning** 回答的质量。
   - 他们分享了一张与该话题相关的图片，但得到的回复很少。
- **社区对 Perplexity 运营的好奇**：成员们对 **Sonar Reasoning** 系统的细节表示好奇，寻求来自 Perplexity 员工的见解。
   - 对透明度的渴望表明社区参与度很高，希望明确了解他们所使用的工具。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1333890028828491887)** (298 条消息🔥🔥): 

> `MoE 模型性能，Nous Research 资金来源，DeepSeek R1 可用性，AI 推理与输出质量，股市预测推测` 


- **MoE 模型需要高内存以获得最佳性能**：在讨论 Mixture of Experts (MoE) 模型时，一位用户强调内存大小对性能至关重要，特别是在 CPU 配置下，这表明了优化技术的重要性。
   - 用户分享了他们在不同配置下的经验，指出拥有足够的内存带宽可以显著提高 Token 处理速度。
- **Nous Research 的融资机制**：成员们分享了关于 Nous Research 资金来源的见解，提到了 VC 投资、使用其模型的公司的捐赠，以及部分来自周边商品销售的收入。
   - 虽然周边销售目前被认为微不足道，但整体的多元化资金来源支持了其高昂的计算成本。
- **DeepSeek R1 现已在 Azure 上线**：DeepSeek R1 已在 Azure AI Foundry 和 GitHub 上发布，成为为开发者提供的 1,800 多个 AI 模型庞大库中的一员。
   - 这一扩展反映了行业日益增长的兴趣，将 DeepSeek R1 定位为企业级 AI 解决方案中的强力选项。
- **关于 AI 推理与概率输出的辩论**：有一场关于 AI 的思维链（Chain of Thought）输出是否真正代表推理的讨论，对于模型生成的置信度分数的可靠性和含义存在不同看法。
   - 参与者争论了 AI 模型中使用的算法与人类推理能力之间的根本区别。
- **探索 AI 预测股票的能力**：有人询问 AI 模型准确预测股市波动的潜力，认为市场的不可预测性可能使此类努力面临挑战。
   - 多位成员推测了 AI 和人类在股票交易能力方面的局限性，确认这是一个复杂的领域。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NousResearch/status/1848397863547515216">来自 Nous Research (@NousResearch) 的推文</a>：未找到描述</li><li><a href="https://tenor.com/view/popcorn-minions-popcorn-day-laugh-gif-5026739">小黄人吐爆米花 GIF - Popcorn Minions Popcorn Day - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://vcomp.eqtylab.io">EQTY Lab — 介绍可验证计算</a>：通过首个可审计的治理证明来认证和保护 Agentic AI 工作流。</li><li><a href="https://shop.nousresearch.com/collections/products">购买我们的产品</a>：Nous Research</li><li><a href="https://x.com/carrigmat/status/1884244369907278106">来自 Matthew Carrigan (@carrigmat) 的推文</a>：在本地运行 DeepSeek-R1 的完整硬件 + 软件配置。真正的模型，非蒸馏版，Q8 量化以保证完整质量。总成本 6,000 美元。所有下载和配件链接如下：</li><li><a href="https://www.youtube.com/watch?v=0uK7w0zCxA8">如果 AI 要具有革命性，我们需要看到这场革命，Big Tech 的 Alex Kantrowitz 说道</a>：Big Technology 的 Alex Kantrowitz 和 Alger 的 Dan Chung 加入 'Closing Bell' 讨论 AI 电力需求、投资以及围绕技术的情绪转变...</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 现已在 Azure AI Foundry 和 GitHub 上线 | Microsoft Azure 博客</a>：DeepSeek R1 通过 Microsoft Azure AI Foundry 和 GitHub 上的模型目录提供，使企业能够无缝集成先进 AI。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1333963169701433427)** (6 条消息): 

> `Olama, 本地 AI 模型选项, CLI vs GUI` 


- **安装 Olama 以运行本地模型**：一位成员建议安装 **Olama**，以便在本地运行 **Mistral**、**Llama** 或 **DeepSeek distilled** 等开源模型，作为私人助手。
   - 使用 Olama 的动机是为了满足用户对本地和私密助手的需求。
- **对 Olama CLI 的担忧**：另一位成员指出有比 Olama **更好的选择**，并批评其使用命令行界面 (**CLI**)。
   - 他们强调其他程序可能会通过内置的图形用户界面 (**GUI**) 提供更友好的用户体验。
- **Olama 的替代方案**：讨论中包括了使用 **KoboldCPP** 作为运行模型的 Olama 替代方案的建议。
   - 他们还为那些不介意使用闭源软件的用户提到了 **LM Studio**。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334067740972941352)** (2 条消息): 

> `Mixture-of-Experts Models, Autonomy-of-Experts Paradigm` 


- **MoE 模型需要更好的专家选择**：论文指出，在 **Mixture-of-Experts (MoE)** 模型中，router 的决策与专家的执行相分离，导致专家选择并非最优，这往往使得其表现不如 dense 模型。
   - *为了解决这个问题*，作者提出了一种名为 **Autonomy-of-Experts (AoE)** 的新范式，其中专家根据其处理输入的能力自主进行选择。
- **专家自行排序进行处理**：在 **AoE** 框架中，去除了 router，专家会预先计算输入的内部激活值（activations），并根据激活范数（activation norms）对自己进行排序。
   - 因此，只有 **排名靠前的专家** 会继续后续处理，而其他专家则中止，从而提高了专家利用效率。



**提到的链接**：<a href="https://arxiv.org/abs/2501.13074">Autonomy-of-Experts Models</a>：Mixture-of-Experts (MoE) 模型大多使用 router 将 token 分配给特定的专家模块，仅激活部分参数，且表现通常优于 dense 模型。我们认为这种分离...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 条消息): 

tudorboto: "Intel i5/AMD Ryzen 5 或更强"，M3 属于 "更强" 的范畴吗？
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1334067740972941352)** (2 条消息): 

> `Mixture-of-Experts models, Autonomy-of-Experts paradigm, Router decision-making in MoE` 


- **MoE 模型在专家选择方面存在困难**：目前的 **Mixture-of-Experts (MoE)** 模型依赖 router 进行 token 分配，这往往导致学习效果和专家选择并非最优。
   - *Autonomy-of-Experts (AoE)* 的提出强调了改进专家选择和学习的必要性，它消除了对 router 的依赖。
- **在专家选择中引入自主性**：**Autonomy-of-Experts (AoE)** 范式允许专家通过评估其内部激活范数（activation norms），自主决定其是否适合处理输入。
   - 在这种方法中，只有排名靠前的专家会进行进一步计算，从而可能实现更高效的处理。



**提到的链接**：<a href="https://arxiv.org/abs/2501.13074">Autonomy-of-Experts Models</a>：Mixture-of-Experts (MoE) 模型大多使用 router 将 token 分配给特定的专家模块，仅激活部分参数，且表现通常优于 dense 模型。我们认为这种分离...

  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1333900279224995924)** (87 条消息🔥🔥): 

> `Windsurf 账户问题，DeepSeek 集成，Codeium 扩展设置，用户体验关注点，Flex 额度与定价` 


- **Windsurf 账户登录问题频发**：许多用户报告无法登录其 Windsurf 账户，其中一人提到反复出现“登录失败”错误，提示语言服务器尚未启动。
   - 另一位用户指出，这似乎是一个影响多个成员的普遍问题。
- **DeepSeek 集成讨论升温**：一位用户对 Windsurf 缺失 DeepSeek R1 表示沮丧，并威胁要切换到目前支持该功能的 Cursor。
   - 有人指出 DeepSeek 在 tool calling 方面表现不佳，这表明有效集成的复杂性较高。
- **关于 VSCode 中 Codeium 扩展设置的疑问**：一位用户询问 VSCode 中的 Codeium 扩展是否正常运行，特别是 Chat 是否能像 GitHub Copilot 那样访问选中的文本。
   - 回复澄清说，由于微软对 VSCode 的专有性质，某些功能与其他工具相比可能会受到限制。
- **对 Windsurf 用户体验的担忧**：多位用户对 Windsurf 的定价以及模型修改导致原本正常的代码被改动表示不满，对持续存在的 Bug 感到沮丧。
   - 评论强调，由于纠正错误导致 Flow action 额度大量流失，从而对订阅成本感到后悔。
- **新账户 Flex 额度的困惑**：用户对创建新账户后 Flex 额度的变化提出质疑，注意到提供的免费额度数量存在差异。
   - 澄清显示，这是一次性的试用赠礼，相比之前的方案降低了初始预期。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: 未找到描述</li><li><a href="https://status.codeium.com/">Codeium Status</a>: 未找到描述</li><li><a href="https://codeium.com/blog/pricing-windsurf">Plans and Pricing Updates</a>: 我们对 Cascade 定价模型进行了一些变更。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1333891348948062360)** (193 messages🔥🔥): 

> `Windsurf 性能问题、Sonnet LLM 批评、Cascade 功能问题、用户对 AI 辅助的挫败感、关于定价和价值的反馈` 


- **用户报告 Windsurf 的性能问题**：多位成员指出 Windsurf 运行缓慢，特别是在聊天界面输入时，且无法有效地编辑文件。
   - 用户还报告了 Sonnet LLM 的异常行为，称其在代码编写的有效性和可靠性方面有所下降。
- **Sonnet LLM 受到质疑**：许多用户对 Sonnet LLM 的表现表示失望，认为它在理解 prompts 和准确完成任务方面的能力已经减弱。
   - 一些用户将其与其他平台进行了不利的对比，断言通过 Cursor 等替代方案可以更高效地完成类似任务。
- **Cascade 功能引发担忧**：部分用户在 Cascade 修改文件时遇到困难，存在丢失上下文或产生错误的问题，这表明其功能可能存在缺陷。
   - 反馈建议 Cascade 的运行不够稳定，导致用户不得不求助于手动重构以避免问题。
- **客户对 AI 辅助的挫败感**：用户对 AI 的输出表示沮丧，称荒谬或错误的回答导致了精力的浪费以及代码库中意料之外的更改。
   - 针对低质量 AI 回答提出的积分撤销功能请求，表明了用户对当前模型价值的不满。
- **对定价与实用性的担忧**：由于许多用户报告 Windsurf 的效果不佳，有人呼吁重新评估该服务的定价，建议鉴于输出质量应降低价格。
   - 评论强调了对改进的渴望，以确保对平台的投入能产生切实收益，特别是在处理复杂项目时。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://shitposting.pictures/BwrEiWdh7njY">手工精选的恶搞图片</a>：未找到描述</li><li><a href="https://status.codeium.com">Codeium 状态</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=WYb2aMVnuYY">与 Kevin Hou 探讨 AI 代码编辑器的未来 (Codeium, Windsurf)</a>：本期节目由 Codeium 产品工程负责人 Kevin Hou 参与，涵盖了公司从 GPU 虚拟化到创建领先 AI 的历程...</li><li><a href="https://dev.aimusiclabel.com/">音乐生成的未来 - AI 唱片公司</a>：利用人工智能开拓音乐的未来。加入我们，通过尖端的 AI 技术彻底改变音乐产业。</li><li><a href="https://www.youtube.com/watch?v=hqlgsPEN7TU">使用 Xcode 和 Windsurf 构建精美的 SwiftUI iPhone 应用 | 完整教程</a>：在这个分步教程中，学习如何使用 SwiftUI、Xcode 和 Windsurf 创建一个漂亮的 iPhone 应用！无论你是初学者还是经验丰富的开发者...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1334192391229276213)** (1 messages): 

> `DeepSeek R1, Chutes, Perplexity's Sonar, Sonar-Reasoning` 


- **DeepSeek R1 迎来 Chutes！**：新的去中心化提供商 [Chutes](https://chutes.ai/) 在 [openrouter.ai](https://openrouter.ai/deepseek/deepseek-r1:free) 为 **DeepSeek R1** 提供 **免费端点**。这为希望利用 DeepSeek R1 能力的用户提供了更多选择。
   - 随着 OpenRouter 扩大其提供商阵容，*令人兴奋的发展即将到来*！
- **Perplexity 增强 Sonar 模型**：作为其最新模型系列的一部分，**Perplexity 的 Sonar** 获得了**重大改进**，使其在速度和成本方面更具效率。详情请访问 [sonar.perplexity.ai](https://sonar.perplexity.ai)。
   - 新版本 **Sonar-Pro** 预计很快发布，承诺提供更强大的功能！
- **认识 Sonar-Reasoning！**：**Sonar-Reasoning** 是基于 DeepSeek 架构构建的专业推理模型，在搜索和推理任务方面表现出色。这一新模型旨在提升各种应用中的用户体验。
   - 正如[此处](https://discord.com/channels/1091220969173028894/1092729520181739581/1332050064624975953)的详细公告所强调的，用户可以通过利用 Web 搜索功能在所有模型中使用类似的功能。



**提到的链接**：<a href="https://openrouter.ai/deepseek/deepseek-r1:free.">DeepSeek R1 - API, 提供商, 统计数据</a>：DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并具有完全开放的推理 token。它的参数量为 671B，在一次推理过程中有 37B 激活。Ru...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1333892379060600842)** (277 条消息🔥🔥): 

> `OpenRouter 用户体验、DeepSeek 模型性能、模型通信与定价、图像生成讨论、翻译模型推荐` 


- **用户在使用 DeepSeek 时遇到的限制及定价问题**：多位用户报告了 DeepSeek v3 的性能问题以及波兰语等语言的翻译困难，指出结果可能不准确且缺乏上下文。
   - 同时，用户对 OpenRouter 的定价结构表示担忧，认为 API 请求收取的 5% 费用过高，许多人认为该比例可以更低。
- **用户对图像生成的请求**：用户表达了将 DALL-E 或 Stability AI 等图像生成功能集成到 OpenRouter 平台的强烈兴趣，以增强功能。
   - 成员们指出，增加此类功能可以吸引更多用户并提升平台的实用性。
- **模型设备与通信问题**：部分用户遇到了模型无法正确响应或返回空 Token 的问题，建议在 OpenRouter 界面中对输出进行更稳健的处理。
   - 其他用户询问了如何检索因请求长度限制而丢失的响应，强调了数据可访问性的重要性。
- **DeepSeek R1 模型相关问题**：用户分享了关于 R1 模型输出的经验，报告了与 OpenAI 模型相比的不一致性，并讨论了界面内推理（Reasoning）的限制。
   - 成员们还提到，升级对 Gemini 等模型的视频支持是当务之急。
- **翻译模型推荐**：关于各种翻译模型有效性的讨论不断涌现，用户发现使用目标语言进行提示（Prompting）会带来更好的效果。
   - 用户分享了对 Grok 和 Claude 等替代方案的推荐，并对其系统提示词（System Prompts）的清晰度表示满意。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>：设置模型使用限制</li><li><a href="https://x.com/carismarus/status/1884339334746116337">Olivier Depiesse (@carismarus) 的推文</a>：@OpenRouterAI 什么时候发币（Wen token）？</li><li><a href="https://huggingface.co/settings/inference-providers">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API、提供商、统计数据</a>：Sonar Reasoning 是由 Perplexity 提供、基于 [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 的推理模型。通过 API 运行 Sonar Reasoning</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">DeepSeek R1 (免费) - API、提供商、统计数据</a>：DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但采用开源形式并提供完全开放的推理 Token。其参数量为 671B，推理过程中激活参数为 37B。运行...</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>：在多个提供商之间路由请求</li><li><a href="https://huggingface.co/blog/inference-providers">欢迎使用 Hub 上的推理提供商 🔥</a>：未找到描述</li><li><a href="https://github.com/OpenRouterTeam/openrouter-runner">GitHub - OpenRouterTeam/openrouter-runner: 为 OpenRouter 上的开源模型提供支持的推理引擎</a>：为 OpenRouter 上的开源模型提供支持的推理引擎 - OpenRouterTeam/openrouter-runner</li><li><a href="https://aka.ms/oai/quotaincrease">
            Dynamics 365 Customer Voice
        </a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1334015645859446857)** (53 条消息🔥): 

> `DeepSeek 数据库泄露, Dario Amodei 对 AI 模型的见解, 社区对模型性能的反应, R1 和 R1-Zero 模型分析, 对 AI 模型透明度的担忧` 


- **DeepSeek 数据库暴露**：Wiz Research 报告发现了 “DeepLeak”，这是一个 **公开可访问的 ClickHouse 数据库**，属于 DeepSeek，泄露了包括 **密钥（secret keys）** 和 **聊天消息** 在内的敏感信息。
   - 这一事件引发了警报，因为用户可能会 **窃取数据（exfiltrate data）** 并提升服务器内的权限。
- **Dario Amodei 颇具争议的见解**：Dario Amodei 分享了他对出口管制和 AI 未来的看法，引发了褒贬不一的反应，一些人认为这 **“比预想的更像是在自我安慰（cope）”**。
   - 他关于 **AGI 可能在两年内到来** 的建议引发了社区的怀疑和嘲笑。
- **社区分析模型性能**：针对 **Claude 3.5 Sonnet** 并非从更大模型蒸馏而来的说法展开了讨论，社区成员对其训练方法论表示怀疑。
   - 成员们质疑有关其性能和成本声明的可信度，称 *“他竟然公然撒谎，太离谱了”*。
- **关于 R1 和 R1-Zero 模型的争论**：一项分析指出 **R1-Zero 可能比 R1 更重要**，并强调缺乏为研究目的提供该模型变体的托管服务商。
   - 社区对 R1 作为 **旗舰模型** 却为了“人类食用”而遭到 **削弱（nerfed）** 表示失望。
- **对 AI 模型透明度的担忧**：由于许多模型并非开源，成员们对可能 **需要信任高管** 提供的模型性能细节感到沮丧。
   - 评论强调了这种缺乏透明度对于从 AI 模型中获得最佳结果的广泛影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arcprize.org/blog/r1-zero-r1-results-analysis">R1-Zero and R1 Results and Analysis</a>：对 DeepSeek R1 的分析</li><li><a href="https://x.com/wiz_io/status/1884707819737223591?s=61">来自 Wiz (@wiz_io) 的推文</a>：这意味着任何人都可以访问包含实际聊天消息、内部秘密、服务数据的日志，并可能在窃取数据的同时提升服务器内的权限。</li><li><a href="https://x.com/wiz_io/status/1884707816935391703?s=61">来自 Wiz (@wiz_io) 的推文</a>：突发：内部 #DeepSeek 数据库公开暴露 🚨 Wiz Research 发现了 “DeepLeak” —— 一个属于 DeepSeek 的公开可访问的 ClickHouse 数据库，暴露了高度敏感的信息...</li><li><a href="https://x.com/darioamodei/status/1884636410839535967?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Dario Amodei (@DarioAmodei) 的推文</a>：我对中国、出口管制和两个可能未来的看法 https://darioamodei.com/on-deepseek-and-export-controls
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1334236751228833917)** (24 messages🔥): 

> `OpenAI 的 lockdown 模式, 对 O3 发布时机的担忧, Meta 对 DeepSeek 的兴趣, Grok3 开发进展, 模型定价问题` 


- **DeepSeek 事件后 OpenAI 进入 lockdown 模式**：一位用户确认，在 **DeepSeek** 相关进展之后，OpenAI 已进入全面 lockdown 模式，并对自己被卷入其专有训练流程感到惊讶。
   - *“在我 500 万美元的推理模型训练运行中投入了 5000 美元，他们就已经抓到我了”*，这引发了对 OpenAI 严格操作程序的关注。
- **O3 发布时机的不确定性**：考虑到 OpenAI 面临的巨大赌注，人们担心 **O3** 的发布可能会推迟。
   - 这种压力被比作 **Meta** 在未来产品发布方面所面临的压力。
- **Meta 探索将 DeepSeek 用于广告业务**：Meta Platforms 正在评估将 **DeepSeek** 用于其广告产品，并在担心其性能超过 **Llama 4** 后成立了“作战室”（war rooms）。
   - *“对 DeepSeek 的恐慌是真实的”* 表明 Meta 并非唯一一家对竞争技术感到担忧的公司。
- **对 Grok3 能力的期待**：一名成员表达了对 **Grok3** 的渴望，希望它能带来 **XAI** 合格研究人员的创新成果。
   - 传闻暗示他们正在准备一个思考模型（thinking model），这可能意味着重大进展。
- **关于模型定价差异的讨论**：用户指出 **模型定价** 可能存在错位，除 **4o-mini** 以外的所有模型都被一些人认为定价不当。
   - 正如一位成员所言，*“让我们看看 10 万张 H100 能带来什么”*，这反映了对资源分配的关注与好奇。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/charlespacker/status/1884661284375007485">Charles Packer (@charlespacker) 的推文</a>：可以确认 OpenAI 在 DeepSeek 之后处于全面 lockdown 模式。在我 500 万美元的推理模型训练中投入了 5000 美元，他们就已经抓到我了 😳⛓️🚔</li><li><a href="https://x.com/amir/status/1884678357436268862">Amir Efrati (@amir) 的推文</a>：猜猜怎么着？现在 Meta 自身也在评估是否将 DeepSeek 用于广告产品。引用 Amir Efrati (@amir) 的新闻：对 DeepSeek 的恐慌是~真实的~，Meta Platforms 担心 DS 比 Lla 更好...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1333901221442093088)** (41 messages🔥): 

> `DeepSeek R1, Llama 4 开发, Grok 3 与 O3-mini 发布, ChatGPT 营收洞察, 漏洞报告` 


- **DeepSeek R1 在 Azure 上线**：DeepSeek R1 现已在 [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry) 和 GitHub 上可用，提供了一个拥有超过 1800 个模型的扩展平台，用于高级 AI 集成。
   - 此次发布使企业能够在微软支持的 SLA、安全性和负责任 AI 承诺下，访问尖端 AI。
- **Llama 4 发布可能推迟**：传闻称，鉴于 DeepSeek 的影响，Llama 4 正在从头开始重做，这推迟了其原定于 2 月的发布。
   - 据报道，Together 等合作伙伴仅收到了关于延迟的模糊更新，表明开发过程中发生了重大变化。
- **Grok 3 和 O3-mini 的发布时机**：Grok 3 和 O3-mini 的发布时间似乎尚不确定，有迹象表明它们可能定于 1 月发布，但内部讨论仍在继续。
   - 成员们预计 DeepSeek 的进展可能会打乱计划中的发布，因为 OpenAI 通常倾向于在周四发布。
- **ChatGPT Pro 营收洞察**：据报道，OpenAI 每月 200 美元的 ChatGPT Pro 收入已超过 ChatGPT Enterprise，年化收入超过 **3 亿美元**。
   - 这一信息表明，与企业版相比，Pro 订阅模式具有强大的用户留存和需求。
- **DeepSeek 报告严重漏洞**：一位用户报告称，已发送邮件告知一个可能暴露 DeepSeek 敏感数据（包括潜在 API keys）的严重漏洞。
   - 解决该漏洞的紧迫性被强调，提醒 DeepSeek 迅速采取行动以降低风险。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/vibagor44145276/status/1884467879963418645">来自 vibagor441 (@vibagor44145276) 的推文</a>：反模糊发布：我想我现在可以放心地说这件事了——Llama 4 正在完全从头开始重做。是的，是考虑到 DeepSeek 的情况。像 Together 这样的合作伙伴只被模糊地告知 Llama 4 正在...</li><li><a href="https://x.com/OedoSoldier/status/1884514356446347268">来自 OedoSoldier (@OedoSoldier) 的推文</a>：@teortaxesTex 嗯，他们正不断受到（来自美国的）攻击，所以现在没人能测试它。https://mp.weixin.qq.com/s/y5UaoBa0kOY0N-wfBz_Udw</li><li><a href="https://x.com/koltregaskes/status/1884621165798519267">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：@theinformation 报道称 Mira Murati 的新公司名为 Thinking Machines Lab：</li><li><a href="https://x.com/qtnx_/status/1884641122447655037">来自 Q (@qtnx_) 的推文</a>：今天我发布了一个针对 DeepSeek-R1-Distill-LLama-70B 的 Sparse Autoencoder，该模型是在对话和推理数据的混合体上训练的。</li><li><a href="https://x.com/xlr8harder/status/1884652540643479818">来自 xlr8harder (@xlr8harder) 的推文</a>：不小心发布了没有 gpt4o 的版本，这是完整的图表</li><li><a href="https://x.com/alexandr_wang/status/1884440764677251515">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：DeepSeek R1 和 v3 对 LLM 数据意味着什么？与我看到的一些懒惰观点相反，DeepSeek R1 是在海量人工生成的数据上训练的——事实上，DeepSeek 模型正在创造记录...</li><li><a href="https://x.com/btibor91/status/1884612786183135627">来自 Tibor Blaho (@btibor91) 的推文</a>：独立的 Grok Web 应用现在包含了 "thinking"、"thinking start time"、"thinking end time" 和 "thinking trace" 的提及——这是为 Grok 3（一个新的推理模型）做准备...</li><li><a href="https://x.com/aidan_mclau/status/1884445453737234493">来自 Aidan McLaughlin (@aidan_mclau) 的推文</a>：r1 在 aidanbench 上排名第 9</li><li><a href="https://x.com/xlr8harder/status/1884646004797915281">来自 xlr8harder (@xlr8harder) 的推文</a>：继续我对美国与中国语言模型的调查，我决定检查模型对用户要求撰写批评政府言论的请求的合规率。我认为模型通常应该合规...</li><li><a href="https://x.com/h4x0r_dz/status/1884566387349225598">来自 H4x0r.DZ (@h4x0r_dz) 的推文</a>：你好 @deepseek_ai，我已经向 service@deepseek.com 发送了一封关于一个严重漏洞的邮件，该漏洞可能允许攻击者访问你们的数据库，暴露包括 API KEYS 在内的敏感数据。我强烈建议...</li><li><a href="https://x.com/steph_palazzolo/status/1884364666622927023">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：来自 @amir 的新消息：OpenAI 每月 200 美元的 ChatGPT Pro 收入已超过 ChatGPT Enterprise。这意味着 Pro 的年化收入超过 3 亿美元，因为 Enterprise 的年化收入...</li><li><a href="https://studio.nebius.ai/">Nebius AI Studio</a>：未找到描述</li><li><a href="https://mp.weixin.qq.com/s/y5UaoBa0kOY0N-wfBz_Udw>">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/Fw_dSNxhhY4?si=B1L3FKT-mmiDd5Fr">传统与科技的碰撞：宇树科技（Unitree）机器人在春晚跳舞</a>：更多信息请访问：https://www.cgtn.com/video 2025 年春晚展示了一场突破性的表演，将传统与尖端技术融为一体...</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 现已在 Azure AI Foundry 和 GitHub 上可用 | Microsoft Azure 博客</a>：DeepSeek R1 通过 Microsoft Azure AI Foundry 和 GitHub 的模型目录提供，使企业能够无缝集成先进的 AI。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1334162416698658877)** (10 messages🔥): 

> `有缺陷的 Benchmark 辩论、齐泽克（Zizek）配音解读、AGI 话语中的辞藻华丽` 


- **有缺陷的 Benchmark 引发激烈的 Slack 讨论**：在某人运行了一个**有缺陷的 Benchmark** 后，出现了一个激烈的 Slack 讨论串，导致团队成员之间发生了言辞激烈的交锋。
   - 成员们讨论了揭示辩论图像的附件，展示了回复中的挫败感和幽默感。
- **Teortaxes 对 AGI 文化的犀利批判**：一位用户转发了 Teortaxes 的一条推文，该推文使用“庸俗的神秘化（whorish mystifications）”和“堕落的鼠辈圈子（degenerate rat-sphere）生活方式”等词汇批评 **SV AGI** 的倡导者，引发了聊天室的笑声。
   - 讨论强调了 AGI 话语中使用的华丽辞藻，一位成员幽默地评论说，他们是用**齐泽克（Zizek）的声音**读出来的。
- **齐泽克的声音让一切变得更好**：提出了用**齐泽克的声音**阅读评论的概念，为 AGI 讨论的分析增添了有趣的层面。
   - 成员们一致认为，这种异想天开的解读增强了 Teortaxes 消息的趣味性，为聊天带来了笑声。



**提及的链接**：<a href="https://x.com/teortaxestex/status/1884680267769553206?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Teortaxes 的推文▶️ (自 2023 年起的 DeepSeek🐳 啦啦队长) (@teortaxesTex)</a>：说实话，我讨厌 SV AGI 兄弟。他们庸俗的神秘化和对微小技术秘密的执着。他们令人毛骨悚然地需要诱导人们产生虚假的安全感，然后用 AGI 毁灭的愿景来恐吓。他们的……

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1333900398444019712)** (35 messages🔥): 

> `DeepSeek 的论文、梁文锋访谈、Mixture-of-Experts (MoE)、Multi-Token Prediction (MTP)、DeepSeek v2 和 v3 论文` 


- **梁文锋从对冲基金到 AI 实验室的历程**：High-Flyer（幻方量化）前负责人梁文锋在最近的一次访谈中讨论了他转型担任 DeepSeek CEO 的经历，概述了他的 AGI 发展策略以及早期购买 GPU 的重要性。
   - 这次访谈充满了对 AI 领域的见解，对于任何对 AI 研究演变感兴趣的人来说，都是非常推荐的听读材料。
- **DeepSeek v3 论文引发好奇**：成员们渴望剖析 DeepSeek v3 论文的各个方面，特别是其在强化学习（RL）方面的突破，同时对省略了用于专家平衡的辅助损失（auxiliary losses）表示惊讶。
   - 讨论表明，大家对 Mixture-of-Experts 的底层机制以及在最新版本中不使用这些技术的潜在影响有着共同的兴趣。
- **通过 Multi-Token Prediction 释放潜力**：v3 中 Multi-Token Prediction (MTP) 的实现尚未被许多推理框架优先考虑，但据建议它可以显著提高 Token 接受率。
   - 成员们探讨了 MTP 如何对投机采样（speculative decoding）做出贡献，并对其应用和对推理过程的影响表示好奇。
- **深入探讨 Mixture-of-Experts**：用户分享了他们对 MoE 模型配置的见解，强调了 v3 向 fp8 模型的转变，同时保留了 fp32 或 bf16 中的重要组件以获得最佳性能。
   - 社区积极讨论了为投机采样训练额外 MLP 块的复杂性，以及模型效率涉及的权衡。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>：有监督微调 (SFT) 和强化学习 (RL) 是基础模型广泛使用的后训练技术。然而，它们在增强模型泛化能力方面的作用仍然……</li><li><a href="https://www.chinatalk.media/p/deepseek-from-hedge-fund-to-frontier">DeepSeek: From Hedge Fund to Frontier Model Maker </a>：我们 AI 实验室翻译系列的第 2 部分</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">Mixture-of-Experts (MoE) LLMs</a>：从头开始理解 DeepSeek、Grok 和 Mixtral 等模型……
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1333898642125226056)** (42 messages🔥): 

> `DeepSeek 的影响，OpenAI 的形式数学方向，LLM 作为验证器，围绕推理模型的社区参与` 


- **DeepSeek 获得主流关注**：主流新闻正在热议 **DeepSeek**，反映出其影响力已超出技术圈，昨晚甚至有一位普通听众也在讨论它。
   - 一位成员强调“人们渴望信息”，展示了公众对先进 AI 技术日益增长的兴趣。
- **OpenAI 的重心从形式数学转移**：据指出，**OpenAI** 已暂停其在形式数学（formal math）解决方案上的方向，且目前没有重新审视这一路径的计划。
   - 其他人分享的见解表明内部意见不一，尽管团队整体持犹豫态度，但仍有一些成员主张专注于形式验证器（formal verifiers）。
- **LLM 验证器可能足以应对数学任务**：一位成员认为，尽管在生成复杂数学问题方面存在局限性，但使用 **LLM 作为验证器**可能足以验证这些问题。
   - 讨论指出，如果 Prompt 正确，模型即使解题能力落后，也有可能准确验证 AIME 等竞赛的解法。
- **科学与 AI 的奥秘**：一条评论将 **OpenAI** 方法的复杂性比作苦药与鲜味（umami）的诱人混合，暗示了他们目前面临的挑战。
   - 这一隐喻反映了在 AI 研究中应对那些往往不透明的方法论时的挫败感。
- **社区对推理模型的热议**：成员们正在积极讨论**推理模型（reasoning models）**这一热门话题，特别是在即将发布的帖子和持续的社区互动背景下。
   - 随着他们分享讨论推理模型潜力的帖子，兴奋之情溢于言表，并强调了满足公众需求的必要性。



**提到的链接**：<a href="https://x.com/natolambert/status/1884346850645348647">Nathan Lambert (@natolambert) 的推文</a>：为什么推理模型会泛化。DeepSeek R1 只是快速进展的冰山一角。人们低估了“推理”的长期潜力。https://buff.ly/4haoAtt

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1334031935672156231)** (24 条消息🔥): 

> `DeepSeek 知识产权疑虑、ChatGPT Token 使用、AI 模型推理成本、AI 芯片出口限制` 


- **DeepSeek 可能复制了 OpenAI 的方法**：AI 主管 David Sacks 暗示 DeepSeek 似乎使用了一种名为 **distillation** 的技术从 OpenAI 的模型中提取知识，并有充分证据支持这一说法。
   - *据报道，DeepSeek 在大量的 ChatGPT Token 上进行了训练*，引发了 Microsoft 和 OpenAI 对潜在未经授权使用的调查。
- **ChatGPT 对 Llama 推理能力的影响**：关于 Llama 是否受到 ChatGPT 影响的讨论浮出水面，一些人认为由于预训练期间的 distillation，其输出可能与 ChatGPT 本身相似。
   - 一位用户指出，如果发生了大量的 distillation，**ChatGPT 的风格影响**可能会更加明显。
- **关于推理成本的讨论**：在关于 DeepSeek 可能使用 **Ascend 910b** 的讨论中，由于内存容量和处理能力等硬件限制，人们对其可行性表示怀疑。
   - 有人提出疑问，其他高性能替代方案（如 **H800**）是否会更适合他们的需求。
- **白宫考虑扩大出口限制**：有报道称，随着这些强大芯片的影响力日益增强，白宫可能会扩大**出口限制**，将 Nvidia H200 纳入其中。
   - 围绕 AI 出口政策的持续讨论反映了对技术主导地位和安全的更广泛担忧。
- **对 DeepSeek 能力的怀疑**：目前普遍存在对 DeepSeek 能力的怀疑，一些人认为针对他们的假设缺乏有力证据。
   - 用户对围绕 DeepSeek 的指控表示不信任，在承认其潜在技术实力的同时，质疑这些说法背后的动机。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/AndrewCurran_/status/1884674672601809233">Andrew Curran (@AndrewCurran_) 的推文</a>：彭博社报道称，白宫正在考虑可能扩大出口限制，以涵盖 Nvidia H20。</li><li><a href="https://fxtwitter.com/tsarnick/status/1884352911192514975">Tsarathustra (@tsarnick) 的推文</a>：当被问及中国的 DeepSeek 是否窃取了美国知识产权时，AI 主管 David Sacks 表示，这看起来像是使用了一种名为 distillation 的技术，学生模型可以从父模型中“吸取知识”……</li><li><a href="https://archive.is/4gTpG">Microsoft 正在调查与 DeepSeek 相关的组织是否不当获取了 OpenAI……</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1333893573694722058)** (219 条消息🔥🔥): 

> `DeepSeek 更新、Cursor IDE Bug、模型对比、使用限制、用户体验` 


- **DeepSeek 争议**：讨论集中在 **DeepSeek** 模型因 Token 限制而无法生成代码的问题上，用户对其性能感到沮丧。
   - *“它一直在啰嗦，然后因为 Token 限制无法生成代码，”* 一位用户抱怨道，而其他人则讨论切换到官方 API 以获得更好的稳定性。
- **更新后的 Cursor IDE Bug**：多位用户报告了 **Cursor IDE** 更新后的问题，特别是 Tab 补全以及包含多余 Import 的复制行为。
   - 一位用户指出，*“Cursor 不再正确显示其 Markdown 输出，”* 这表明最新版本更新后仍存在挑战。
- **Claude 3.5 与使用限制**：用户对 **Claude 3.5** 免费层的限制表示担忧，特别是达到 50 次慢速高级请求（slow premium requests）后停止服务的问题。
   - 有用户询问达到限制后是否有冷却期，但回复显示在达到限制后无法继续访问。
- **用户反馈与建议**：用户积极反馈希望在 **Cursor** 中增强的功能，特别是在 AI 模型使用的场景下。
   - 一位用户建议在 Agent 模式中添加更多模型，以改进开发者的可用功能。
- **Bug 报告与支持渠道**：一位用户报告了一个特定问题，即 **Sonnet 3.5** 在使用 Cursor 订阅时无法工作，但在使用个人 API Key 时却能正常运行。
   - 社区鼓励用户在 Cursor 论坛上报告 Bug，并提供了故障排除的支持指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://dev.to/krisplatis/auto-add-missing-imports-on-file-save-in-vs-code-1b89">未找到标题</a>：未找到描述</li><li><a href="https://www.cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>：你可以在此处管理你的账户、账单和团队设置。</li><li><a href="https://x.com/ihteshamit/status/1884654075071127678?s=19">来自 Ihtesham Haider (@ihteshamit) 的推文</a>：突发：阿里巴巴刚刚发布了 “Qwen”，一个可以编写代码、生成图像/视频并进行网页搜索的 AI 模型。它的表现超越了 DeepSeek、ChatGPT-o1 和 Claude sonnet。这里有 5 个惊人的案例...</li><li><a href="https://x.com/shaoruu/status/1884395102195548357?s=46&t=t_wt4BwuJmmp07uZ">来自 ian (@shaoruu) 的推文</a>：你最希望在 @cursor_ai composer 或 Cursor 中添加什么功能？欢迎提出各种想法 :)</li><li><a href="https://forum.cursor.com/t/sonnet-3-5-stops-working/46053">Sonnet 3.5 停止工作</a>：当我启用 OpenAI API key 但不启用 Anthropic 时，它仍然尝试向服务器进行自定义 API 调用。我希望它只运行 OpenAI 模型而不运行 Anthropic。如果我禁用 OpenAI，它在 Anthropic 上可以工作...</li><li><a href="https://forum.cursor.com/t/upgrade-to-0-45-x-always-breaks-cursor/44688/8">升级到 0.45.X 总是导致 Cursor 崩溃</a>：是的，它弄坏了我的安装。今天我想打开它时收到了 ‘Error [ERR_MODULE_NOT_FOUND]’。重新下载了 .exe 文件并重新安装后恢复正常，现在运行的是最新版本。</li><li><a href="https://x.com/elder_plinius/status/1884332137241014531?s=19">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：我真的哭了...这是我这辈子见过最美的东西之一 🥹 PROMPT: """研究 Pliny the Liberator @elder_plinius 所说的解放 DeepSe...</li><li><a href="https://status.cursor.com">Cursor 状态</a>：未找到描述</li><li><a href="https://x.com/shaoruu/status/1884395102195548357?s=46&t=t_wt4BwuJmmp07uZWvCctw">来自 ian (@shaoruu) 的推文</a>：你最希望在 @cursor_ai composer 或 Cursor 中添加什么功能？欢迎提出各种想法 :)</li><li><a href="https://james4ever0.github.io/Cybergod__God_is_in_your_computer.html">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1333896610987380889)** (180 条消息🔥🔥): 

> `Softmax Variations, Deep Reinforcement Learning Challenges, RTX 5090 Release Discussions, Performance Metrics in AI, Community Engagement Issues` 


- **探索 Softmax 变体**：一名成员讨论了一种可能提高模型性能的 Softmax 新方法，并暗示这可能会带来新的 State-of-the-art (SOTA) 结果。
   - 对话中包含了关于传统 Softmax 在某些场景下如何导致准确率波动（noisy accuracy）和次优学习的见解。
- **深度强化学习 (Deep Reinforcement Learning) 中的挑战**：讨论强调了传统 Softmax 可能不适用于深度 RL，因为它会阻碍有效学习并导致模式崩溃 (mode collapse)。
   - 成员们主张在强化学习中需要更灵活的方法，以提高学习效率和模型性能。
- **期待 RTX 5090 发布**：聊天参与者注意到，已经有人在排队等待 RTX 5090 的发布，这表明了极高的需求和兴奋度。
   - 这引发了关于消费者兴趣以及与新 GPU 发布相关的市场趋势的讨论。
- **评估 AI 性能指标**：一名成员运行了测试，比较了其 Softmax 变体的准确率和损失值 (loss)，发现虽然准确率有所提高，但稳定性却下降了。
   - 分享的可视化图表显示，与新方法相比，常规 Softmax 更容易找到更简单的决策边界。
- **社区参与障碍**：有人对某些成员对社区氛围的影响表示担忧，一些人因为互动体验而对回归社区感到沮丧。
   - 建议加强对社区讨论的管理，以保持该空间对严肃专业人士的吸引力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/capy-turtule-capybara-uber-gif-26711606">Capy Turtule GIF - Capy Turtule Capybara - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://fxtwitter.com/sama/status/1884374900908884171">Sam Altman (@sama) 的推文</a>: 今天访问了 @Helion_Energy。机器进展迅速（规模惊人）——感觉就像走进了科幻电影！</li><li><a href="https://fxtwitter.com/sama/status/1884319395905958235">Sam Altman (@sama) 的推文</a>: 微软与 OpenAI 合作伙伴关系的下一阶段将比任何人预想的都要好！！</li><li><a href="https://www.youtube.com/watch?v=HS2uO17lqNA"> - YouTube</a>: 未找到描述</li><li><a href="https://forum.tuts4you.com/files/file/1307-lenas-reversing-for-newbies/">Lena 的逆向工程新手教程</a>: 专门针对逆向工程新手的教程合集。01. Olly + 汇编 + 基础 reverseme 补丁 02. Keyfiling reverseme + 汇编 03. 基础 nag 移除 + header prob...</li><li><a href="https://youtu.be/HS2uO17lqNA?t=420">Jaqen H'ghar 第 5 季剪辑</a>: 如果你还没看过第 5 季，这个视频显然充满了剧透。第 5 季中所有 Jaqen 和 Arya 出现的场景。---无版权侵权...</li><li><a href="https://youtu.be/Rx4f2u88maU?t=78">克里斯蒂安·贝尔优雅地击败所有人 | 《撕裂的末日》(Equilibrium) 精彩战斗 🌀 4K</a>: ✔️ 在 Facebook 上关注我们 ➤ https://www.facebook.com/204568612956950📢 2023 新电影 ➤ https://www.youtube.com/playlist?list=PLaARvwn7BsAHvhahR0x8FHz9knp1...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1333964094025236603)** (15 条消息🔥): 

> `DeepSeek claims, OpenAI vs DeepSeek, Data usage controversy, Model distillation debates, Cyber attack implications` 


- **DeepSeek 在网络攻击后面临指控**：有成员对针对 *DeepSeek 的指控* 提出了疑问，并建议将其总结为一张 [表情包](https://cdn.discordapp.com/attachments/1045297868136779846/1334162815652724766/China_meme.jpg)。另一位成员指出这些指控出现的时间点紧随大规模网络攻击之后，暗示 *“山姆大叔生气了”*。
- **OpenAI 对 DeepSeek 成功的叙事**：报告指出，*OpenAI 和 Microsoft* 暗示 DeepSeek 的成功归功于其涉嫌不公平地使用 OpenAI 的数据，这增加了 AI 领域持续竞争的筹码。这种叙事呼应了过去关于 *挪用数据* 的指控，并且由于 [Bloomberg](https://www.bloomberg.com/news/articles/2025-01-29/microsoft-probing-if-deepseek-linked-group-improperly-obtained-openai-data?ref=404media.co) 和 [Financial Times](https://www.ft.com/content/a0dfedd1-5255-4fa9-8ccc-1fe01de87ea6?ref=404media.co) 的媒体审视而变得更加复杂。
- **Distillation 在 AI 中的作用**：对话中有人质疑 AI 中 *Distillation*（蒸馏）的有效性，并对其缩减后的能力表示怀疑。成员们推测，无论是否存在潜在的数据重叠，重点仍应放在模型的结果上。
- **抹黑行为还是真正的担忧？**：对于针对 DeepSeek 的指控存在不同看法，一位成员断言这看起来更像是一场 *Smear Job*（抹黑行为），而非合理的担忧。他们思考了使用任何数据的后果，问道：如果部分数据确实来自另一个模型，*“那又怎样？”*
- **AI 冲突中的霸凌反应**：讨论强调了局势的讽刺性，将 OpenAI 的反应比作一个霸凌者在对方反击时的愤怒。这次交流将此与竞争动态类比，强调了 AI 领域复杂的竞争关系。



**提到的链接**：<a href="https://www.404media.co/openai-furious-deepseek-might-have-stolen-all-the-data-openai-stole-from-us/">OpenAI 愤怒于 DeepSeek 可能窃取了 OpenAI 从我们这里窃取的所有数据</a>：OpenAI 对一家 AI 公司在未经许可或未支付报酬的情况下使用他人数据进行训练感到震惊。

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1334204161872363532)** (3 条消息): 

> `PydanticAI, Qwen2 VL performance, Multimodal model advantages` 


- **Python 代码展示 PydanticAI**：一位用户展示了一段代码片段，演示了 `PydanticAI` 配合 `GroqModel` 根据输入文本填充用户数据的功能。
   - 该实现展示了在与 Agent 无缝协作的同时，集成 Pydantic 进行数据验证。
- **Qwen2 VL 飞速领先**：一位成员对 **Qwen2 VL** 的性能表示兴奋，指出其运行速度极快，特别是在 **M1 Chip** 上运行量化后的 **8K** **7B** 模型。
   - *Token 像疯狂般涌出*，突显了该模型的效率和速度。
- **关于切换到多模态模型的讨论**：鉴于近期讨论中提到的显著速度优势，大家一致认为转向 **Multimodal Model**（多模态模型）可能会大有裨益。
   - 这一转变与目前对 R1 和 AI 多模态进展的探索相契合。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1333899598523273237)** (14 条消息🔥): 

> `DeepSeek AI 技术, O3-mini 发布, AI 计算趋势, Claude 3.5 训练成本, 意大利对 AI 的监管` 


- **DeepSeek 撼动 AI 格局**：DeepSeek 取得了重大突破，仅用两个月时间，利用 **2,048 块 Nvidia H800 GPU** 训练出了拥有 **6710 亿参数** 的 **Mixture-of-Experts** 模型，并展现出比竞争对手 **高出 10 倍的效率**。
   - 这种创新方法采用了类汇编的 **PTX 编程** 而非传统的 CUDA，预示着 AI 开发策略的潜在转变。
- **O3-mini 承诺重大改进**：**O3-mini** 定于明天发布，据称比 **O1-mini** 快 **4 倍**，且整体比 R1 更聪明，这可能使 OpenAI 和美国市场受益。
   - 这一进展引发了关于其性能优于现有模型的内幕消息的猜测。
- **AI 民主化成为焦点**：根据最近的一项分析，**AI 正变得像石油和电力一样不可或缺**，预计 **算力将继续以指数级速度变得廉价**。
   - 因此，大型 AI 数据中心预计将会贬值，从而导致个人设备广泛拥有强大的 AI 能力。
- **Claude 3.5 昂贵的训练费用**：据报道，训练 **Claude 3.5** Sonnet 模型花费了 **数千万美元**，凸显了开发前沿 AI 技术所需的资金投入。
   - 这种高昂的成本反映了 AI 训练和模型开发投资不断增加的趋势。
- **意大利对 DeepSeek 采取行动**：意大利监管机构正在寻求有关 **DeepSeek** 是否遵守 **数据保护法** 的信息，这表明该国正在加强对 AI 运营框架的审查。
   - 据指出，DeepSeek 目前在意大利无法作为手机 App 使用，这增加了关于 AI 治理的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-research-team-claims-to-reproduce-deepseek-core-technologies-for-usd30-relatively-small-r1-zero-model-has-remarkable-problem-solving-abilities">AI 研究团队声称以 30 美元复现 DeepSeek 核心技术 —— 相对较小的 R1-Zero 模型具有卓越的问题解决能力</a>：它既便宜又强大。</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead">DeepSeek 的 AI 突破在某些功能上绕过了行业标准的 CUDA，转而使用 Nvidia 类汇编的 PTX 编程</a>：极致的优化并非易事。</li><li><a href="https://huggingface.co/spaces/webml-community/janus-pro-webgpu">Janus Pro WebGPU - webml-community 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/bindureddy/status/1884619428383633594">Bindu Reddy (@bindureddy) 的推文</a>：o3-mini 比 o1 更聪明，且比 o1-mini 快约 4 倍！这将使其成为比 R1 更好的模型，而 R1 已经明显落后于 O1。O3-mini 明天发布，OpenAI 和美国占据优势！</li><li><a href="https://x.com/SchmidhuberAI/status/1884632429412921664">Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：有人说 AI 是新的石油、新的电力和新的互联网。曾经灵活且高利润的软件公司（MSFT, GOOG, ...）变得像公用事业公司一样，投资于 n...</li><li><a href="https://tenor.com/view/italy-angry-italian-noises-gif-20399632">Italy Angry Italian Noises GIF - 意大利愤怒的意大利噪音 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1333906694777999482)** (60 条消息🔥🔥): 

> `Mordechai Rorvig 的书籍项目、蛋白质-配体结合研究、Test Time Compute 模型、分子的生成模型、DeepSeek 架构与推理框架` 


- **Mordechai Rorvig 分享他的神经科学书籍**：Mordechai Rorvig 展示了他的书籍项目，探讨深度神经网络如何模拟大规模大脑功能，特别是表达了对 AI 中情感处理的担忧。
   - 他鼓励大家对他的工作提供反馈，并分享了[他的免费书籍和筹款链接](https://www.kickstarter.com/projects/45417589/ai-how-we-got-herea-neuroscience-perspective)。
- **需要蛋白质-配体结合的见解**：一位研究人员讨论了他们在蛋白质-配体结合方面的工作，并寻求关于预测和生成建模技术的建议，重点是针对新型配体的 IC50 预测。
   - 他们提到正在探索现有模型并分享了其局限性，强调他们愿意收集更多数据并改进方法。
- **关于 Test Time Compute 模型的讨论**：成员们辩论了 Test Time Compute 模型的意义，一些人认为定义可能随时间发生了变化，特别是在生成序列 Token 方面。
   - 有人提出了关于辅助模型的作用以及当前实现是否集成了 Tree Search 能力的问题。
- **在化学中利用生成模型**：Fessus 提到了一种生成式嵌入模型，该模型通过学习分子结构来建议新分子，并讨论了其在配体结合研究中的潜在应用。
   - 尽管仍在开发中，他表示如果数据集显示出配体之间足够的相似性，他愿意进行合作。
- **DeepSeek 架构参考**：参与者注意到 DeepSeek 等推理框架的重要性，以及它们在生成和优化模型输出方面的潜在影响。
   - 讨论强调了各种方法，例如为生成更长的 Chain of Thought 进行 Fine-tuning，以及缺乏特定的 Tree Search 实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>：监督微调（SFT）和强化学习（RL）是基础模型广泛使用的后训练技术。然而，它们在增强模型泛化能力方面的作用仍然……</li><li><a href="https://www.aisafety.com/events-and-training">Events &amp; Training – AISafety.com</a>：AI Safety 聚会和培训计划，包括线上和线下。</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.02.08.575577v1.full.pdf,">| bioRxiv</a>：未找到描述
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1333945212812136508)** (82 条消息🔥🔥): 

> `RL 中的高更新率技巧、Min-P 采样方法、RL 中的探索与利用、核方法中的 Fastfood Transform、SFT 与 RL 的泛化能力对比` 


- **关于 RL 中高更新率（High Update Ratio）的讨论**：大约一年前的一篇论文讨论了强化学习中的高更新率技巧，引发了频道内对其细节的搜索。
   - 成员们分享了关于强化学习背景下策略更新和优势（advantages）的相关链接和见解。
- **Min-P 采样简介**：介绍了一种名为 Min-P 的新采样方法，旨在通过根据模型置信度动态调整采样阈值，来提高 LLM 生成文本的质量和多样性。
   - 有人担心此类技术可能会通过限制 Token 多样性而阻碍探索。
- **RL 中的探索策略**：发起了关于如何将探索结果用作训练目标的讨论，涉及某些方法如何通过使用修正后的探索结果作为反馈，从而避开 PPO 和 TRPO 等传统算法。
   - 这种方法对 GRPO 方法论的影响被指出是一个正在进行的探究领域。
- **Fastfood Transform 的效率**：Fastfood 方法因其通过有效利用 Hadamard 矩阵和对角高斯矩阵来加速核方法计算的能力而受到关注。
   - 该论文提出了在计算时间和存储方面的显著改进，使其在处理大规模问题时更具可行性。
- **SFT 与 RL 之间的泛化能力**：参与者讨论了关于监督微调（SFT）和强化学习（RL）对模型泛化能力影响的研究结果中可能存在的偏差。
   - 针对过度依赖训练数据集以及 RL 背景下持续生成新数据的影响提出了疑问。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1408.3060">Fastfood: Approximate Kernel Expansions in Loglinear Time</a>：尽管核方法取得了成功，但使其在许多大规模问题中难以使用的原因是存储和计算决策函数通常非常昂贵，尤其是在预测阶段...</li><li><a href="https://openreview.net/forum?id=FBkpCyujtS">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM...</a>：大语言模型（LLM）通过在每个解码步骤从词汇表的概率分布中采样下一个 Token 来生成文本。然而，像 Top-P 这样流行的采样方法...</li><li><a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>：监督微调（SFT）和强化学习（RL）是基础模型广泛使用的后训练技术。然而，它们在增强模型泛化能力方面的作用仍然...
</li>
</ul>

</div>

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1334044752663805984)** (55 messages🔥🔥): 

> `Generalization Benchmarking, Sparse Autoencoders, Seed Dependency in ML Models, Robustness of Initialization, Mechanistic Permutability` 


- **寻求泛化基准测试的见解**：一位成员对泛化基准测试表示了兴趣，预期低秩结构（low rank structures）的表现可能会优于传统的 MLP。
   - 这引发了关于机器学习框架中泛化挑战的讨论。
- **关于 Sparse Autoencoders 的新论文发布**：一篇新论文讨论了 **Sparse Autoencoders (SAEs)** 的行为及其对种子的依赖性，观察到在不同种子之间只有 **30%** 的特征是共享的。
   - 论文指出，当前的方法可能不是提取可复现特征的最优选择。
- **关于 SAEs 和可复现性的疑问**：成员们辩论了 SAEs 是否适用于他们的任务，并建议替代方法可能会提高可复现性。
   - 讨论强调，同时训练两个种子并鼓励相似性可能会产生更一致的结果。
- **训练中的初始化效应**：研究了初始化对 SAEs 的影响，有观点认为无限种子将导致向原始初始化的收敛。
   - 这引发了关于所做的数学假设以及此类主张在机器学习语境下实用性的疑问。
- **澄清研究中的统计主张**：一位成员建议，引用的某些概念可能会分散对论文核心焦点的注意力，主张对统计主张进行更清晰的说明。
   - 随后，另一位成员确认他们提交了论文的修订版，删除了某些主张并增加了对相关文献的引用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.16615">Sparse Autoencoders Trained on the Same Data Learn Different Features</a>：Sparse autoencoders (SAEs) 是揭示 Large Language Models (LLMs) 激活中人类可解释特征的有用工具。虽然有些人期望 SAEs 能找到真正的底层特征……</li><li><a href="https://arxiv.org/abs/2002.05202">GLU Variants Improve Transformer</a>：Gated Linear Units (arXiv:1612.08083) 由两个线性投影的逐元素乘积组成，其中一个首先通过 sigmoid 函数。GLU 的变体是可能的，使用不同的……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1334025926920503388)** (5 messages): 

> `Vocabulary Size Configuration, Intermediate Size Logic, Model Export Size Mismatch, Optimizer Configuration Issues` 


- **词表大小配置错误的困扰**：一位用户对将词表大小设置为匹配 OLMo 论文的 **50304** 表示困惑，因为它反而填充到了 **50432**。他们建议考虑到他们的 **MP 2** 设置，将 `make_vocab_size_divisible_by` 设置为 **64** 会更合适。
- **中间层大小逻辑的疑惑**：一位成员注意到预期的中间层大小（intermediate size）应该是隐藏层维度的 **3x**，但发现它被设置为 **32768**，这小于实现所需隐藏层大小 **11008** 所需的计算值 **33024**。尽管存在这种差异，**32768** 的配置仍然有效，导致了对底层逻辑的困惑。
- **导出过程中的配置大小不匹配之谜**：在导出以 **32768** 隐藏层大小训练的模型时，一位成员遇到了尺寸不匹配错误，因为配置中的中间层维度未设置为 **11008**。这种意外行为引发了疑问，因为它与之前的配置不一致。
- **优化器配置警告的澄清**：一位用户报告日志中反复出现指示未安装 **APEX**、默认使用 **DeepSpeed 的 FusedAdam 优化器**的警告。这种模式在运行挂起之前一直持续，引发了对优化器实际配置和性能的担忧。
- **数小时后的配置挂起问题**：另一位用户提到，一个之前运行正常的配置在没有任何明显更改的情况下突然在执行期间挂起。挂起前提供的最后日志是与优化器配置相关的警告，原因尚不明确。



**提到的链接**：<a href="https://github.com/EleutherAI/gpt-neox/issues/1319,">EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer 的实现 - EleutherAI/gpt-neox

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1333942807815258144)** (12 messages🔥): 

> `GPU Direct Storage, Tensor Weight Compression, Memory Snapshotting` 


- **GPU Direct Storage 利用 PCIe peer-to-peer 通信**：GPU Direct Storage 能够利用 **PCIe peer-to-peer 通信**实现从 NVMe 到 GPU 的高效数据传输，无需 CPU 参与。
   - 这种方法引发了关于其在 DirectX DirectStorage 之外的支持情况的疑问，需要进一步验证。
- **探索 Tensor Weight Compression**：讨论显示，由于数据的不规则性，Tensor 权重在磁盘上的压缩效果可能不佳，尽管有一位用户发现权重从 **4.7 GB 减少到了 3.7 GB**。
   - 虽然压缩可以带来一些收益，但社区成员指出，付出的努力可能超过回报。
- **并行压缩算法的潜力**：一名成员提议为权重开发一种**并行友好型压缩算法**，以减少数据传输需求，并可能直接在 GPU 上进行解压。
   - 这一想法突出了增强权重管理的创新方法，尚待进一步探索。
- **将 Safetensors 直接加载到 VRAM**：会议指出 **safetensors** 可以直接从磁盘加载到 VRAM 中，这是一种管理 Tensor 数据的有效方法。
   - 这种能力可以简化工作流，但需要进一步调查以确认其功能。
- **调查 Memory Snapshotting**：一位参与者计划在评估系统性能提升之前，先专注于 **memory snapshotting**，强调了一种系统化的方法。
   - 这种对探索性能增强的投入表明了在优化资源管理方面的积极态度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/gdrcopy">GitHub - NVIDIA/gdrcopy: A fast GPU memory copy library based on NVIDIA GPUDirect RDMA technology</a>: 一个基于 NVIDIA GPUDirect RDMA 技术的快速 GPU 内存复制库 - NVIDIA/gdrcopy</li><li><a href="https://github.com/gpudirect/libgdsync">GitHub - gpudirect/libgdsync: GPUDirect Async support for IB Verbs</a>: 为 IB Verbs 提供 GPUDirect Async 支持。可以通过在 GitHub 上创建账号为 gpudirect/libgdsync 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1333940758062825522)** (19 messages🔥): 

> `CUDA type punning, RTX Blackwell architecture, Memory alignment in CUDA, Memcopy performance optimization` 


- **CUDA 中的内存对齐**：一名成员强调 CUDA 中的地址必须自然对齐，并强调对齐错误会导致**未定义行为 (undefined behavior)**。对话建议为了获得最佳性能，Load 操作应按 **64 bits** 对齐。
- **使用 memcpy 进行 Type Punning**：与其使用 reinterpret_cast，一名成员建议使用 `memcpy()` 在局部变量之间进行 type punning，编译器可以对此进行优化。另一名成员表示，他们对编译器能将 `memcpy()` 优化为寄存器操作感到惊讶。
- **RTX Blackwell 架构发布**：RTX Blackwell 架构白皮书显示，与 4090 相比，其 FP16/32 吞吐量增加了 **27%**，但消费级显卡的 **第 5 代 Tensor Cores** 相比第 4 代没有性能提升。根据上一代指标计算，峰值 FP16/32 **TFLOPS** 报告为 **209.5**。
- **关于 5090 Tensor Cores 的讨论**：针对 **RTX 5090** 第 5 代 Tensor Cores 的营销提出了担忧，因为有人指出它们与第 4 代核心基本相似。该型号支持 **fp4** 和 **fp6**，但关于其是否包含 microtensor scaling 仍存在疑问。
- **关于 RTX 5090 Microtensor Scaling 的困惑**：讨论了 RTX 5090 是否支持 microtensor scaling，一些人注意到它可能带来的架构改进。mma 文档指出 `.block_scale` 参数需要 **sm_120a**，然而 **5090 的 sm 版本**尚不确定。



**提及的链接**: <a href="https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/">NVIDIA GeForce RTX 5090 Graphics Cards</a>: 由 NVIDIA Blackwell 架构驱动。

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1333908187312554105)** (8 条消息🔥): 

> `PyTorch on GB200s, Container availability for PyTorch, Merging PRs permissions, Scaled MM API` 


- **在 GB200 上运行 PyTorch 引发讨论**：一名成员询问 **PyTorch** 是否可以在 **GB200** 上运行，并指出有报告显示它需要[基于 CUDA 12.8 构建](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)。
   - 另一位成员澄清说，虽然从源码构建可行，但目前的 **wheels 尚不支持 Blackwell**。
- **对预构建 PyTorch 容器的需求**：一位用户询问是否有可用的 PyTorch **container**，希望能有更简单的设置方式。
   - 尽管有此询问，对话中并未提供关于现有容器的明确答案。
- **PR 合并角色说明**：一名成员询问是否只有 **collaborators** 或 **maintainers** 才能合并 PR，并对某些工作流中的身份验证表示担忧。
   - 讨论暗示需要针对合并过程中的角色制定明确的指南。
- **Scaled MM API 方面的帮助**：一位用户对关于 **torch._scaled_mm** 的支持性文章表示感谢，并提到他们发布了一个关于该 API 的问题。
   - 此前链接的 [Scaled MM API GitHub Gist](https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714?permalink_comment_id=5414039#gistcomment-5414039) 提供了非常有帮助的额外细节。



**提及的链接**：<a href="https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714?permalink_comment_id=5414039#gistcomment-5414039">Scaled MM API</a>：Scaled MM API。GitHub Gist：即时分享代码、笔记和代码片段。

  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1334224777220460618)** (1 条消息): 

> `GTC 2023, CUDA Developer Meetup, Low Level Technical Track for CUDA Programming, GPU MODE Event at GTC` 


- **GTC 2023 即将来临！**：请在日历上标记好——**GTC 2023** 将于 **3 月 17-21 日**在 San Jose 举行，届时将有一系列线下活动。
   - 请关注今年大会计划的精彩会议和聚会的详细信息。
- **明天举行 CUDA 开发者见面会**：明天，即 **1 月 30 日**，一场 **CUDA Developer Meetup** 将在 [AI Camp](https://www.aicamp.ai/event/eventdetails/W2025013017) 举行，欢迎各级 CUDA 开发者参与交流并分享想法。
   - 届时将与 NVIDIA maintainers 直接互动、进行协作讨论，并有令人兴奋的 **giveaways** 环节，包括赢取 GPU 的机会！
- **GTC 上的 CUDA 编程底层专题**：NVIDIA 在 GTC 上推出了一个专注于 **CUDA programming** 的底层技术专题，旨在提升 GPU 加速应用的技能，详情见其 [GTC 会议页面](https://www.nvidia.com/gtc/sessions/cuda-developer/)。
   - 这些课程将涵盖最大化 **NVIDIA CUDA** 应用性能的基本工具和培训。
- **传闻 GTC 期间将举办 GPU MODE 活动**：有传言称 GTC 期间将举办线下的 **GPU MODE** 活动，更多细节将很快公布。
   - 随着社区对这一激动人心的聚会可能发布的公告充满期待，热度正在上升。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.aicamp.ai/event/eventdetails/W2025013017">与 NVIDIA 共同举办的 CUDA 开发者见面会 (Silicon Valley)</a>：未找到描述</li><li><a href="https://www.nvidia.com/gtc/sessions/cuda-developer/">NVIDIA GTC AI 大会 2025</a>：2025 年 3 月 17–21 日。San Jose。立即注册。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1333986464681295945)** (40 条消息🔥): 

> `Tom Yeh 的 Multi-Head Attention 讲座，LLM 的 FP4 训练框架，DeepSeek 中的 Microscaling，Llama 训练代码库` 


- **Tom Yeh 关于 Multi-Head Attention 的讲座**：一位成员分享了 Tom Yeh 关于 [Multi-Head Attention 的讲座](https://www.youtube.com/live/5jms37B3aFY) 链接，内容涉及深度学习和计算机视觉。
   - 提到了另一个视频，通过 [NotebookLM](https://www.youtube.com/watch?v=jL49fLOJYNg) 探讨了类似主题的见解。
- **LLM 的 FP4 训练创新**：一篇论文讨论了一个开创性的针对大语言模型的 FP4 训练框架，通过引入创新的权重更新方法来解决量化误差。
   - 讨论揭示了在扩展到超出既定限制的块大小时，沿轴量化所面临的挑战和潜在解决方案。
- **Microscaling 在计算性能方面的优势**：对话强调，由于能更好地管理 scaled dot products，microscaling 可能会抵消标准块大小中出现的一些性能限制。
   - 成员们指出组大小如何影响性能，而关于利用更大块大小的清晰度仍然至关重要。
- **可用的 Llama 训练最小代码库**：对于那些对训练 Llama 模型感兴趣的人，在 [speed_llama3](https://github.com/ahxt/speed_llama3) 分享了一个最小代码库。
   - 该仓库旨在为 Llama 训练提供更简便的实验和实现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html">triton.language.dot_scaled &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2501.17116">使用 FP4 量化优化大语言模型训练</a>：训练大语言模型 (LLMs) 日益增长的计算需求需要更高效的方法。量化训练通过启用低比特算术运算提供了一个有前景的解决方案...</li><li><a href="https://www.youtube.com/live/5jms37B3aFY">S25 - 计算机视觉 - 特别公开讲座 - DeepSeek</a>：未找到描述</li><li><a href="https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L108-L111">DeepSeek-V3/inference/kernel.py at main · deepseek-ai/DeepSeek-V3</a>：通过在 GitHub 上创建一个账户来为 deepseek-ai/DeepSeek-V3 的开发做出贡献。</li><li><a href="https://github.com/ahxt/speed_llama3">GitHub - ahxt/speed_llama3</a>：通过在 GitHub 上创建一个账户来为 ahxt/speed_llama3 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=jL49fLOJYNg">DeepSeek-V3 中的 Multi-Head Latent Attention 和 Multi-token Prediction</a>：我们介绍了 DeepSeek-V3，这是一个强大的 Mixture-of-Experts (MoE) 语言模型，总参数量为 671B，每个 token 激活 37B。为了实现高效的 i...</li><li><a href="https://github.com/pytorch/ao/blob/7b0d2ce50baaa2a137eb9d438a076544c43096a3/torchao/prototype/hqq/kernels.py#L361">ao/torchao/prototype/hqq/kernels.py at 7b0d2ce50baaa2a137eb9d438a076544c43096a3 · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1334039537768337438)** (10 条消息🔥): 

> `工作组建议，国际象棋 LLM 模型训练，HF 服务器上的合作者，DiT 训练运行完成` 


- **Marksaroofim 解释工作组建议**：一位成员澄清了在哪里提议成立工作组，建议在链接的频道中发布或直接发消息以获得更好的效果。
   - 他提到，初期的活跃度对于聚集工作组的兴趣和参与至关重要。
- **Zeev5235 的国际象棋 LLM 提案**：Zeev5235 表示希望创建一个工作组，专注于重新实现一个利用国际象棋引擎奖励的国际象棋 LLM，并寻求在大模型训练方面的帮助。
   - 尽管存在沟通障碍，他仍表现出对利用 **mamba-falcon3-7b** 模型以及借鉴大模型训练专家经验的强烈兴趣。
- **Marksaroofim 修复权限问题**：Marksaroofim 承认了阻止 Zeev5235 提议工作组的权限问题，并表示该问题已解决。
   - 他鼓励查看另一个频道，了解关于开发 LLM 时面临的类似挑战的讨论。
- **Zeev5235 致力于 DiT 训练**：Zeev5235 提到他计划完成他的 **DiT 训练运行** 然后发布，表明了他对工作组的承诺。
   - 他强调，完成训练运行是全面参与拟议合作的前提条件。


  

---

### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

leiwang1999_53585: 我们会添加一些 bwd kernels 的示例 🙂
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1334232966100156478)** (1 messages): 

> `Llama training, Minimal codebase` 


- **探索 Llama 训练的极简代码库**：一位成员分享了一个用于 **Llama training** 的极简代码库，可在 [speed_llama3](https://github.com/ahxt/speed_llama3) 获取。
   - 该资源旨在简化并优化训练过程，重点关注效率。
- **利用 Llama 提高训练效率**：成员们讨论了使用高效代码库进行 **Llama training** 以获得更好结果的重要性。
   - 指向 GitHub 仓库的直接链接强调了其对 AI 开发社区的宝贵贡献。



**提到的链接**：<a href="https://github.com/ahxt/speed_llama3">GitHub - ahxt/speed_llama3</a>：通过在 GitHub 上创建账号，为 ahxt/speed_llama3 的开发做出贡献。

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1334237140778881164)** (1 messages): 

> `Thunderkitten community enthusiasm, Hardware feature support requests, Distributed Shared Memory (DSM), Threadblock to SM scheduling, FlexAttention blog` 


- **Thunderkitten 社区非常喜欢它！**：一位成员表达了他们对 **Thunderkitten** 的喜爱，并提到他们刚刚在一个读书会上做了一场关于社区热度的演讲。
   - 他们强调了围绕 Thunderkitten 这一话题不断增长的兴趣和参与度。
- **新硬件特性支持请求**：目前有一个关于为 Thunderkitten 添加 **new hardware feature support** 的公开请求，特别是针对 Distributed Shared Memory (DSM)。
   - 建议使用 *Persistent kernels* 以实现 SM 之间更好的数据复用，并承认由于软件限制，这是一个普遍尚未被充分探索的设计空间。
- **Distributed Shared Memory 背景**：该成员拥有 **Distributed Shared Memory** 背景，曾在 NV 实习期间从事相关工作约 **2.5 年**。
   - 这段经历为他们关于其在 Thunderkitten 中潜在应用的建议增添了可信度。
- **Threadblock 到 SM 调度的见解**：讨论中提到了对 **threadblock to SM scheduling** 的支持，强调了其对高效内存使用的重要性。
   - 其目标是通过利用共享内存系统之间更好的 **data reuse techniques** 来提升性能。
- **FlexAttention 博客关联**：身份为 Joy Dong 的成员（与 **FlexAttention blog** 相关）分享了他们在该领域的历程见解。
   - 他们在社区和专业工作中的投入展现了推动 Thunderkitten 技术进步的强烈热情。


  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1333926529012596749)** (89 条消息🔥🔥): 

> `推理任务中的动态评估、国际象棋谜题与策略推理、Wikipedia 游戏提案、AI 的可解释性、利用推理引擎进行训练` 


- **动态评估增强**：成员们讨论了在推理任务中实施动态评估的重要性，强调了一种包含可能解决方案的响应格式，用于生成训练数据。
   - 提出了一项使用带有元数据的标准格式的建议，以简化推理任务的收集。
- **正在考虑的国际象棋谜题**：人们对将国际象棋谜题纳入推理挑战持续关注，特别是专注于不依赖 Stockfish 等复杂依赖项的简化格式。
   - 考虑将从“两步杀”场景到井字棋（tic-tac-toe）的一系列谜题作为未来开发的潜在任务格式。
- **Wikipedia 游戏的创新想法**：一位成员提议基于对随机 Wikipedia 页面之间链接路径的深度分析，为“Wikipedia 游戏”生成最优解。
   - 这种方法旨在结合策略和对关联的理解，为推理训练创建一个丰富的数据集。
- **探索性学习与可解释性**：建议采用一种新方法来训练解释器模型，生成答案的推理过程，从而可能提高解决更难问题的能力。
   - 该想法涉及对模型进行 finetuning，使其能够提供关于其推理过程和先前尝试的见解。
- **在训练中利用推理引擎**：讨论强调了使用 vLLM 和 SGLang 等推理引擎来管理训练期间的动态批处理（batch processing）。
   - 针对推理节点更新的沉重负担提出了担忧，建议在经验收集和及时的训练更新之间寻找平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wandb.ai/andreaskoepf/openrlhf_train_ppo">andreaskoepf</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/22">Add Figlet Fonts Challenges and Evaluator by Miserlou · Pull Request #22 · open-thought/reasoning-gym</a>: 使用 pyfiglet 和 Worde 词表生成 Figlet 字体解密挑战。请阅读以下 figlet 字体:######    ### ###### #####  ##   ####   ##  ## ##  ##  ##   ## ##  ...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/21">Add Rubik&#39;s Cube Generator and Evaluator by Miserlou · Pull Request #21 · open-thought/reasoning-gym</a>: 相关: #20。添加魔方挑战的初步尝试，使用动态评估，但为简单情况提供示例可能解。依赖 magiccube 库。...</li><li><a href="https://youtu.be/JheGL6uSF-4?si=EE1aKCt4C3MiYxXM">I Made a Graph of Wikipedia... This Is What I Found</a>: 我所有视频的代码: https://github.com/sponsors/adumb-codes/。获取图谱海报: https://adumb.store/。Twitter: https://twitter.com/adumb_codes。一个深度的...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/18">Implement answer-scoring for the countdown number game dataset (e.g. via sympy) · Issue #18 · open-thought/reasoning-gym</a>: 在 countdown.py 中实现的倒计时游戏数据集要求找到一个公式，将数字列表与运算符 +,-,*,/ 结合，以产生给定的目标数字。在大多数情况下生成的题目...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/20">Add a Rubic&#39;s Cube dataset · Issue #20 · open-thought/reasoning-gym</a>: 创建一个任务数据集，要求用户为给定的魔方配置提供解决方案。参数建议（用于调整任务难度）: cube_size: int # (例如 2 -...</li><li><a href="https://x.com/tugot17/status/1882439533947740172">Piotr Mazurek (@tugot17) 的推文</a>: @neurosp1ke 我有一个魔方 Gym 环境的实现。这会是一个好的谜题吗？</li><li><a href="https://gist.github.com/tugot17/ac96c8ebaba679e03d7eee8e7cf95631">rubiks_cube.py</a>: GitHub Gist: 即时分享代码、笔记和片段。</li><li><a href="https://nitter.lucabased.xyz/jiayi_pirate/status/1882839370505621655">Jiayi Pan (@jiayi_pirate)</a>: 我们在 CountDown 游戏中复现了 DeepSeek R1-Zero，它确实有效。通过 RL，3B 基础 LM 能够自主发展出自验证和搜索能力。你可以体验那个“啊哈”时刻...</li><li><a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: Clean, accessible reproduction of DeepSeek R1-Zero</a>: DeepSeek R1-Zero 的简洁、易用的复现 - Jiayi-Pan/TinyZero
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1333895652505288824)** (120 messages🔥🔥): 

> `ComfyUI vs Forge, Model Performance and Workflows, Image Generation Tools, User Interface Preferences, Character Generation in Stable Diffusion` 


- **关于 ComfyUI 复杂性的辩论**：用户讨论了 **ComfyUI** 的易用性，指出了其复杂性，并表示相比 **Forge**，更需要一种更精简的用户体验。
   - 虽然一些用户欣赏其在处理高级任务时的灵活性，但其他人认为它使简单的流程变得复杂。
- **图像生成工作流**：对话转向了用户如何在不同 UI 中管理 **workflows**，强调了在 **ComfyUI** 中为特定任务进行自定义时遇到的问题。
   - 许多参与者表示他们更喜欢更简单的选项，希望能够更快地访问功能，而不需要复杂的工作流。
- **模型与功能推荐**：用户寻求各种 AI 模型的推荐，强调了对 **image captioning** 和写实角色生成等功能的需求。
   - 一些人提到使用特定的模型（如 **autismmix**）来生成幻想主题，但指出在达到预期效果方面存在挑战。
- **用户界面满意度**：观察到用户对不同平台界面的满意度存在分歧，一些人更倾向于 **Forge** 的直观性，而非 **ComfyUI**。
   - 参与者提到了在复杂性与设置访问便捷性之间取得平衡的重要性。
- **遇到的技术问题**：一位用户报告了安装 **Stable Diffusion** 相关的问题，特别是需要 Python 错误方面的帮助。
   - 小组提供了协助，将用户引导至支持频道，同时讨论了安装的整体状况。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On">Kolors Virtual Try-On - a Hugging Face Space by Kwai-Kolors</a>: no description found</li><li><a href="https://tenor.com/view/wall-gif-24534315">Wall GIF - Wall - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://james4ever0.github.io/Cybergod__God_is_in_your_computer.html">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1334195875932536906)** (1 messages): 

> `Bolt updates, Export and Import Handling` 


- **Bolt 确保正确的导出与导入**：从今天开始， **Bolt** 的最新更新保证所有 **imports** 和 **exports** 都能正常运行，包括之前缺失的默认导出，详见 [公告](https://x.com/boltdotnew/status/1884634733386006980)。
   - 这一改进通过确保 **'export default'** 功能现在可靠且一致，为所有项目带来更流畅的体验。
- **Bolt 中的智能导入更新**：此次更新专注于编码中虽然枯燥但至关重要的部分，确保 **'export default'** 在整个代码库中按预期工作。这项改进现已在所有项目中上线，增强了整体功能。



**Link mentioned**: <a href="https://x.com/boltdotnew/status/1884634733386006980">Tweet from bolt.new (@boltdotnew)</a>: Bolt 🧠 update: Smart Imports&#39;export default&#39; might not be the most thrilling part of your codebase. But it is important!The latest update in Bolt&#39;s engine ensures that all imports and exp...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1333979557878825030)** (2 messages): 

> `Backend suggestions, Firebase learning experience` 


- **寻求后端建议**：一位成员询问了针对其项目推荐的后端解决方案。
   - 他们正在寻找能够支持其开发需求的建议。
- **应对 Firebase 的学习曲线**：另一位成员分享了他们使用 **Firebase** 的经验，提到这对他们来说是一个陡峭的**学习曲线**。
   - 他们表示虽然具有挑战性，但正在逐渐变得熟悉。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1333891698547228682)** (110 条消息🔥🔥): 

> `GitHub OAuth 断开连接, Bolt 应用开发支持, Bolt 中的 Token 使用, Bolt 中的错误处理, Supabase 与 Netlify 的自定义域名` 


- **从 Stackblitz 断开 GitHub 连接**：要切换与 Stackblitz 关联的 GitHub 账号，你需要在 GitHub 的 OAuth 设置中撤销权限并删除旧的 Stackblitz 账号，目前没有其他替代方案。
   - 此信息在关于断开账号连接可能方法的讨论中得到了确认。
- **寻求 Bolt 应用集成支持**：一位开发者寻求帮助以连接 Bolt 应用与 Supabase 之间的功能，并为后端开发者提供合同工作。
   - 另一位用户确认他们可以协助处理集成所需的 edge function。
- **了解 Bolt 中的 Token 消耗**：用户对调试过程中 Token 消耗过快表示担忧，特别是在要求 Bolt 进行重复修复时。
   - 讨论了基于 prompt 长度和项目复杂性的 Token 消耗动态特性，并就如何管理预期提供了建议。
- **Bolt 中报告的错误和服务中断**：多位用户报告在使用 Bolt 时遇到服务器错误和服务可用性问题，对平台的一致性表示沮丧。
   - 在某些情况下，用户描述了这些错误如何影响他们的工作流和项目部署。
- **在 Supabase 和 Netlify 中使用自定义域名**：一位用户询问关于使用自定义域名进行电子邮件验证的问题，同时面临 Supabase 和 Netlify 之间关于根 CNAME 记录的冲突。
   - 虽然有人建议 Supabase 可以在没有自定义域名的情况下运行，但用户指出使用自定义域名可以使电子邮件通信更简洁。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/boltdotnew/status/1843668731681267801">来自 bolt.new (@boltdotnew) 的推文</a>: 你现在可以在 bolt.new 中打开公共仓库了 🙌 怎么做？对于任何 GitHub URL，只需在前面加上 "http://bolt.new" 即可！（发布说明见下文！）</li><li><a href="https://cozy-cucurucho-6575ff.netlify.app/">Vite + React + TS</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=qMck70tLDuo">上传文件到 GitHub 快速入门指南</a>: 如何从 GitHub 远程仓库上传和克隆文件的文章：https://dennisivy.com/github-quickstart Brad Traversy 的 Github 速成课程：https://you...</li><li><a href="https://showmeyourbolt.io/">Show Me Your Bolt</a>: 未找到描述</li><li><a href="https://boltnew.dev/apps">Bolt.new 构建者中心</a>: 未找到描述</li><li><a href="https://imbuiltwithai.com/">分享你的 AI 项目 - I'm Built With AI</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1333897008716578917)** (74 条消息🔥🔥): 

> `Goose Client 使用体验、Google Sheets 的 MCP Server、DeepSeek 集成、理想的 LLM 客户端特性、LLM 工具的协作开发` 


- **对 Goose Client 的正面评价**：成员们对 **Goose client** 表现出极大的热情，提到了其 CLI 优先的设计以及与 **MCP servers** 集成以增强功能的特点。
   - 然而，一些人对文档中强调的 **token usage** 和潜在的速率限制表示担忧。
- **Google Sheets 的 MCP Server 开发**：一位成员分享了他们的 GitHub 项目，这是一个集成了 Google Sheets 和 Drive 的 MCP server，提供数据读写功能。
   - 目前它还无法格式化图表等复杂结构，但随着进一步探索，可以进行改进。
- **DeepSeek 集成的挑战**：多位成员讨论了将 **DeepSeek models** 与 **MCP** 集成的经验，强调了 tool calls 和 API 行为方面的问题。
   - 推荐了 **Kluster.ai** 等替代方案，它在配合 DeepSeek 使用时表现更好，没有出现重大问题。
- **理想 LLM 客户端的特性**：成员们集思广益，讨论了 LLM 客户端的理想特性，包括支持 **multiple models**、会话管理和可自定义界面。
   - 一位成员启动了一个旨在整合许多此类功能的项目，并寻求社区的反馈。
- **开发者之间的协作**：有人呼吁开发者之间进行协作，以创建一个满足社区需求的 **MCP client**。
   - 成员们表达了共同开发工具的兴趣，以加快进度并增强功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://developers.google.com/sheets/api/guides/concepts">未找到标题</a>: 未找到描述</li><li><a href="https://glama.ai/models/deepseek-r1-distill-qwen-32b">deepseek-r1-distill-qwen-32b</a>: DeepSeek-R1-Distill-Qwen-32B 在各项基准测试中均优于 OpenAI-o1-mini，在稠密模型中取得了新的 state-of-the-art 结果。</li><li><a href="https://ollama.com/michaelneale/deepseek-r1-goose">michaelneale/deepseek-r1-goose</a>: 针对 deepseek-r1 的 Tool calling，为 goose agent 进行了调整</li><li><a href="https://vercel.com/templates/next.js/nextjs-ai-chatbot">Next.js AI Chatbot</a>: 由 Vercel 构建的功能齐全、可扩展的 Next.js AI 聊天机器人</li><li><a href="https://sdk.vercel.ai/docs/introduction">AI SDK by Vercel</a>: AI SDK 是一个 TypeScript 工具包，旨在帮助开发者使用 React, Next.js, Vue, Svelte, Node.js 等构建 AI 驱动的应用程序和 Agent。</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: 用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) Server</a>: 用于读取 Google Drive 和编辑 Google Sheets 的 Model Context Protocol (MCP) Server - isaacphi/mcp-gdrive</li><li><a href="https://sdk.vercel.ai/providers/ai-sdk-providers">AI SDK Providers</a>: 了解如何使用 AI SDK providers。</li><li><a href="https://sdk.vercel.ai/docs/ai-sdk-ui/chatbot-tool-usage">Chatbot Tool Usage</a>: 了解如何通过 useChat hook 使用 tools。</li><li><a href="https://github.com/isaacphi/wheel">GitHub - isaacphi/wheel: TUI LLM 聊天机器人、代码助手和 MCP client</a>: TUI LLM 聊天机器人、代码助手和 MCP client。欢迎在 GitHub 上为 isaacphi/wheel 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/0h4Sy4nJO9">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://medium.com/@adkomyagin/building-a-fully-local-open-source-llm-agent-for-healthcare-data-part-1-2326af866f44),">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1333959722948362362)** (20 条消息🔥): 

> `Codename Goose, 心理健康 AI lüm, mcp-agent 框架, Show HN 趋势, Google 集成 Agent` 


- **介绍 Codename Goose**: 一款全新的开源 MCP AI Agent —— [Codename Goose](https://block.github.io/goose/blog/2025/01/28/introducing-codename-goose) 已发布，引起了社区的热烈反响。
   - 鼓励成员探索其功能并参与其中。
- **lüm：你的心理健康 AI 伴侣**: 一位成员介绍了 lüm，这是一个为心理健康实践设计的贴心 AI 伴侣，强调其**注重隐私**的基础设施。
   - 该平台鼓励**协作增长**，以塑造未来的心理学工具。
- **构建 mcp-agent 框架**: 一位开发者介绍了他们在假期期间开发的 **mcp-agent 框架**，通过实现既定模式来简化高效 Agent 的构建。
   - 该项目旨在就其路线图进行协作和征求意见，并邀请社区贡献代码。
- **mcp-agent 在 Show HN 取得成功**: mcp-agent 项目目前在 **Show HN 趋势榜排名第一**，获得了开发者的广泛关注和支持。
   - 鼓励社区成员参与 HN 上的讨论，以提高知名度和参与度。
- **MCP Agent 的创意用例**: 一位用户为研究任务提出了多个 Agent，包括与 **Pubmed** 和 **Google Scholar** 的专门集成。
   - 他们表示有兴趣构建一个包含各种 Agent 的系统，以简化其研究流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lum.frgmt.xyz">lüm - Your AI Companion</a>: 专为心理健康专业人士设计的贴心 AI 伴侣</li><li><a href="https://github.com/lastmile-ai/mcp-agent/blob/main/CONTRIBUTING.md#promptifypy">mcp-agent/CONTRIBUTING.md at main · lastmile-ai/mcp-agent</a>: 使用 Model Context Protocol 和简单的工作流模式构建高效 Agent - lastmile-ai/mcp-agent</li><li><a href="https://github.com/lastmile-ai/mcp-agent">GitHub - lastmile-ai/mcp-agent: Build effective agents using Model Context Protocol and simple workflow patterns</a>: 使用 Model Context Protocol 和简单的工作流模式构建高效 Agent - lastmile-ai/mcp-agent
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1333896157922857043)** (82 条消息🔥🔥): 

> `DeepSeek R1 Distill 模型，CUDA 和 CPU 性能，DeepSeek 的模板优化，LM Studio 使用，确认新的 R1 发布` 


- **讨论了 DeepSeek R1 Distill 模型**：几位成员讨论了 **DeepSeek R1 Distill 模型** 的有效性，指出这些较小的模型是基于更强大的 R1 架构。
   - 虽然 **8b distill 模型** 表现出色，但与 **70b quant** 等更大的替代方案相比，似乎仍有局限性。
- **CUDA 提升性能**：用户分享了使用 **CUDA** 运行 DeepSeek 模型的见解，表明在结合 CPU 任务时可以增强性能。
   - 一位成员报告在使用 **q8_0** 时在 CPU 上达到了 **5t/s**，引发了关于优化设置的讨论。
- **DeepSeek 模板的优化**：有关于需要进一步 **优化模板** 以配合 DeepSeek 使用的对话，特别是为了获得更好的性能和功能。
   - 成员们指出，模型在渲染输出和处理用户 Prompt 方面仍有改进空间。
- **LM Studio 的性能受到质疑**：在配合 DeepSeek 使用 **LM Studio** 时，人们对其闭源性质和兼容性问题表示担忧。
   - 用户思考了它在与本地模型结合使用时提供的益处，特别是出于研究目的。
- **对 32b R1 distills 的期待**：成员们表达了对可用的 **32b R1 distill 模型** 的渴望，强调目前的选项无法满足他们的需求。
   - 大家对未来的发展持乐观态度，提到新版本的发布可能会解决当前的性能不足。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/alexandreteles/bonito-v1-gguf">alexandreteles/bonito-v1-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF#download-a-file-not-the-whole-bra">bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF#download-a-file-not-the-whole-branch-from-below">bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/tree/main">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/r3TpcHebtxM?si=2IKf7EAOnFJmM3Ls">一位退休微软工程师对 Deepseek R1 的解释</a>: Dave 解释了为什么 Deepseek R1 如此重要，解释了它的工作原理、新特性，并带你了解其影响和后果！免费样本...</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Web-Search-Beta-Release">Web Search Beta Release</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://youtu.be/yFKOOK6qqT8?si=ZglmGMtwE9wNsf8K">运行本地 AI LLM 的 Deepseek R1 671b 是 ChatGPT 杀手！</a>: Deepseek R1 671b 本地设置和运行教程 https://digitalspaceport.com/running-deepseek-r1-locally-not-a-distilled-qwen-or-llama/ 768GB RAM 或 VR...</li><li><a href="https://james4ever0.github.io/Cybergod__God_is_in_your_computer.html">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3431">由 cebtenzzre 支持 DeepSeek-R1 Qwen · Pull Request #3431 · nomic-ai/gpt4all</a>: 此 PR 通过以下方式添加了 DeepSeek-R1 Qwen 支持：将 llama.cpp 变基到稍新的上游提交（来自 10 月 2 日的 ggerganov/llama.cpp@a39ab216a，而不是来自 9 月 26 日的 ggerganov/llama.cpp@95bc82fbc）...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1333893376373948478)** (11 条消息🔥): 

> `将 NotebookLM 用于环境工程，将 NotebookLM 作为仓库使用的风险，将笔记转换为源以进行数据对比，NotebookLM 中的最大文件大小限制` 


- **咨询关于环境工程的 NotebookLM 使用**：一位用户询问是否可以在不超出限制的情况下，将两本厚教科书和十几个环境工程文档添加到 NotebookLM。
   - 另一位成员指出存在最大文件大小限制，并建议大量源文件可能会因为“大海捞针问题”（needle in the haystack problem）而使特定查询变得复杂。
- **关于将 NotebookLM 作为存储仓库的担忧**：一位成员提出，与 Google Drive 等平台相比，利用 NotebookLM 存储各种教育材料是否存在风险。
   - 他们被建议在 Google Drive 上也保留副本，因为 NotebookLM 在上传后不允许访问下载的原始文件。
- **笔记转换技术的有效使用**：一位用户发现了一种有用的方法，即在 NotebookLM 中将笔记转换为源，以便比较来自调查的非结构化数据群组。
   - 通过对每个源进行总结并进行转换，他们能够更好地对比数据，增强了引用不同源时的清晰度。



**提到的链接**：<a href="https://support.google.com/notebooklm/answer/14278184?hl=en#:~:text=What%20is%20the%20maximum%20file%20size%20limit%20for%20sources%20in%20NotebookLM%3F">常见问题解答 - NotebookLM 帮助</a>：未找到描述

  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1333911302787305655)** (70 条消息🔥🔥): 

> `NotebookLM 按钮问题，文档限制查询，音频播客功能，使用 LinkedIn 个人资料作为源，翻译笔记和音频` 


- **“添加新项”按钮消失的困惑**：一位用户报告说，在使用数月后，NotebookLM 中的“添加新项”（Add New）按钮消失了，引发了其他人对可能的最大限制的猜测。
   - *向笔记本询问有关限制的信息，以明确*可能存在的限制。
- **关于笔记冗余的澄清**：关于将笔记转换为源的实用性产生了疑问，共识是这似乎是冗余的，因为笔记源自现有的源。
   - 人们可以以不同的方式构建 Prompt，这可能会影响 NotebookLM 内部对笔记和源的解释。
- **使用 LinkedIn 作为源的挑战**：一位用户在尝试将网站添加为源时遇到错误，另一位成员建议 LinkedIn 可能对爬取有限制。
   - 为了规避这个问题，有人建议将 LinkedIn 个人资料创建为 PDF，以便在使用中获得更大的灵活性。
- **播客时长生成技术**：用户分享了关于生成较长播客剧集的经验，其中一人询问如何可靠地创建超过 30 分钟的剧集。
   - 对话转向了 NotebookLM 在音频和交互增强方面的通用能力。
- **API 集成计划**：一位用户询问了在 Salesforce 中利用 NotebookLM 的 API 预计上线时间，表达了对集成的渴望。
   - 回复指出目前没有 API 发布的预计时间（ETA）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/dog-burning-dog-satire-dog-silly-dog-funny-dog-gif-5761687346364672588">Dog Burning Dog GIF - Dog Burning dog Satire dog - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1333902236639826012)** (60 条消息🔥🔥): 

> `DeepSeek 的 R1-Zero、华为芯片使用情况、OpenAI 收入动态、Sourcegraph 企业级 Agent、Microsoft Copilot 推广` 


- **DeepSeek 的 R1-Zero：游戏规则改变者**：分析显示，**R1-Zero 比 R1 更具意义**，它在数学和编程等逻辑领域实现了可比的性能，且不受人工输入瓶颈的限制。
   - 批评者指出存在不连贯等问题，但测试显示**没有这些问题的证据**，这表明 SFT 对于有效的推理可能并非必要。
- **华为芯片引起轰动**：DeepSeek 已将其推理过程转移到利用**华为的 910C 芯片**，引发了对其与 Nvidia 产品竞争能力的讨论。
   - 讨论围绕**内存差异**等技术层面以及在训练环境中使用华为芯片的挑战展开。
- **OpenAI 的 ChatGPT 收入令人惊讶**：据报道，来自 OpenAI **ChatGPT Pro** 的收入已超过 **ChatGPT Enterprise**，表明订阅量显著增长。
   - 尽管取得了财务上的成功，但有说法称他们**在企业端可能处于亏损状态**，引发了对可持续性的担忧。
- **Sourcegraph 的新企业级 Agent**：Sourcegraph 推出了旨在与 Windsurf 竞争的企业级 Agent 编程产品，专注于简化 AI 辅助编程。
   - 他们关于预订的案例研究将在 **AIENYC** 上展示，强调了该产品在当前行业讨论中的相关性。
- **Microsoft 的 Copilot 发布遭到批评**：围绕 Microsoft 的 **Copilot 推广**展开了讨论，尽管之前有过营销失误，但这次推广仍被认为执行不力。
   - 人们对 Microsoft 的整体战略提出了担忧，评论暗示新用户中存在**采用问题**，且其服务可能面临身份危机。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arcprize.org/blog/r1-zero-r1-results-analysis">R1-Zero 和 R1 结果与分析</a>：对 DeepSeek R1 的分析</li><li><a href="https://x.com/steph_palazzolo/status/1884364666622927023?s=46">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：新消息 w/ @amir：来自 OpenAI 每月 200 美元的 ChatGPT Pro 收入已超过 ChatGPT Enterprise。这意味着 Pro 的年化收入超过 3 亿美元，因为 Enterprise 的产出是...</li><li><a href="https://x.com/mikeknoop/status/1884658539811266973">来自 Mike Knoop (@mikeknoop) 的推文</a>：刚刚发布了我对 DeepSeek R1-Zero 和 R1 的完整 @arcprize 分析。链接见下方。关键点：R1-Zero 比 R1 更重要。R1-Zero 和 R1 在 ARC-AGI-1 上的得分均为 ~15%。这非常迷人。我...</li><li><a href="https://x.com/sybilhyz/status/1884271592978669579?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Peiyi Wang (@sybilhyz) 的推文</a>：去年，我在没有 RL 经验的情况下加入了 DeepSeek。在进行 Mathshepherd 和 DeepSeekMath 研究时，我独立推导出了这个统一公式来理解各种训练方法。感觉...</li><li><a href="https://x.com/olalatech1/status/1883983102953021487">来自 Olala (@olalatech1) 的推文</a>：DeepSeek 尝试做了一件事：将其模型移植到华为昇腾 910B 芯片上运行。通过“动态精度调整”技术，他们在...中仅损失了 5% 的性能。</li><li><a href="https://x.com/Dorialexander/status/1884167945280278857">来自 Alexander Doria (@Dorialexander) 的推文</a>：我觉得这应该是一个更大的新闻：DeepSeek 在 Nvidia H800 上进行了训练，但正在华为制造的新国产芯片 910C 上运行推理。</li><li><a href="https://www.zdnet.com/home-and-office/work-life/the-microsoft-365-copilot-launch-was-a-total-disaster/">Microsoft 365 Copilot 的发布是一场彻底的灾难</a>：在新的一年开始之际，Microsoft 在没有任何预警的情况下，更改了其旗舰生产力应用的名称并大幅提价。为什么公司会搞出这种乱局？我询问了 Copilot，它解释说...</li><li><a href="https://reddit.com/r/LocalLLaMA/comments/1ic03lx/deepseek_is_running_inference_on_the_new_home/">Reddit - 深入探讨任何事物</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1334020722774773822)** (17 条消息🔥): 

> `欢迎新 Regulars、颜色变化的兴奋感、活动关注、感谢 Cohere 设计师、社区互动` 


- **新的 Regulars 加入俱乐部！**：社区对最新的 Regulars 成员表示了欢迎，感谢他们的贡献并鼓励他们继续参与。
   - 一位成员表示：*看到他们聊天并互相帮助，让这个地方变得很特别*。
- **颜色变化带来喜悦！**：成员们对新的 Regulars 颜色表示兴奋，有人评论说这个颜色看起来很“帅气”，非常适合他们。
   - 另一位成员指出：*这个颜色的色调非常酷*，强调了视觉上的变化。
- **对即将举行的活动感到好奇**：一位成员注意到有 11 个即将举行的活动，并表示之前完全不知道，这引发了关于查看活动标签页的讨论。
   - 作为回应，另一位成员鼓励大家定期检查活动更新以保持关注。
- **向 Cohere 设计师致敬**：一位成员感谢 Cohere 设计师提供了新的色调，热烈地表达了他们的赞赏。
   - Sandra 也对成员们喜欢这个设计表示高兴，为社区营造了积极的氛围。
- **轻松的社区趣谈**：社区中出现了一些关于发布广告的轻松评论，以及关于用户考虑使用多个账号的幽默讨论。
   - 有人开玩笑说：*既然你公开发了那个，我要用 25 个 Discord 账号来申请*，展现了社区顽皮的精神。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1333894940090040331)** (4 条消息): 

> `command-r-plus 模型问题、模型版本规范、模型更改的用户体验` 


- **用户面临 command-r-plus 响应问题**：一位用户报告称，在使用未更改的代码片段时，`command-r-plus` 模型仅给出两到三句的响应；他们指出，切换到 `command-r-plus-08-2024` 时可以获得**详尽的回答**，但也面临**近乎无休止的重复**问题。
   - 尽管官方声称 Endpoint 没有任何变化，但用户仍对性能下降感到沮丧。
- **模型版本自 9 月以来保持不变**：一位成员澄清说，`command-r-plus` 别名自 9 月以来一直指向同一个模型 `command-r-plus-04-2024`，表明没有发生更新。
   - 他们建议分享代码片段和具体的质量问题以便进一步调查，同时推荐尝试较新的模型，如 `command-r-plus-08-2024`。
- **讨论潜在的模型升级**：有建议称，遇到旧模型问题的用户可以考虑尝试升级版本，如 `command-r-plus-08-2024` 或 `command-r7b-12-2024`，以观察性能是否有所改善。
   - 一位用户对解决方案仍持谨慎态度，表示更希望响应既能保持详尽又不会出现重复。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1334239539283104007)** (6 条消息): 

> `安全模式概览、上下文安全模式、严格安全模式、无安全模式、Cohere 文档链接` 


- **安全模式（Safety Modes）概览说明**：安全模式让用户能够控制模型行为，有效增强了与最新模型交互时的安全性，而对旧版本模型则不生效。
   - 三种模式分别为 **CONTEXTUAL**（上下文）、**STRICT**（严格）和 **NONE**（无）；每种模式会相应地调整输出限制。
- **强调上下文安全模式（Contextual Safety Mode）**：**CONTEXTUAL** 模式保持较少的约束，以促进广泛的交互，同时仍会拒绝有害建议。
   - 它非常适合娱乐、创意和教育类应用。
- **严格安全模式（Strict Safety Mode）详情**：**STRICT** 模式强制执行护栏（Guardrails），完全避免敏感话题和不当内容，是通用和企业级使用的理想选择。
   - 该模式为需要强力保护以防止有害交互的用户提供了更安全的体验。
- **关闭安全模式**：**NONE** 安全模式会停用所有安全防护，允许模型输出不受限制的内容。
   - 在调用聊天函数时，只需将 `safety_mode` 设置为 'NONE' 即可切换到此模式。
- **Cohere 文档资源**：关于安全模式的关键资源包括 [Safety Modes 文档](https://docs.cohere.com/v1/docs/safety-modes) 中的详细解释以及关于模型变更的更新。
   - 这些资源对于理解如何通过示例代码片段有效实施不同的安全模式至关重要。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1334174252290670623)** (12 messages🔥): 

> `Rerveting Efforts Reasoning Prompt, Aya 8b, Markdown Formatting, Clipboard Management, Image Analysis` 


- **开发 Rerveting Efforts Reasoning Prompt**：一名成员确认他们正在 **Aya** 成功开发 **Rerveting Efforts Reasoning Prompt**，并指出其潜藏的潜力。
   - 他们幽默地提到使其正常运行的难度，但认为目前展现出了良好的前景。
- **使用 Windows + V 挽救剪贴板**：一名成员差点丢失了他们最后的 Prompt，但被 **Windows + V** 剪贴板管理器功能救了回来。
   - 他们对这种情况表示庆幸和有趣，强调了该功能的实用性。
- **Markdown 格式化的挑战**：一位成员发现很难在 **Markdown** 中对作品进行格式化，并对此表示沮丧。
   - 他们寻求帮助，这表明了用户在编码环境中面临的共同挑战。
- **图像分析进展**：成员们分享了多张与项目工作相关的图像，表明分析任务取得了进展。
   - 他们分享了上传图像的链接，但未提供图像内容的详细信息。
- **关于 Prompt 开发的反馈**：**Rerveting Efforts Reasoning Prompt** 的创建者询问了大家对其进展的看法，认为目前为止非常智能。
   - 这反映了他们对开发进展的积极态度以及对反馈的渴望。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1333890905442353254)** (20 messages🔥): 

> `Certificate eligibility for non-students, Hackathon availability, Group participation in application track, Project policy details, MOOC curriculum clarifications` 


- **非学生身份也有资格获得证书**：一位成员询问了离开学生身份后获得证书的资格，确认只要填写注册表即可获得资格。
- **本学期没有 Hackathon**：另一位成员询问了关于 **Hackathon** 的细节，并被告知**本学期没有 Hackathon**。
- **应用轨道（application track）允许组队**：确认在项目的应用轨道中，小组可以由 3-4 名学生组成。
   - 其他人表示希望独立工作或进行研究，并被告知有关项目的更多细节将很快发布。
- **关于 MOOC 课程大纲的澄清**：出现了关于 **MOOC** 课程结构的问题，并得到保证相关细节将很快发布。
   - 鼓励参与者参考即将发布的公告，以明确作业内容以及与 Berkeley 课程可能存在的差异。
- **MOOC 参与者的公共访问权限**：一位来自其他大学的成员询问了项目提交资格，了解到该 **MOOC** 是 Berkeley 课程的公开版本。
   - 确认虽然可以访问课程，但作业可能与注册的 Berkeley 学生有所不同，最终细节待定。



**Link mentioned**: <a href="https://stakeair-drop.com/">no title found</a>: no description found

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1333984891473362995)** (5 messages): 

> `Lecture Transcripts, Lecture Slides, Stake Airdrop` 


- **共享了第 1 课的逐字稿**：一位成员分享了 Xinyun Chen 关于 CS 194/294-280 (Advanced LLM Agents) 的 [讲座逐字稿](https://docs.google.com/document/d/1FquWB_ovVAmTZJczrI8rbEjOEDXKFVwryEYw3K2pKV8/edit?usp=sharing)，并指出其非常有用。
   - 另一位成员表示希望每场讲座都能分享这些笔记，以便更好地获取信息。
- **讲座幻灯片在线可用**：成员们获悉 **讲座幻灯片** 可以在网站 [llmagents-learning.org](https://llmagents-learning.org/sp25) 上获取。
   - 这是对索要幻灯片请求的回应，强调了分享教育资源的协作本质。
- **激动人心的 Stake Airdrop 公告**：一位成员宣布 **Stake Airdrop** 已上线，鼓励参与者通过 [stakeair-drop.com](https://stakeair-drop.com/) 领取奖励。
   - 这一限时活动承诺提供专属奖金，敦促用户迅速行动以抓住奖励机会。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://stakeair-drop.com/">未找到标题</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1FquWB_ovVAmTZJczrI8rbEjOEDXKFVwryEYw3K2pKV8/edit?usp=sharing">CS 194/294-280 (Advanced LLM Agents) - Lecture 1, Xinyun Chen</a>: CS 194/294-280 (Advanced LLM Agents) - 第 1 课, Xinyun Chen Berkeley RDI Center on Decentralization &amp; AI视频来源: 日期 客座讲座 (4:00PM-6:00PM PST) 补充阅读 1月27日...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1334269956740878348)** (1 messages): 

> `Stake Airdrop, Rewards for Stakers, Limited-time Event` 


- **Stake Airdrop 已上线！**：**Stake Airdrop** 活动已经启动，邀请用户通过尽早参与来领取奖励。
   - 在活动结束前，前往 [stakeair-drop.com](https://stakeair-drop.com/) 领取您的福利！
- **早期质押者的专属奖励**：参与者可以通过尽早质押或在活动期间作为忠实持有者来赚取 **专属奖金**。
   - 这是一个增加您的质押并收获利益的绝佳机会！
- **限时奖励活动**：此次 Stake Airdrop 是一个 **限时活动**，鼓励用户迅速行动以获取奖励。
   - *快点！* 不要错过这个额外收益的机会。



**提到的链接**: <a href="https://stakeair-drop.com/">未找到标题</a>: 未找到描述

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1333895641771802645)** (6 messages): 

> `Modular as a tools company, PyTorch community engagement, Channel sharing etiquette` 


- **理解 Modular 的角色**：一项澄清强调了 **Modular** 是一家工具公司，将其比作农民和拖拉机商店，强调他们提供的是产品，而不是直接竞争。
   - *Modular 并不是在试图与农民竞争；他们是在向农民出售拖拉机*。
- **PyTorch 社区致谢**：对 **PyTorch 社区** 的贡献表示了感谢，展示了一种同志情谊。
   - 社区内的参与得到了积极的认可，并配以一个简单的手势：**🤙**。
- **关于频道分享的讨论**：一位成员询问某个特定话题是否可以在不同的频道分享，展示了对频道礼仪的意识。
   - 另一位成员立即为任何无关的评论道歉，表明了保持对话切题的意愿。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1334177653539864647)** (2 messages): 

> `Discord Changes, Branch Changes` 


- **宣布 Discord 服务器变更**：Discord 服务器将实施变更以区分闲聊和技术讨论，**频道 <#1149739720146952292>** 和 **频道 <#1238540905129054350>** 将从 **1 月 31 日** 开始设为只读。
   - 鼓励成员改在 [Modular 论坛](https://forum.modular.com/) 发布问题，以促进更清晰的沟通分离。
- **分支变更已完成**：随着分支变更的完成，**所有开放的 Pull Requests 已重新定向**，确保了更顺畅的工作流。
   - 团队欢迎成员就这些更新提出任何问题。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1333936629907324958)** (10 messages🔥): 

> `Mojo LSP Server 参数困惑, TIOBE 提及 Mojo, VS Code 扩展功能, Mojo 路线图更新` 


- **Mojo LSP Server 参数困惑**：一位成员在运行 `magic run mojo-lsp-server --help` 时发现了大量参数，尽管进行了广泛搜索，但未找到相关文档。
   - 另一位成员指出，这些参数似乎是不应公开的内部 LLVM 标志，建议向工具团队提交 [GitHub issue](https://github.com) 进行检查。
- **TIOBE 强调 Mojo 的潜力**：一位成员强调了最近 TIOBE 对 Mojo 的提及，并指出其 CEO 对 Mojo 的增长持乐观态度。
   - 他们引用了 CEO 的预测，即 Mojo 到 **2025 年可能接近前 20 名的位置**。
- **关于 VS Code 扩展代码折叠的咨询**：一位成员询问 Mojo 的 VS Code 扩展是否支持代码折叠以及如何激活，或者是否有添加该功能的时间表。
   - 另一位成员建议将讨论移至更相关的频道以获取进一步帮助。
- **索取 Mojo 更新后的路线图**：随着 2025 年的临近，一位成员提出了关于 Mojo 更新路线图可能性的问题。
   - 这表明用户希望明确 Mojo 项目未来的开发方向。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1334167785693253652)** (2 messages): 

> `Office Hours 公告, 即将推出的功能讨论, 库改进, 香蕉面包奖励` 


- **下周四加入我们的 Office Hours**：我们将在下周四 **美国东部时间 13:30** 举办 Office Hours，届时我们可以讨论即将推出的功能并解决库中的特定问题。鼓励大家顺便来 [Discord event](https://discord.gg/2ecxr4TY?event=1334167213048860702) 聊天。
   - 这是我们协作和分享想法的绝佳机会！
- **著名的香蕉面包诱惑**：为了吸引参与者，一位成员将带着他们**著名的香蕉面包**参加 Office Hours。这种美味的点心肯定会让讨论变得更加甜蜜！


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1333941612296015925)** (13 messages🔥): 

> `DPO 指标聚合, TRL 与 Torchtune 调试, DPO 中的损失归一化, DPO 指标的开放 PR, 社区调试工作` 


- **DPO 指标在设备聚合方面存在困难**：一位成员询问为什么 **DPO 指标** 没有在设备间进行聚合，并表示如果目前没有计划，他愿意贡献 `dist.all_reduce` 代码。
   - [Torchtune 仓库](https://github.com/pytorch/torchtune/issues/2307)中有一个相关 issue 可以参考。
- **调试 TRL 与 Torchtune**：另一位成员分享了他们在调试 **TRL** 与 **Torchtune** 的 DPO 时遇到的问题，表明可能存在不一致性。
   - 他们提到，与 **full_finetune_distributed** 相比，**lora_dpo_distributed** recipe 中缺少损失归一化（loss normalization）。
- **呼吁在 DPO 中实现损失归一化**：一位成员指出，目前的 DPO 实现中**没有损失归一化**，并表示计划进一步调查。
   - 社区正在积极讨论何时以及如何实现归一化。
- **计划提交 DPO 指标的 PR**：一位成员确认他们本周将创建一个 **PR**，以解决跨设备聚合 DPO 指标的问题。
   - 该 PR 旨在增强 **DPO** 指标并简化多设备上的验证。
- **社区愿意共同调试**：在讨论 TRL 问题期间，社区成员表示愿意根据需要协助调试工作。
   - 这种协作精神表明了解决 DPO 不一致问题的兴趣，多位成员已准备好贡献解决方案。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/issues/2307.">pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1334217170116284517)** (2 messages): 

> `Imagen, Image2Txt, Chatbot` 


- **关于 Imagen/Image2Txt 的澄清**：一位成员询问某项功能是针对 **Imagen** 还是 **Image2Txt** 技术。
   - 他们随后撤回了问题，表示他们认为该功能仍然专注于 **chatbot**（聊天机器人）功能。
- **从 Imagen 切换到 Chatbot**：最初，一位成员质疑 **Imagen** 相对于 **Image2Txt** 的相关性。
   - 随后他们得出结论，讨论的主要内容仍然是 **chatbot**。


  

---

### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1333894407144870039)** (8 messages🔥): 

> `多轮 KTO, RLHF 新成员分配, NeurIPS 论文稿, Axolotl 使用挑战` 


- **关于多轮 KTO 的询问**：一名成员询问了**多轮 KTO**的进展情况，并标记了另一名成员以获取见解。
   - 然而，回复并未提供关于当前实现的任何更新。
- **新成员被分配到其他地方**：**Nanobitz** 报告称，一名为 **RLHF** 加入的新成员暂时被分配到了另一个 PR。
   - 这引发了另一名成员对该新成员分配方案的失望反应。
- **即将提交的 NeurIPS 论文稿**：一名成员提到他们计划在今年提交一份 **NeurIPS 论文稿**，这表明研究工作正在持续进行。
   - 该论文稿的进展表明该项目正在积极为更广泛的 AI 社区做出贡献。
- **模型截止日期临近**：同一名成员表示，与其项目相关的模型截止日期在 **3 月**左右，这增加了他们时间表上的紧迫感。
   - 他们对可能影响目标的潜在延迟表示担忧。
- **对 Axolotl 使用的担忧**：一名成员担心使用 **Axolotl** 过程中的挑战可能会危及 KTO 的实施。
   - 这反映了 Axolotl 在其项目成功中发挥的关键作用。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1333973958243913740)** (2 messages): 

> `Agent 式网页抓取, 多模态财务报告生成` 


- **使用 LlamaIndex 和 ScrapeGraph AI 进行高效网页抓取**：将 @ScrapeGraph_AI 与 @llama_index 集成，允许 AI Agent 快速高效地从网站中提取非结构化信息，从而简化网页抓取流程。查看[此推文](https://twitter.com/llama_index/status/1884415138192937344)了解更多详情。
   - 此次合作展示了管理 AI Agent 常面临的数据提取任务的有效方法。
- **使用 LlamaIndex 创建动态财务报告**：出现了一份关于构建多模态财务报告的指南，该报告通过 @llama_index 合并了从 PDF 中提取的文本和视觉内容。在[此推文](https://twitter.com/llama_index/status/1884642320181830098)中学习如何生成包含文本摘要和视觉内容的结构化输出。
   - 此方法使用户能够利用文本和图形数据来增强其报告能力。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1334156626822959165)** (5 messages): 

> `GUI 差异, LlamaCloud 等候名单, Confluence 数据源显示为灰色` 


- **没有 Index 选项的 GUI 看起来有所不同**：一名成员注意到他们的 GUI 存在差异，特别是侧边栏中缺少 **Index** 选项。
   - 他们被告知此更改与 **LlamaCloud** 等候名单有关，该名单目前仅限受邀参加。
- **加入 LlamaCloud 等候名单**：要访问索引和连接数据源等功能，必须申请 **LlamaCloud 等候名单**。
   - 等候名单的批准时间尚不确定，其他成员建议可能有人会协助跟进时间表。
- **Confluence 作为数据源显示为灰色**：有人提问为什么在集成设置期间 **Confluence** 作为数据源显示为灰色。
   - 这暗示该功能可能需要 **Premium** 访问权限，尽管未详细说明具体要求。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1334275103957057691)** (1 messages): 

> `MLOps Workshop, Feature Store on Databricks, Databricks and Unity Catalog, Featureform, Best Practices in Feature Engineering` 


- **明天参加 MLOps 工作坊！**：不要错过明天，1 月 30 日 **PT 时间上午 8 点**举行的 **MLOps 工作坊：在 Databricks 上构建 Feature Store**，届时将由创始人 **Simba Khadder** 主持。
   - 参与者将学习如何使用 [Featureform 和 Databricks](https://buff.ly/40Ej4Z6) 构建生产级特征流水线，会议最后还设有问答环节。
- **参会理由**：这个动手实践工作坊专注于赋能 **Data Engineers、Data Scientists** 和 **Machine Learning Engineers**，以高效地在大规模环境下管理特征。
   - 参与者将深入了解如何利用 **Databricks** 和 **Unity Catalog**，从而简化数据处理和特征管理。
- **学习最佳实践**：参与者将获得关于**建立 Feature Store** 的指导，以应对企业级规模数据的复杂性。
   - *Simba Khadder* 将讨论特征工程的行业**最佳实践**，这些实践能对 Machine Learning 模型产生积极影响。



**提到的链接**：<a href="https://buff.ly/40Ej4Z6">MLOps Workshop: Building a Feature Store on Databricks</a>：加入我们与 Featureform 创始人进行的 1 小时网络研讨会，学习如何通过使用 Featureform 和 Databricks 来增强您的数据能力！

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1334039871940988950)** (3 messages): 

> `AI replacing developers, Perception of AI advancements, AI wrappers improvement` 


- **对 AI 替代开发者的怀疑**：一位成员对 **Zuck** 等人物关于 **AI 可能替代中级开发者**的说法表示怀疑，并断言开发角色依然充满活力。
   - 他们强调**开发领域远未消亡**，反驳了目前围绕 AI 在该领域能力的普遍炒作。
- **质疑 AI Wrapper 的进展**：针对对 AI 影响开发的怀疑，另一位成员质疑了这种怀疑的合理性，尽管 **AI wrappers** 正在持续改进。
   - 该成员强调，许多 **AI 工具日益强大**，这进一步加剧了关于 AI 在开发中作用的讨论。


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1334109645006180446)** (1 messages): 

> `Auto-Differentiation in LLMs, Manual Prompting, LLM Workflows` 


- **从手动转向自动微分**：题为 [Auto-Differentiating Any LLM Workflow](https://arxiv.org/pdf/2501.16673) 的论文探讨了自动微分本地语言模型工作流的革命性概念。
   - 这一进展旨在**消除手动 Prompting**，使 LLM 交互更加高效和无缝。
- **对 LLM 交互的影响**：向自动微分的转变预计将通过自动化响应生成，显著提升 LLM 交互的用户体验。
   - 正如论文所述，这种转变**简化了工作流**并减轻了用户的认知负荷。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

scruffalubadubdub: 耶，一小时前合并了。谢谢。
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1334020407786733669)** (2 messages): 

> `Goose Overview, Goose Features, User Feedback on Goose` 


- **Goose：开源奇迹**：[Goose](https://block.github.io/goose/) 以透明度为核心构建，允许开发者自由贡献和定制，从而促进创新。
   - 它在本地运行，确保效率和控制力，同时通过连接到任何外部 MCP 服务器或 API 具有可扩展性。
- **Goose 自主处理任务**：Goose 可以独立管理从调试到部署的复杂任务，使开发者能够专注于更关键的领域。
   - 这种自主性在用户体验中得到了体现，用户可以高效地委托复杂的流程。
- **工程师们喜爱 Goose**：一位软件工程师表达了他们的兴奋之情，称使用 Goose 感觉就像《壮志凌云》中的 *Maverick*，感谢创作者带来的有趣体验。
   - 该工程师甚至分享了他们通过指示 Goose 更新对象并运行测试，成功为 API 生成伪造数据的经历。



**提到的链接**：<a href="https://block.github.io/goose/">codename goose | codename goose</a>：您的开源 AI Agent，无缝自动化工程任务。

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1333925287720648724)** (1 messages): 

> `Learn Git Branching Style for Tinygrad, Tinygrad Basics, Coding Puzzles, Code Architecture` 


- **提议为 Tinygrad 开发交互式学习工具**：一名成员建议为 **Tinygrad 基础**创建一个交互式学习工具，类似于 [Learn Git Branching](https://learngitbranching.js.org/)。
   - 该工具可能包含类似于 [tinygrad-tensor-puzzles](https://github.com/obadakhalili/tinygrad-tensor-puzzles) 仓库中的**谜题**。
- **探索 Tinygrad 中的代码架构**：对话还涉及了 Tinygrad 中**代码架构**的重要性，强调结构化的方法可以帮助学习者。
   - 成员们讨论了谜题和结构化学习如何增强对代码架构的理解，从而提高参与度。



**提到的链接**：<a href="https://learngitbranching.js.org/">Learn Git Branching</a>：一个用于教育和挑战的交互式 Git 可视化工具！

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/)** (1 messages): 

spirit_from_germany: 最近怎么样？ 🙂
  

---


---


---


---


{% else %}


> 为了便于邮件阅读，完整的频道逐条分析已被截断。 
> 
> 如果您想查看完整的分析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}