import os
import re
import requests
import frontmatter
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置区域 ---
UPSTREAM_REPO = "smol-ai/ainews-web-2025"
TARGET_DIR = "markdownsrc/content/issues"
OUTPUT_DIR = "docs"
API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
MAX_WORKERS = 5  # 并行线程数，根据你的 API Rate Limit 调整

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_upstream_files():
    """获取上游仓库的文件列表"""
    url = f"https://api.github.com/repos/{UPSTREAM_REPO}/contents/{TARGET_DIR}"
    print(f"Checking upstream: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return [f for f in response.json() if f['name'].endswith('.md')]
    print(f"Error checking upstream: {response.status_code}")
    return []

def split_markdown_by_headers(text):
    """
    按 H1-H4 (#, ##, ###, ####) 切分 Markdown 内容。
    注意：必须忽略代码块 (```) 内部的 # 符号。
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    in_code_block = False
    
    # 正则匹配 # 开头的标题 (H1-H4)
    header_pattern = re.compile(r'^(#{1,4})\s')

    for line in lines:
        # 检测代码块状态
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        
        # 如果是标题，且不在代码块内，且当前块不为空 -> 切割
        if not in_code_block and header_pattern.match(line):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)
            
    # 添加最后一块
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks

def translate_text_chunk(text, is_metadata=False):
    """
    调用 LLM 翻译单个文本块
    """
    if not text.strip():
        return ""

    # 针对不同内容调整 Prompt
    if is_metadata:
        system_prompt = "你是一个翻译助手。请将提供的文本翻译成中文，保留原意。"
    else:
        system_prompt = """
        你是一个专业的技术翻译专家。请将以下 Markdown 片段翻译成中文。
        1. **必须保留 Markdown 格式**：不要修改标题层级、链接、加粗、代码块。
        2. **专有名词保留英文**：如 LLM, Transformer, Agent, GitHub, CUDA 等。
        3. **代码块内容不要翻译**：保留代码块内的原始代码。
        4. **只返回翻译结果**：不要包含"这是翻译"等废话。
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 建议使用 mini 或 deepseek-chat 以节省成本
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # 失败则返回原文

def process_single_file(file_info):
    """
    处理单个文件的完整流程：下载 -> 解析 -> 并行翻译 -> 组装 -> 保存
    """
    filename = file_info['name']
    local_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(local_path):
        print(f"跳过已存在: {filename}")
        return

    print(f"开始处理: {filename} ...")
    
    # 1. 下载原始内容
    raw_content = requests.get(file_info['download_url']).text
    
    # 2. 解析 Frontmatter (元数据)
    post = frontmatter.loads(raw_content)
    
    # 3. 翻译 Metadata (串行，因为内容少)
    if 'title' in post.metadata:
        post.metadata['title'] = translate_text_chunk(post.metadata['title'], is_metadata=True)
    if 'description' in post.metadata:
        post.metadata['description'] = translate_text_chunk(post.metadata['description'], is_metadata=True)
    
    # 4. 切分正文 (Body)
    body_chunks = split_markdown_by_headers(post.content)
    print(f"[{filename}] 切分为 {len(body_chunks)} 个片段，准备并行翻译...")

    # 5. 并行翻译正文
    translated_chunks = [None] * len(body_chunks) # 预分配位置保证顺序
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务，保留索引以便后续按顺序组装
        future_to_index = {
            executor.submit(translate_text_chunk, chunk): i 
            for i, chunk in enumerate(body_chunks)
        }
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                translated_chunks[idx] = future.result()
            except Exception as e:
                print(f"片段 {idx} 翻译出错: {e}")
                translated_chunks[idx] = body_chunks[idx] # 出错回退到原文

    # 6. 重新组装
    post.content = "\n\n".join(translated_chunks)
    
    # 7. 保存
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))
    
    print(f"完成并保存: {filename}")

def update_index():
    """生成简单的 index.md"""
    if not os.path.exists(OUTPUT_DIR): return
    
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md') and f != 'index.md'], reverse=True)
    index_content = "# AI News 中文同步版\n\n> 自动同步自 smol-ai/ainews-web-2025，由 AI 并行翻译。\n\n"
    
    for f in files:
        try:
            post = frontmatter.load(os.path.join(OUTPUT_DIR, f))
            title = post.get('title', f.replace('.md', ''))
            date = post.get('date', '')
            index_content += f"- [{title}](./{f}) *{date}*\n"
        except:
            index_content += f"- [{f}](./{f})\n"
            
    with open(os.path.join(OUTPUT_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write(index_content)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    upstream_files = get_upstream_files()
    
    # 这里也可以并行处理不同的文件，但为了避免 API Rate Limit，建议文件级别串行，文件内部并行
    for file_info in upstream_files:
        process_single_file(file_info)

    update_index()

if __name__ == "__main__":
    main()
