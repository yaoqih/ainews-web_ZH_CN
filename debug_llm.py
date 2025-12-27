
import os
import requests
import sys
from openai import OpenAI

# --- 1. 基础配置打印 ---
print("="*40)
print("1. 环境变量检查")
print("="*40)

api_key = os.environ.get("LLM_API_KEY")
base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

if not api_key:
    print("❌ 错误: 未找到 LLM_API_KEY 环境变量！")
    print("   如果你在本地运行，请确保 `export LLM_API_KEY=sk-...`")
    print("   如果你在 GitHub Actions 运行，请检查 Secrets 配置。")
    sys.exit(1)
else:
    # 只打印前几位和后几位，防止泄露
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"✅ LLM_API_KEY 已设置: {masked_key}")

print(f"ℹ️ LLM_BASE_URL: {base_url}")


# --- 2. 网络连通性测试 ---
print("\n" + "="*40)
print("2. 网络连通性测试 (Ping Base URL)")
print("="*40)

# 通常 Base URL 是 https://api.xxx.com/v1，我们要测试 https://api.xxx.com
# 简单的处理方式是去掉 /v1 或直接请求
try:
    print(f"正在尝试连接: {base_url} ...")
    # 很多 API endpoint 直接 GET 会返回 404 或 405，这没关系，只要不是连接超时就行
    # 我们设置 10秒超时
    response = requests.get(base_url, timeout=10)
    print(f"✅ 网络连接成功! 状态码: {response.status_code}")
    print(f"   响应头: {response.headers.get('content-type', 'unknown')}")
except requests.exceptions.ConnectionError:
    print(f"❌ 连接失败: 无法连接到 {base_url}")
    print("   可能原因: DNS 解析失败、防火墙阻拦、或 URL 拼写错误。")
except requests.exceptions.Timeout:
    print(f"❌ 连接超时: {base_url} 在 10秒内无响应。")
except Exception as e:
    print(f"❌ 发生未知网络错误: {e}")


# --- 3. OpenAI SDK 调用测试 ---
print("\n" + "="*40)
print("3. LLM API 调用测试 (Hello World)")
print("="*40)

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    print("正在发送测试请求 (Model: gpt-4o-mini)...")
    # 这里的 model 建议用你确定支持的模型，或者通用一点的 gpt-3.5-turbo / gpt-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 如果你用的是 DeepSeek，记得改成 deepseek-chat
        messages=[
            {"role": "user", "content": "Say 'Connection Successful' if you can hear me."}
        ],
        max_tokens=20
    )
    
    content = response.choices[0].message.content
    print(f"✅ API 调用成功!")
    print(f"🤖 模型回复: {content}")

except Exception as e:
    print(f"❌ API 调用失败!")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误详情: {e}")
    
    # 针对性建议
    err_str = str(e)
    if "401" in err_str:
        print("💡 建议: 你的 API Key 可能无效或过期。")
    elif "404" in err_str:
        print("💡 建议: 模型名称可能错误 (检查 model参数) 或 Base URL 不正确。")
    elif "429" in err_str:
        print("💡 建议: 触发了速率限制 (Rate Limit) 或额度用尽。")
    elif "500" in err_str or "502" in err_str:
        print("💡 建议: 服务端崩溃，或者网关错误。")

print("\n" + "="*40)
print("测试结束")
print("="*40)
