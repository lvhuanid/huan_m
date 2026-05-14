import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Optional


# 了解至少一种“强制 schema”的方法（Instructor 或 JSON mode）。
load_dotenv()

MODEL = os.getenv("LLM_MODEL_ID")
client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# ollama_client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )
#   """调用 Ollama 获取文本的向量表示"""
# response = ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=text
# )
# resp =  ollama_client.embeddings.create(
#     model=EMBEDDING_MODEL,
#     input=["今天天气真好", "下雨了，好冷"]
# )

# prompt = """
# 从以下文本中提取人名、公司和职位，以 JSON 格式返回。
# 只返回 JSON，不要任何解释。
# 文本：王小蒙在北京智源科技有限公司担任算法工程师。
# 输出格式示例：{"name": "...", "company": "...", "position": "..."}
# """

# response = client.chat.completions.create(
#     model=model,
#      messages=[{"role": "user", "content": prompt}],
#     temperature=0,
#     response_format={"type": "json_object"}
# )

# content = response.choices[0].message.content
# # 有时候模型会包在 ```json 代码块里，需要处理一下
# if "```json" in content:
#     content = content.split("```json")[1].split("```")[0].strip()
# data = json.loads(content)
# print(data)

# class PersonInfo(BaseModel):
#     name: str
#     company: str | None
#     position: str
#     skills: list[str]

# prompt = """
# 从文本中提取人物信息，返回一个 JSON 对象，包含字段 name, company, position, skills。
# 只返回 JSON，不要额外解释。
# 文本：张三在腾讯担任前端工程师，擅长 React 和 TypeScript。
# """

# response = client.chat.completions.create(
#     model=model,
#     messages=[{"role": "user", "content": prompt}],
#     temperature=0,
#     response_format={"type": "json_object"}   # 强制 JSON 输出
# )

# # 获取返回的字符串并解析
# raw = response.choices[0].message.content

# # 有时候模型会包一层 ```json ... ```，需要去除
# if "```json" in raw:
#     raw = raw.split("```json")[1].split("```")[0].strip()

# data = json.loads(raw)

# # 用 Pydantic 验证和实例化
# try:
#     info = PersonInfo(**data)
#     print(info.name)    # 张三
#     print(info.skills)  # ['React', 'TypeScript']
# except ValidationError as e:
#     print("输出不符合预设结构:", e)


class WorkExperience(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    duration: Optional[str] = None

class Education(BaseModel):
    school: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    graduation_year: Optional[str] = None

class Resume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[str] = []
    work_experience: list[WorkExperience] = []
    education: list[Education] = []

# -------------------- 核心解析函数 --------------------
def parse_resume(text: str) -> dict:
    """
    输入简历文本，返回结构化的字典。
    """
    # 精心设计的 Prompt
    prompt = f"""
你是一名专业的简历解析器。请从以下简历文本中提取信息，并以 JSON 格式返回。
严格遵守以下规则：
1. 只返回一个 JSON 对象，不要包含任何其他文字。
2. JSON 必须符合以下结构：
{{
    "name": "姓名",
    "email": "邮箱",
    "phone": "电话",
    "skills": ["技能1", "技能2"],
    "work_experience": [
        {{
            "company": "公司名",
            "position": "职位",
            "duration": "工作时间段，如 2020.06 - 至今"
        }}
    ],
    "education": [
        {{
            "school": "学校名",
            "degree": "学历（如本科、硕士）",
            "major": "专业",
            "graduation_year": "毕业年份"
        }}
    ]
}}
3. 如果某个字段没有找到，skills/ work_experience/ education 用空数组，其他字段用 null，不要编造任何信息。
4. 提取的技能名称请尽量使用通用名称，如 "Python" 而非 "python编程"。

简历文本：
{text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}   # 强制 JSON 输出
        )
    except Exception as e:
        raise RuntimeError(f"API 调用失败: {e}")

    # 获取返回字符串
    content = response.choices[0].message.content

    # 清洗可能存在的 Markdown 代码块标记
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # 解析 JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"模型返回的不是合法的 JSON:\n{content}")

    # 用 Pydantic 验证并返回字典（自动填充默认值、类型检查）
    try:
        resume = Resume(**data)
        return resume.model_dump()  # 转为字典
    except ValidationError as e:
        raise RuntimeError(f"输出结构与预期不符: {e}")

# -------------------- 测试 --------------------
if __name__ == "__main__":
    sample_resume = """
    张三 | 前端工程师
    邮箱：zhangsan@example.com
    电话：13800138000
    技能：React, TypeScript, Vue.js, Webpack
    工作经历：
    腾讯科技（2020年6月 - 至今）  前端工程师
    阿里巴巴（2018年7月 - 2020年5月） 前端开发实习生
    教育背景：
    清华大学  计算机科学与技术  本科  2020年毕业
    """

    result = parse_resume(sample_resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))