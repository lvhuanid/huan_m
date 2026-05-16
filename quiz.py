import json
import sys
import os
import click
from datetime import datetime

# ---------- 辅助函数 ----------
def get_data_path(filename):
    """获取数据文件的路径，兼容 PyInstaller 打包后的 _MEIPASS 环境"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, filename)

def load_words(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)

def get_quiz_dir():
    """返回固定存储目录 ~/.quiz，若不存在则创建"""
    quiz_dir = os.path.expanduser("~/.quiz")
    os.makedirs(quiz_dir, exist_ok=True)
    return quiz_dir


def get_error_file():
    return os.path.join(get_quiz_dir(), "errors.json")

def get_progress_file():
    return os.path.join(get_quiz_dir(), "progress.json")


def init_error_session():
    """
    初始化错误记录：在 errors.json 末尾新建一条记录（时间 + 空单词列表）
    返回该记录在数组中的索引
    """
    error_file = get_error_file()
    try:
        if os.path.exists(error_file):
            with open(error_file, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        else:
            sessions = []
    except (json.JSONDecodeError, IOError):
        sessions = []

    new_session = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "words": []
    }
    sessions.append(new_session)
    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    return len(sessions) - 1   # 新记录的索引

def add_error_word(session_index, en, zh):
    """实时添加错误单词到指定 session 中（若已存在则忽略）"""
    error_file = get_error_file()
    try:
        with open(error_file, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, IOError, FileNotFoundError):
        return

    if session_index < 0 or session_index >= len(sessions):
        return

    words = sessions[session_index]["words"]
    # 去重：同一 session 内不重复添加
    already_exists = any(w["en"] == en for w in words)
    if not already_exists:
        words.append({"en": en, "zh": zh})
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)


def load_progress():
    prog_file = get_progress_file()
    try:
        if os.path.exists(prog_file):
            with open(prog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = data.get("index", 0)
            if isinstance(idx, int) and idx >= 0:
                return idx
    except Exception:
        pass
    return 0

def save_progress(index):
    prog_file = get_progress_file()
    with open(prog_file, "w", encoding="utf-8") as f:
        json.dump({"index": index}, f)

# ---------- 命令 ----------
@click.command()
@click.option(
    "-f", "--file",
    default=None,
    type=click.Path(dir_okay=False),
    help="自定义单词 JSON 文件路径（默认使用内置 a.json）"
)
@click.option(
    "--start",
    type=int,
    default=None,
    help="指定起始索引（0 表示第一个单词）"
)
@click.option(
    "--word",
    type=str,
    default=None,
    help="指定从某个英文单词开始（忽略大小写）"
)
def cli(file, start, word):
    """
    简单单词测验：中文释义 -> 英文输入，支持断点续测。

    \b
    - 自动记忆进度（progress.json），下次启动从上次结束位置继续
    - --start 或 --word 可手动指定起始位置
    - 答错的单词会立即记录到 errors.json，不怕中途退出
    """
    # --- 获取单词数据 ---
    if file is None:
        data_file = get_data_path("a.json")
    else:
        data_file = file
        if not os.path.isfile(data_file):
            raise click.BadParameter(f"文件不存在: {data_file}")

    words = load_words(data_file)
    if not words:
        click.echo("没有找到任何单词，退出。")
        return

    total = len(words)

    # --- 确定起始索引 ---
    if word is not None:
        found_idx = None
        for i, (en, _) in enumerate(words):
            if en.strip().lower() == word.strip().lower():
                found_idx = i
                break
        if found_idx is None:
            click.echo(f"错误：单词列表中未找到 '{word}'")
            return
        start_idx = found_idx
    elif start is not None:
        start_idx = start
        if start_idx < 0 or start_idx >= total:
            click.echo(f"错误：起始索引 {start_idx} 超出范围 (0 ~ {total-1})")
            return
    else:
        start_idx = load_progress()
        if start_idx >= total:
            click.echo("之前已完成所有单词，从头开始。")
            start_idx = 0
        elif start_idx > 0:
            click.echo(f"从上次结束位置继续（第 {start_idx+1} 个单词）。")

    click.echo(f"共 {total} 个单词，从第 {start_idx+1} 个开始，按 Ctrl+C 可随时退出。\n")

    # --- 初始化错误记录（实时写入用）---
    session_idx = init_error_session()   # 新开一条错误记录
    error_set = set()                    # 内存去重

    # --- 测验循环 ---
    for idx in range(start_idx, total):
        en, zh = words[idx]
        zh_clean = zh.strip().replace("\r", "")
        click.echo(zh_clean)

        while True:
            user_input = click.prompt("请输入英文", type=str).strip()
            if user_input.lower() == en.strip().lower():
                click.echo("✓ 正确")
                save_progress(idx + 1)      # 答对保存进度
                break
            else:
                click.echo(en.strip())
                # 避免重复记录
                if en.strip() not in error_set:
                    error_set.add(en.strip())
                    # 立即写入 errors.json
                    add_error_word(session_idx, en.strip(), zh_clean)

    # --- 结束后处理 ---
    if error_set:
        click.echo("\n⚠️  本次答错的单词如下，建议重点复习：")
        for en, zh in [(e, z) for e, z in
                       # 重新遍历一次拿到完整释义（也可从内存中找）
                       [(en, zh) for en, zh in [(w[0].strip(), w[1].strip().replace("\r", "")) for w in words]
                        if en in error_set]
                      ]:
            click.echo(f"  {en}  {zh}")
        click.echo("（错误单词已实时记录到 errors.json）")
    else:
        click.echo("\n🎉 全部正确，太棒了！")

    # 若已完成全部单词，重置进度
    if start_idx + (total - start_idx) >= total:
        save_progress(0)
        click.echo("已完成全部单词，进度已重置。")

    click.echo("所有单词已完成。")

if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n\n已退出，当前进度已自动保存。")
        sys.exit(0)
    # pyinstaller --onefile --add-data "a.json:." quiz.py
    # sudo cp dist/quiz /usr/local/bin/
    # sudo chmod +x /usr/local/bin/quiz
    # quiz

    # quiz --word removal
    # --start
    # quiz --file /path/to/your/words.json