import json
import sys
import os
import click
from datetime import datetime

# ---------- 配置文件路径 ----------
def get_quiz_dir():
    quiz_dir = os.path.expanduser("~/.quiz")
    os.makedirs(quiz_dir, exist_ok=True)
    return quiz_dir

def get_error_file():
    return os.path.join(get_quiz_dir(), "errors.json")

def get_progress_file():
    return os.path.join(get_quiz_dir(), "progress.json")

def get_data_path(filename):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, filename)

# ---------- 错误记录 ----------
def add_error_word(session_index_ref, en, zh, wrong_input):
    error_file = get_error_file()
    try:
        if os.path.exists(error_file):
            with open(error_file, "r", encoding="utf-8") as f:
                sessions = json.load(f)
        else:
            sessions = []
    except:
        sessions = []

    if session_index_ref[0] is None:
        new_session = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "words": []
        }
        sessions.append(new_session)
        session_index_ref[0] = len(sessions) - 1

    session_idx = session_index_ref[0]
    if session_idx < 0 or session_idx >= len(sessions):
        return

    words = sessions[session_idx]["words"]
    existing = next((w for w in words if w["en"] == en), None)
    if existing:
        if wrong_input not in existing.get("wrong", []):
            existing.setdefault("wrong", []).append(wrong_input)
    else:
        words.append({"en": en, "zh": zh, "wrong": [wrong_input]})

    with open(error_file, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

# ---------- 进度管理 ----------
def load_progress():
    prog_file = get_progress_file()
    try:
        if os.path.exists(prog_file):
            with open(prog_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = data.get("index", 0)
            if isinstance(idx, int) and idx >= 0:
                return idx
    except:
        pass
    return 0

def save_progress(index):
    with open(get_progress_file(), "w", encoding="utf-8") as f:
        json.dump({"index": index}, f)

# ---------- 核心输入函数（含跳过） ----------
def input_word_until_correct(en, zh, session_index, error_set):
    target_lower = en.strip().lower()
    user_input = ""

    while True:
        try:
            char = click.getchar(echo=False)
        except KeyboardInterrupt:
            raise

        # 跳过
        # if char.lower() == 's' and user_input == '':
        #     click.echo()
        #     click.echo(en.strip())
        #     if en not in error_set:
        #         error_set.add(en)
        #         add_error_word(session_index, en.strip(), zh, "[SKIP]")
        #     return

        # 回车
        if char in ('\n', '\r'):
            if user_input.lower() == target_lower:
                click.echo()
                return
            else:
                click.echo()
                click.echo(en.strip())
                if en not in error_set:
                    error_set.add(en)
                    add_error_word(session_index, en.strip(), zh, user_input)
                user_input = ""
                continue

        # 退格
        elif char in ('\x7f', '\b'):
            if user_input:
                user_input = user_input[:-1]
                click.echo('\b \b', nl=False)

        # Ctrl+C
        elif char == '\x03':
            raise KeyboardInterrupt()

        # 普通字符
        elif char.isprintable():
            user_input += char
            click.echo(char, nl=False)
            if user_input.lower() == target_lower:
                click.echo()
                return

def input_word_review(en, zh):
    target_lower = en.strip().lower()
    user_input = ""
    while True:
        try:
            char = click.getchar(echo=False)
        except KeyboardInterrupt:
            raise

        if char.lower() == 's' and user_input == '':
            click.echo()
            click.echo(en.strip())
            return

        if char in ('\n', '\r'):
            if user_input.lower() == target_lower:
                click.echo()
                return
            else:
                click.echo()
                click.echo(en.strip())
                user_input = ""
                continue

        elif char in ('\x7f', '\b'):
            if user_input:
                user_input = user_input[:-1]
                click.echo('\b \b', nl=False)

        elif char == '\x03':
            raise KeyboardInterrupt()

        elif char.isprintable():
            user_input += char
            click.echo(char, nl=False)
            if user_input.lower() == target_lower:
                click.echo()
                return

# ---------- 命令组 ----------
@click.group()
def cli():
    """单词测验工具 - 学习、复习、翻阅"""
    pass

# ---------- learn ----------
@cli.command()
@click.option("-f", "--file", default=None, type=click.Path(dir_okay=False),
              help="自定义单词 JSON 文件（默认内置 a.json）")
@click.option("--start", type=int, default=None, help="起始索引（0 为第一个）")
@click.option("--word", type=str, default=None, help="从指定英文单词开始")
def learn(file, start, word):
    """根据中文释义输入英文，支持断点续测，S 键跳过"""
    if file is None:
        data_file = get_data_path("a.json")
    else:
        data_file = file
        if not os.path.isfile(data_file):
            raise click.BadParameter(f"文件不存在: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        click.echo("没有找到任何单词，退出。")
        return

    total = len(words)

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
            click.echo(f"错误：起始索引 {start_idx} 超出范围 (0~{total-1})")
            return
    else:
        start_idx = load_progress()
        if start_idx >= total:
            click.echo("已完成全部单词，从头开始。")
            start_idx = 0
        elif start_idx > 0:
            click.echo(f"从上次结束位置继续（第 {start_idx+1} 个）。")

    click.echo(f"共 {total} 个单词，从第 {start_idx+1} 个开始，按 Ctrl+C 退出。")
    click.echo("（输入为空时按 S 键跳过当前单词）\n")

    session_index = [None]
    error_set = set()

    for idx in range(start_idx, total):
        en, zh = words[idx]
        zh_clean = zh.strip().replace("\r", "")
        click.echo(zh_clean)
        input_word_until_correct(en, zh_clean, session_index, error_set)
        save_progress(idx + 1)

    if error_set:
        click.echo("\n⚠️  本次答错的单词：")
        for en, zh in [(e, z) for e, z in
                       [(w[0].strip(), w[1].strip().replace("\r", "")) for w in words]
                       if e in error_set]:
            click.echo(f"  {en}  {zh}")
        click.echo("（错误及跳过记录已保存到 errors.json）")
    else:
        click.echo("\n🎉 全部正确！")

    if start_idx + (total - start_idx) >= total:
        save_progress(0)
        click.echo("已完成全部单词，进度已重置。")
    click.echo("学习结束。")

# ---------- review ----------
@cli.command()
@click.option("-t", "--time", "session_time", default=None, type=str,
              help="指定要复习的错误记录时间，默认最新")
def review(session_time):
    """复习之前答错的单词，同样支持 S 键跳过"""
    error_file = get_error_file()
    if not os.path.exists(error_file):
        click.echo("错误记录文件不存在，请先使用 learn 学习。")
        return

    try:
        with open(error_file, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    except:
        click.echo("错误记录文件损坏。")
        return

    sessions = [s for s in sessions if s.get("words")]

    if not sessions:
        click.echo("暂无有效的错误记录。")
        return

    if session_time is not None:
        target_session = next((s for s in sessions if s.get("time") == session_time), None)
        if target_session is None:
            click.echo(f"未找到时间为 '{session_time}' 的错误记录。")
            return
    else:
        target_session = sessions[-1]

    words = target_session.get("words", [])
    if not words:
        click.echo("该记录中没有错误单词。")
        return

    click.echo(f"开始复习 {len(words)} 个错题（记录时间：{target_session['time']}）。")
    click.echo("（输入为空时按 S 键跳过当前单词）\n")

    for w in words:
        zh = w["zh"].strip().replace("\r", "")
        en = w["en"].strip()
        click.echo(zh)
        input_word_review(en, zh)

    click.echo("\n🎉 错题复习完成！")

# ---------- browse（翻阅单词） ----------
@cli.command()
@click.option("-f", "--file", default=None, type=click.Path(dir_okay=False),
              help="自定义单词 JSON 文件（默认内置 a.json）")
@click.option("--start", type=int, default=0, help="起始索引（默认 0）")
@click.option("--page", type=int, default=100, help="每页显示的单词数量（默认 100）")
def browse(file, start, page):
    """翻阅单词表，每页显示指定数量，按任意键翻页，q 退出"""
    if file is None:
        data_file = get_data_path("a.json")
    else:
        data_file = file
        if not os.path.isfile(data_file):
            raise click.BadParameter(f"文件不存在: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        words = json.load(f)

    if not words:
        click.echo("没有找到任何单词。")
        return

    total = len(words)
    idx = start if start >= 0 and start < total else 0

    while idx < total:
        # 清除屏幕（可选，便于阅读）
        click.clear()
        click.echo(f"📖 单词表（{idx+1}-{min(idx+page, total)} / {total}）")
        click.echo("-" * 50)

        for i in range(idx, min(idx+page, total)):
            en, zh = words[i]
            zh_clean = zh.strip().replace("\r", "")
            click.echo(f"{i+1:4d}. {en:<20} {zh_clean}")

        click.echo("-" * 50)
        click.echo("按任意键继续翻页，按 q 退出...", nl=False)

        try:
            key = click.getchar()
        except KeyboardInterrupt:
            click.echo("\n已退出。")
            return

        if key.lower() == 'q':
            click.echo("\n已退出翻阅。")
            return
        else:
            idx += page

    click.echo("\n已是最后一页。")

# ---------- 入口 ----------
if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n已退出。")
        sys.exit(0)
    # pyinstaller --onefile --add-data "a.json:." quiz.py
    # sudo cp dist/quiz /usr/local/bin/
    # sudo chmod +x /usr/local/bin/quiz
    # quiz

    # quiz --word removal
    # --start
    # quiz --file /path/to/your/words.json
    # cat ~/.quiz/errors.json

#     quiz learn                    # 默认从进度继续
# quiz learn --start 0          # 从头开始
# quiz learn --word remove      # 从 "remove" 开始
# quiz learn -f mywords.json    # 用外部文件

# quiz review                   # 复习最新一次错误记录
# quiz review -t "2026-05-16 10:01:16"   # 复习指定时间的错误记录

# quiz browse                           # 从第 1 个开始，每页 100 个
# quiz browse --start 50                # 从第 51 个开始
# quiz browse --page 50                 # 每页显示 50 个
# quiz browse --start 100 --page 50     # 从第 101 个开始，每页 50 个
# quiz browse -f mywords.json  