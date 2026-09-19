#!/usr/bin/env python3
"""
批量生成 SelfPlay 对战日志摘要报告。

功能：
  1. 扫描 all_output_dir/selfplay/selfplay_battles/detailed_battles/ 下的所有 JSON 日志
  2. 调用 format_battle_detail.generate_report() 生成摘要
  3. 追加到同一个累积输出文件
  4. 通过 manifest 记录已处理的文件，支持增量处理（只跑新文件）

用法:
  batch_battle_reports.py <output_dir>
  batch_battle_reports.py <output_dir> --force
  batch_battle_reports.py <output_dir> --output reports.md
"""
import hashlib
import json
import os
import sys
import glob
import time
import re

from format_battle_detail import generate_report


# ─── 路径常量 ──────────────────────────────────

SELFPLAY_BATTLES_REL = "selfplay/selfplay_battles"
DETAILED_REL = f"{SELFPLAY_BATTLES_REL}/detailed_battles"
OUTPUT_REL = f"{SELFPLAY_BATTLES_REL}/all_reports.md"
MANIFEST_REL = f"{SELFPLAY_BATTLES_REL}/.report_manifest.json"
SHA256_MANIFEST_REL = f"{SELFPLAY_BATTLES_REL}/.sha256_manifest.json"


# ─── Manifest 管理 ─────────────────────────────

def load_manifest(manifest_path):
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed": {}, "last_run": None}


def save_manifest(manifest_path, manifest):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_manifest_key(json_path):
    with open(json_path) as f:
        data = json.load(f)
    ep = data.get("episode", 0)
    opp = data.get("opponent_id", "unknown")
    return f"{ep}_{opp}"


def is_processed(sha256_manifest, key, sha256, manifest, filename, size, mtime):
    if sha256_manifest is not None and key in sha256_manifest:
        return sha256_manifest[key] == sha256
    entry = manifest.get("processed", {}).get(filename)
    if entry is None:
        return False
    if entry.get("size") != size:
        return False
    if entry.get("mtime") != mtime:
        return False
    return True


# ─── 范围过滤 ──────────────────────────────────

def filter_by_range(output_path, ep_range, ep_list):
    with open(output_path) as f:
        content = f.read()

    header_pattern = re.compile(r'^## Episode (\d+)\n', re.MULTILINE)
    matches = list(header_pattern.finditer(content))
    if not matches:
        return content

    preamble = content[:matches[0].start()]
    sections = []
    for i, m in enumerate(matches):
        ep_num = int(m.group(1))
        section_start = m.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((ep_num, content[section_start:section_end]))

    if ep_range is not None:
        start_ep, end_ep = ep_range
        keep = [sec for ep, sec in sections if start_ep <= ep <= end_ep]
    elif ep_list is not None:
        keep = [sec for ep, sec in sections if ep in ep_list]
    else:
        return content

    return preamble + "".join(keep)


# ─── 主函数 ────────────────────────────────────

def batch_process(output_dir, force=False, output_path=None, quiet=False, ep_range=None, ep_list=None):
    detailed_dir = os.path.join(output_dir, DETAILED_REL)
    if not os.path.isdir(detailed_dir):
        print(f"错误：detailed_battles 目录不存在 — {detailed_dir}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = os.path.join(output_dir, OUTPUT_REL)
    manifest_path = os.path.join(output_dir, MANIFEST_REL)
    sha256_manifest_path = os.path.join(output_dir, SHA256_MANIFEST_REL)

    # 收集所有 JSON 文件
    all_files = sorted(glob.glob(os.path.join(detailed_dir, "*.json")))
    if not all_files:
        print("没有找到对战日志文件", file=sys.stderr)
        return

    # 加载 manifests
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = load_manifest(manifest_path)
    sha256_manifest = load_manifest(sha256_manifest_path) if os.path.exists(sha256_manifest_path) else {}
    processed_count = 0
    skipped_count = 0

    # 确定需要处理的文件
    to_process = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        stat = os.stat(fpath)
        size = stat.st_size
        mtime = stat.st_mtime
        key = _get_manifest_key(fpath)
        sha256 = _compute_sha256(fpath)

        if not force and is_processed(sha256_manifest, key, sha256, manifest, fname, size, mtime):
            skipped_count += 1
            continue
        to_process.append((fname, fpath, size, mtime, key, sha256))

    if not to_process:
        if not quiet:
            print(f"所有 {skipped_count} 个文件已处理，无新文件")
        return

    # 打开输出文件（追加模式）
    is_new = not os.path.exists(output_path)
    with open(output_path, "a") as out:
        if is_new:
            dirname = os.path.basename(output_dir.rstrip("/"))
            out.write(f"# SelfPlay 对战摘要报告\n\n")
            out.write(f"训练目录: {dirname}\n")
            out.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        total = len(to_process)
        for i, (fname, fpath, size, mtime, key, sha256) in enumerate(to_process, 1):
            if not quiet:
                pct = i / total * 100
                print(f"  [{i}/{total}] {pct:5.1f}%  {fname}", file=sys.stderr)

            report = generate_report(fpath)

            ep = fname  # fallback
            if fname.startswith("selfplay_") and fname.endswith(".json"):
                parts = fname.split("_")
                if len(parts) >= 2:
                    try:
                        ep = f"Episode {int(parts[1])}"
                    except ValueError:
                        ep = fname

            out.write(f"---\n\n")
            out.write(f"## {ep}\n\n")
            out.write(f"源文件: {fname}\n\n")
            out.write(report)
            out.write("\n\n")

            # 更新 SHA256 manifest
            sha256_manifest[key] = sha256
            processed_count += 1

    save_manifest(sha256_manifest_path, sha256_manifest)

    output_size = os.path.getsize(output_path)
    print(f"\n完成: {processed_count} 个新文件已处理", file=sys.stderr)
    if skipped_count:
        print(f"跳过: {skipped_count} 个已处理文件", file=sys.stderr)
    print(f"输出: {output_path} ({output_size / 1024:.0f} KB)", file=sys.stderr)

    # 范围过滤
    if ep_range is not None or ep_list is not None:
        filtered = filter_by_range(output_path, ep_range, ep_list)
        print(filtered, end="")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量处理 SelfPlay 对战日志")
    parser.add_argument("output_dir", nargs="?", help="训练输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有文件")
    parser.add_argument("--output", help="输出文件路径（默认在 selfplay/selfplay_battles/all_reports.md）")
    parser.add_argument("--quiet", action="store_true", help="安静模式")

    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--ep-range", help="episode 范围，例如 400-450")
    filter_group.add_argument("--ep-list", help="episode 列表，例如 425,430")

    args = parser.parse_args()

    ep_range = None
    ep_list = None
    if args.ep_range:
        parts = args.ep_range.split("-")
        try:
            ep_range = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            parser.error("--ep-range 格式错误，应为 N-M，例如 400-450")
    if args.ep_list:
        try:
            ep_list = [int(x) for x in args.ep_list.split(",")]
        except ValueError:
            parser.error("--ep-list 格式错误，应为 N,M,...，例如 425,430")

    if not args.output_dir:
        if not args.output:
            parser.error("纯范围过滤模式需要指定 --output")
        if ep_range is None and ep_list is None:
            parser.error("纯范围过滤模式需要指定 --ep-range 或 --ep-list")
        filtered = filter_by_range(args.output, ep_range, ep_list)
        print(filtered, end="")
        return

    batch_process(
        output_dir=args.output_dir,
        force=args.force,
        output_path=args.output,
        quiet=args.quiet,
        ep_range=ep_range,
        ep_list=ep_list,
    )


if __name__ == "__main__":
    main()
