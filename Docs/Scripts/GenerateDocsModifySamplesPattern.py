import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnx.tools.net_drawer import GetPydotGraph


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "Docs" / "Sample" / "ModifyPattern"


def _format_introductions(introductions):
    lines = ["## Introduction", ""]
    paragraph = []
    code_block = []

    def flush_paragraph():
        if paragraph:
            lines.append(" ".join(part.strip() for part in paragraph))
            lines.append("")
            paragraph.clear()

    def flush_code_block():
        if code_block:
            lines.append("```text")
            lines.extend(code_block)
            lines.append("```")
            lines.append("")
            code_block.clear()

    for source_line in introductions:
        line = source_line.strip()
        if not line or re.fullmatch(r"=+", line):
            flush_paragraph()
            flush_code_block()
            continue

        section = re.fullmatch(r"([A-Z][A-Z0-9 /_-]*):(?:\s*(.*))?", line)
        if section:
            flush_paragraph()
            flush_code_block()
            lines.extend([f"### {section.group(1).title()}", ""])
            if section.group(2):
                paragraph.append(section.group(2))
            continue

        if source_line[:1].isspace():
            flush_paragraph()
            code_block.append(line)
            continue

        flush_code_block()
        inline_label = re.match(r"([^:]+):\s+(.+)", line)
        if inline_label and not line.startswith(("-", "*")):
            line = f"**{inline_label.group(1)}:** {inline_label.group(2)}"

        if re.match(r"^(?:[-*]|\d+[.)])\s+", line):
            flush_paragraph()
            lines.append(line)
        else:
            paragraph.append(line)

    flush_paragraph()
    flush_code_block()
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _load_introductions(pattern_path):
    with pattern_path.open("r", encoding="utf-8") as pattern_file:
        pattern_document = json.load(pattern_file)

    if not isinstance(pattern_document, dict) or len(pattern_document) != 1:
        raise ValueError(f"Expected one named pattern in {pattern_path}")

    pattern = next(iter(pattern_document.values()))
    introductions = pattern.get("Introductions") if isinstance(pattern, dict) else None
    if not isinstance(introductions, list) or not all(
        isinstance(line, str) for line in introductions
    ):
        raise ValueError(
            f'Pattern in {pattern_path} must contain a string-list "Introductions"'
        )
    return introductions


def _write_pattern_markdown(pattern_name, introductions, search_image, replace_image):
    markdown_path = OUTPUT_DIR / f"{pattern_name}.md"
    lines = [
        f"# {pattern_name}",
        "",
        *_format_introductions(introductions),
        "",
        "## SearchBy",
        "",
        f"![SearchBy]({search_image.name})",
        "",
        "## ReplaceBy",
        "",
        f"![ReplaceBy]({replace_image.name})",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_markdown(patterns_by_type):
    lines = [
        "# Modify Sample Patterns",
        "",
        "Select a pattern to view its introduction and SearchBy/ReplaceBy graphs.",
    ]
    for pattern_type in sorted(patterns_by_type, key=str.casefold):
        lines.extend(
            [
                "",
                f"## {pattern_type}",
                "",
            ]
        )
        lines.extend(
            f"- [{pattern_name}]({pattern_name}.md)"
            for pattern_name in sorted(
                patterns_by_type[pattern_type], key=str.casefold
            )
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_graphs_and_markdown(row, onnx_index, modify_sample_index):
    onnx_path = ROOT_DIR / row[onnx_index]
    pattern_path = ROOT_DIR / row[modify_sample_index]
    pattern_name = pattern_path.stem
    introductions = _load_introductions(pattern_path)

    search_image = OUTPUT_DIR / f"{pattern_name}_SearchBy.png"
    replace_image = OUTPUT_DIR / f"{pattern_name}_ReplaceBy.png"

    with tempfile.TemporaryDirectory(prefix="modify-pattern-") as temp_dir:
        input_model = Path(temp_dir) / "input.onnx"
        output_model = Path(temp_dir) / "output.onnx"

        subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "Tools" / "ONMACreateGraph.py"),
                "--input",
                str(onnx_path),
                "--output",
                str(input_model),
            ],
            cwd=ROOT_DIR,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "Tools" / "ONMAModifyModel.py"),
                "--modify",
                str(pattern_path),
                "--input",
                str(input_model),
                "--output",
                str(output_model),
            ],
            cwd=ROOT_DIR,
            check=True,
        )

        for model_path, image_path in (
            (input_model, search_image),
            (output_model, replace_image),
        ):
            model = onnx.load(str(model_path))
            graph = GetPydotGraph(model.graph, name=model.graph.name, rankdir="LR")
            graph.write_png(str(image_path))

    _write_pattern_markdown(pattern_name, introductions, search_image, replace_image)
    print(f"Created documentation for {pattern_name}")
    return pattern_name


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Markdown documentation and SearchBy/ReplaceBy images for "
            "each unique ModifySample in a CSV file."
        )
    )
    parser.add_argument(
        "--csv_file",
        type=Path,
        required=True,
        help="CSV input containing Onnx and ModifySample columns",
    )
    args = parser.parse_args()

    with args.csv_file.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        header = next(reader, None)
        if header is None:
            parser.error("input CSV is empty")

        try:
            onnx_index = header.index("Onnx")
            modify_sample_index = header.index("ModifySample")
            type_index = header.index("Type")
        except ValueError:
            parser.error(
                'input CSV must contain "Onnx", "ModifySample", and "Type" columns'
            )

        first_row_by_pattern = {}
        for row in reader:
            if len(row) <= max(onnx_index, modify_sample_index, type_index):
                parser.error(f"invalid CSV row: {row}")
            first_row_by_pattern.setdefault(row[modify_sample_index], row)

    if not first_row_by_pattern:
        parser.error("input CSV contains no pattern rows")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns_by_type = {}
    for row in first_row_by_pattern.values():
        pattern_name = export_graphs_and_markdown(
            row, onnx_index, modify_sample_index
        )
        patterns_by_type.setdefault(row[type_index], []).append(pattern_name)
    _write_summary_markdown(patterns_by_type)


if __name__ == "__main__":
    main()
