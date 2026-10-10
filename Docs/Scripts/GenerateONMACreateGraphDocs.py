"""Generate a standalone HTML or Markdown reference for a tagged Python script."""

import argparse
import ast
import base64
import html
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import quote


SOURCE_PATH = Path(__file__).with_name("ONMACreateGraph.py")


def _literal_keyword(call: ast.Call, name: str, fallback: str) -> str:
    for keyword in call.keywords:
        if keyword.arg == name:
            try:
                return str(ast.literal_eval(keyword.value))
            except (ValueError, TypeError):
                return fallback
    return fallback


def _arguments(tree: ast.Module) -> list[tuple[str, str, str]]:
    arguments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flags = ", ".join(ast.literal_eval(arg) for arg in node.args)
        except (ValueError, TypeError):
            continue
        help_text = _literal_keyword(node, "help", "No help text provided.")
        default = _literal_keyword(node, "default", "None")
        arguments.append((flags, help_text, default))
    return arguments


def _usage(docstring: str) -> str:
    match = re.search(r"Usage:\s*(.*?)(?:\n\s*\n|$)", docstring, re.DOTALL)
    return match.group(1).strip() if match else "No usage instructions provided."


def _tagged_section(source: str, tag: str, fallback: str) -> str:
    pattern = (
        r"(?m)^[ \t]*#[ \t]*\["
        + re.escape(tag)
        + r"\][ \t]*\r?\n[ \t]*([\"']{3})(.*?)\1"
    )
    match = re.search(pattern, source, re.DOTALL)
    return match.group(2).strip() if match else fallback


def _resolve_tagged_path(path: str, source_path: Path) -> Path | None:
    candidate_path = Path(path).expanduser()
    candidates = (
        [candidate_path]
        if candidate_path.is_absolute()
        else [
            source_path.parent / candidate_path,
            source_path.parent.parent / candidate_path,
            candidate_path,
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _images(source: str, source_path: Path) -> str:
    image_paths = _tagged_section(source, "Image", "").splitlines()
    figures = []
    for image_path in (path.strip() for path in image_paths):
        if not image_path:
            continue

        image_file = _resolve_tagged_path(image_path, source_path)
        if image_file is None:
            raise FileNotFoundError(f"Image referenced in [Image] section was not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(image_file.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported image type in [Image] section: {image_path}")

        encoded_image = base64.b64encode(image_file.read_bytes()).decode("ascii")
        figures.append(
            '<figure><img src="data:{};base64,{}" alt="{}">'
            "<figcaption>{}</figcaption></figure>".format(
                mime_type,
                encoded_image,
                html.escape(image_file.name, quote=True),
                html.escape(image_path),
            )
        )
    return "\n".join(figures)


def _references(source: str, source_path: Path, output_path: Path) -> str:
    reference_paths = _tagged_section(source, "Reference", "").splitlines()
    links = []
    for reference_path in (path.strip() for path in reference_paths):
        if not reference_path:
            continue

        reference_file = _resolve_tagged_path(reference_path, source_path)
        if reference_file is None:
            raise FileNotFoundError(
                f"File referenced in [Reference] section was not found: {reference_path}"
            )

        relative_path = os.path.relpath(
            reference_file.resolve(), output_path.resolve().parent
        ).replace(os.sep, "/")
        links.append(
            '<li><a href="{}"><code>{}</code></a></li>'.format(
                quote(relative_path, safe="/:"),
                html.escape(reference_path),
            )
        )
    return "\n".join(links)


def _relative_href(path: Path, output_path: Path) -> str:
    relative_path = os.path.relpath(
        path.resolve(), output_path.resolve().parent
    ).replace(os.sep, "/")
    return quote(relative_path, safe="/:")


def _markdown(source: str, source_path: Path, output_path: Path) -> str:
    tree = ast.parse(source)
    purpose = _tagged_section(source, "Purpose", "No purpose provided.")
    usage = _usage(_tagged_section(source, "Usage", ""))
    title = source_path.name
    rows = [
        (
            ", ".join(ast.literal_eval(arg) for arg in node.args),
            _literal_keyword(node, "help", "No help text provided."),
            _literal_keyword(node, "default", "None"),
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and all(
            isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            for arg in node.args
        )
    ]

    lines = [
        f"# {title}",
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Usage",
        "",
        "```text",
        usage,
        "```",
        "",
        "## Command-line arguments",
        "",
        "| Option | Description | Default |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        "| {} | {} | `{}` |".format(
            flags.replace("|", r"\|"),
            help_text.replace("|", r"\|").replace("\n", " "),
            default.replace("|", r"\|"),
        )
        for flags, help_text, default in rows
    )
    if not rows:
        lines.append("| No command-line arguments detected. | | |")

    image_paths = [
        path.strip()
        for path in _tagged_section(source, "Image", "").splitlines()
        if path.strip()
    ]
    image_links = []
    for image_path in image_paths:
        image_file = _resolve_tagged_path(image_path, source_path)
        if image_file is None:
            raise FileNotFoundError(
                f"Image referenced in [Image] section was not found: {image_path}"
            )
        image_links.append(
            f"![{Path(image_path).name}]({_relative_href(image_file, output_path)})"
        )
    if image_links:
        lines.extend(["", "## Images", ""])
        lines.extend(image_links)

    reference_paths = [
        path.strip()
        for path in _tagged_section(source, "Reference", "").splitlines()
        if path.strip()
    ]
    reference_links = []
    for reference_path in reference_paths:
        reference_file = _resolve_tagged_path(reference_path, source_path)
        if reference_file is None:
            raise FileNotFoundError(
                f"File referenced in [Reference] section was not found: {reference_path}"
            )
        reference_links.append(
            f"- [`{reference_path}`]({_relative_href(reference_file, output_path)})"
        )
    if reference_links:
        lines.extend(["", "## References", "", *reference_links])

    return "\n".join(lines).rstrip() + "\n"


def _render(
    source: str,
    source_path: Path = SOURCE_PATH,
    output_path: Path | None = None,
) -> str:
    if output_path is None:
        output_path = source_path.with_suffix(".html")

    tree = ast.parse(source)
    purpose = _tagged_section(source, "Purpose", "No purpose provided.")
    usage = _tagged_section(source, "Usage", "")
    images = _images(source, source_path)
    references = _references(source, source_path, output_path)
    title = source_path.name
    options = "\n".join(
        "<tr><td><code>{}</code></td><td>{}</td><td><code>{}</code></td></tr>".format(
            html.escape(flags),
            html.escape(help_text),
            html.escape(default),
        )
        for flags, help_text, default in _arguments(tree)
    )
    if not options:
        options = '<tr><td colspan="3">No command-line arguments detected.</td></tr>'

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - HTML Reference</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }}
    h1, h2 {{ line-height: 1.25; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #8886; padding: .6rem; text-align: left; }}
    th {{ background: #8882; }}
    pre {{ overflow-x: auto; padding: 1rem; border-radius: .4rem; background: #8882; }}
    code {{ font-family: ui-monospace, monospace; }}
    figure {{ margin: 1rem 0; }}
    figure img {{ display: block; max-width: 100%; height: auto; }}
    figcaption {{ color: #888; font-size: .9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <section>
      <h2>Purpose</h2>
      <p>{purpose}</p>
    </section>
    <section>
      <h2>Usage</h2>
      <pre><code>{usage}</code></pre>
    </section>
    <section>
      <h2>Command-line arguments</h2>
      <table>
        <thead><tr><th>Option</th><th>Description</th><th>Default</th></tr></thead>
        <tbody>{options}</tbody>
      </table>
    </section>
    {image_section}
    {reference_section}
  </main>
</body>
</html>
""".format(
        title=html.escape(title),
        purpose=html.escape(purpose),
        usage=html.escape(_usage(usage)),
        options=options,
        image_section=(
            "<section><h2>Images</h2>{}</section>".format(images) if images else ""
        ),
        reference_section=(
            "<section><h2>References</h2><ul>{}</ul></section>".format(references)
            if references
            else ""
        ),
        source=html.escape(source),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML or Markdown documentation for a tagged Python source file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=SOURCE_PATH,
        help="Python source file to document (default: Tools/ONMACreateGraph.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: input file with the selected format suffix)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("html", "md", "markdown"),
        default="html",
        help="Output format (default: html)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_format = "md" if args.format == "markdown" else args.format
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_suffix(f".{output_format}")
    )
    source = input_path.read_text(encoding="utf-8")
    document = (
        _render(source, input_path, output_path)
        if output_format == "html"
        else _markdown(source, input_path, output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"Documentation written to {output_path}")


if __name__ == "__main__":
    main()
