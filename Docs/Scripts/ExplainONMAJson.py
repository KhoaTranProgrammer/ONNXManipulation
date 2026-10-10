"""Generate an HTML or Markdown explanation of an ONMA graph JSON file.

Usage:
python Docs/Scripts/ExplainONMAJson.py --input Docs/Sample/Conv.json --format html
python Docs/Scripts/ExplainONMAJson.py --input Docs/Sample/Conv.json --output Docs/Sample/Conv.md --format md
"""

import argparse
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote


DEFAULT_INPUT = Path(__file__).parent.parent / "Sample" / "Conv.json"


def _display(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _shape(tensor: dict[str, Any]) -> str:
    shape = tensor.get("shape")
    return _display(shape) if shape is not None else "Not specified"


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, list):
        return [number for item in value for number in _numeric_values(item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def _tensor_interpretation(tensor: dict[str, Any]) -> str:
    shape = tensor.get("shape")
    if isinstance(shape, list) and len(shape) == 4:
        return (
            f"One batch, {shape[1]} channel(s), "
            f"{shape[2]} × {shape[3]} spatial dimensions (NCHW)."
        )
    if isinstance(shape, list):
        return f"A rank-{len(shape)} tensor with the dimensions shown in its shape."
    return "Tensor dimensions are not specified."


def _kernel_matrix(tensor: dict[str, Any], kernel: list[Any]) -> str | None:
    data = tensor.get("data")
    if (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], list)
        and len(data[0]) == 1
        and isinstance(data[0][0], list)
        and len(data[0][0]) == kernel[0]
        and all(isinstance(row, list) and len(row) == kernel[1] for row in data[0][0])
    ):
        return "\n".join(" ".join(str(value) for value in row) for row in data[0][0])
    return None


def _is_average_kernel(tensor: dict[str, Any], kernel: list[Any]) -> bool:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[:2] != [1, 1]
        or not all(isinstance(size, int) and size > 0 for size in kernel)
    ):
        return False
    values = _numeric_values(tensor.get("data"))
    expected_value = 1 / (kernel[0] * kernel[1])
    return len(values) == kernel[0] * kernel[1] and all(
        math.isclose(value, expected_value) for value in values
    )


def _conv_output_shape(
    activation_shape: Any, weight_shape: Any, attributes: dict[str, Any]
) -> list[int] | None:
    if (
        not isinstance(activation_shape, list)
        or len(activation_shape) != 4
        or not isinstance(weight_shape, list)
        or len(weight_shape) != 4
        or not all(
            isinstance(value, int) and value > 0
            for value in activation_shape + weight_shape
        )
    ):
        return None

    kernel = attributes.get("kernel_shape", weight_shape[-2:])
    strides = attributes.get("strides", [1, 1])
    dilations = attributes.get("dilations", [1, 1])
    pads = attributes.get("pads", [0, 0, 0, 0])
    if not (
        isinstance(kernel, list)
        and len(kernel) == 2
        and isinstance(strides, list)
        and len(strides) == 2
        and isinstance(dilations, list)
        and len(dilations) == 2
        and isinstance(pads, list)
        and len(pads) == 4
        and all(
            isinstance(value, int) and value > 0
            for value in kernel + strides + dilations
        )
        and all(isinstance(value, int) and value >= 0 for value in pads)
    ):
        return None

    height = math.floor(
        (
            activation_shape[2]
            + pads[0]
            + pads[2]
            - dilations[0] * (kernel[0] - 1)
            - 1
        )
        / strides[0]
        + 1
    )
    width = math.floor(
        (
            activation_shape[3]
            + pads[1]
            + pads[3]
            - dilations[1] * (kernel[1] - 1)
            - 1
        )
        / strides[1]
        + 1
    )
    return [activation_shape[0], weight_shape[0], height, width]


def _conv_explanation(
    node: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    initializers: dict[str, dict[str, Any]],
) -> str:
    node_inputs = node.get("inputs", [])
    weight = initializers.get(node_inputs[1]) if len(node_inputs) > 1 else None
    weight_shape = weight.get("shape") if weight else None
    attributes = node.get("attributes", {})
    kernel = attributes.get(
        "kernel_shape",
        weight_shape[-2:] if isinstance(weight_shape, list) and len(weight_shape) >= 2 else None,
    )
    strides = attributes.get("strides", [1, 1])
    dilations = attributes.get("dilations", [1, 1])
    pads = attributes.get("pads", [0, 0, 0, 0])

    explanation = (
        f"This node applies an ONNX Conv operation to "
        f"`{node_inputs[0] if node_inputs else 'the activation input'}`"
    )
    if len(node_inputs) > 1:
        explanation += f" using `{node_inputs[1]}` as its weight input"
    explanation += "."
    if kernel is not None:
        explanation += f" The kernel shape is `{_display(kernel)}`."
    if strides is not None:
        explanation += f" The strides are `{_display(strides)}`."
    if dilations != [1, 1]:
        explanation += f" The dilations are `{_display(dilations)}`."
    if pads is not None:
        explanation += (
            f" Padding is `{_display(pads)}` (ONNX order: "
            "beginning pads followed by ending pads for each spatial axis)."
        )

    activation = inputs.get(node_inputs[0]) if node_inputs else None
    activation_shape = activation.get("shape") if activation else None
    if (
        isinstance(activation_shape, list)
        and len(activation_shape) == 4
        and isinstance(weight_shape, list)
        and len(weight_shape) == 4
        and isinstance(kernel, list)
        and len(kernel) == 2
        and isinstance(strides, list)
        and len(strides) == 2
        and isinstance(dilations, list)
        and len(dilations) == 2
        and isinstance(pads, list)
        and len(pads) == 4
        and all(isinstance(v, int) and v > 0 for v in strides)
        and all(isinstance(v, int) and v > 0 for v in dilations)
        and all(isinstance(v, int) and v >= 0 for v in pads)
        and all(isinstance(v, int) and v > 0 for v in kernel)
        and all(isinstance(v, int) and v > 0 for v in activation_shape[2:])
    ):
        height = math.floor(
            (
                activation_shape[2]
                + pads[0]
                + pads[2]
                - dilations[0] * (kernel[0] - 1)
                - 1
            )
            / strides[0]
            + 1
        )
        width = math.floor(
            (
                activation_shape[3]
                + pads[1]
                + pads[3]
                - dilations[1] * (kernel[1] - 1)
                - 1
            )
            / strides[1]
            + 1
        )
        explanation += (
            f" For the declared input shape `{_display(activation_shape)}`, "
            f"the calculated spatial output size is `{height} × {width}` "
            "using the Conv kernel, stride, and padding settings."
        )

    if (
        weight is not None
        and isinstance(weight_shape, list)
        and len(weight_shape) == 4
        and weight_shape[:2] == [1, 1]
        and isinstance(kernel, list)
        and len(kernel) == 2
        and all(isinstance(size, int) and size > 0 for size in kernel)
        and len(node_inputs) == 2
    ):
        values = _numeric_values(weight.get("data"))
        expected_value = 1 / (kernel[0] * kernel[1])
        if len(values) == kernel[0] * kernel[1] and all(
            math.isclose(value, expected_value) for value in values
        ):
            explanation += (
                f" The `{node_inputs[1]}` weights are all "
                f"`{_display(expected_value)}` and sum to 1, "
                "so each output is the average of its input window."
            )

    if len(node_inputs) > 2:
        explanation += f" The third input, `{node_inputs[2]}`, is the optional bias."
    elif weight is not None:
        explanation += " No bias input is specified."
    return explanation


def _sections(
    data: dict[str, Any], input_path: Path, output_path: Path
) -> list[tuple[str, Any]]:
    graph = data.get("graph")
    if not isinstance(graph, dict):
        raise ValueError("The input JSON must contain a 'graph' object.")

    inputs = graph.get("inputs", [])
    outputs = graph.get("outputs", [])
    initializers = graph.get("initializers", [])
    nodes = graph.get("nodes", [])
    if not all(isinstance(items, list) for items in (inputs, outputs, initializers, nodes)):
        raise ValueError("Graph inputs, outputs, initializers, and nodes must be arrays.")

    input_by_name = {
        tensor["name"]: tensor for tensor in inputs if isinstance(tensor, dict) and "name" in tensor
    }
    initializer_by_name = {
        tensor["name"]: tensor
        for tensor in initializers
        if isinstance(tensor, dict) and "name" in tensor
    }
    content: list[tuple[str, Any]] = [
        (
            "paragraph",
            "This sample describes an ONMA graph containing its input tensors, "
            "constant initializers, computation nodes, and outputs.",
        ),
        ("heading", "Graph overview"),
        (
            "list",
            (
                True,
                [
                    f"Provide input tensor `{tensor.get('name', 'Unnamed')}` "
                    f"with shape `{_shape(tensor)}`."
                    for tensor in inputs
                    if isinstance(tensor, dict)
                ]
                + [
                    f"Apply the `{node.get('op_type', 'Unknown')}` node "
                    f"`{node.get('name', f'Node {index}')}` using inputs "
                    f"`{_display(node.get('inputs', []))}`."
                    for index, node in enumerate(nodes, start=1)
                    if isinstance(node, dict)
                ]
                + [
                    f"Produce output tensor `{tensor.get('name', 'Unnamed')}` "
                    f"with shape `{_shape(tensor)}`."
                    for tensor in outputs
                    if isinstance(tensor, dict)
                ],
            ),
        ),
        ("heading", "Top-level model"),
        (
            "table",
            (
                ["Field", "Meaning", "Sample value"],
                [
                    [
                        "name",
                        "Human-readable model name.",
                        str(data.get("name", "Not specified")),
                    ],
                    [
                        "graph",
                        "Inputs, outputs, initializers, and computation nodes.",
                        "Object",
                    ],
                ],
            ),
        ),
        ("heading", "Input and output tensors"),
        (
            "paragraph",
            "Four-dimensional tensor shapes use ONNX NCHW order: batch, channels, height, width.",
        ),
        (
            "table",
            (
                ["Tensor", "Shape", "Interpretation", "Data type"],
                [
                    [
                        str(tensor.get("name", "Unnamed")),
                        _shape(tensor),
                        _tensor_interpretation(tensor),
                        str(tensor.get("data_type", "Not specified")),
                    ]
                    for tensor in inputs + outputs
                    if isinstance(tensor, dict)
                ],
            ),
        ),
        (
            "paragraph",
            "Input entries declare tensor metadata, not runtime input values. "
            "Values must be supplied when the generated model runs.",
        ),
        ("heading", "Weight initializers"),
    ]

    consumers = {
        tensor_name: node
        for node in nodes
        if isinstance(node, dict)
        for tensor_name in node.get("inputs", [])
    }
    initializer_rows = [
        [
            str(tensor.get("name", "Unnamed")),
            _shape(tensor),
            str(tensor.get("data_type", "Not specified")),
            (
                "Convolution weight tensor."
                if isinstance(consumers.get(tensor.get("name")), dict)
                and consumers[tensor.get("name")].get("op_type") == "Conv"
                else "Constant tensor data stored in the JSON."
            ),
        ]
        for tensor in initializers
        if isinstance(tensor, dict)
    ]
    content.append(
        (
            "table",
            (["Name", "Shape", "Data type", "Purpose"], initializer_rows),
        )
    )
    if not initializers:
        content.append(("paragraph", "This graph does not define any initializers."))

    for tensor in initializers:
        if not isinstance(tensor, dict):
            continue
        conv_consumers = [
            node
            for node in nodes
            if isinstance(node, dict)
            and node.get("op_type") == "Conv"
            and tensor.get("name") in node.get("inputs", [])
        ]
        if not conv_consumers:
            continue
        weight_shape = tensor.get("shape")
        attributes = conv_consumers[0].get("attributes", {})
        kernel = attributes.get(
            "kernel_shape",
            weight_shape[-2:]
            if isinstance(weight_shape, list) and len(weight_shape) >= 2
            else None,
        )
        if (
            isinstance(weight_shape, list)
            and len(weight_shape) == 4
            and isinstance(kernel, list)
            and len(kernel) == 2
        ):
            content.append(
                (
                    "paragraph",
                    f"`{tensor.get('name')}` has shape `{_shape(tensor)}`: "
                    f"{weight_shape[0]} output channel(s), "
                    f"{weight_shape[1]} input channel(s), and a "
                    f"`{_display(kernel)}` kernel.",
                )
            )
            matrix = _kernel_matrix(tensor, kernel)
            if matrix is not None:
                content.extend(
                    [
                        ("pre", f"{tensor.get('name')}[0, 0]:\n{matrix}"),
                        (
                            "paragraph",
                            "At each output position, the kernel is multiplied "
                            "element by element with an input window and the "
                            "products are summed.",
                        ),
                    ]
                )

    content.append(("heading", "Convolution node"))
    if not nodes:
        content.append(("paragraph", "This graph does not define any nodes."))
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            raise ValueError(f"Graph node {index} must be a JSON object.")
        name = node.get("name", f"Node {index}")
        op_type = node.get("op_type", "Unknown operation")
        if op_type == "Conv":
            attribute_meanings = {
                "kernel_shape": "Height and width of the kernel.",
                "strides": "Step size vertically and horizontally.",
                "pads": (
                    "Padding at the beginning of each spatial axis, followed "
                    "by padding at the end of each axis."
                ),
                "dilations": "Spacing between kernel elements.",
                "group": "Number of groups dividing input and output channels.",
            }
            attributes = node.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError(f"Attributes for node {name!r} must be an object.")
            node_rows = [
                ["name", str(name), "Identifies this graph node."],
                ["op_type", str(op_type), "ONNX convolution operator."],
                [
                    "inputs",
                    _display(node.get("inputs", [])),
                    "Input activation, weights, and optional bias, in that order.",
                ],
                [
                    "outputs",
                    _display(node.get("outputs", [])),
                    "Names of the tensors produced by this node.",
                ],
            ]
            node_rows.extend(
                [
                    [
                        str(attribute),
                        _display(value),
                        attribute_meanings.get(
                            str(attribute), "ONNX operator attribute."
                        ),
                    ]
                    for attribute, value in attributes.items()
                ]
            )
            content.extend(
                [
                    ("subheading", f"{name} — Conv"),
                    ("table", (["Field", "Value", "Meaning"], node_rows)),
                    (
                        "paragraph",
                        _conv_explanation(node, input_by_name, initializer_by_name),
                    ),
                ]
            )

            node_inputs = node.get("inputs", [])
            activation = input_by_name.get(node_inputs[0]) if node_inputs else None
            weight = (
                initializer_by_name.get(node_inputs[1])
                if len(node_inputs) > 1
                else None
            )
            attributes = node.get("attributes", {})
            calculated_shape = _conv_output_shape(
                activation.get("shape") if activation else None,
                weight.get("shape") if weight else None,
                attributes,
            )
            if calculated_shape:
                kernel = attributes.get("kernel_shape", weight["shape"][-2:])
                if len(node_inputs) == 2 and _is_average_kernel(weight, kernel):
                    content.append(
                        (
                            "note",
                            f"The weights are a normalized `{_display(kernel)}` "
                            f"kernel, so the {calculated_shape[2] * calculated_shape[3]} "
                            "output values are overlapping input-window averages.",
                        )
                    )
                strides = attributes.get("strides", [1, 1])
                dilations = attributes.get("dilations", [1, 1])
                pads = attributes.get("pads", [0, 0, 0, 0])
                input_shape = activation["shape"]
                height_calculation = (
                    f"floor(({input_shape[2]} + {pads[0]} + {pads[2]} "
                    f"- {dilations[0]} × ({kernel[0]} - 1) - 1) "
                    f"/ {strides[0]}) + 1 = {calculated_shape[2]}"
                )
                width_calculation = (
                    f"floor(({input_shape[3]} + {pads[1]} + {pads[3]} "
                    f"- {dilations[1]} × ({kernel[1]} - 1) - 1) "
                    f"/ {strides[1]}) + 1 = {calculated_shape[3]}"
                )
                content.extend(
                    [
                        ("subheading", "Why is the output this size?"),
                        (
                            "paragraph",
                            f"For input `{_shape(activation)}`, kernel "
                            f"`{_display(kernel)}`, stride `{_display(strides)}`, "
                            f"and padding `{_display(pads)}`, the spatial output "
                            "dimensions are calculated as "
                            f"`{height_calculation}` for height and "
                            f"`{width_calculation}` for width. "
                            f"The calculated output shape is "
                            f"`{_display(calculated_shape)}`.",
                        ),
                    ]
                )
                declared_output = next(
                    (
                        tensor
                        for tensor in outputs
                        if isinstance(tensor, dict)
                        and tensor.get("name") in node.get("outputs", [])
                    ),
                    None,
                )
                if declared_output:
                    declared_shape = declared_output.get("shape")
                    matches = declared_shape == calculated_shape
                    content.append(
                        (
                            "note",
                            (
                                f"The declared output shape `{_display(declared_shape)}` "
                                "matches the calculated shape."
                                if matches
                                else f"The declared output shape `{_display(declared_shape)}` "
                                f"differs from the calculated shape `{_display(calculated_shape)}`."
                            ),
                        )
                    )
            continue

        content.append(("subheading", f"{name} — {op_type}"))
        content.append(
            (
                "table",
                (
                    ["Field", "Value"],
                    [
                        ["Inputs", _display(node.get("inputs", []))],
                        ["Outputs", _display(node.get("outputs", []))],
                    ],
                ),
            )
        )
        attributes = node.get("attributes", {})
        if attributes:
            content.append(
                (
                    "table",
                    (
                        ["Attribute", "Value"],
                        [[str(key), _display(value)] for key, value in attributes.items()],
                    ),
                )
            )

    relative_input = os.path.relpath(
        input_path.resolve(), output_path.resolve().parent
    ).replace(os.sep, "/")
    content.extend(
        [
            ("heading", "Source file"),
            ("link", (input_path.name, f'{relative_input}')),
        ]
    )
    return content


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", r"\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _render_markdown(title: str, sections: list[tuple[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for kind, value in sections:
        if kind == "heading":
            lines.extend([f"## {value}", ""])
        elif kind == "subheading":
            lines.extend([f"### {value}", ""])
        elif kind == "paragraph":
            lines.extend([value, ""])
        elif kind == "table":
            headers, rows = value
            lines.extend([_markdown_table(headers, rows), ""])
        elif kind == "list":
            ordered, items = value
            lines.extend(
                [
                    f"{index}. {item}" if ordered else f"- {item}"
                    for index, item in enumerate(items, start=1)
                ]
            )
            lines.append("")
        elif kind == "pre":
            lines.extend(["```text", value, "```", ""])
        elif kind == "note":
            lines.extend([f"> **Expected behavior:** {value}", ""])
        elif kind == "link":
            label, href = value
            lines.extend([f"- [{label}]({quote(href, safe='/')})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_html(title: str, sections: list[tuple[str, Any]]) -> str:
    body = []

    def inline_markup(text: str) -> str:
        pieces = re.split(r"(`[^`]+`)", text)
        return "".join(
            "<code>{}</code>".format(html.escape(piece[1:-1]))
            if piece.startswith("`") and piece.endswith("`")
            else html.escape(piece)
            for piece in pieces
        )

    def render_table(headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{inline_markup(cell)}</th>" for cell in headers)
        table_rows = "".join(
            "<tr>{}</tr>".format(
                "".join(f"<td>{inline_markup(cell)}</td>" for cell in row)
            )
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{table_rows}</tbody></table>"

    for kind, value in sections:
        if kind == "heading":
            body.append(f"<h2>{html.escape(value)}</h2>")
        elif kind == "subheading":
            body.append(f"<h3>{inline_markup(value)}</h3>")
        elif kind == "paragraph":
            body.append(f"<p>{inline_markup(value)}</p>")
        elif kind == "table":
            headers, rows = value
            body.append(render_table(headers, rows))
        elif kind == "list":
            ordered, items = value
            tag = "ol" if ordered else "ul"
            body.append(
                f"<{tag}>"
                + "".join(f"<li>{inline_markup(item)}</li>" for item in items)
                + f"</{tag}>"
            )
        elif kind == "pre":
            body.append(f"<pre><code>{html.escape(value)}</code></pre>")
        elif kind == "note":
            body.append(
                f'<div class="note"><strong>Expected behavior:</strong> '
                f"{inline_markup(value)}</div>"
            )
        elif kind == "link":
            label, href = value
            body.append(
                '<p><a href="{}">{}</a></p>'.format(
                    html.escape(quote(href, safe="/"), quote=True),
                    html.escape(label),
                )
            )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #8886; padding: .6rem; text-align: left; vertical-align: top; }}
    th {{ background: #8882; }}
    code {{ font-family: ui-monospace, monospace; }}
    pre {{ overflow-x: auto; padding: 1rem; border-radius: .4rem; background: #8882; }}
    .note {{ border-left: .25rem solid #8888; padding: .5rem 1rem; }}
  </style>
</head>
<body>
  <main>
    <h1>{}</h1>
    {}
  </main>
</body>
</html>
""".format(
        html.escape(title),
        html.escape(title),
        "\n    ".join(body),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain an ONMA graph JSON file in HTML or Markdown."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="ONMA JSON input (default: Docs/Sample/Conv.json from the repository root)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output document path (default: input file with selected format extension)",
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
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The input JSON root must be an object.")

    title = f"Explanation of {input_path.name}"
    sections = _sections(data, input_path, output_path)
    document = (
        _render_html(title, sections)
        if output_format == "html"
        else _render_markdown(title, sections)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"Explanation written to {output_path}")


if __name__ == "__main__":
    main()
