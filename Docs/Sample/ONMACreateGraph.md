# ONMACreateGraph.py

## Purpose

This tool is designed to convert a model represented in the ONMA JSON format into an ONNX file.
It reads and parses the input ONMA JSON file, processes the model structure and its associated parameters,
and then generates an ONNX model file that can be used with ONNX-compatible frameworks and inference runtimes.

## Usage

```text
python Tools/ONMACreateGraph.py --input Sample/Conv.json --output Sample/Conv.onnx
```

## Command-line arguments

| Option | Description | Default |
| --- | --- | --- |
| --input, -in | Create graph from json | `Sample/Conv.json` |
| --output, -ou | Output onnx file | `Sample/Conv.onnx` |

## Images

![ConvSample.gif](ConvSample.gif)

## References

- [`Sample/Conv.json`](../../Sample/Conv.json)
- [`Sample/Conv.onnx`](../../Sample/Conv.onnx)
