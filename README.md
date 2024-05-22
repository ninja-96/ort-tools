# ORT tools

Userful tools for ONNX models

## Installation

Install using `pip`\
From source:

```bash
pip3 install git+https://github.com/ninja-96/ort-tools
```

## Getting Started

1. Quantize ONNX mode
```bash
python3 -m ort_tools.quant -m <path to onnx model>
```

2. Speedtest ONNX model
```bash
python3 -m ort_tools.speedtest -m <path to onnx model>
```

### Note

- Quantization module quantize model using random input data.
- You must set `--shapes` argument to run tools:
```bash
python3 -m ort_tools.speedtest -m ./model.onnx --shape images:1x3x224x224:fp32
```

If ONNX model has 2 or more inputs, just enumerate input names separated by commas:
```bash
python3 -m ort_tools.speedtest -m ./model.onnx --shape images:1x3x224x224:fp32,gray_images:1x3x224x224:fp32
```

## Built with

- [onnx](https://github.com/onnx/onnx) - Open standard for machine learning interoperability 
- [onnxruntime](https://github.com/microsoft/onnxruntime) - ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator 

## Versioning

All versions available, see the [tags on this repository](https://github.com/ninja-96/ort-tools/tags).

## Authors

- **Oleg Kachalov** - _Initial work_ - [ninja-96](https://github.com/ninja-96)

See also the list of [contributors](https://github.com/ninja-96/ort-tools/contributors) who participated in this project.

## License

This project is licensed under the GPL-3.0 license - see the [LICENSE.md](./LICENSE) file for details.
