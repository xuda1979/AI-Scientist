# AI Scientist

AI Scientist orchestrates the generation, review, and refinement of research papers using large language models.  The workflow can target open-source models and run on CPUs, GPUs, or NPUs.

## OpenAI OSS 120B Model

An optional placeholder for the open-source **OpenAI OSS 120B** model lives in `models/openai_120b/`.  Obtain the official source and weights and place them in that directory.  Select it by setting `"default_model": "openai-oss-120b"` in your configuration.

## NPU Support

The workflow can execute on Ascend 910B NPUs.  Specify `"device": "npu"` and install the required `torch` and `torch-npu` packages along with the appropriate Ascend drivers.

## Modular Review and LaTeX Fix Cycles

Review, edit, and revision steps are decomposed into separate API calls that write their artifacts to disk, allowing parallel execution across multiple devices.  The `latex_fix_cycle` iteratively compiles and patches LaTeX sources until they succeed, saving logs of each attempt.

## Testing

Run the test suite with:

```bash
pytest
```
