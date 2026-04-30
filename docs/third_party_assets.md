# Third-Party Assets

| Asset | Version/revision | Role | License | Source/citation | Included in release |
|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | cached Hugging Face revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`; converted from `Qwen/Qwen2.5-7B-Instruct` with `mlx-lm` 0.18.1 per model card | evaluated actor model | Apache-2.0 | https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit | no, referenced |
| Qwen tokenizer bundled with `mlx-community/Qwen2.5-7B-Instruct-4bit` | same cached revision as actor model | tokenization | Apache-2.0 | https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit | no, referenced |
| `mlx-lm` | project requirement `>=0.21`; local verification used 0.31.3; exact original generation package version was not retained in v1.0 run summaries | MLX text generation backend | MIT | https://pypi.org/project/mlx-lm/ | code dependency |
| `BAAI/bge-small-en-v1.5` | cached Hugging Face revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | dense retrieval embedding model | MIT | https://huggingface.co/BAAI/bge-small-en-v1.5 | no, referenced |
| `sentence-transformers` | project requirement `>=3.0`; local verification used 5.4.1 | embedding model runtime | Apache-2.0 | https://pypi.org/project/sentence-transformers/ | dependency file |
| `numpy` | project requirement `>=1.26`; local verification used 2.4.4 | array operations and dense index loading | BSD-3-Clause family license expression in package metadata | https://pypi.org/project/numpy/ | dependency file |
| `pydantic` | project requirement `>=2.0`; local verification used 2.13.3 | data validation schemas | MIT | https://pypi.org/project/pydantic/ | dependency file |
| `pytest` | project requirement `>=8.0`; local verification used 9.0.3 for current tests | test runner | MIT | https://pypi.org/project/pytest/ | dependency file |
| MEMTRACE synthetic corpus | v1.0 | new benchmark asset | MIT, anonymized submission copy | paper and release bundle | yes |
| MEMTRACE traces | v1.0 main 324-trace set plus 72 calibration traces | new benchmark asset | MIT, anonymized submission copy | paper and release bundle | yes |

The anonymized review artifact withholds author identity in the license header; the public camera-ready release should restore the non-anonymous copyright holder.
