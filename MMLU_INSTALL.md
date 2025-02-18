# MMLU Install part
Steps 1-2 is also a part of pyproject.toml so you can also do `pip install -e .` on this directory.
1. do `pip install --upgrade lm_eval`
2. do `pip install peft==0.10.0`
3. if you get problems with awq not having WQ_MODULE_GEMM you need to go to `site_packages/peft/import_utils.py` and modify the following line at the bottom of the file:
```python
def is_auto_awq_available():
    return False # change this
```
