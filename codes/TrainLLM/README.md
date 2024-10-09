# Code to Train Crab
This is the code to fine-tune crab-Llama3.1 using LoRA.
Most of the codebase is from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
1. Setup environment: `./install_deps.sh`
2. Run LoRA fine-tuning: `torchrun --nproc_per_node 2 python run_finetune.py`
3. Merge LoRA weights: `python run_export.py`
4. Run benchmark: `python run_benchmark.py`