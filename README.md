# Data and code for "Crab: A Novel Configurable Role-Playing LLM with Assessing Benchmark"

## Model
Our pretrained LLMs can be downloaded from https://huggingface.co/HeAAAAA/Crab/tree/main/Crab/model

## Data
### Training data for Crab
`data\train\crab_train_data.json`

### Training data for RoleRM
`data\train\rolerm_train_data.jsonl`

### Benchmark
`data\benchmark\benchmark.json`
This is the system prompt and queries to benchmark the role-playing capabilities on each model.
We also include a reference response for each query.

### result
`data\result\*`
Each record contains the generated responses on our benchmarks and their scores judged by RoleRM.
The record with `uid 439` is the case study presents at Table 5.

Note: Due to the size limitation, we only provide a random subset for each datasets: 1000 for training data, 100 for benchmark data and results here. Complete data will publish at Huggingface later.

## Code
### Training data for Crab-Llama3.1
See `codes\TrainLLM\README.md`.
