import json
import random
random.seed(42)


input_file = "data_score.json"
out_train_file = "data_score_train.json"
out_test_file = "data_score_test.json"

with open(input_file, 'r') as f:
    data = json.load(f)

random.shuffle(data)

test_num = int(len(data)*0.1)
print(test_num)

test_data = data[0:test_num]
train_data = data[test_num:]

with open(out_train_file, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(out_test_file, 'w') as f:
    json.dump(test_data, f, indent=4)