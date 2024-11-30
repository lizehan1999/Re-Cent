import json
import os


def data2train(filename):
    labels = set()
    final_data = []
    with open(filename, "r") as f:
        for i in f:
            json_data = json.loads(i)
            data_dict = {"tokenized_text": [], "subject": [], "object": []}
            for t in json_data['triplets']:
                data_dict["tokenized_text"] = t['tokens']
                head = [t['head'][0], t['head'][-1], t['label']]
                assert t['head'][-1] - t['head'][0] == len(t['head']) - 1
                tail = [t['tail'][0], t['tail'][-1], t['label']]
                assert t['tail'][-1] - t['tail'][0] == len(t['tail']) - 1
                data_dict["subject"].append(head)
                data_dict["object"].append(tail)
                labels.add(t['label'])
                labels.add(t['label'])
            final_data.append(data_dict)
    return list(labels), final_data


def data_process(in_file, out_file):
    _, train = data2train(in_file + "train.jsonl")
    dev_labels, dev = data2train(in_file + "dev.jsonl")
    test_labels, test = data2train(in_file + "test.jsonl")
    with open(out_file + "train.json", "w") as f:
        json.dump(train, f)
    with open(out_file + "dev.json", "w") as f:
        json.dump(dev, f)
    with open(out_file + "test.json", "w") as f:
        json.dump(test, f)
    with open(out_file + "dev_labels.json", "w") as f:
        json.dump(dev_labels, f)
    with open(out_file + "test_labels.json", "w") as f:
        json.dump(test_labels, f)


if __name__ == "__main__":
    for dataset in ["fewrel"]:
        for m in [5, 10, 15]:
            for i in range(5):
                in_file = "../../ori_data/{0}/unseen_{1}_seed_{2}/".format(dataset, m, i)
                out_file = "../../data/{0}_m{1}_{2}/".format(dataset, m, i)
                os.makedirs(out_file, exist_ok=True)
                data_process(in_file, out_file)
