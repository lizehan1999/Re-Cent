import glob
import json
import os
import torch
from .utils import log


def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, dev_labels, test_labels = None, None, None, None, None
    for p in paths:
        if "train.json" in p:
            with open(p, "r", encoding='utf-8') as f:
                train = json.load(f)
        elif "dev.json" in p:
            with open(p, "r", encoding='utf-8') as f:
                dev = json.load(f)
        elif "test.json" in p:
            with open(p, "r", encoding='utf-8') as f:
                test = json.load(f)
        elif "dev_labels.json" in p:
            with open(p, "r", encoding='utf-8') as f:
                dev_labels = json.load(f)
        elif "test_labels.json" in p:
            with open(p, "r", encoding='utf-8') as f:
                test_labels = json.load(f)

    return train, dev, test, dev_labels, test_labels


# create dataset
def create_dataset(path):
    train_dataset, dev_dataset, test_dataset, dev_labels, test_labels = open_content(path)
    dev_labels = [label.lower() for label in dev_labels]
    test_labels = [label.lower() for label in test_labels]
    return train_dataset, dev_dataset, test_dataset, dev_labels, test_labels


@torch.no_grad()
def get_for_one_path(path, model):
    # load the dataset
    _, dev_dataset, test_dataset, dev_labels, test_labels = create_dataset(path)
    data_name = path.split("/")[-1]  # get the name of the dataset
    flat_ner = True

    # evaluate the model
    dev_single_info, dev_multi_info, dev_sub_f1, dev_ob_f1 = model.evaluate(dev_dataset, flat_ner=flat_ner,
                                                                            threshold=model.decoder_threshold,
                                                                            batch_size=12,
                                                                            entity_types=dev_labels,
                                                                            gamma=model.ent2triplet_gamma)
    test_single_info, test_multi_info, test_sub_f1, test_ob_f1 = model.evaluate(test_dataset, flat_ner=flat_ner,
                                                                                threshold=model.decoder_threshold,
                                                                                batch_size=12,
                                                                                entity_types=test_labels,
                                                                                gamma=model.ent2triplet_gamma)

    print("dev_subject_entity_f1:", dev_sub_f1)
    print("dev_object_entity_f1:", dev_ob_f1)
    print("dev_single_info:", dev_single_info["precision"], dev_single_info["recall"], dev_single_info["f1"])
    print("dev_multi_info:", dev_multi_info["precision"], dev_multi_info["recall"], dev_multi_info["f1"])
    print("test_subject_entity_f1:", test_sub_f1)
    print("test_object_entity_f1:", test_ob_f1)
    print("test_single_info:", test_single_info["precision"], test_single_info["recall"], test_single_info["f1"])
    print("test_multi_info:", test_multi_info["precision"], test_multi_info["recall"], test_multi_info["f1"])
    return data_name, dev_single_info, dev_multi_info, test_single_info, test_multi_info, dev_sub_f1, dev_ob_f1, test_sub_f1, test_ob_f1


def get_for_all_path(model, epochs, steps, log_dir, data_path):
    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()
    data_name, dev_single_info, dev_multi_info, test_single_info, test_multi_info, dev_sub_f1, dev_ob_f1, test_sub_f1, test_ob_f1 = get_for_one_path(
        data_path, model)

    log(log_dir, '##############################################')
    log(log_dir, 'decoder_threshold: ' + str(model.decoder_threshold))
    log(log_dir, 'ent2triplet_gamma: ' + str(model.ent2triplet_gamma))
    log(log_dir, 'epochs: ' + str(epochs))
    log(log_dir, 'step: ' + str(steps))
    log(log_dir, data_name)
    log(log_dir, 'dev_entity_info:')
    log(log_dir, '    subject entity f1: ' + str(dev_sub_f1))
    log(log_dir, '    object entity f1: ' + str(dev_ob_f1))
    log(log_dir, 'dev_single_info:')
    log(log_dir, '    precision: ' + dev_single_info['precision'])
    log(log_dir, '    recall: ' + dev_single_info['recall'])
    log(log_dir, '    f1: ' + dev_single_info['f1'])
    log(log_dir, 'dev_multi_info:')
    log(log_dir, '    precision: ' + dev_multi_info['precision'])
    log(log_dir, '    recall: ' + dev_multi_info['recall'])
    log(log_dir, '    f1: ' + dev_multi_info['f1'])
    log(log_dir, 'test_entity_info:')
    log(log_dir, '    subject entity f1: ' + str(test_sub_f1))
    log(log_dir, '    object entity f1: ' + str(test_ob_f1))
    log(log_dir, 'test_single_info:')
    log(log_dir, '    precision: ' + test_single_info['precision'])
    log(log_dir, '    recall: ' + test_single_info['recall'])
    log(log_dir, '    f1: ' + test_single_info['f1'])
    log(log_dir, 'test_multi_info:')
    log(log_dir, '    precision: ' + test_multi_info['precision'])
    log(log_dir, '    recall: ' + test_multi_info['recall'])
    log(log_dir, '    f1: ' + test_multi_info['f1'])

    entity_info = [dev_sub_f1, dev_ob_f1, test_sub_f1, test_ob_f1]

    return dev_single_info, dev_multi_info, test_single_info, test_multi_info, entity_info
