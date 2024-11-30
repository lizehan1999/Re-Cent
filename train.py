import json
import os
import random
import torch
from tqdm import tqdm
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
from transformers import AutoTokenizer

from reCent import ReCent, ReCentConfig
from reCent.data_processing import SpanProcessor
from reCent.data_processing.tokenizer import WordsSplitter
from reCent.data_processing.collator import DataCollator
from reCent.evaluation import get_for_all_path, log
import argparse
import yaml

os.unsetenv('LD_LIBRARY_PATH')
seed = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


set_seed(seed)


def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


class Trainer:
    def __init__(self, config, device='cuda', seed=0):
        self.config = config
        self.lr_encoder = float(self.config.lr_encoder)
        self.lr_others = float(self.config.lr_others)
        self.weight_decay_encoder = float(self.config.weight_decay_encoder)
        self.weight_decay_other = float(self.config.weight_decay_other)

        self.device = device
        self.seed = seed

        self.model_config = ReCentConfig(
            model_name=config.model_name,
            max_width=config.max_width,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            loss_alpha=config.loss_alpha,
            loss_gamma=config.loss_gamma,
            max_types=config.max_types,
            shuffle_types=config.shuffle_types,
            random_drop=config.random_drop,
            max_neg_type_ratio=config.max_neg_type_ratio,
            max_len=config.max_len,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model_config.class_token_index = len(tokenizer)
        tokenizer.add_tokens([self.model_config.relation_token, self.model_config.sep_token])
        self.model_config.vocab_size = len(tokenizer)

        words_splitter = WordsSplitter()

        self.data_processor = SpanProcessor(self.model_config, tokenizer, words_splitter, preprocess_text=True)

        self.optimizer = None

        self.model_config.loss_beta = config.loss_beta
        self.model_config.so_neg_num = config.so_neg_num

        self.best_dev_single_info = None
        self.best_dev_multi_info = None
        self.best_test_single_info = None
        self.best_test_multi_info = None
        self.best_entity_info = None

    def create_optimizer(self, opt_model, **optimizer_kwargs):
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        encoder_parameters = [name for name, _ in opt_model.named_parameters() if "token_rep_layer" in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if
                    (n in decay_parameters and n not in encoder_parameters and p.requires_grad)
                ],
                "weight_decay": self.weight_decay_other,
                "lr": self.lr_others,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if
                    (n not in decay_parameters and n not in encoder_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.lr_others,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if
                    (n in decay_parameters and n in encoder_parameters and p.requires_grad)
                ],
                "weight_decay": self.weight_decay_encoder,
                "lr": self.lr_encoder,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if
                    (n not in decay_parameters and n in encoder_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.lr_encoder,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def setup_model_and_optimizer(self, device=None):
        if device is None:
            device = self.device
        model = ReCent(self.model_config, data_processor=self.data_processor).to(device)
        model.resize_token_embeddings([self.model_config.relation_token, self.model_config.sep_token],
                                      set_class_token_index=False,
                                      add_tokens_to_tokenizer=False)
        optimizer = self.create_optimizer(model.model)

        model.decoder_threshold = self.config.decoder_threshold
        model.ent2triplet_gamma = self.config.ent2triplet_gamma

        return model, optimizer

    def create_dataloader(self, dataset, sampler=None, shuffle=True):
        collator = DataCollator(self.config, data_processor=self.data_processor, prepare_labels=True)
        data_loader = DataLoader(dataset, batch_size=self.config.train_batch_size, num_workers=12,
                                 worker_init_fn=lambda k: set_seed(seed + k),
                                 shuffle=shuffle, collate_fn=collator, sampler=sampler)
        return data_loader

    def init_scheduler(self, optimizer, num_warmup_steps, num_steps):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
        return scheduler

    def epochs_train(self, model, optimizer, train_loader, val_data_dir, num_epochs, device='cuda', rank=None):
        model.train()

        warmup_ratio = self.config.warmup_ratio
        log_dir = self.config.log_dir

        num_warmup_steps = int(num_epochs * len(train_loader) * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)

        scheduler = self.init_scheduler(optimizer, num_warmup_steps,
                                        num_epochs * len(train_loader))
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(num_epochs):
            pbar = tqdm(range(len(train_loader)))
            for i, x in zip(pbar, train_loader):
                optimizer.zero_grad()

                for k, v in x.items():
                    if isinstance(v, torch.Tensor):
                        x[k] = v.to(device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = model(alpha=self.config.loss_alpha,
                                 gamma=self.config.loss_gamma,
                                 **x).loss

                if torch.isnan(loss).any():
                    print("Warning: NaN loss detected")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                del x
                torch.cuda.empty_cache()

                description = f"epoch: {epoch} | step: {i} | loss: {loss.item():.2f}"
                pbar.set_description(description)
            if rank is None or rank == 0:
                if val_data_dir != "none":
                    dev_single_info, dev_multi_info, test_single_info, test_multi_info, entity_info = get_for_all_path(
                        model, epoch, (epoch + 1) * len(train_loader), log_dir, val_data_dir)
                    if self.best_dev_single_info is None or float(self.best_dev_single_info['f1']) < float(
                            dev_single_info['f1']):
                        self.best_dev_single_info = dev_single_info
                        self.best_test_single_info = test_single_info
                    if self.best_dev_multi_info is None or float(self.best_dev_multi_info['f1']) < float(
                            dev_multi_info['f1']):
                        self.best_dev_multi_info = dev_multi_info
                        self.best_test_multi_info = test_multi_info
                        self.best_entity_info = entity_info

                model.train()

    def run(self):
        data_dir = self.config.data_dir + self.config.dataset + "_" + self.config.rel_num + "_" + str(self.seed)
        with open(data_dir + "/train.json", 'r') as f:
            data = json.load(f)
        random.shuffle(data)

        model, optimizer = self.setup_model_and_optimizer()

        train_loader = self.create_dataloader(data, shuffle=True)

        self.epochs_train(model, optimizer, train_loader, data_dir, num_epochs=self.config.num_epochs,
                          device=self.device)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    config = load_config_as_namespace(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device

    log(config.log_dir, '====================START====================')
    for k, v in vars(config).items():
        log(config.log_dir, str(k) + ': ' + str(v))

    avg_best_dev_single_info = dict(
        precision=0,
        recall=0,
        f1=0,
    )
    avg_best_dev_multi_info = dict(
        precision=0,
        recall=0,
        f1=0,
    )
    avg_best_test_single_info = dict(
        precision=0,
        recall=0,
        f1=0,
    )
    avg_best_test_multi_info = dict(
        precision=0,
        recall=0,
        f1=0,
    )
    avg_best_entity_info = dict(
        dev_sub=0,
        dev_ob=0,
        test_sub=0,
        test_ob=0,
    )
    for s in range(config.seed_num):
        log(config.log_dir, '--------------------start--------------------')
        log(config.log_dir, 'seed_num: ' + str(s))

        trainer = Trainer(config,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          seed=s)
        trainer.run()

        avg_best_dev_single_info['precision'] += float(trainer.best_dev_single_info['precision'])
        avg_best_dev_single_info['recall'] += float(trainer.best_dev_single_info['recall'])
        avg_best_dev_single_info['f1'] += float(trainer.best_dev_single_info['f1'])

        avg_best_dev_multi_info['precision'] += float(trainer.best_dev_multi_info['precision'])
        avg_best_dev_multi_info['recall'] += float(trainer.best_dev_multi_info['recall'])
        avg_best_dev_multi_info['f1'] += float(trainer.best_dev_multi_info['f1'])

        avg_best_test_single_info['precision'] += float(trainer.best_test_single_info['precision'])
        avg_best_test_single_info['recall'] += float(trainer.best_test_single_info['recall'])
        avg_best_test_single_info['f1'] += float(trainer.best_test_single_info['f1'])

        avg_best_test_multi_info['precision'] += float(trainer.best_test_multi_info['precision'])
        avg_best_test_multi_info['recall'] += float(trainer.best_test_multi_info['recall'])
        avg_best_test_multi_info['f1'] += float(trainer.best_test_multi_info['f1'])

        avg_best_entity_info["dev_sub"] += float(trainer.best_entity_info[0])
        avg_best_entity_info["dev_ob"] += float(trainer.best_entity_info[1])
        avg_best_entity_info["test_sub"] += float(trainer.best_entity_info[2])
        avg_best_entity_info["test_ob"] += float(trainer.best_entity_info[3])

        log(config.log_dir, '---------------------end---------------------\n\n')

    avg_best_dev_single_info['precision'] /= config.seed_num
    avg_best_dev_single_info['recall'] /= config.seed_num
    avg_best_dev_single_info['f1'] /= config.seed_num

    avg_best_dev_multi_info['precision'] /= config.seed_num
    avg_best_dev_multi_info['recall'] /= config.seed_num
    avg_best_dev_multi_info['f1'] /= config.seed_num

    avg_best_test_single_info['precision'] /= config.seed_num
    avg_best_test_single_info['recall'] /= config.seed_num
    avg_best_test_single_info['f1'] /= config.seed_num

    avg_best_test_multi_info['precision'] /= config.seed_num
    avg_best_test_multi_info['recall'] /= config.seed_num
    avg_best_test_multi_info['f1'] /= config.seed_num

    avg_best_entity_info["dev_sub"] /= config.seed_num
    avg_best_entity_info["dev_ob"] /= config.seed_num
    avg_best_entity_info["test_sub"] /= config.seed_num
    avg_best_entity_info["test_ob"] /= config.seed_num

    log(config.log_dir, 'average_best_dev_entity_info:')
    log(config.log_dir, '    sub_f1: ' + str(avg_best_entity_info['dev_sub']))
    log(config.log_dir, '    ob_f1: ' + str(avg_best_entity_info['dev_ob']))

    log(config.log_dir, 'average_best_dev_single_info:')
    log(config.log_dir, '    precision: ' + str(avg_best_dev_single_info['precision']))
    log(config.log_dir, '    recall: ' + str(avg_best_dev_single_info['recall']))
    log(config.log_dir, '    f1: ' + str(avg_best_dev_single_info['f1']))

    log(config.log_dir, 'average_best_dev_multi_info:')
    log(config.log_dir, '    precision: ' + str(avg_best_dev_multi_info['precision']))
    log(config.log_dir, '    recall: ' + str(avg_best_dev_multi_info['recall']))
    log(config.log_dir, '    f1: ' + str(avg_best_dev_multi_info['f1']))

    log(config.log_dir, 'average_best_test_entity_info:')
    log(config.log_dir, '    sub_f1: ' + str(avg_best_entity_info['test_sub']))
    log(config.log_dir, '    ob_f1: ' + str(avg_best_entity_info['test_ob']))

    log(config.log_dir, 'average_best_test_single_info:')
    log(config.log_dir, '    precision: ' + str(avg_best_test_single_info['precision']))
    log(config.log_dir, '    recall: ' + str(avg_best_test_single_info['recall']))
    log(config.log_dir, '    f1: ' + str(avg_best_test_single_info['f1']))

    log(config.log_dir, 'average_best_test_multi_info:')
    log(config.log_dir, '    precision: ' + str(avg_best_test_multi_info['precision']))
    log(config.log_dir, '    recall: ' + str(avg_best_test_multi_info['recall']))
    log(config.log_dir, '    f1: ' + str(avg_best_test_multi_info['f1']))

    log(config.log_dir, '=====================END=====================\n\n\n\n')
