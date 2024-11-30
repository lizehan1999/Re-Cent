from typing import Optional, Union

import torch
from torch import nn

from .modeling.base import SpanModel
from .data_processing import SpanProcessor
from .data_processing.collator import DataCollator
from .decoding import SpanDecoder
from .evaluation import Evaluator, evaluate_single_multi, score
from .config import ReCentConfig

from huggingface_hub import PyTorchModelHubMixin


class ReCent(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ReCentConfig,
                 data_processor: Optional[Union[SpanProcessor]] = None,
                 encoder_from_pretrained: bool = True):

        super().__init__()
        self.config = config

        self.model = SpanModel(config, encoder_from_pretrained)
        self.data_processor = data_processor
        self.decoder = SpanDecoder(config)

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output

    @property
    def device(self):
        device = next(self.model.parameters()).device
        return device

    def resize_token_embeddings(self, add_tokens,
                                set_class_token_index=True,
                                add_tokens_to_tokenizer=True,
                                pad_to_multiple_of=None) -> nn.Embedding:
        if set_class_token_index:
            self.config.class_token_index = len(self.data_processor.transformer_tokenizer) + 1
        if add_tokens_to_tokenizer:
            self.data_processor.transformer_tokenizer.add_tokens(add_tokens)
        new_num_tokens = len(self.data_processor.transformer_tokenizer)
        model_embeds = self.model.token_rep_layer.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.vocab_size = model_embeds.num_embeddings
        if self.config.encoder_config is not None:
            self.config.encoder_config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def evaluate(self, test_data, flat_ner=False, multi_label=True, threshold=0.2, batch_size=12, entity_types=None,
                 gamma=1.3):
        self.eval()
        dataset = test_data
        collator = DataCollator(self.config, data_processor=self.data_processor,
                                return_tokens=True,
                                return_entities=True,
                                return_id_to_classes=True,
                                prepare_labels=False,
                                entity_types=entity_types)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

        device = self.device
        all_preds_sub, all_preds_ob = [], []
        all_trues_sub, all_trues_ob = [], []
        single_preds, single_golds, multi_preds, multi_golds = [], [], [], []

        # Iterate over data batches
        for batch in data_loader:
            # Move the batch to the appropriate device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Perform predictions
            outputs = self.model(**batch)
            model_output = outputs.logits
            [model_output_rep_sub, model_output_rep_ob] = outputs.span_rep_so
            prompts_embedding = outputs.prompts_embedding

            if not isinstance(model_output[0], torch.Tensor):
                model_output[0] = torch.from_numpy(model_output[0])
            if not isinstance(model_output[1], torch.Tensor):
                model_output[1] = torch.from_numpy(model_output[1])

            decoded_outputs_sub, rep_selected_sub = self.decoder.decode(
                batch['tokens'], batch['id_to_classes'],
                model_output[0], model_output_rep_sub, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
            )
            decoded_outputs_ob, rep_selected_ob = self.decoder.decode(
                batch['tokens'], batch['id_to_classes'],
                model_output[1], model_output_rep_ob, flat_ner=flat_ner, threshold=threshold, multi_label=multi_label
            )
            all_preds_sub.extend(decoded_outputs_sub)
            all_preds_ob.extend(decoded_outputs_ob)
            all_trues_sub.extend(batch["entities_sub"])
            all_trues_ob.extend(batch["entities_ob"])

            classes_to_id = {v: k for k, v in batch['id_to_classes'].items()}

            single_pred, single_gold, multi_pred, multi_gold = evaluate_single_multi(batch["entities_sub"],
                                                                                     decoded_outputs_sub,
                                                                                     batch["entities_ob"],
                                                                                     decoded_outputs_ob,
                                                                                     rep_selected_sub,
                                                                                     rep_selected_ob, self.model,
                                                                                     prompts_embedding,
                                                                                     classes_to_id, gamma)
            single_preds.extend(single_pred)
            single_golds.extend(single_gold)
            multi_preds.extend(multi_pred)
            multi_golds.extend(multi_gold)

        evaluator_sub = Evaluator(all_trues_sub, all_preds_sub)
        evaluator_ob = Evaluator(all_trues_ob, all_preds_ob)

        sub_out, sub_f1 = evaluator_sub.evaluate()
        ob_out, ob_f1 = evaluator_ob.evaluate()

        single_info = score(single_preds, single_golds)
        multi_info = score(multi_preds, multi_golds)

        return single_info, multi_info, sub_f1, ob_f1
