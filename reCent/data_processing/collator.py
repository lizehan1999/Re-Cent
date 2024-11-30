from .processor import SpanProcessor

class DataCollator:
    def __init__(self, config, tokenizer=None, words_splitter=None, data_processor=None,
                 return_tokens: bool = False,
                 return_id_to_classes: bool = False,
                 return_entities: bool = False,
                 prepare_labels: bool = False,
                 entity_types=None):
        self.config = config
        if data_processor is None:
            self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
        else:
            self.data_processor = data_processor
        self.prepare_labels = prepare_labels
        self.return_tokens = return_tokens
        self.return_id_to_classes = return_id_to_classes
        self.return_entities = return_entities
        self.entity_types = entity_types

    def __call__(self, input_x):
        raw_batch = self.data_processor.collate_raw_batch(input_x, entity_types=self.entity_types)

        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=self.prepare_labels)

        model_input.update({"span_idx": raw_batch['span_idx'] if 'span_idx' in raw_batch else None,
                            "span_mask_sub": raw_batch["span_mask_sub"] if 'span_mask_sub' in raw_batch else None,
                            "span_mask_ob": raw_batch["span_mask_ob"] if 'span_mask_ob' in raw_batch else None,
                            "text_lengths": raw_batch['seq_length'],
                            "sub_ob_pair": raw_batch['sub_ob_pair'],
                            })
        if self.return_tokens:
            model_input['tokens'] = raw_batch['tokens']
        if self.return_id_to_classes:
            model_input['id_to_classes'] = raw_batch['id_to_classes']
        if self.return_entities:
            model_input['entities_sub'] = raw_batch['entities_sub']
            model_input['entities_ob'] = raw_batch['entities_ob']

        return model_input
