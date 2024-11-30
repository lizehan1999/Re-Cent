# Re-Cent

This repository contains the official implementation for the paper: **Re-Cent: A Relation-Centric Framework for Joint Zero-Shot Relation Triplet Extraction.** The paper has been accepted to appear at **COLING 2025**.

## Requirements

The main requirements are:

- python==3.9.7
- pytorch==1.13.1
- transformers==4.44.2
- numpy==1.22.3

## Run

1. Download the split datasets and place them in the `/ori_data` folder.
2. Run `/reCent/data_processing/data_processing.py`.
3. Download the pretrained [DeBERTa-v3 weights](https://huggingface.co/microsoft/deberta-v3-base) and place them in the `./BERT_MODELS/deberta-v3-base` folder.
4. Set the hyperparameters in `config.yaml` and run `train.py`

## Cite

```
-
```



## Acknowledgement

The framework of Re-Cent is based on [GLiNER](https://github.com/urchade/GLiNER). Their contributions have greatly helped in the development and implementation of our code. We appreciate their efforts and the open-source community for fostering collaboration and knowledge sharing.
