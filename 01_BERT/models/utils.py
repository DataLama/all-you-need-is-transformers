import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from transformers import EvalPrediction


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    
    def ap_nlp_custom_metrics(preds, labels):
        acc = (preds == labels).mean()
        tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
        tpr = tp/(tp+fn) # sensitivity, recall
        tnr = tn/(tn+fp) # specificity
        f1 = f1_score(y_true=labels, y_pred=preds)
        mcc = matthews_corrcoef(y_true=labels, y_pred=preds)
        return {
            "mcc":mcc,
            "tpr": tpr,
            "tnr": tnr,
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
    
    return ap_nlp_custom_metrics(preds, p.label_ids)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )