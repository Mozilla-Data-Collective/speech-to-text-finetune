import argparse
from functools import partial
from typing import Dict, Tuple
import evaluate
import os
import torch
from loguru import logger
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, 
                          Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, 
                          Trainer)

from transformers.models.wav2vec2.modeling_wav2vec2 import \
    WAV2VEC2_ADAPTER_SAFE_FILE
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from safetensors.torch import save_file as safe_save_file
from speech_to_text_finetune.config import load_config
from speech_to_text_finetune.data_process import (
    DataCollatorCTCWithPadding,
    load_dataset_from_dataset_id,
    load_subset_of_dataset, preprocess_for_mms, get_mms_dataset_prep_fn
)
from speech_to_text_finetune.utils import (
    get_hf_username,
    create_model_card,
    compute_wer_cer_metrics, 
    make_vocab
)


def load_mms_model_with_adapters(
        processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    """
    Loads and freezes the base 1b model, adds adapter layers, and makes them 
    trainable.

    Args:
      processor (Wav2Vec2Processor): a Wav2Vec2 processor object.
    Returns:
      Model updated with adapter layers.
    """
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    model.init_adapter_layers()
    model.freeze_base_model()

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

    return model


def run_finetuning(
    config_path: str = "config.yaml",
) -> Tuple[Dict, Dict]:
    """
    Complete pipeline for preprocessing the Common Voice dataset and then
    finetuning a Whisper model on it.

    Args:
        config_path (str): yaml filepath that follows the format defined in
                           config.py

    Returns:
        Tuple[Dict, Dict]: evaluation metrics from the baseline and the
                           finetuned models
    """
    cfg = load_config(config_path)

    language_id = cfg.language.lower().replace(" ", "-")

    if cfg.repo_name == "default":
        cfg.repo_name = f"{cfg.model_id.split('/')[1]}-{language_id}"
    local_output_dir = f"./artifacts/{cfg.repo_name}"
    if not os.path.exists(local_output_dir):
        os.mkdir(local_output_dir)

    logger.info(
        f"Finetuning starts soon, results saved locally at {local_output_dir}"
    )

    hf_repo_name = ""
    if cfg.training_hp.push_to_hub:
        hf_username = get_hf_username()
        hf_repo_name = f"{hf_username}/{cfg.repo_name}"
        logger.info(
            f"Results will also be uploaded in HF at {hf_repo_name}. "
            f"Private repo is set to {cfg.training_hp.hub_private_repo}."
        )

    device = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    logger.info(
        f"Loading {cfg.model_id} on {device} and configuring "
        f"it for {cfg.language}."
    )

    logger.info(f"Loading {cfg.dataset_id}. Language selected {cfg.language}")
    dataset, save_proc_dataset_dir = load_dataset_from_dataset_id(
        dataset_id=cfg.dataset_id,
    )
    dataset["train"] = load_subset_of_dataset(dataset["train"],
                                              cfg.n_train_samples)
    dataset["test"] = load_subset_of_dataset(dataset["test"],
                                             cfg.n_test_samples)
    logger.info("Processing dataset...")

    dataset = preprocess_for_mms(dataset)

    logger.info("Building new vocabulary from training data")
    make_vocab(dataset["train"], language_id, local_output_dir)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        local_output_dir, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|",
        target_lang=language_id
    )
    tokenizer.save_pretrained(local_output_dir)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, 
        tokenizer=tokenizer
    )
    prepare_dataset = get_mms_dataset_prep_fn(processor)
    for split in ("train", "test"):
        dataset[split] = dataset[split].map(prepare_dataset)

    model = load_mms_model_with_adapters(processor)
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    mms_hps = {
        hp: v for hp, v in cfg.training_hp.model_dump().items()
        if "generat" not in hp
    }
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        hub_model_id=hf_repo_name,
        report_to=["tensorboard"],
        **mms_hps
    )
    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_wer_cer_metrics,
            processor=processor,
            wer=wer,
            cer=cer,
            normalizer=BasicTextNormalizer(),
        ),
        processing_class=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    logger.info(
        f"Before finetuning, run evaluation on the baseline model "
        f"{cfg.model_id} to easily compare performance"
        f" before and after finetuning"
    )
    baseline_eval_results = trainer.evaluate()
    logger.info(
        f"Baseline evaluation complete. Results:\n\t {baseline_eval_results}"
    )

    logger.info(
        f"Start finetuning job on {dataset['train'].num_rows} audio samples. "
        f"Monitor training metrics in real time in a local tensorboard server "
        f"by running in a new terminal: "
        f"tensorboard --logdir {training_args.output_dir}/runs"
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Stopping the finetuning job prematurely...")
    else:
        logger.info("Finetuning job complete.")

    logger.info(
        f"Start evaluation on {dataset['test'].num_rows} audio samples."
    )
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation complete. Results:\n\t {eval_results}")

    adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(language_id)
    adapter_file = os.path.join(training_args.output_dir, adapter_file)

    safe_save_file(model._get_adapters(),
                   adapter_file,
                   metadata={"format": "pt"})

    model_card = create_model_card(
        model_id=cfg.model_id,
        dataset_id=cfg.dataset_id,
        language_id=language_id,
        language=cfg.language,
        n_train_samples=dataset["train"].num_rows,
        n_eval_samples=dataset["test"].num_rows,
        baseline_eval_results=baseline_eval_results,
        ft_eval_results=eval_results,
    )
    model_card.save(f"{local_output_dir}/README.md")

    if cfg.training_hp.push_to_hub:
        logger.info(
            f"Uploading model and eval results to HuggingFace: {hf_repo_name}"
        )
        try:
            trainer.push_to_hub()
        except Exception as e:
            logger.info(f"Did not manage to upload final model. See: \n{e}")
        model_card.push_to_hub(hf_repo_name)

    logger.info(
        f"Find your final, best performing model at {local_output_dir}"
    )
    return baseline_eval_results, eval_results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_to_config", "-c", 
                           default="example_data/config.yaml", 
                           help="Path to the experiment config yaml file")
    args = argparser.parse_args()
    run_finetuning(config_path=args.path_to_config)
