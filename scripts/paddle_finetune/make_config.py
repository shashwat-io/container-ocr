"""
Step 4 -- PaddleX fine-tuning config generator.

Writes a PP-OCRv6_medium_rec training config in the exact schema PaddleX
3.7.2 expects, verified directly against the installed package's own
template (site-packages/paddlex/configs/modules/text_recognition/
PP-OCRv6_medium_rec.yaml) rather than guessed from docs, since formats can
shift between PaddleX versions (per the spec).

Base model is PP-OCRv6_medium_rec -- confirmed against AGENTS.md as the
model already used for inference (app/inference/ocr.py's
paddlex.create_pipeline('OCR', ...) resolves to PP-OCRv6_medium under the
hood; AGENTS.md's 02-07-2026 changelog entry confirms this explicitly), so
fine-tuning starts from the same model tier that's actually deployed.
"""
import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'dataset')
DEFAULT_OUTPUT_CONFIG = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'pp_ocrv6_medium_rec_finetune.yaml')

# Kept in one string (not a Python dict dumped via yaml.safe_dump) so the key
# order and comments match PaddleX's own template file exactly -- this is
# meant to be diffable against
# site-packages/paddlex/configs/modules/text_recognition/PP-OCRv6_medium_rec.yaml
CONFIG_TEMPLATE = """\
Global:
  model: PP-OCRv6_medium_rec
  mode: train # check_dataset/train/evaluate/predict
  dataset_dir: "{dataset_dir}"
  device: {device}
  output: "{output_dir}"

CheckDataset:
  convert:
    enable: False
    src_dataset_type: null
  split:
    enable: False
    train_percent: null
    val_percent: null

Train:
  epochs_iters: {epochs}
  batch_size: {batch_size}
  learning_rate: {learning_rate}
  pretrain_weight_path: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
  resume_path: null
  log_interval: 1
  eval_interval: 1
  save_interval: 1

Evaluate:
  weight_path: "{output_dir}/best_accuracy/best_accuracy.pdparams"
  log_interval: 1

Export:
  weight_path: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams

Predict:
  batch_size: 1
  model_dir: "{output_dir}/best_accuracy/inference"
  input: "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
  kernel_option:
    run_mode: paddle
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    ap.add_argument('--output-dir', default=os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'training_output'))
    ap.add_argument('--output-config', default=DEFAULT_OUTPUT_CONFIG)
    ap.add_argument('--device', default='cpu',
                     help="'cpu' for smoke test / local dev, 'gpu:0' (or 'gpu:0,1' etc.) on Kaggle.")
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=0.001)
    args = ap.parse_args()

    for required in ('train.txt', 'val.txt', 'dict.txt'):
        p = os.path.join(args.dataset_dir, required)
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: {p} not found -- run generate_labels.py first.")

    config_text = CONFIG_TEMPLATE.format(
        dataset_dir=args.dataset_dir,
        device=args.device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    output_config_dir = os.path.dirname(args.output_config)
    if output_config_dir:
        os.makedirs(output_config_dir, exist_ok=True)
    with open(args.output_config, 'w') as f:
        f.write(config_text)
    print(f"Wrote config to {args.output_config}")


if __name__ == '__main__':
    main()
