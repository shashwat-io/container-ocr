"""
Step 4 (invocation) / Step 5 (smoke test) -- run PaddleX training from a
config produced by make_config.py.

The `paddlex` CLI as installed (3.7.2) only exposes pipeline predict/serve/
install subcommands, not a training subcommand -- confirmed by reading
paddlex/paddlex_cli.py directly rather than assuming a `-c config.yaml`
CLI pattern from older docs. The actual training entry point in this
version is the Python API below (paddlex.build_dataset_checker /
paddlex.build_trainer), confirmed by reading paddlex/__init__.py and the
BaseTrainer/BaseDatasetChecker classes directly.

Model registration is NOT automatic in this installed version:
paddlex.build_trainer()/build_model() look up model metadata (config path,
supported APIs) in a registry populated by *importing*
paddlex.repo_apis.<Repo>_api (e.g. PaddleOCR_api) -- that import runs
register_model_info() as a side effect. Nothing in the build_trainer/
build_model call path imports it automatically (traced build_trainer's
source directly: it's a bare `try: pass except ModuleNotFoundError: pass`,
i.e. a no-op in this version) -- so without an explicit import here,
build_trainer raises `UnsupportedParamError: '<model>' is not a registered
model name` for EVERY model, regardless of device or dataset. Confirmed by
reproducing: get_registered_model_info('PP-OCRv6_medium_rec') raises
KeyError before importing paddlex.repo_apis.PaddleOCR_api, and returns the
correct config entry immediately after.

Usage:
    python3 scripts/paddle_finetune/train.py --config data/paddle_finetune/pp_ocrv6_medium_rec_finetune.yaml

--train-ratio (default 1.0): PaddleX's own PP-OCRv6_medium_rec.yaml template
hardcodes Train.dataset.ratio_list: [0.5] -- every "epoch" trains on a fresh
random 50% subsample of train.txt, re-drawn each epoch, not a full pass. Not
exposed anywhere in make_config.py's high-level schema (ratio_list has no
CONFIG_TEMPLATE field), so it silently stayed at the vendor default for every
run so far, including the checkpoint currently in
paddle_backup_inspect/checkpoints_backup/. Overridden here to 1.0 (full pass
per epoch) since 1,668 train rows is already small -- halving it further per
epoch has no evident upside. Confirmed safe to set via trainer.pdx_config
directly, since paddlex.modules.text_recognition.trainer.py's own
update_config() (called at the top of trainer.train()) never touches
ratio_list -- so this override survives it, whichever order they run in.
"""
import argparse

import paddlex
import paddlex.repo_apis.PaddleOCR_api  # noqa: F401 -- import side effect: registers PP-OCRv6_medium_rec etc.
from paddlex.utils.config import parse_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--skip-dataset-check', action='store_true',
                     help='Skip CheckDataset (it validates train.txt/val.txt/dict.txt format and image existence).')
    ap.add_argument('--train-ratio', type=float, default=1.0,
                     help="Fraction of train.txt sampled per epoch (PaddleX/PaddleOCR's own "
                          "Train.dataset.ratio_list). Vendor default is 0.5 (resampled fresh "
                          "each epoch); default here is 1.0, a full pass every epoch.")
    args = ap.parse_args()

    config = parse_config(args.config)

    if not args.skip_dataset_check:
        print("=== CheckDataset ===")
        checker = paddlex.build_dataset_checker(config)
        meta = checker.check()
        print(meta)

    print("=== Train ===")
    trainer = paddlex.build_trainer(config)
    trainer.pdx_config.update({"Train.dataset.ratio_list": [args.train_ratio]})
    print(f"Set Train.dataset.ratio_list = [{args.train_ratio}] "
          f"(vendor default is [0.5] -- see module docstring above)")
    trainer.train()


if __name__ == '__main__':
    main()
