# PaddleOCR/PaddleX fine-tuning pipeline

Rebuilt from `ocr_pipeline_rebuild_spec.md` (the original `ocr_pipeline.py` was
lost). Run in order:

```
python3 scripts/paddle_finetune/extract_crops.py [--drive-labels-csv PATH]
python3 scripts/paddle_finetune/generate_labels.py
python3 scripts/paddle_finetune/make_config.py --device cpu --epochs 2 --batch-size 4   # smoke test config
python3 scripts/paddle_finetune/train.py --config data/paddle_finetune/pp_ocrv6_medium_rec_finetune.yaml
```

For the real Kaggle run, regenerate the config with `--device gpu:0` (or
`gpu:0,1` for T4x2) and real `--epochs`/`--batch-size`.

## Current state (as of this rebuild)

- **35 of 377** `manual_labels.csv` rows survive the label-quality + leakage
  filters (30 train / 5 val after the video-level split). This is small --
  expected, given the conservative leakage policy below, not a bug.
- Step 1-3 verified end-to-end locally against the real dataset and the real
  installed `paddlex==3.7.2` dataset checker (`CheckDataset` passes: 30
  train / 5 val samples, format confirmed valid).
- Step 4's config is generated and confirmed schema-correct against the
  installed package's own template.
- **Step 5 (smoke test): PASSED**, actually executed locally (CPU) end to
  end -- not just written. Training ran through 6+ epochs on the 30-crop
  train set, loss dropped sanely (CTCLoss+NRTRLoss combined: ~117-122 at
  step 1-2 down to ~73-74 by step 16-18, never NaN/flat), eval ran and
  produced real metrics, and `best_accuracy/best_accuracy.pdparams` saved
  and was confirmed to reload (441 parameter tensors). Log:
  `data/paddle_finetune/training_output_local/train.log`.

  Getting to a working local training environment required fixing four
  separate, unrelated environment bugs (none in our pipeline code) -- listed
  here since Kaggle may hit some of the same ones:
  1. `paddlex.build_trainer()` doesn't auto-register model metadata in this
     paddlex version; needed an explicit
     `import paddlex.repo_apis.PaddleOCR_api` before calling it (now in
     `train.py`).
  2. `paddlex --install PaddleOCR` crashed with an internal pip resolver
     `AssertionError` (pip 22.0.2 bug). Fixed by patching
     `site-packages/paddlex/utils/install.py` to add
     `--use-deprecated=legacy-resolver` to its internal pip call. This is a
     change to the installed package, not this repo -- redo it if the
     Kaggle environment hits the same crash (`pip --version` there will
     tell you if it's likely: older pip = more likely).
  3. `repo_apis/PaddleOCR_api/configs/` (where paddlex looks up each
     model's native architecture YAML) ships **empty in the paddlex 3.7.2
     wheel for every repo, not just PaddleOCR** -- confirmed no repo_api
     submodule has this directory. Worked around by symlinking the real
     file from the cloned repo:
     `ln -s .../repo_manager/repos/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml
        .../repo_apis/PaddleOCR_api/configs/PP-OCRv6_medium_rec.yaml`
  4. Installing PaddleOCR's own `requirements.txt` silently downgraded
     `Pillow` to 9.0.1 (breaks the pinned `Pillow==10.0.0` this project's
     Flask app relies on) and, separately, having both `opencv-python` and
     `opencv-contrib-python` installed side by side corrupted the shared
     `cv2` namespace (`cv2.__file__` became `None`, missing `CV_8U` and
     `imshow` -- this also broke `ultralytics`/YOLO import, i.e. the live
     app, until fixed). Fixed by re-pinning Pillow back to 10.0.0 and
     force-reinstalling a single `opencv-contrib-python` build. If
     re-running any PaddleOCR dependency install, re-check both afterward:
     `python3 -c "import PIL; print(PIL.__version__)"` (want `10.0.0`) and
     `python3 -c "import cv2; print(cv2.__file__, cv2.CV_8U)"` (want a real
     path, not `None`).

  For the real Kaggle run: regenerate the config with `--device gpu:0` (or
  `gpu:0,1` for T4x2) and real `--epochs`/`--batch-size` (the smoke test
  used the CPU config's defaults, 20 epochs/batch_size 8, which is more
  than a true smoke test needs but was left running since it was already
  producing valid results). Given only 30 usable train crops, this smoke
  run and "the real run" are nearly the same size -- there isn't much
  headroom for a separate larger full-scale run without loosening the
  leakage policy below or labeling more crops.

## Key findings from this rebuild (read before changing the filters)

1. **Filenames alone can't detect test-split leakage.** Images in this
   dataset are video frames (`<video_id>_<frame_offset>_jpg.rf.<roboflow_
   hash>.jpg`); Roboflow assigns a fresh hash per export, so the same exact
   frame gets a different filename in each split. Comparing raw filenames
   found 0 overlaps between `manual_labels.csv` and the test set --
   comparing `video_id` (the part before the first `_`) found 119/377 rows
   are frame-identical duplicates of a test-set image, and 222/377 more
   share a source video with a test-set image at a different offset. The
   pipeline matches on `video_id`, not filename.
2. **User's explicit choice**: exclude every row whose `video_id` appears
   *anywhere* in the test split (not just exact-duplicate frames). This is
   the conservative option -- it's why only 35/377 rows survive. See
   `EXCLUDE_WHOLE_SOURCE_VIDEO` in `extract_crops.py` if this needs
   revisiting later; loosening it should get the same explicit sign-off this
   did, not a casual flag flip.
3. **The train/valid split for fine-tuning also groups by `video_id`**, not
   just by exact file path -- caught one real case
   (`z4944210769393_e588d7aaa69420580cf57da233ec59b2`) where the identical
   photo was exported twice under two different Roboflow hashes; a per-file
   split had put one copy in train and its duplicate in val.
4. **Drive freshness check** (`extract_crops.py --drive-labels-csv PATH`)
   only actually verifies anything if you pass a real synced/mounted Drive
   path; without it, the script loudly warns rather than silently trusting
   the local copy.
5. **Crop logic assumption, not yet confirmed**: `Labelling.ipynb` (where
   the ground-truth `label_text` was actually typed against a rendered crop)
   only exists in Kaggle/Colab sessions -- it wasn't available in this repo
   to diff against. `crop_utils.py` instead reuses
   `app/inference/ocr.py:crop_detection` (padding=10) so training crops are
   at least consistent with what the deployed model sees at inference. If
   the notebook used different padding, a training crop could show more/less
   border than what the labeler actually verified against. Confirm against
   the real notebook before trusting this at scale.
6. **Base model**: PP-OCRv6_medium_rec, matching the model already deployed
   in `app/inference/ocr.py` (per AGENTS.md).
7. **Pinned versions** (added to `requirements.txt`, matching what's
   actually installed and confirmed working for dataset-check locally):
   `paddlepaddle==3.3.1`, `paddlex==3.7.2`, `paddleocr==3.7.0`.
