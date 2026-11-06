# Container OCR

A multi-user Flask web app that reads shipping-container codes from photos. Users upload an image, a YOLOv8 model finds the container code, an OCR pipeline reads it, and the result is checked against the ISO 6346 standard before it is saved. I built it during a six-week software development internship at Vehant Technologies (Jun–Jul 2026), and it was deployed on the company's GPU server.

**Result:** on a 786-image benchmark, end-to-end accuracy went from 61.2% to 67.5%.

## What it does

- **Detection:** YOLOv8s finds the container-code region in each uploaded image.
- **Recognition:** the cropped region goes through a tiered OCR pipeline (`app/inference/ocr.py`). It preprocesses the crop, tries several rotations, merges fragments and corrects commonly confused characters. A fine-tuned TrOCR model was evaluated alongside it.
- **Validation:** every code is checked with the ISO 6346 check-digit algorithm (4 letters, 6 digits, 1 check digit), and low-confidence reads are flagged for review.
- **Review workflow:** users can correct a code in the browser, and every edit is written to an audit log.
- **Accounts and roles:** login and signup with bcrypt-hashed passwords and admin approval of new users. An admin dashboard covers user management, password resets and per-user activity.
- **Export:** results can be downloaded per upload, and past exports are kept as a history.

## Stack

Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt and Flask-Migrate (Alembic), served with Gunicorn. The frontend is plain HTML, CSS and vanilla JavaScript. The database is SQLite in development and PostgreSQL in production. ML: YOLOv8 (Ultralytics), PaddleOCR/PaddleX, Hugging Face TrOCR and OpenCV.

## Security

Before the code was open-sourced, I audited it and fixed 28 issues, including 3 critical ones (IDOR, open redirect and XSS).

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
cp .env.example .env          # USE_STUB_MODEL=true runs the app without model weights
flask db upgrade
flask create-user --admin     # create the first admin account
python run.py                 # http://localhost:5000
```

The trained model weights are not included in this repository. With `USE_STUB_MODEL=true`, the full upload, review and export flow works using a stub predictor.

## Repository layout

- `app/`: the Flask application (`auth`, `admin`, `main`, `inference`, `models`)
- `migrations/`: database migrations
- `scripts/`: dataset preparation, fine-tuning and evaluation tools
- `benchmark*.py` and `*.csv`: the benchmark harness and its recorded runs
- Top-level `scratch_*` scripts and sample folders are research and diagnostic work from the internship, kept for reference.
