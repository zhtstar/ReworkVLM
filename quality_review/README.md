# ReworkVLM Review

Flask and SQLite interface for inspecting an image, its model-generated draft and initial opinion, and decisions from multiple reviewers. The history keeps each decision, author, time, corrected annotation and reason. An adjudicator can finalize a sample or escalate it for further review, and export the full history as JSONL.

## Run locally

From this folder, with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m flask --app app add-user adjudicator --role adjudicator
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and sign in with the account you created. The add-user command prompts for a password of at least five characters. Create additional accounts with `python -m flask --app app add-user reviewer1`. Follow the next section to populate the review queue.

## Load images and drafts

Place images under `images/` or set `REWORKVLM_IMAGE_ROOT`. Use one UTF-8 JSON object per line in the import file:

```json
{"id":"sample-001","image":"batch-a/001.png","dataset":"Local inspection","category":"assembly","draft_annotation":{"decision":"anomaly","action_type":"replace"},"draft_opinion":"The left fastener appears to be a different type."}
```

Use a stable `id` and an image path relative to the root. `dataset`, `category`, `draft_annotation` and `draft_opinion` are optional. Run `python -m flask --app app import-items drafts.jsonl` to load the drafts. Imports can be run again for identical records.

To load the final annotations as reference material, set the two dataset roots and use the separate import mode:

```bash
export REWORKVLM_LOCO_ROOT="$PWD/images/mvtec_loco_anomaly_detection"
export REWORKVLM_VID_ROOT="$PWD/images/VID-AD"
python -m flask --app app import-items ../ReworkVLM-Correct/annotations.jsonl --from-annotations
```

In PowerShell, use `$env:REWORKVLM_LOCO_ROOT = '...'` and `$env:REWORKVLM_VID_ROOT = '...'`. The final annotations appear under **Released reference annotation**. The app displays any input drafts supplied separately in the draft format. It starts with an empty human review history.

For a team installation, serve with `waitress-serve --listen=127.0.0.1:8080 --call app:create_app` behind an HTTPS reverse proxy. Set `REWORKVLM_HTTPS=1` and a shared `REWORKVLM_SECRET_KEY` when using HTTPS. Set `REWORKVLM_DB` to a persistent location; the default is `instance/reviews.sqlite`. Review history is visible to signed-in reviewers. Only adjudicators can finalize and export.
