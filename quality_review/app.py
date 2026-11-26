"""Local annotation review with an append-only SQLite history."""

import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sqlite3

import click
from flask import Flask, Response, abort, flash, g, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY, sample_key TEXT UNIQUE NOT NULL, image TEXT NOT NULL,
    dataset TEXT NOT NULL, category TEXT NOT NULL, draft TEXT, opinion TEXT, reference TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, sample_id INTEGER NOT NULL REFERENCES samples(id),
    reviewer TEXT NOT NULL REFERENCES users(username), kind TEXT NOT NULL,
    verdict TEXT NOT NULL, annotation TEXT, reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS reviews_sample ON reviews(sample_id, id);
"""
VERDICTS = {"accept": "Accept", "correct": "Correct", "escalate": "Escalate"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("REWORKVLM_SECRET_KEY"),
        DATABASE=Path(os.getenv("REWORKVLM_DB", Path(app.instance_path) / "reviews.sqlite")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("REWORKVLM_HTTPS") == "1",
        MAX_CONTENT_LENGTH=1024 * 1024,
        MAX_FORM_MEMORY_SIZE=512 * 1024,
    )
    if not app.secret_key:
        secret_path = Path(app.instance_path) / "session.key"
        secret_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            with secret_path.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_hex(32))
            secret_path.chmod(0o600)
        except FileExistsError:
            pass
        app.secret_key = secret_path.read_text(encoding="utf-8")

    def db():
        if "db" not in g:
            file = app.config["DATABASE"]
            file.parent.mkdir(exist_ok=True, parents=True)
            g.db = sqlite3.connect(file, timeout=20)
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys = ON")
        return g.db

    with app.app_context():
        db().execute("PRAGMA journal_mode = WAL")
        db().executescript(SCHEMA)
        db().close()

    @app.teardown_appcontext
    def close_db(_error):
        if connection := g.pop("db", None):
            connection.close()

    def image_file(name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name or not relative.parts:
            raise ValueError("Expected a relative image path")
        roots = {"mvtec_loco_anomaly_detection": os.getenv("REWORKVLM_LOCO_ROOT"),
                 "VID-AD": os.getenv("REWORKVLM_VID_ROOT")}
        specific = roots.get(relative.parts[0])
        root = Path(specific or os.getenv("REWORKVLM_IMAGE_ROOT", Path(app.root_path) / "images")).resolve()
        parts = relative.parts[1:] if specific else relative.parts
        path = root.joinpath(*parts).resolve()
        if path.suffix.lower() not in EXTENSIONS or not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(24)
        return session["csrf"]

    app.jinja_env.globals.update(csrf_token=csrf_token, verdicts=VERDICTS)

    @app.template_filter("pretty")
    def pretty(value):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, str) else json.dumps(parsed, indent=2, ensure_ascii=False)

    @app.before_request
    def authenticate():
        if request.method == "POST" and (not session.get("csrf") or not hmac.compare_digest(
                session["csrf"], request.form.get("csrf", ""))):
            abort(400, "The form expired. Reload the page and try again.")
        g.user = None
        if username := session.get("username"):
            g.user = db().execute("SELECT username, role FROM users WHERE username=?", (username,)).fetchone()
            if g.user is None:
                session.clear()
        if g.user is None and request.endpoint not in {"login", "static", None}:
            return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user = db().execute("SELECT * FROM users WHERE username=?", (request.form.get("username"),)).fetchone()
            if user is None or not check_password_hash(user["password"], request.form.get("password", "")):
                flash("Invalid username or password.", "error")
                return render_template("login.html"), 401
            session.clear()
            session["username"] = user["username"]
        if session.get("username"):
            return redirect(url_for("queue"))
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    def queue():
        query = request.args.get("q", "").strip()[:120]
        dataset = request.args.get("dataset", "")
        status = request.args.get("status", "")
        page = max(1, request.args.get("page", 1, type=int))
        states = {"pending": "review_count = 0", "reviewed": "review_count > 0 AND settled = 0",
                  "final": "settled = 1"}
        statement = """
            WITH overview AS (
                SELECT s.id, s.sample_key, s.dataset, s.category, s.image, COUNT(r.id) AS review_count,
                       COALESCE(MAX(r.kind = 'adjudication' AND r.verdict != 'escalate'), 0) AS settled
                FROM samples s LEFT JOIN reviews r ON r.sample_id=s.id
                WHERE (? = '' OR s.dataset = ?)
                  AND (? = '' OR instr(lower(s.sample_key || ' ' || s.image || ' ' || s.category), lower(?)) > 0)
                GROUP BY s.id
            )
            SELECT * FROM overview WHERE """ + states.get(status, "1=1") + " ORDER BY id LIMIT 31 OFFSET ?"
        results = db().execute(statement, (dataset, dataset, query, query, min(page - 1, 100000) * 30)).fetchall()
        datasets = db().execute("SELECT DISTINCT dataset FROM samples ORDER BY dataset").fetchall()
        return render_template("queue.html", samples=results[:30], more=len(results) > 30, page=page,
                               query=query, dataset=dataset, datasets=datasets, status=status)

    def sample_or_404(sample_id):
        row = db().execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
        if row is None:
            abort(404)
        return row

    @app.route("/samples/<int:sample_id>", methods=["GET", "POST"])
    def sample(sample_id):
        item = sample_or_404(sample_id)
        if request.method == "POST":
            kind = request.form.get("kind", "")
            verdict = request.form.get("verdict", "")
            reason = request.form.get("reason", "").strip()
            correction = request.form.get("correction", "").strip()
            if kind not in {"review", "adjudication"} or verdict not in VERDICTS or not reason or len(reason) > 8000:
                abort(400, "Select a decision and enter a reason of up to 8,000 characters.")
            if kind == "adjudication" and g.user["role"] != "adjudicator":
                abort(403)
            if correction:
                try:
                    document = json.loads(correction)
                except (ValueError, RecursionError):
                    abort(400, "The corrected annotation must be a JSON object.")
                if not isinstance(document, dict):
                    abort(400, "The corrected annotation must be a JSON object.")
                correction = json.dumps(document, ensure_ascii=False)
            if verdict == "correct" and not correction:
                abort(400, "A corrected annotation is required for this decision.")
            with db() as connection:
                connection.execute("BEGIN IMMEDIATE")
                final = connection.execute("""SELECT 1 FROM reviews
                    WHERE sample_id=? AND kind='adjudication' AND verdict!='escalate'""", (sample_id,)).fetchone()
                if final:
                    abort(409, "This sample already has a final decision.")
                connection.execute("""INSERT INTO reviews(sample_id, reviewer, kind, verdict, annotation, reason)
                    VALUES (?, ?, ?, ?, ?, ?)""", (sample_id, g.user["username"], kind, verdict, correction or None, reason))
            flash("Review saved." if kind == "review" else "Adjudication saved.", "success")
            return redirect(url_for("sample", sample_id=sample_id))

        reviews = db().execute("SELECT * FROM reviews WHERE sample_id=? ORDER BY id DESC", (sample_id,)).fetchall()
        final = next((row for row in reviews if row["kind"] == "adjudication" and row["verdict"] != "escalate"), None)
        previous = db().execute("SELECT MAX(id) FROM samples WHERE id<?", (sample_id,)).fetchone()[0]
        following = db().execute("SELECT MIN(id) FROM samples WHERE id>?", (sample_id,)).fetchone()[0]
        return render_template("sample.html", item=item, reviews=reviews, final=final, previous=previous, following=following)

    @app.get("/samples/<int:sample_id>/image")
    def image(sample_id):
        try:
            path = image_file(sample_or_404(sample_id)["image"])
        except (ValueError, FileNotFoundError):
            abort(404)
        response = send_file(path, conditional=True, max_age=0)
        response.cache_control.private = True
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/export")
    def export():
        if g.user["role"] != "adjudicator":
            abort(403)
        def stream(rows):
            for row in rows:
                data = dict(row)
                if data["annotation"] is not None:
                    data["annotation"] = json.loads(data["annotation"])
                yield json.dumps(data, ensure_ascii=False) + "\n"

        records = db().execute("""SELECT s.sample_key, s.image, s.dataset, r.reviewer, r.kind, r.verdict,
            r.annotation, r.reason, r.created_at FROM reviews r JOIN samples s ON r.sample_id=s.id ORDER BY r.id""").fetchall()
        return Response(stream(records), mimetype="application/x-ndjson",
                        headers={"Content-Disposition": "attachment; filename=reworkvlm-reviews.jsonl"})

    @app.cli.command("add-user")
    @click.argument("username")
    @click.option("--role", type=click.Choice(["reviewer", "adjudicator"]), default="reviewer")
    @click.password_option()
    def add_user(username, role, password):
        if not re.fullmatch(r"[\w.\-]{1,64}", username) or len(password) < 5:
            raise click.ClickException("Choose a simple username and a password of at least 5 characters.")
        try:
            with db() as connection:
                connection.execute("INSERT INTO users VALUES (?, ?, ?)", (username, generate_password_hash(password), role))
        except sqlite3.IntegrityError as error:
            raise click.ClickException("This account already exists.") from error
        click.echo(f"Created {role} {username}")

    @app.cli.command("import-items")
    @click.argument("file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
    @click.option("--from-annotations", is_flag=True, help="Import released annotations as reference context.")
    def import_items(file, from_annotations):
        added = 0
        with db() as connection, file.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                    image = row["image"]
                    image_file(image)
                    dataset = row.get("dataset", "")
                    if from_annotations:
                        reference = row["annotation"]
                        sample_key = image
                        category = reference["category"]
                        draft, opinion = None, None
                        reference = json.dumps({"annotation": reference, "reasoning": row.get("reasoning")}, ensure_ascii=False)
                    else:
                        sample_key = row["id"]
                        category = row.get("category", "")
                        draft = json.dumps(row["draft_annotation"], ensure_ascii=False) if row.get("draft_annotation") is not None else None
                        opinion, reference = row.get("draft_opinion"), None
                    if not isinstance(sample_key, str) or not sample_key or opinion is not None and not isinstance(opinion, str):
                        raise ValueError("Invalid sample ID or model opinion")
                    values = (sample_key, image, dataset, category, draft, opinion, reference)
                    cursor = connection.execute("""INSERT INTO samples(sample_key,image,dataset,category,draft,opinion,reference)
                        VALUES (?,?,?,?,?,?,?) ON CONFLICT(sample_key) DO NOTHING""", values)
                    if not cursor.rowcount:
                        existing = connection.execute("""SELECT sample_key,image,dataset,category,draft,opinion,reference
                            FROM samples WHERE sample_key=?""", (sample_key,)).fetchone()
                        if tuple(existing) != values:
                            raise ValueError("A sample with this ID already has different content")
                    added += cursor.rowcount
                except (TypeError, KeyError, ValueError, FileNotFoundError, sqlite3.IntegrityError) as error:
                    raise click.ClickException(f"Line {line_number}: {error}") from error
        click.echo(f"Imported {added} new samples")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
