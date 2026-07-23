"""
Human labeling / review queue for the collection pipeline.

Nothing reaches the training set just because a YouTube search query
happened to return it. A reviewer watches each candidate clip and either
confirms the label, corrects it, or rejects the clip outright.

Run locally (one domain per instance):
    rehab-review --domain lower_limb
    rehab-review --domain upper_body --port 5051
Then open http://127.0.0.1:5050 (or the port you chose)

Security note: this serves raw video files over local HTTP with no
authentication. It's meant to run on localhost / a trusted internal network
only -- do not expose this port publicly. It is a labeling tool, not a
patient-facing service.
"""
import os
import csv
import glob
import shutil
import secrets
import argparse
from datetime import datetime

from flask import Flask, request, redirect, url_for, send_from_directory, session, render_template_string

from ..domains import get_domain, DOMAIN_NAMES

REVIEW_LOG_FIELDS = ["filename", "query_class", "decision", "confirmed_class",
                      "accepted_filename", "reason", "reviewer", "reviewed_at"]

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # local single-user tool, ephemeral is fine

DOMAIN = None  # set from --domain
BASE_DIR = PENDING_DIR = RAW_DIR = REVIEW_LOG = None


def configure(domain_name):
    global DOMAIN, BASE_DIR, PENDING_DIR, RAW_DIR, REVIEW_LOG
    DOMAIN = get_domain(domain_name)
    BASE_DIR = os.path.join("datasets", DOMAIN.name)
    PENDING_DIR = os.path.join(BASE_DIR, "pending_review")
    RAW_DIR = os.path.join(BASE_DIR, "raw")
    REVIEW_LOG = os.path.join(BASE_DIR, "review_log.csv")
    for c in DOMAIN.classes:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)


def log_decision(row):
    file_exists = os.path.exists(REVIEW_LOG)
    with open(REVIEW_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def next_pending(class_filter=None):
    pattern = os.path.join(PENDING_DIR, class_filter or "*", "*.mp4")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def pending_counts():
    return {c: len(glob.glob(os.path.join(PENDING_DIR, c, "*.mp4"))) for c in DOMAIN.classes}


PAGE = """
<!doctype html>
<title>{{ domain_name }} dataset review</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; }
  video { width: 100%; max-height: 480px; background: #000; }
  .counts { font-size: 0.85rem; color: #555; margin-bottom: 1rem; }
  fieldset { margin-top: 1rem; }
  button { padding: 0.5rem 1rem; margin-right: 0.5rem; cursor: pointer; }
  .accept { background: #d4f7dc; }
  .reject { background: #f7d4d4; }
  select, input[type=text] { padding: 0.3rem; }
</style>
<h2>{{ domain_name }} review queue</h2>
<div class="counts">
  Pending per class:
  {% for c, n in counts.items() %} {{c}}={{n}} {% endfor %}
</div>
{% if video %}
  <p><b>File:</b> {{ filename }} &nbsp; <b>Query-based guess:</b> {{ query_class }}</p>
  <video controls src="{{ video_url }}"></video>
  <form method="post" action="{{ url_for('submit') }}">
    <input type="hidden" name="filepath" value="{{ filepath }}">
    <input type="hidden" name="query_class" value="{{ query_class }}">
    <input type="hidden" name="class_filter" value="{{ class_filter or '' }}">
    <fieldset>
      <label>Reviewer name: <input type="text" name="reviewer" value="{{ reviewer }}" required></label>
    </fieldset>
    <fieldset>
      <label>Actual exercise shown:
        <select name="confirmed_class">
          {% for c in classes %}
            <option value="{{c}}" {% if c == query_class %}selected{% endif %}>{{c}}</option>
          {% endfor %}
        </select>
      </label>
      <button class="accept" type="submit" name="decision" value="accept">Accept with this label</button>
    </fieldset>
    <fieldset>
      <label>Rejection reason:
        <select name="reason">
          <option value="wrong_exercise">Wrong / unrelated exercise</option>
          <option value="not_a_patient_or_demo">Not a real exercise demonstration</option>
          <option value="poor_visibility">Relevant body region not clearly visible</option>
          <option value="multiple_people">Multiple people / not solo</option>
          <option value="low_quality">Low quality / unusable footage</option>
          <option value="other">Other</option>
        </select>
      </label>
      <button class="reject" type="submit" name="decision" value="reject">Reject</button>
    </fieldset>
  </form>
{% else %}
  <p>Queue is empty for this filter. Nothing left to review.</p>
{% endif %}
<hr>
<p>Filter: <a href="{{ url_for('index') }}">all</a>
{% for c in classes %} | <a href="{{ url_for('index', class_filter=c) }}">{{c}}</a>{% endfor %}
</p>
"""


@app.route("/")
def index():
    class_filter = request.args.get("class_filter")
    path = next_pending(class_filter)
    counts = pending_counts()
    if not path:
        return render_template_string(PAGE, video=False, counts=counts, classes=DOMAIN.classes,
                                       class_filter=class_filter, domain_name=DOMAIN.name)

    query_class = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    return render_template_string(
        PAGE, video=True, counts=counts, classes=DOMAIN.classes,
        filepath=path, filename=filename, query_class=query_class,
        video_url=url_for("serve_video", query_class=query_class, filename=filename),
        reviewer=session.get("reviewer", ""), class_filter=class_filter, domain_name=DOMAIN.name,
    )


@app.route("/video/<query_class>/<filename>")
def serve_video(query_class, filename):
    directory = os.path.join(PENDING_DIR, query_class)
    return send_from_directory(directory, filename)


@app.route("/submit", methods=["POST"])
def submit():
    filepath = request.form["filepath"]
    query_class = request.form["query_class"]
    decision = request.form["decision"]
    reviewer = request.form.get("reviewer", "unknown").strip() or "unknown"
    session["reviewer"] = reviewer
    filename = os.path.basename(filepath)

    if not os.path.exists(filepath):
        return redirect(url_for("index"))

    if decision == "accept":
        confirmed_class = request.form["confirmed_class"]
        existing = len(glob.glob(os.path.join(RAW_DIR, confirmed_class, "*.mp4")))
        new_name = f"{confirmed_class}_{existing + 1:04d}.mp4"
        dest = os.path.join(RAW_DIR, confirmed_class, new_name)
        shutil.move(filepath, dest)
        log_decision({
            "filename": filename, "query_class": query_class, "decision": "accept",
            "confirmed_class": confirmed_class, "accepted_filename": new_name, "reason": "",
            "reviewer": reviewer, "reviewed_at": datetime.now().isoformat(),
        })
    else:
        reason = request.form.get("reason", "other")
        os.remove(filepath)  # rejected clips are not retained
        log_decision({
            "filename": filename, "query_class": query_class, "decision": "reject",
            "confirmed_class": "", "accepted_filename": "", "reason": reason,
            "reviewer": reviewer, "reviewed_at": datetime.now().isoformat(),
        })

    class_filter = request.form.get("class_filter") or None
    return redirect(url_for("index", class_filter=class_filter) if class_filter else url_for("index"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    configure(args.domain)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
