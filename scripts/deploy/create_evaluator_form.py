"""
Create the Gap2Idea evaluator Google Form via the Forms API.

Loads 15 pipeline-generated ideas from runs/{ai,ml,math}/artifacts/ideas.tsv,
builds a multi-section form (description + 3 Likert scales per idea), and
prints both the private Edit URL and the public Share URL.

Prerequisite (run ONCE, from your terminal):

    gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/forms.body,https://www.googleapis.com/auth/drive.file

This re-runs your OAuth flow and adds the two scopes the Forms API needs.
Whichever Google account you sign in with becomes the form's OWNER (their Drive).

Then run:

    .venv/Scripts/python.exe scripts/deploy/create_evaluator_form.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]

DOMAIN_LABELS = {"ai": "AI", "ml": "Classical ML", "math": "Mathematics"}
REPO = Path(__file__).resolve().parents[2]


def truncate(s, n):
    s = str(s or "").strip().replace("\r", " ").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def load_ideas():
    blocks, n = [], 0
    for key in ("ai", "ml", "math"):
        df = pd.read_csv(REPO / "runs" / key / "artifacts" / "ideas.tsv", sep="\t")
        for _, r in df.iterrows():
            n += 1
            blocks.append({
                "domain": DOMAIN_LABELS[key],
                "n": n,
                "title": truncate(r["title"], 110),
                "rq": truncate(r["research_question"], 280),
                "method": truncate(r["method_sketch"], 380),
                "evalp": truncate(r["evaluation_plan"], 320),
                "risks": truncate(r["assumptions_and_risks"], 280),
            })
    return blocks


def main():
    ideas = load_ideas()
    print(f"Loaded {len(ideas)} ideas (5 per domain).")

    try:
        creds, _ = default(scopes=SCOPES)
    except Exception as e:
        print("\nERROR: could not load application-default credentials.")
        print("Did you run the gcloud command from the script header? Details:")
        print(f"  {e}")
        sys.exit(2)

    forms = build("forms", "v1", credentials=creds, cache_discovery=False)

    # Step 1: create form (Forms API only accepts info.title at creation time)
    try:
        form = forms.forms().create(body={
            "info": {"title": "Gap2Idea — Research Idea Evaluation"}
        }).execute()
    except HttpError as e:
        print("\nERROR: Forms API create failed. Likely your gcloud ADC doesn't have")
        print("the forms.body / drive.file scopes. Re-run the gcloud command from")
        print("the script header. Server said:")
        print(f"  {e}")
        sys.exit(3)

    form_id = form["formId"]
    print(f"Created form: {form_id}")

    # Step 2: build the item-creation requests in order
    items = []

    items.append({
        "title": "Name / pseudonym (optional)",
        "questionItem": {"question": {"required": False, "textQuestion": {}}},
    })
    items.append({
        "title": "Which domain is closest to your expertise?",
        "questionItem": {"question": {
            "required": False,
            "choiceQuestion": {
                "type": "RADIO",
                "options": [{"value": v} for v in
                    ["AI / LLMs", "Classical ML / statistics",
                     "Mathematics", "Other"]],
            },
        }},
    })

    current_domain = None
    for idea in ideas:
        if idea["domain"] != current_domain:
            items.append({
                "title": f"Section: {idea['domain']}",
                "description":
                    f"Five ideas in {idea['domain']}. Score each on three axes.",
                "pageBreakItem": {},
            })
            current_domain = idea["domain"]

        desc = (
            f"Research question: {idea['rq']}\n\n"
            f"Method sketch: {idea['method']}\n\n"
            f"Evaluation plan: {idea['evalp']}\n\n"
            f"Assumptions and risks: {idea['risks']}"
        )
        items.append({
            "title": f"Idea {idea['n']} — {idea['title']}",
            "description": desc,
            "textItem": {},
        })
        for axis, low, high in [
            ("Novelty", "Not novel", "Highly novel"),
            ("Feasibility", "Very hard", "Clearly feasible"),
            ("Potential impact", "Marginal", "Field-shifting"),
        ]:
            items.append({
                "title": f"Idea {idea['n']} — {axis}",
                "questionItem": {"question": {
                    "required": True,
                    "scaleQuestion": {
                        "low": 1, "high": 5,
                        "lowLabel": low, "highLabel": high,
                    },
                }},
            })

    items.append({"title": "Wrap-up", "pageBreakItem": {}})
    items.append({
        "title": "Any general comments? (optional)",
        "questionItem": {"question": {
            "required": False,
            "textQuestion": {"paragraph": True},
        }},
    })

    # Step 3: batchUpdate in chunks of 30 (well under the 500-req cap, easy to
    # diagnose if one chunk fails)
    BATCH = 30
    appended = 0
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        requests = [
            {"createItem": {"item": item, "location": {"index": appended + j}}}
            for j, item in enumerate(chunk)
        ]
        try:
            forms.forms().batchUpdate(
                formId=form_id, body={"requests": requests}
            ).execute()
        except HttpError as e:
            print(f"\nERROR in batch starting at item {start}: {e}")
            print("(Partial form was created — open the Edit URL below to inspect.)")
            break
        appended += len(chunk)
        print(f"  Added items {start + 1}-{start + len(chunk)} of {len(items)}")

    # Step 4: set the form's description (separate API call: updateFormInfo)
    description = (
        "Score 15 research ideas (5 AI, 5 classical ML, 5 mathematics). "
        "Each idea is graded on three 1-5 Likert axes: Novelty, Feasibility, "
        "Potential impact. Expected time: 12-15 minutes. "
        "Responses are anonymous unless you fill in your name."
    )
    try:
        forms.forms().batchUpdate(formId=form_id, body={"requests": [{
            "updateFormInfo": {
                "info": {"description": description},
                "updateMask": "description",
            },
        }]}).execute()
    except HttpError as e:
        print(f"WARN: could not set description: {e}")

    # Step 5: fetch final URLs
    full = forms.forms().get(formId=form_id).execute()
    edit_url = f"https://docs.google.com/forms/d/{form_id}/edit"
    share_url = full.get("responderUri", "(open Edit URL and use Send -> link)")

    print()
    print("=" * 60)
    print("FORM CREATED")
    print("=" * 60)
    print(f"Edit URL  (private, keep for yourself): {edit_url}")
    print(f"Share URL (give to evaluators):        {share_url}")
    print()
    print("Next: in the form, click Responses -> Sheets icon to link a sheet")
    print("you can later export to CSV for scripts/eval/analyze_eval_responses.py.")


if __name__ == "__main__":
    main()
