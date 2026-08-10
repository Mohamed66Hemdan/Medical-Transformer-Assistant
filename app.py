"""
Medical Transformer Assistant — with Explainable Predictions
==============================================================
Same engine as before (NER, zero-shot classification, QA, summarization,
word-importance / attention / counterfactual explainability). This version
adds:

  1. A restyled, dark UI (the default Streamlit white background is gone).
  2. A "Report Type" toggle — Clinical / Doctor vs. Patient-Friendly — that
     changes what's shown on screen.
  3. A "Download PDF" button that generates a matching PDF: the full
     technical report for clinicians, or a plain-language summary with a
     disclaimer for patients.

Run with:
    pip install streamlit torch transformers matplotlib reportlab
    streamlit run medical_assistant_app.py
"""

import io
from datetime import date

import streamlit as st
import torch
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CANDIDATE_LABELS = [
    "Respiratory disease",
    "Cardiovascular disease",
    "Gastrointestinal disease",
    "Neurological disease",
    "Infectious disease",
]

NER_MODEL = "d4data/biomedical-ner-all"
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
QA_MODEL = "deepset/roberta-base-squad2"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
NLI_MODEL = "roberta-large-mnli"

SAMPLE_TEXT = (
    "The patient has persistent cough, fever, and shortness of breath. "
    "Chest X-ray showed bilateral infiltrates."
)

DISCLAIMER = (
    "This AI-generated report is intended for decision support only. It does "
    "not constitute a clinical diagnosis. Clinical judgment and final "
    "decisions remain with the responsible healthcare professional."
)

SCORE_NOTE = (
    "This score reflects the model's relative classification confidence "
    "among the candidate categories below, not a clinical diagnosis or a "
    "probability of disease."
)

BUCKET_ORDER = ["Symptoms", "Imaging Findings", "Diagnostic Procedures", "Other Clinical Mentions"]


IMAGING_KEYWORDS = {
    "infiltrate", "infiltrates", "opacity", "opacities", "consolidation",
    "effusion", "nodule", "nodules", "mass", "lesion", "lesions",
    "atelectasis", "pneumothorax", "cardiomegaly", "bilateral", "unilateral",
}


def extract_pdf_text(uploaded_file):
    """Extract plain text from an uploaded PDF report. Returns '' on any
    extraction failure (e.g. a scanned/image-only PDF with no text layer)."""
    try:
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(p.strip() for p in pages if p.strip()).strip()
    except Exception:
        return ""


def bucket_entity(entity_group, word=""):
    """Keyword bucketing so raw NER entity_group labels (which vary by
    model, e.g. 'Sign_symptom', 'Disease_disorder', 'Diagnostic_procedure')
    become plain-language, clinically-sensible categories for the UI and PDF.

    Imaging-report vocabulary (infiltrate, opacity, effusion...) is pulled out
    of "Symptoms" into its own "Imaging Findings" bucket even when the NER
    model tags it as a symptom/sign, since presenting it as a patient-reported
    symptom is clinically misleading."""
    g = entity_group.lower()
    w = word.lower().strip()

    if w in IMAGING_KEYWORDS or any(k in w for k in IMAGING_KEYWORDS):
        return "Imaging Findings"
    if "symptom" in g or "sign" in g:
        return "Symptoms"
    if "diagnostic" in g or "procedure" in g:
        return "Diagnostic Procedures"
    if "disease" in g or "disorder" in g or "finding" in g:
        return "Imaging Findings"
    return "Other Clinical Mentions"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    device = 0 if torch.cuda.is_available() else -1

    ner_pipe = pipeline(
        "token-classification",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=device,
    )
    classifier = pipeline("zero-shot-classification", model=CLASSIFIER_MODEL, device=device)

    qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    qa_model.eval()

    sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    sum_model.eval()

    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, output_attentions=True
    )
    nli_model.eval()

    return {
        "ner": ner_pipe,
        "classifier": classifier,
        "qa_tokenizer": qa_tokenizer,
        "qa_model": qa_model,
        "sum_tokenizer": sum_tokenizer,
        "sum_model": sum_model,
        "nli_tokenizer": nli_tokenizer,
        "nli_model": nli_model,
    }


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------
def run_ner(models, text):
    return models["ner"](text)


def run_classification(models, text, candidate_labels=CANDIDATE_LABELS):
    return models["classifier"](text, candidate_labels)


def run_summarization(models, text, max_length=45, min_length=10):
    tok, model = models["sum_tokenizer"], models["sum_model"]
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs, max_length=max_length, min_length=min_length, num_beams=4, do_sample=False
        )
    return tok.decode(summary_ids[0], skip_special_tokens=True)


def run_qa(models, text, question):
    tok, model = models["qa_tokenizer"], models["qa_model"]
    inputs = tok(question, text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tok.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
    start_score = torch.softmax(outputs.start_logits, dim=-1)[0, start].item()
    end_score = torch.softmax(outputs.end_logits, dim=-1)[0, end - 1].item()
    confidence = (start_score + end_score) / 2
    return {"answer": answer, "score": confidence}


def word_importance(models, text, label, top_k=8):
    words = text.split()
    base_score = models["classifier"](text, [label])["scores"][0]
    importances = []
    for i in range(len(words)):
        occluded = " ".join(words[:i] + words[i + 1:])
        if not occluded.strip():
            continue
        occluded_score = models["classifier"](occluded, [label])["scores"][0]
        importances.append((words[i], base_score - occluded_score))
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances[:top_k], base_score


def attention_visualization(models, text, label, top_k=8):
    tok, model = models["nli_tokenizer"], models["nli_model"]
    hypothesis = f"This text is about {label}."
    inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_layer_attn = outputs.attentions[-1][0]
    avg_attn = last_layer_attn.mean(dim=0)
    cls_attn = avg_attn[0]
    tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])
    pairs = [
        (t.replace("Ġ", ""), s)
        for t, s in zip(tokens, cls_attn.tolist())
        if t not in tok.all_special_tokens
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


def counterfactual_analysis(models, text, phrase, label):
    base_score = models["classifier"](text, [label])["scores"][0]
    if phrase not in text:
        return None
    cf_text = text.replace(phrase, "").replace("  ", " ").strip()
    cf_score = models["classifier"](cf_text, [label])["scores"][0]
    return {"base_score": base_score, "cf_score": cf_score, "delta": base_score - cf_score}


def make_bar_png(labels, values, xlabel, dark=True):
    """Render a horizontal bar chart to PNG bytes (used for both the UI and the PDF)."""
    bg = "#1c1f2b" if dark else "white"
    fg = "#e6e6f0" if dark else "black"
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.barh(labels[::-1], values[::-1], color="#5eead4")
    ax.set_xlabel(xlabel, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=bg, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------
def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#0f766e")
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "SubHeading",
            parent=styles["Heading4"],
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "Note",
            parent=styles["Normal"],
            textColor=colors.HexColor("#5b6472"),
            fontSize=8.5,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            textColor=colors.HexColor("#7a1f1f"),
            backColor=colors.HexColor("#fdecea"),
            borderPadding=8,
            spaceBefore=12,
        )
    )
    return styles


def _pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 7.5)
    canvas.setFillColor(colors.HexColor("#7a7a7a"))
    canvas.drawString(
        0.6 * inch, 0.4 * inch,
        "AI-generated report for decision support only \u2014 not a clinical diagnosis.",
    )
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _clinical_findings_flowables(styles, entities):
    """Render extracted entities grouped into clinically-sensible buckets, in
    a fixed clinical order, skipping any bucket that's empty."""
    flows = []
    buckets = {}
    for ent in entities:
        buckets.setdefault(bucket_entity(ent["entity_group"], ent["word"]), []).append(ent["word"])
    if not buckets:
        flows.append(Paragraph("No entities detected.", styles["Normal"]))
        return flows
    ordered = [b for b in BUCKET_ORDER if b in buckets] + [b for b in buckets if b not in BUCKET_ORDER]
    for bucket in ordered:
        items = buckets[bucket]
        flows.append(Paragraph(bucket, styles["SubHeading"]))
        flows.append(Paragraph("&bull; " + "<br/>&bull; ".join(items), styles["Normal"]))
    return flows


def generate_doctor_pdf(report_text, result, counterfactual=None, qa_pair=None, patient_name="", report_date=""):
    styles = _pdf_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Medical AI Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph("Clinical / Professional View", styles["Normal"]))
    meta_bits = []
    if patient_name:
        meta_bits.append(f"Patient: {patient_name}")
    meta_bits.append(f"Analysis date: {report_date}" if report_date else "")
    story.append(Paragraph(" &nbsp;|&nbsp; ".join(b for b in meta_bits if b), styles["Note"]))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#0f766e")))

    story.append(Paragraph("Patient Report", styles["SectionHeading"]))
    story.append(Paragraph(report_text, styles["Normal"]))

    story.append(Paragraph("Extracted Clinical Information", styles["SectionHeading"]))
    story.extend(_clinical_findings_flowables(styles, result["entities"]))

    story.append(Paragraph("AI Text Classification", styles["SectionHeading"]))
    story.append(
        Paragraph(
            f"<b>AI predicted category:</b> {result['top_label']}<br/>"
            f"<b>Model score:</b> {result['top_score']:.1%}",
            styles["Normal"],
        )
    )
    story.append(Paragraph(f"\u26a0 {SCORE_NOTE}", styles["Note"]))
    table_data = [["Category", "Model score"]] + [
        [label, f"{score:.1%}"]
        for label, score in zip(result["classification"]["labels"], result["classification"]["scores"])
    ]
    t = Table(table_data, colWidths=[3.2 * inch, 1.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7f6")]),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(t)

    story.append(Paragraph("Explainability", styles["SectionHeading"]))
    words = [w for w, _ in result["word_importance"]]
    scores = [s for _, s in result["word_importance"]]
    story.append(Paragraph("<b>Words that most influenced the AI classification</b> (highest first):", styles["Normal"]))
    story.append(
        Paragraph(
            ", ".join(f"{i+1}. {w}" for i, w in enumerate(words)),
            styles["Normal"],
        )
    )
    story.append(Paragraph("Higher contribution indicates greater influence on the model's predicted category.", styles["Note"]))
    img_buf = make_bar_png(words, scores, "Contribution to confidence", dark=False)
    story.append(RLImage(img_buf, width=5 * inch, height=2.8 * inch))

    # Advanced/optional: only appears if the physician explicitly ran and
    # chose to include a counterfactual test — never shown by default.
    if counterfactual:
        story.append(Paragraph("Advanced Analysis \u2014 Counterfactual", styles["SectionHeading"]))
        story.append(
            Paragraph(
                f"Original model score for \u201c{result['top_label']}\u201d: {counterfactual['base_score']:.1%}<br/>"
                f"After removing the tested phrase: {counterfactual['cf_score']:.1%}<br/>"
                f"Change: {counterfactual['delta']:+.1%}",
                styles["Normal"],
            )
        )

    story.append(Paragraph("Clinical Summary", styles["SectionHeading"]))
    story.append(Paragraph(result["summary"], styles["Normal"]))

    if qa_pair:
        story.append(Paragraph("Question &amp; Answer", styles["SectionHeading"]))
        story.append(
            Paragraph(
                f"<b>Q:</b> {qa_pair['question']}<br/><b>A:</b> {qa_pair['answer']} "
                f"({qa_pair['score']:.0%} confidence)",
                styles["Normal"],
            )
        )

    story.append(Paragraph(DISCLAIMER, styles["Disclaimer"]))
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    buf.seek(0)
    return buf


def generate_patient_pdf(report_text, result):
    styles = _pdf_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.7 * inch)
    story = []

    story.append(Paragraph("Your Report Summary", styles["ReportTitle"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#0f766e")))

    buckets = {}
    for ent in result["entities"]:
        buckets.setdefault(bucket_entity(ent["entity_group"], ent["word"]), []).append(ent["word"].lower())
    symptoms = buckets.get("Symptoms", [])
    findings = buckets.get("Imaging Findings", []) + buckets.get("Other Clinical Mentions", [])

    story.append(Paragraph("Report Summary", styles["SectionHeading"]))
    if symptoms:
        story.append(Paragraph(f"The report mentions {', '.join(symptoms)}.", styles["Normal"]))
    else:
        story.append(Paragraph(result["summary"], styles["Normal"]))

    story.append(Paragraph("Main Findings", styles["SectionHeading"]))
    if findings:
        story.append(Paragraph(f"The report also mentions {', '.join(findings)}.", styles["Normal"]))
    else:
        story.append(Paragraph("No additional findings were highlighted.", styles["Normal"]))

    story.append(Paragraph("AI Interpretation", styles["SectionHeading"]))
    story.append(
        Paragraph(
            f"The text was classified as being most consistent with a "
            f"{result['top_label'].lower()}-related condition.",
            styles["Normal"],
        )
    )

    story.append(Paragraph("Important", styles["SectionHeading"]))
    story.append(Paragraph(DISCLAIMER, styles["Disclaimer"]))

    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg-0: #0a0c12;
    --bg-1: #10141d;
    --panel: #161b26;
    --panel-border: #262c3d;
    --text: #e8eaf2;
    --text-dim: #9aa3b8;
    --accent: #2dd4bf;
    --accent-dark: #0f766e;
    --danger-bg: #3a1d1d;
    --danger-border: #7a3a3a;
    --danger-text: #f3c9c9;
}

.stApp {
    background: radial-gradient(circle at top left, var(--bg-1) 0%, var(--bg-0) 55%, #060709 100%);
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }

/* Base text: every default Streamlit text element gets an explicit light
   color so nothing renders dark-on-dark. */
p, span, li, label, div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}
h1, h2, h3 { color: var(--accent) !important; font-weight: 700 !important; }
h4, h5 { color: var(--text) !important; }
.subtitle { color: var(--text-dim) !important; margin-top: -6px; margin-bottom: 22px; font-size: 1.02rem; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-dim) !important; }

/* Real Streamlit bordered containers used for every "card" section below —
   replaces the old markdown div hack, which rendered as empty bars because
   an opening/closing <div> pair split across separate st.markdown() calls
   never actually wraps the widgets in between. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 14px !important;
    margin-bottom: 18px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 4px 6px; }

.pill {
    display: inline-block;
    background: rgba(45, 212, 191, 0.12);
    color: var(--accent);
    border: 1px solid var(--accent-dark);
    border-radius: 999px;
    padding: 3px 13px;
    font-size: 0.82rem;
    margin: 3px 5px 3px 0;
}
.bucket-label {
    color: var(--text-dim) !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 10px 0 4px 0;
}
.disclaimer-box {
    background: var(--danger-bg);
    border: 1px solid var(--danger-border);
    color: var(--danger-text) !important;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 10px;
}
.disclaimer-box * { color: var(--danger-text) !important; }
.score-note {
    color: var(--text-dim) !important;
    font-size: 0.82rem;
    font-style: italic;
    margin-top: -4px;
}
.upload-label {
    color: var(--accent) !important;
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 4px;
}
.upload-divider {
    color: var(--text-dim) !important;
    font-size: 0.8rem;
    text-align: center;
    margin: 14px 0 6px 0;
}
[data-testid="stFileUploaderDropzone"] {
    background: #0d1017 !important;
    border: 1.5px dashed var(--accent-dark) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: var(--accent-dark) !important;
    color: white !important;
    border: none !important;
}
.score-row { margin-bottom: 14px; }
.score-row-label {
    display: flex;
    justify-content: space-between;
    color: var(--text) !important;
    font-size: 0.92rem;
    margin-bottom: 4px;
}
.score-row-label span { color: var(--accent) !important; font-weight: 700; }
.score-bar-track {
    background: #0d1017;
    border: 1px solid var(--panel-border);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
}
.score-bar-fill {
    background: linear-gradient(90deg, var(--accent-dark), var(--accent));
    height: 100%;
    border-radius: 999px;
}

.stTextArea textarea, .stTextInput input {
    background: #0d1017 !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    border: 1px solid var(--panel-border) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--text-dim) !important; }

/* Buttons: cover every Streamlit button variant/data-testid so primary and
   secondary buttons both pick up styling consistently. */
.stButton > button,
button[kind="primary"], button[kind="secondary"],
button[data-testid="baseButton-primary"], button[data-testid="baseButton-secondary"] {
    border-radius: 10px !important;
    padding: 0.5rem 1.3rem !important;
    font-weight: 600 !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}
button[kind="primary"], button[data-testid="baseButton-primary"] {
    background: var(--accent-dark) !important;
    color: white !important;
    border: 1px solid var(--accent-dark) !important;
}
button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
    background: #0d9488 !important;
}
button[kind="secondary"], button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent-dark) !important;
}
button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {
    background: rgba(45, 212, 191, 0.10) !important;
    border-color: var(--accent) !important;
}
.stDownloadButton > button {
    background: var(--panel) !important;
    color: var(--accent) !important;
    border-radius: 10px !important;
    border: 1px solid var(--accent) !important;
    font-weight: 600 !important;
    width: 100%;
}

[data-testid="stMetricValue"] { color: var(--accent) !important; }
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; }
[data-testid="stMetricDelta"] { color: var(--text) !important; }

.stProgress > div > div { background: var(--accent) !important; }
.stProgress > div { background: #0d1017 !important; }

[data-testid="stAlert"] { border-radius: 10px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--bg-1) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Medical Transformer Assistant", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("🩺 Medical Transformer Assistant")
st.markdown(
    '<p class="subtitle">Explainable multi-task analysis of a medical report — '
    "NER, classification, summarization, and Q&A, all on the same report.</p>",
    unsafe_allow_html=True,
)

models = load_models()

if "report_text" not in st.session_state:
    st.session_state.report_text = SAMPLE_TEXT
if "result" not in st.session_state:
    st.session_state.result = None
if "qa_pair" not in st.session_state:
    st.session_state.qa_pair = None
if "cf_result" not in st.session_state:
    st.session_state.cf_result = None
if "cf_in_pdf" not in st.session_state:
    st.session_state.cf_in_pdf = False

with st.sidebar:
    st.markdown("### Report type")
    report_type = st.radio(
        "Who is this report for?",
        ["Clinical / Doctor", "Patient-Friendly"],
        label_visibility="collapsed",
    )
    st.caption(
        "**Clinical / Doctor** shows full model outputs — entities, confidence "
        "scores, explainability, counterfactual analysis.\n\n"
        "**Patient-Friendly** shows a plain-language summary only, with a "
        "medical disclaimer, no technical AI details."
    )

with st.container(border=True):
    if report_type == "Clinical / Doctor":
        patient_name = st.text_input("Patient name *", key="patient_name")
        report_date = date.today().strftime("%Y-%m-%d")
    else:
        patient_name, report_date = "", ""

    st.markdown('<p class="upload-label">📎 Upload a PDF report — the text below fills in automatically</p>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Upload a PDF medical report",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.get("_last_uploaded_name"):
        extracted = extract_pdf_text(uploaded_pdf)
        st.session_state["_last_uploaded_name"] = uploaded_pdf.name
        if extracted:
            st.session_state.report_text = extracted
            st.rerun()
        else:
            st.warning("Couldn't extract text from this PDF — it may be a scanned image without a text layer. Paste the report text below instead.")

    st.markdown('<p class="upload-divider">— or paste / edit the report text directly —</p>', unsafe_allow_html=True)
    report_text = st.text_area("Medical report", value=st.session_state.report_text, height=140)
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=False)
    if report_type == "Clinical / Doctor" and analyze_clicked and not patient_name.strip():
        st.error("Patient name is required before running the analysis.")
        analyze_clicked = False

if analyze_clicked and report_text.strip():
    st.session_state.report_text = report_text
    st.session_state.qa_pair = None
    st.session_state.cf_result = None
    st.session_state.cf_in_pdf = False

    with st.spinner("Step 1/4 — Understanding (NER)..."):
        entities = run_ner(models, report_text)
    with st.spinner("Step 2/4 — Classifying..."):
        clf = run_classification(models, report_text)
        top_label = clf["labels"][0]
        top_score = clf["scores"][0]
    with st.spinner("Step 3/4 — Explaining..."):
        importances, base_score = word_importance(models, report_text, top_label)
    with st.spinner("Step 4/4 — Summarizing..."):
        summary = run_summarization(models, report_text)

    st.session_state.result = {
        "entities": entities,
        "classification": clf,
        "top_label": top_label,
        "top_score": top_score,
        "word_importance": importances,
        "summary": summary,
    }

result = st.session_state.result

if result:
    if report_type == "Clinical / Doctor":
        with st.container(border=True):
            st.subheader("① Understand")
            buckets = {}
            for ent in result["entities"]:
                buckets.setdefault(bucket_entity(ent["entity_group"], ent["word"]), []).append(ent["word"])
            if buckets:
                ordered = [b for b in BUCKET_ORDER if b in buckets] + [b for b in buckets if b not in BUCKET_ORDER]
                for bucket in ordered:
                    items = buckets[bucket]
                    st.markdown(f'<p class="bucket-label">{bucket}</p>', unsafe_allow_html=True)
                    st.markdown(
                        " ".join(f'<span class="pill">{w}</span>' for w in items),
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No entities detected.")

        with st.container(border=True):
            st.subheader("② Classify")
            c1, c2 = st.columns([1, 2])
            c1.metric("AI predicted category", result["top_label"], f"{result['top_score']:.1%} model score")
            with c2:
                max_score = max(result["classification"]["scores"])
                for label, score in zip(result["classification"]["labels"], result["classification"]["scores"]):
                    pct = score * 100
                    width_pct = (score / max_score) * 100 if max_score else 0
                    st.markdown(
                        f'''<div class="score-row">
                            <div class="score-row-label">{label}<span>{pct:.1f}%</span></div>
                            <div class="score-bar-track"><div class="score-bar-fill" style="width:{width_pct:.1f}%"></div></div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
            st.markdown(f'<p class="score-note">⚠ {SCORE_NOTE}</p>', unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("③ Explain")
            words = [w for w, _ in result["word_importance"]]
            scores = [s for _, s in result["word_importance"]]
            st.markdown("**Words that most influenced the classification**")
            st.image(make_bar_png(words, scores, "Contribution to confidence"))
            st.caption("Higher contribution indicates greater influence on the model's predicted category.")

            with st.expander("🔬 Advanced Analysis — Counterfactual (optional)"):
                st.caption(
                    "Test what happens to the model's score if a specific phrase is removed from "
                    "the report. This is a deeper diagnostic tool for the AI itself — not part of "
                    "the routine reading of a report."
                )
                phrase = st.text_input(
                    "Phrase to remove (must appear verbatim in the report)",
                    value=words[0] if words else "",
                )
                if st.button("Run counterfactual"):
                    cf = counterfactual_analysis(models, st.session_state.report_text, phrase, result["top_label"])
                    st.session_state.cf_result = cf
                    st.session_state.cf_in_pdf = False
                if st.session_state.cf_result:
                    cf = st.session_state.cf_result
                    st.write(
                        f"Original: **{cf['base_score']:.1%}** → without phrase: **{cf['cf_score']:.1%}** "
                        f"(Δ {cf['delta']:+.1%})"
                    )
                    st.session_state.cf_in_pdf = st.checkbox(
                        "Add this counterfactual result to the PDF report",
                        value=st.session_state.get("cf_in_pdf", False),
                    )

        with st.container(border=True):
            st.subheader("④ Clinical Summary")
            st.info(result["summary"])

        with st.container(border=True):
            st.subheader("⑤ Q&A")
            question = st.text_input("Ask a question about this report")
            if st.button("Get answer") and question.strip():
                answer = run_qa(models, st.session_state.report_text, question)
                st.session_state.qa_pair = {
                    "question": question,
                    "answer": answer["answer"],
                    "score": answer["score"],
                }
            if st.session_state.qa_pair:
                qa = st.session_state.qa_pair
                st.success(f"**{qa['answer']}**  ({qa['score']:.0%} confidence)")
            else:
                st.caption("No question asked yet — this section won't appear in the PDF unless answered.")

        pdf_buf = generate_doctor_pdf(
            st.session_state.report_text,
            result,
            counterfactual=st.session_state.cf_result if st.session_state.get("cf_in_pdf") else None,
            qa_pair=st.session_state.qa_pair,
            patient_name=patient_name,
            report_date=report_date,
        )
        st.download_button(
            "⬇️ Download Clinical PDF Report",
            data=pdf_buf,
            file_name="medical_ai_analysis_clinical.pdf",
            mime="application/pdf",
        )

    else:  # Patient-Friendly
        buckets = {}
        for ent in result["entities"]:
            buckets.setdefault(bucket_entity(ent["entity_group"], ent["word"]), []).append(ent["word"].lower())
        symptoms = buckets.get("Symptoms", [])
        findings = buckets.get("Imaging Findings", []) + buckets.get("Other Clinical Mentions", [])

        with st.container(border=True):
            st.subheader("📄 Your Report Summary")
            if symptoms:
                st.write(f"The report mentions {', '.join(symptoms)}.")
            else:
                st.write(result["summary"])

        with st.container(border=True):
            st.subheader("🩺 Main Findings")
            if findings:
                st.write(f"The report also mentions {', '.join(findings)}.")
            else:
                st.write("No additional findings were highlighted.")

        with st.container(border=True):
            st.subheader("💬 Simple Explanation")
            st.write(
                f"The text was classified as being most consistent with a "
                f"{result['top_label'].lower()}-related condition."
            )

        st.markdown(f'<div class="disclaimer-box">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

        pdf_buf = generate_patient_pdf(st.session_state.report_text, result)
        st.download_button(
            "⬇️ Download Patient-Friendly PDF",
            data=pdf_buf,
            file_name="medical_ai_summary_patient.pdf",
            mime="application/pdf",
        )
else:
    st.info("Enter a report and click **Analyze** to run the full pipeline.")