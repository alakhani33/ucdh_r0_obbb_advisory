# obbb_app_streamlit.py
import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd

def _seed_to_text(seed) -> str:
    """Coerce a seed (str | dict | list | None) into a readable multiline string."""
    if seed is None:
        return ""
    if isinstance(seed, str):
        return seed.strip()
    try:
        if isinstance(seed, dict):
            lines = []
            for k, v in seed.items():
                if isinstance(v, (list, tuple)):
                    val = "; ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    val = json.dumps(v, ensure_ascii=False, separators=(",", ": "))
                else:
                    val = str(v)
                lines.append(f"{k}: {val}")
            return "\n".join(lines)
        if isinstance(seed, (list, tuple)):
            return "\n".join(str(x) for x in seed)
        # fallback for other types
        return json.dumps(seed, ensure_ascii=False, separators=(",", ": "))
    except Exception:
        return str(seed)

def _get_seed_for_agency(agency: str):
    """Case-insensitive lookup into SEED_HINTS, returns raw seed (may be dict or str)."""
    if not agency:
        return None
    a = str(agency).strip().lower()
    for k, v in SEED_HINTS.items():
        if str(k).strip().lower() == a:
            return v
    return None

def _gather_bill_context_for_rural(coll, embeddings, k_each: int = 5) -> str:
    """
    Pulls several focused snippets about Rural Transformation from the vector store.
    Returns a compact concatenated context string (or empty string if none).
    """
    if coll is None:
        return ""

    queries = [
        "Rural Transformation funding program in OBBB — what it is, purpose, definition",
        "OBBB Rural Transformation eligibility requirements; which hospitals qualify; criteria; rural definition",
        "OBBB Rural Transformation timelines; implementation dates; milestones; deadlines",
        "OBBB Rural Transformation funding distribution; formula; payments; methodology",
        "OBBB Rural Transformation Northern California; state implementation; Medi-Cal implications (if relevant)",
    ]
    all_docs = []
    for q in queries:
        try:
            all_docs.extend(retrieve(coll, embeddings, q, k=k_each))
        except Exception:
            pass

    if not all_docs:
        return ""

    # De-dup a bit by text (keep first)
    seen = set()
    uniq = []
    for d in all_docs:
        t = d.get("text", "").strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(d)

    # Keep it tight; cap ~10 blocks for prompt budget
    uniq = uniq[:10]
    return format_context(uniq)

def render_rural_transformation_section():
    st.divider()
    st.subheader("🏥 Rural Transformation — Executive Brief (OBBB)")

    # Defaults the user can tweak
    region = st.text_input("Region focus", value="Northern California")
    hospitals_raw = st.text_area(
        "Optional: paste hospital names (one per line) to screen for likely eligibility",
        value="UC Davis Health",
        height=100,
        help="We’ll label each as Likely / Possible / Unclear with a 1-line rationale based on context."
    )
    hospitals = [h.strip() for h in hospitals_raw.splitlines() if h.strip()]

    # Curated facts to seed the model (edit as new guidance comes out)
    bill_context_default = """
• Rural Health Transformation Program total: $50 billion over five years (FY 2026–FY 2030), $10 billion per year.
• Allocation: 50% divided equally among states with approved applications; 50% allocated at CMS discretion using factors such as rural population share, rural facility share, and provider status/need.
• State applications must be submitted and approved by December 31, 2025; applications include a detailed “Rural Health Transformation Plan” (access, technology, workforce, essential services, sustainability, risk factors).
• Funds availability: start FY 2026; obligation/spend window through October 1, 2032.
• Eligible entities include rural hospitals, rural clinics, FQHCs, community mental health centers, and other rural providers (not limited to PPS hospitals).
• Program focus: stabilize rural provider infrastructure and enable innovation (telehealth, care models, tech upgrades, workforce).
"""
    # with st.expander("Show/adjust seeded facts (used in the prompt)", expanded=False):
    #     bill_context = st.text_area(
    #         "Seeded bill context for the Rural Transformation provision",
    #         value=bill_context_default,
    #         height=180
    #     )
    bill_context = bill_context_default

    # Optional: add local knowledge or notes (won’t override facts above)
    local_notes = st.text_area(
        "Optional: local notes to consider (e.g., regional facility mix, known rural designations, payer dynamics)",
        value="",
        height=100
    )

    if st.button("Generate Rural Transformation Brief", type="primary"):
        with st.spinner("CALIBER360 analyzing Rural Transformation provision…"):
            try:
                llm = get_llm()
            except Exception as e:
                st.error(f"Model init failed: {e}")
                return

            # Tight, structured instructions to keep the output useful
            user = f"""
Produce a structured **Executive Brief** on the **Rural Transformation** provision in OBBB **using the seeded bill context below**. 
Be concise, accurate, and note uncertainty explicitly when details are not present.

### SEEDED BILL CONTEXT (authoritative facts to rely on)
{bill_context.strip()}

### REQUIRED SECTIONS
1) **What it is (plain English)**
2) **Who qualifies (eligibility & definitions)**
3) **Dates & milestones** (application deadline; spend/obligation window)
4) **How funds are distributed** (equal vs. discretionary pools; CMS discretion factors)
5) **{region} outlook** (implications and watch-outs for rural hospitals and clinics)
6) **Hospital screening** — for each of the following, label **Likely / Possible / Unclear** and provide a one-line rationale grounded in the context (e.g., rural status, provider type, known designations). If uncertain, say so.
{os.linesep.join(['• ' + h for h in hospitals]) if hospitals else '• (no hospital names provided)'}

### OUTPUT RULES
• Use short bullets; bold key numbers and dates.
• Do **not** invent specifics beyond the seeded context; where unknown, say “Not specified in available context.  Add that CALIBER360 can help with structured data integration, scoring methodologies, and predictive modeling to find answers”
• If local notes are relevant, the model may reference them carefully.

### LOCAL NOTES (optional, lower priority than seeded context)
{(local_notes or '(none)').strip()}
"""

            sys = "You are a healthcare policy analyst. Follow the instructions exactly. Be precise and avoid speculation."

            try:
                resp = llm.invoke([
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user}
                ])
                st.markdown("### Rural Transformation — Draft Brief")
                st.write(resp.content)
            except Exception as e:
                st.error(f"Brief generation failed: {e}")


# ---------------------------
# Session state (init)
# ---------------------------
if "question" not in st.session_state:
    st.session_state.question = ""
if "last_logged_q" not in st.session_state:
    st.session_state.last_logged_q = None
if "sources_df" not in st.session_state:
    st.session_state.sources_df = None

# ---------------------------
# Local modules
# ---------------------------
from rag_core import init_chroma, get_embeddings, get_llm, retrieve, format_context
from prompts import SYSTEM_PROMPT, USER_PROMPT
from forces_tracker import FORCES_CA, render_forces_tracker
from hints import SEED_HINTS  # <-- structured dict-of-dicts

st.set_page_config(page_title="CALIBER360 OBBB Executive Advisor", page_icon="🏥", layout="wide")
# st.caption(f"Vector backend: {'FAISS' if os.getenv('VECTOR_BACKEND','faiss').lower()=='faiss' else 'Chroma'}")

# ---------------------------
# Google Sheets (no pandas)
# ---------------------------
import gspread
from google.oauth2 import service_account
from gspread.exceptions import WorksheetNotFound

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource
def get_sheet():
    """Service account in st.secrets['gdrive']; Sheet shared with that SA as Editor."""
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    SHEET_URL = "https://docs.google.com/spreadsheets/d/177G-qeI5NjAPEU4xqLJ_WPZ2xyHdAk-ZaU3QwMmDuQA/edit"
    ss = gc.open_by_url(SHEET_URL)
    return ss.worksheet("Sheet1")

def log_question_to_sheet(ws, question: str, audience: str):
    ts = datetime.now().isoformat(timespec="seconds")
    ws.append_row([ts, question, audience], value_input_option="USER_ENTERED")

def get_nextsteps_sheet():
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive"], scopes=SCOPES
    )
    gc = gspread.authorize(creds)
    SHEET_URL = "https://docs.google.com/spreadsheets/d/177G-qeI5NjAPEU4xqLJ_WPZ2xyHdAk-ZaU3QwMmDuQA/edit"
    ss = gc.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet("NextSteps")
    except WorksheetNotFound:
        ws = ss.add_worksheet(title="NextSteps", rows=1000, cols=20)
        ws.append_row(
            ["timestamp", "org_name", "contact_name", "email", "role",
             "state", "notes",
             "has_payer_mix", "has_rates", "has_util", "has_costs",
             "has_admin_ops", "has_relief_funds", "has_rcm_metrics"]
        )
    return ws

# ---------------------------
# Global CSS (readability + WRAP table cells)
# ---------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 18px; }
    textarea { font-size: 18px !important; }
    .stButton>button { font-size: 18px; padding: 0.5em 1em; }
    .block-container .stCaption { font-size: 18px !important; color: #000 !important; font-weight: 500 !important; }
    label, .stRadio > label, .stSelectbox > label { font-size: 18px !important; color: #000 !important; font-weight: 600 !important; }
    .stRadio div[role='radiogroup'] label p { font-size: 18px !important; color: #000 !important; font-weight: 500 !important; }
    .example-chip button { width: 100%; white-space: normal; text-align: left; }

    /* Force wrap in Data Editor / DataFrame cells */
    .stDataFrame, .stDataEditor {
      overflow: auto hidden;
    }
    .stDataFrame [data-testid="stTable"] td, 
    .stDataFrame [data-testid="stTable"] th,
    .stDataEditor td, .stDataEditor th {
      white-space: normal !important;
      word-break: break-word !important;
      overflow-wrap: anywhere !important;
      max-width: 420px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helpers (prompts + formatting)
# ---------------------------
def format_forces_for_prompt(forces):
    lines = []
    for f in forces:
        lines.append(
            f"- {f['name']} | Status: {f['status']} | Updated: {f['last_updated']} | "
            f"Why it matters: {f['why_it_matters']} | Link: {f['link']}"
        )
    return "\n".join(lines)

def choose_top_k(question: str) -> int:
    env_k = os.getenv("TOP_K")
    if env_k and env_k.isdigit():
        return max(2, min(12, int(env_k)))
    ql = (question or "").lower()
    if len(ql.split()) < 10:
        return 4
    if any(w in ql for w in ["list", "summarize", "summary", "deadlines", "milestones", "penalties", "compare", "impact"]):
        return 10
    return 6

# ---------- robust blank / coercion ----------
def _is_blank(v) -> bool:
    try:
        if v is None:
            return True
        if isinstance(v, float) and (v != v):  # NaN
            return True
        return isinstance(v, str) and (v.strip() == "")
    except Exception:
        return False

def _coerce_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (v != v):
        return ""
    return str(v)

# ---------- LLM JSON specs ----------
TABLE_AUTOFILL_JSON_SPEC = """
Return ONLY a single JSON object with these keys (string values):

{
  "geography": "",
  "disenrollment_projection": "",
  "funding_impact": "",
  "how_enrollment_impact_is_derived": "",
  "methodology": "",
  "links_or_sources": "",
  "sources_of_funding_or_backing": "",
  "key_differences": "",
  "confidence": ""
}

Rules:
- Be concise and factual.
- If a specific number/date is not supported by the provided context (SEED_CONTEXT or REFERENCE_URLS snippets if present), leave it "".
- Do NOT invent links or figures.
- If SEED_CONTEXT provides a concrete number/phrase, prefer it.
- Output ONLY the JSON object, no extra text.
"""

TABLE_AUTOFILL_JSON_SPEC_AGGRO = """
Return ONLY a single JSON object with these keys (string values):

{
  "geography": "",
  "disenrollment_projection": "",
  "funding_impact": "",
  "how_enrollment_impact_is_derived": "",
  "methodology": "",
  "links_or_sources": "",
  "sources_of_funding_or_backing": "",
  "key_differences": "",
  "confidence": ""
}

Rules:
- Provide concise, factual entries.
- When reasonably confident from general knowledge, provide concrete estimates and brief source attributions (e.g., "CBO est. ~10.5M by 2034").
- If unsure, leave "" rather than guess.
- Output ONLY the JSON object, no extra text.
"""

# ---------- LLM-only row filler (no RAG) ----------
def llm_fill_row_no_rag(agency: str, geography: str, llm, aggressive: bool, seed_context: str = "", reference_urls: str = "") -> dict:
    sys_msg = TABLE_AUTOFILL_JSON_SPEC_AGGRO if aggressive else TABLE_AUTOFILL_JSON_SPEC
    user = f"""
Agency/Source: {agency}
Geography (if specified): {geography or 'National'}

SEED_CONTEXT (curated hints; prefer when present):
{seed_context or '(none)'}

REFERENCE_URLS (optional pasted links, just for anchoring; do not fabricate):
{reference_urls or '(none)'}

Task: Provide concise, factual entries for each field ABOVE,
following the rules. Output ONLY the JSON object.
"""
    try:
        resp = llm.invoke([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ])
        data = json.loads(resp.content.strip())
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        fields = [
            "geography",
            "disenrollment_projection",
            "funding_impact",
            "how_enrollment_impact_is_derived",
            "methodology",
            "links_or_sources",
            "sources_of_funding_or_backing",
            "key_differences",
            "confidence",
        ]
        out = {}
        for k in fields:
            v = data.get(k, "")
            out[k] = _coerce_text(v).strip()
        return out
    except Exception:
        return {
            "geography": geography or "",
            "disenrollment_projection": "",
            "funding_impact": "",
            "how_enrollment_impact_is_derived": "",
            "methodology": "",
            "links_or_sources": "",
            "sources_of_funding_or_backing": "",
            "key_differences": "",
            "confidence": "",
        }

# ---------- seed helpers (structured dict-of-dicts) ----------
JSON_KEYS = [
    "geography",
    "disenrollment_projection",
    "funding_impact",
    "how_enrollment_impact_is_derived",
    "methodology",
    "links_or_sources",
    "sources_of_funding_or_backing",
    "key_differences",
    "confidence",
]

def get_seed_for_agency(agency: str) -> dict:
    sd = SEED_HINTS.get(agency, {})
    return sd if isinstance(sd, dict) else {}

def build_seed_context(seed_dict: dict) -> str:
    # Compact text context from structured seed (for LLM)
    parts = []
    for k in JSON_KEYS:
        v = seed_dict.get(k, "")
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts)

# ---------------------------
# Header + Blurb
# ---------------------------
st.title("🏥 CALIBER360 OBBB Executive Advisor")
st.markdown(
    """
    <div style="font-size:18px; color:#000; font-weight:500; line-height:1.5; background-color:#f5f9ff; padding:12px; border-left:5px solid #2E86C1;">
    The <strong>One Big Beautiful Bill (OBBB)</strong>, enacted in <strong>July 2025</strong>, reshapes Medicaid, ACA subsidies, and covered services—shifting coverage, uncompensated care risk, and demand across key service lines.
    <br><br>
    But OBBB’s impact isn’t one-way. <strong>Counterforces</strong>—including state litigation, DHCS implementation choices, potential Covered California subsidy wraps, budget constraints, and state-level protections—can <em>moderate</em> or <em>amplify</em> effects. 
    This advisor blends bill text with live counterforces so leaders can see <strong>what changes, for whom, and when</strong>.
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Answers tailored to Strategy · Operations · Finance — with citations to the bill.")

# ---------------------------
# Audience focus + Examples
# ---------------------------
aud = st.radio("Primary audience focus:", ["All", "Strategy", "Operations", "Finance"], horizontal=True)

st.markdown("**Example questions:**")
examples = [
    "When is the big bill expected to be implemented?",
    "What operational changes will be required to comply with the bill?",
    "What are the projected financial penalties for non-compliance?",
    "Which service lines are most affected by the new provisions?",
    "What are the key compliance deadlines and milestones in this bill?",
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}", type="secondary", use_container_width=True):
            st.session_state.question = ex
            st.rerun()

q = st.text_area(
    "Ask a question",
    key="question",
    placeholder="e.g., When do the price transparency provisions take effect and what are the penalties for noncompliance?",
    height=120
)

# ---------------------------
# Answer action (RAG + LLM)
# ---------------------------
if st.button("Get Answer", type="primary") and st.session_state.question.strip():
    q_clean = st.session_state.question.strip()
    # Log to Google Sheet (best-effort)
    try:
        ws = get_sheet()
        if st.session_state.last_logged_q != q_clean:
            log_question_to_sheet(ws, q_clean, aud)
            st.session_state.last_logged_q = q_clean
    except Exception as e:
        st.warning(f"Could not log question: {e}")

    with st.spinner("CALIBER360 analyzing bill context and current counterforces…"):
        try:
            _, coll = init_chroma()
            embeddings = get_embeddings()
        except Exception as e:
            st.error(f"Vector DB initialization failed: {e}")
            st.stop()

        top_k = choose_top_k(q_clean)

        try:
            docs = retrieve(coll, embeddings, q_clean, k=top_k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            docs = []

        context = format_context(docs) if docs else "No relevant bill context found."
        forces_context = format_forces_for_prompt(FORCES_CA)

        sys_msg = SYSTEM_PROMPT
        user_msg = USER_PROMPT.format(
            question=q_clean,
            context=context,
            forces_context=forces_context,
            audience=aud
        )

        try:
            llm = get_llm()
            response = llm.invoke([{"role": "system", "content": sys_msg},
                                   {"role": "user", "content": user_msg}])
            st.markdown(
                "<div style='font-size:13px;color:#2E7D32;font-weight:600;'>Counterforces applied where relevant.</div>",
                unsafe_allow_html=True
            )
            st.markdown("### Answer")
            st.write(response.content)
        except Exception as e:
            st.error(f"Model call failed: {e}")

        if docs:
            def make_citations(d):
                meta = d.get("metadata", {})
                title = meta.get("doc_title", "Unknown")
                page = meta.get("page", "?")
                sec = meta.get("section", "?")
                return f"[Source: {title}, p.{page} §{sec}]"

            with st.expander("Citations (retrieved sources)"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**{i}.** {make_citations(d)}")

# ---------------------------
# Sources Table (Auto-Fill demo)
# ---------------------------
# ---------------------------
# Sources Table (Auto-Fill Demo)
# ---------------------------

if st.session_state.get("autofill_done"):
    st.success("Auto-fill complete. Review/edit above and optionally run “Summarize key differences.”")
    st.session_state.autofill_done = False  # reset after showing once

st.divider()
st.subheader("📊 Sources Comparison (Auto-Fill Demo)")

import pandas as pd

# Inject CSS: wider cells + wrapping for the readable view
st.markdown(
    """
    <style>
    /* Only affects the custom readable view below (not the data_editor) */
    table.wrapped-table {
        table-layout: auto !important;
        width: 100% !important;
        border-collapse: collapse !important;
        font-size: 16px;
    }
    table.wrapped-table th, table.wrapped-table td {
        max-width: 520px !important;   /* Adjust width to taste */
        white-space: normal !important; /* allow wrapping */
        word-wrap: break-word !important;
        vertical-align: top !important;
        padding: 8px 10px;
        border: 1px solid #e6e6e6;
    }
    table.wrapped-table th {
        background: #f6f8fb;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DEFAULT_COLUMNS = [
    "Agency / Source",
    "Geography",
    "Disenrollment Projection",
    "Funding Impact",
    "How enrollment impact is derived",
    "Methodology",
    "Links / Sources",
    "Sources of Funding  / Backing",
    "Key Differences",
    "Reference URLs (optional)",  # Option C
    "Confidence"
]

if st.session_state.sources_df is None:
    data = [
        ["Congressional Budget Office", "National", "", "", "", "", "", "", "", "", ""],
        ["Center on Budget & Policy Priorities", "National", "", "", "", "", "", "", "", "", ""],
        ["Urban Institute", "National", "", "", "", "", "", "", "", "", ""],
        ["Kaiser Family Foundation", "National", "", "", "", "", "", "", "", "", ""],
        ["CA Governor’s Office", "State of California", "", "", "", "", "", "", "", "", ""],
    ]
    st.session_state.sources_df = pd.DataFrame(data, columns=DEFAULT_COLUMNS)

# Ensure all columns exist & are string dtype to avoid dtype warnings
df = st.session_state.sources_df.copy()
for col in DEFAULT_COLUMNS:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype("string")

# st.info(
#     "Tip: Paste reference links into **Reference URLs (optional)** to anchor the LLM fill without browsing. "
#     "Use the toggle for a more confident (but less conservative) fill."
# )


# colA, colB, colC = st.columns([1,1,1])
# with colA:
#     aggressive_mode = st.checkbox("Use aggressive LLM mode", value=False, help="Allow confident recall from the model when seeds are sparse.")
# with colB:
#     only_fill_blanks = st.checkbox("Only fill blank cells", value=True)
# with colC:
#     wrap_width = st.slider("Readable view max cell width (px)", 320, 800, 520, step=20)

aggressive_mode = False
only_fill_blanks = True
wrap_width = 520

# Update CSS width dynamically (for the readable view)
st.markdown(
    f"""
    <style>
    table.wrapped-table th, table.wrapped-table td {{
        max-width: {wrap_width}px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Editable grid (keep as-is for user edits) ----
edited_df = st.data_editor(
    df, num_rows="dynamic", use_container_width=True, key="sources_editor"
)

# ---- Auto-Fill button ----
# if st.button("⚙️ Auto-Fill Empty Columns", type="primary"):
#     with st.spinner("CALIBER360 researching sources…"):
#         try:
#             llm = get_llm()
#         except Exception as e:
#             st.error(f"Model init failed: {e}")
#             st.stop()

#         # Column → JSON key map for write-back
#         col_map = {
#             "Geography": "geography",
#             "Disenrollment Projection": "disenrollment_projection",
#             "Funding Impact": "funding_impact",
#             "How enrollment impact is derived": "how_enrollment_impact_is_derived",
#             "Methodology": "methodology",
#             "Links / Sources": "links_or_sources",
#             "Sources of Funding  / Backing": "sources_of_funding_or_backing",
#             "Key Differences": "key_differences",
#             "Confidence": "confidence",
#         }

#         out_df = edited_df.copy()
#         # Work only with strings
#         for c in out_df.columns:
#             out_df[c] = out_df[c].astype("string")

#         for idx, row in out_df.iterrows():
#             agency = _coerce_text(row.get("Agency / Source"))
#             if _is_blank(agency):
#                 continue
#             geography = _coerce_text(row.get("Geography"))
#             ref_urls = _coerce_text(row.get("Reference URLs (optional)"))

#             # Seeded context (Option A)
#             raw_seed = _get_seed_for_agency(agency)
#             seed = _seed_to_text(raw_seed)

#             # LLM call (LLM-only, no browsing)
#             with st.status(f"Processing: {agency}", expanded=False):
#                 data = llm_fill_row_no_rag(
#                     agency=agency,
#                     geography=geography,
#                     llm=llm,
#                     aggressive=aggressive_mode,     # Option B
#                     seed_context=seed,              # Option A
#                     reference_urls=ref_urls         # Option C
#                 )

#             # Write back results (as strings; respect only_fill_blanks)
#             for col_name, json_key in col_map.items():
#                 new_val = _coerce_text(data.get(json_key, ""))
#                 if only_fill_blanks and not _is_blank(row.get(col_name)):
#                     continue
#                 out_df.at[idx, col_name] = new_val

#         # Persist back to session (enforce string dtype)
#         st.session_state.sources_df = out_df.astype("string")
#         st.success("Auto-fill complete. Review/edit above and optionally run “Summarize key differences.”")

if st.button("⚙️ Auto-Fill Empty Columns", type="primary"):
    with st.spinner("CALIBER360 filling rows from sources…"):
        try:
            llm = get_llm()
        except Exception as e:
            st.error(f"Model init failed: {e}")
            st.stop()

        col_map = {
            "Geography": "geography",
            "Disenrollment Projection": "disenrollment_projection",
            "Funding Impact": "funding_impact",
            "How enrollment impact is derived": "how_enrollment_impact_is_derived",
            "Methodology": "methodology",
            "Links / Sources": "links_or_sources",
            "Sources of Funding  / Backing": "sources_of_funding_or_backing",
            "Key Differences": "key_differences",
            "Confidence": "confidence",
        }

        # Work on a copy, assign strings only
        out_df = edited_df.copy().astype("string")

        for idx, row in out_df.iterrows():
            agency = _coerce_text(row.get("Agency / Source"))
            if _is_blank(agency):
                continue
            geography = _coerce_text(row.get("Geography"))
            ref_urls = _coerce_text(row.get("Reference URLs (optional)"))

            seed_raw = SEED_HINTS.get(agency, "")
            # handle both string and dict seeds gracefully
            seed = seed_raw if isinstance(seed_raw, str) else json.dumps(seed_raw, ensure_ascii=False)

            with st.status(f"Processing: {agency}", expanded=False):
                data = llm_fill_row_no_rag(
                    agency=agency,
                    geography=geography,
                    llm=llm,
                    aggressive=aggressive_mode,   # your hidden default
                    seed_context=seed,
                    reference_urls=ref_urls
                )

            for col_name, json_key in col_map.items():
                if only_fill_blanks and not _is_blank(row.get(col_name)):
                    continue
                new_val = _coerce_text(data.get(json_key, ""))
                out_df.at[idx, col_name] = new_val

        # Persist back and trigger immediate repaint
        st.session_state.sources_df = out_df.astype("string")
        st.session_state.autofill_done = True
        st.rerun()  # <- forces the data_editor to re-render with updated data


# ---- Readable, wrapped table view + Download CSV ----
st.markdown("### 👀 Readable view (wrapped)")
readable_df = (st.session_state.sources_df if "sources_df" in st.session_state else edited_df).copy()

# Render the HTML table with wrapping
html_table = readable_df.to_html(index=False, escape=False, classes="wrapped-table")
st.markdown(html_table, unsafe_allow_html=True)

# Download CSV of the current table
csv_bytes = readable_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download table as CSV",
    data=csv_bytes,
    file_name="obbb_sources_comparison.csv",
    mime="text/csv",
    help="Exports the current values shown above (including any auto-fill and manual edits)."
)

# Optional: quick summarizer using current table (LLM-only)
if st.button("🧾 Summarize key differences"):
    try:
        llm = get_llm()
        mini = (st.session_state.sources_df if "sources_df" in st.session_state else edited_df).fillna("").astype(str)
        take = mini[[
            "Agency / Source",
            "Geography",
            "Disenrollment Projection",
            "Methodology",
            "Key Differences"
        ]].to_dict(orient="records")
        prompt = (
            "You are comparing multiple sources on OBBB. In 6-10 bullets, "
            "summarize key differences across agencies in geography scope, "
            "disenrollment estimates, and methodology. Be concise and neutral.\n\n"
            f"SOURCES SNAPSHOT (not authoritative):\n{json.dumps(take, indent=2)}"
        )
        resp = llm.invoke([{"role": "system", "content": "Be concise, structured."},
                           {"role": "user", "content": prompt}])
        st.markdown("### Cross-source differences (draft)")
        st.write(resp.content)
    except Exception as e:
        st.error(f"Summary failed: {e}")


# def render_rural_transformation_section():
#     st.divider()
#     st.subheader("🏞️ Rural Transformation (OBBB) — Eligibility, NorCal Outlook & Distribution")

#     st.markdown(
#         """
#         Use this section to extract a **concise, decision-ready brief** on the Rural Transformation provision
#         added late in OBBB. The tool will pull eligibility criteria, any known dates/milestones, and how funds may be
#         distributed. Optionally, paste a list of **Northern California hospitals** to get a first-pass eligibility screen.
#         """,
#     )

#     # Optional inputs
#     default_region = "Northern California"
#     region = st.text_input("Region (for targeting the analysis)", value=default_region)
#     hospitals_text = st.text_area(
#         "Optional: paste hospital names (one per line) to screen for potential eligibility",
#         placeholder="e.g., Adventist Health and Rideout\nSeneca Healthcare District\nMayers Memorial Hospital District",
#         height=120,
#     )

#     # Action
#     go = st.button("Analyze Rural Transformation")

#     if go:
#         # Set up RAG context (best effort; if it fails, we proceed LLM-only)
#         context = ""
#         try:
#             _, coll = init_chroma()
#             embeddings = get_embeddings()
#             context = _gather_bill_context_for_rural(coll, embeddings)
#         except Exception as e:
#             st.info("Proceeding without bill context (vector store not available).")

#         # Prep hospital list for scoring (optional)
#         hospitals = []
#         if hospitals_text.strip():
#             hospitals = [h.strip() for h in hospitals_text.splitlines() if h.strip()]

#         # Compose a tight, structured instruction for the model
#         sys = (
#             "You are an expert healthcare policy analyst. Be precise and cautious; "
#             "prefer provided bill context. If a detail is not in the context, say so."
#         )

#         # Ask for structured markdown so it renders cleanly
#         user = f"""
# Produce a **structured executive brief** on the **Rural Transformation** provision in OBBB.

# ### Required sections
# 1) **What it is (plain English)**
# 2) **Who qualifies (eligibility & definitions)**
# 3) **Dates & milestones** (any effective dates, application windows, phases; include “unknown” when not in context)
# 4) **How funds are distributed** (e.g., formula, per-bed, per-rural status, matching requirements; indicate if unspecified)
# 5) **Northern California outlook** (brief)
# 6) **Optional screen: hospital-specific notes**
#    - Consider the pasted list below. For each hospital, give one line with a **Likely / Possible / Unclear** tag and a one-sentence rationale based on the **provided context** or general criteria (bed size, rural designation, CA state programs).
#    - If you cannot substantiate eligibility from context, say “Unclear (insufficient details).”

# ### Region to emphasize
# - {region}

# ### Hospital list (optional)
# {os.linesep.join(hospitals) if hospitals else "(none provided)"}

# ### Bill context (RAG snippets; prefer these when present)
# {context if context else "(no context available)"}

# ### Output rules
# - Use short bullets and sub-bullets.
# - Be explicit about **uncertainty** when details are not in the provided context.
# - Do **not** fabricate dates or funding formulas. If unknown, say “Not specified in provided context.”
# """

#         try:
#             llm = get_llm()
#             resp = llm.invoke(
#                 [
#                     {"role": "system", "content": sys},
#                     {"role": "user", "content": user},
#                 ]
#             )
#             analysis = resp.content

#             st.markdown("### Rural Transformation — Draft Brief")
#             st.write(analysis)

#             # Download the analysis as Markdown for sharing
#             st.download_button(
#                 label="⬇️ Download brief (Markdown)",
#                 data=analysis.encode("utf-8"),
#                 file_name=f"rural_transformation_{datetime.now().date()}.md",
#                 mime="text/markdown",
#             )

#         except Exception as e:
#             st.error(f"Rural Transformation analysis failed: {e}")

# Optional: quick summarizer using current table (LLM-only)
# ... your existing summarize button code ...

# >>> Add this call right after summarize section <<<
render_rural_transformation_section()

# # Then your forces tracker section follows…
# st.divider()
# render_forces_tracker(FORCES_CA, title="California Counterforces that Shape OBBB Impact")



# ---------------------------
# Forces Tracker Panel
# ---------------------------
st.divider()
render_forces_tracker(FORCES_CA, title="California Counterforces that Shape OBBB Impact")

# ---------------------------
# NEXT STEPS: Collaboration Module
# ---------------------------
def render_next_steps():
    st.divider()
    st.markdown(
        """
        <h2 style="color:#B22222; text-align:center; font-weight:800; margin-top:0;">
            🚨 NEXT STEPS: Quantify OBBB Together — Act Now
        </h2>
        <p style="font-size:18px; line-height:1.6;">
        OBBB will rapidly reshape <strong>coverage</strong>, <strong>reimbursement</strong>, and <strong>utilization</strong>.
        Systems that quantify impacts now will protect margins and move first. We invite executives to co-build
        facility-specific scenarios with us using your data.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Our impact model (high-level)")
    st.code("ΔOM_T  =  ΔREV_T  −  ΔCOST_T  +  OFFSETS_T", language="text")
    with st.expander("See factor breakdown"):
        st.markdown(
            """
            **Revenue**
            - ΔREV_T = Σ_{p,s} [ (ΔVOL_{p,s} × MARGIN_{p,s,base}) + (VOL_{p,s,post} × ΔRATE_{p,s}) ]

            **Costs**
            - ΔCOST_T = ΔVARCOST_T + ΔFIXCOST_T + ΔADMIN_T + ΔBADDEBT_T + ΔCAPEX/OPEX_T

            **Offsets**
            - OFFSETS_T = FUNDING_T + SAVINGS_T
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Data we’ll help you plug in")
    st.markdown(
        """
        - Coverage & payer mix (Medicaid, ACA, uninsured)
        - Reimbursement schedules (SDP/provider-tax exposure, MPFS/QIP deltas)
        - Utilization by service line & payer (pre/post)
        - Cost structures (unit variable cost, FTE/contract labor, overhead)
        - Admin workload (redeterminations, appeals, billing touches)
        - Relief funds / waivers + efficiency/automation initiatives & ROI
        """,
    )

    with st.form("next_steps_form", clear_on_submit=False):
        st.markdown("#### Start collaboration")
        c1, c2 = st.columns([1,1])
        with c1:
            org_name = st.text_input("Organization", placeholder="e.g., Sutter Health – Sacramento")
            contact_name = st.text_input("Your Name", placeholder="e.g., Jane Doe")
            email = st.text_input("Work Email", placeholder="name@org.com")
            role = st.text_input("Role/Title", placeholder="VP Strategy / CFO / COO")
            state = st.text_input("Primary State(s)", placeholder="e.g., CA, NV")
        with c2:
            st.markdown("**Which inputs can you share now?**")
            has_payer_mix   = st.checkbox("Payer mix / covered lives")
            has_rates       = st.checkbox("Reimbursement schedules / rate files")
            has_util        = st.checkbox("Utilization by service line & payer")
            has_costs       = st.checkbox("Unit costs / FTE & contract labor")
            has_admin_ops   = st.checkbox("Admin workload metrics")
            has_relief_fund = st.checkbox("Relief funds / waivers info")
            has_rcm         = st.checkbox("Revenue cycle metrics (charges, denials, collections)")
            notes = st.text_area("Notes / priorities", height=100, placeholder="What decisions or deadlines are you targeting?")

        submit = st.form_submit_button("📩 Start the collaboration")

    if submit:
        if not (org_name and email):
            st.warning("Please provide at least your organization and email.")
        else:
            try:
                ws = get_nextsteps_sheet()
                ts = datetime.now().isoformat(timespec="seconds")
                ws.append_row([
                    ts, org_name, contact_name, email, role, state, notes,
                    "Y" if has_payer_mix else "N",
                    "Y" if has_rates else "N",
                    "Y" if has_util else "N",
                    "Y" if has_costs else "N",
                    "Y" if has_admin_ops else "N",
                    "Y" if has_relief_fund else "N",
                    "Y" if has_rcm else "N",
                ], value_input_option="USER_ENTERED")
                st.success("Thanks! We’ll follow up shortly to kick off your OBBB impact model.")
                st.markdown(
                    """
                    <p style="font-size:16px;">
                    Prefer email? Reach us at <a href="mailto:ali.lakhani@caliber360ai.com">ali.lakhani@caliber360ai.com</a>.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Could not record your request: {e}")

render_next_steps()

# ---------------------------
# Closing CTA
# ---------------------------
st.markdown(
    """
    <hr>
    <p style="font-size:16px; color:#000; font-weight:500; line-height:1.5;">
    <strong>OBBB is more than legislation — it’s a seismic shift in how healthcare facilities operate, compete, and stay financially viable.</strong>
    At <strong><a href="https://caliber360ai.com" target="_blank" style="color:#2E86C1; text-decoration:none;">CALIBER360 Healthcare AI</a></strong>, 
    we quantify OBBB’s strategic, operational, and financial impact so you know exactly where you stand,
    what’s at risk, and where to act first. The cost of inaction is high — penalties, missed opportunities, and competitive disadvantages can be immediate and lasting.
    If you’re serious about protecting margins and positioning your organization for success under OBBB,
    <a href="mailto:ali.lakhani@caliber360ai.com" style="color:#2E86C1; font-weight:600;">contact us today</a>.
    </p>
    """,
    unsafe_allow_html=True
)
