import os
import re
import ssl
import smtplib
import pandas as pd
import pdfplumber
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import gradio as gr

# ===========================
# ✅ Hugging Face API Setup
# ===========================
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "tiiuae/falcon-7b-instruct"  # Can switch to falcon-1b-instruct for lower latency
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def hf_generate(prompt, max_new_tokens=200, temperature=0.0):
    """Call Hugging Face Inference API for text generation."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature}
    }
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return str(result)
    except Exception as e:
        return f"⚠️ LLM generation failed: {e}"

LOGO_PATH = "logo drivool.png"

def header(canvas, doc):
    canvas.saveState()
    if os.path.exists(LOGO_PATH):
        try:
            canvas.drawImage(LOGO_PATH, doc.leftMargin, A4[1] - 60,
                             width=120, height=40,
                             preserveAspectRatio=True, mask='auto')
        except:
            pass
    canvas.restoreState()

def clean_generation(raw_output, snippets_to_strip=None):
    text = raw_output or ""
    if snippets_to_strip:
        for s in snippets_to_strip:
            if not s:
                continue
            text = text.replace(s, "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and re.match(r'^(write|based|data|generate|please)', lines[0].lower()):
        lines = lines[1:]
    if len(lines) == 1:
        lines[0] = re.sub(r'^(Write|Based on).*?:\s*', '', lines[0], flags=re.IGNORECASE)
    return "\n".join(lines).strip()

def split_into_bullets(text):
    bullets = []
    for raw_line in text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = re.split(r"(?:^|\s)(?:[-•]\s+|\d+\.\s+)", raw_line)
        for part in parts:
            part = part.strip()
            if part:
                bullets.append(part)
    return bullets

def generate_llm_text(section, df):
    """Generate a specific section using Hugging Face API and return cleaned text."""
    base_context = f"Data shows {len(df)} incidents across {df['Project Code'].nunique()} sites and criteria counts: {df['Criteria'].value_counts().to_dict()}."

    if section == "overview":
        prompt = ("Write a short professional OVERVIEW paragraph (2-3 sentences) "
                  "summarizing the incident period. Respond ONLY with the paragraph.")
        kwargs = dict(max_new_tokens=120, temperature=0.0)

    elif section == "observations":
        prompt = ("Write exactly 3 key security observations as separate lines, each starting with '- '. "
                  "Keep each observation concise (one sentence). Respond ONLY with those 3 lines.")
        kwargs = dict(max_new_tokens=160, temperature=0.0)

    elif section == "recommendations":
        prompt = ("Write 4 recommendations. Each recommendation MUST start with '- ' on its own line. "
                  "If you include sub-points for a recommendation, put them on the next lines starting with '  • '. "
                  "Respond ONLY with the bullets and sub-bullets.")
        kwargs = dict(max_new_tokens=220, temperature=0.0)

    elif section == "conclusion":
        prompt = ("Write a short professional CONCLUSION paragraph (2-3 sentences) summarizing security posture "
                  "and next steps. Respond ONLY with the paragraph.")
        kwargs = dict(max_new_tokens=100, temperature=0.0)

    else:
        return ""

    full_prompt = base_context + "\n\n" + prompt
    raw_resp = hf_generate(full_prompt, **kwargs)
    return clean_generation(raw_resp, snippets_to_strip=[base_context, prompt, full_prompt])

def generate_debrief(file):
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip()

        try:
            df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format="%d-%b-%y")
        except:
            df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], dayfirst=True)

        start_date = df['Date of Incident'].min().strftime("%d %B %Y")
        end_date = df['Date of Incident'].max().strftime("%d %B %Y")

        category_points = [f"• <b>{crit}</b>: {len(group)} cases ({', '.join(group['Project Code'].unique())})"
                           for crit, group in df.groupby("Criteria")]

        site_points = []
        for site, group in df.groupby("Project Code"):
            site_points.append(f"• <b>{site}</b>: {len(group)} incidents")
            for _, row in group.iterrows():
                site_points.append(f"&nbsp;&nbsp;&nbsp;• {row['Criteria']} ({row['Location / Place of Incident']})")

        # Generate sections via HF API
        overview_text = generate_llm_text("overview", df) or f"{len(df)} incidents across {df['Project Code'].nunique()} sites were reviewed this period."
        obs_bullets = split_into_bullets(generate_llm_text("observations", df)) or ["No key observations identified."]
        rec_bullets = split_into_bullets(generate_llm_text("recommendations", df)) or ["No recommendations provided."]
        conclusion_text = generate_llm_text("conclusion", df) or "Security posture remains under monitoring; follow-up actions recommended."

        pdf_file = "security_debrief.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=90, bottomMargin=50)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="TitleCenter", parent=styles['Title'], alignment=TA_CENTER, fontSize=18, spaceAfter=12))
        styles.add(ParagraphStyle(name="SectionHeader", parent=styles['Heading2'], fontSize=14, spaceBefore=12, spaceAfter=8, textColor="#003366"))
        styles.add(ParagraphStyle(name="BulletCustom", parent=styles['Normal'], fontSize=11, leading=16, leftIndent=20))
        styles.add(ParagraphStyle(name="BodyTextCustom", parent=styles['Normal'], fontSize=11, leading=16, alignment=TA_LEFT))

        story = []
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Executive Security Debrief</b>", styles["TitleCenter"]))
        story.append(Paragraph(f"<b>Period:</b> {start_date} – {end_date}", styles["BodyTextCustom"]))
        story.append(Paragraph("<b>Prepared for:</b> Security Leadership Team", styles["BodyTextCustom"]))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Overview", styles["SectionHeader"]))
        story.append(Paragraph(overview_text, styles["BodyTextCustom"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Incident Summary by Category", styles["SectionHeader"]))
        for point in category_points:
            story.append(Paragraph(point, styles["BulletCustom"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Incident Summary by Site", styles["SectionHeader"]))
        for point in site_points:
            story.append(Paragraph(point, styles["BulletCustom"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Key Observations", styles["SectionHeader"]))
        for bullet in obs_bullets:
            story.append(Paragraph(f"• {bullet}", styles["BulletCustom"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Recommendations", styles["SectionHeader"]))
        for bullet in rec_bullets:
            story.append(Paragraph(f"• <font color='#003366'><b>{bullet}</b></font>", styles["BulletCustom"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Conclusion", styles["SectionHeader"]))
        story.append(Paragraph(conclusion_text, styles["BodyTextCustom"]))

        doc.build(story, onFirstPage=header, onLaterPages=header)
        return "✅ Debrief Generated Successfully", pdf_file

    except Exception as e:
        return f"❌ Error generating debrief: {str(e)}", None

def extract_text_from_pdf(pdf_path):
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n\n"
    return text_content.strip()

def send_email_with_pdf(receiver_email, pdf_path, full_report_text):
    try:
        sender_email = os.getenv("EMAIL_USER", "safetybot137@gmail.com")
        password = os.getenv("EMAIL_PASS", "yutu jkio dere zacv")

        message = MIMEMultipart("alternative")
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "Security Debrief Report"

        html_text = full_report_text
        for section in ["Overview", "Incident Summary by Category", "Incident Summary by Site",
                        "Key Observations", "Recommendations", "Conclusion"]:
            html_text = html_text.replace(section, f"<b>{section}</b>")

        html_body = f"""
        <html>
        <body style="font-family:Arial, sans-serif;">
        <h2 style="color:#003366;">Executive Security Debrief</h2>
        <pre style="white-space:pre-wrap; font-family:Arial;">{html_text}</pre>
        <hr>
        <p><i>This report is auto-generated. Please refer to the attached PDF for the official formatted version.</i></p>
        </body>
        </html>
        """
        message.attach(MIMEText(html_body, "html"))

        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment; filename=security_debrief.pdf")
            message.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

        return f"✅ Email sent to {receiver_email}"

    except Exception as e:
        return f"❌ Failed to send email: {e}"

def gradio_app(file, email):
    result, pdf_path = generate_debrief(file)
    if not pdf_path:
        return "❌ Failed to generate PDF", None

    full_report_text = extract_text_from_pdf(pdf_path)
    email_status = None
    if email and "@" in email:
        email_status = send_email_with_pdf(email, pdf_path, full_report_text)

    return (email_status or "✅ Report generated"), pdf_path

with gr.Blocks() as demo:
    gr.Markdown("## 🛡️ Security Debrief Report Generator")
    with gr.Row():
        file_input = gr.File(label="Upload Incident Data (Excel)", type="filepath")
        email_input = gr.Textbox(label="Recipient Email (Optional)", placeholder="Enter email to send report")
    generate_btn = gr.Button("Generate Report")
    output_msg = gr.Textbox(label="Status")
    file_output = gr.File(label="Download PDF", type="filepath")
    
    generate_btn.click(fn=gradio_app, inputs=[file_input, email_input], outputs=[output_msg, file_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))

