import os
import json
from datetime import datetime
from flask import Flask, request

# ===== LINE Bot =====
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

# ===== AI / Image =====
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ===== Google Sheet =====
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# =========================
# Flask App
# =========================
app = Flask(__name__)


# =========================
# LINE CONFIG (ENV)
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =========================
# GOOGLE SHEET CONFIG
# =========================
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

google_creds = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
google_creds["private_key"] = google_creds["private_key"].replace("\\n", "\n")

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, scope
)

client = gspread.authorize(creds)

# 👉 ใส่ Spreadsheet ID จริง
sheet = client.open_by_key(
    "PUT_YOUR_SPREADSHEET_ID_HERE"
).sheet1


def log_to_sheet(disease_name):
    now = datetime.now().strftime("%Y-%m-%d")
    sheet.append_row([""] * 12 + [now, disease_name])
    print("✅ บันทึกลง Google Sheet:", disease_name)


# =========================
# LOAD MODEL
# =========================
device = "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 9)

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "mobilenetv2_chatbot.pth")

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
class_names = checkpoint["class_names"]

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CONF_THRESHOLD = 85  # %


def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence

    disease = class_names[pred.item()]
    return disease, confidence


# =========================
# FLASK ROUTE
# =========================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return "OK"


# =========================
# LINE IMAGE HANDLER
# =========================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    content = line_bot_api.get_message_content(message_id)

    image_path = "input.jpg"
    with open(image_path, "wb") as f:
        for chunk in content.iter_content():
            f.write(chunk)

    disease, confidence = predict_image(image_path)

    if disease is None:
        reply = (
            "📷 ไม่สามารถวิเคราะห์ภาพได้อย่างแม่นยำ\n\n"
            "กรุณาส่งภาพมะเขือเทศใหม่อีกครั้ง "
            "โดยให้ภาพชัด เห็นใบหรืออาการผิดปกติ และมีแสงเพียงพอ 🙏"
        )
    else:
        log_to_sheet(disease)
        reply = (
            f"🌱 ผลการวิเคราะห์โรคมะเขือเทศ\n\n"
            f"🦠 โรคที่พบ: {disease}\n"
            f"📊 ความมั่นใจ: {confidence:.2f}%"
        )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# =========================
# RUN (สำหรับ Local เท่านั้น)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
