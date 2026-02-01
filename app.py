import os
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, ImageMessage, TextSendMessage

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

import json
import os

google_creds = json.loads(os.getenv("GOOGLE_CREDENTIALS"))

creds = ServiceAccountCredentials.from_json_keyfile_dict(
    google_creds, scope
)

client = gspread.authorize(creds)

sheet = client.open_by_key(
    "SPREADSHEET_ID_ของคุณ"
).sheet1
def log_to_sheet(disease_name):
    now = datetime.now().strftime("%Y-%m-%d")

    sheet.append_row(
        ["" for _ in range(12)] + [now, disease_name]
    )

    print("✅ บันทึกลง Google Sheet:", disease_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 9)

checkpoint = torch.load(
    "mobilenetv2_chatbot.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])
class_names = checkpoint["class_names"]

model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
CONF_THRESHOLD = 85  # %

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item() * 100

    if confidence < CONF_THRESHOLD:
        return None, confidence

    disease = class_names[pred.item()]
    return disease, confidence
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return "OK"
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
            "📷 ไม่สามารถวิเคราะห์ภาพได้อย่างแม่นยำ\n"
            "กรุณาส่งภาพมะเขือเทศใหม่อีกครั้ง"
        )
    else:
        log_to_sheet(disease)
        reply = (
            f"🌱 พบโรค: {disease}\n"
            f"📊 ความมั่นใจ: {confidence:.2f}%"
        )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
