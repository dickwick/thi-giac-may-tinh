import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import json

# load css
def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.markdown("<div class='title'>🐾 Nhận diện động vật AI</div>",unsafe_allow_html=True)

st.write("Tải ảnh động vật để AI nhận diện")

# load classes

with open("classes.json") as f:
    classes = json.load(f)

# bảng dịch 100 animals

animal_vi = {
"alligator":"Cá sấu Mỹ",
"ant":"Kiến",
"bear":"Gấu",
"bee":"Ong",
"butterfly":"Bướm",
"camel":"Lạc đà",
"cat":"Mèo",
"cheetah":"Báo gêpa",
"chicken":"Gà",
"cobra":"Rắn hổ mang",
"cow":"Bò",
"crab":"Cua",
"crow":"Quạ",
"deer":"Hươu",
"dog":"Chó",
"dolphin":"Cá heo",
"duck":"Vịt",
"eagle":"Đại bàng",
"elephant":"Voi",
"fox":"Cáo",
"frog":"Ếch",
"giraffe":"Hươu cao cổ",
"goat":"Dê",
"goldfish":"Cá vàng",
"gorilla":"Khỉ đột",
"hamster":"Chuột hamster",
"horse":"Ngựa",
"jaguar":"Báo đốm",
"kangaroo":"Chuột túi",
"koala":"Koala",
"lion":"Sư tử",
"lobster":"Tôm hùm",
"monkey":"Khỉ",
"octopus":"Bạch tuộc",
"owl":"Cú",
"panda":"Gấu trúc",
"parrot":"Vẹt",
"penguin":"Chim cánh cụt",
"pig":"Lợn",
"rabbit":"Thỏ",
"rhinoceros":"Tê giác",
"scorpion":"Bọ cạp",
"seal":"Hải cẩu",
"seagull":"Mòng biển",
"shark":"Cá mập",
"sheep":"Cừu",
"snake":"Rắn",
"spider":"Nhện",
"squid":"Mực",
"swan":"Thiên nga",
"tiger":"Hổ",
"turtle":"Rùa",
"vulture":"Kền kền",
"whale":"Cá voi",
"wolf":"Sói",
"zebra":"Ngựa vằn"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTForImageClassification.from_pretrained(
"google/vit-base-patch16-224",
num_labels=len(classes),
ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load("vit_animals_best.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor()
])

file = st.file_uploader("📷 Tải ảnh lên", type=["jpg","png","jpeg"])

if file:

    image = Image.open(file).convert("RGB")

    st.image(image,width=400)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(img).logits

        probs = torch.softmax(outputs, dim=1)[0]

        pred = torch.argmax(probs).item()

        confidence = float(probs[pred])*100

    label_en = classes[pred]

    label_vi = animal_vi.get(label_en,label_en)

    st.markdown(
        f"<div class='result-box'>🐾 Dự đoán: <b>{label_vi}</b><br>Độ tin cậy: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )