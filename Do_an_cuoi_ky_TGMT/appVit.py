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


animal_vi = {
"alligator": "Cá sấu Mỹ",
"ant": "Kiến",
"bear": "Gấu",
"bee": "Ong",
"beetle": "Bọ cánh cứng",
"buffalo": "Trâu",
"butterfly": "Bướm",
"camel": "Lạc đà",
"cat": "Mèo",
"chameleon": "Tắc kè hoa",
"cheetah": "Báo gêpa",
"chicken": "Gà",
"chimpanzee": "Tinh tinh",
"cobra": "Rắn hổ mang",
"cockroach": "Gián",
"cow": "Bò",
"crab": "Cua",
"cricket": "Dế",
"crocodile": "Cá sấu",
"crow": "Quạ",
"deer": "Hươu",
"dog": "Chó",
"dolphin": "Cá heo",
"dragonfly": "Chuồn chuồn",
"duck": "Vịt",
"eagle": "Đại bàng",
"elephant": "Voi",
"elk": "Nai sừng tấm",
"falcon": "Chim ưng",
"flamingo": "Hồng hạc",
"fox": "Cáo",
"frog": "Ếch",
"gecko": "Tắc kè",
"giraffe": "Hươu cao cổ",
"goat": "Dê",
"goldfish": "Cá vàng",
"goose": "Ngỗng",
"gorilla": "Khỉ đột",
"grasshopper": "Châu chấu",
"guinea pig": "Chuột lang",
"hamster": "Chuột hamster",
"hippopotamus": "Hà mã",
"horse": "Ngựa",
"hyena": "Linh cẩu",
"jaguar": "Báo đốm",
"kangaroo": "Chuột túi",
"koala": "Gấu koala",
"komodo dragon": "Rồng Komodo",
"ladybug": "Bọ rùa",
"lemur": "Vượn cáo",
"leopard": "Báo hoa mai",
"lion": "Sư tử",
"lizard": "Thằn lằn",
"lobster": "Tôm hùm",
"magpie": "Chim ác là",
"mantis": "Bọ ngựa",
"monkey": "Khỉ",
"moose": "Nai sừng tấm Bắc Mỹ",
"octopus": "Bạch tuộc",
"orangutan": "Đười ươi",
"otter": "Rái cá",
"owl": "Cú mèo",
"panda": "Gấu trúc",
"parrot": "Vẹt",
"peacock": "Công",
"pelican": "Bồ nông",
"penguin": "Chim cánh cụt",
"pig": "Lợn",
"python": "Trăn",
"rabbit": "Thỏ",
"rhinoceros": "Tê giác",
"salamander": "Kỳ giông",
"scorpion": "Bọ cạp",
"sea lion": "Sư tử biển",
"seagull": "Mòng biển",
"seal": "Hải cẩu",
"shark": "Cá mập",
"sheep": "Cừu",
"shrimp": "Tôm",
"skunk": "Chồn hôi",
"snake": "Rắn",
"sparrow": "Chim sẻ",
"spider": "Nhện",
"squid": "Mực",
"stingray": "Cá đuối",
"stork": "Cò",
"swan": "Thiên nga",
"termite": "Mối",
"tiger": "Hổ",
"toad": "Cóc",
"tortoise": "Rùa cạn",
"turkey": "Gà tây",
"turtle": "Rùa",
"vulture": "Kền kền",
"walrus": "Hải mã",
"whale": "Cá voi",
"wolf": "Sói",
"wombat": "Gấu túi mũi trần",
"woodpecker": "Chim gõ kiến",
"zebra": "Ngựa vằn"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTForImageClassification.from_pretrained(
"google/vit-base-patch16-224",
num_labels=len(classes),
ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load("model/vit_animals_best.pth", map_location=device))
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
