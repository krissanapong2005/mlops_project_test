from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io, torch
from PIL import Image
from torchvision import transforms
import mlflow.pytorch

MODEL_URI = "models:/cifar10-classifier@staging"  # ถ้าไม่มี Registry ให้เปลี่ยนเป็น runs:/<run_id>/model

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
prep = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
])

app = FastAPI(title="CIFAR10 Classifier API")

model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        x = prep(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0]
        top = int(torch.argmax(p).item())
        top5 = torch.topk(p, 5).indices.cpu().tolist()
        return {"class": CLASSES[top], "prob": float(p[top].cpu().item()),
                "top5": [{"class": CLASSES[i], "prob": float(p[i].cpu().item())} for i in top5]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
