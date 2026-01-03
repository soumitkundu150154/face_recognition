import cv2, torch
from facenet_pytorch import MTCNN, InceptionResnetV1

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to("cuda" if torch.cuda.is_available() else "cpu")

img = cv2.imread("authorized_faces/person1.jpg")  # put one sample image here
if img is None:
    print("Place a sample image at authorized_faces/example.jpg and re-run test.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with torch.no_grad():
    t = mtcnn(img_rgb)
print("mtcnn output type:", type(t), "shape:", None if t is None else t.shape)
if t is not None:
    if t.ndim == 3:
        t = t.unsqueeze(0)
    emb = resnet(t.to("cuda" if torch.cuda.is_available() else "cpu"))
    print("Embedding shape:", emb.shape)
