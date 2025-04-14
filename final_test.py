import torch
import matplotlib.pyplot as plt
import numpy as np
from model import LMFusion
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = LMFusion(
    hidden_size=128,
    num_attention_heads=4,
    intermediate_size=256, 
    num_layers=4,
    img_token_dim=4,
    img_size=64,
    vocab_size=tokenizer.vocab_size
)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()
prompt="A beautiful sunset over the ocean with palm trees"
inputs=tokenizer(prompt, return_tensors="pt", padding=True)
input_ids=inputs["input_ids"].to(device)
attention_mask=inputs["attention_mask"].to(device)

print(f"Generating image for prompt: '{prompt}'")
print(f"This will take some time as it runs {50} denoising steps...")

with torch.no_grad():
    generated_image = model.generate_image(
        input_ids=input_ids,
        attention_mask=attention_mask,
        steps=50
    )
img=generated_image[0].permute(1, 2, 0).cpu().numpy()
img=(img + 1) / 2
img=np.clip(img, 0, 1)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title(prompt)
plt.savefig('generated_image.png')
plt.show()
print(f"Image generated and saved")
