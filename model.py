import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNetDownsampler(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, t):
        t_emb = self.time_embed(t.reshape(-1, 1))
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = F.silu(x)
        x = self.pool(x)
        
        t_emb = t_emb.reshape(x.shape[0], -1, 1, 1)
        x = x + t_emb
        
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)
        return x

class UNetUpsampler(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, t, skip_x):
        b, seq_len, c = x.shape
        h = w = int(math.sqrt(seq_len))
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        
        t_emb = self.time_embed(t.reshape(-1, 1))
        t_emb = t_emb.reshape(x.shape[0], -1, 1, 1)
        
        x = self.up(x)
        x = x + t_emb
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x

class ModalitySpecificAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, text_weights=None, img_weights=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        self.text_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.text_o = nn.Linear(hidden_size, hidden_size)
        
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_o = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, text_hidden_states, img_hidden_states, attention_mask=None):
        batch_size = text_hidden_states.size(0)
        
        text_qkv = self.text_qkv(text_hidden_states)
        text_qkv = text_qkv.reshape(batch_size, -1, 3, self.num_attention_heads, self.head_dim)
        text_q, text_k, text_v = text_qkv.unbind(dim=2)
        text_q = text_q.transpose(1, 2)
        text_k = text_k.transpose(1, 2)
        text_v = text_v.transpose(1, 2)
        
        img_qkv = self.img_qkv(img_hidden_states)
        img_qkv = img_qkv.reshape(batch_size, -1, 3, self.num_attention_heads, self.head_dim)
        img_q, img_k, img_v = img_qkv.unbind(dim=2)
        img_q = img_q.transpose(1, 2)
        img_k = img_k.transpose(1, 2)
        img_v = img_v.transpose(1, 2)
        
        combined_k = torch.cat([img_k, text_k], dim=2)
        combined_v = torch.cat([img_v, text_v], dim=2)        
        text_attn_weights = torch.matmul(text_q, combined_k.transpose(2, 3)) * self.scaling        
        text_seq_len = text_hidden_states.size(1)
        img_seq_len = img_hidden_states.size(1)
        total_seq_len = text_seq_len + img_seq_len
        
        causal_mask = torch.triu(
            torch.ones(text_seq_len, total_seq_len, dtype=torch.bool, device=text_hidden_states.device),
            diagonal=img_seq_len + 1
        )
        
        text_attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        
        if attention_mask is not None:
            # Reshape attention mask for text_attn_weights
            # Expand attention_mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            # and then broadcast to match the shape of text_attn_weights
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Apply attention mask only to the text part (not image part)
            # Create a padded mask of the right shape
            padded_mask = torch.zeros(batch_size, 1, 1, total_seq_len, 
                                      device=text_hidden_states.device)
            padded_mask[:, :, :, img_seq_len:] = (1.0 - extended_attention_mask) * -10000.0
            text_attn_weights = text_attn_weights + padded_mask
            
        text_attn_probs = F.softmax(text_attn_weights, dim=-1)
        text_attn_output = torch.matmul(text_attn_probs, combined_v)
        text_attn_output = text_attn_output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        text_attn_output = self.text_o(text_attn_output)
        
        img_attn_weights = torch.matmul(img_q, combined_k.transpose(2, 3)) * self.scaling
        
        if attention_mask is not None:
            # Apply the same mask transformation for image attention
            img_attn_weights = img_attn_weights + padded_mask
            
        img_attn_probs = F.softmax(img_attn_weights, dim=-1)
        img_attn_output = torch.matmul(img_attn_probs, combined_v)
        img_attn_output = img_attn_output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        img_attn_output = self.img_o(img_attn_output)
        
        return text_attn_output, img_attn_output

class ModalitySpecificFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, text_weights=None, img_weights=None):
        super().__init__()
        
        self.text_gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.text_down_proj = nn.Linear(intermediate_size, hidden_size)
        self.text_up_proj = nn.Linear(hidden_size, intermediate_size)
        
        self.img_gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.img_down_proj = nn.Linear(intermediate_size, hidden_size)
        self.img_up_proj = nn.Linear(hidden_size, intermediate_size)
    
    def forward(self, text_hidden_states, img_hidden_states):
        text_gate = F.silu(self.text_gate_proj(text_hidden_states))
        text_up = self.text_up_proj(text_hidden_states)
        text_output = self.text_down_proj(text_gate * text_up)
        
        img_gate = F.silu(self.img_gate_proj(img_hidden_states))
        img_up = self.img_up_proj(img_hidden_states)
        img_output = self.img_down_proj(img_gate * img_up)
        
        return text_output, img_output

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.SiLU()
        )
        
        self.fc_mu = nn.Conv2d(256, latent_dim, 1)
        self.fc_var = nn.Conv2d(256, latent_dim, 1)
        
        self.decoder_input = nn.Conv2d(latent_dim, 256, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

class LMFusionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, layer_norm_eps=1e-5, text_weights=None, img_weights=None):
        super().__init__()
        
        self.text_input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.text_post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.img_input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.img_post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.attention = ModalitySpecificAttention(
            hidden_size, 
            num_attention_heads
        )
        
        self.ffn = ModalitySpecificFFN(
            hidden_size, 
            intermediate_size
        )
    
    def forward(self, text_hidden_states, img_hidden_states, attention_mask=None):
        text_residual = text_hidden_states
        img_residual = img_hidden_states
        
        text_hidden_states = self.text_input_layernorm(text_hidden_states)
        img_hidden_states = self.img_input_layernorm(img_hidden_states)
        
        text_attn_output, img_attn_output = self.attention(
            text_hidden_states,
            img_hidden_states,
            attention_mask
        )
        
        text_hidden_states = text_residual + text_attn_output
        img_hidden_states = img_residual + img_attn_output
        
        text_residual = text_hidden_states
        img_residual = img_hidden_states
        
        text_hidden_states = self.text_post_attention_layernorm(text_hidden_states)
        img_hidden_states = self.img_post_attention_layernorm(img_hidden_states)
        
        text_ffn_output, img_ffn_output = self.ffn(text_hidden_states, img_hidden_states)
        
        text_hidden_states = text_residual + text_ffn_output
        img_hidden_states = img_residual + img_ffn_output
        
        return text_hidden_states, img_hidden_states

class RandomEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)
        
    def forward(self, input_ids):
        return nn.functional.embedding(input_ids, self.weight)

class RandomLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, hidden_states):
        return nn.functional.linear(hidden_states, self.weight, self.bias)

class LMFusion(nn.Module):
    def __init__(self, hidden_size=128, num_attention_heads=4, intermediate_size=256, num_layers=4, img_token_dim=4, img_size=64, vocab_size=32000):
        super().__init__()
        
        self.embed_tokens = RandomEmbedding(vocab_size, hidden_size)
        self.lm_head = RandomLMHead(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        
        self.vae = VAE(input_channels=3, latent_dim=img_token_dim)
        self.unet_down = UNetDownsampler(img_token_dim, hidden_size)
        self.unet_up = UNetUpsampler(hidden_size, img_token_dim)
        
        self.layers = nn.ModuleList([
            LMFusionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            ) for _ in range(num_layers)
        ])
        
    def add_noise(self, img, t):
        alpha_bar = torch.cos(t * math.pi/2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        noise = torch.randn_like(img)
        noisy_img = torch.sqrt(alpha_bar) * img + torch.sqrt(1 - alpha_bar) * noise
        
        return noisy_img, noise
    
    def forward(self, input_ids, images=None, t=None, attention_mask=None, generation_mode=False):
        batch_size = input_ids.shape[0]        
        text_hidden_states = self.embed_tokens(input_ids)
        if images is not None and t is not None:
            img_mu, img_log_var = self.vae.encode(images)
            img_z = self.vae.reparameterize(img_mu, img_log_var)
            noisy_img_z, target_noise = self.add_noise(img_z, t)
            img_hidden_states = self.unet_down(noisy_img_z, t)
        else:
            img_hidden_states = torch.zeros((batch_size, 0, text_hidden_states.shape[-1]), device=text_hidden_states.device)
            target_noise = None
        
        for layer in self.layers:
            text_hidden_states, img_hidden_states = layer(
                text_hidden_states, 
                img_hidden_states,
                attention_mask
            )
        
        outputs = {}
        
        logits = self.lm_head(text_hidden_states)
        outputs["logits"] = logits
        
        if images is not None and t is not None and not generation_mode:
            pred_noise = self.unet_up(img_hidden_states, t, noisy_img_z)
            outputs["pred_noise"] = pred_noise
            outputs["target_noise"] = target_noise
        
        if generation_mode and img_hidden_states.shape[1] > 0:
            predicted_z = self.unet_up(img_hidden_states, torch.zeros_like(t), None)
            generated_image = self.vae.decode(predicted_z)
            outputs["generated_image"] = generated_image
        
        return outputs
        
    def generate_image(self, input_ids, attention_mask=None, steps=50):
        batch_size = input_ids.shape[0]
        
        img_z = torch.randn(batch_size, 4, 16, 16, device=input_ids.device)
        
        for i in range(steps-1, -1, -1):
            t = torch.ones(batch_size, device=input_ids.device) * i / steps
            
            text_hidden_states = self.embed_tokens(input_ids)
            img_hidden_states = self.unet_down(img_z, t)
            
            for layer in self.layers:
                text_hidden_states, img_hidden_states = layer(
                    text_hidden_states, 
                    img_hidden_states,
                    attention_mask
                )
            
            pred_noise = self.unet_up(img_hidden_states, t, img_z)
            
            alpha = torch.cos(torch.tensor(i / steps, device=input_ids.device) * math.pi/2) ** 2
            alpha_next = torch.cos(torch.tensor((i-1) / steps, device=input_ids.device) * math.pi/2) ** 2 if i > 0 else torch.ones_like(alpha)
            
            img_z = (img_z - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            
            if i > 0:
                noise = torch.randn_like(img_z)
                sigma = ((1 - alpha_next) / (1 - alpha)).sqrt() * (1 - alpha/alpha_next).sqrt()
                img_z = alpha_next.sqrt() * img_z + (1 - alpha_next - sigma**2).sqrt() * pred_noise + sigma * noise
        
        generated_image = self.vae.decode(img_z)
        
        return generated_image

class MockLMFusionDataset(Dataset):
    def __init__(self, num_samples=10, seq_length=32, img_size=64, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.img_size = img_size
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones_like(input_ids)
        image = torch.rand(3, self.img_size, self.img_size) * 2 - 1
        t = torch.rand(1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "t": t
        }

class LMFusionLoss(nn.Module):
    def __init__(self, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    def forward(self, outputs, targets):
        losses = {}
        
        if "logits" in outputs:
            logits = outputs["logits"]
            input_ids = targets["input_ids"]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            text_loss = self.text_loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            losses["text_loss"] = text_loss
        
        if "pred_noise" in outputs and "target_noise" in outputs:
            image_loss = F.mse_loss(
                outputs["pred_noise"], 
                outputs["target_noise"]
            )
            losses["image_loss"] = image_loss
        
        if "text_loss" in losses and "image_loss" in losses:
            losses["total_loss"] = losses["text_loss"] + losses["image_loss"]
        elif "text_loss" in losses:
            losses["total_loss"] = losses["text_loss"]
        elif "image_loss" in losses:
            losses["total_loss"] = losses["image_loss"]
        
        return losses

def lmfusion_dry_run(batch_size=2, num_steps=3, tiny_model=True):
    print("\n===== LMFusion Model Dry Run Test =====")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Creating mock dataset and dataloader...")
    dataset = MockLMFusionDataset(num_samples=batch_size*num_steps, seq_length=16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Initializing LMFusion model with random weights...")
    
    if tiny_model:
        hidden_size = 32
        num_attention_heads = 4
        intermediate_size = 64
        num_layers = 2
    else:
        hidden_size = 512
        num_attention_heads = 8
        intermediate_size = 1024
        num_layers = 4
    
    try:
        model = LMFusion(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            img_token_dim=4,
            img_size=64,
            vocab_size=32000
        )
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        
        model = model.to(device)
        
        loss_fn = LMFusionLoss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        print("\nPerforming forward and backward passes...")
        for step, batch in enumerate(tqdm(dataloader, total=num_steps)):
            if step >= num_steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                images=batch["image"],
                t=batch["t"],
                attention_mask=batch["attention_mask"]
            )
            
            losses = loss_fn(outputs, batch)
            
            print(f"\nStep {step+1}/{num_steps} Losses:")
            for loss_name, loss_val in losses.items():
                print(f"  {loss_name}: {loss_val.item():.4f}")
            
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
        
        print("\nTesting generation mode...")
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"][:1]
            
            gen_outputs = model(input_ids=input_ids, generation_mode=True)
            
            print(f"Generation output keys: {list(gen_outputs.keys())}")
            if "logits" in gen_outputs:
                print(f"Logits shape: {gen_outputs['logits'].shape}")
        
        print("\n✅ All tests passed! LMFusion model is working correctly.")
        return True
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    lmfusion_dry_run(batch_size=2, num_steps=3, tiny_model=True)

