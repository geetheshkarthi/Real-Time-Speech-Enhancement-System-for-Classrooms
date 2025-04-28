import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

# -------------------------------------------
# Early Stopping
# -------------------------------------------
class EarlyStopping:
    def _init_(self, patience=8, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def check_early_stop(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# -------------------------------------------
# Improved Comb Filter with Dynamic Alpha
# -------------------------------------------
class CombFilter(nn.Module):
    def _init_(self, max_delay=240):
        super(CombFilter, self)._init_()
        self.alpha = nn.Parameter(torch.tensor(0.9))
        self.delay = nn.Parameter(torch.clamp(torch.tensor(100.0) + torch.randn(1) * 5, 1, 240))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha * x.var(dim=-1, keepdim=True))
        delay = int(torch.clamp(self.delay, 1, 240).item())
        delayed_signal = F.pad(x, (delay, 0))[:, :, :-delay]
        return x + alpha * delayed_signal

# -------------------------------------------
# STFT and ISTFT Processing
# -------------------------------------------
class STFTProcessor(nn.Module):
    def _init_(self, n_fft=512, hop_length=128):
        super()._init_()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

    def forward(self, waveform):
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window.to(waveform.device), return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        return magnitude, phase  # Returning magnitude and phase separately

    def inverse(self, magnitude, phase):
        stft_complex = magnitude * torch.exp(1j * phase)  # Reconstruct STFT
        waveform = torch.istft(stft_complex, n_fft=self.n_fft, hop_length=self.hop_length,
                               window=self.window.to(magnitude.device))
        return waveform

# -------------------------------------------
# Envelope Postfilter
# -------------------------------------------
def envelope_postfilter(magnitude, alpha=0.02):
    smooth_gain = torch.sin((torch.pi / 2) * magnitude)  # Inspired by formant postfiltering
    return (1 - alpha) * magnitude + alpha * smooth_gain


# -------------------------------------------
# Enhanced PercepNet Model
# -------------------------------------------
class PercepNet(nn.Module):
    def _init_(self, n_fft=512, hop_length=128):
        super(PercepNet, self)._init_()
        self.stft = STFTProcessor(n_fft, hop_length)
        
        # Make sure first Conv1D expects freq_bins as input channels
        self.conv1 = nn.Sequential(nn.Conv1d(257, 64, 5, 1, 2), nn.ReLU(), nn.Dropout(0.3))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 5, 1, 2), nn.ReLU(), nn.Dropout(0.3))
        
        self.gru = nn.GRU(128, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 257)  # Output should match frequency bins
    
    def forward(self, x):
        magnitude, phase = self.stft(x.squeeze(1))  # [batch, freq_bins=257, time_steps=126]
        
        # Keep (batch, freq_bins, time_steps) for Conv1D
        x = magnitude  # Already in [batch, freq_bins=257, time_steps=126]
        
        x = self.conv1(x)  # [batch, 64, time_steps]
        x = self.conv2(x)  # [batch, 128, time_steps]
        
        x = x.permute(0, 2, 1)  # Prepare for GRU [batch, time_steps, features]
        x, _ = self.gru(x)
        
        x = self.fc(x)  # Map back to 257 frequency bins
        x = x.permute(0, 2, 1)  # Convert back to [batch, freq_bins, time_steps]

        # Apply enhancement
        enhanced_magnitude = magnitude * (1 + 0.1 * x)  # Modify magnitude
        enhanced_waveform = self.stft.inverse(enhanced_magnitude, phase)  # ISTFT reconstruction
        
        return enhanced_waveform.unsqueeze(1)  # Add back channel dimension



# -------------------------------------------
# Dataset with STFT Processing & Length Matching
# -------------------------------------------
class AudioDataset(Dataset):
    def _init_(self, noisy_dir, clean_dir, sample_rate=16000, max_length=16000):
        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.resample = T.Resample(orig_freq=48000, new_freq=16000)
        self.max_length = max_length

    def pad_or_trim(self, waveform):
        if waveform.shape[1] < self.max_length:
            pad_size = self.max_length - waveform.shape[1]
            return F.pad(waveform, (0, pad_size))
        return waveform[:, :self.max_length]

    def _getitem_(self, idx):
        noisy_waveform, _ = sf.read(os.path.join(self.noisy_dir, self.noisy_files[idx]))
        clean_waveform, _ = sf.read(os.path.join(self.clean_dir, self.clean_files[idx]))

        noisy_waveform = torch.tensor(noisy_waveform).unsqueeze(0).float()
        clean_waveform = torch.tensor(clean_waveform).unsqueeze(0).float()

        noisy_waveform = self.resample(noisy_waveform)
        clean_waveform = self.resample(clean_waveform)

        # Match lengths
        noisy_waveform = self.pad_or_trim(noisy_waveform)
        clean_waveform = self.pad_or_trim(clean_waveform)

        return noisy_waveform, clean_waveform

    def _len_(self):
        return len(self.noisy_files)


# -------------------------------------------
# Training Function
# -------------------------------------------
def train(model, dataloader, optimizer, criterion, scheduler, early_stopper, ema_model, epochs=100):
    scaler = torch.amp.GradScaler()
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            noisy, clean = noisy.cuda(), clean.cuda()

            # Check tensor devices
            # print(f"Noisy device: {noisy.device}, Clean device: {clean.device}")
            # print(f"Model conv1 weight device: {model.conv1[0].weight.device}")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                output = model(noisy)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema_model.update_parameters(model)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # ✅ Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_percepnet_new.pth")
            print("✅ Best Model Saved as 'best_percepnet_new.pth'")

        # ✅ Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"percepnet_epoch_{epoch}.pth")
            print(f"📌 Checkpoint saved: percepnet_epoch_{epoch}.pth")

        # Early Stopping
        if early_stopper.check_early_stop(avg_loss):
            print("🛑 Early Stopping Triggered. Training Stopped.")
            break

    # Check model parameter devices after training
    for param in model.parameters():
        print(f"Model parameter device: {param.device}")


# -------------------------------------------
# Main Script
# -------------------------------------------
if _name_ == "_main_":
    torch.backends.cudnn.benchmark = True
    model = PercepNet().cuda()
    ema_model = AveragedModel(model)
    dataset = AudioDataset("noisy_speech", "clean_speech")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=8)

    train(model, dataloader, optimizer, criterion, scheduler, early_stopper, ema_model, epochs=100)
    print("✅ Model Training Complete!")