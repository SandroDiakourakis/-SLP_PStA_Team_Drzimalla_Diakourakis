# %%
from torch import nn
import soundfile as sf
import torch
# %%
# 8 audio dateien (wav) / pro individuum, das zu klassifizieren ist
#%%
# 1 datei pro individuum laden
wav, sr = sf.read("audio.wav", dtype="float32")
# features extrahieren
# wav2vec model laden
model = torch.hub.load("pytorch/fairseq", "wav2vec2_large_lv60k")
model.eval()
features_fuer_ein_datei = model(torch.tensor(wav).unsqueeze(0))
#%% 
# trainiere svm classifier mit features pro task (aeiou pa ta ka)
8 svms , 1 pro task

# %%
[a, e, i, o, u,  pa, ta, ka]
[0, 1, 1, 0, 1, 0, 1, 0]

# %%
feature extrahieren pro file, 8 zusammfasen fuer ein inviduum

class FileProcesser(torch.Module):
    def __init__(self, features_dim): # features fuer eine datei, schon gepooled über die zeit
        super().__init__()
        self.linlayer = nn.Linear(features_dim, 128)  # reduce to 128 dim

    def forward(self, features):
        out_feats = self.linlayer(features)
        out_feats = nn.functional.relu(out_feats)
        return out_feats
        
class MetaProcesser(torch.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fp1 = FileProcesser(features_dim=input_dim)
        self.fp2 = FileProcesser(features_dim=input_dim)
        self.fp3 = FileProcesser(features_dim=input_dim)
        self.fp4 = FileProcesser(features_dim=input_dim)
        self.fp5 = FileProcesser(features_dim=input_dim)
        self.fp6 = FileProcesser(features_dim=input_dim)
        self.fp7 = FileProcesser(features_dim=input_dim)
        self.fp8 = FileProcesser(features_dim=input_dim) 
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features_for_all_files):
        out1 = self.fp1(features_for_all_files[0])
        out2 = self.fp2(features_for_all_files[1])
        out3 = self.fp3(features_for_all_files[2])
        out4 = self.fp4(features_for_all_files[3])
        out5 = self.fp5(features_for_all_files[4])
        out6 = self.fp6(features_for_all_files[5])
        out7 = self.fp7(features_for_all_files[6])
        out8 = self.fp8(features_for_all_files[7])
        features = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], dim=-1)
        out = self.classifier(features)
        return out
# for classifcxation: torch.argmax(torch.softmax())

# %%
class FileProcesserDeciscions(torch.Module):
    def __init__(self, features_dim, num_classes): # features fuer eine datei, schon gepooled über die zeit
        super().__init__()
        self.linlayer = nn.Linear(features_dim, 128)  # reduce to 128 dim
        self.linlayer2 = nn.Linear(128, num_classes)  # reduce to 128 dim

    def forward(self, features):
        out_feats = self.linlayer(features)
        out_feats = nn.functional.relu(out_feats)
        out_feats = self.linlayer2(out_feats)
        return out_feats


        
class MetaProcesser(torch.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fp1 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp2 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp3 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp4 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp5 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp6 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp7 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes)
        self.fp8 = FileProcesserDeciscions(features_dim=input_dim, num_classes=num_classes) 
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features_for_all_files):
        out1 = self.fp1(features_for_all_files[0])
        out2 = self.fp2(features_for_all_files[1])
        out3 = self.fp3(features_for_all_files[2])
        out4 = self.fp4(features_for_all_files[3])
        out5 = self.fp5(features_for_all_files[4])
        out6 = self.fp6(features_for_all_files[5])
        out7 = self.fp7(features_for_all_files[6])
        out8 = self.fp8(features_for_all_files[7])
        features = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], dim=-1)
        out = self.classifier(features)
        return out