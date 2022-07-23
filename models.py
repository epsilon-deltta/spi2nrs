import torch
from torch import nn
import torchaudio
class Wave2tf(torch.nn.Module):
    def __init__(self):
        super(Wave2tf,self).__init__()

        # feature_extraction
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.wave2vec = bundle.get_model()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        tf_enc = nn.TransformerEncoder(encoder_layer, num_layers=3)

        
        conv1d_0 = nn.Conv1d(281,140,3)
        gelu_0 = nn.GELU()
        bn_0 = nn.BatchNorm1d(140)
        convb_0 = nn.Sequential(conv1d_0,gelu_0,bn_0)

        conv1d_1 = nn.Conv1d(140,70,3)
        gelu_1 = nn.GELU()
        bn_1 = nn.BatchNorm1d(70)
        convb_1 = nn.Sequential(conv1d_1,gelu_1,bn_1)

        conv1d_2 = nn.Conv1d(70,35,3)
        gelu_2 = nn.GELU()
        bn_2 = nn.BatchNorm1d(35)
        convb_2 = nn.Sequential(conv1d_2,gelu_2,bn_2)

        lin_3 = nn.Linear(26,14)
        relu_3 = nn.ReLU()
        linb_0 = nn.Sequential(lin_3,relu_3)

        lin_4 = nn.Linear(14,7)
        relu_4 = nn.ReLU()
        bn_4 = nn.BatchNorm1d(35)
        linb_1 = nn.Sequential(lin_4,relu_4,bn_4)

        fl_5 = nn.Flatten()
        lin_5 = nn.Linear(245,2)

        final = nn.Sequential(fl_5,lin_5)
        convbs = nn.Sequential(convb_0,convb_1,convb_2)
        lins   = nn.Sequential(linb_0,linb_1)

        self.blocks = nn.ModuleDict({'attention':tf_enc,'conv_blocks':convbs,'lin_blocks':lins,'fc':final})
        
    def forward(self,out):
        out, _ = self.wave2vec(out)
        out    = self.blocks['attention'](out)
        out    = self.blocks['conv_blocks'](out)
        out    = self.blocks['lin_blocks'](out)
        out    = self.blocks['fc'](out)
        return out