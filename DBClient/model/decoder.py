import torch 
from torch import nn  
from vector_quantize_pytorch import ResidualVQ
from soundstream.decoder import Decoder as SoundStreamDecoder
class QuantDecoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        # Initialize quantizer and decoder as empty modules
        self.decoder = SoundStreamDecoder(C=40, D=256)

        self.quantizer = ResidualVQ(
            num_quantizers=16,
            codebook_size=1024,
            dim=256,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2
        )
        
        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Load quantizer and decoder state dicts
        self.quantizer.load_state_dict(state_dict["quantizer"])
        self.decoder.load_state_dict(state_dict["decoder"])

    def forward(self, indices):
        codes = self.quantizer.get_output_from_indices(indices)
        return self.decoder(codes.permute(0, 2, 1))