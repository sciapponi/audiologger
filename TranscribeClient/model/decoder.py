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

        # Process and load quantizer weights
        quantizer_state_dict = state_dict["quantizer"]
        for key, value in quantizer_state_dict.items():
            # Reshape weights if they have an extra dimension
            if value.ndim == 3 and value.shape[0] == 1:
                quantizer_state_dict[key] = value.squeeze(0)
            elif value.ndim == 2 and value.shape[0] == 1:
                quantizer_state_dict[key] = value.squeeze(0)

        self.quantizer.load_state_dict(quantizer_state_dict)

        # Load decoder weights
        self.decoder.load_state_dict(state_dict["decoder"])

    def forward(self, indices):
        codes = self.quantizer.get_output_from_indices(indices)
        return self.decoder(codes.permute(0, 2, 1))