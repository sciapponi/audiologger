import torch
from torch import nn
from vector_quantize_pytorch import ResidualVQ
from soundstream.decoder import Decoder as SoundStreamDecoder


class QuantDecoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        # Initialize quantizer and decoder
        self.decoder = SoundStreamDecoder(C=40, D=256)
        self.quantizer = ResidualVQ(
            num_quantizers=16,
            codebook_size=1024,
            dim=256,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2,
            sync_codebook=False
        )

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Adjust any weights with unexpected shapes in the state_dict
        state_dict["quantizer"] = self._reshape_quantizer_weights(
            self.quantizer.state_dict(), state_dict["quantizer"]
        )
        state_dict["decoder"] = self._reshape_decoder_weights(
            self.decoder.state_dict(), state_dict["decoder"]
        )

        # Load quantizer and decoder state dicts
        self.quantizer.load_state_dict(state_dict["quantizer"])
        self.decoder.load_state_dict(state_dict["decoder"])

    def _reshape_quantizer_weights(self, model_state, checkpoint_state):
        reshaped_state = {}
        for key, value in checkpoint_state.items():
            if key in model_state:  # Only process keys present in the model
                expected_shape = model_state[key].shape
                if value.shape != expected_shape:
                    # Handle case where shape is [1, 1024, 256] -> [1024, 256]
                    if len(value.shape) == 3 and value.shape[0] == 1:
                        reshaped_state[key] = value.squeeze(0)
                    # Handle case where shape is [1, 1024] -> [1024]
                    elif len(value.shape) == 2 and value.shape[0] == 1:
                        reshaped_state[key] = value.squeeze(0)
                    else:
                        raise ValueError(
                            f"Unexpected shape for key '{key}': {value.shape}, expected: {expected_shape}"
                        )
                else:
                    reshaped_state[key] = value
            else:
                print(f"Warning: Key '{key}' not found in the model's state_dict.")
        return reshaped_state

    def _reshape_decoder_weights(self, model_state, checkpoint_state):
        # Similar reshaping logic can be applied for the decoder if necessary
        return checkpoint_state

    def forward(self, indices):
        codes = self.quantizer.get_output_from_indices(indices)
        return self.decoder(codes.permute(0, 2, 1))