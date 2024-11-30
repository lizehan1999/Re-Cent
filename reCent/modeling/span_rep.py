import torch
from torch import nn

from .layers import create_projection_layer


def extract_elements(sequence, indices):
    B, L, D = sequence.shape

    expanded_indices = indices.unsqueeze(2).expand(-1, -1, D)
    extracted_elements = torch.gather(sequence, 1, expanded_indices)

    return extracted_elements


class SpanMarkerV0(nn.Module):
    """
    Marks and projects span endpoints using an MLP.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)

        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, h: torch.Tensor, span_idx: torch.Tensor) -> torch.Tensor:
        B, L, D = h.size()

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = torch.cat([start_span_rep, end_span_rep], dim=-1).relu()

        return self.out_project(cat).view(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """
    Various span representation approaches
    """

    def __init__(self, hidden_size, max_width, **kwargs):
        super().__init__()
        self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)

    def forward(self, x, *args):
        return self.span_rep_layer(x, *args)
