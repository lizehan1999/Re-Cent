import torch
from torch import nn


class SoRepLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.4):
        super().__init__()
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def forward(self, sub: torch.Tensor, ob: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            sub: [bs, sub_num, hidden_size]
            ob: [bs, ob_num, hidden_size]
        Returns:
            output_rep: [bs, sub_num, ob_num, hidden_size]
        '''
        B, sub_num, D = sub.size()
        B, ob_num, D = ob.size()

        sub_expanded = sub.unsqueeze(2).expand(B, sub_num, ob_num, D)  # [bs, sub_num, ob_num, hidden_size]
        ob_expanded = ob.unsqueeze(1).expand(B, sub_num, ob_num, D)  # [bs, sub_num, ob_num, hidden_size]

        concat = torch.cat([sub_expanded, ob_expanded], dim=-1).relu()  # [bs, sub_num, ob_num, hidden_size * 2]

        output_rep = self.out_project(concat)  # [bs, sub_num, ob_num, hidden_size]

        return output_rep


def create_projection_layer(hidden_size: int, dropout: float, out_dim: int = None) -> nn.Sequential:
    if out_dim is None:
        out_dim = hidden_size

    return nn.Sequential(
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim)
    )
