import numpy as np
import torch.nn as nn
import torch


class FuzzifyLayer(nn.Module):
    def __init__(self, input_dim, num_mfs, membership_type='gaussian'):
        super(FuzzifyLayer, self).__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs
        self.centers = nn.Parameter(torch.randn(input_dim, num_mfs))
        self.widths = nn.Parameter(torch.randn(input_dim, num_mfs).abs())
        self.membership_type = membership_type

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        if self.membership_type == 'gaussian':
            membership_values = torch.exp(-0.5 * torch.pow((x_expanded - self.centers) / self.widths, 2))
        elif self.membership_type == 'triangular':
            diff = torch.abs(x_expanded - self.centers)
            membership_values = torch.clamp(1 - diff / self.widths, min=0)
        return membership_values
    
class RuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, tnorm='product'):
        super(RuleLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tnorm=tnorm
        self.register_buffer("antecedent_idx", torch.empty(0, self.input_dim, dtype=torch.long))
        self.register_buffer("consequent_idx", torch.empty(0, self.output_dim, dtype=torch.long))

    @property
    def n_rules(self):
        return self.antecedent_idx.size(0)

    def forward(self, memberships):
        if self.n_rules == 0:
            return memberships.new_zeros(memberships.size(0), 0)
        batch_size, n_inputs, max_L = memberships.shape
        assert n_inputs == self.input_dim, "Input dimension mismatch."

        #fixherelater
        rule_strengths = torch.ones(batch_size, 1)

        for i in range(self.input_dim):
            rule_strengths = rule_strengths * x[:, i, :].unsqueeze(2)

        rule_strengths = rule_strengths.view(batch_size, -1)
        return rule_strengths
    
class GenSoFNN(nn.Module):
    def __init__(self, input_dim, num_mfs, output_dim):
        super(GenSoFNN, self).__init__()
        self.fuzzify_layer = FuzzifyLayer(input_dim, num_mfs)
        self.rule_layer = RuleLayer(input_dim, num_mfs)
        self.output_layer = nn.Linear(num_mfs ** input_dim, output_dim)

    def forward(self, x):
        fuzzified = self.fuzzify_layer(x)
        rule_strengths = self.rule_layer(fuzzified)
        output = self.output_layer(rule_strengths)
        return output