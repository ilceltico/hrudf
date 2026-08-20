"""
Implementation of DeepSDF (Park et al., CVPR 2019)
and similar models.
"""

import torch
import torch.nn as nn

from .activation import get_activation
from .features import get_input_features


class DeepSDF(nn.Module):
	"""DeepSDF network from Park et al., CVPR 2019."""

	def __init__(self, latent_dim=256, hidden_dim=512, n_layers=8, in_insert=[4],
				 dropout=0.2, weight_norm=True, last_tanh=False,
				 layer_norm=False, activation="relu", features=None,out_dim = 1,in_dim=3,
				 **kwargs):
		super().__init__()
		self.latent_dim = latent_dim
		self.in_insert = in_insert

		if features is None or features == "none":
			self.features = None
			feats_dim = in_dim
		else:
			self.features = get_input_features(features)
			feats_dim = self.features.outdim

		self.fcs = nn.ModuleList()
		for n in range(n_layers):
			layer = []

			# Fully-connected
			in_d = out_d = hidden_dim
			if n == 0:
				in_d = latent_dim + feats_dim
			if n == n_layers - 1:
				out_d = out_dim
			elif n + 1 in self.in_insert:
				out_d = hidden_dim - (latent_dim + feats_dim)

			fc = nn.Linear(in_d, out_d)
			if weight_norm:
				fc = nn.utils.weight_norm(fc)
			layer.append(fc)

			# Normalization
			if not weight_norm and layer_norm and n < n_layers - 1:
				layer.append(nn.LayerNorm(out_d))

			# Activation
			if n < n_layers - 1:
				layer.append(get_activation(activation))
			elif last_tanh:
				layer.append(nn.Tanh())

			# Dropout
			if dropout > 0. and n < n_layers - 1:
				layer.append(nn.Dropout(dropout))

			# Combine them
			self.fcs.append(nn.Sequential(*layer))

	def forward(self, x):
		# Separate latent from positions
		lat = x[..., :self.latent_dim]
		xyz = x[..., self.latent_dim:]

		feats = self.features(xyz) if self.features is not None else xyz
		x = torch.cat([lat, feats], dim=-1)

		out = x
		for i, fc in enumerate(self.fcs):
			out = fc(out)

			if (i + 1) in self.in_insert:
				out = torch.cat([out, x], dim=-1)

		return out
