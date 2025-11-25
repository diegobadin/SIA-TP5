"""
Variational Autoencoder (VAE) Implementation

This module implements a VAE with:
- Variational encoder (outputs mu and log_var)
- Reparameterization trick
- KL divergence loss
- Combined reconstruction + KL loss
"""

from __future__ import annotations
from typing import List, Sequence, Optional, Tuple
import numpy as np
import copy

from src.mlp.mlp import MLP
from src.mlp.activations import Activation, TANH, SIGMOID, IDENTITY
from src.mlp.erorrs import Loss, MSELoss
from src.mlp.optimizers import Optimizer, Adam, SGD, Momentum


def reparameterize(mu: np.ndarray, log_var: np.ndarray, 
                  epsilon: Optional[np.ndarray] = None,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Reparameterization trick: z = μ + σ * ε where ε ~ N(0, I)
    
    Args:
        mu: Mean vector (n_samples, latent_dim)
        log_var: Log-variance vector (n_samples, latent_dim)
        epsilon: Optional pre-sampled epsilon (for gradient flow)
        rng: Random number generator
        
    Returns:
        Sampled latent vector z (n_samples, latent_dim)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if epsilon is None:
        epsilon = rng.normal(0, 1, size=mu.shape)
    
    # σ = exp(0.5 * log_var)
    std = np.exp(0.5 * log_var)
    z = mu + std * epsilon
    return z


def kl_divergence_loss(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    Formula: KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))
    
    Args:
        mu: Mean vectors (n_samples, latent_dim)
        log_var: Log-variance vectors (n_samples, latent_dim)
        
    Returns:
        Average KL divergence (scalar)
    """
    # KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
    return float(np.mean(kl))


def _clone_optimizer(optimizer: Optimizer) -> Optimizer:
    """
    Create a new optimizer instance with the same parameters.
    
    Args:
        optimizer: Original optimizer instance
        
    Returns:
        New optimizer instance with same parameters
    """
    if isinstance(optimizer, Adam):
        return Adam(lr=optimizer.lr, beta1=optimizer.b1, beta2=optimizer.b2, eps=optimizer.eps)
    elif isinstance(optimizer, SGD):
        return SGD(lr=optimizer.lr)
    elif isinstance(optimizer, Momentum):
        return Momentum(lr=optimizer.lr, beta=optimizer.beta)
    else:
        # Fallback: try to create with same parameters
        # This handles custom optimizers
        return copy.deepcopy(optimizer)


class VariationalEncoder:
    """
    Variational encoder that outputs distribution parameters (μ, log_var).
    
    Architecture:
    - Shared base: [input_dim] → [hidden_layers] → [latent_dim]
    - Two heads: mu_head and log_var_head both output [latent_dim]
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_layers: Sequence[int],
                 activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        # Shared base network (up to last hidden layer)
        if len(hidden_layers) == 0:
            # Direct encoding: input -> latent
            base_sizes = [input_dim, latent_dim]
            base_acts = [activations[0] if len(activations) > 0 else TANH]
        else:
            base_sizes = [input_dim] + list(hidden_layers)
            # Base network needs activations for each hidden layer
            # The activations list should have one activation per hidden layer
            if len(activations) == len(hidden_layers):
                base_acts = list(activations)
            elif len(activations) > len(hidden_layers):
                # If more activations provided, use first len(hidden_layers)
                base_acts = list(activations[:len(hidden_layers)])
            else:
                # If fewer activations, pad with TANH
                base_acts = list(activations) + [TANH] * (len(hidden_layers) - len(activations))
        
        # Create separate optimizer instances for each MLP to avoid shape conflicts
        optimizer_base = _clone_optimizer(optimizer)
        optimizer_mu = _clone_optimizer(optimizer)
        optimizer_log_var = _clone_optimizer(optimizer)
        
        self.base_mlp = MLP(
            layer_sizes=base_sizes,
            activations=base_acts,
            loss=loss,
            optimizer=optimizer_base,
            w_init_scale=w_init_scale,
            seed=seed
        )
        
        # Two separate heads for mu and log_var
        hidden_dim = hidden_layers[-1] if len(hidden_layers) > 0 else input_dim
        
        self.mu_head = MLP(
            layer_sizes=[hidden_dim, latent_dim],
            activations=[IDENTITY], # IDENTITY: linear output for μ, regularization is handled by the KL term
            loss=loss,
            optimizer=optimizer_mu,
            w_init_scale=w_init_scale,
            seed=seed
        )
        
        self.log_var_head = MLP(
            layer_sizes=[hidden_dim, latent_dim],
            activations=[IDENTITY], # IDENTITY: linear output for log σ²
            loss=loss,
            optimizer=optimizer_log_var,
            w_init_scale=w_init_scale,
            seed=seed
        )
        
        self.latent_dim = latent_dim
        self.optimizer = optimizer
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input to distribution parameters.
        
        Args:
            x: Input data (n_samples, input_dim) or (input_dim,)
            
        Returns:
            Tuple (mu, log_var) both of shape (n_samples, latent_dim) or (latent_dim,)
        """
        # Pass through shared base (handle both single sample and batch)
        if x.ndim == 1:
            h = self.base_mlp.forward(x)
        else:
            h = self.base_mlp.predict(x)
        
        # Get mu and log_var from separate heads
        if h.ndim == 1:
            mu = self.mu_head.forward(h)
            log_var = self.log_var_head.forward(h)
        else:
            mu = self.mu_head.predict(h)
            log_var = self.log_var_head.predict(h)
        
        # Clip log_var for numerical stability
        log_var = np.clip(log_var, -10, 10)
        
        return mu, log_var


class VAEDecoder:
    """
    Decoder for VAE (similar to standard decoder).
    """
    def __init__(self,
                 latent_dim: int,
                 output_dim: int,
                 hidden_layers: Sequence[int],
                 activations: Sequence[Activation],
                 loss: Loss,
                 optimizer: Optimizer,
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        layer_sizes = [latent_dim] + list(hidden_layers) + [output_dim]
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(f"Se necesitan {len(layer_sizes) - 1} activaciones, se proporcionaron {len(activations)}")
        
        self.mlp = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss=loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent vector to output.
        
        Args:
            z: Latent vector (n_samples, latent_dim) or (latent_dim,)
            
        Returns:
            Decoded output (n_samples, output_dim) or (output_dim,)
        """
        if z.ndim == 1:
            return self.mlp.forward(z)
        return self.mlp.predict(z)


class VAE:
    """
    Variational Autoencoder combining encoder, reparameterization, and decoder.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 encoder_hidden: Sequence[int],
                 decoder_hidden: Sequence[int],
                 encoder_activations: Sequence[Activation],
                 decoder_activations: Sequence[Activation],
                 reconstruction_loss: Loss,
                 optimizer: Optimizer,
                 beta: float = 1.0,  # KL weight
                 w_init_scale: float = 0.05,
                 seed: Optional[int] = None):
        
        self.encoder = VariationalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_layers=encoder_hidden,
            activations=encoder_activations,
            loss=reconstruction_loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_layers=decoder_hidden,
            activations=decoder_activations,
            loss=reconstruction_loss,
            optimizer=optimizer,
            w_init_scale=w_init_scale,
            seed=seed
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        self.optimizer = optimizer
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        
        # Training history
        self.history = {
            "reconstruction_loss": [],
            "kl_loss": [],
            "total_loss": []
        }
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input data (n_samples, input_dim)
            
        Returns:
            Tuple (x_recon, mu, log_var, z)
        """
        mu, log_var = self.encoder.encode(x)
        z = reparameterize(mu, log_var, rng=self.rng)
        x_recon = self.decoder.decode(z)
        return x_recon, mu, log_var, z
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 1,
            shuffle: bool = True, verbose: bool = False) -> dict:
        """
        Train the VAE using proper backpropagation.
        
        Strategy:
        1. Backprop reconstruction loss through decoder → get ∂L_recon/∂z
        2. Flow gradients through reparameterization → get ∂L_recon/∂μ, ∂L_recon/∂log_var
        3. Compute KL gradients directly on encoder → get ∂L_KL/∂μ, ∂L_KL/∂log_var
        4. Combine gradients and backprop through encoder
        
        Args:
            X: Training data (n_samples, input_dim) in range [0, 1]
            epochs: Number of training epochs
            batch_size: Batch size
            shuffle: Whether to shuffle data
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        for epoch in range(epochs):
            if shuffle:
                self.rng.shuffle(indices)
            
            epoch_recon_loss = []
            epoch_kl_loss = []
            epoch_total_loss = []
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = X[batch_indices]
                
                # Forward pass to compute losses
                mu, log_var = self.encoder.encode(x_batch)
                epsilon = self.rng.normal(0, 1, size=mu.shape)
                z = reparameterize(mu, log_var, epsilon=epsilon, rng=self.rng)
                x_recon = self.decoder.decode(z)
                
                # Compute losses for logging
                recon_loss_val = self.reconstruction_loss.value(x_recon, x_batch)
                kl_loss_val = kl_divergence_loss(mu, log_var)
                total_loss = recon_loss_val + self.beta * kl_loss_val
                
                epoch_recon_loss.append(recon_loss_val)
                epoch_kl_loss.append(kl_loss_val)
                epoch_total_loss.append(total_loss)
                
                # Perform proper backpropagation step using same forward values
                self._train_step(
                    x_batch,
                    mu=mu,
                    log_var=log_var,
                    epsilon=epsilon,
                    z=z,
                    x_recon=x_recon,
                )
            
            # Record epoch averages
            self.history["reconstruction_loss"].append(np.mean(epoch_recon_loss))
            self.history["kl_loss"].append(np.mean(epoch_kl_loss))
            self.history["total_loss"].append(np.mean(epoch_total_loss))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Recon: {self.history['reconstruction_loss'][-1]:.6f}, "
                      f"KL: {self.history['kl_loss'][-1]:.6f}, "
                      f"Total: {self.history['total_loss'][-1]:.6f}")
        
        return self.history
    
    def _backprop_decoder(self, x: np.ndarray, z: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
        """
        Backpropagate reconstruction loss through decoder to get gradient w.r.t. z.
        
        Args:
            x: Target (original input)
            z: Decoder input (latent code)
            x_recon: Decoder output (reconstruction)
            
        Returns:
            Gradient w.r.t. z: ∂L_recon/∂z
        """
        # Use decoder's backprop_to_input method
        if z.ndim == 1:
            delta_z = self.decoder.mlp.backprop_to_input(z, x)
        else:
            # Handle batch: backprop each sample
            delta_z_list = []
            for i in range(len(z)):
                delta_z_list.append(self.decoder.mlp.backprop_to_input(z[i], x[i]))
            delta_z = np.array(delta_z_list)
        return delta_z
    
    def _reparameterization_gradients(self, delta_z: np.ndarray, mu: np.ndarray, 
                                      log_var: np.ndarray, epsilon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients through reparameterization: z = μ + σ * ε
        
        Args:
            delta_z: ∂L_recon/∂z (from decoder backprop)
            mu: Mean vectors
            log_var: Log-variance vectors
            epsilon: Random samples used in reparameterization
            
        Returns:
            Tuple (∂L_recon/∂μ, ∂L_recon/∂log_var)
        """
        sigma = np.exp(0.5 * log_var)
        
        # ∂z/∂μ = 1, so ∂L_recon/∂μ = ∂L_recon/∂z
        delta_mu_recon = delta_z
        
        # ∂z/∂log_var = 0.5 * σ * ε
        # So ∂L_recon/∂log_var = (∂L_recon/∂z) * (0.5 * σ * ε)
        delta_log_var_recon = delta_z * 0.5 * sigma * epsilon
        
        return delta_mu_recon, delta_log_var_recon
    
    def _kl_gradients(self, mu: np.ndarray, log_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute KL divergence gradients directly on encoder outputs.
        
        KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))
        
        Args:
            mu: Mean vectors
            log_var: Log-variance vectors
            
        Returns:
            Tuple (∂L_KL/∂μ, ∂L_KL/∂log_var)
        """
        # ∂L_KL/∂μ = μ
        delta_mu_kl = mu
        
        # ∂L_KL/∂log_var = 0.5 * (exp(log_var) - 1)
        delta_log_var_kl = 0.5 * (np.exp(log_var) - 1)
        
        return delta_mu_kl, delta_log_var_kl
    
    def _update_decoder(self, x: np.ndarray, z: np.ndarray, x_recon: np.ndarray):
        """Update decoder using reconstruction loss."""
        # Update decoder weights using standard backpropagation
        if z.ndim == 1:
            self.decoder.mlp._backward_update(z, x)
        else:
            # Batch update: update each sample
            for i in range(len(z)):
                self.decoder.mlp._backward_update(z[i], x[i])
    
    def _backprop_encoder(self, x: np.ndarray, delta_mu_total: np.ndarray, 
                         delta_log_var_total: np.ndarray):
        """
        Backpropagate combined gradients through encoder and update weights.
        
        Args:
            x: Input data
            delta_mu_total: Combined gradient w.r.t. μ (recon + KL)
            delta_log_var_total: Combined gradient w.r.t. log_var (recon + KL)
        """
        # Forward through base to get hidden representation
        if x.ndim == 1:
            h = self.encoder.base_mlp.forward(x)
        else:
            h = self.encoder.base_mlp.predict(x)
        
        # Backprop through mu_head to get gradient w.r.t. h from mu branch
        # We need to manually backpropagate delta_mu_total through mu_head
        if h.ndim == 1:
            # Single sample: use backprop_to_input with a dummy target
            # Actually, we need to manually compute gradient w.r.t. h
            # Let's use the MLP's internal backpropagation
            # We'll treat delta_mu_total as if it's the gradient at output
            # and backpropagate through mu_head layers
            delta_h_mu = self._backprop_through_mlp_to_input(
                self.encoder.mu_head, h, delta_mu_total
            )
        else:
            # Batch: handle each sample
            delta_h_mu_list = []
            for i in range(len(h)):
                delta_h_mu_i = self._backprop_through_mlp_to_input(
                    self.encoder.mu_head, h[i], delta_mu_total[i]
                )
                delta_h_mu_list.append(delta_h_mu_i)
            delta_h_mu = np.array(delta_h_mu_list)
        
        # Backprop through log_var_head to get gradient w.r.t. h from log_var
        if h.ndim == 1:
            delta_h_log_var = self._backprop_through_mlp_to_input(
                self.encoder.log_var_head, h, delta_log_var_total
            )
        else:
            delta_h_log_var_list = []
            for i in range(len(h)):
                delta_h_log_var_i = self._backprop_through_mlp_to_input(
                    self.encoder.log_var_head, h[i], delta_log_var_total[i]
                )
                delta_h_log_var_list.append(delta_h_log_var_i)
            delta_h_log_var = np.array(delta_h_log_var_list)
        
        # Combine gradients from both heads
        delta_h_total = delta_h_mu + delta_h_log_var
        
        # Update mu_head weights
        if h.ndim == 1:
            # For single sample, we need to update mu_head
            # Use a simplified update: treat delta_mu_total as target gradient
            self._update_mlp_with_gradient(self.encoder.mu_head, h, delta_mu_total)
        else:
            # Batch update for mu_head
            for i in range(len(h)):
                self._update_mlp_with_gradient(self.encoder.mu_head, h[i], delta_mu_total[i])
        
        # Update log_var_head weights
        if h.ndim == 1:
            self._update_mlp_with_gradient(self.encoder.log_var_head, h, delta_log_var_total)
        else:
            # Batch update for log_var_head
            for i in range(len(h)):
                self._update_mlp_with_gradient(self.encoder.log_var_head, h[i], delta_log_var_total[i])
        
        # Update base_mlp weights
        if x.ndim == 1:
            # For base_mlp, we need to backpropagate delta_h_total
            # We'll use a dummy target approach
            self._update_mlp_with_gradient(self.encoder.base_mlp, x, delta_h_total)
        else:
            # Batch update for base_mlp
            for i in range(len(x)):
                self._update_mlp_with_gradient(self.encoder.base_mlp, x[i], delta_h_total[i])
    
    def _backprop_through_mlp_to_input(self, mlp: MLP, x: np.ndarray, delta_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate a gradient delta through MLP to get gradient at input.
        
        Args:
            mlp: MLP to backpropagate through
            x: Input to MLP
            delta_out: Gradient at output layer
            
        Returns:
            Gradient at input layer
        """
        # Forward pass to get activations and zs
        zs, acts = mlp._forward_full(x)
        
        # Start with output gradient
        deltas: List[np.ndarray] = [None] * len(mlp.layers)  # type: ignore
        deltas[-1] = delta_out
        
        # Backpropagate through hidden layers
        for l in reversed(range(len(mlp.layers)-1)):
            deltas[l] = mlp.layers[l].backprop_delta(deltas[l+1], mlp.layers[l+1].weights)
        
        # Get gradient at input
        if len(mlp.layers) > 0:
            W0 = mlp.layers[0].weights[:, 1:]  # Remove bias column
            delta_input = W0.T @ deltas[0]
            return delta_input
        return delta_out
    
    def _update_mlp_with_gradient(self, mlp: MLP, x: np.ndarray, delta_out: np.ndarray):
        """
        Update MLP weights using a gradient at the output layer.
        
        Args:
            mlp: MLP to update
            x: Input to MLP
            delta_out: Gradient at output layer
        """
        # Forward pass to get activations
        zs, acts = mlp._forward_full(x)
        
        # Compute deltas for all layers
        deltas: List[np.ndarray] = [None] * len(mlp.layers)  # type: ignore
        deltas[-1] = delta_out
        
        # Backpropagate through hidden layers
        for l in reversed(range(len(mlp.layers)-1)):
            deltas[l] = mlp.layers[l].backprop_delta(deltas[l+1], mlp.layers[l+1].weights)
        
        # Update weights
        mlp.optimizer.begin_step()
        for l, layer in enumerate(mlp.layers):
            g = layer.grad_w(acts[l], deltas[l])
            step = mlp.optimizer.update(l, g)
            layer.weights -= step
    
    def _update_encoder(self, x: np.ndarray, mu: np.ndarray, log_var: np.ndarray, 
                       z: np.ndarray, x_recon: np.ndarray):
        """Update encoder using KL + reconstruction gradients."""
        # This method is kept for compatibility but will be replaced by _backprop_encoder
        pass  # Encoder update handled in _train_step
    
    def _train_step(self, x_batch: np.ndarray, 
                    mu, 
                    log_var ,
                    epsilon, 
                    z , 
                    x_recon):
        """
        Single VAE training step with proper backpropagation.
        
        Optional cached forward values (mu, log_var, epsilon, z, x_recon)
        can be supplied to avoid redundant computation.
        """
        
        # 1. Backprop through decoder to get ∂L_recon/∂z
        delta_z = self._backprop_decoder(x_batch, z, x_recon)
        
        # 2. Update decoder weights (standard backprop)
        self._update_decoder(x_batch, z, x_recon)
        
        # 3. Flow gradients through reparameterization
        delta_mu_recon, delta_log_var_recon = self._reparameterization_gradients(
            delta_z, mu, log_var, epsilon
        )
        
        # 4. Compute KL gradients (directly on encoder, no decoder)
        delta_mu_kl, delta_log_var_kl = self._kl_gradients(mu, log_var)
        
        # 5. Combine gradients
        delta_mu_total = delta_mu_recon + self.beta * delta_mu_kl
        delta_log_var_total = delta_log_var_recon + self.beta * delta_log_var_kl
        
        # 6. Backprop through encoder and update weights
        self._backprop_encoder(x_batch, delta_mu_total, delta_log_var_total)
    
    def generate(self, n_samples: int = 10, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate new samples by sampling from prior N(0, I) and decoding.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed
            
        Returns:
            Generated samples (n_samples, input_dim)
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        
        # Sample from prior: z ~ N(0, I)
        z = rng.normal(0, 1, size=(n_samples, self.latent_dim))
        
        # Decode
        x_gen = self.decoder.decode(z)
        return x_gen
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent representation (mean μ) for input data.
        
        Args:
            X: Input data (n_samples, input_dim)
            
        Returns:
            Mean latent vectors (n_samples, latent_dim)
        """
        mu, _ = self.encoder.encode(X)
        return mu

