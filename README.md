# CS772 Assignment 2 — Implementing and Analyzing Variational Autoencoder (VAE)

This assignment implements and analyzes a **Variational Autoencoder (VAE)** using PyTorch on the **MNIST handwritten digit dataset**.

The assignment focuses not only on implementing a VAE, but also on understanding its mathematical objective, comparing different KL-divergence estimation strategies, studying the effect of the $\beta$ parameter, analyzing posterior collapse, and investigating the geometry of the learned latent space.

---

## Objectives

The main objectives of this assignment are:

- Understand the **Evidence Lower Bound (ELBO)** objective used to train VAEs.
- Derive and implement the **reconstruction loss**.
- Derive the **closed-form KL divergence** between two Gaussian distributions.
- Implement the **reparameterization trick** for differentiable sampling.
- Train a VAE on MNIST and analyze its reconstruction and KL losses.
- Compare **closed-form KL** with a **Monte Carlo KL approximation**.
- Study the effect of different $\beta$ values using a **$\beta$-VAE**.
- Investigate **posterior collapse** in latent dimensions.
- Analyze the geometry of the learned latent space using interpolation.
- Visualize the latent space using a 2-dimensional VAE.

---

## Dataset

The experiments use the **MNIST handwritten digit dataset**.

- Training samples: 60,000
- Test samples: 10,000
- Image size: $28 \times 28$
- Image type: Grayscale
- Pixel values: Normalized to $[0,1]$

The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## VAE Formulation

The VAE assumes the latent-variable model

$$
p_\theta(x,z) = p_\theta(x|z)p(z)
$$

where the prior over the latent variable is

$$
p(z) = \mathcal{N}(0,I).
$$

The encoder learns an approximate posterior

$$
q_\phi(z|x)
=
\mathcal{N}
\left(
\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))
\right).
$$

The VAE is trained by maximizing the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(\theta,\phi;x)
=
\mathbb{E}_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}
\left(
q_\phi(z|x)\|p(z)
\right).
$$

The implementation therefore consists of two major loss components:

1. Reconstruction loss
2. KL-divergence regularization

---

## Model Architecture

The VAE consists of an encoder, a latent representation, and a decoder.

### Encoder

The input MNIST image is flattened from $28 \times 28$ into 784 dimensions.

```text
784
 ↓
Linear(784 → 512)
 ↓
ReLU
 ↓
Linear(512 → 256)
 ↓
ReLU
 ↓
 ┌───────────────┐
 ↓               ↓
μ             log(σ²)
