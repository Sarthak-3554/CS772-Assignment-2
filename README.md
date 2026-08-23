# CS772 Assignment 2 — Implementing and Analyzing Variational Autoencoder (VAE)

This assignment implements and analyzes a **Variational Autoencoder (VAE)** using PyTorch on the **MNIST handwritten digit dataset**.

The assignment focuses on both the implementation and mathematical understanding of VAEs, including the Evidence Lower Bound (ELBO), reconstruction loss, KL divergence, the reparameterization trick, Monte Carlo KL estimation, β-VAE, posterior collapse, and latent-space geometry.

---

## Objectives

The main objectives of this assignment are:

- Understand the **Evidence Lower Bound (ELBO)** objective used to train VAEs.
- Derive and implement the **reconstruction loss**.
- Derive the **closed-form KL divergence** between Gaussian distributions.
- Implement the **reparameterization trick** for differentiable sampling.
- Train a VAE on MNIST and analyze its reconstruction and KL losses.
- Compare **closed-form KL** with a **Monte Carlo KL approximation**.
- Study the effect of different β values using a **β-VAE**.
- Investigate **posterior collapse** and inactive latent dimensions.
- Analyze the geometry of the learned latent space using interpolation.
- Visualize the latent space using a 2-dimensional VAE.

---

## Dataset

The experiments use the **MNIST handwritten digit dataset**.

- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28
- Image type: Grayscale
- Pixel values: [0, 1]

The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## VAE Formulation

The VAE assumes the latent-variable model:

$$p_\theta(x, z) = p_\theta(x \mid z)\, p(z)$$

where the prior over the latent variable is:

$$p(z) = \mathcal{N}(0, I)$$

The encoder learns an approximate posterior:

$$q_\phi(z \mid x) = \mathcal{N}\left(\mu_\phi(x),\ \operatorname{diag}(\sigma_\phi^2(x))\right)$$

The VAE is trained by maximizing the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - D_{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right)$$

The implementation therefore consists of two major loss components:

1. Reconstruction loss
2. KL-divergence regularization

---

## Model Architecture

The VAE consists of an encoder, a latent representation, and a decoder.

### Encoder

The input MNIST image is flattened from 28 × 28 into 784 dimensions.

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
 μ            log(σ²)
```

The encoder produces two vectors:

- `mu` — latent mean
- `logvar` — logarithm of latent variance

The default latent dimension is:

```text
Z_DIM = 10
```

### Decoder

The decoder maps the sampled latent representation back to image space:

```text
Latent z (10)
      ↓
Linear(10 → 256)
      ↓
ReLU
      ↓
Linear(256 → 512)
      ↓
ReLU
      ↓
Linear(512 → 784)
      ↓
Sigmoid
      ↓
28 × 28 reconstructed image
```

---

## Reparameterization Trick

Directly sampling from

$$z \sim \mathcal{N}(\mu, \sigma^2)$$

would prevent standard backpropagation through the sampling operation.

The implementation uses the reparameterization trick:

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

with:

$$\sigma = \exp\left(\tfrac{1}{2}\log \sigma^2\right)$$

This separates the randomness from the learnable parameters and allows gradients to propagate through $\mu$ and $\sigma$.

---

# Part 1 — Mathematical Derivations

## Q1.1 — Reconstruction Loss

The decoder likelihood is assumed to be Gaussian:

$$p_\theta(x \mid z) = \mathcal{N}(f_\theta(z), \sigma^2 I)$$

The assignment derives why maximizing the log-likelihood under this assumption corresponds to minimizing **Mean Squared Error (MSE)**.

It also analyzes:

- The effect of using a deterministic decoder.
- The relationship between reconstruction loss and generated image quality.
- An alternative Bernoulli likelihood for MNIST.
- Why Binary Cross-Entropy can be more principled when pixels are modeled as Bernoulli variables.

---

## Q1.2 — Closed-Form KL Divergence

For:

$$q_\phi(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2)) \qquad p(z) = \mathcal{N}(0, I)$$

the closed-form KL divergence is:

$$D_{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right) = -\frac{1}{2}\sum_{j} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

The assignment derives this expression starting from the definition of KL divergence.

---

# Part 2 — VAE Implementation

## Q2.1 — Reparameterization

The VAE implements differentiable latent sampling using the reparameterization trick:

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std
```

This allows gradients to flow through the encoder parameters during backpropagation.

---

## Q2.2 — VAE Loss

The implementation uses **MSE** for reconstruction and the closed-form KL divergence.

### Reconstruction Loss

$$\mathcal{L}_{recon} = \sum_i (x_i - \hat{x}_i)^2$$

### KL Loss

$$\mathcal{L}_{KL} = -\frac{1}{2}\sum_j \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### Total Objective

$$\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{KL}$$

The implementation follows the specified reduction:

```python
.sum(dim=1).mean()
```

which sums over pixels/latent dimensions and then averages over the batch.

---

## Q2.3 — Training

The baseline VAE is trained using:

| Parameter | Value |
|---|---:|
| Batch Size | 128 |
| Latent Dimension | 10 |
| Epochs | 20 |
| Learning Rate | 10⁻³ |
| Optimizer | Adam |
| Reconstruction Loss | MSE |
| KL Coefficient (β) | 1.0 |

The training loop separately records:

- Reconstruction loss
- KL divergence
- Total loss

A loss curve is generated after training.

---

# Part 2.4 — Monte Carlo KL Approximation

Instead of using the analytical KL expression, the assignment also estimates KL divergence using Monte Carlo sampling.

The KL divergence can be written as:

$$D_{KL}(q \,\|\, p) = \mathbb{E}_q\left[\log q(z \mid x) - \log p(z)\right]$$

which can be approximated using $L$ samples:

$$D_{KL}(q \,\|\, p) \approx \frac{1}{L}\sum_{\ell=1}^{L} \left[\log q(z^{(\ell)} \mid x) - \log p(z^{(\ell)})\right]$$

The assignment compares different numbers of Monte Carlo samples:

```text
L = 1
L = 10
L = 100
```

against the closed-form KL baseline.

### Analysis

The experiment demonstrates the trade-off between:

- Computational cost
- Estimator variance
- Number of Monte Carlo samples

Using fewer samples produces a cheaper but noisier estimate, while increasing the number of samples provides a more stable estimate.

The assignment also examines why high-variance KL gradients can be problematic during the early stages of VAE training.

---

# Part 3 — β-VAE and Posterior Collapse

The standard VAE objective is extended using a weighting factor β:

$$\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right] - \beta\, D_{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right)$$

The standard VAE corresponds to $\beta = 1$.

The assignment evaluates:

```text
β = 0.1
β = 1.0
β = 10.0
```

For each value of β, reconstruction and KL losses are recorded and compared.

---

## Posterior Collapse

Posterior collapse occurs when the encoder effectively ignores the input and produces a posterior close to the prior:

$$q_\phi(z \mid x) \approx p(z)$$

The assignment investigates this behavior by measuring the variance of encoder means across the test dataset:

$$\operatorname{Var}_x\left[\mu_\phi(x)\right]$$

A low variance indicates that a latent dimension is relatively inactive.

The variance of each latent dimension is visualized for different β values to investigate how increasing the KL penalty affects the learned latent representation.

---

# Part 4 — Geometry of the Latent Space

## Q4.1 — Linear Interpolation

The trained β = 1 VAE is used to investigate the smoothness of its latent space.

A test image of digit **2** and a test image of digit **7** are encoded to obtain their mean latent vectors $\mu_A$ and $\mu_B$.

Ten interpolated latent vectors are generated using:

$$z_t = (1 - t)\mu_A + t\mu_B, \qquad t \in \{0.0, 0.1, \ldots, 1.0\}$$

Each interpolated latent vector is passed through the decoder to generate an image.

This experiment investigates whether the decoder produces a smooth transition between the two digit representations.

---

## Q4.2 — 2-D Latent Space Visualization

As a bonus experiment, the VAE is retrained using:

```text
Z_DIM = 2
```

The complete MNIST test set is encoded, and the resulting two-dimensional latent representations are visualized according to their digit classes.

This provides an intuitive visualization of:

- Class separation
- Latent-space clustering
- Similarity between digit representations
- Structure learned by the VAE

---

# Results

The baseline VAE was trained for 20 epochs using the MNIST training set.

The recorded training losses were:

```text
Epoch 01 → Reconstruction: 43.5904 | KL: 3.4020 | Total: 46.9924
Epoch 10 → Reconstruction: 19.0243 | KL: 9.8330 | Total: 28.8572
Epoch 20 → Reconstruction: 17.4062 | KL: 10.2918 | Total: 27.6980
```

The reconstruction loss decreased substantially during training, while the KL divergence increased and gradually stabilized.

Overall, the experiments investigate:

- The behavior of the VAE reconstruction and regularization terms.
- The accuracy and variance of Monte Carlo KL estimation.
- The effect of β on the learned latent representation.
- Inactive latent dimensions and posterior collapse.
- Smoothness of the latent space through interpolation.
- Structure and clustering in a two-dimensional latent representation.

---

# Visualizations

The notebook generates several visualizations.

### VAE Training Loss

The training curve displays:

- Reconstruction Loss
- KL Divergence
- Total Loss

### Monte Carlo KL Comparison

The KL comparison evaluates:

- Closed-form KL
- Monte Carlo KL with L = 1
- Monte Carlo KL with L = 10
- Monte Carlo KL with L = 100

### β-VAE Loss Curves

Loss curves are compared for:

```text
β = 0.1
β = 1.0
β = 10.0
```

### Active Latent Dimensions

The variance of the encoder means is used to identify relatively inactive latent dimensions.

### Latent Interpolation

The decoder generates intermediate images between the latent representations of digit 2 and digit 7.

### 2-D Latent Space

A two-dimensional VAE is used to visualize the learned MNIST latent representations.

---

# Requirements

Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

The notebook automatically uses a GPU when CUDA is available:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

A CUDA-enabled GPU is recommended for faster training.

---

# Project Structure

```text
CS772-VAE/
│
├── cs772-hw-2.ipynb
├── vae_loss_curves.png
├── mc_kl_comparison.png
├── beta_vae_loss_curves.png
├── active_dims.png
└── README.md
```

---

# Key Concepts

This assignment provides hands-on implementation and analysis of:

- Variational Autoencoders
- Evidence Lower Bound (ELBO)
- Reconstruction likelihood
- Mean Squared Error
- KL Divergence
- Reparameterization Trick
- Monte Carlo Estimation
- β-VAE
- Posterior Collapse
- Active Latent Dimensions
- Latent-Space Interpolation
- Latent-Space Visualization
- Generative Modeling

---

# References

1. Kingma, D. P. & Welling, M. (2013). **Auto-Encoding Variational Bayes.** ICLR 2014. https://arxiv.org/abs/1312.6114
2. Higgins, I. et al. (2017). **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.** ICLR 2017.
3. Lucas, J. et al. (2019). **Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse.** NeurIPS 2019.Make sure Docker Desktop is installed and running first.

**4. Run it**
```bash
python3 main.py
```

Then just type what you want:
```
> git status
> push my changes to main
```

## What's not done yet

- No logging/history of past commands yet.
- No automated test suite that scores the agent's accuracy — just manual
  testing and unit tests so far.
- No memory between separate runs — each time you start `main.py`, it
  starts fresh.
