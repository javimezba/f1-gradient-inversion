# Federated Learning Gradient Inversion

Reconstructing clients' private training images from the gradients they share in federated learning, from the point of view of a malicious server with white-box access to the global models.

Built for the *European Championship in Trustworthy AI 2026* (CISPA Helmholtz Center) with team **MalagAI**.

## The problem

In federated learning, clients never send their data, only gradients. The question is how much of the data those gradients leak.

Setup: 12 global models (MLP, CNN and ViT architectures with different activations), and for each one the gradients of a batch of **128 private 64×64 images**. Some models were initialised with **trap weights** (Boenisch et al.), a malicious initialisation that makes most neurons fire for only one image of the batch. Goal: reconstruct all 1,536 images. Score: average **SSIM**, with each guess matched to a distinct ground-truth image.

## Approach: one attack per architecture

**MLP: analytic attack.** If a neuron in the first linear layer is activated by a single sample, its gradients satisfy `x = ∂L/∂W_k / ∂L/∂b_k`. That is the image itself, recovered exactly, with no optimisation. Trap weights make this happen for many neurons. It is near-perfect with ReLU, and blurrier with sigmoid/tanh, where several samples contribute to each neuron.

**CNN: analytic attack + conv inversion.** The same formula applied to the first fully connected layer returns the *feature map* `act(conv(x))`, not the image. Since the conv weights are known, the conv is then inverted by optimisation:
- stride 1: more equations than unknowns, so the image is recovered almost exactly;
- stride 4/8: information is lost in the downsampling, so the result is a low-frequency approximation.

**ViT: gradient matching.** The patch-embedding bias mixes all patches, so there is no clean analytic attack. The ViT forward pass is rebuilt from its weights, and dummy images are optimised until their gradients match the shared ones (cosine distance + total-variation prior, as in Geiping et al.). Labels are inferred from the output-layer gradients.

**Diverse candidate selection.** The analytic attacks produce many more candidates than images, with lots of near-duplicates. Candidates are ranked by reliability and smoothness, and de-duplicated by similarity *after removing the common mean*, so the 128 guesses cover 128 different images.

## Usage

```bash
pip install -r requirements.txt
python fl_reconstruction.py
```

It expects `models/model{1..12}.pt` and `gradients/model{1..12}.pt` (challenge data, not included) and writes `reconstructions.pt`: a dict of 12 float32 tensors of shape `(128, 3, 64, 64)` in `[0, 1]`. A GPU is recommended for the ViT models.

## Limitations

- The ViT attack is best-effort: gradient matching on a 128-image batch is hard, and results are far below the analytic attacks.
- Sigmoid/tanh MLPs give mixed reconstructions, because the trap-weight isolation works best with ReLU.

## References

- Boenisch et al., *When the Curious Abandon Honesty: Federated Learning Is Not Private*, EuroS&P 2023
- Zhu et al., *Deep Leakage from Gradients*, NeurIPS 2019
- Geiping et al., *Inverting Gradients: How easy is it to break privacy in federated learning?*, NeurIPS 2020
