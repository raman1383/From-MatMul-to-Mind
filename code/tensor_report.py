




# ---

# Create sample tensors matching training anomalies
w = torch.randn(1024, 1024) * 0.000001  # Defective init
act = torch.randn(32, 128, 768) * 12.0  # Exploding activation
grad_vanish = torch.randn(512, 512) * 1e-9  # Vanishing gradient

# Inspect
inspect_tensor(w, name="Linear_Proj_Weight", category="weight").print_summary()

inspect_tensor(
    act, name="Transformer_Layer1_Act", category="activation"
).print_summary()

inspect_tensor(
    grad_vanish, name="Embedding_Grad", category="gradient"
).print_summary()