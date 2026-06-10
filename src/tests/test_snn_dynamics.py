import torch
from spikingjelly.activation_based import functional, neuron
from src.models import NMNIST_SNN

def test_snn_state_reset():
    """
    Verify the 2 temporal properties of the model : 
    1. forward() should reset the network state to ensure statelessness entre appels.
    2. The neurons should have temporal memory that accumulate activations over time steps, not just react instantaneously.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMNIST_SNN().to(device)
    model.eval()

    batch_size, T, C, H, W = 2, 16, 2, 34, 34
    x = torch.ones(T, batch_size, C, H, W).to(device)

    out_forward_a = model(x)
    out_forward_b = model(x)

    assert torch.equal(out_forward_a, out_forward_b), "Echec: Le forward() ne nettoie pas le reseau."

    functional.reset_net(model)

    out_raw_a = model.network(x).mean(dim=0)
    out_raw_b = model.network(x).mean(dim=0)

    assert not torch.equal(out_raw_a, out_raw_b), "Echec: Les neurones n'ont pas de memoire temporelle."

def test_snn_sparsity_bounds():
    """
    Verify that the sparsity of the SNN remains within reasonable bounds.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMNIST_SNN().to(device)
    model.eval()

    batch_size, T, C, H, W = 4, 16, 2, 34, 34
    torch.manual_seed(67)
    x = (torch.randn(T, batch_size, C, H, W) * 10.0).to(device)

    firing_rates = []

    def hook(module, input, output):
        firing_rates.append(output.detach().mean().item())

    hooks = []
    for m in model.modules():
        if isinstance(m, neuron.LIFNode):
            hooks.append(m.register_forward_hook(hook))

    functional.reset_net(model)
    _ = model(x)

    for h in hooks:
        h.remove()

    avg_firing_rate = sum(firing_rates) / len(firing_rates)
    sparsity = (1.0 - avg_firing_rate) * 100

    assert sparsity < 100.0, f"Reseau mort (Sparsity: {sparsity}%). Aucun spike genere."
    assert sparsity > 0.0, f"Reseau epileptique (Sparsity: {sparsity}%). Tire en continu."
    assert sparsity > 50.0, f"Inefficacite energetique (Sparsity: {sparsity}%). Plus de 50% d'activite."

def test_deterministic_inference():
    """
    Verify that the model produces the exact same output for the same input during evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMNIST_SNN().to(device)
    model.eval()
    
    batch_size, T, C, H, W = 2, 16, 2, 34, 34
    torch.manual_seed(67)
    events = (torch.rand(T, batch_size, C, H, W) > 0.8).float().to(device)
    
    with torch.no_grad():
        out1 = model(events)
        out2 = model(events)
        
    assert torch.equal(out1, out2), "Echec: L'inference n'est pas deterministe."

def test_output_tensor_shape():
    """
    Verify that the final output tensor shape matches (Batch, Num_Classes) regardless of the time steps.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMNIST_SNN().to(device)
    model.eval()
    
    batch_size = 8
    num_classes = 10
    T = 16
    C, H, W = 2, 34, 34
    
    events = torch.zeros((T, batch_size, C, H, W)).to(device)
    
    with torch.no_grad():
        output = model(events)
        
    assert output.shape == (batch_size, num_classes), f"Forme incorrecte: {output.shape} au lieu de ({batch_size}, {num_classes})"