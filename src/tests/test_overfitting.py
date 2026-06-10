import hashlib
import pytest
import torch
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from src.data_loaders import get_nmnist_loaders
from src.models import DVSGesture_SNN, NMNIST_SNN


@pytest.fixture(scope="module")
def nmnist_loaders():
    return get_nmnist_loaders(batch_size=8, n_time_bins=10)

@pytest.fixture(scope="module")
def train_loader(nmnist_loaders):
    return nmnist_loaders[0]

@pytest.fixture(scope="module")
def val_loader(nmnist_loaders):
    return nmnist_loaders[1]

def _sample_hash(sample: torch.Tensor) -> str:
    return hashlib.md5(sample.contiguous().cpu().numpy().tobytes()).hexdigest()

def _batch_sample_hashes(events: torch.Tensor) -> set[str]:
    if events.ndim == 5:
        samples = events.transpose(0, 1)
    else:
        samples = events
    return {_sample_hash(sample) for sample in samples}

def _dropout_layers(model: torch.nn.Module):
    return [module for module in model.modules() if "Dropout" in type(module).__name__]

def test_regularization_train_vs_eval():
    """
    Verify that regularization mechanisms (like Dropout) are present, respect
    train()/eval() mode, and remain deterministic in eval() mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMNIST_SNN().to(device)
    
    batch_size, T, C, H, W = 2, 16, 2, 34, 34
    x = torch.rand(T, batch_size, C, H, W).to(device)
    
    model.train()
    dropout_layers = _dropout_layers(model)
    assert dropout_layers, "Aucune couche de Dropout trouvée dans NMNIST_SNN."
    assert all(layer.training for layer in dropout_layers), "Les couches de Dropout doivent être actives en mode train."
    
    model.eval()
    out_eval_1 = model(x)
    functional.reset_net(model)
    out_eval_2 = model(x)
    
    assert all(not layer.training for layer in dropout_layers), "Les couches de Dropout doivent être désactivées en mode eval."
    assert torch.equal(out_eval_1, out_eval_2), "Comportement stochastique anormal en mode eval."

def test_no_data_leakage(train_loader: DataLoader, val_loader: DataLoader):
    """
    Verify that there is strictly zero intersection between the training 
    and validation datasets to prevent artificial validation accuracy.
    """
    train_hashes = set()
    
    for events, _ in train_loader:
        train_hashes.update(_batch_sample_hashes(events))
        
    leakage_count = 0
    
    for events, _ in val_loader:
        leakage_count += sum(
            1 for sample_hash in _batch_sample_hashes(events) if sample_hash in train_hashes
        )
            
    assert leakage_count == 0, f"Fuite de données : {leakage_count} échantillons partagés entre Train et Val."

def test_reproducibility_eval_DVSGesture():
    """
    Verify that DVSGesture_SNN is deterministic in eval() mode.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DVSGesture_SNN().to(device)
    model.eval()

    batch_size, T, C, H, W = 2, 4, 2, 128, 128
    x = torch.rand(T, batch_size, C, H, W).to(device)

    out_eval_1 = model(x)
    functional.reset_net(model)
    out_eval_2 = model(x)

    assert torch.equal(out_eval_1, out_eval_2), "Comportement non déterministe en mode eval."