import torch
import tonic
import tonic.transforms as transforms

def get_dvs_gesture_loaders(batch_size=16, n_time_bins=10):
    tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
    tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    frame_transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
    ])

    train_set = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=frame_transform)
    test_set = tonic.datasets.DVSGesture(save_to='./data', train=False, transform=frame_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=False)
    )
    
    return train_loader, test_loader