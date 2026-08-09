import torch


def save_checkpoint(optimizer, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dict = torch.load(filename, map_location=device)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch