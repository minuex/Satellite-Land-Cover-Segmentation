from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def log_train_loss(loss, epoch, step, train_loader):
    writer.add_scalar("Loss/train", loss, epoch * len(train_loader) + step)

def log_validation_loss(loss, epoch):
    writer.add_scalar("Loss/valid", loss, epoch)

def close_writer():
    writer.close()
