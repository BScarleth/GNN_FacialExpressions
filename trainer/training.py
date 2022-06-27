import torch, time
import torch.nn.functional as F
from .utils import timeSince
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, n_iters, dir, print_every = 10, plot_every = 10):
        self.n_iters = n_iters
        self.dir = dir

        self.print_loss_total = 0  # Reset every print_every
        self.print_acc_total = 0  # Reset every print_every

        self.plot_loss_total = 0  # Reset every plot_every
        self.plot_acc_total = 0  # Reset every plot_every
        self.plot_loss_val_total = 0  # Reset every plot_every

        self.print_every = print_every
        self.plot_every = plot_every

    def train_model(self, dataloader, val_loader, model, optimizer, device="cpu"):
        model.to(device)
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        start = time.time()
        writer = SummaryWriter()

        print(next(model.parameters()).device)
        print("\nTraining started!")
        for iter in range(1, self.n_iters + 1):
            model.train()
            epoch_loss = 0
            for batch in dataloader:
                batch.to(device)
                predictions = model(batch)
                _, indices = torch.max(predictions, dim=1)
                loss = F.cross_entropy(predictions, batch.y)
                epoch_loss += float(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del loss
                batch.to("cpu")
                torch.cuda.empty_cache()
            epoch_acc, epoch_val_loss = self.evaluate(val_loader, model, device)
            self.set_metrics(epoch_loss, epoch_acc, epoch_val_loss, iter, start, dataloader, writer)
        writer.flush()

        torch.save(model.state_dict(), "{}/pointnet_model_{}".format(self.dir, t))

    def set_metrics(self, epoch_loss, epoch_acc, epoch_val_loss, iter, start, dataloader, writer):
        self.print_loss_total += (epoch_loss / len(dataloader))
        self.plot_loss_total += (epoch_loss / len(dataloader))

        self.print_acc_total += epoch_acc
        self.plot_acc_total += epoch_acc

        self.plot_loss_val_total += epoch_val_loss

        if iter % self.print_every == 0:
            print_loss_avg = self.print_loss_total / self.print_every
            self.print_loss_total = 0

            print_acc_avg = self.print_acc_total / self.print_every
            self.print_acc_total = 0

            print('%s (%d %d%%) loss: %.4f acc:  %.4f' % (
                timeSince(start, iter / self.n_iters), iter, iter / self.n_iters * 100, print_loss_avg, print_acc_avg))

        if iter % self.plot_every == 0:
            plot_loss_avg = self.plot_loss_total / self.plot_every
            self.plot_loss_total = 0
            writer.add_scalar("Loss/train", plot_loss_avg, iter)

            plot_acc_avg = self.plot_acc_total / self.plot_every
            self.plot_acc_total = 0
            writer.add_scalar("Acc/train", plot_acc_avg, iter)

            plot_loss_val_avg = self.plot_loss_val_total / self.plot_every
            self.plot_loss_val_total = 0
            writer.add_scalar("Loss/Validation", plot_loss_val_avg, iter)

    def evaluate(self, dataloader, model, device="cpu"):
        model.eval()
        correct = 0

        epoch_loss = 0
        for batch in dataloader:
            batch.to(device)
            with torch.no_grad():
                predictions = model(batch)
                _, indices = torch.max(predictions, dim=1)
            loss = F.cross_entropy(predictions, batch.y)
            epoch_loss += float(loss)

            del loss
            correct += torch.sum(indices == batch.y)
            batch.to("cpu")
            torch.cuda.empty_cache()

        return float(correct/len(dataloader.dataset)), float((epoch_loss / len(dataloader)))

