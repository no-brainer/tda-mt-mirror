import gc
import os

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import MetricTracker, inf_loop


class Trainer:

    def __init__(self, model, criterion, optimizer, device, train_dataloader, val_dataloader,
                 scheduler, writer, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.cur_epoch = kwargs.get("start_epoch", 1)
        self.num_epochs = kwargs.get("num_epochs", 100)

        self.len_epoch = kwargs.get("len_epoch", None)
        if self.len_epoch is None:
            self.len_epoch = len(train_dataloader)
            self.train_dataloader = train_dataloader
        else:
            self.train_dataloader = inf_loop(train_dataloader)

        self.val_dataloader = val_dataloader

        self.scheduler = scheduler

        self.max_gradient_norm = kwargs.get("max_gradient_norm", 10.0)

        self.device = device

        self.save_step = kwargs.get("save_epoch_step", 5)
        self.save_dir = kwargs.get("save_dir", "./saved")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        self.log_step = kwargs.get("log_step", 50)
        self.writer = writer

        self.train_metrics = MetricTracker(["loss", "grad_norm"])
        self.test_metrics = MetricTracker(["loss"])

    @staticmethod
    def move_batch_to_device(batch, device):
        for key in ["src_enc", "trg_enc"]:
            batch[key] = batch[key].to(device)
        return batch

    def train(self):
        try:
            self._train()
        except KeyboardInterrupt as e:
            print("Interrupted by user. Saving checkpoint...")
            self._save_checkpoint()
            raise e

    def warmup_allocation(self):
        """
        Following https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
        """
        max_src_len, max_trg_len = 0, 0
        for data in self.train_dataloader.dataset:
            if max_src_len < len(data["src_enc"]):
                max_src_len = len(data["src_enc"])

            if max_trg_len < len(data["trg_enc"]):
                max_trg_len = len(data["trg_enc"])

        batch_size = len(next(iter(self.train_dataloader))["src"])

        batch = {
            "src_enc": torch.randint(0, 10, (batch_size, max_src_len)),
            "trg_enc": torch.randint(0, 10, (batch_size, max_trg_len)),
        }

        batch = self.move_batch_to_device(batch, self.device)
        batch["logits"] = self.model(**batch)
        loss = self.criterion(**batch)
        loss.backward()

        self.model.zero_grad()

    def _train(self):
        if self.device == "cuda":
            self.warmup_allocation()

        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.num_epochs + 1):
            self._train_epoch(epoch)

            if epoch % self.save_step == 0:
                self._save_checkpoint()

            self.cur_epoch += 1

    def process_batch(self, batch, metrics, is_training):
        batch = self.move_batch_to_device(batch, self.device)

        outputs = self.model(**batch)

        batch["loss"] = self.criterion(**batch)
        batch["lengths"] = batch["trg_length"]
        batch["log_probs"] = F.log_softmax(outputs[:, :, :-1], dim=-1)

        if is_training:
            batch["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        metrics.update("loss", batch["loss"].item())

        return batch

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader, 1):
            if batch_idx > self.len_epoch:
                break

            self.optimizer.zero_grad()
            try:
                batch = self.process_batch(batch, self.train_metrics, is_training=True)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise e

                gc.collect()
                print("Skipping batch --- OOM")
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad

                torch.cuda.empty_cache()
                continue

            self.train_metrics.update("grad_norm", self.get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, "train")
                self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0])
                for metric_name, value in self.train_metrics.all_avg().items():
                    self.writer.add_scalar(metric_name, value)

                self._log_predictions(**batch)
                self.train_metrics.reset()

        self._valid_epoch(epoch)

    def _valid_epoch(self, epoch):
        self.model.eval()
        refs = []
        preds = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = self.process_batch(batch, self.test_metrics, is_training=False)

                predictions = self.decode_predictions(batch["log_probs"], batch["lengths"])
                refs.extend(batch["trg"])
                preds.extend(predictions)

            self.writer.set_step(epoch * self.len_epoch, "val")

            for metric_name, value in self.test_metrics.all_avg().items():
                self.writer.add_scalar(metric_name, value)

            self._log_predictions(**batch)
            self.test_metrics.reset()

            bleu = sacrebleu.BLEU(sacrebleu.Namespace(tokenize=None, force=True))
            bleu_data = bleu.corpus_score(preds, [refs])
            self.writer.add_scalar("bleu", bleu_data.score)

    def _log_predictions(self, src, trg, log_probs, lengths, num_examples=5, *args, **kwags):
        predictions = self.decode_predictions(log_probs[:num_examples], lengths[:num_examples])

        data = list(zip(src[:num_examples], trg[:num_examples], predictions))
        lines = [
            "<tr> <th>Source</th> <th>Reference</th> <th>Prediction</th> </tr>"
        ]
        for source, reference, prediction in data:
            lines.append(
                f"<tr> <td>{source}</td> <td>{reference}</td> <td>{prediction}</td> </tr>"
            )
        lines = "\n".join(lines)
        lines = f"<table> {lines} </table>"

        self.writer.add_text("predictions", lines)

    def _save_checkpoint(self):
        state = {
            "epoch": self.cur_epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        filename = os.path.join(self.save_dir, f"checkpoint_epoch{self.cur_epoch}.pth")
        print(f"Saving checkpoint to {filename}")
        torch.save(state, filename)

    def decode_predictions(self, log_probs, lengths):
        idx = [
            logits[:length].argmax(dim=-1)
            for logits, length in zip(log_probs, lengths)
        ]
        return self.train_dataloader.dataset.tokenizer.decode_trg(idx)

    @torch.no_grad()
    def get_grad_norm(self):
        params = [p for p in self.model.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2).cpu() for p in params]), p=2)
        return total_norm.item()
