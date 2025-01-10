import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        C.device = 'auto'
        C.num_workers = 4
        C.max_iters = 10000  # Defina conforme necessário
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        C.log_interval = 100  # Iterações entre logs
        C.save_interval = 1000  # Iterações entre salvamentos de checkpoint
        C.checkpoint_path = './checkpoints'
        C.num_classes = 2  # Para classificação binária
        C.system = CfgNode(work_dir='./training_logs')
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = defaultdict(list)
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Rodando no dispositivo:", self.device)
        self.iter_num = 0
        self.iter_time = time.time()
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config)
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                sampler=torch.utils.data.SequentialSampler(self.val_dataset),
                shuffle=False,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        else:
            val_loader = None

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            inputs, labels = batch
            logits = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.iter_num += 1
            self.trigger_callbacks('on_batch_end')

            # Logging
            if self.iter_num % config.log_interval == 0:
                elapsed = time.time() - self.iter_time
                self.iter_dt = elapsed
                self.iter_time = time.time()
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    correct = (predictions == labels).sum().item()
                    accuracy = correct / labels.size(0)
                print(f"Iteração {self.iter_num}: Loss = {loss.item():.4f}, Acurácia = {accuracy*100:.2f}%, Tempo/iter = {elapsed:.2f}s")

            # Validação
            if val_loader and self.iter_num % config.log_interval == 0:
                self.evaluate(val_loader)

            # Salvamento de checkpoint
            if self.iter_num % config.save_interval == 0:
                self.save_checkpoint()

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                if val_loader:
                    self.evaluate(val_loader)
                break

    def evaluate(self, val_loader):
        self.model.eval()
        total_correct = 0
        total = 0
        total_loss = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * inputs.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total += labels.size(0)
        avg_loss = total_loss / total
        accuracy = total_correct / total
        print(f"Validação: Loss = {avg_loss:.4f}, Acurácia = {accuracy*100:.2f}%")
        self.model.train()

    def save_checkpoint(self):
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(self.config.checkpoint_path, f"checkpoint_iter_{self.iter_num}.pt")
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Checkpoint salvo em {checkpoint_file}")
