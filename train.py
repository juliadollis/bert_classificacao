import os
import argparse
from torch.utils.data import Dataset
from bpe import BPETokenizer
from model import BERT
from trainer import Trainer
from utils import set_seed, CfgNode
import torch

from torch.utils.data import random_split

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, max_length=self.max_length)
        return tokens.squeeze(0), torch.tensor(label, dtype=torch.long)

def load_imdb_dataset(tokenizer, max_length, split_ratio=0.8):
    from torchtext.datasets import IMDB
    import itertools

    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    train_texts = []
    train_labels = []
    for label, text in train_iter:
        train_texts.append(text)
        train_labels.append(1 if label == 'pos' else 0)

    test_texts = []
    test_labels = []
    for label, text in test_iter:
        test_texts.append(text)
        test_labels.append(1 if label == 'pos' else 0)

    # Split o conjunto de treino em treino e validação
    num_train = int(len(train_texts) * split_ratio)
    train_texts, val_texts = train_texts[:num_train], train_texts[num_train:]
    train_labels, val_labels = train_labels[:num_train], train_labels[num_train:]

    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description="Treinamento de BERT para Classificação no IMDB")
    parser.add_argument('--config', type=str, default=None, help="Caminho para arquivo de configuração JSON")
    parser.add_argument('--override', nargs='*', default=[], help="Argumentos para sobrescrever a configuração no formato --arg=value")
    parser.add_argument('--seed', type=int, default=42, help="Semente para reprodutibilidade")
    args = parser.parse_args()

    # Configuração
    config = BERT.get_default_config()
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config.merge_from_dict(config_dict)
    if args.override:
        config.merge_from_args(args.override)
    set_seed(args.seed)

    # Inicialização do Tokenizer
    tokenizer = BPETokenizer()

    # Carregamento do Dataset
    print("Carregando dataset IMDB...")
    train_dataset, val_dataset, test_dataset = load_imdb_dataset(tokenizer, max_length=config.block_size)

    # Inicialização do Modelo
    print("Inicializando o modelo BERT para classificação...")
    model = BERT(config)

    # Inicialização do Trainer
    trainer = Trainer(config, model, train_dataset, val_dataset)

    # Iniciar o Treinamento
    print("Iniciando o treinamento...")
    trainer.run()

    # Avaliar no conjunto de teste
    print("Avaliando no conjunto de teste...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=torch.utils.data.SequentialSampler(test_dataset),
        shuffle=False,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    trainer.evaluate(test_loader)

    # Salvar o modelo final
    os.makedirs(config.checkpoint_path, exist_ok=True)
    final_model_path = os.path.join(config.checkpoint_path, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final salvo em {final_model_path}")

    # Inferência Simples
    sample_text = "This movie was absolutely fantastic! I loved every moment of it."
    print(f"\nRealizando inferência para o texto de exemplo: \"{sample_text}\"")
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(sample_text, max_length=config.block_size).to(config.device)
        logits = model(tokens)
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        print(f"Sentimento Predito: {sentiment}")

if __name__ == '__main__':
    main()
