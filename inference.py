import argparse
from bpe import BPETokenizer
from model import BERT
from utils import CfgNode
import torch

def main():
    parser = argparse.ArgumentParser(description="Inferência com o Modelo BERT para Classificação no IMDB")
    parser.add_argument('--checkpoint', type=str, required=True, help="Caminho para o checkpoint do modelo")
    parser.add_argument('--text', type=str, required=True, help="Texto para inferência")
    parser.add_argument('--block_size', type=int, default=128, help="Comprimento máximo da sequência")
    parser.add_argument('--device', type=str, default='auto', help="Dispositivo para inferência ('cuda', 'cpu', ou 'auto')")
    args = parser.parse_args()

    # Configuração
    config = BERT.get_default_config()
    config.block_size = args.block_size
    config.vocab_size = 50257
    config.num_classes = 2

    # Inicialização do Tokenizer
    tokenizer = BPETokenizer()

    # Inicialização do Modelo
    model = BERT(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    
    # Configuração do Dispositivo
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model.to(device)
    model.eval()

    # Inferência
    tokens = tokenizer(args.text, max_length=config.block_size).to(device)
    with torch.no_grad():
        logits = model(tokens)
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        print(f"Texto: \"{args.text}\"")
        print(f"Sentimento Predito: {sentiment}")

if __name__ == '__main__':
    main()
