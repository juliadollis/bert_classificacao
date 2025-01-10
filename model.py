import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import CfgNode as CN

class NewGELU(nn.Module):
    def forward(self, x):
        # Implementação oficial do GELU, aproximada
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):
    """
    Atenção padrão (não causal), típica de BERT.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        # Projeções lineares para Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reformata para múltiplas cabeças
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Produto interno para obter pontuações de atenção
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Se houver um attention_mask (1 onde há token, 0 onde é padding), expandimos para (B, 1, 1, T)
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            expanded_mask = attention_mask[:, None, None, :]
            # Preenche com -inf onde a máscara é 0 (token de padding)
            att = att.masked_fill(expanded_mask == 0, float('-inf'))

        # Normaliza as pontuações
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # Aplica atenção
        y = att @ v

        # Rearranja de volta para (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Projeção final
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """
    Bloco Transformer básico: Self-Attention + MLP + skips + LayerNorm.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class BertForSequenceClassification(nn.Module):
    """
    Modelo BERT simplificado, mas agora voltado para classificação.
    Em vez de prever o próximo token, teremos uma cabeça de classificação binária.
    """
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'bert'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.num_labels = 2  # default para classificação binária (ex: IMDB)
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        
        # Ajusta parâmetros, se necessário
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given, "Defina config.model_type OU (n_layer, n_head, n_embd)."

        if type_given:
            # Exemplo de presets. Ajuste conforme achar melhor
            config.merge_from_dict({
                'gpt-mini':   dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':  dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':   dict(n_layer=3, n_head=3, n_embd=48),
                # 'bert': ... etc. 
            }.get(config.model_type, {}))

        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.num_labels = config.num_labels

        # Transformer base
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Cabeça de classificação final
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

        self.apply(self._init_weights)

        # Ajuste especial de inicialização para c_proj em atenção
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters()) + sum(p.numel() for p in self.classifier.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids: (B, T)
        attention_mask: (B, T) com 1 para tokens válidos e 0 para padding
        labels: (B,) com o rótulo (0 ou 1)
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.block_size, f"Sequência de tamanho {t} mas block_size é {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        tok_emb = self.transformer.wte(input_ids)      # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)            # (1, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Passa por todos os blocos Transformer
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)

        # Normalização final
        x = self.transformer.ln_f(x)

        # Extraímos apenas o embedding do primeiro token [CLS] (aqui assumimos idx=0)
        # ou poderíamos ter um embedding especial, mas simplificamos:
        cls_embedding = x[:, 0, :]  # shape: (B, n_embd)

        # Logits de classificação
        logits = self.classifier(cls_embedding)  # shape: (B, num_labels)

        # Se os rótulos forem fornecidos, calculamos a perda
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss
