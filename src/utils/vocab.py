class RemiVocab:
    def __init__(self, tokens=None):
        if tokens is None:
            tokens = []
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(tokens)}
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.add_token(self.pad_token)
        self.add_token(self.eos_token)
        self.add_token(self.unk_token)

    def add_token(self, token):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def to_i(self, token):
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def to_token(self, idx):
        return self.id_to_token.get(idx, self.unk_token)

    def encode(self, sequence):
        return [self.to_i(token) for token in sequence]

    def decode(self, indices):
        return [self.to_token(idx) for idx in indices]