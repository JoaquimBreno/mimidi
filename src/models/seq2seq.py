class SimpleSeq2Seq(nn.Module):
    def __init__(self, n_tokens, d_model=256, n_layers=2):
        super(SimpleSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.fc = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

    def encode(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

    def decode(self, x, hidden, cell):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(lstm_out)
        return output

    def generate(self, start_token, max_length=100):
        generated = [start_token]
        hidden, cell = self.encode(torch.tensor([[start_token]]))
        
        for _ in range(max_length):
            input_tensor = torch.tensor([[generated[-1]]])
            output = self.decode(input_tensor, hidden, cell)
            next_token = output.argmax(dim=-1).item()
            generated.append(next_token)
            hidden, cell = hidden, cell  # Update hidden and cell states

        return generated