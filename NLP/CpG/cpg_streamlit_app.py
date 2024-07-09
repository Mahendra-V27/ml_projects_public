import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn

# Define model classes
class CpGPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=1, output_size=1):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.classifier(out[:, -1, :])
        return out

class CpgEmbPred(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout_prob=0.2):
        super(CpgEmbPred, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings)
        output = self.linear(lstm_out[:, -1, :])
        return output

# Define function to load models
def load_model(model_name, model_paths):
    if model_name in ["CPG Prediction Model", "CPG Prediction Model Padded"]:
        model = CpGPredictor()
    elif model_name in ["CP Count Model Padding with Embedding", "CP Prediction Embedding Model"]:
        model = CpgEmbPred(input_size=6, embedding_dim=16, hidden_size=128, num_layers=3, dropout_prob=0.2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    state_dict = torch.load(model_paths[model_name])
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    else:
        model = state_dict  # If state_dict is already a model, return it directly
    model.eval()
    return model

# Model paths
model_paths = {
    "CPG Prediction Model": "CP_Pred_model.pth",
    "CPG Prediction Model Padded": "CPG_Pred_model_padded.pth",
    "CP Count Model Padding with Embedding": "CPG_count_model_padding_with_emb.pth",
    "CP Prediction Embedding Model": "CP_pred_emb_model.pth"
}

# Define prediction function
def predict(input_string, model_name):
    model = load_model(model_name, model_paths)

    if "Embedding" in model_name:
        nucleotide_to_index = {'N': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, 'pad': 0}

        def string_to_tensor(s, mapping):
            max_length = 128
            indices = [mapping[c] for c in s]
            padding_length = max_length - len(indices)
            indices.extend([0] * padding_length)
            tensor = torch.LongTensor([indices])
            return tensor
    else:
        nucleotide_to_index = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

        def string_to_tensor(s, mapping):
            indices = [mapping[c] for c in s]
            tensor = torch.tensor(indices)
            one_hot = F.one_hot(tensor, num_classes=len(mapping))
            one_hot = one_hot.unsqueeze(0).float()
            return one_hot

    test_tensor = string_to_tensor(input_string, nucleotide_to_index)

    if "Embedding" in model_name:
        embeddings = model.embedding(test_tensor)
        lstm_out, _ = model.lstm(embeddings)
        output = model.linear(lstm_out[:, -1, :])
    else:
        model.eval()
        with torch.no_grad():
            output = model(test_tensor)

    if isinstance(output, torch.Tensor):
        output = output.squeeze().item()
    return output

# Streamlit app
def main():
    st.title("DNA Sequence Prediction App")
    input_string = st.text_input("Enter DNA Sequence:")
    model_name = st.selectbox("Select Model:", list(model_paths.keys()))

    if st.button("Predict"):
        prediction = predict(input_string, model_name)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
