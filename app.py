from flask import Flask, request, jsonify, send_file, render_template
import joblib
import pandas as pd
import torch
from torch import nn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Define VAE (same as before)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.mu = nn.Linear(32, latent_dim)
        self.logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Load preprocessor and example dataset
preprocessor = joblib.load("vae_preprocessor.joblib")
new_df = pd.read_csv("student_habits_performance.csv")
X_new_processed = preprocessor.transform(new_df)

# Load VAE model
vae = VAE(input_dim=X_new_processed.shape[1])
vae.load_state_dict(torch.load("vae_model.pth"))
vae.eval()

def generate_synthetic(row_count):
    with torch.no_grad():
        z = torch.randn(row_count, vae.mu.out_features)
        synthetic = vae.decoder(z).numpy()
    return synthetic

def postprocess_synthetic(synthetic):
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    synthetic_num = synthetic[:, :len(num_cols)]
    synthetic_cat = synthetic[:, len(num_cols):]

    numerical_df = pd.DataFrame(
        preprocessor.named_transformers_['num'].inverse_transform(synthetic_num),
        columns=num_cols
    )

    categorical_df = pd.DataFrame(
        preprocessor.named_transformers_['cat'].inverse_transform(synthetic_cat),
        columns=cat_cols
    )

    final_df = pd.concat([numerical_df, categorical_df], axis=1)
    return final_df

@app.route('/generate', methods=['POST'])
def generate():
    try:
        row_count = request.form.get("row_count", default=100, type=int)
        if not (1 <= row_count <= 10000):
            return jsonify({"error": "row_count must be between 1 and 10,000"}), 400

        synthetic = generate_synthetic(row_count)
        final_df = postprocess_synthetic(synthetic)

        output_path = "synthetic_data.csv"
        final_df.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('csv_file')
        if file is None or file.filename == '':
            return jsonify({"error": "No CSV file uploaded"}), 400

        row_count = request.form.get("row_count", default=100, type=int)
        if not (1 <= row_count <= 10000):
            return jsonify({"error": "row_count must be between 1 and 10,000"}), 400

        # Read uploaded CSV
        df = pd.read_csv(file)

        # Validate required columns
        expected_cols = list(preprocessor.transformers_[0][2]) + list(preprocessor.transformers_[1][2])
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Uploaded CSV missing columns: {missing_cols}"}), 400

        # Preprocess uploaded data
        processed = preprocessor.transform(df)

        # Generate synthetic data with matching input dim
        synthetic = generate_synthetic(row_count)

        # Postprocess synthetic data
        final_df = postprocess_synthetic(synthetic)

        output_path = "synthetic_data_uploaded.csv"
        final_df.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
