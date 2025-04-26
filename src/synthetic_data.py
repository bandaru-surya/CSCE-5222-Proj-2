import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from src.vae_model import VAE, vae_loss_function, DataNormalizer

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data_for_vae(df):
    """Encode categorical variables for VAE training"""
    df_encoded = df.copy()
    le_stage = LabelEncoder()
    le_treatment = LabelEncoder()
    df_encoded['Stage'] = le_stage.fit_transform(df['Stage'])
    df_encoded['TreatmentType'] = le_treatment.fit_transform(df['TreatmentType'])
    return df_encoded, le_stage, le_treatment

def train_vae(data, epochs=100, batch_size=32, learning_rate=1e-3):
    """Train VAE model on the data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = data.shape[1]
    vae = VAE(input_dim=input_dim).to(device)
    normalizer = DataNormalizer()
    
    normalizer.fit(data)
    normalized_data = normalizer.transform(data)
    tensor_data = torch.FloatTensor(normalized_data).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i+batch_size]
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae_loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(tensor_data):.4f}')
    
    return vae, normalizer

def generate_original_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic patient data using statistical distributions"""
    np.random.seed(seed)
    
    # Generate realistic patient data
    ages = np.random.normal(loc=68, scale=10, size=n_samples).clip(40, 90)
    tumor_size = np.random.lognormal(mean=1.5, sigma=0.6, size=n_samples)
    ca19_9 = np.random.lognormal(mean=5, sigma=1.2, size=n_samples)
    stages = np.random.choice(['I', 'II', 'III', 'IV'], size=n_samples, p=[0.1, 0.2, 0.35, 0.35])
    treatment = np.random.choice(['Surgery', 'Chemo', 'Radio', 'Palliative'], 
                               size=n_samples, p=[0.25, 0.45, 0.15, 0.15])
    
    # Generate survival times based on stage
    survival = []
    for s in stages:
        base = np.random.exponential(scale=18)
        factor = {'I':1.2, 'II':0.9, 'III':0.6, 'IV':0.35}[s]
        surv = max(1, np.round(base * factor + np.random.normal(0, 2), 1))
        survival.append(surv)
    
    return pd.DataFrame({
        'Age': np.round(ages, 1),
        'TumorSize_cm': np.round(tumor_size, 2),
        'CA19_9': np.round(ca19_9, 1),
        'Stage': stages,
        'TreatmentType': treatment,
        'SurvivalMonths': survival
    })

def generate_synthetic_patients(original_df: pd.DataFrame, n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic patient data using VAE"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Train VAE on original data
    df_encoded, le_stage, le_treatment = prepare_data_for_vae(original_df)
    vae, normalizer = train_vae(df_encoded.values, epochs=150)
    
    # Generate new samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 10).to(device)
        synthetic_data = vae.decode(z)
        synthetic_data = normalizer.inverse_transform(synthetic_data.cpu().numpy())
    
    # Convert to DataFrame and clean up
    synthetic_df = pd.DataFrame(synthetic_data, columns=df_encoded.columns)
    synthetic_df['Age'] = synthetic_df['Age'].clip(40, 90).round(1)
    synthetic_df['TumorSize_cm'] = synthetic_df['TumorSize_cm'].clip(0.1, 15).round(2)
    synthetic_df['CA19_9'] = synthetic_df['CA19_9'].clip(0, 10000).round(1)
    synthetic_df['SurvivalMonths'] = synthetic_df['SurvivalMonths'].clip(1, 60).round(1)
    
    # Convert categorical variables back
    synthetic_df['Stage'] = np.clip(np.round(synthetic_df['Stage']), 0, len(le_stage.classes_) - 1).astype(int)
    synthetic_df['TreatmentType'] = np.clip(np.round(synthetic_df['TreatmentType']), 0, len(le_treatment.classes_) - 1).astype(int)
    synthetic_df['Stage'] = le_stage.inverse_transform(synthetic_df['Stage'])
    synthetic_df['TreatmentType'] = le_treatment.inverse_transform(synthetic_df['TreatmentType'])
    
    return synthetic_df

if __name__ == "__main__":
    print("Generating original dataset...")
    original_df = generate_original_data(1000)
    original_df.to_csv(DATA_DIR / "synthetic_data.csv", index=False)
    print("Original dataset saved to data/synthetic_data.csv")
    print(f"Generated {len(original_df)} original samples")
    
    print("\nGenerating VAE-augmented dataset...")
    vae_df = generate_synthetic_patients(original_df, 5000)
    vae_df.to_csv(DATA_DIR / "vae_augmented_data.csv", index=False)
    print("VAE-augmented dataset saved to data/vae_augmented_data.csv")
    print(f"Generated {len(vae_df)} VAE-augmented samples")
