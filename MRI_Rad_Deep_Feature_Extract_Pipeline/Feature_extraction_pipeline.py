# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:52:01 2026

@author: LIHAP9
"""

import torch
import numpy as np
import SimpleITK as sitk
from torchvision import models, transforms
from radiomics import featureextractor
from neuroCombat import medical_combat

# --- 1. Model Initialization ---

# Deep Learning: Pretrained Vision Transformer (ViT-Base)
# We use a standard ViT and strip the head to get the 768-dim embedding
vit_model = models.vit_b_16(weights="DEFAULT")
vit_model.heads = torch.nn.Identity() 
vit_model.eval()

# Radiomics: Initialize extractor with some basic settings
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder') # Extracts 18 intensity features

# --- 2. Mock Data Generator ---

def generate_mock_3d_data():
    """Generates a mock 3D image and a mask with liver(1) and spleen(2) labels"""
    # Create a 3D volume: 30 slices of 224x224
    img_array = np.random.rand(30, 224, 224).astype(np.float32)
    # Create a mask with two spheres representing liver and spleen
    mask_array = np.zeros((30, 224, 224), dtype=np.int16)
    mask_array[5:15, 50:150, 50:150] = 1 # Mock Liver
    mask_array[20:25, 100:150, 100:150] = 2 # Mock Spleen
    
    image = sitk.GetImageFromArray(img_array)
    mask = sitk.GetImageFromArray(mask_array)
    return image, mask

# --- 3. The Pipeline ---

subject_rad_features = []
subject_deep_features = []
site_labels = [0]*5 + [1]*5 # 5 subjects from Site A, 5 from Site B

for i in range(10):
    print(f"Processing Subject {i+1}/10...")
    
    # --- Load 3D data (Mock) ---
    img_t1, mask_t1 = generate_mock_3d_data()
    img_t2, mask_t2 = generate_mock_3d_data()
    img_b0, mask_b0 = generate_mock_3d_data() # Use b0 for segmentation
    img_adc, _ = generate_mock_3d_data()      # ADC uses b0's mask
    
    subject_modality_data = [
        (img_t1, mask_t1), 
        (img_t2, mask_t2), 
        (img_adc, mask_b0) # Applying b0 mask to ADC map
    ]
    
    rad_parts = []
    deep_parts = []

    for img, mask in subject_modality_data:
        for label in [1, 2]: # Liver=1, Spleen=2
            
            # TRACK A: Radiomics (3D)
            try:
                rad_result = extractor.execute(img, mask, label=label)
                # Filter out diagnostic info, keep only feature values
                feat_values = np.array([float(val) for key, val in rad_result.items() if not key.startswith('diagnostics')])
                rad_parts.append(feat_values)
            except Exception as e:
                rad_parts.append(np.zeros(18)) # Fallback

            # TRACK B: Deep Learning (ViT Slice-wise)
            img_array = sitk.GetArrayFromImage(img)
            mask_array = sitk.GetArrayFromImage(mask)
            slice_embeddings = []
            
            for z in range(img_array.shape[0]):
                if np.any(mask_array[z] == label):
                    # Prepare slice for ViT (resize/normalize/3-channel)
                    slice_2d = img_array[z]
                    # Simple transform: Normalize to [0,1] and stack to 3 channels for ViT
                    slice_rgb = np.stack([slice_2d]*3, axis=0) 
                    slice_tensor = torch.from_numpy(slice_rgb).unsqueeze(0)
                    
                    with torch.no_grad():
                        embedding = vit_model(slice_tensor).numpy() # (1, 768)
                        slice_embeddings.append(embedding)
            
            # Average Pooling across all slices containing the organ
            if slice_embeddings:
                avg_embedding = np.mean(slice_embeddings, axis=0).flatten()
            else:
                avg_embedding = np.zeros(768)
            deep_parts.append(avg_embedding)

    # Concatenate the 6 vectors (3 modalities x 2 organs)
    subject_rad_features.append(np.concatenate(rad_parts))
    subject_deep_features.append(np.concatenate(deep_parts))

# --- 4. Harmonization (ComBat) ---

# Convert to (Features, Subjects) for ComBat
rad_matrix = np.array(subject_rad_features).T
deep_matrix = np.array(subject_deep_features).T

print("\nHarmonizing across sites...")
# medical_combat returns the harmonized matrix
rad_harmonized = medical_combat(rad_matrix, site_labels)
deep_harmonized = medical_combat(deep_matrix, site_labels)

print(f"Pipeline Complete.")
print(f"Final Rad Feature Vector Size: {rad_harmonized.shape[0]} (6 vectors concatenated)")
print(f"Final Deep Feature Vector Size: {deep_harmonized.shape[0]} (6 vectors concatenated)")