# MultiModal-CVAE-Alzheimers

Multimodal conditional variational autoencoder for PET-MRI representation learning and individualized abnormality detection in Alzheimer's disease.

# Multimodal CVAE for Alzheimer's Disease

This repository contains code developed during my research internship (8CP) to complete my master education in cognitive science at Rheinland-Pfälzische Technische Universität (RPTU), Germany
conducted at Institute of Neuroscience and Medicine – Brain and Behaviour (INM-7), Cognitive and Clinical Neuroscience research group. Forschungszentrum Jülich (FZJ), for multimodal neuroimaging analysis
in Alzheimer's disease.

The project investigates a multimodal conditional variational
autoencoder (CVAE) for jointly modelling PET and structural MRI
features.

## Overview

The model uses separate encoders for the two imaging modalities and
combines their latent representations using a Product of Experts
(PoE) approach.

Demographic information, including age and sex, is incorporated as
conditioning information.

The pipeline includes:

- PET and MRI feature preparation
- demographic encoding
- multimodal CVAE modelling
- Product of Experts latent representation
- reconstruction-based analysis
- 5-fold cross-validation
- reconstruction error analysis
- individualized deviation analysis
- Z-score based identification of abnormal regions

## Model

The multimodal CVAE contains:

1. A modality-specific encoder for PET features
2. A modality-specific encoder for MRI features
3. A Product of Experts to combine the latent representations
4. A shared latent representation
5. Separate decoders for PET and MRI reconstruction

## Data

The datasets used in this project are not included in this repository.

They are subject to access restrictions and applicable data-use
agreements.
