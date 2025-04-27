# Fingerprint Data Analysis

## Background
The article *Unveiling Intra-Person Fingerprint Similarity via Deep Contrastive Learning* discovered that intra-person fingerprint similarities do exist, meaning that fingerprints from the same person tend to have high similarity scores. This suggests that even when a system records a slightly different fingerprint than the one presented, the algorithm developed in the article can still identify whether the fingerprints originate from the same individual. This finding challenges the traditional belief that no two fingerprints are identical.

## Goals
This project aims to:
- Apply dimensionality reduction techniques — including PCA, PPCA, and NMF — to extract low-dimensional embeddings of fingerprints, and evaluate intra-person similarities using these embeddings.
  - Perform dimensionality reduction on various fingerprint representations (raw images, binarized images, ridge orientation maps, and minutiae) to assess how image details and noise impact performance. This also serves to verify the article's claim that minutiae are less critical.
- Investigate fingerprint clustering:
  - If intra-person similarities exist, explore whether fingerprints can be clustered effectively.
  - Such clustering could assist forensic investigations by grouping fingerprints when multiple individuals’ prints are found at the same crime scene.
  - Methods: classical clustering (e.g., K-means, spectral clustering) and mixture models (graphical models).
- Explore fingerprint regeneration:
  - Investigate whether fingerprints can be reconstructed from extracted features and evaluate the precision of regenerated fingerprints.

## Key References and Related Work
- **Similarity and Matching:**
  - Reducing the number of comparisons via Euclidean distances in feature space: [Fingerprint classification using SVM combinations](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=fb61dc7bdbb7e57e739648b38207b41e2e41e52b).
- **Challenges in Fingerprint Recognition:**
  - Addressing challenges such as lighting variability, noise, and acquisition artifacts using feature extraction, hyperparameter tuning, augmentation, and transfer learning: [Optimization strategies article](https://phoenixpublication.net/index.php/TTI/article/view/3484).
- **Fingerprint Classification and Matching:**
  - Comprehensive review: [Fingerprint classification: A review](https://link.springer.com/article/10.1007/s10044-004-0204-7).
  - *Handbook of Fingerprint Recognition*:
    - Singular points and coarse ridge-line shapes assist classification but are insufficient for accurate matching.
    - Intra-class variations stem from displacement, rotation, pressure variability, skin condition changes, and noise.
    - Minutiae extraction is critical but difficult for low-quality images; local orientation and ridge frequency may be more reliably extracted.
- **Cross-Finger Matching:**
  - Performance on cross-finger matching is significantly lower than on same-finger matching tasks.
- **Algorithms and Techniques:**
  - [Minutiae-based matching using phase correlation](https://core.ac.uk/download/pdf/143875633.pdf).
  - [Fingerprint recognition using minutia score matching](https://arxiv.org/ftp/arxiv/papers/1001/1001.4186.pdf).

## Project Structure
- Introduction to fingerprint features:
  - Ridge orientation
  - Singularity points
  - Minutiae
  - Ridge frequency and ridge count
- Challenges:
  - High-dimensional feature spaces
  - Noise sensitivity and acquisition variability

## Data Description
The dataset used is the [NIST Special Database 302](https://www.nist.gov/itl/iad/image-group/nist-special-database-302), containing the following subsets:

| Dataset | Description | Size |
|--------|-------------|------|
| SD 302a | Challenger rolled friction ridge images (PNG) | 2 GB |
| SD 302b | Operator-assisted rolled fingerprints and 4-4-2 slap impressions | 4.5 GB |
| SD 302c | Palm images and segmented fingerprint images | 11 GB |
| SD 302d | Plain fingerprint images from auxiliary devices | 465 MB |
| SD 302e | Latent distal phalanx images | 5.3 GB |
| SD 302f | Unprocessed photographs from prototype devices | 63 GB |
| SD 302g | Annotated exemplar IRR transactions | 860 MB |
| SD 302h | Annotated latent LFFS transactions | 2.9 GB |
| SD 302i | Latent COMP transactions | 5.2 GB |

### Impression Types
- **Slap:** Four fingers placed flat together; captures central area only.
- **Rolled:** Finger rolled nail-to-nail; captures full ridge detail.
- **Plain:** Single finger placed flat; captures central region.

## Data and Materials Availability
- The fingerprint dataset is publicly available at:  
  [NIST Special Database 302](https://www.nist.gov/itl/iad/image-group/nist-special-database-302).

## Project Workflow
### Step 1:
- Extract participant IDs and corresponding finger IDs.
- For each participant:
  - Perform PCA on extracted fingerprint features.
  - Conduct hypothesis testing on intra-person fingerprint similarities.