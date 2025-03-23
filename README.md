# Fingerprint Data Analysis
## Background
The article *Unveiling Intra-Person Fingerprint Similarity via Deep Contrastive Learning* found that intra-person fingerprint similarities exist, meaning that the similarity scores among fingerprints from the same person are high. This suggests that even if a system records a fingerprint different from the one presented, the algorithm developed in the article can still determine whether the fingerprints come from the same person. This finding challenges the traditional belief that no two fingerprints are identical.

## Goal
The goal of this project is to 
- Apply dimensionality reduction methods — such as PCA, PPCA, and NMF — to extract low-dimensional embeddings of fingerprints, and then test intra-person similarities using these embeddings. Compare the performance of these methods with the representations extracted by the neural networks used in the article.
  - Perform dimensionality reduction on raw data, binarized data, ridge orientation, and minutiae to evaluate how image details and noise influence the performance of each method. Also, test the article's claim that minutiae are less important.
- Investigate fingerprint clustering — If intra-person fingerprint similarities do exist, explore whether clustering fingerprints is possible. This could be useful for identifying clusters of fingerprints when multiple individuals’ fingerprints are found at the same crime scene, aiding in crime investigations based on fingerprint clusters.
  
## Data
SD 302a: Challenger rolled friction ridge images (PNG).  [2 GB]
SD 302b: Operator-assisted rolled fingerprint impressions and 4-4-2 slap impressions ("baseline") (PNG). [4.5 GB]
SD 302c: Palm images and fingerprint images segmented from upper palms (PNG). [11 GB]
SD 302d: Plain fingerprint images from auxiliary devices (PNG). [465 MB]
SD 302e: Latent distal phalanx images (PNG). [5.3 GB]
SD 302f: Unprocessed photographs from Challenger T's prototype device (JPEG). [63 GB]
SD 302g: Exemplar IRR transactions annotated to EFS Profile 2 (EBTS). [860 MB]
SD 302h: Latent LFFS transactions annotated to EFS Profile 2 (EBTS). [2.9 GB]
SD 302i: Latent COMP transactions (302h to 302g) (EBTS). [5.2 GB]

## Data and Materials Availability
The fingerprint data set is obtained from https://www.nist.gov/itl/iad/image-group/nist-special-database-302.