# Fingerprint Data Analysis
## Background
The article *Unveiling Intra-Person Fingerprint Similarity via Deep Contrastive Learning* found that intra-person fingerprint similarities exist, meaning that the similarity scores among fingerprints from the same person are high. This suggests that even if a system records a fingerprint different from the one presented, the algorithm developed in the article can still determine whether the fingerprints come from the same person. This finding challenges the traditional belief that no two fingerprints are identical.

## Goal
The goal of this project is to 
- Apply dimensionality reduction methods — such as PCA, PPCA, and NMF — to extract low-dimensional embeddings of fingerprints, and then test intra-person similarities using these embeddings. Compare the performance of these methods with the representations extracted by the neural networks used in the article.
  - Perform dimensionality reduction on raw data, binarized data, ridge orientation, and minutiae to evaluate how image details and noise influence the performance of each method. Also, test the article's claim that minutiae are less important.
- Investigate fingerprint clustering — If intra-person fingerprint similarities do exist, explore whether clustering fingerprints is possible. This could be useful for identifying clusters of fingerprints when multiple individuals’ fingerprints are found at the same crime scene, aiding in crime investigations based on fingerprint clusters. This can be done using classical clustering methods, e.g. K-means and spectral clustering, or using mixture models (graphical models). 
- Can we regenerate the fingerprints using the extracted features? How precise are the regenerated fingerprints?

## Citations
- human experts
- Reduce the number of fingerprints that we need to compare against to. [The similarity of two fingerprints can be defined as the Euclidean distance between them in the feature space and query fingerprints are matched against all other prints that fall within a given radius.] https://link.springer.com/article/10.1007/s10044-004-0204-7 [The first type of classifiers are usually called one-vs-all, while classifiers of the second type are called pairwise classifiers. For the one-vs-all a test point is classified into the class whose associated classifier has the highest score among all classifiers. Fingerprint Classification with Combinations of Support Vector Machines https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=fb61dc7bdbb7e57e739648b38207b41e2e41e52b]
- Overcome the challanges. challenges such as lighting variability. Optimization strategies, such as feature extraction, hyperparameter tuning, dataset augmentation, and transfer learning, significantly enhance accuracy and efficiency. https://phoenixpublication.net/index.php/TTI/article/view/3484
- Fingerprint classification: a review https://link.springer.com/article/10.1007/s10044-004-0204-7
- Handbook of Fingerprint Recognition https://link.springer.com/book/10.1007/978-3-030-83624-5
  - Singular points and coarse ridge-line shape are useful for fingerprint classification and indexing, but their distinctiveness is not sufficient for accurate matching. 
  - Matching: The main factors responsible for the intra-class variations are displacement, rotation, partial overlap, nonlinear distortion, variable pressure, changing skin condition, noise, and feature extraction errors. 
  - automatic minutiae-based fingerprint matching is inspired by the manual procedure
  - Feature-based matching: minutiae extraction is difficult in extremely low-quality fingerprint images, whereas other features of the fingerprint ridge pattern (e.g., local orientation and frequency, ridge shape, and texture information) may be extracted more reliably than minutiae, even though their distinctiveness is generally lower.
- The performance of cross-finger matching is lower than that of sam-finger matching
- A Minutiae-based Fingerprint Matching Algorithm Using Phase Correlation https://core.ac.uk/download/pdf/143875633.pdf
- FINGERPRINT RECOGNITION USING MINUTIA SCORE MATCHING https://arxiv.org/ftp/arxiv/papers/1001/1001.4186.pdf
  - 

## Structure of the delivarable
- Introduce ridge orientation, singularity, minutiae and frequency, ridge count...
- Challanges of this analysis: hign dimensional problem
  
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

- Impression Type
  - "Slap" refers to a method of capturing fingerprints by placing four fingers (index to pinky) flat on the scanner at once, rather than rolling one finger at a time. It is a non-rolled, plain impression. Central area only
  - Rolled: The finger is physically rolled from one side to the other. Captures more ridge detail than slap/plain impressions. Full ridge detail
  - Plain: One finger placed flat. Partial (center only)

## Data and Materials Availability
The fingerprint data set is obtained from https://www.nist.gov/itl/iad/image-group/nist-special-database-302.

## Step1:
- Get participants' ID and their finger IDs'.
- For each participant, conduct PCA.
- Conduct hypothesis testing.