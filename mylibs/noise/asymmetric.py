import numpy as np

# Flowers flip pairs — only one direction per pair to simulate annotator confusion.
# Visually similar classes: rose→tulip, daisy→sunflower, dandelion→daisy.
FLOWERS_FLIP_PAIRS = {
    1: 2,   # rose      → tulip     
    2: 1,   # tulip     → rose   
    0: 3,   # daisy     → sunflower 
    3: 0,   # sunflower → daisy     
    1: 0,   # dandelion → daisy     
    0: 1,   # daisy → dandelion     
    1: 3,   # dandelion → sunflower
    3: 1,   # sunflower → dandelion 
}


def asymmetric_noise(
    labels: np.ndarray,
    noise_rate: float,
    flip_pairs: dict = FLOWERS_FLIP_PAIRS,
    seed: int = 42,
) -> np.ndarray:
    """
    Asymmetric noise: labels only flip within confusable pairs.
    More realistic than uniform — annotators confuse similar-looking flowers.
    """
    rng   = np.random.default_rng(seed)
    noisy = labels.copy()
    for i in range(len(labels)):
        if labels[i] in flip_pairs and rng.random() < noise_rate:
            noisy[i] = flip_pairs[labels[i]]
    return noisy
