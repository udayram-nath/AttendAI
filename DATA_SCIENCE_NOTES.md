# AttendAI — Data Science Deep Dive

## Face Recognition Pipeline Analysis

### 1. Face Detection

**Technique**: Haar Cascade Classifiers (OpenCV)
- Cascade of boosted classifiers trained on positive/negative face samples
- AdaBoost algorithm for feature selection
- ~24×24 pixel minimum detection window

**Parameters**:
```python
scaleFactor = 1.1      # Image pyramid scale
minNeighbors = 4       # Quality threshold (higher = stricter)
minSize = (30, 30)     # Minimum face dimensions
```

**Trade-offs**:
- Lower `scaleFactor` (1.05) → Better detection, slower
- Higher `minNeighbors` (5+) → Fewer false positives

### 2. Face Alignment

**Why?** Neural networks expect normalized, aligned faces
- Translation: Center face in frame
- Rotation: Align face to frontal position
- Scaling: Consistent input size (224×224 for DeepFace)

### 3. Face Encoding (Embedding)

**DeepFace (FaceNet)**:
- Input: 224×224 RGB image
- Deep CNN: 25+ layers
- Output: 128-dimensional vector
- Training: Triplet loss on 500K+ celebrity faces
- Learned representation: Face identity in vector space

**Vector Space Properties**:
- Same person → Vectors close (cosine ~ 1.0)
- Different people → Vectors far (cosine ~ -1.0)
- **Distance metric**: Cosine similarity

### 4. Face Matching

**Algorithm**: Nearest Neighbor Search
```python
def match_face(unknown_encoding, known_encodings):
    # Compute distances
    distances = [cosine_distance(unknown, known) for known in known_encodings]
    
    # Find closest match
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    
    # Decision threshold
    if min_distance < THRESHOLD:
        return known_identities[min_index]
    else:
        return "Unknown"
```

**Threshold Selection**:
- THRESHOLD = 0.35 (cosine)
- Balances False Positive vs False Negative
- ROC curve analysis: Find optimal operating point

### 5. Confirmation Strategy

**Problem**: Single-frame detection unreliable

**Solution**: Multi-frame confirmation
```python
CONFIRM_FRAMES = 3  # Require 3 consecutive matches

# Benefits:
# - Reduces false positives by ~90%
# - Increases latency by ~100ms
# - Confidence score increases with streak
```

## Statistical Analysis

### Attendance Distribution

**Metrics to Track**:
```
Daily Attendance Rate = (Present / Total) × 100
Attendance Trend = 7-day moving average
Absentee Pattern = Same day weekly?
```

**Outlier Detection**:
- Students absent 4+ days consecutively → Flag
- Unusual attendance time → Alert

### Performance Evaluation

**Confusion Matrix**:
```
                  Predicted
                  Match  No-Match
Actual  Match     TP      FN
        No-Match  FP      TN

Accuracy = (TP + TN) / Total
Precision = TP / (TP + FP)     # "Is match correct?"
Recall = TP / (TP + FN)        # "Did we find all matches?"
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**ROC Curve**:
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Find threshold maximizing TPR - FPR

## Data Quality Issues

### 1. Imbalanced Classes
- More "Absent" than "Present" historically
- **Solution**: Weighted loss, oversampling, threshold adjustment

### 2. Seasonal Variations
- Higher absence during exams, holidays
- **Solution**: Separate baselines per semester

### 3. Lighting Conditions
- Poor lighting → Poor embeddings
- **Solution**: Image preprocessing (histogram equalization)

## Optimization Techniques

### Hardware Acceleration

**GPU Deployment**:
```bash
# CUDA-enabled DeepFace
import torch
torch.cuda.is_available()  # Check GPU
```
- Speedup: 10-20× faster encoding
- Memory tradeoff: ~2GB additional

### Vectorized Operations

```python
# Slow (Python loop)
for known_enc in known_encodings:
    dist = cosine_distance(unknown, known_enc)

# Fast (NumPy vectorized)
distances = cosine_distances([unknown], known_encodings)[0]
```

### Caching Strategy

```python
known_faces_cache = load_known_faces()  # Once at startup
# Reuse for all frames → 1000× faster than reload each time
```

## Advanced Topics

### Adversarial Robustness

**Threat**: Adversarial examples (perturbations that fool model)

**Defense**:
- Ensemble multiple models
- Liveness detection (prevent spoofing)
- Multi-biometric fusion

### Privacy Considerations

**GDPR Compliance**:
- Store only embeddings (not raw faces)
- Implement face deletion workflow
- Audit access logs

### Scalability

**Current Architecture**:
- 1000 students: ~5 seconds matching

**Scaling to 10,000+**:
- Use approximate nearest neighbor (ANN)
- FAISS library: Search 1M vectors in milliseconds
- Partition by department/class

## Research Papers Implemented

1. **Schroff et al. (2015)** - FaceNet: Triplet loss learning
2. **Kanade et al. (1973)** - Haar Cascades foundation
3. **He et al. (2016)** - ResNet architecture (DeepFace backbone)
4. **Parkhi et al. (2015)** - VGGFace embeddings

---

*Last Updated: 2024 | Data Science Notes for Academic Reference*