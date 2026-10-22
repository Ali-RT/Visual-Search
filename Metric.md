# Recommendation Metrics — Text Definitions + Formulas + Sarah (K=5) Calculations

## Use case (single-user example)
- **User:** Sarah  
- **Ground truth (future positives)** `G` = {Inception, The Matrix, Interstellar}  → |G| = 3  
- **Top-5 recommendations** `R5` = [Toy Story, Inception, Up, The Matrix, Batman]  
- **Hits in Top-5** = {Inception (rank 2), The Matrix (rank 4)} → hits = 2  
- **Binary relevance vector** over ranks 1..5: `rel = [0, 1, 0, 1, 0]`

---

## A) Retrieval (Candidate Generation) — Accuracy Metrics

### 1) Recall@K
**Definition:** Of all items the user truly liked in the test period, what fraction appear anywhere in the top-K list?  
**Formula:**
\[
\text{Recall@K}=\frac{|G \cap R_K|}{|G|}
\]
**Sarah (K=5):**
\[
\text{Recall@5}=\frac{2}{3}=0.6667
\]

---

### 2) Precision@K
**Definition:** Of the top-K items you recommended, what fraction are actually relevant?  
**Formula:**
\[
\text{Precision@K}=\frac{|G \cap R_K|}{K}
\]
**Sarah (K=5):**
\[
\text{Precision@5}=\frac{2}{5}=0.4
\]

---

### 3) Hit Rate@K (HR@K) / Success@K
**Definition:** Binary metric—did you get *at least one* relevant item in the top-K?  
**Formula:**
\[
\text{HR@K}=\mathbb{1}\{|G \cap R_K|>0\}
\]
**Sarah (K=5):**
\[
\text{HR@5}=1
\]

---

## B) Ranking (Ordering) — Position-Sensitive Metrics

### 4) Reciprocal Rank@K (RR@K)
**Definition:** Rewards the model for placing the *first* relevant item as high as possible.  
**Formula:**
\[
\text{RR@K}=
\begin{cases}
\frac{1}{\text{rank of first relevant item}} & \text{if any hit in top-K}\\
0 & \text{otherwise}
\end{cases}
\]
**Sarah (K=5):** first relevant item is at rank 2  
\[
\text{RR@5}=\frac{1}{2}=0.5
\]

---

### 5) Mean Reciprocal Rank@K (MRR@K)
**Definition:** Average RR@K across users. (Macro-average; each user counts equally.)  
**Formula:**
\[
\text{MRR@K}=\frac{1}{|U|}\sum_{u \in U}\text{RR@K}_u
\]
**Sarah-only batch (|U|=1):**
\[
\text{MRR@5}=0.5
\]

---

### 6) Average Precision@K (AP@K)
**Definition:** For users with multiple relevant items, AP rewards models that keep relevant items near the top by averaging precision at the ranks where hits
