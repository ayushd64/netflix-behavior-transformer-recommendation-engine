# Real-Time Behavior Sequence Transformer (BST) Two-Stage Recommendation Engine

A production-grade, stateful, two-stage recommendation framework engineered over the MovieLens-100k dataset. The architecture leverages a deep **Behavior Sequence Transformer (BST)** utilizing Multi-Head Self-Attention for real-time temporal retrieval, paired with an **Optimized Neural Collaborative Filtering (NCF)** network for precise candidate ranking. 

The entire ecosystem runs behind a multi-threaded **FastAPI** web routing microservice capable of processing live session tracking updates with sub-4ms performance footprints.

---

## 🚀 System Architecture Performance Benchmarks
* **Average Pipeline Latency:** `3.25 ms`
* **P95 Latency (95th Percentile Floor):** `3.44 ms`
* **Throughput Capacity:** Optimized for ultra-low latency candidate sifting across multi-million item vector spaces.

---

## 🗺️ End-to-End System Design Flow

```text
  [ LIVE INCOMING USER TRAFFIC ] -> /watch/?user_id=196&title=Star Wars (1977)
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: CANDIDATE RETRIEVAL (Transformer Encoder Layer)                    │
│ • Captures rolling click streams dynamically in memory.                     │
│ • Injects temporal positional context matrices directly into item lookups.  │
│ • Multi-Head Self-Attention models short-term behavior transitions.         │
│ • Outputs contextual session vector to query Faiss Index.                   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ (Retrieves 20 Raw Catalog Candidates)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PRECISE NEURAL RANKING (Deep Neural Collaborative Filtering)       │
│ • Maps raw candidates to vectorized PyTorch input batch matrices.           │
│ • Feeds tensors through a deep Multi-Layer Perceptron (MLP) network.        │
│ • Calculates exact predicted star ratings and returns a sorted list.         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
                [ ULTRA-LOW LATENCY JSON LIVE RESPONSE OUTPUT ]

## 📈 Evolutionary Project Milestones & Progression

### Milestone 1: Foundations of Collaborative Filtering
* Built the mathematical core of item-based and user-based recommendations using raw **Cosine Similarity** arrays executed through pure Python loops.
* Transitioned manual loops to vectorized matrix dot products via **NumPy**, achieving massive performance scaling.

### Milestone 2: Parametric Deep Learning via Optimized NCF
* Migrated from non-parametric calculations to a trained parametric model using **Neural Collaborative Filtering (NCF)** in **PyTorch**.
* Implemented dense structural embedding lookup layers for users and items, processed through a deep **Multi-Layer Perceptron (MLP)** with custom Dropout regularization.
* Optimized model weights utilizing stochastic gradient descent to accurately predict user-item rating values.

### Milestone 3: Real-Time Vector Scaling via Faiss
* Extracted the learned weights from the item embedding matrix of the neural network to construct dense 64-dimensional semantic vectors for every movie.
* Loaded these vectors into a **Faiss Index (Facebook AI Similarity Search)** to handle index indexing. This replaced global database scanning with approximate nearest neighbor clustering math, collapsing sifting latencies down to microseconds.

### Milestone 4: Multi-Item Session Blending
* Built an active, stateful web server layer using **FastAPI** to track rolling user session patterns.
* Applied an **Exponential Time-Decay Heuristic** ($0.5^{\text{position}}$) to merge item histories. This ensured that a user's absolute latest click exerted significantly more pull on the search vector than older history.

### Milestone 5: The Graduation Architecture (Behavior Sequence Transformer)
* Upgraded the linear decay math into an industry-grade **Sequential Behavior Transformer**.
* Preprocessed the training corpus into 96,228 chronological sequence windows mapping a user's past 4 clicks to their next target interaction label.
* Developed a PyTorch Transformer architecture combining **Item Lookup Embeddings** with a learned **Positional Embedding Matrix** to maintain temporal structural memory.
* Passed data sequences through a stack of **Multi-Head Self-Attention Encoder Layers** to learn complex contextual actions dynamically, replacing heuristics with pure sequential deep learning.

---

## 🛠️ Core Technology Stack
* **Deep Learning Framework:** PyTorch (Core Neural Architectures, Attention Blocks, Linear Layers)
* **Vector Vector Engine:** Faiss (Facebook AI Similarity Search for L2-Space Projections)
* **Asynchronous Web Gateway:** FastAPI & Uvicorn (Stateful Session Management & High-Throughput Routing)
* **Data Processing Pipeline:** Pandas, NumPy, Scikit-Learn
* **Visualization Engine:** Plotly Express & PCA (Dimensionality Reduction Diagnostics)


🗂️ Project Repository Map

├── data/                             # Raw MovieLens dataset files (Git ignored)
├── streaming_api.py                  # Live, two-stage Transformer-backed production API
├── train_sequence_transformer.py     # Training suite for the Multi-Head Attention Encoder
├── build_realtime_index.py           # Compiles Faiss binary catalogs from embedding models
├── benchmark_pipeline.py             # Performance measurement and latency profiling utility
├── visualize_faiss.py                # PCA dimensional reduction scatter plot diagnostic script
└── .gitignore                        # Protection filters guarding big binaries/dependencies

⚙️ Deployment & Production Execution
1. Fire Up the Production Server
Start your asynchronous API engine to load your weights and initialize the vector indexes:
uvicorn streaming_api:app --reload
2. Stream Live Watch Actions
Simulate live user click flows by firing incoming GET requests directly to the server:
[http://127.0.0.1:8000/watch/?user_id=196&title=Toy](http://127.0.0.1:8000/watch/?user_id=196&title=Toy) Story (1995)
[http://127.0.0.1:8000/watch/?user_id=196&title=Star](http://127.0.0.1:8000/watch/?user_id=196&title=Star) Wars (1977)
3. Run Performance Diagnostics
Profile the response speeds under multi-request loads to verify your system constraints:
python benchmark_pipeline.py

