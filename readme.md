# Generating Captions for Google Street View Images

## 1. Run the script (Already done, don't need to caption again)
```
python data_caption.py Tokyo
```

## 2. Generate Semantic Forest
```
python generate_semantic_forest.py --input-dir /Users/danielxie/E-RAG/Embodied-RAG/graph/tokyo_streetview_graph.gml --name half_tokyp
```

## 3. Evaluate Retrieval
```
python evaluation.py --graph-dir /Users/danielxie/E-RAG/Embodied-RAG/semantic_forests/CMU_500/semantic_forest_CMU_500.gml --vector-db /Users/danielxie/E-RAG/Embodied-RAG/semantic_forests/CMU_500/vector_db --image-dir /Users/danielxie/E-RAG/data_generation/datasets/CMU_500/images/merged

```

# evaluate in batch
```
python evaluation.py --mode batch --graph-dir /Users/danielxie/E-RAG/Embodied-RAG/semantic_forests/CMU_500/semantic_forest_CMU_500.gml --vector-db /Users/danielxie/E-RAG/Embodied-RAG/semantic_forests/CMU_500/vector_db --image-dir /Users/danielxie/E-RAG/data_generation/datasets/CMU_500/images/merged
```