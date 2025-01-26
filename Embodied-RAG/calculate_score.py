import json
import numpy as np

def analyze_batch_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    top1_metrics = {
        'correctness': [],
        'affirmness': [],
        'spatial_relativity': [],
        'overall_score': []
    }
    
    top5_metrics = {
        'correctness': [],
        'affirmness': [],
        'spatial_relativity': [],
        'overall_score': []
    }
    
    # Initialize timing metrics
    timing_metrics = {
        'total_time': [],
        'retrieval_time': [],
        'response_time': []
    }
    
    # Collect metrics from successful queries
    successful_queries = 0
    for result in data['results']:
        if 'result' in result and result['result'].get('success', False):
            successful_queries += 1
            
            # Top 1 metrics
            top1 = result['result']['top_1']
            for metric in top1_metrics:
                if metric in top1:
                    top1_metrics[metric].append(top1[metric])
            
            # Top 5 metrics
            top5 = result['result']['top_5']
            for metric in top5_metrics:
                if metric in top5:
                    top5_metrics[metric].append(top5[metric])
            
            # Timing metrics
            timing = result['result']['timing']
            for metric in timing_metrics:
                if metric in timing:
                    timing_metrics[metric].append(timing[metric])
    
    # Calculate averages
    top1_averages = {
        metric: np.mean(values) for metric, values in top1_metrics.items() if values
    }
    top5_averages = {
        metric: np.mean(values) for metric, values in top5_metrics.items() if values
    }
    timing_averages = {
        metric: np.mean(values) for metric, values in timing_metrics.items() if values
    }
    
    return {
        'total_queries': data['total_queries'],
        'successful_queries': successful_queries,
        'top1_averages': top1_averages,
        'top5_averages': top5_averages,
        'timing_averages': timing_averages
    }

# Analyze both files
file1 = '/Users/danielxie/E-RAG/Embodied-RAG/experiment_logs/batch_results_20250119_143734.json'
file2 = '/Users/danielxie/E-RAG/Embodied-RAG/experiment_logs/batch_results_20250119_150730.json'

results1 = analyze_batch_results(file1)
results2 = analyze_batch_results(file2)

print("File 1 Results (Explicit Queries):")
print(f"Total Queries: {results1['total_queries']}")
print(f"Successful Queries: {results1['successful_queries']}")
print("\nTop 1 Averages:")
for metric, value in results1['top1_averages'].items():
    print(f"{metric}: {value:.4f}")
print("\nTop 5 Averages:")
for metric, value in results1['top5_averages'].items():
    print(f"{metric}: {value:.4f}")
print("\nTiming Averages (seconds):")
for metric, value in results1['timing_averages'].items():
    print(f"{metric}: {value:.2f}")

print("\n" + "="*50 + "\n")

print("File 2 Results (Implicit Queries):")
print(f"Total Queries: {results2['total_queries']}")
print(f"Successful Queries: {results2['successful_queries']}")
print("\nTop 1 Averages:")
for metric, value in results2['top1_averages'].items():
    print(f"{metric}: {value:.4f}")
print("\nTop 5 Averages:")
for metric, value in results2['top5_averages'].items():
    print(f"{metric}: {value:.4f}")
print("\nTiming Averages (seconds):")
for metric, value in results2['timing_averages'].items():
    print(f"{metric}: {value:.2f}")
    