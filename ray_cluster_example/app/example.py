import ray

# Initialize Ray, automatically connects to the Ray cluster
ray.init(address='auto')

@ray.remote
def square(x):
    return x * x

if __name__ == "__main__":
    # Distribute computation across the Ray cluster
    futures = [square.remote(i) for i in range(100)]
    results = ray.get(futures)
    
    print(results)