import random

def iterSample(iterable, samplesize):
    results = []

    for i, v in enumerate(iterable):
        r = random.randint(0, i)
        if r < samplesize:
            if i < samplesize:
                results.insert(r, v) # add first samplesize items in random order
            else:
                results[r] = v # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")

    return results


def get_scatter_features(entity):
    features = [{"name":c.metadata["real_name"], "id":c.name} for c in entity.get_column_info()]
    return sorted(features, key=lambda x: x["name"])