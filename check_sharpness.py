import pickle
from pathlib import Path

# Make pickle able to find CacheState/CachedFace
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sort_photos import CacheState, CachedFace
sys.modules['__main__'].CacheState = CacheState
sys.modules['__main__'].CachedFace = CachedFace

with open(Path.home() / '.face_sort_cache' / 'cache.pkl', 'rb') as f:
    cache = pickle.load(f)

sharps = sorted([c.sharpness for c in cache.faces])
n = len(sharps)
print(f'Total faces: {n}')
print(f'Min:    {sharps[0]:.1f}')
print(f'10th %: {sharps[n//10]:.1f}')
print(f'25th %: {sharps[n//4]:.1f}')
print(f'Median: {sharps[n//2]:.1f}')
print(f'75th %: {sharps[3*n//4]:.1f}')
print(f'90th %: {sharps[9*n//10]:.1f}')
print(f'Max:    {sharps[-1]:.1f}')
print()
for t in (30, 40, 60, 80, 100):
    below = sum(1 for s in sharps if s < t)
    print(f'Below {t:3d}: {below:5d} faces ({100*below/n:.1f}%)')
