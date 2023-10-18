import sys
import os

# Add folder with python assets to sys.path, so we can import stuff:
path = os.path.dirname(os.getcwd())
print(f'Setting base bath to "{path}"')
sys.path.insert(0, path)