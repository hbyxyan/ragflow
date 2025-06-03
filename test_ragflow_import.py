import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("sys.path:", sys.path)

try:
    import ragflow
    print("Successfully imported ragflow module.")
    print(f"ragflow version: {ragflow.__version__}") # 如果有 __version__ 属性
    print(f"ragflow location: {ragflow.__file__}") # 打印模块路径
except ImportError as e:
    print(f"Failed to import ragflow: {e}")
except AttributeError: # Catching if __version__ or __file__ is missing
    print("Successfully imported ragflow, but __version__ or __file__ attribute is missing.")
    print(f"ragflow object: {ragflow}")
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
