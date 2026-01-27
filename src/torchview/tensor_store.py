"""Disk-based tensor storage using memory-mapped files.

This module provides efficient storage for tensor data during model visualization,
avoiding memory issues with large models by storing tensors to disk.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TensorStore:
    """Manages disk-based storage for tensor data using memory-mapped files.
    
    Tensors are stored as numpy memmap files in a directory (temporary by default).
    This allows visualization of large models without running out of memory,
    as tensor data is only loaded when needed.
    
    Example:
        >>> store = TensorStore()  # Uses temp directory
        >>> store = TensorStore(cache_dir="/my/cache")  # Custom location
        >>> path = store.save(np.array([1, 2, 3]), "node_123")
        >>> data = store.load(path)
        >>> store.cleanup()
    """
    
    def __init__(
        self, 
        cache_dir: str | Path | None = None,
        prefix: str = "torchview_"
    ) -> None:
        """Initialize the tensor store.
        
        Args:
            cache_dir: Optional directory to store tensor files. If None,
                      a temporary directory is created (auto-cleaned on exit).
                      If provided, a subdirectory with the prefix is created.
            prefix: Prefix for the directory name (used with temp dir or 
                   as subdirectory name if cache_dir is provided).
        """
        self._is_temp = cache_dir is None
        
        if cache_dir is None:
            # Use system temp directory
            self._temp_dir = tempfile.mkdtemp(prefix=prefix)
        else:
            # Use provided directory, create a subdirectory with unique name
            base_dir = Path(cache_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            # Create unique subdirectory
            self._temp_dir = tempfile.mkdtemp(prefix=prefix, dir=str(base_dir))
        
        self._file_counter = 0
        self._paths: list[str] = []
        
        # Register cleanup on interpreter exit
        atexit.register(self.cleanup)
    
    @property
    def cache_dir(self) -> str:
        """Return the path to the cache directory."""
        return self._temp_dir
    
    def save(self, data: NDArray[np.floating], node_id: str) -> str:
        """Save tensor data to a memory-mapped file.
        
        Args:
            data: Numpy array to save.
            node_id: Identifier for the node (used in filename).
            
        Returns:
            Path to the saved memmap file.
        """
        # Sanitize node_id for use in filename
        safe_id = node_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        filename = f"tensor_{self._file_counter}_{safe_id}.npy"
        self._file_counter += 1
        
        filepath = Path(self._temp_dir) / filename
        
        # Create memmap file and write data
        # We save as a regular .npy file which can be memory-mapped on load
        np.save(str(filepath), data)
        
        self._paths.append(str(filepath))
        return str(filepath)
    
    def load(self, path: str) -> NDArray[np.floating]:
        """Load tensor data from a memory-mapped file.
        
        Args:
            path: Path to the memmap file.
            
        Returns:
            Numpy array (memory-mapped for large files).
        """
        # Use mmap_mode='r' for read-only memory mapping
        # This doesn't load the entire file into RAM
        return np.load(path, mmap_mode='r')
    
    def load_copy(self, path: str) -> NDArray[np.floating]:
        """Load tensor data as a full copy in memory.
        
        Use this when you need to modify the data or when
        memory-mapped access would be slower (small tensors).
        
        Args:
            path: Path to the saved file.
            
        Returns:
            Numpy array (fully loaded into memory).
        """
        return np.load(path, mmap_mode=None)
    
    def cleanup(self) -> None:
        """Delete all cached tensor files and the temporary directory.
        
        This is automatically called when the Python interpreter exits,
        but can also be called manually to free disk space earlier.
        """
        try:
            if Path(self._temp_dir).exists():
                shutil.rmtree(self._temp_dir)
                self._paths.clear()
        except Exception:
            # Ignore errors during cleanup (directory may already be gone)
            pass
    
    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.cleanup()
    
    def __len__(self) -> int:
        """Return the number of stored tensors."""
        return len(self._paths)
    
    def __repr__(self) -> str:
        return f"TensorStore(dir={self._temp_dir!r}, tensors={len(self._paths)})"
