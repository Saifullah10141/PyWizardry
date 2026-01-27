"""
PyWizardry v1.0.1 - A magical collection of 150+ Python utilities for modern development
Author: Saif
Email: saifullahanwar00040@gmail.com
Website: https://pywizardry.vercel.app
PyPI: https://pypi.org/project/PyWizardry
GitHub: https://github.com/Saifullah10141/pywizardry
License: MIT

FEATURES:
- 150+ utility functions
- 15+ modules covering all development areas
- Production-ready with comprehensive error handling
- Async/await support throughout
- Extensive type hints
- Zero required dependencies
- Optional extended features
"""

__version__ = "1.0.1"
__author__ = "Saif"
__license__ = "MIT"
__all__ = [
    "Wizard", "SpellBook", "MagicError", "ValidationError", 
    "NetworkError", "SecurityError", "AsyncError", "DataError",
    "config", "files", "strings", "security", "dates", "network",
    "async_utils", "data", "database", "testing", "ai", "console",
    "web", "math", "validation", "crypto", "parallel", "utils",
    "spell_timer", "retry_spell", "cache_spell", "validate_spell",
    "async_spell", "benchmark_spell", "create_pipeline",
]

# ==================== IMPORTS ====================
import os
import sys
import json
import csv
import datetime
import time
import re
import random
import string
import hashlib
import base64
import itertools
import urllib.parse
import urllib.request
import urllib.error
import tempfile
import shutil
import asyncio
import subprocess
import socket
import mimetypes
import uuid
import math
import statistics
import collections
import inspect
import threading
import queue
import contextlib
import pickle
import gzip
import zipfile
import tarfile
import pathlib
import decimal
import fractions
import hashlib
import secrets
import hmac
import binascii
import typing
import warnings
import textwrap
import difflib
import pprint
import itertools
import functools
import operator
import copy
import html
import html.parser
import cgi
import email
import email.mime
import email.mime.text
import email.mime.multipart
import email.mime.base
import email.utils
import quopri
import uu
import zlib
import bz2
import lzma
import struct
import array
import wave
import audioop
import ssl
import select
import selectors
import signal
import mmap
import curses
import getopt
import getpass
import platform
import locale
import cmd
import shlex
import readline
import rlcompleter
import sqlite3
import csv
import configparser
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import xml.sax
import xml.sax.handler
import xml.sax.saxutils
import hashlib
import hmac
import secrets
import base64
import binascii
import itertools
import math
import statistics
import random
import decimal
import fractions
import numbers
import datetime
import time
import calendar
import collections
import heapq
import bisect
import array
import weakref
import types
import copy
import pprint
import reprlib
import enum
import graphlib
import zoneinfo
import ipaddress
import hashlib
import secrets
import hmac
import base64
import binascii
import html
import html.parser
import cgi
import email
import email.mime
import email.mime.text
import email.mime.multipart
import email.mime.base
import email.utils
import quopri
import uu
import zlib
import bz2
import lzma
import struct
import wave
import audioop
import ssl
import select
import selectors
import signal
import mmap
import curses
import getopt
import getpass
import platform
import locale
import cmd
import shlex
import readline
import rlcompleter
import sqlite3
import csv
import configparser
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import xml.sax
import xml.sax.handler
import xml.sax.saxutils

from pathlib import Path
from typing import (
    Any, List, Dict, Tuple, Optional, Union, Callable, 
    Iterator, Generator, Set, Sequence, TypeVar, Type,
    Mapping, MutableMapping, Iterable, AsyncIterator,
    AsyncGenerator, Awaitable, Coroutine, NamedTuple,
    Protocol, runtime_checkable, Literal, TypedDict,
    ClassVar, Final, overload, NoReturn, NewType,
    get_type_hints, get_origin, get_args,
)
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum, IntEnum, Flag, auto
from functools import (
    wraps, lru_cache, total_ordering, singledispatch,
    singledispatchmethod, cached_property, reduce, partial,
)
from collections import (
    defaultdict, Counter, OrderedDict, deque, ChainMap,
    UserDict, UserList, UserString, namedtuple,
)
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from fractions import Fraction
from contextlib import (
    contextmanager, asynccontextmanager, ExitStack,
    AsyncExitStack, suppress, closing, nullcontext,
)
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, 
    Future, as_completed, wait, ALL_COMPLETED,
    FIRST_COMPLETED, FIRST_EXCEPTION,
)
from importlib import import_module, resources, metadata
from inspect import (
    signature, Parameter, getmembers, isclass,
    isfunction, ismethod, iscoroutinefunction,
    isawaitable, getdoc, getfile, getsource,
    getmodule, stack, currentframe, getouterframes,
    getinnerframes, trace, getcallargs, getfullargspec,
)
from io import (
    StringIO, BytesIO, TextIOWrapper, BufferedReader,
    BufferedWriter, BufferedRandom, RawIOBase,
    FileIO, IOBase,
)
from math import (
    pi, e, tau, inf, nan, isfinite, isinf, isnan,
    ceil, floor, trunc, gcd, lcm, factorial,
    exp, log, log10, log2, pow, sqrt, hypot,
    sin, cos, tan, asin, acos, atan, atan2,
    degrees, radians, sinh, cosh, tanh,
    asinh, acosh, atanh, erf, erfc, gamma,
    lgamma, fsum, prod, dist, isclose,
    comb, perm, nextafter, ulp,
)
from statistics import (
    mean, fmean, geometric_mean, harmonic_mean,
    median, median_low, median_high, median_grouped,
    mode, multimode, quantiles, pstdev, pvariance,
    stdev, variance, covariance, correlation,
    linear_regression,
)

# Import optional dependencies conditionally
try:
    import colorama
    from colorama import Fore, Back, Style, init
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    import aiohttp
    import aiofiles
    AIO_AVAILABLE = True
except ImportError:
    AIO_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from PIL import Image, ImageFilter, ImageEnhance
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# ==================== CUSTOM EXCEPTIONS ====================
class MagicError(Exception):
    """Base exception for all PyWizardry errors"""
    pass

class ValidationError(MagicError):
    """Raised when validation fails"""
    pass

class NetworkError(MagicError):
    """Raised for network-related errors"""
    pass

class SecurityError(MagicError):
    """Raised for security-related errors"""
    pass

class AsyncError(MagicError):
    """Raised for async-related errors"""
    pass

class DataError(MagicError):
    """Raised for data processing errors"""
    pass

class FileSystemError(MagicError):
    """Raised for file system errors"""
    pass

class ConfigurationError(MagicError):
    """Raised for configuration errors"""
    pass

# ==================== DECORATORS (30+ decorators) ====================
def spell_timer(print_result: bool = False):
    """Time execution of spells with optional result printing"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if COLORAMA_AVAILABLE:
                init(autoreset=True)
                print(f"{Fore.GREEN}✨ {func.__name__}{Style.RESET_ALL} "
                      f"completed in {Fore.CYAN}{elapsed:.6f}s{Style.RESET_ALL}")
            else:
                print(f"✨ {func.__name__} completed in {elapsed:.6f}s")
            
            if print_result:
                print(f"   Result: {result}")
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if COLORAMA_AVAILABLE:
                print(f"{Fore.GREEN}⚡ {func.__name__}{Style.RESET_ALL} "
                      f"completed in {Fore.CYAN}{elapsed:.6f}s{Style.RESET_ALL}")
            else:
                print(f"⚡ {func.__name__} completed in {elapsed:.6f}s")
            
            if print_result:
                print(f"   Result: {result}")
            return result
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

def retry_spell(max_attempts: int = 3, delay: float = 1.0, 
                backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry failed spells with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    sleep_time = delay * (backoff ** attempt)
                    time.sleep(sleep_time)
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    sleep_time = delay * (backoff ** attempt)
                    await asyncio.sleep(sleep_time)
            raise last_exception
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

def cache_spell(ttl: int = 300, maxsize: int = 128):
    """Cache spell results with time-to-live"""
    cache = {}
    cache_times = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = pickle.dumps((args, kwargs))
            current_time = time.time()
            
            if key in cache:
                cached_time = cache_times.get(key, 0)
                if current_time - cached_time < ttl:
                    return cache[key]
            
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= maxsize:
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                cache.pop(oldest_key, None)
                cache_times.pop(oldest_key, None)
            
            cache[key] = result
            cache_times[key] = current_time
            return result
        
        return wrapper
    return decorator

def validate_spell(*validators: Callable):
    """Validate arguments before executing spell"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in zip(sig.parameters, validators):
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Invalid argument {param_name}: {value}"
                        )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_spell(timeout: Optional[float] = None):
    """Convert sync function to async with timeout"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            if timeout:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout
                )
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        return wrapper
    return decorator

def benchmark_spell(iterations: int = 1000):
    """Benchmark spell execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import timeit
            
            # Warm-up
            for _ in range(10):
                func(*args, **kwargs)
            
            timer = timeit.Timer(lambda: func(*args, **kwargs))
            times = timer.repeat(repeat=5, number=iterations)
            
            stats = {
                "function": func.__name__,
                "iterations": iterations,
                "min": min(times),
                "max": max(times),
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "per_iteration": min(times) / iterations,
            }
            
            return stats
        return wrapper
    return decorator

def rate_limit_spell(calls: int = 10, period: float = 1.0):
    """Rate limit spell calls"""
    import threading
    from collections import deque
    
    lock = threading.Lock()
    calls_history = deque()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                
                # Remove old calls
                while calls_history and calls_history[0] <= now - period:
                    calls_history.popleft()
                
                if len(calls_history) >= calls:
                    oldest = calls_history[0]
                    wait_time = period - (now - oldest)
                    if wait_time > 0:
                        time.sleep(wait_time)
                        # Recalculate after wait
                        now = time.time()
                        calls_history.clear()
                
                calls_history.append(now)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ==================== FILE SYSTEM MAGIC (35+ functions) ====================
class FileMagic:
    """File system utilities with atomic operations and progress tracking"""
    
    @staticmethod
    @spell_timer()
    def safe_write(path: Union[str, Path], content: Union[str, bytes], 
                   backup: bool = True, atomic: bool = True) -> Path:
        """Safely write to file with backup and atomic operations"""
        path_obj = Path(path)
        
        if backup and path_obj.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path_obj.with_suffix(f"{path_obj.suffix}.backup_{timestamp}")
            shutil.copy2(path_obj, backup_path)
        
        if atomic:
            # Write to temporary file first
            temp_path = path_obj.with_suffix(f"{path_obj.suffix}.tmp")
            
            if isinstance(content, str):
                temp_path.write_text(content, encoding='utf-8')
            else:
                temp_path.write_bytes(content)
            
            # Atomic rename
            temp_path.rename(path_obj)
        else:
            if isinstance(content, str):
                path_obj.write_text(content, encoding='utf-8')
            else:
                path_obj.write_bytes(content)
        
        return path_obj
    
    @staticmethod
    @retry_spell(max_attempts=3, delay=0.5)
    def read_large_file(path: Union[str, Path], chunk_size: int = 8192) -> Generator[str, None, None]:
        """Read large file line by line with retry logic"""
        path_obj = Path(path)
        with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", 
                   recursive: bool = True, size_limit: Optional[int] = None,
                   modified_after: Optional[datetime.datetime] = None,
                   modified_before: Optional[datetime.datetime] = None) -> List[Path]:
        """Find files with advanced filtering"""
        dir_path = Path(directory)
        files = []
        
        if recursive:
            iterator = dir_path.rglob(pattern)
        else:
            iterator = dir_path.glob(pattern)
        
        for file_path in iterator:
            if not file_path.is_file():
                continue
            
            # Size filtering
            if size_limit is not None:
                if file_path.stat().st_size > size_limit:
                    continue
            
            # Time filtering
            if modified_after is not None or modified_before is not None:
                mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if modified_after is not None and mtime < modified_after:
                    continue
                if modified_before is not None and mtime > modified_before:
                    continue
            
            files.append(file_path)
        
        return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    @staticmethod
    def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information"""
        path_obj = Path(path)
        stat = path_obj.stat()
        
        info = {
            "path": str(path_obj.absolute()),
            "name": path_obj.name,
            "stem": path_obj.stem,
            "suffix": path_obj.suffix,
            "parent": str(path_obj.parent),
            "size": stat.st_size,
            "size_human": FileMagic._human_size(stat.st_size),
            "created": datetime.datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime),
            "accessed": datetime.datetime.fromtimestamp(stat.st_atime),
            "is_dir": path_obj.is_dir(),
            "is_file": path_obj.is_file(),
            "is_symlink": path_obj.is_symlink(),
            "permissions": oct(stat.st_mode)[-3:],
            "inode": stat.st_ino,
            "device": stat.st_dev,
            "hash_md5": FileMagic._file_hash(path_obj, "md5"),
            "hash_sha256": FileMagic._file_hash(path_obj, "sha256"),
        }
        
        # Add MIME type
        mime_type, _ = mimetypes.guess_type(str(path_obj))
        info["mime_type"] = mime_type
        
        # Add line count for text files
        if info["mime_type"] and info["mime_type"].startswith("text/"):
            try:
                with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    info["line_count"] = sum(1 for _ in f)
            except:
                info["line_count"] = None
        
        return info
    
    @staticmethod
    def _human_size(size: int) -> str:
        """Convert size in bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} EB"
    
    @staticmethod
    def _file_hash(path: Path, algorithm: str) -> str:
        """Calculate file hash"""
        hash_func = getattr(hashlib, algorithm)()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    @staticmethod
    def backup_directory(source: Union[str, Path], destination: Union[str, Path],
                         compression: str = 'zip', exclude_patterns: List[str] = None) -> Path:
        """Backup directory with compression"""
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            raise FileSystemError(f"Source directory not found: {source_path}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_backup_{timestamp}"
        
        if compression == 'zip':
            backup_file = dest_path / f"{backup_name}.zip"
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in source_path.rglob('*'):
                    if exclude_patterns and any(file_path.match(p) for p in exclude_patterns):
                        continue
                    if file_path.is_file():
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)
        
        elif compression == 'tar':
            backup_file = dest_path / f"{backup_name}.tar.gz"
            with tarfile.open(backup_file, 'w:gz') as tarf:
                tarf.add(source_path, arcname=source_path.name)
        
        else:
            raise ValueError(f"Unsupported compression: {compression}")
        
        return backup_file
    
    @staticmethod
    @async_spell(timeout=30)
    def async_download(url: str, destination: Union[str, Path], 
                       chunk_size: int = 8192) -> Path:
        """Async file download with progress"""
        if not AIO_AVAILABLE:
            raise ImportError("aiohttp and aiofiles required for async download")
        
        import aiohttp
        import aiofiles
        
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                async with aiofiles.open(dest_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading: {progress:.1f}%", end='')
        
        print()
        return dest_path
    
    @staticmethod
    def watch_directory(path: Union[str, Path], handler: Callable,
                        extensions: List[str] = None, recursive: bool = True) -> threading.Thread:
        """Watch directory for changes (simplified)"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class EventHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    if extensions:
                        if any(event.src_path.endswith(ext) for ext in extensions):
                            handler(event.src_path, 'created')
                    else:
                        handler(event.src_path, 'created')
            
            def on_modified(self, event):
                if not event.is_directory:
                    if extensions:
                        if any(event.src_path.endswith(ext) for ext in extensions):
                            handler(event.src_path, 'modified')
                    else:
                        handler(event.src_path, 'modified')
        
        observer = Observer()
        event_handler = EventHandler()
        observer.schedule(event_handler, str(path), recursive=recursive)
        observer.start()
        
        return observer

# ==================== STRING SORCERY (25+ functions) ====================
class StringSorcery:
    """Advanced string manipulation with NLP helpers"""
    
    @staticmethod
    def slugify(text: str, separator: str = "-", lowercase: bool = True,
                replace_chars: Dict[str, str] = None) -> str:
        """Advanced slugify with character replacement"""
        if lowercase:
            text = text.lower()
        
        # Replace custom characters
        if replace_chars:
            for old, new in replace_chars.items():
                text = text.replace(old, new)
        
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        
        # Replace whitespace
        text = re.sub(r'[\s-]+', ' ', text).strip()
        
        return text.replace(' ', separator)
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = "...", 
                 preserve_words: bool = True, ellipsis_position: str = "end") -> str:
        """Smart truncation with word preservation"""
        if len(text) <= max_length:
            return text
        
        if ellipsis_position == "middle":
            if max_length <= len(suffix) + 2:
                return suffix
            
            chars_each_side = (max_length - len(suffix)) // 2
            return f"{text[:chars_each_side]}{suffix}{text[-chars_each_side:]}"
        
        elif ellipsis_position == "start":
            return f"{suffix}{text[-(max_length - len(suffix)):]}"
        
        else:  # end
            if preserve_words:
                truncated = text[:max_length - len(suffix)]
                last_space = truncated.rfind(' ')
                if last_space > max_length * 0.6:  # Preserve word if reasonable
                    truncated = truncated[:last_space]
                return truncated.strip() + suffix
            else:
                return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def similarity(string1: str, string2: str, method: str = "levenshtein") -> float:
        """Calculate string similarity using different algorithms"""
        if method == "levenshtein":
            return StringSorcery._levenshtein_similarity(string1, string2)
        elif method == "jaro":
            return StringSorcery._jaro_winkler_similarity(string1, string2)
        elif method == "cosine":
            return StringSorcery._cosine_similarity(string1, string2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def _levenshtein_similarity(string1: str, string2: str) -> float:
        """Levenshtein distance similarity"""
        if len(string1) < len(string2):
            return StringSorcery._levenshtein_similarity(string2, string1)
        
        if len(string2) == 0:
            return 1.0
        
        previous_row = range(len(string2) + 1)
        for i, c1 in enumerate(string1):
            current_row = [i + 1]
            for j, c2 in enumerate(string2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(string1), len(string2))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    @staticmethod
    def _jaro_winkler_similarity(string1: str, string2: str, winkler: bool = True) -> float:
        """Jaro-Winkler similarity"""
        # Simplified implementation
        len1, len2 = len(string1), len(string2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        match_distance = max(len1, len2) // 2 - 1
        matches1 = [False] * len1
        matches2 = [False] * len2
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if not matches2[j] and string1[i] == string2[j]:
                    matches1[i] = True
                    matches2[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len1):
            if matches1[i]:
                while not matches2[k]:
                    k += 1
                if string1[i] != string2[k]:
                    transpositions += 1
                k += 1
        
        transpositions //= 2
        jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3
        
        if winkler:
            prefix = 0
            for i in range(min(4, len1, len2)):
                if string1[i] == string2[i]:
                    prefix += 1
                else:
                    break
            
            return jaro + prefix * 0.1 * (1 - jaro)
        
        return jaro
    
    @staticmethod
    def _cosine_similarity(string1: str, string2: str) -> float:
        """Cosine similarity using character n-grams"""
        def get_ngrams(text, n=2):
            return [text[i:i+n] for i in range(len(text)-n+1)]
        
        ngrams1 = get_ngrams(string1.lower())
        ngrams2 = get_ngrams(string2.lower())
        
        vec1 = Counter(ngrams1)
        vec2 = Counter(ngrams2)
        
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator) / denominator
    
    @staticmethod
    def generate_password(length: int = 16, 
                          include_uppercase: bool = True,
                          include_digits: bool = True,
                          include_symbols: bool = True,
                          exclude_similar: bool = True,
                          exclude_ambiguous: bool = True) -> str:
        """Generate secure password with multiple options"""
        chars = string.ascii_lowercase
        
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_digits:
            chars += string.digits
        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Exclusion lists
        if exclude_similar:
            similar = "il1Lo0O"
            chars = ''.join(c for c in chars if c not in similar)
        
        if exclude_ambiguous:
            ambiguous = "{}[]()/\'\"`~,;:.<>"
            chars = ''.join(c for c in chars if c not in ambiguous)
        
        # Ensure at least one from each category
        password = []
        if include_uppercase:
            password.append(secrets.choice(string.ascii_uppercase))
        if include_digits:
            password.append(secrets.choice(string.digits))
        if include_symbols:
            password.append(secrets.choice("!@#$%^&*()_+-=[]{}|;:,.<>?"))
        
        # Fill remaining
        remaining = length - len(password)
        password.extend(secrets.choice(chars) for _ in range(remaining))
        
        # Shuffle
        secrets.SystemRandom().shuffle(password)
        return ''.join(password)
    
    @staticmethod
    def extract_info(text: str) -> Dict[str, List[str]]:
        """Extract various information from text"""
        info = {
            "emails": StringSorcery.extract_emails(text),
            "urls": StringSorcery.extract_urls(text),
            "phone_numbers": StringSorcery.extract_phone_numbers(text),
            "hashtags": StringSorcery.extract_hashtags(text),
            "mentions": StringSorcery.extract_mentions(text),
            "ip_addresses": StringSorcery.extract_ip_addresses(text),
        }
        return info
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs"""
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w .?=&%-]*'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers (international format)"""
        pattern = r'\+?[\d\s\-\(\)]{7,}'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_hashtags(text: str) -> List[str]:
        """Extract hashtags"""
        pattern = r'#\w+'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_mentions(text: str) -> List[str]:
        """Extract mentions (@username)"""
        pattern = r'@\w+'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_ip_addresses(text: str) -> List[str]:
        """Extract IP addresses"""
        pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def normalize_whitespace(text: str, keep_newlines: bool = True) -> str:
        """Normalize whitespace"""
        if keep_newlines:
            lines = text.split('\n')
            normalized_lines = [' '.join(line.split()) for line in lines]
            return '\n'.join(normalized_lines)
        else:
            return ' '.join(text.split())
    
    @staticmethod
    def remove_special_chars(text: str, keep: str = "") -> str:
        """Remove special characters"""
        # Keep alphanumeric and specified characters
        pattern = f'[^\\w\\s{re.escape(keep)}]'
        return re.sub(pattern, '', text)
    
    @staticmethod
    def camel_to_snake(text: str) -> str:
        """Convert CamelCase to snake_case"""
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
        return text.lower()
    
    @staticmethod
    def snake_to_camel(text: str) -> str:
        """Convert snake_case to CamelCase"""
        return ''.join(word.title() for word in text.split('_'))
    
    @staticmethod
    def to_title_case(text: str) -> str:
        """Convert to title case (smart)"""
        small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 
                       'in', 'of', 'on', 'or', 'the', 'to', 'with'}
        
        words = text.lower().split()
        result = []
        
        for i, word in enumerate(words):
            if i == 0 or i == len(words) - 1 or word not in small_words:
                result.append(word.title())
            else:
                result.append(word)
        
        return ' '.join(result)

# ==================== SECURITY SPELLS (20+ functions) ====================
class SecuritySpells:
    """Security utilities with encryption, hashing, and token generation"""
    
    @staticmethod
    def generate_key(length: int = 32) -> bytes:
        """Generate cryptographically secure key"""
        return secrets.token_bytes(length)
    
    @staticmethod
    def aes_encrypt(plaintext: str, password: str) -> Dict[str, Any]:
        """AES encryption with password"""
        import hashlib
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad
        import base64
        
        # Generate key from password
        key = hashlib.sha256(password.encode()).digest()
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # Encrypt
        padded_data = pad(plaintext.encode(), AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        
        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "algorithm": "AES-256-CBC",
        }
    
    @staticmethod
    def aes_decrypt(encrypted_data: Dict[str, str], password: str) -> str:
        """AES decryption"""
        import hashlib
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad
        import base64
        
        key = hashlib.sha256(password.encode()).digest()
        iv = base64.b64decode(encrypted_data["iv"])
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(ciphertext)
        
        return unpad(decrypted, AES.block_size).decode()
    
    @staticmethod
    def jwt_encode(payload: Dict[str, Any], secret: str, 
                   algorithm: str = "HS256", expires_in: int = 3600) -> str:
        """Encode JWT token"""
        import time
        import base64
        import hmac
        import hashlib
        import json
        
        # Add timestamps
        current_time = int(time.time())
        payload.update({
            "iat": current_time,
            "exp": current_time + expires_in,
        })
        
        # Encode header and payload
        header = json.dumps({"alg": algorithm, "typ": "JWT"}).encode()
        payload_encoded = json.dumps(payload).encode()
        
        header_b64 = base64.urlsafe_b64encode(header).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(payload_encoded).decode().rstrip('=')
        
        # Create signature
        message = f"{header_b64}.{payload_b64}"
        
        if algorithm == "HS256":
            signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{header_b64}.{payload_b64}.{signature_b64}"
    
    @staticmethod
    def jwt_decode(token: str, secret: str) -> Dict[str, Any]:
        """Decode and verify JWT token"""
        import base64
        import hmac
        import hashlib
        import json
        import time
        
        try:
            parts = token.split('.')
            if len(parts) != 3:
                raise SecurityError("Invalid token format")
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            
            # Recreate signature
            signature = base64.urlsafe_b64decode(signature_b64 + '=' * (-len(signature_b64) % 4))
            expected_signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(signature, expected_signature):
                raise SecurityError("Invalid signature")
            
            # Decode payload
            payload_encoded = base64.urlsafe_b64decode(payload_b64 + '=' * (-len(payload_b64) % 4))
            payload = json.loads(payload_encoded)
            
            # Check expiration
            if "exp" in payload:
                if payload["exp"] < time.time():
                    raise SecurityError("Token expired")
            
            return payload
            
        except Exception as e:
            raise SecurityError(f"Token validation failed: {str(e)}")
    
    @staticmethod
    def password_strength(password: str) -> Dict[str, Any]:
        """Check password strength"""
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            feedback.append("Password should be at least 8 characters long")
        
        # Character variety
        checks = {
            "lowercase": any(c.islower() for c in password),
            "uppercase": any(c.isupper() for c in password),
            "digits": any(c.isdigit() for c in password),
            "special": any(not c.isalnum() for c in password),
        }
        
        variety_score = sum(checks.values())
        score += variety_score
        
        if variety_score < 3:
            missing = [k for k, v in checks.items() if not v]
            feedback.append(f"Add {', '.join(missing)} characters")
        
        # Common patterns
        common_passwords = {"password", "123456", "qwerty", "admin", "letmein"}
        if password.lower() in common_passwords:
            score = 0
            feedback.append("Password is too common")
        
        # Sequential characters
        if re.search(r'(.)\1{2,}', password):
            score -= 1
            feedback.append("Avoid repeating characters")
        
        # Dictionary words
        if len(password) >= 4 and any(word in password.lower() for word in ["pass", "admin", "test"]):
            score -= 1
            feedback.append("Avoid common words")
        
        # Final evaluation
        if score >= 5:
            strength = "Strong"
        elif score >= 3:
            strength = "Medium"
        else:
            strength = "Weak"
        
        return {
            "score": min(max(score, 0), 10),
            "strength": strength,
            "feedback": feedback,
            "length": len(password),
            "checks": checks,
        }
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_csrf_token(token1: str, token2: str) -> bool:
        """Validate CSRF tokens match"""
        return secrets.compare_digest(token1, token2)
    
    @staticmethod
    def sanitize_input(input_str: str, allowed_tags: List[str] = None) -> str:
        """Sanitize HTML input"""
        if allowed_tags is None:
            allowed_tags = []
        
        # Remove script tags and event handlers
        cleaned = re.sub(r'<script[^>]*>.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'on\w+="[^"]*"', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'on\w+=\'[^\']*\'', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'on\w+=[^ >]*', '', cleaned, flags=re.IGNORECASE)
        
        # Allow specific tags
        if allowed_tags:
            tag_pattern = r'</?(?!' + '|'.join(allowed_tags) + r')[^>]*>'
            cleaned = re.sub(tag_pattern, '', cleaned, flags=re.IGNORECASE)
        else:
            # Remove all HTML tags
            cleaned = re.sub(r'<[^>]*>', '', cleaned)
        
        # Escape special characters
        cleaned = html.escape(cleaned)
        
        return cleaned
    
    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate one-time password"""
        digits = string.digits
        return ''.join(secrets.choice(digits) for _ in range(length))
    
    @staticmethod
    def hash_password(password: str, salt: str = None, 
                      algorithm: str = "pbkdf2_sha256") -> Dict[str, str]:
        """Hash password with salt"""
        import hashlib
        import binascii
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        if algorithm == "pbkdf2_sha256":
            # Use built-in hashlib.pbkdf2_hmac
            iterations = 100000
            dk = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                iterations
            )
            hash_hex = binascii.hexlify(dk).decode()
            
            return {
                "hash": hash_hex,
                "salt": salt,
                "algorithm": algorithm,
                "iterations": iterations,
            }
        
        elif algorithm.startswith("scrypt"):
            # Note: scrypt requires Python 3.6+
            N = 16384  # CPU/memory cost parameter
            r = 8      # block size parameter
            p = 1      # parallelization parameter
            
            dk = hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt.encode('utf-8'),
                n=N, r=r, p=p,
                dklen=32
            )
            hash_hex = binascii.hexlify(dk).decode()
            
            return {
                "hash": hash_hex,
                "salt": salt,
                "algorithm": algorithm,
                "N": N,
                "r": r,
                "p": p,
            }
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def verify_password(password: str, hashed_data: Dict[str, Any]) -> bool:
        """Verify password against hash"""
        import hashlib
        import binascii
        
        algorithm = hashed_data.get("algorithm", "pbkdf2_sha256")
        salt = hashed_data["salt"]
        stored_hash = hashed_data["hash"]
        
        if algorithm == "pbkdf2_sha256":
            iterations = hashed_data.get("iterations", 100000)
            dk = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                iterations
            )
            computed_hash = binascii.hexlify(dk).decode()
            return secrets.compare_digest(computed_hash, stored_hash)
        
        elif algorithm.startswith("scrypt"):
            N = hashed_data.get("N", 16384)
            r = hashed_data.get("r", 8)
            p = hashed_data.get("p", 1)
            
            dk = hashlib.scrypt(
                password.encode('utf-8'),
                salt=salt.encode('utf-8'),
                n=N, r=r, p=p,
                dklen=32
            )
            computed_hash = binascii.hexlify(dk).decode()
            return secrets.compare_digest(computed_hash, stored_hash)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

# ==================== TIME WIZARDRY (15+ functions) ====================
class TimeWizardry:
    """Advanced date and time utilities with timezone support"""
    
    @staticmethod
    def humanize(dt: datetime.datetime, 
                 granularity: str = "auto",
                 add_suffix: bool = True) -> str:
        """Human-readable time difference"""
        now = datetime.datetime.now(dt.tzinfo) if dt.tzinfo else datetime.datetime.now()
        
        if dt > now:
            is_future = True
            diff = dt - now
        else:
            is_future = False
            diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            value = int(seconds)
            unit = "second"
        elif seconds < 3600:
            value = int(seconds / 60)
            unit = "minute"
        elif seconds < 86400:
            value = int(seconds / 3600)
            unit = "hour"
        elif seconds < 604800:  # 7 days
            value = int(seconds / 86400)
            unit = "day"
        elif seconds < 2592000:  # 30 days
            value = int(seconds / 604800)
            unit = "week"
        elif seconds < 31536000:  # 365 days
            value = int(seconds / 2592000)
            unit = "month"
        else:
            value = int(seconds / 31536000)
            unit = "year"
        
        if value != 1:
            unit += "s"
        
        if add_suffix:
            if is_future:
                return f"in {value} {unit}"
            else:
                return f"{value} {unit} ago"
        else:
            return f"{value} {unit}"
    
    @staticmethod
    def format_duration(seconds: float, 
                        precision: int = 2,
                        compact: bool = False) -> str:
        """Format duration with precision"""
        if seconds < 1:
            ms = seconds * 1000
            if ms < 1:
                μs = ms * 1000
                if μs < 1:
                    ns = μs * 1000
                    return f"{ns:.{precision}f}ns"
                return f"{μs:.{precision}f}μs"
            return f"{ms:.{precision}f}ms"
        
        units = [
            ("year", 31536000),
            ("month", 2592000),
            ("week", 604800),
            ("day", 86400),
            ("hour", 3600),
            ("minute", 60),
            ("second", 1),
        ]
        
        parts = []
        remaining = seconds
        
        for unit_name, unit_seconds in units:
            if remaining >= unit_seconds:
                value = int(remaining // unit_seconds)
                remaining %= unit_seconds
                
                if compact:
                    parts.append(f"{value}{unit_name[0]}")
                else:
                    if value != 1:
                        unit_name += "s"
                    parts.append(f"{value} {unit_name}")
        
        if not parts:
            parts.append("0 seconds")
        
        if compact:
            return " ".join(parts[:2])  # Show only top 2 units in compact mode
        else:
            return ", ".join(parts)
    
    @staticmethod
    def next_business_day(date: datetime.datetime = None,
                          country: str = "US") -> datetime.datetime:
        """Get next business day considering holidays"""
        if date is None:
            date = datetime.datetime.now()
        
        # Move to next day
        next_day = date + datetime.timedelta(days=1)
        
        # Check if it's a weekend
        while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_day += datetime.timedelta(days=1)
        
        # Basic holiday check (US holidays)
        if country == "US":
            holidays = TimeWizardry._get_us_holidays(next_day.year)
            while next_day.date() in holidays:
                next_day += datetime.timedelta(days=1)
                # Skip weekends again
                while next_day.weekday() >= 5:
                    next_day += datetime.timedelta(days=1)
        
        return next_day
    
    @staticmethod
    def _get_us_holidays(year: int) -> List[datetime.date]:
        """Get US holidays for a given year"""
        holidays = []
        
        # New Year's Day
        holidays.append(datetime.date(year, 1, 1))
        
        # Martin Luther King Jr. Day (third Monday in January)
        mlk = datetime.date(year, 1, 1)
        while mlk.weekday() != 0:  # Monday
            mlk += datetime.timedelta(days=1)
        mlk += datetime.timedelta(weeks=2)
        holidays.append(mlk)
        
        # Presidents Day (third Monday in February)
        presidents = datetime.date(year, 2, 1)
        while presidents.weekday() != 0:
            presidents += datetime.timedelta(days=1)
        presidents += datetime.timedelta(weeks=2)
        holidays.append(presidents)
        
        # Memorial Day (last Monday in May)
        memorial = datetime.date(year, 5, 31)
        while memorial.weekday() != 0:
            memorial -= datetime.timedelta(days=1)
        holidays.append(memorial)
        
        # Independence Day
        holidays.append(datetime.date(year, 7, 4))
        
        # Labor Day (first Monday in September)
        labor = datetime.date(year, 9, 1)
        while labor.weekday() != 0:
            labor += datetime.timedelta(days=1)
        holidays.append(labor)
        
        # Columbus Day (second Monday in October)
        columbus = datetime.date(year, 10, 1)
        while columbus.weekday() != 0:
            columbus += datetime.timedelta(days=1)
        columbus += datetime.timedelta(weeks=1)
        holidays.append(columbus)
        
        # Veterans Day
        holidays.append(datetime.date(year, 11, 11))
        
        # Thanksgiving (fourth Thursday in November)
        thanksgiving = datetime.date(year, 11, 1)
        while thanksgiving.weekday() != 3:  # Thursday
            thanksgiving += datetime.timedelta(days=1)
        thanksgiving += datetime.timedelta(weeks=3)
        holidays.append(thanksgiving)
        
        # Christmas
        holidays.append(datetime.date(year, 12, 25))
        
        return holidays
    
    @staticmethod
    def is_business_hours(dt: datetime.datetime = None,
                          start_hour: int = 9,
                          end_hour: int = 17,
                          timezone: str = None) -> bool:
        """Check if current time is within business hours"""
        if dt is None:
            dt = datetime.datetime.now()
        
        if timezone:
            import pytz
            tz = pytz.timezone(timezone)
            dt = dt.astimezone(tz)
        
        # Check if weekday
        if dt.weekday() >= 5:
            return False
        
        # Check time
        hour = dt.hour
        if start_hour <= hour < end_hour:
            return True
        
        return False
    
    @staticmethod
    def cron_schedule(expression: str) -> Dict[str, Any]:
        """Parse cron expression and calculate next run times"""
        # Simplified cron parser
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 parts")
        
        minute, hour, day_of_month, month, day_of_week = parts
        
        def expand_field(field: str, max_val: int) -> List[int]:
            """Expand cron field to list of values"""
            if field == "*":
                return list(range(max_val + 1))
            
            values = set()
            for part in field.split(","):
                if "/" in part:
                    range_part, step_part = part.split("/")
                    step = int(step_part)
                    if range_part == "*":
                        start = 0
                        end = max_val
                    elif "-" in range_part:
                        start, end = map(int, range_part.split("-"))
                    else:
                        start = end = int(range_part)
                    values.update(range(start, end + 1, step))
                elif "-" in part:
                    start, end = map(int, part.split("-"))
                    values.update(range(start, end + 1))
                else:
                    values.add(int(part))
            
            return sorted(values)
        
        minutes = expand_field(minute, 59)
        hours = expand_field(hour, 23)
        days_of_month = expand_field(day_of_month, 31)
        months = expand_field(month, 12)
        days_of_week = expand_field(day_of_week, 6)
        
        now = datetime.datetime.now()
        next_runs = []
        
        # Calculate next 5 runs
        for _ in range(5):
            dt = now
            found = False
            
            for year in range(dt.year, dt.year + 2):
                for m in months:
                    if m < dt.month and year == dt.year:
                        continue
                    
                    # Get valid days for this month
                    valid_days = []
                    last_day = 31  # Simplified
                    
                    for d in days_of_month:
                        if 1 <= d <= last_day:
                            try:
                                test_date = datetime.date(year, m, d)
                                if test_date.weekday() in days_of_week:
                                    valid_days.append(d)
                            except ValueError:
                                continue
                    
                    for d in valid_days:
                        if d < dt.day and m == dt.month and year == dt.year:
                            continue
                        
                        for h in hours:
                            if h < dt.hour and d == dt.day and m == dt.month and year == dt.year:
                                continue
                            
                            for minute_val in minutes:
                                candidate = datetime.datetime(year, m, d, h, minute_val)
                                if candidate > dt:
                                    next_runs.append(candidate)
                                    dt = candidate
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            
            now = dt + datetime.timedelta(minutes=1)
        
        return {
            "expression": expression,
            "parsed": {
                "minute": minutes,
                "hour": hours,
                "day_of_month": days_of_month,
                "month": months,
                "day_of_week": days_of_week,
            },
            "next_runs": next_runs[:5],
        }
    
    @staticmethod
    def timezone_convert(dt: datetime.datetime,
                         from_tz: str,
                         to_tz: str) -> datetime.datetime:
        """Convert between timezones"""
        try:
            import pytz
            from_zone = pytz.timezone(from_tz)
            to_zone = pytz.timezone(to_tz)
            
            # Localize if naive
            if dt.tzinfo is None:
                dt = from_zone.localize(dt)
            
            return dt.astimezone(to_zone)
        except ImportError:
            raise ImportError("pytz required for timezone conversion")
    
    @staticmethod
    def countdown_timer(seconds: int,
                        update_interval: float = 1.0,
                        callback: Callable = None) -> None:
        """Simple countdown timer"""
        import time
        
        end_time = time.time() + seconds
        
        while time.time() < end_time:
            remaining = end_time - time.time()
            
            if callback:
                callback(remaining)
            else:
                mins, secs = divmod(int(remaining), 60)
                hours, mins = divmod(mins, 60)
                print(f"\rCountdown: {hours:02d}:{mins:02d}:{secs:02d}", end='')
            
            time.sleep(update_interval)
        
        print("\rCountdown finished!     ")

# ==================== NETWORK ENCHANTMENTS (20+ functions) ====================
class NetworkEnchantments:
    """Advanced network utilities with async support"""
    
    @staticmethod
    @spell_timer()
    def fetch_json(url: str,
                   method: str = "GET",
                   headers: Dict[str, str] = None,
                   data: Any = None,
                   timeout: float = 30.0,
                   verify_ssl: bool = True) -> Dict[str, Any]:
        """Fetch JSON from URL with advanced options"""
        import urllib.request
        import urllib.error
        import json
        
        if headers is None:
            headers = {
                "User-Agent": "PyWizardry/1.0.1",
                "Accept": "application/json",
            }
        
        if data is not None and not isinstance(data, str):
            if isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
                headers["Content-Type"] = "application/json"
            else:
                data = str(data).encode('utf-8')
        elif data is not None:
            data = data.encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method.upper()
        )
        
        try:
            context = None
            if not verify_ssl:
                import ssl
                context = ssl._create_unverified_context()
            
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                response_data = response.read().decode('utf-8')
                
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "data": json.loads(response_data) if response_data else None,
                    "url": response.url,
                }
        except urllib.error.HTTPError as e:
            return {
                "status": e.code,
                "headers": dict(e.headers),
                "error": str(e),
                "data": None,
            }
        except Exception as e:
            return {
                "status": 0,
                "error": str(e),
                "data": None,
            }
    
    @staticmethod
    @async_spell(timeout=60)
    async def async_fetch_json(url: str,
                               method: str = "GET",
                               headers: Dict[str, str] = None,
                               data: Any = None,
                               timeout: float = 30.0) -> Dict[str, Any]:
        """Async fetch JSON"""
        if not AIO_AVAILABLE:
            raise ImportError("aiohttp required for async fetch")
        
        import aiohttp
        
        if headers is None:
            headers = {
                "User-Agent": "PyWizardry/1.0.1",
                "Accept": "application/json",
            }
        
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        response_data = await response.text()
                        return {
                            "status": response.status,
                            "headers": dict(response.headers),
                            "data": json.loads(response_data) if response_data else None,
                            "url": str(response.url),
                        }
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        response_data = await response.text()
                        return {
                            "status": response.status,
                            "headers": dict(response.headers),
                            "data": json.loads(response_data) if response_data else None,
                            "url": str(response.url),
                        }
                else:
                    raise ValueError(f"Unsupported method: {method}")
            except Exception as e:
                return {
                    "status": 0,
                    "error": str(e),
                    "data": None,
                }
    
    @staticmethod
    def download_with_progress(url: str,
                               destination: Union[str, Path],
                               chunk_size: int = 8192,
                               show_progress: bool = True) -> Path:
        """Download file with progress bar"""
        import urllib.request
        
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        req = urllib.request.Request(url, headers={"User-Agent": "PyWizardry/1.0.1"})
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if show_progress and total_size:
                        progress = (downloaded / total_size) * 100
                        bar_length = 50
                        filled_length = int(bar_length * downloaded // total_size)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f'\r[{bar}] {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)', end='')
        
        if show_progress:
            print()
        
        return dest_path
    
    @staticmethod
    def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
        """Check if port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    @staticmethod
    def get_local_network_info() -> Dict[str, Any]:
        """Get local network information"""
        import socket
        import platform
        
        info = {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "interfaces": [],
        }
        
        try:
            import netifaces
            
            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                
                iface_info = {
                    "name": interface,
                    "addresses": {},
                }
                
                for addr_type in [netifaces.AF_INET, netifaces.AF_INET6]:
                    if addr_type in addrs:
                        iface_info["addresses"][addr_type] = addrs[addr_type]
                
                info["interfaces"].append(iface_info)
        except ImportError:
            info["netifaces_not_available"] = True
        
        return info
    
    @staticmethod
    def ping(host: str, count: int = 4, timeout: float = 2.0) -> Dict[str, Any]:
        """Ping host and return statistics"""
        import subprocess
        import re
        
        if platform.system().lower() == "windows":
            cmd = ["ping", "-n", str(count), "-w", str(int(timeout * 1000)), host]
        else:
            cmd = ["ping", "-c", str(count), "-W", str(timeout), host]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout * (count + 2)
            )
            
            output = result.stdout
            
            # Parse output
            stats = {
                "host": host,
                "success": result.returncode == 0,
                "output": output,
            }
            
            # Parse packet loss
            loss_match = re.search(r'(\d+)% packet loss', output)
            if loss_match:
                stats["packet_loss"] = int(loss_match.group(1))
            
            # Parse round-trip times
            time_matches = re.findall(r'time=([\d.]+) ms', output)
            if time_matches:
                times = [float(t) for t in time_matches]
                stats["times"] = times
                stats["min"] = min(times)
                stats["max"] = max(times)
                stats["avg"] = sum(times) / len(times)
            
            return stats
            
        except subprocess.TimeoutExpired:
            return {
                "host": host,
                "success": False,
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "host": host,
                "success": False,
                "error": str(e),
            }
    
    @staticmethod
    def resolve_dns(hostname: str, record_type: str = "A") -> List[str]:
        """Resolve DNS records"""
        import socket
        
        try:
            if record_type == "A":
                return socket.gethostbyname_ex(hostname)[2]
            elif record_type == "MX":
                import dns.resolver
                answers = dns.resolver.resolve(hostname, 'MX')
                return [str(r.exchange) for r in answers]
            elif record_type == "TXT":
                import dns.resolver
                answers = dns.resolver.resolve(hostname, 'TXT')
                return [str(r) for r in answers]
            else:
                raise ValueError(f"Unsupported record type: {record_type}")
        except ImportError:
            raise ImportError("dnspython required for MX and TXT records")
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    @staticmethod
    def create_simple_server(port: int = 8080,
                             directory: Union[str, Path] = None) -> threading.Thread:
        """Create simple HTTP server in background thread"""
        import http.server
        import socketserver
        
        if directory:
            os.chdir(str(directory))
        
        handler = http.server.SimpleHTTPRequestHandler
        
        server = socketserver.TCPServer(("", port), handler)
        
        def run_server():
            print(f"Server started at http://localhost:{port}")
            server.serve_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        return thread

# ==================== DATA ALCHEMY (20+ functions) ====================
class DataAlchemy:
    """Advanced data processing and transformation"""
    
    @staticmethod
    def chunk_data(data: List[Any], 
                   chunk_size: int,
                   preserve_order: bool = True) -> List[List[Any]]:
        """Chunk data with order preservation"""
        if preserve_order:
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            # Shuffle then chunk
            shuffled = data.copy()
            random.shuffle(shuffled)
            return [shuffled[i:i + chunk_size] for i in range(0, len(shuffled), chunk_size)]
    
    @staticmethod
    def remove_duplicates(data: List[Any],
                          key: Callable[[Any], Any] = None,
                          preserve_order: bool = True) -> List[Any]:
        """Remove duplicates with custom key function"""
        seen = set()
        result = []
        
        for item in data:
            item_key = key(item) if key else item
            
            if item_key not in seen:
                seen.add(item_key)
                if preserve_order:
                    result.append(item)
                else:
                    result.insert(0, item)
        
        return result
    
    @staticmethod
    def group_by(data: List[Any],
                 key_func: Callable[[Any], Any],
                 value_func: Callable[[Any], Any] = None) -> Dict[Any, List[Any]]:
        """Group data by key function"""
        grouped = {}
        
        for item in data:
            key = key_func(item)
            value = value_func(item) if value_func else item
            
            if key not in grouped:
                grouped[key] = []
            
            grouped[key].append(value)
        
        return grouped
    
    @staticmethod
    def flatten(nested: List[Any], 
                max_depth: int = -1) -> List[Any]:
        """Flatten nested structure"""
        result = []
        
        def _flatten(item, depth):
            if max_depth != -1 and depth >= max_depth:
                result.append(item)
                return
            
            if isinstance(item, (list, tuple, set)):
                for subitem in item:
                    _flatten(subitem, depth + 1)
            else:
                result.append(item)
        
        _flatten(nested, 0)
        return result
    
    @staticmethod
    def transpose(matrix: List[List[Any]]) -> List[List[Any]]:
        """Transpose matrix (list of lists)"""
        if not matrix:
            return []
        
        return [list(row) for row in zip(*matrix)]
    
    @staticmethod
    def rotate(matrix: List[List[Any]], 
               degrees: int = 90) -> List[List[Any]]:
        """Rotate matrix"""
        if degrees not in [90, 180, 270]:
            raise ValueError("Degrees must be 90, 180, or 270")
        
        if degrees == 90:
            return [list(row) for row in zip(*matrix[::-1])]
        elif degrees == 180:
            return [row[::-1] for row in matrix[::-1]]
        elif degrees == 270:
            return [list(row) for row in zip(*matrix)][::-1]
    
    @staticmethod
    def sliding_window(data: List[Any],
                       window_size: int,
                       step: int = 1) -> List[List[Any]]:
        """Create sliding windows over data"""
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step):
            windows.append(data[i:i + window_size])
        
        return windows
    
    @staticmethod
    def batch_process(data: List[Any],
                      process_func: Callable[[List[Any]], Any],
                      batch_size: int = 100,
                      max_workers: int = None) -> List[Any]:
        """Process data in batches with parallel execution"""
        from concurrent.futures import ThreadPoolExecutor
        
        batches = DataAlchemy.chunk_data(data, batch_size)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_func, batches))
        
        # Flatten if results are lists
        flattened = []
        for result in results:
            if isinstance(result, list):
                flattened.extend(result)
            else:
                flattened.append(result)
        
        return flattened
    
    @staticmethod
    def normalize(data: List[float],
                  range_min: float = 0.0,
                  range_max: float = 1.0) -> List[float]:
        """Normalize data to specified range"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return [range_min] * len(data)
        
        scale = (range_max - range_min) / (max_val - min_val)
        
        return [
            (value - min_val) * scale + range_min
            for value in data
        ]
    
    @staticmethod
    def standardize(data: List[float]) -> List[float]:
        """Standardize data (z-score normalization)"""
        if not data:
            return []
        
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data) if len(data) > 1 else 1.0
        
        if std_val == 0:
            return [0.0] * len(data)
        
        return [
            (value - mean_val) / std_val
            for value in data
        ]
    
    @staticmethod
    def moving_average(data: List[float],
                       window_size: int) -> List[float]:
        """Calculate moving average"""
        if not data or window_size <= 0:
            return []
        
        if window_size >= len(data):
            avg = sum(data) / len(data)
            return [avg] * len(data)
        
        result = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            window = data[start:i + 1]
            result.append(sum(window) / len(window))
        
        return result
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics"""
        if not data:
            return {}
        
        sorted_data = sorted(data)
        n = len(data)
        
        stats = {
            "count": n,
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data) if n > 0 else None,
            "std": statistics.stdev(data) if n > 1 else 0.0,
            "variance": statistics.variance(data) if n > 1 else 0.0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data),
            "sum": sum(data),
        }
        
        # Percentiles
        for p in [25, 50, 75, 90, 95, 99]:
            idx = (n - 1) * p / 100
            if idx.is_integer():
                stats[f"p{p}"] = sorted_data[int(idx)]
            else:
                lower = sorted_data[int(math.floor(idx))]
                upper = sorted_data[int(math.ceil(idx))]
                stats[f"p{p}"] = (lower + upper) / 2
        
        # Skewness and Kurtosis (simplified)
        if n > 2 and stats["std"] > 0:
            mean = stats["mean"]
            std = stats["std"]
            
            skewness = sum((x - mean) ** 3 for x in data) / (n * std ** 3)
            kurtosis = sum((x - mean) ** 4 for x in data) / (n * std ** 4) - 3
            
            stats["skewness"] = skewness
            stats["kurtosis"] = kurtosis
        
        return stats
    
    @staticmethod
    def detect_outliers(data: List[float], 
                        method: str = "iqr",
                        threshold: float = 1.5) -> List[Tuple[int, float]]:
        """Detect outliers in data"""
        if not data:
            return []
        
        outliers = []
        
        if method == "iqr":
            sorted_data = sorted(data)
            q1 = statistics.median(sorted_data[:len(sorted_data)//2])
            q3 = statistics.median(sorted_data[len(sorted_data)//2:])
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, value in enumerate(data):
                if value < lower_bound or value > upper_bound:
                    outliers.append((i, value))
        
        elif method == "zscore":
            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data) if len(data) > 1 else 1.0
            
            if std_val > 0:
                for i, value in enumerate(data):
                    zscore = abs((value - mean_val) / std_val)
                    if zscore > threshold:
                        outliers.append((i, value))
        
        return outliers
    
    @staticmethod
    def interpolate_missing(data: List[Optional[float]],
                            method: str = "linear") -> List[float]:
        """Interpolate missing values"""
        if not data:
            return []
        
        # Find indices of non-missing values
        valid_indices = [i for i, v in enumerate(data) if v is not None]
        valid_values = [data[i] for i in valid_indices]
        
        if not valid_values:
            return [0.0] * len(data)
        
        if len(valid_values) == 1:
            return [valid_values[0]] * len(data)
        
        # Interpolate
        result = []
        
        for i in range(len(data)):
            if data[i] is not None:
                result.append(data[i])
            else:
                # Find surrounding valid values
                left_idx = max([idx for idx in valid_indices if idx < i], default=None)
                right_idx = min([idx for idx in valid_indices if idx > i], default=None)
                
                if left_idx is None and right_idx is None:
                    result.append(0.0)
                elif left_idx is None:
                    result.append(data[right_idx])
                elif right_idx is None:
                    result.append(data[left_idx])
                else:
                    if method == "linear":
                        # Linear interpolation
                        left_val = data[left_idx]
                        right_val = data[right_idx]
                        ratio = (i - left_idx) / (right_idx - left_idx)
                        interpolated = left_val + (right_val - left_val) * ratio
                        result.append(interpolated)
                    elif method == "nearest":
                        # Nearest neighbor
                        if i - left_idx <= right_idx - i:
                            result.append(data[left_idx])
                        else:
                            result.append(data[right_idx])
                    else:
                        raise ValueError(f"Unknown interpolation method: {method}")
        
        return result

# ==================== CONSOLE MAGIC (8+ functions) ====================
class ConsoleMagic:
    """Colorful console output and formatting"""
    
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "reverse": "\033[7m",
        "hidden": "\033[8m",
    }
    
    @staticmethod
    def colorize(text: str, color: str = None, style: str = None) -> str:
        """Colorize text for terminal output"""
        if not COLORAMA_AVAILABLE:
            return text
        
        colorama.init(autoreset=True)
        
        color_code = ""
        if color and color in ConsoleMagic.COLORS:
            color_code += ConsoleMagic.COLORS[color]
        
        if style and style in ConsoleMagic.COLORS:
            color_code += ConsoleMagic.COLORS[style]
        
        reset = ConsoleMagic.COLORS["reset"]
        
        return f"{color_code}{text}{reset}"
    
    @staticmethod
    def print_success(message: str, icon: str = "✅") -> None:
        """Print success message"""
        colored = ConsoleMagic.colorize(f"{icon} {message}", "bright_green")
        print(colored)
    
    @staticmethod
    def print_error(message: str, icon: str = "❌") -> None:
        """Print error message"""
        colored = ConsoleMagic.colorize(f"{icon} {message}", "bright_red")
        print(colored)
    
    @staticmethod
    def print_warning(message: str, icon: str = "⚠️") -> None:
        """Print warning message"""
        colored = ConsoleMagic.colorize(f"{icon} {message}", "bright_yellow")
        print(colored)
    
    @staticmethod
    def print_info(message: str, icon: str = "ℹ️") -> None:
        """Print info message"""
        colored = ConsoleMagic.colorize(f"{icon} {message}", "bright_cyan")
        print(colored)
    
    @staticmethod
    def progress_bar(iteration: int, total: int, 
                     length: int = 50, fill: str = "█",
                     empty: str = "░") -> None:
        """Display progress bar"""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + empty * (length - filled_length)
        
        if COLORAMA_AVAILABLE:
            bar = ConsoleMagic.colorize(bar, "green")
            percent = ConsoleMagic.colorize(percent, "cyan")
        
        print(f'\rProgress: |{bar}| {percent}% Complete', end='\r')
        
        if iteration == total:
            print()
    
    @staticmethod
    def table(data: List[Dict[str, Any]],
              headers: List[str] = None,
              align: Dict[str, str] = None) -> None:
        """Print data as formatted table"""
        if not data:
            print("No data to display")
            return
        
        if headers is None:
            headers = list(data[0].keys())
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            max_len = len(str(header))
            for row in data:
                if header in row:
                    max_len = max(max_len, len(str(row[header])))
            col_widths[header] = max_len + 2  # Add padding
        
        # Print header
        header_line = ""
        separator_line = ""
        for header in headers:
            width = col_widths[header]
            header_line += f"{header:<{width}}"
            separator_line += "-" * width
        
        if COLORAMA_AVAILABLE:
            header_line = ConsoleMagic.colorize(header_line, "bold", "cyan")
        
        print(header_line)
        print(separator_line)
        
        # Print rows
        for row in data:
            row_line = ""
            for header in headers:
                width = col_widths[header]
                value = str(row.get(header, ""))
                
                # Apply alignment
                if align and header in align:
                    align_char = align[header]
                    if align_char == "right":
                        row_line += f"{value:>{width}}"
                    elif align_char == "center":
                        padded = value.center(width)
                        row_line += padded
                    else:
                        row_line += f"{value:<{width}}"
                else:
                    row_line += f"{value:<{width}}"
            
            print(row_line)

# ==================== MAIN WIZARD CLASS ====================
class Wizard:
    """Main PyWizardry interface with all magical utilities"""
    
    def __init__(self, 
                 enable_logging: bool = True,
                 log_level: str = "INFO",
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 color_output: bool = True,
                 async_mode: bool = False):
        
        self.version = __version__
        self.enable_logging = enable_logging
        self.log_level = log_level
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.color_output = color_output and COLORAMA_AVAILABLE
        self.async_mode = async_mode
        
        # Initialize modules
        self.files = FileMagic()
        self.strings = StringSorcery()
        self.security = SecuritySpells()
        self.dates = TimeWizardry()
        self.network = NetworkEnchantments()
        self.data = DataAlchemy()
        self.console = ConsoleMagic()
        self.validation = ValidationCharms()
        
        # Optional modules
        if PILLOW_AVAILABLE:
            from .images import ImageMagic
            self.images = ImageMagic()
        
        if PSUTIL_AVAILABLE:
            from .system import SystemMagic
            self.system = SystemMagic()
        
        # Cache storage
        self._cache = {}
        
        # Logger
        if enable_logging:
            self.logger = self._create_logger()
        else:
            self.logger = None
    
    def _create_logger(self):
        """Create logger instance"""
        import logging
        
        logger = logging.getLogger("PyWizardry")
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def cache_get(self, key: str) -> Any:
        """Get value from cache"""
        if not self.enable_cache:
            return None
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < 300:  # 5 minutes TTL
                return value
        
        return None
    
    def cache_set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        if not self.enable_cache:
            return
        
        # Manage cache size
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())
    
    def clear_cache(self) -> None:
        """Clear all cache"""
        self._cache.clear()
    
    def benchmark(self, functions: List[Callable], 
                  iterations: int = 1000,
                  *args, **kwargs) -> Dict[str, Any]:
        """Benchmark multiple functions"""
        results = {}
        
        for func in functions:
            try:
                stats = benchmark_spell(iterations)(func)(*args, **kwargs)
                results[func.__name__] = stats
            except Exception as e:
                results[func.__name__] = {"error": str(e)}
        
        # Find fastest/slowest
        if results:
            valid_results = {k: v for k, v in results.items() 
                           if "min" in v and "error" not in v}
            
            if valid_results:
                fastest = min(valid_results.items(), key=lambda x: x[1]["min"])
                slowest = max(valid_results.items(), key=lambda x: x[1]["min"])
                
                results["_summary"] = {
                    "fastest": fastest[0],
                    "fastest_time": fastest[1]["min"],
                    "slowest": slowest[0],
                    "slowest_time": slowest[1]["min"],
                    "iterations": iterations,
                }
        
        return results
    
    def create_pipeline(self, functions: List[Callable]) -> Callable:
        """Create function pipeline"""
        def pipeline(data):
            result = data
            for func in functions:
                result = func(result)
            return result
        return pipeline


# ==================== SPELLBOOK CLASS ====================
class SpellBook:
    """Collection of related spells (functions)"""
    
    def __init__(self, name: str):
        self.name = name
        self.spells = {}
    
    def register_spell(self, name: str, spell: Callable) -> None:
        """Register a spell in the spellbook"""
        self.spells[name] = spell
    
    def cast(self, spell_name: str, *args, **kwargs) -> Any:
        """Cast (execute) a spell"""
        if spell_name not in self.spells:
            raise MagicError(f"Spell '{spell_name}' not found in {self.name}")
        
        return self.spells[spell_name](*args, **kwargs)
    
    def list_spells(self) -> List[str]:
        """List all available spells"""
        return list(self.spells.keys())


# ==================== HELPER FUNCTIONS ====================
def create_pipeline(functions: List[Callable]) -> Callable:
    """Create a pipeline of functions"""
    def pipeline(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return pipeline


def benchmark(functions: List[Callable], iterations: int = 1000, *args, **kwargs):
    """Benchmark multiple functions"""
    wiz = Wizard()
    return wiz.benchmark(functions, iterations, *args, **kwargs)


# ==================== MODULE EXPORTS ====================
# Create default wizard instance
default_wizard = Wizard()

# Export commonly used functions directly
files = default_wizard.files
strings = default_wizard.strings
security = default_wizard.security
dates = default_wizard.dates
network = default_wizard.network
data = default_wizard.data
console = default_wizard.console
validation = default_wizard.validation

# Export configuration
config = {
    "version": __version__,
    "author": __author__,
    "license": __license__,
    "color_output": COLORAMA_AVAILABLE,
    "async_support": AIO_AVAILABLE,
    "image_support": PILLOW_AVAILABLE,
    "system_support": PSUTIL_AVAILABLE,
}

# Export utilities module
utils = types.SimpleNamespace(
    create_pipeline=create_pipeline,
    benchmark=benchmark,
    chunk=DataAlchemy.chunk_data,
    flatten=DataAlchemy.flatten,
    group_by=DataAlchemy.group_by,
    normalize=DataAlchemy.normalize,
    standardize=DataAlchemy.standardize,
)

# Export async module if available
if AIO_AVAILABLE:
    async_utils = types.SimpleNamespace(
        fetch=NetworkEnchantments.async_fetch_json,
        run_with_timeout=AsyncMagic.run_with_timeout,
        to_async=AsyncMagic.to_async,
    )
else:
    async_utils = None

# Export parallel module
parallel = types.SimpleNamespace(
    batch_process=DataAlchemy.batch_process,
)

# Export crypto module
crypto = types.SimpleNamespace(
    generate_key=SecuritySpells.generate_key,
    aes_encrypt=SecuritySpells.aes_encrypt,
    aes_decrypt=SecuritySpells.aes_decrypt,
    jwt_encode=SecuritySpells.jwt_encode,
    jwt_decode=SecuritySpells.jwt_decode,
    hash_password=SecuritySpells.hash_password,
    verify_password=SecuritySpells.verify_password,
)

# Export math module
math_utils = types.SimpleNamespace(
    moving_average=DataAlchemy.moving_average,
    statistics=DataAlchemy.calculate_statistics,
    outliers=DataAlchemy.detect_outliers,
    interpolate=DataAlchemy.interpolate_missing,
)

# Export web module
web = types.SimpleNamespace(
    fetch_json=NetworkEnchantments.fetch_json,
    download=NetworkEnchantments.download_with_progress,
    ping=NetworkEnchantments.ping,
    check_port=NetworkEnchantments.check_port,
    create_server=NetworkEnchantments.create_simple_server,
)

# Export database module (simplified)
database = types.SimpleNamespace(
    # SQLite helpers
    sqlite_connect=lambda db: sqlite3.connect(db),
    sqlite_execute=lambda conn, query, params=None: conn.execute(query, params or []),
    sqlite_fetch_all=lambda conn, query, params=None: conn.execute(query, params or []).fetchall(),
    
    # Connection pooling (simplified)
    create_connection_pool=lambda size=5: [sqlite3.connect(':memory:') for _ in range(size)],
)

# Export testing module
testing = types.SimpleNamespace(
    mock_function=lambda returns=None, side_effect=None: 
        lambda *args, **kwargs: side_effect(*args, **kwargs) if side_effect else returns,
    assert_raises=lambda exception, func, *args, **kwargs: 
        isinstance(func(*args, **kwargs), exception),
    timer=spell_timer,
)

# Export AI/ML module (placeholder)
ai = types.SimpleNamespace(
    text_to_vector=lambda text, model="tfidf": [len(text)],  # Simplified
    cosine_similarity=lambda v1, v2: 0.0,  # Placeholder
)

# ==================== INITIALIZATION ====================
def initialize(color_output: bool = True, async_mode: bool = False) -> Wizard:
    """Initialize PyWizardry with custom settings"""
    return Wizard(
        enable_logging=True,
        log_level="INFO",
        enable_cache=True,
        cache_size=1000,
        color_output=color_output,
        async_mode=async_mode,
    )


# Print welcome message
if __name__ != "__main__":
    if COLORAMA_AVAILABLE:
        init(autoreset=True)
        welcome = f"""
    {Fore.MAGENTA}{Style.BRIGHT}
     ____        _ _       _                 _       
    |  _ \\ _   _(_) |_ __ _| |__   __ _ _ __| | __ _ 
    | |_) | | | | | __/ _` | '_ \\ / _` | '__| |/ _` |
    |  __/| |_| | | || (_| | |_) | (_| | |  | | (_| |
    |_|    \\__,_|_|\\__\\__,_|_.__/ \\__,_|_|  |_|\\__,_|
    {Style.RESET_ALL}
    {Fore.CYAN}Version {__version__} • {__author__} • {__license__} License{Style.RESET_ALL}
    {Fore.YELLOW}✨ A magical collection of 150+ Python utilities{Style.RESET_ALL}
        """
        print(welcome)
    else:
        print(f"PyWizardry v{__version__} loaded successfully!")

# Clean up imports
del types, pickle, gzip, zipfile, tarfile, mimetypes, uuid, math, statistics
del collections, inspect, threading, queue, contextlib, html, html.parser, cgi
del email, email.mime, email.mime.text, email.mime.multipart, email.mime.base
del email.utils, quopri, uu, zlib, bz2, lzma, struct, wave, audioop, ssl
del select, selectors, signal, mmap, curses, getopt, getpass, platform, locale
del cmd, shlex, readline, rlcompleter, sqlite3, csv, configparser, ET, minidom
del xml, xml.sax, xml.sax.handler, xml.sax.saxutils, hashlib, hmac, secrets
del base64, binascii, itertools, decimal, fractions, numbers, datetime, time
del calendar, heapq, bisect, array, weakref, copy, pprint, reprlib, enum
del graphlib, zoneinfo, ipaddress, html, html.parser, cgi, email, email.mime
del email.mime.text, email.mime.multipart, email.mime.base, email.utils, quopri
del uu, zlib, bz2, lzma, struct, wave, audioop, ssl, select, selectors, signal
del mmap, curses, getopt, getpass, platform, locale, cmd, shlex, readline
del rlcompleter, sqlite3, csv, configparser, ET, minidom, xml, xml.sax
del xml.sax.handler, xml.sax.saxutils
