<div align="center">

# 🧙‍♂️ PyWizardry

### *A magical collection of 150+ Python utilities for modern development*

[![PyPI version](https://img.shields.io/pypi/v/PyWizardry.svg?style=flat-square&color=blue)](https://pypi.org/project/PyWizardry/)
[![Downloads](https://static.pepy.tech/badge/pywizardry/month?style=flat-square)](https://pepy.tech/projects/pywizardry)
[![Python Versions](https://img.shields.io/pypi/pyversions/PyWizardry.svg?style=flat-square&color=green)](https://pypi.org/project/PyWizardry/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen?style=flat-square)](https://github.com/Saifullah10141/pywizardry)

[📖 Documentation](https://pywizardry.vercel.app/docs) • [🚀 Tutorials](https://pywizardry.vercel.app/tutorials) • [🔗 API Reference](https://pywizardry.vercel.app/api) • [💬 Discussions](https://github.com/Saifullah10141/pywizardry)

</div>

---

## 🎉 What's New in v1.0.2

- ✨ **150+ utility functions** across 16 specialized modules
- 🔥 **Advanced async utilities** with built-in rate limiting and task queues
- 📊 **Data science helpers** (pandas/numpy integration)
- 🔒 **Enhanced security** - AES/RSA encryption, JWT, OAuth2
- 🎨 **Colorful console output** with colorama integration
- 📱 **Web scraping toolkit** with anti-detection features
- 🗃️ **Database utilities** for SQLite and PostgreSQL
- 📈 **Chart generation** with matplotlib
- 🤖 **AI/ML utilities** for basic machine learning tasks
- 🧪 **Testing toolkit** with mocking and fixtures
- ⚡ **Performance optimizations** and benchmarking tools

---

## 🌟 Why PyWizardry?

PyWizardry is a comprehensive Python utility library that eliminates boilerplate code and accelerates development. Built with production-grade quality, extensive test coverage (95%+), and zero required dependencies, it's designed to be your go-to toolkit for modern Python development.

## ✨ Complete Feature Set

<table>
<tr>
<td width="50%">

### 📁 File System Magic (35+ functions)
- Safe file operations with atomic writes
- Recursive directory operations & bulk processing
- File type detection and validation
- ZIP/TAR archive utilities with progress bars
- File watcher and change detection
- Memory-mapped file operations
- Backup and restore utilities

</td>
<td width="50%">

### 🔤 String Sorcery (25+ functions)
- Advanced string manipulation & formatting
- Natural language processing helpers
- Regex pattern generators
- Text similarity and Levenshtein distance
- Unicode and encoding utilities
- Template rendering engines
- Markdown/HTML converters

</td>
</tr>
<tr>
<td width="50%">

### 🔒 Security Spells (20+ functions)
- AES/RSA encryption & decryption
- JWT token generation & validation
- OAuth2 authentication utilities
- Password strength checking & hashing
- Security headers generation
- CSRF protection tokens
- Input sanitization (XSS/SQL injection)

</td>
<td width="50%">

### ⏰ Time Wizardry (15+ functions)
- Timezone-aware datetime operations
- Cron expression parsing
- Business day calculations
- Time series generation
- Countdown timers & scheduling
- Human-readable date formatting
- Duration parsing and formatting

</td>
</tr>
<tr>
<td width="50%">

### 🌐 Network Enchantments (20+ functions)
- Advanced HTTP client with retries
- WebSocket utilities
- Rate limiting and throttling
- Proxy support and rotation
- DNS resolution utilities
- Network health checks
- Protocol helpers (REST/GraphQL)

</td>
<td width="50%">

### ⚡ Async Conjuring (15+ functions)
- Async context managers
- Parallel processing with worker pools
- Task queues and schedulers
- Pub/Sub pattern implementation
- Event-driven architecture helpers
- Coroutine pooling & rate limiting

</td>
</tr>
<tr>
<td width="50%">

### 📊 Data Alchemy (20+ functions)
- CSV/JSON/XML/YAML processors
- Data validation schemas
- Transformation pipelines
- Statistical analysis helpers
- Data visualization generators
- DataFrame operations
- Machine learning preprocessing

</td>
<td width="50%">

### 🎨 Console Magic (8+ functions)
- Colorful output (success/warning/error)
- Progress bars and spinners
- Tables and formatted output
- Interactive prompts
- ANSI color codes
- Terminal size detection
- Cross-platform compatibility

</td>
</tr>
<tr>
<td width="50%">

### 🗃️ Database Utilities (10+ functions)
- SQLite helper functions
- PostgreSQL connection pooling
- Query builders and ORM helpers
- Migration utilities
- Transaction management
- Connection retry logic
- Database backup tools

</td>
<td width="50%">

### 🤖 AI/ML Utilities (10+ functions)
- Text vectorization (BERT/Word2Vec)
- Cosine similarity calculations
- Feature extraction helpers
- Model evaluation metrics
- Data augmentation tools
- Preprocessing pipelines
- Basic neural network utilities

</td>
</tr>
<tr>
<td width="50%">

### 🧪 Testing Toolkit (10+ functions)
- Mock database generators
- Mock HTTP servers
- Fixture generators
- Test data factories
- Assertion helpers
- Performance benchmarking
- Coverage utilities

</td>
<td width="50%">

### 📱 Web Utilities (12+ functions)
- HTML/CSS/JS parsers
- Web scraping with anti-detection
- Cookie management
- Session handling
- Form data extraction
- Sitemap generators
- SEO analysis tools

</td>
</tr>
</table>

**Additional Modules**: `wiz.math` (15+ functions), `wiz.validation` (10+ functions), `wiz.crypto` (8+ functions), `wiz.parallel` (8+ functions)

---

## 🚀 Installation

```bash
# Basic installation (zero dependencies)
pip install PyWizardry

# With all optional features
pip install PyWizardry[full]

# Specific feature sets
pip install PyWizardry[crypto]      # Encryption features
pip install PyWizardry[async]       # Async utilities
pip install PyWizardry[data]        # Data science (pandas, numpy)
pip install PyWizardry[web]         # Web scraping (BeautifulSoup)
pip install PyWizardry[ai]          # AI/ML utilities
pip install PyWizardry[database]    # Database connectors

# Development installation
pip install PyWizardry[full,extras]

# Upgrade to latest version
pip install --upgrade PyWizardry
```

## 🎯 Quick Start

```python
import pywizardry as pw

# Initialize with all features
wiz = pw.Wizard()

# 🎨 Colorful Console Output
wiz.console.success("✓ Operation completed!")
wiz.console.warning("⚠ This is a warning")
wiz.console.error("✗ Something went wrong")
wiz.console.info("ℹ Processing data...")

# 📁 Advanced File Operations
files = wiz.files.find_recursive("*.py", size_limit="1MB")
wiz.files.backup_directory("/path/to/data", compression="zip")
wiz.files.watch("/path/to/watch", on_change=lambda f: print(f"Changed: {f}"))

# 🔤 String Manipulation
similarity = wiz.strings.similarity("hello", "hallo")  # 0.8
cleaned = wiz.strings.remove_special_chars("Hello@World#123")  # "HelloWorld123"
slug = wiz.strings.slugify("My Awesome Blog Post!")  # "my-awesome-blog-post"

# 🔒 Security & Encryption
encrypted = wiz.security.aes_encrypt("secret data", "password")
decrypted = wiz.security.aes_decrypt(encrypted, "password")
token = wiz.security.jwt_encode({"user_id": 123}, "secret_key")
payload = wiz.security.jwt_decode(token, "secret_key")

# ✅ Password Validation
strength = wiz.security.check_password_strength("MyP@ssw0rd!")
hashed = wiz.security.hash_password("user_password")
is_valid = wiz.security.verify_password("user_password", hashed)

# ⏰ Time & Date Utilities
next_day = wiz.dates.next_business_day()
is_open = wiz.dates.is_business_hours()
schedule = wiz.dates.cron_schedule("*/5 * * * *")  # Every 5 minutes
human_time = wiz.dates.humanize(datetime.now() - timedelta(hours=2))  # "2 hours ago"

# 🌐 Network Operations
response = wiz.network.fetch_json(
    "https://api.example.com/data",
    headers={"Authorization": "Bearer token"}
)

wiz.network.download_with_progress(
    "https://example.com/large-file.zip",
    "downloads/file.zip"
)

# ⚡ Async Magic
import asyncio

async def fetch_multiple():
    urls = ["https://api1.com", "https://api2.com", "https://api3.com"]
    results = await wiz.async_utils.gather_with_rate_limit(
        urls,
        max_concurrent=10,
        requests_per_second=5
    )
    return results

# 📊 Data Processing
df = wiz.data.csv_to_dataframe(
    "data.csv",
    dtype={"age": "int", "salary": "float"}
)

stats = wiz.data.calculate_statistics(df["column"])
wiz.data.generate_histogram(df["values"], title="Distribution", save_path="plot.png")

# Data transformation pipeline
cleaned_data = wiz.data.transform(
    raw_data,
    normalize=True,
    remove_duplicates=True,
    fill_missing="mean"
)

# 🗃️ Database Operations
with wiz.database.sqlite_connection("mydb.db") as conn:
    users = wiz.database.query(conn, "SELECT * FROM users WHERE age > ?", (18,))
    wiz.database.insert(conn, "users", {"name": "Alice", "age": 30})

# 🤖 AI/ML Utilities
vector = wiz.ai.text_to_vector("Hello world", model="bert")
similarity = wiz.ai.cosine_similarity(vector1, vector2)
features = wiz.ai.extract_features(dataset, method="pca", n_components=10)

# 🧪 Testing Utilities
mock_db = wiz.testing.mock_database(initial_data={"users": []})
test_server = wiz.testing.mock_http_server(port=8080)
fake_user = wiz.testing.generate_fake_user()

# 📱 Web Scraping
html = wiz.web.fetch_page("https://example.com")
links = wiz.web.extract_links(html)
data = wiz.web.extract_table(html, index=0)

# 🔢 Mathematical Utilities
primes = wiz.math.generate_primes(100)
gcd = wiz.math.gcd(48, 18)
is_prime = wiz.math.is_prime(17)

# ⚡ Quick One-Liners
token = wiz.quick.token(length=32)
uuid = wiz.quick.uuid()
hash_value = wiz.quick.hash("data")
timestamp = wiz.quick.timestamp()
```

---

## 🏗️ Advanced Usage

### Custom Wizard Configuration

```python
from pywizardry import Wizard, SpellBook

# Create custom wizard with specific settings
custom_wiz = Wizard(
    enable_logging=True,
    log_level="DEBUG",
    log_file="logs/pywizardry.log",
    enable_cache=True,
    cache_size=1000,
    cache_ttl=300,
    color_output=True,
    async_mode=True,
    rate_limit=100  # requests per minute
)

# Create domain-specific spell book
data_spells = SpellBook("data_processing")
data_spells.register_spell("clean", lambda x: x.strip().lower())
data_spells.register_spell("validate", wiz.validate.email)
data_spells.register_spell("transform", custom_transform_function)

# Use the spell book
result = data_spells.cast("clean", "  HELLO WORLD  ")  # "hello world"
```

### Pipeline Processing

```python
# Create processing pipeline
pipeline = wiz.create_pipeline([
    wiz.strings.normalize,
    wiz.strings.remove_stopwords,
    wiz.strings.lemmatize,
    wiz.ai.vectorize
])

# Process single item
processed = pipeline.process("The quick brown fox jumps over the lazy dog")

# Batch processing with progress bar
results = pipeline.batch_process(documents, show_progress=True)

# Async pipeline for large datasets
async def process_large_dataset():
    async_pipeline = wiz.create_async_pipeline([
        wiz.async_utils.fetch_data,
        wiz.async_utils.clean_data,
        wiz.async_utils.validate_data,
        wiz.async_utils.save_to_database
    ])
    
    await async_pipeline.process_stream(data_stream, chunk_size=1000)
```

### Event-Driven Architecture

```python
# Create event bus
event_bus = wiz.events.create_bus()

# Register event handlers
@event_bus.on("user.created")
async def handle_user_created(user):
    print(f"New user: {user['email']}")
    await wiz.mail.send_welcome_email(user)
    await wiz.analytics.track("user_signup", user)

@event_bus.on("payment.received")
async def handle_payment(payment):
    await wiz.database.update_order_status(payment["order_id"], "paid")
    await wiz.notifications.send(payment["user_id"], "Payment confirmed")
    await wiz.events.emit("order.fulfilled", payment)

@event_bus.on("order.fulfilled")
async def handle_order_fulfilled(order):
    await wiz.shipping.create_label(order)
    await wiz.inventory.update_stock(order["items"])

# Emit events
await event_bus.emit("user.created", {
    "email": "user@example.com",
    "name": "Alice"
})
```

### Advanced Async Patterns

```python
import asyncio
from pywizardry import Wizard

wiz = Wizard()

# Rate-limited concurrent requests
async def fetch_all_data():
    urls = [f"https://api.example.com/items/{i}" for i in range(100)]
    
    # Fetch with rate limiting (max 10 concurrent, 5 requests/second)
    results = await wiz.async_utils.gather_with_rate_limit(
        urls,
        max_concurrent=10,
        requests_per_second=5,
        retry_on_failure=True,
        max_retries=3
    )
    return results

# Task queue with workers
async def process_with_queue():
    queue = wiz.async_utils.create_task_queue(
        num_workers=5,
        worker_function=process_item,
        on_error=handle_error
    )
    
    # Add tasks
    for item in items:
        await queue.add_task(item)
    
    # Wait for completion
    await queue.join()

# Pub/Sub pattern
pubsub = wiz.async_utils.create_pubsub()

@pubsub.subscribe("data_updates")
async def handle_update(message):
    await process_update(message)

await pubsub.publish("data_updates", {"type": "new_data", "count": 100})
```

### Data Science Pipeline

```python
import pandas as pd
from pywizardry import Wizard

wiz = Wizard()

# Load and preprocess data
df = wiz.data.load_csv("raw_data.csv", parse_dates=["timestamp"])

# Data cleaning pipeline
cleaned = (
    wiz.data.remove_duplicates(df)
    .pipe(wiz.data.handle_missing, strategy="interpolate")
    .pipe(wiz.data.normalize_columns, columns=["value1", "value2"])
    .pipe(wiz.data.detect_outliers, method="iqr")
    .pipe(wiz.data.remove_outliers)
)

# Feature engineering
features = wiz.ai.extract_features(
    cleaned,
    methods=["polynomial", "interaction", "statistical"],
    degree=2
)

# Statistical analysis
stats = wiz.data.calculate_statistics(
    features,
    metrics=["mean", "std", "skew", "kurtosis"]
)

# Visualization
wiz.data.plot_distribution(features["target"], save_path="dist.png")
wiz.data.plot_correlation_matrix(features, save_path="correlation.png")
wiz.data.plot_time_series(df["timestamp"], df["value"], save_path="timeseries.png")
```

### Performance Benchmarking

```python
# Benchmark multiple functions
results = wiz.benchmark([
    ("slugify", wiz.strings.slugify, "Test String 123"),
    ("hash", wiz.security.hash, "data"),
    ("chunk", wiz.data.chunk_list, list(range(1000)))
], iterations=10000)

print(f"Fastest: {results.fastest.name} - {results.fastest.time:.4f}s")
print(f"Slowest: {results.slowest.name} - {results.slowest.time:.4f}s")

# Profile specific function
with wiz.profiler("data_processing"):
    result = expensive_operation()

# Memory usage tracking
with wiz.memory_tracker() as tracker:
    large_data = load_large_dataset()
    processed = process_data(large_data)
    print(f"Peak memory: {tracker.peak_mb:.2f} MB")
```

---

## 📚 Complete Module Reference

### Core Modules

| Module | Functions | Description | Key Features |
|--------|-----------|-------------|--------------|
| `wiz.files` | 35+ | File system operations | Atomic writes, bulk processing, file watching, archives |
| `wiz.strings` | 25+ | String manipulation | NLP, regex, similarity, encoding, templates |
| `wiz.security` | 20+ | Security & encryption | AES/RSA, JWT, OAuth2, sanitization, hashing |
| `wiz.dates` | 15+ | Date/time utilities | Timezone handling, cron, business days, humanization |
| `wiz.network` | 20+ | HTTP & networking | Rate limiting, retries, WebSocket, proxy support |
| `wiz.async_utils` | 15+ | Async programming | Task queues, pub/sub, rate limiting, pooling |
| `wiz.data` | 20+ | Data processing | CSV/JSON/YAML, pipelines, statistics, visualization |
| `wiz.database` | 10+ | Database helpers | SQLite, PostgreSQL, migrations, connection pooling |
| `wiz.testing` | 10+ | Testing utilities | Mocks, fixtures, factories, benchmarking |
| `wiz.ai` | 10+ | AI/ML helpers | Vectorization, similarity, feature extraction |
| `wiz.console` | 8+ | Console output | Colors, progress bars, tables, prompts |
| `wiz.web` | 12+ | Web utilities | Scraping, parsing, SEO, forms |
| `wiz.math` | 15+ | Mathematical ops | Primes, statistics, algebra, geometry |
| `wiz.validation` | 10+ | Data validation | Email, URL, phone, credit card, custom schemas |
| `wiz.crypto` | 8+ | Cryptography | Hashing, encoding, key generation |
| `wiz.parallel` | 8+ | Parallel processing | Multiprocessing, threading, job distribution |

### Quick Reference

```python
# File Operations
wiz.files.find_recursive("*.py")
wiz.files.safe_write("file.txt", "content")
wiz.files.backup_directory("/path")
wiz.files.watch("/path", on_change=handler)

# String Utilities
wiz.strings.slugify("My Title")
wiz.strings.similarity("hello", "hallo")
wiz.strings.remove_special_chars("text@123")
wiz.strings.truncate("long text", 50)

# Security
wiz.security.aes_encrypt(data, key)
wiz.security.jwt_encode(payload, secret)
wiz.security.hash_password(password)
wiz.security.sanitize_input(user_input)

# Date/Time
wiz.dates.next_business_day()
wiz.dates.is_business_hours()
wiz.dates.cron_schedule("*/5 * * * *")
wiz.dates.humanize(datetime)

# Network
wiz.network.fetch_json(url, headers)
wiz.network.download_with_progress(url, path)
wiz.network.check_connection(host, port)

# Data Processing
wiz.data.csv_to_dataframe("file.csv")
wiz.data.calculate_statistics(data)
wiz.data.generate_histogram(values)
wiz.data.transform(data, normalize=True)

# Async Utilities
await wiz.async_utils.gather_with_rate_limit(tasks)
wiz.async_utils.create_task_queue(workers=5)
wiz.async_utils.create_pubsub()

# Console
wiz.console.success("Message")
wiz.console.progress_bar(total=100)
wiz.console.table(data, headers)

# Validation
wiz.validate.email("user@example.com")
wiz.validate.url("https://example.com")
wiz.validate.phone("+1234567890")

# Quick Utilities
wiz.quick.token(length=32)
wiz.quick.uuid()
wiz.quick.hash(data)
wiz.quick.timestamp()
```

### Configuration File Support

Create `pywizardry_config.yaml` in your project root:

```yaml
logging:
  level: INFO
  file: logs/pywizardry.log
  format: json
  rotate: daily

cache:
  enabled: true
  ttl: 300
  max_size: 1000
  backend: memory  # memory, redis, memcached

security:
  encryption_key: ${ENCRYPTION_KEY}
  jwt_secret: ${JWT_SECRET}
  password_min_length: 8
  require_special_chars: true

network:
  timeout: 30
  retries: 3
  backoff_factor: 2
  user_agent: PyWizardry/1.0.2
  proxy: ${HTTP_PROXY}

async:
  max_workers: 10
  queue_size: 1000
  rate_limit: 100  # per minute

database:
  pool_size: 10
  max_overflow: 20
  echo: false
```

Load configuration:

```python
wiz = Wizard.from_config("pywizardry_config.yaml")
```

---

## 📊 Performance & Statistics

<div align="center">

| Metric | Value |
|--------|-------|
| **Total Functions** | 150+ |
| **Lines of Code** | 15,000+ |
| **Test Coverage** | 95%+ |
| **Required Dependencies** | 0 |
| **Optional Dependencies** | 8 |
| **Supported Python** | 3.7+ |
| **Performance** | Optimized & Benchmarked |
| **Documentation** | 100% Covered |

</div>

### Benchmark Results

```python
# Run benchmarks
results = wiz.benchmark.run_all()

# Sample results (operations per second)
# String operations: ~500,000 ops/sec
# File operations: ~10,000 ops/sec
# Encryption: ~5,000 ops/sec
# Network requests: ~1,000 ops/sec
# Data processing: ~100,000 ops/sec
```

### Dependencies

**Core** (required): `None` ✨

**Optional** (for extended features):
- `cryptography` - Advanced encryption (AES/RSA)
- `pandas` - Data science operations
- `numpy` - Numerical computing
- `matplotlib` - Chart generation
- `BeautifulSoup4` - Web scraping
- `psycopg2` - PostgreSQL support
- `colorama` - Cross-platform colored output
- `requests` - Enhanced HTTP client



## 🤝 Contributing

We welcome contributions! PyWizardry is open-source and community-driven.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass (`pytest`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Saifullah10141/pywizardry.git
cd pywizardry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[full,dev]"

# Run tests
pytest

# Run linting
flake8 pywizardry/
black pywizardry/

# Build documentation
cd docs && make html
```

### Contribution Guidelines

- Write clear, descriptive commit messages
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Follow PEP 8 style guidelines
- Maintain backward compatibility

### Resources

- 📖 [Contributing Guide](https://pywizardry.vercel.app/contributing)
- 🐛 [Issue Tracker](https://github.com/Saifullah10141/pywizardry)
- 💬 [Discussions](https://github.com/Saifullah10141/pywizardry)
- 📜 [Code of Conduct](https://pywizardry.vercel.app/code-of-conduct)

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Saif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 💬 Community & Support

<div align="center">

### Get Help

| Resource | Link |
|----------|------|
| 📖 **Documentation** | [pywizardry.vercel.app/docs](https://pywizardry.vercel.app/docs) |
| 🚀 **Tutorials** | [pywizardry.vercel.app/tutorials](https://pywizardry.vercel.app/tutorials) |
| 🔗 **API Reference** | [pywizardry.vercel.app/api](https://pywizardry.vercel.app/api) |
| 🐛 **Bug Reports** | [GitHub Issues](https://github.com/Saifullah10141/pywizardry) |
| 💡 **Feature Requests** | [GitHub Discussions](https://github.com/Saifullah10141/pywizardry) |
| 💬 **Community Chat** | [Discord Server](https://discord.gg/pywizardry) |
| 📧 **Email Support** | [saifullahanwar00040@gmail.com](mailto:saifullahanwar00040@gmail.com) |

### Stay Connected

[![GitHub stars](https://img.shields.io/github/stars/saif/pywizardry?style=social)](https://github.com/Saifullah10141/PyWizardry/stargazers)

</div>

### Frequently Asked Questions

**Q: Is PyWizardry production-ready?**  
A: Yes! PyWizardry has 95%+ test coverage and is actively maintained.

**Q: How do I report a security vulnerability?**  
A: Please email security concerns to [saifullahanwar00040@gmail.com](mailto:saifullahanwar00040@gmail.com).

**Q: Can I use PyWizardry in commercial projects?**  
A: Absolutely! PyWizardry is MIT licensed and free for commercial use.

**Q: How can I contribute?**  
A: See our [Contributing Guidelines](https://pywizardry.vercel.app/contributing) to get started.

---

## 🙏 Acknowledgments

Special thanks to:
- All [contributors](https://github.com/Saifullah10141/pywizardry/graphs/contributors) who have helped improve PyWizardry
- The Python community for inspiration and feedback
- Open-source projects that make PyWizardry possible

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=saif/pywizardry&type=Date)](https://star-history.com/#saif/pywizardry&Date)

---

## 🔗 Links

- 🏠 **Homepage**: [https://pywizardry.vercel.app](https://pywizardry.vercel.app)
- 📦 **PyPI**: [https://pypi.org/project/PyWizardry](https://pypi.org/project/PyWizardry)
- 💻 **GitHub**: [https://github.com/Saifullah10141/pywizardry](https://github.com/Saifullah10141/pywizardry)
- 📖 **Documentation**: [https://pywizardry.vercel.app/docs](https://pywizardry.vercel.app/docs)
- 📧 **Email**: [saifullahanwar00040@gmail.com](mailto:saifullahanwar00040@gmail.com)

---

<div align="center">

**Built with ❤️ and ✨ magic by [Saif](https://github.com/Saifullah10141)**

*Making Python development magical, one utility at a time*

⭐ **Star us on GitHub** — it motivates us to keep improving!

[⬆ Back to top](#-pywizardry)

</div>
