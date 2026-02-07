# Arbiter Framework

## Quick Start

Install (when published):
```bash
pip install arbiter-framework
```

Or use directly:
```python
from arbiter import ArbiterAgent

agent = ArbiterAgent()
result = agent.execute_action("reason", {"prompt": "..."})
```

## Development

Run tests:
```bash
python test_arbiter.py -v
```

Run examples:
```bash
python examples.py
```

## Files

- `arbiter.py` - Core framework (7 theorems, zero dependencies)
- `test_arbiter.py` - 155-test adversarial suite
- `examples.py` - Practical usage examples
- `THEOREMS.md` - Mathematical proofs
- `README.md` - Full documentation
- `setup.py` - Package configuration

## License

MIT License - See LICENSE file
