"""Plain SSL method name constants, kept free of TensorFlow imports.

`main.py` needs this list to build `argparse` choices before any heavy dependency is imported, so
`python main.py --help` works even on a machine that doesn't have TensorFlow installed yet.
"""
METHODS = ("barlow_twins", "byol", "moco", "simclr", "simsiam")
