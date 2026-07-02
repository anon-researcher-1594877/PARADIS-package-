"""pytest configuration – suppress matplotlib display and compiler errors during tests."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot imports

# torch.compile requires a C++ compiler (cl.exe / gcc) to generate optimised code.
# On machines without one, fall back silently to eager (un-compiled) execution so
# tests still exercise the correct logic.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
