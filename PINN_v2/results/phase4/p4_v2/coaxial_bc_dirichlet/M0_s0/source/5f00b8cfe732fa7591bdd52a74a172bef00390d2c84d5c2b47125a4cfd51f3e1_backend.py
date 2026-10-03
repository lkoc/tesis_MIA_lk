"""UFL adapter so the same conductivity expressions run in numpy, torch and FEniCSx."""
import ufl


class UFLBackend:
    sin=staticmethod(ufl.sin); cos=staticmethod(ufl.cos); tanh=staticmethod(ufl.tanh)
    log=staticmethod(ufl.ln); sqrt=staticmethod(ufl.sqrt); exp=staticmethod(ufl.exp)
    @staticmethod
    def where(condition,a,b): return ufl.conditional(condition,a,b)