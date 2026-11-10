# Manufactured sine field on a unit square

- Identifier: `mms_sine`
- Family: manufactured solution (MMS)
- Origin: V3 synthetic verification problem.

The exact field is declared in `reference.exact` and is used only for evaluation. The source and four Dirichlet traces are declared independently in the JSON, allowing PINN and FEM to solve the same problem without hidden boundary rules.
