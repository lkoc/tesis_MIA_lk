# Historical v2 verification snapshots

These JSON files are pre-activation snapshots of analytical and manufactured-solution verification
cases. The active definitions now live recursively under `cases/v2/references/verification/` and are
listed by `CaseDefinition.load_all()`.

The annulus and rectangular MMS cases are general 2D heat-conduction tests, not cable cases. Both the
PINN and FEniCSx verification solvers support them. These snapshots remain here only for historical
comparison and are not part of active catalog discovery.