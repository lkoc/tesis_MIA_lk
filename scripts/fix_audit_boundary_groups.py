from pathlib import Path
p=Path('pinn_cables/pinn/train.py')
s=p.read_text(encoding='utf-8')
s=s.replace('bc_parts: list[torch.Tensor] = []','bc_parts = {name: [] for name in ("dirichlet", "neumann", "robin")}')
s=s.replace('bc_parts.append(', 'bc_parts[bc.bc_type].append(')
s=s.replace('''        if bc_parts:
            all_bc = torch.cat(bc_parts, dim=0)
            losses["bc_dirichlet"] = mse(all_bc)''','''        for kind, parts in bc_parts.items():
            if parts:
                losses[f"bc_{kind}"] = mse(torch.cat(parts, dim=0))''')
s=s.replace('''        if bc_parts:
            losses["bc_dirichlet"] = mse(torch.cat(bc_parts, dim=0))''','''        for kind, parts in bc_parts.items():
            if parts:
                losses[f"bc_{kind}"] = mse(torch.cat(parts, dim=0))''')
a=s.index('        # Interface flux-continuity losses.')
b=s.index('        ifc_flux_losses:',a)
s=s[:a]+'''        # Approximate two-sided traces at each material interface. A globally
        # smooth network still cannot represent an exact derivative jump;
        # Benchmarks uses separate networks for the discontinuous MMS case.
'''+s[b:]
p.write_text(s,encoding='utf-8')
