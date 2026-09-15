import json,sys
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if job['ampacity']:
    from Benchmarks.full_ampacity import train_ampacity
    train_ampacity(job['case'],job['configuration'],job['reference'],job['output'],job['limit'])
else:
    from Benchmarks.full_domain import FullDomain
    from Benchmarks.full_pinn import train
    train(FullDomain(job['case'],job['mode']),job['configuration'],job['output'])
