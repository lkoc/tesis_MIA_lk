"""Bounded second-derivative workload; this is not time-to-accepted-solution."""
from pathlib import Path
from datetime import datetime, timezone
import json, platform, statistics, subprocess, sys, time
import torch

OUT=Path(__file__).resolve().parent

def model_and_points(device,dtype):
    torch.manual_seed(913)
    model=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.Tanh(),torch.nn.Linear(32,32),
        torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).to(device=device,dtype=dtype)
    points=torch.rand((1536,2),dtype=dtype,device=device).requires_grad_(True)
    return model,points

def step(model,points):
    model.zero_grad(set_to_none=True)
    points.grad=None
    y=model(points)
    grad=torch.autograd.grad(y,points,torch.ones_like(y),create_graph=True)[0]
    lap=0
    for j in range(2):
        lap=lap+torch.autograd.grad(grad[:,j],points,torch.ones_like(grad[:,j]),create_graph=True)[0][:,j]
    loss=((lap+torch.sin(points[:,0])*torch.sin(points[:,1]))**2).mean()
    loss.backward()
    return loss

def main():
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
        platform=platform.platform(),torch=torch.__version__,cuda=torch.cuda.is_available(),
        xpu=torch.xpu.is_available(),limitations='Isolated synthetic derivative/backward microbenchmark; no optimizer, interfaces, acceptance test, campaign throughput or GPU speed claim.',
        cpu_rows=[],xpu_checks=[])
    command="$c=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_ComputerSystem; $g=Get-CimInstance Win32_VideoController; @{cpu=$c.Name;cores=$c.NumberOfCores;logical=$c.NumberOfLogicalProcessors;ram_bytes=$m.TotalPhysicalMemory;gpu=$g.Name;driver=$g.DriverVersion}|ConvertTo-Json"
    result['hardware']=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True))
    if '--xpu-only' in sys.argv:
        result['cpu_rows']=json.loads((OUT/'compute_probe.json').read_text())['cpu_rows']
    for threads in ([] if '--xpu-only' in sys.argv else [1,2,4,8]):
        torch.set_num_threads(threads)
        model,points=model_and_points('cpu',torch.float64)
        for _ in range(10):step(model,points)
        samples=[]
        for _ in range(3):
            start=time.perf_counter()
            for _ in range(25):step(model,points)
            samples.append((time.perf_counter()-start)/25)
        row=dict(threads=threads,seconds_per_step=samples,median_seconds=statistics.median(samples))
        result['cpu_rows'].append(row)
        print(json.dumps(row),flush=True)
    if result['xpu']:
        result['xpu_name']=torch.xpu.get_device_name(0)
        for dtype in [torch.float64,torch.float32]:
            row=dict(dtype=str(dtype))
            try:
                model,points=model_and_points('xpu',dtype)
                loss=step(model,points)
                torch.xpu.synchronize()
                row.update(derivative_backward_supported=True,finite=bool(torch.isfinite(loss).item()))
                for _ in range(10):step(model,points)
                torch.xpu.synchronize()
                samples=[]
                for _ in range(3):
                    torch.xpu.synchronize()
                    start=time.perf_counter()
                    for _ in range(25):step(model,points)
                    torch.xpu.synchronize()
                    samples.append((time.perf_counter()-start)/25)
                row.update(seconds_per_step=samples,median_seconds=statistics.median(samples))
            except Exception as error:
                row.update(derivative_backward_supported=False,error=str(error))
            result['xpu_checks'].append(row)
            print(json.dumps(row),flush=True)
    (OUT/'compute_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
