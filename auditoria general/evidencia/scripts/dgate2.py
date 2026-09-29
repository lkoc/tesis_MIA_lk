import json, os
os.chdir(r'c:\usr\ths_mia_fiis\tesis_MIA_lk\Benchmarks\explicit_study')
d = json.load(open('D/summary.json', encoding='utf-8'))
print('case | seed | acc | violation | RMSE K | Tmax err K | NRMSE % | balance % | seconds')
for r in d:
    print(f"{r['case']:22s} {r['seed']:3d} {str(r['accepted']):5s} {r['violation']:7.2f} {r['rmse_K']:6.3f} {r['Tmax_error_K']:6.3f} {r['nrmse_pct']:6.2f} {r['balance_pct']:6.3f} {r['seconds']:7.0f}")
import glob, collections
j = json.load(open('D/production/xlpe_discrete_layers/seed71/pinn_seed71.json', encoding='utf-8'))
def walk(o, p=''):
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in k.lower() for s in ('viol', 'reject', 'fail', 'gate', 'interface_T', 'jump')):
                print('  ', p + '/' + k, '=', json.dumps(v, ensure_ascii=False)[:200])
            if isinstance(v, (dict, list)) and len(p) < 40: walk(v, p + '/' + k)
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]): walk(v, p + f'[{i}]')
walk(j)
