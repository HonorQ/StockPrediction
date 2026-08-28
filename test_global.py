import requests

codes = 'us.DJI,us.SPX,us.IXIC,hk.HSI,hf.CL,hf.GC'
url = 'https://qt.gtimg.cn/q=' + codes
resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)

for line in resp.text.strip().split(';'):
    if '=' in line and 'none_match' not in line:
        key = line.split('=')[0].replace('v_', '').strip()
        seg = line.split('"')
        raw = seg[1] if len(seg) > 1 else ''
        if raw:
            p = raw.split('~')
            if len(p) >= 6:
                print(f'{key}: price={p[3]}, change={p[4]}, change_pct={p[5]}')
            elif len(p) >= 2:
                print(f'{key}: price={p[0]}, change={p[1]}')
