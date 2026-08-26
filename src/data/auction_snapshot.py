# -*- coding: utf-8 -*-
"""集合竞价快照落库（竞价强度因子的数据积累管道）。"""


def _save_auction_snapshot(db, today: str, fresh) -> None:
    """把 9:25 全市场竞价快照写入 auction_snapshots 表。

    竞价强度因子的数据积累：量比(竞价撮合量/近5日均量)、开盘涨幅、
    换手率。9:25 时刻的最新价即开盘撮合价，涨跌幅即开盘涨幅。
    全市场约5000行，批量 OR REPLACE 幂等。
    """
    code_col = '代码' if '代码' in fresh.columns else 'code'
    name_col = '名称' if '名称' in fresh.columns else 'name'
    price_col = '最新价' if '最新价' in fresh.columns else 'price'
    pct_col = '涨跌幅' if '涨跌幅' in fresh.columns else 'pctChg'
    vr_col = '量比' if '量比' in fresh.columns else None
    turn_col = '换手率' if '换手率' in fresh.columns else None
    amt_col = '成交额' if '成交额' in fresh.columns else None

    rows = []
    for _, r in fresh.iterrows():
        try:
            code = str(r.get(code_col, '') or '')
            price = float(r.get(price_col, 0) or 0)
            if not code or price <= 0:
                continue
            rows.append({
                'code': code,
                'name': str(r.get(name_col, '') or ''),
                'open_price': price,
                'open_pct': float(r.get(pct_col, 0) or 0),
                'volume_ratio': (float(r.get(vr_col, 0) or 0)
                                 if vr_col else None),
                'turnover': (float(r.get(turn_col, 0) or 0)
                             if turn_col else None),
                'amount': (float(r.get(amt_col, 0) or 0)
                           if amt_col else None),
            })
        except (TypeError, ValueError):
            continue
    n = db.save_auction_snapshot(today, rows)
    print(f'    竞价快照: {n} 只')
