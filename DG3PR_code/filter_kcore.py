"""
Filtro k-core para los CSV generados por prepare_amazon_data.py.
Elimina (de forma iterativa) clientes y productos con menos de k interacciones.
Hace copia de seguridad del CSV original (*.full.csv) y reescribe los tres CSV de raw_data/.

Uso:
  python filter_kcore.py --k 2 --single_region
"""
import argparse
import os
import shutil

import pandas as pd

SHIP = 'ADS_AI_MRT_SALES_SHIPPING.csv'
CLS = 'DIM_BREEDS_FILECLASS.csv'
CUST = 'DIM_CUST.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='raw_data')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--single_region', action='store_true',
                   help='todos los clientes en una sola region (east) para no repartir los pocos datos en 3')
    a = p.parse_args()

    ship_path = os.path.join(a.dir, SHIP)
    full_path = ship_path.replace('.csv', '.full.csv')
    if not os.path.exists(full_path):
        shutil.copy(ship_path, full_path)  # copia de seguridad, solo la primera vez
    d = pd.read_csv(full_path)  # siempre parte del original completo
    print('original: filas', len(d), 'clientes', d['cust_dk'].nunique(), 'productos', d['forsale_breeds_dk'].nunique())

    while True:
        n = len(d)
        d = d[d.groupby('cust_dk')['cust_dk'].transform('size') >= a.k]
        d = d[d.groupby('forsale_breeds_dk')['cust_dk'].transform('size') >= a.k]
        if len(d) == n:
            break

    if a.single_region:
        d = d.assign(province_nm='上海')

    d.to_csv(ship_path, index=False)
    cls = pd.read_csv(os.path.join(a.dir, CLS))
    cls[cls['forsale_breeds_dk'].isin(d['forsale_breeds_dk'])].to_csv(os.path.join(a.dir, CLS), index=False)
    cust = pd.read_csv(os.path.join(a.dir, CUST))
    cust[cust['cust_dk'].isin(d['cust_dk'])].to_csv(os.path.join(a.dir, CUST), index=False)

    print('k-core %d: filas %d clientes %d productos %d' %
          (a.k, len(d), d['cust_dk'].nunique(), d['forsale_breeds_dk'].nunique()))
    print('dias con ventas por region:')
    print(d.groupby('province_nm')['shipping_dt'].nunique().to_string())
    per_day = d.groupby(['province_nm', 'shipping_dt'])['forsale_breeds_dk'].nunique()
    print('minimo de productos distintos por region/dia (debe ser >= 30):', per_day.min())


if __name__ == '__main__':
    main()