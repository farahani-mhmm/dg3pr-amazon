"""
Convierte Amazon "Grocery and Gourmet Food" (5-core reviews + metadata) al esquema que espera
raw_data_process.py de DG3PR. No hace falta modificar train_dg3pr.py.

Entradas
  --reviews  reviews_Grocery_and_Gourmet_Food_5.json (o .json.gz)   [reviewerID, asin, unixReviewTime]
  --meta     meta_Grocery_and_Gourmet_Food.json.gz                   [asin, price, categories]
             (descargar de https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
              meta_Grocery_and_Gourmet_Food.json.gz ; el archivo de reviews NO trae precio ni categoria)

Salidas (en --out_dir, por defecto raw_data/)
  ADS_AI_MRT_SALES_SHIPPING.csv, DIM_BREEDS_FILECLASS.csv, DIM_CUST.csv

Mapeo
  reviewerID -> cust_dk          asin -> forsale_breeds_dk / forsale_breeds_nm
  price      -> guide_price      categories[0][category_level] -> breeds_class_nm
  review     -> una "venta" (sale_qty = 1)

Adaptaciones (Amazon no tiene dias ni provincias chinas):
  - TIEMPO: las reviews son escasas por dia, asi que se agrupan en ventanas y a cada ventana se le
    asigna un "dia" sintetico consecutivo (2020-12-24 .. 2021-01-17 por defecto = 25 ventanas, las
    mismas fechas que usa train_dg3pr.py). Por defecto (--bucket_days 0) los datos se reparten en
    --num_buckets ventanas con igual numero de reviews (sin ventanas vacias); con --bucket_days 30 se usan solo las ultimas 25 ventanas
    mensuales. Asi "train_date_days=7" = 7 ventanas.
  - PRECIO: en la metadata de Amazon solo una fraccion de los productos tiene precio; los demas se
    descartan (el script imprime cuantos quedan).
  - REGIONES: no hay geografia; cada cliente se asigna de forma determinista (hash) a una de
    3 regiones, usando provincias que ya estan en las listas east/middle/west de train_dg3pr.py.
  - PRECIO: es estatico por producto, lo que encaja con el "precio guia" que cambia poco.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os

import pandas as pd

# una provincia representativa por region (las listas de train_dg3pr.py)
REGION_PROVINCES = ['上海', '河南', '四川']  # east, middle, west


def open_any(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8')


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # metadata 2014 viene como literales de Python (comillas simples)
        return ast.literal_eval(line)


def load_reviews(path):
    rows = []
    with open_any(path) as f:
        for line in f:
            d = parse_line(line)
            if d:
                rows.append((d['reviewerID'], d['asin'], d['unixReviewTime']))
    return pd.DataFrame(rows, columns=['cust_dk', 'asin', 'ts'])


def load_meta(path, category_level):
    rows = []
    with open_any(path) as f:
        for line in f:
            d = parse_line(line)
            if not d or 'asin' not in d:
                continue
            price = d.get('price')
            cats = d.get('categories') or [[]]
            path_cats = cats[0] if cats and cats[0] else []
            cls = path_cats[category_level] if len(path_cats) > category_level else None
            rows.append((d['asin'], price, cls))
    return pd.DataFrame(rows, columns=['asin', 'guide_price', 'breeds_class_nm'])


def stable_region(cust_id):
    h = int(hashlib.md5(cust_id.encode()).hexdigest(), 16)
    return REGION_PROVINCES[h % 3]


def parse_args():
    p = argparse.ArgumentParser(description='Amazon Grocery -> DG3PR raw CSVs')
    p.add_argument('--reviews', required=True)
    p.add_argument('--meta', required=True)
    p.add_argument('--out_dir', default='raw_data')
    p.add_argument('--bucket_days', type=int, default=0,
                   help='0 = automatico: --num_buckets ventanas con igual numero de reviews (recomendado, '
                        'evita ventanas vacias); 30 = mensual, usando solo las ultimas ventanas')
    p.add_argument('--num_buckets', type=int, default=25, help='numero de dias sinteticos (25 = 2020-12-24..2021-01-17)')
    p.add_argument('--last_day', default='2021-01-17', help='ultimo dia sintetico')
    p.add_argument('--category_level', type=int, default=1,
                   help='nivel de categories[0] usado como clase (0=Grocery & Gourmet Food, 1=subcategoria)')
    p.add_argument('--min_class_items', type=int, default=20, help='descarta clases con menos productos')
    p.add_argument('--min_price', type=float, default=0.01)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rev = load_reviews(a.reviews)
    meta = load_meta(a.meta, a.category_level)
    print('reviews:', len(rev), ' meta:', len(meta))

    meta['guide_price'] = pd.to_numeric(meta['guide_price'], errors='coerce')
    meta = meta.dropna(subset=['guide_price', 'breeds_class_nm'])
    meta = meta[meta['guide_price'] >= a.min_price].drop_duplicates('asin')
    cls_count = meta['breeds_class_nm'].value_counts()
    meta = meta[meta['breeds_class_nm'].isin(cls_count[cls_count >= a.min_class_items].index)]
    print('productos con precio y clase:', len(meta), ' clases:', meta['breeds_class_nm'].nunique())

    df = rev.merge(meta, on='asin', how='inner')
    print('reviews con precio y clase: %d de %d (%.1f%%), productos: %d de %d'
          % (len(df), len(rev), 100.0 * len(df) / len(rev), df['asin'].nunique(), rev['asin'].nunique()))

    # ventanas de tiempo -> dias sinteticos consecutivos
    if a.bucket_days > 0:  # ventanas de ancho fijo, solo las ultimas num_buckets
        span = a.bucket_days * 86400
        idx_from_end = (df['ts'].max() - df['ts']) // span
        df = df[idx_from_end < a.num_buckets].copy()
        idx_from_end = idx_from_end[df.index]
    else:  # automatico: num_buckets ventanas con el MISMO numero de reviews (orden temporal conservado)
        qbin = pd.qcut(df['ts'].rank(method='first'), a.num_buckets, labels=False)
        idx_from_end = (a.num_buckets - 1) - qbin
        df = df.copy()
    last = pd.to_datetime(a.last_day)
    df['shipping_dt'] = (last - pd.to_timedelta(idx_from_end, unit='D')).dt.strftime('%Y-%m-%d')

    # una compra por (cliente, producto, dia)
    df = df.drop_duplicates(['cust_dk', 'asin', 'shipping_dt'])

    out = pd.DataFrame({
        'shipping_dt': df['shipping_dt'],
        'org_inv_dk': 'AMZ',
        'forsale_breeds_nm': df['asin'],
        'cust_dk': df['cust_dk'],
        'province_nm': df['cust_dk'].map(stable_region),
        'forsale_breeds_dk': df['asin'],
        'fg_grade_nm': 'A',
        'sale_qty': 1.0,
        'sale_price': df['guide_price'],
        'avg_wt': 1.0,
        'guide_price': df['guide_price'],
    })
    out.to_csv(os.path.join(a.out_dir, 'ADS_AI_MRT_SALES_SHIPPING.csv'), index=False)

    items = df[['asin', 'breeds_class_nm']].drop_duplicates('asin')
    pd.DataFrame({
        'org_inv_dk': 'AMZ',
        'forsale_breeds_dk': items['asin'],
        'breeds_class_nm': items['breeds_class_nm'],
    }).to_csv(os.path.join(a.out_dir, 'DIM_BREEDS_FILECLASS.csv'), index=False)

    pd.DataFrame({'cust_dk': df['cust_dk'].unique(), 'inner_company_ind': 0}) \
        .to_csv(os.path.join(a.out_dir, 'DIM_CUST.csv'), index=False)

    # diagnostico de las condiciones que exige el codigo
    print('\nfilas:', len(out), ' clientes:', out['cust_dk'].nunique(), ' productos:', out['forsale_breeds_dk'].nunique())
    cov = out.groupby(['province_nm'])['shipping_dt'].nunique()
    print('dias con ventas por region (necesitas', a.num_buckets, 'en las 3):')
    print(cov.to_string())
    per_day = out.groupby(['province_nm', 'shipping_dt'])['forsale_breeds_dk'].nunique()
    print('minimo de productos distintos por region/dia (debe ser >= 30):', per_day.min())
    print('\nSiguiente paso: python raw_data_process.py && python train_dg3pr.py')


if __name__ == '__main__':
    main()