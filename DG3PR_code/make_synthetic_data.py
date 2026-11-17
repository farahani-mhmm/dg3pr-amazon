"""
Genera un dataset sintético con el mismo esquema que los datos privados de DG3PR.

Escribe en <out_dir>/ los tres CSV que espera raw_data_process.py:
  - ADS_AI_MRT_SALES_SHIPPING.csv  (ventas: cliente x producto x día)
  - DIM_BREEDS_FILECLASS.csv       (producto -> clase, por almacén)
  - DIM_CUST.csv                   (cliente -> inner_company_ind)

El generador cuida las condiciones implícitas del código:
  - todos los días del rango hay ventas en las tres regiones (east / middle / west), porque
    cada grafo dinámico diario necesita aristas de todos los tipos;
  - cada (clase, almacén) tiene varias ventas con precios distintos por día (jenkspy);
  - cada cliente solo compra de 2 clases, así nunca compra todos los productos
    (el muestreo negativo entraría en un bucle infinito);
  - el precio guía cambia pocas veces (como un precio de lista): la arista producto -> precio exacto
    del día de evaluación tiene que existir en el grafo de entrenamiento o el producto se descarta;
  - cada día se vende todo el catálogo en cada región, así hay >= 30 candidatos por cliente
    (utils/eval.py evalúa hasta top-30 y falla con menos candidatos).

Hay una señal aprendible: cada cliente tiene productos favoritos y una sensibilidad al precio.
Además se agregan unas pocas filas "sucias" (nulos, negativos, duplicados) para ejercitar la limpieza.

Uso:
  python make_synthetic_data.py                       # defaults: 2020-12-24 .. 2021-01-17
  python make_synthetic_data.py --customers 80 --seed 1
"""
import argparse
import os

import numpy as np
import pandas as pd

REGIONS = {
    'east': ['上海', '江苏', '浙江', '山东'],
    'middle': ['河南', '湖北', '湖南', '广东'],
    'west': ['四川', '重庆', '陕西', '云南'],
}
# clase -> (precio guía base por kg, peso promedio en kg)
CLASSES = {
    'lechon': (38.0, 7.0),
    'engorde': (16.0, 110.0),
    'reproductor': (25.0, 140.0),
    'descarte': (12.0, 180.0),
}
GRADES = ['A', 'B', 'C']


def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic data for DG3PR')
    parser.add_argument('--out_dir', type=str, default='raw_data')
    parser.add_argument('--start', type=str, default='2020-12-24', help='primer día con ventas')
    parser.add_argument('--end', type=str, default='2021-01-17', help='último día con ventas')
    parser.add_argument('--customers', type=int, default=60, help='clientes por región')
    parser.add_argument('--breeds_per_class', type=int, default=10)
    parser.add_argument('--price_change_prob', type=float, default=0.05,
                        help='prob. diaria de que cambie el precio guía de un producto')
    parser.add_argument('--buy_prob', type=float, default=0.75, help='prob. de que un cliente compre en un día')
    parser.add_argument('--dirty_frac', type=float, default=0.01, help='fracción de filas sucias extra')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    days = pd.date_range(args.start, args.end)

    # productos: comunes a todas las regiones, cada uno con un precio base
    breeds = []
    for class_index, (class_name, (base_price, base_weight)) in enumerate(CLASSES.items()):
        for i in range(args.breeds_per_class):
            breeds.append({
                'forsale_breeds_dk': 'BR{}{:02d}'.format(class_index, i),
                'forsale_breeds_nm': '{}_{:02d}'.format(class_name, i),
                'breeds_class_nm': class_name,
                'base_price': base_price * rng.uniform(0.75, 1.25),
                'base_weight': base_weight,
            })
    breeds = pd.DataFrame(breeds)
    class_names = list(CLASSES)

    # precio guía diario por producto: constante por tramos, con saltos ocasionales de +-1..5%
    jumps = rng.normal(0, 0.03, size=(len(days), len(breeds)))
    jumps *= rng.random(size=jumps.shape) < args.price_change_prob
    jumps[0] = 0
    daily_price = np.round(breeds['base_price'].values * np.cumprod(1 + jumps, axis=0), 2)

    rows, customers = [], []

    def sale(day_index, org, cust, b):
        breed = breeds.loc[b]
        guide_price = daily_price[day_index, b]
        rows.append({
            'shipping_dt': days[day_index].strftime('%Y-%m-%d'),
            'org_inv_dk': org,
            'forsale_breeds_nm': breed['forsale_breeds_nm'],
            'cust_dk': cust['cust_dk'],
            'province_nm': cust['province'],
            'forsale_breeds_dk': breed['forsale_breeds_dk'],
            'fg_grade_nm': GRADES[rng.integers(len(GRADES))],
            'sale_qty': int(max(1, rng.poisson(cust['scale']))),
            'sale_price': round(guide_price * rng.uniform(0.95, 1.05), 2),
            'avg_wt': round(breed['base_weight'] * rng.uniform(0.85, 1.15), 2),
            'guide_price': guide_price,
        })

    for region, provinces in REGIONS.items():
        org = 'ORG_{}'.format(region.upper())
        region_customers = []
        for c in range(args.customers):
            fav_classes = rng.choice(class_names, size=2, replace=False)
            candidates = breeds[breeds['breeds_class_nm'].isin(fav_classes)].index.values
            region_customers.append({
                'cust_dk': 'C{}{:03d}'.format(region[0].upper(), c),
                'province': provinces[rng.integers(len(provinces))],
                'candidates': candidates,
                'affinity': rng.gamma(0.6, 1.0, size=len(candidates)),  # pocos productos favoritos
                'price_sensitivity': rng.uniform(0, 6),
                'scale': rng.lognormal(3.5, 0.6),
            })
        customers.extend(cust['cust_dk'] for cust in region_customers)

        for d in range(len(days)):
            sold = set()
            for cust in region_customers:
                if rng.random() > args.buy_prob:
                    continue
                candidates = cust['candidates']
                rel_price = daily_price[d, candidates] / breeds.loc[candidates, 'base_price'].values
                weights = cust['affinity'] * np.exp(-cust['price_sensitivity'] * (rel_price - 1))
                n_items = min(rng.integers(1, 4), len(candidates))
                for b in rng.choice(candidates, size=n_items, replace=False, p=weights / weights.sum()):
                    sale(d, org, cust, b)
                    sold.add(b)
            # el resto del catálogo lo compra algún cliente interesado en esa clase
            for b in set(breeds.index) - sold:
                buyers = [cust for cust in region_customers if b in cust['candidates']]
                if buyers:
                    sale(d, org, buyers[rng.integers(len(buyers))], b)

    shipping = pd.DataFrame(rows)

    # filas sucias: nulos, negativos y duplicados (las elimina raw_data_process.py)
    n_dirty = int(len(shipping) * args.dirty_frac)
    nulls = shipping.sample(n_dirty, random_state=args.seed).copy()
    nulls['guide_price'] = np.nan
    negatives = shipping.sample(n_dirty, random_state=args.seed + 1).copy()
    negatives['sale_qty'] = -negatives['sale_qty']
    duplicates = shipping.sample(n_dirty, random_state=args.seed + 2)
    shipping = pd.concat([shipping, nulls, negatives, duplicates]).sample(frac=1, random_state=args.seed)

    breed_class = pd.DataFrame([
        {'org_inv_dk': 'ORG_{}'.format(region.upper()), 'forsale_breeds_dk': b.forsale_breeds_dk,
         'breeds_class_nm': b.breeds_class_nm}
        for region in REGIONS for b in breeds.itertuples()
    ])
    cust = pd.DataFrame({
        'cust_dk': customers,
        'inner_company_ind': (rng.random(len(customers)) < 0.1).astype(int),
    })

    os.makedirs(args.out_dir, exist_ok=True)
    shipping.to_csv(os.path.join(args.out_dir, 'ADS_AI_MRT_SALES_SHIPPING.csv'), index=False)
    breed_class.to_csv(os.path.join(args.out_dir, 'DIM_BREEDS_FILECLASS.csv'), index=False)
    cust.to_csv(os.path.join(args.out_dir, 'DIM_CUST.csv'), index=False)

    print('Dataset sintético en {}/'.format(args.out_dir))
    print('  días: {} .. {} ({})'.format(args.start, args.end, len(days)))
    print('  ventas: {} (+{} filas sucias)'.format(len(rows), len(shipping) - len(rows)))
    print('  clientes: {} ({} por región) | productos: {} | clases: {}'.format(
        len(customers), args.customers, len(breeds), len(CLASSES)))


if __name__ == '__main__':
    main()
