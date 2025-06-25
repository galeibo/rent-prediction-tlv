def prepare_data(df, st, with_price):
    import pandas as pd
    import numpy as np
    import joblib
    import os
    import re
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if ((st == 'test' and with_price) or st == 'train'):
        df = df.dropna(subset=['price'])
        df = df[(df['price'] > 1000) & (df['price'] < 17000)]

    df = df.dropna(subset=['property_type', 'address', 'floor', 'neighborhood'])
    df = df[~df.duplicated(subset=['address', 'room_num', 'neighborhood', 'floor'], keep='first')]

    df['garden_area'] = df['garden_area'].fillna(0)
    df['days_to_enter'] = df['days_to_enter'].fillna(0).replace(-1, 0)
    df['num_of_payments'] = df['num_of_payments'].fillna(12).replace(0, 12)
    df['handicap'] = df['handicap'].fillna(0)
    df['num_of_images'] = df['num_of_images'].fillna(0)

    if 'description' in df.columns:
        df = df.drop('description', axis=1)

    df['monthly_arnona'] = df['monthly_arnona'].fillna(df.groupby('neighborhood')['monthly_arnona'].transform('mean')).fillna(0)
    df['building_tax'] = df['building_tax'].fillna(df.groupby('neighborhood')['building_tax'].transform('mean')).fillna(0)

    def fix_floor(floor, total_floors):
        if pd.isna(total_floors):
            total_floors = ''
        try:
            if int(floor) > int(total_floors):
                floors = re.findall(r"(\d{1,2})(\d{2})?", str(floor))
                return pd.Series(floors[0] if floors else [np.nan, np.nan])
            else:
                return pd.Series([floor, total_floors])
        except:
            if "קרקע" in str(floor):
                floor = str(floor).replace('קרקע', '0')
            floors = re.findall(r"\d+", str(floor))
            return pd.Series(floors[:2] + [np.nan]*(2 - len(floors)))

    df[['floor', 'total_floors']] = df.apply(lambda row: fix_floor(row['floor'], row['total_floors']), axis=1)
    df = df.dropna(subset=['total_floors'])

    df = df[(df['area'] >= 10) & (df['area'] <= 200)]
    df = df[df['room_num'] > 0]

    def handle_distance_from_center(df):
        is_all_nan = df.groupby('neighborhood')['distance_from_center'].transform(lambda x: x.isna().all())
        df = df.loc[~is_all_nan]
        df['distance_from_center'] = df['distance_from_center'].fillna(df.groupby('neighborhood')['distance_from_center'].transform('mean'))
        return df

    df = handle_distance_from_center(df)

    df = df.astype({
        'floor': int,
        'garden_area': int,
        'days_to_enter': int,
        'num_of_payments': int,
        'monthly_arnona': int,
        'building_tax': int,
        'total_floors': int,
        'num_of_images': int,
        'handicap': int,
        'distance_from_center': float
    })

    df['property_type'] = df['property_type'].replace({
        'דירה להשכרה': 'דירה',
        'דירת גן להשכרה': 'דירת גן',
        'גג/פנטהאוז להשכרה': 'גג/פנטהאוז',
        'גג/ פנטהאוז': 'גג/פנטהאוז',
        'באתר מופיע ערך שלא ברשימה הסגורה': 'דירה',
        'החלפת דירות': 'דירה',
        'Квартира': 'דירה',
        'דופלקס': 'גג/פנטהאוז',
        'דו משפחתי': "פרטי/קוטג'",
        'מרתף/פרטר': 'כללי',
        'סטודיו/לופט': 'כללי',
        'יחידת דיור': 'כללי'
    })

    df = df[~df['property_type'].isin(['מחסן', 'חניה'])]

    def convert_to_meters(row):
        if "דיזינגוף" not in str(row['address']) and row['distance_from_center'] < 20:
            return row['distance_from_center'] * 1000
        return row['distance_from_center']

    df['distance_from_center'] = df.apply(convert_to_meters, axis=1)
    df = df[df['distance_from_center'] < 25000]

    amenities = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap', 'has_bars',
                 'has_safe_room', 'has_balcony', 'is_furnished', 'is_renovated']
    df['num_of_amenities'] = df[amenities].sum(axis=1)
    df['is_central'] = (df['distance_from_center'] < 1500).astype(int)

    encoder_path = 'encoder_property_type.pkl'
    if st == 'train':
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', dtype=int)
        encoded = encoder.fit_transform(df[['property_type']])
        joblib.dump(encoder, encoder_path)
    else:
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            encoded = encoder.transform(df[['property_type']])
        else:
            raise FileNotFoundError(f"{encoder_path} not found. Please run in 'train' mode first.")

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['property_type']), index=df.index)
    df = pd.concat([df.drop(columns=['property_type']), encoded_df], axis=1)

    # neighborhood_encoded לפני מחיקה
    if st == 'train':
        neighborhood_map = df.groupby('neighborhood')['price'].mean()
        neighborhood_map.to_csv('neighborhood_map.csv')
    else:
        neighborhood_map = pd.read_csv('neighborhood_map.csv', index_col=0)['price']
    df['neighborhood_encoded'] = df['neighborhood'].map(neighborhood_map).fillna(0)

    # מחיקת עמודות מיותרות
    drop_cols = ['neighborhood', 'address', 'days_to_enter', 'num_of_images', 'num_of_payments', 'distance_from_center']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if st == 'train':
        df = df[[col for col in df.columns if col != 'price'] + ['price']]

    scaler_path = 'scaler.pkl'
    if with_price:
        features = df.drop('price', axis=1)
        y = df['price']
    else:
        features = df.copy()
        y = None

    binary_cols = [col for col in features.columns if set(features[col].unique()) <= {0, 1} and col != 'garden_area']
    continuous_cols = [col for col in features.columns if col not in binary_cols]

    if st == 'train':
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[continuous_cols])
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            features_scaled = scaler.transform(features[continuous_cols])
        else:
            raise FileNotFoundError(f"{scaler_path} not found. Please run in 'train' mode first.")

    df_scaled = pd.concat([
        pd.DataFrame(features_scaled, columns=continuous_cols).reset_index(drop=True),
        features[binary_cols].reset_index(drop=True)
    ], axis=1)

    if with_price and y is not None:
        df_scaled['price'] = y.reset_index(drop=True)

    return df_scaled
