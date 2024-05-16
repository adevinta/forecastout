import pandas as pd
from typing import List


class FeatureEncoder:

    @staticmethod
    def numerical_encode(
            df: pd.DataFrame,
            encode_feats: List[str],
            encode_value: str) -> pd.DataFrame:

        for encode_feat in encode_feats:
            df_encoded = (
                df.groupby(encode_feat)[encode_value].mean().reset_index()
            )
            df_encoded.rename(
                columns={encode_value: encode_feat+'_num_enc'},
                inplace=True
            )
            df = df.merge(
                df_encoded,
                on=encode_feat, how='left')
        return df
