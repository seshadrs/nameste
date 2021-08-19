"""
Run like this: 
streamlit run predictor_demo.py
"""

import streamlit as st
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pycountry
import plotly.express as px
from typing import List


@st.cache
def get_model_and_artifacts():
    return load_model("nameste_gender_clfr.model"), joblib.load(
        "nameste_gender_clfr.artifacts"
    )


@st.cache
def get_country_codes(artifacts):
    return artifacts["country_code_encoder"].categories_[0]


@st.cache
def get_country_names(artifacts):
    countries = [
        pycountry.countries.get(alpha_2=code) for code in get_country_codes(artifacts)
    ]
    return [c.name if c else "" for c in countries]


def predict(name, model, artifacts: dict):
    def get_names_matrix(names: pd.Series):
        # padded, categogical character sequences
        return to_categorical(
            pad_sequences(
                artifacts["name_tokenizer"].texts_to_sequences(names),
                maxlen=artifacts["max_name_length"],
                padding="post",
                truncating="post",
            ),
            num_classes=artifacts["num_name_token_classes"],
        )

    def get_country_codes_matrix(codes: pd.Series):
        return (
            artifacts["country_code_encoder"]
            .transform(codes.to_numpy().reshape(-1, 1))
            .todense()
        )

    name = name.lower().strip()
    return model.predict(
        [
            get_names_matrix(pd.Series([name] * len(get_country_codes(artifacts)))),
            get_country_codes_matrix(pd.Series(get_country_codes(artifacts))),
        ]
    )


def get_gender_buckets(maleness_scores: List[float]):
    def get_bucket(score: float):
        if score < 0.2:
            return "feminine"
        elif score < 0.4:
            return "somewhat-feminine"
        elif score < 0.6:
            return "unisex"
        elif score < 0.8:
            return "somewhat-masculine"
        else:
            return "masculine"

    return [get_bucket(score) for score in maleness_scores]


model, artifacts = get_model_and_artifacts()
st.write("🧑 Enter a name to predict the perceived gender")
name = st.text_input("Type name and hit ↵", "Daenerys")

if name:
    maleness_scores = [x[0] for x in predict(name, model, artifacts)]
    res_df = pd.DataFrame(
        list(
            zip(
                get_country_names(artifacts),
                get_country_codes(artifacts),
                maleness_scores,
                get_gender_buckets(maleness_scores),
            )
        ),
        columns=["Country", "Country-Code", "Pred-Maleness", "Pred-Gender"],
    )
    res_df = res_df.sort_values("Pred-Maleness")

    fig = px.choropleth(
        res_df,
        title="Predicted perceived-gender",
        locations="Country",
        locationmode="country names",
        scope="world",
        color="Pred-Gender",
        color_discrete_map={
            "feminine": "#ff4dff",
            "somewhat-feminine": "#ffccff",
            "unisex": "#99ff99",
            "somewhat-masculine": "#ccffff",
            "masculine": "#3399ff",
        },
        projection="mercator",
        hover_data=["Country", "Pred-Gender", "Pred-Maleness"],
        height=600,
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.write(fig)

    res_df = res_df.drop(["Country-Code"], axis=1)
    st.write(res_df)
