import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

# =================================================================================================

RAW_PATH = "src/data/raw/"
PROCESSED_PATH = "src/data/processed/"

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES = [
    "RENDA_MEDIA",
    "RENDA_MEDIANA",
    "TAXA_ALFABETIZACAO",
    "TAXA_URBANIZACAO",
    # "ESCOLA_POR_10k_HABITANTES",
    "ALUNOS_POR_DOCENTE",
    "ESCOLAS_COM_INTERNET"
]

TARGET = "IDEB"

IDEB_CLASSES = [
    "MUITO BAIXO",
    "BAIXO",
    "MÉDIO",
    "ALTO"
]

# =================================================================================================

def NormalizeCodCity(column):
    return (
        column.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(7)
    )

def PrepareIDEB():
    ideb = pd.read_csv(
        RAW_PATH + "ideb_2.csv",
        sep=";",
        encoding="utf-8",
        dtype={"CO_MUNICIPIO": "string"},
        na_values=["-"]
    )

    ideb2023 = ideb.loc[
        ideb["REDE"] == "Pública",
        [
            "SG_UF",
            "CO_MUNICIPIO",
            "NO_MUNICIPIO",
            "VL_OBSERVADO_2023"
        ]
    ].copy()

    ideb2023 = ideb2023.rename(columns={
        "SG_UF": "UF",
        "CO_MUNICIPIO": "COD",
        "NO_MUNICIPIO": "NAME",
        "VL_OBSERVADO_2023": "IDEB"
    })

    ideb2023["COD"] = NormalizeCodCity(ideb2023["COD"])

    ideb2023["IDEB"] = pd.to_numeric(
        ideb2023["IDEB"],
        errors="coerce"
    )

    ideb2023 = ideb2023.dropna(subset=["IDEB"])

    return ideb2023

def PrepareIncome():
    incomeRaw = pd.read_csv(
        RAW_PATH + "tabela10295(1).csv",
        sep=";",
        encoding="utf-8",
        dtype={"Cód.": "string"}
    )

    income = incomeRaw[
        [
            "Cód.",
            "Valor do rendimento nominal médio mensal domiciliar per capita dos moradores em domicílios particulares permanentes ocupados, exclusive os cuja condição no domicílio era pensionista, empregado(a) doméstico(a) ou parente do(a) empregado(a) doméstico(a) (Reais)",
            "Valor do rendimento nominal mediano mensal domiciliar per capita dos moradores em domicílios particulares permanentes ocupados, exclusive os cuja condição no domicílio era pensionista, empregado(a) doméstico(a) ou parente do(a) empregado(a) doméstico(a) (Reais)"
        ]
    ].copy()

    income = income.rename(columns={
        "Cód.": "COD",
        "Valor do rendimento nominal médio mensal domiciliar per capita dos moradores em domicílios particulares permanentes ocupados, exclusive os cuja condição no domicílio era pensionista, empregado(a) doméstico(a) ou parente do(a) empregado(a) doméstico(a) (Reais)": "RENDA_MEDIA",
        "Valor do rendimento nominal mediano mensal domiciliar per capita dos moradores em domicílios particulares permanentes ocupados, exclusive os cuja condição no domicílio era pensionista, empregado(a) doméstico(a) ou parente do(a) empregado(a) doméstico(a) (Reais)": "RENDA_MEDIANA"
    })

    income["COD"] = NormalizeCodCity(income["COD"])

    for column in ["RENDA_MEDIA", "RENDA_MEDIANA"]:
        income[column] = pd.to_numeric(
            income[column]
            .astype("string")
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        )

    return income

def PrepareLiteracy():
    literacyRaw = pd.read_csv(
        RAW_PATH + "tabela9543.csv",
        sep=";",
        encoding="utf-8",
        dtype={"Cód.": "string"}
    )

    literacy = literacyRaw[
        ["Cód.", "Total"]
    ].copy()

    literacy = literacy.rename(columns={
        "Cód.": "COD",
        "Total": "TAXA_ALFABETIZACAO"
    })

    literacy["COD"] = NormalizeCodCity(literacy["COD"])

    literacy["TAXA_ALFABETIZACAO"] = (
        literacy["TAXA_ALFABETIZACAO"]
        .astype("string")
        .str.replace(",", ".", regex=False)
    )

    literacy["TAXA_ALFABETIZACAO"] = pd.to_numeric(
        literacy["TAXA_ALFABETIZACAO"],
        errors="coerce"
    )

    return literacy

def PrepareSchools():
    schoolsRaw = pd.read_csv(
        RAW_PATH + "microdados_ed_basica_2022.csv",
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    schools = schoolsRaw.loc[
        (schoolsRaw["TP_SITUACAO_FUNCIONAMENTO"] == 1) &
        (schoolsRaw["TP_DEPENDENCIA"].isin([1, 2, 3])) &
        (schoolsRaw["IN_FUND_AI"] == 1),
        [
            "CO_MUNICIPIO",
            "CO_ENTIDADE",
            "IN_INTERNET",
            "QT_MAT_FUND_AI",
            "QT_DOC_FUND_AI"
        ]
    ].copy()

    schools = schools.rename(columns={
        "CO_MUNICIPIO": "COD",
        "CO_ENTIDADE": "COD_ESCOLA",
        "IN_INTERNET": "ESCOLAS_COM_INTERNET",
        "QT_MAT_FUND_AI": "MATRICULAS",
        "QT_DOC_FUND_AI": "DOCENTES"
    })

    schools["COD"] = NormalizeCodCity(schools["COD"])

    cities = schools.groupby("COD").agg(
        COD_ESCOLA=("COD_ESCOLA", "nunique"),
        ESCOLAS_COM_INTERNET=("ESCOLAS_COM_INTERNET", "mean"),
        MATRICULAS=("MATRICULAS", "sum"),
        DOCENTES=("DOCENTES", "sum")
    ).reset_index()

    cities["ALUNOS_POR_DOCENTE"] = (
        cities["MATRICULAS"] /
        cities["DOCENTES"].replace(0, np.nan)
    )

    cities["ESCOLAS_COM_INTERNET"] *= 100

    return cities

def PrepareUrbanization():
    urbanizationRaw = pd.read_csv(
        RAW_PATH + "tabela9923.csv",
        sep=";",
        encoding="utf-8",
        dtype={"Cód.": "string"}
    )

    urbanization = urbanizationRaw[
        ["Cód.", "UrbanaPorcentagem"]
    ].copy()

    urbanization = urbanization.rename(columns={
        "Cód.": "COD",
        "UrbanaPorcentagem": "TAXA_URBANIZACAO"
    })

    urbanization["COD"] = NormalizeCodCity(urbanization["COD"])

    urbanization["TAXA_URBANIZACAO"] = (
        urbanization["TAXA_URBANIZACAO"]
        .astype("string")
        .str.replace(",", ".", regex=False)
    )

    urbanization["TAXA_URBANIZACAO"] = pd.to_numeric(
        urbanization["TAXA_URBANIZACAO"],
        errors="coerce"
    )

    return urbanization

# =================================================================================================

def MergeDatasets(
        ideb2023, 
        cities, 
        income,
        literacy,
        urbanization
    ):

    dataset = ideb2023.merge(
        cities,
        on="COD",
        how="left",
        validate="one_to_one"
    )

    dataset = dataset.merge(
        income,
        on="COD",
        how="left",
        validate="one_to_one"
    )

    dataset = dataset.merge(
        literacy,
        on="COD",
        how="left",
        validate="one_to_one"
    )

    dataset = dataset.merge(
        urbanization,
        on="COD",
        how="left",
        validate="one_to_one"
    )

    return dataset

# =================================================================================================

def AnalyzeDataset(dataset):
    print("\nDataset: ")

    print("Dimensões: ", dataset.shape)

    print("\nValores Ausentes: ")
    print(dataset.isna().sum())

    print("\nTipos: ")
    print(dataset.dtypes)

    print("\nDuplicados: ")
    print(dataset["COD"].duplicated().sum())

    print("\nAmostra: ")
    print(dataset.head(10))

def SaveDataset(dataset):
    dataset.to_csv(
        PROCESSED_PATH + "FinalDataset.csv",
        sep=";",
        index=False,
        encoding="utf-8"
    )

    print("\nDataset salvo com sucesso")

# =================================================================================================

def PrepareML(dataset):
    X = dataset[FEATURES].copy()
    Y = dataset[TARGET].copy()

    xTrain, xTest, yTrain, yTest = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return xTrain, xTest, yTrain, yTest

# =================================================================================================

def EvaluateRegression(yReal, yPred, name):
    mse = mean_squared_error(yReal, yPred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yReal, yPred)
    r2 = r2_score(yReal, yPred)

    print(f"\n{name}: ")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

def RunLinearRegression(dataset):
    xTrain, xTest, yTrain, yTest = PrepareML(dataset)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    model.fit(xTrain, yTrain)

    yTrainPred = model.predict(xTrain)
    yTestPred = model.predict(xTest)

    EvaluateRegression(
        yTrain,
        yTrainPred,
        "Linear Regression - Train"
    )
    EvaluateRegression(
        yTest,
        yTestPred,
        "Linear Regression - Test"
    )

    linearRegression = model.named_steps["regression"]

    print("\nWeights: ")
    for feature, coef in zip(FEATURES, linearRegression.coef_):
        print(f"{feature}: {coef:.4f}")

    print(f"\nBias: {linearRegression.intercept_:.4f}")

    print("\nPrevisões: ")
    predictions = pd.DataFrame({
        "Real": yTest.values,
        "Previsto": yTestPred
    })

    print(predictions.head(10))

    return model

# =================================================================================================

def AnalyzeCorrelation(dataset):
    columns = FEATURES + [TARGET]

    pearson = dataset[columns].corr(method="pearson")
    spearman = dataset[columns].corr(method="spearman")

    print("\n========== PEARSON ==========")
    print(pearson[TARGET].sort_values(ascending=False))

    print("\n========== SPEARMAN ==========")
    print(spearman[TARGET].sort_values(ascending=False))

# =================================================================================================

def main():
    ideb2023 = PrepareIDEB()
    cities = PrepareSchools()
    income = PrepareIncome()
    literacy = PrepareLiteracy()
    urbanization = PrepareUrbanization()

    dataset = MergeDatasets(
        ideb2023,
        cities,
        income,
        literacy,
        urbanization
    )

    AnalyzeDataset(dataset)
    SaveDataset(dataset)

    linealModel = RunLinearRegression(dataset)

    AnalyzeCorrelation(dataset)

if __name__ == "__main__":
    main()