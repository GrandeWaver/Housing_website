import streamlit as st
import numpy as np
import pydeck as pdk
import tarfile
from six.moves import urllib
import pandas as pd
import os
import streamlit.components.v1 as components
import altair as alt
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import sys
from sklearn.impute import SimpleImputer
os.environ['NUMEXPR_MAX_THREADS'] = '12'


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


#with st_stdout("code"):
    #print("Prints as st.code()")

#with st_stdout("info"):
    #print("Prints as st.info()")

#with st_stdout("markdown"):
    #print("Prints as st.markdown()")

#with st_stdout("success"), st_stderr("error"):
    #print("You can print regular success messages")
    #print("And you can redirect errors as well at the same time", file=sys.stderr)


#pobieranie danych
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("zestawy danych", "mieszkania")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")

        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()

housing.rename(columns={"longitude": "lon", "latitude": "lat", "housing_median_age": "Mediana wieku mieszkań",
                        "total_rooms": "Całk. liczba pokoi", "total_bedrooms": "Całk. liczba sypialni", "population": "Populacja",
                        "households": "Rodziny", "median_income": "Mediana dochodów", "median_house_value": "Mediana cen mieszkań",
                        "ocean_proximity": "Odległość do oceanu"}, inplace=True)

st.title("Prognozowanie cen mieszkań w dystryktach stanu Kalifornia z pomocą algorytmu sztucznej inteligencji")
st.subheader("Twoim zadaniem jest stworzenie modelu, który wyręczy masę ekspertów obecnie zajmujących się wyliczanien cen ręcznie. "
             "Jest to bardzo kosztowna i czasochłonna czynność, a przy pomocy naszego modelu możemy ją znacznie uprościć. "
             "Pragram pobierze i rozpakuje dane co pozwoli nam je wyświetlić na stronie internetowej.")

with st_stdout("code"):
    print("Dane pobrane zostały ze strony: ")
    print(HOUSING_URL)

if st.checkbox('Wyświetl surowe dane'):
    st.subheader("Dane dystryktów na których będziemy pracować:")
    st.write(housing)
    st.text("(Wszystkie dane zostały opracowane na podstawie spisu ludności Kalifornii z 1990 roku.)")


st.subheader("Nasz zbiór danych składa się z 20640 przykładów, nie jest on duży jak na standardy Sztucznej Inteligencji, ale nadaje się idealnie dla początkujących analityków.")


#Mapa cen mieszkań
df_mapa = housing[["lon", "lat", "Mediana cen mieszkań"]]


def mapa_cl():
    layer = pdk.Layer(
        'HexagonLayer',
        df_mapa,
        get_position=['lon', 'lat'],
        auto_highlight=True,
        elevation_scale=30,
        pickable=True,
        elevation_range=[0, 10000],
        extruded=True,
        coverage=2)
    view_state = pdk.ViewState(
        longitude=-119.535,
        latitude=37.5323,
        zoom=5.3,
        pitch=20.5,
        bearing=17.36)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, mapbox_key='pk.eyJ1Ijoid2F2ZXItIiwiYSI6ImNraGtjZGtmdDE5dDEycm8xeXZweXRkYWYifQ.KqAMWCqBJOhDjaHPZdDHtA')
    r.to_html("hexagon_layer.html")
    return r.show()


mapa = mapa_cl()
st.header('**Tak prezentują się ceny domów w Kalifornii:**')
HtmlFile = open("hexagon_layer.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height=800)


#Korelacja między medianą dochodów a medianą zarobków
st.header('**Poszukiwanie korelacji**')
housing.rename(columns={"lon": "długość geograficzna", "lat": "szerokość geograficzna"}, inplace=True)
metrics = ["Mediana dochodów", "Mediana cen mieszkań", "Mediana wieku mieszkań", "Populacja", "Rodziny", "długość geograficzna", "szerokość geograficzna"]
cols = st.selectbox('', metrics)


if cols == "Mediana dochodów":
    x = [1.3, 7]
    y = [10000, 500000]
    wskaznik = pd.DataFrame({
      'x': x,
      'y': y })
    rule = alt.Chart(wskaznik).mark_line(color='red').encode(x='x', y='y')
    bar = alt.Chart(housing).mark_circle().encode(x=cols, y='Mediana cen mieszkań')
    st.altair_chart(bar + rule)
    with st_stdout("info"):
        print('Na powyższym wykresie widzimy, że mediana dochodów idzie w góre wraz ze wzrostem mediany cen mieszkań.')
else:
    bar = alt.Chart(housing).mark_circle().encode(x=cols, y='Mediana cen mieszkań')
    st.altair_chart(bar)


#korelacja za pomocą corr_matrix
st.subheader('Możemy również wyliczyć współczynnik korelacji liniowej pomiędzy każdą parą wartości.')
corr_matrix = housing.corr()
korelacje = corr_matrix["Mediana cen mieszkań"].sort_values(ascending=False)
korelacje
st.subheader('Wartości współczynnika korelacji mieszczą się w zakresie pomiędzy -1 a 1. Wartości zbliżone do 1 wskazują silną korelacje dodatnią; na przykład mediana cen mieszkań prawie w 70% przypdaków '
             'rośnie wraz ze wzrostem mediany dochodów. Z kolei wartości zbliżone do -1 mówią nam, że istnieje silna korelacja ujemna; widzimy niewielką korelacje ujemna pomiędzy szerokością geograficzną,'
             'a medianą cen mieszkań (przykładowo ceny nieznacznie spadają, im bardziej kierujemy się na północ). Natomiast wartości bliskie zera oznaczają brak korelacji.')


#Eksperymentowanie z kombinacjami atrybutów
st.header('**Eksperymentowanie z kombinacjami atrybutów**')
st.subheader("Powinniśmy teraz wypróbować różne kombinacje atrybutów. Przykładowo, całkowita liczba pomieszczeń w dystrykcie nie jest zbyt wartościowym atrybutem, jeśli nie znamy liczby przebywających "
             "tam rodzin. W rzeczywistości interesuje nas liczba pokojów przypadających na rodzinę. Również całkowita liczba sypialni sama w sobie nic nam nie mówi: prawdopodobnie należałoby ją porównać "
             "z całkowitą liczbą pomieszczeń. Inna ciekawą kombinacją atrybutów jest określenie zależności pomiędzy populacją a liczbą rodzin.")

st.subheader("Po podzieleniu odpowiednich atrybutów przez siebie, otrzymujemy nowe. Teraz czas przyjrzeć się uzyskanej tablicy korelacji.")
housing["Pokoje na rodzinę"] = housing["Całk. liczba pokoi"]/housing["Rodziny"]
housing["Sypialnie na pokoje"] = housing["Całk. liczba sypialni"]/housing["Całk. liczba pokoi"]
housing["Populacja na rodzinę"]=housing["Populacja"]/housing["Rodziny"]

corr_matrix = housing.corr()
corr_matrix = corr_matrix["Mediana cen mieszkań"].sort_values(ascending=False)
corr_matrix
st.subheader("Nowy atrybut **sypialnie na pokoje** jest znacznie bardziej skorelowany z medianą cen mieszkań niż całkowita liczba pomieszczeń lub sypialni. Najwidoczniej mieszkania o mniejszym współczynniku"
             "liczby sypialni do liczby pomieszczeń okazują się droższe. Liczba pokojów przypadająca na rodzinę również dostarcza nam ciekawe informacje; wraz z powierzchnią domu rośnie jego cena.")

# wypełnienie luki w danych w tabeli całk. liczba pokoi
st.header("**Przygotowywanie danych**")
st.subheader("Zanim zaczniemy szukać najlepszego modelu do prognozowania cen, musimy rozwiązać problem luki w danych.")
with st_stdout("markdown"):
    print("Widzimy to na poniższym wycinku z konsoli:")
st.image("output.png")

#wybierz metodę wypełnienia luki oraz usuwamy kolumne Odległośćdo oceanu
opcje = [" ", "Uzupełnij dane medianą wartości", "Uzupełnij dane średnią",
         "Uzupełnij dane najczęściej występującą wartością"]
cols = st.selectbox("Wybierz jedną z trzech strategii:", opcje)

if cols == " ":
    housing
    housing_cat = housing["Odległość do oceanu"]  # na później
    with st_stdout("info"):
        print('Zauważmy, że w wierszach takich jak np. 290, 2334, 2028 i wielu innych, zamiast normalnej wartości znajduje się NaN.')
elif cols == "Uzupełnij dane medianą wartości":
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    housing_cat = housing["Odległość do oceanu"]  # na później
    housing_num = housing.drop("Odległość do oceanu", axis=1)
    X = imputer.fit_transform(housing_num)
    housing1 = pd.DataFrame(X, columns=housing_num.columns)
    housing = housing1
    strategy = SimpleImputer(strategy="median")
    housing
    with st_stdout("info"):
        print('Po wypełnieniu luki w pustych miejscach znajdują się już wartości')
elif cols == "Uzupełnij dane średnią":
    strategy = SimpleImputer(strategy="mean")
    housing_cat = housing["Odległość do oceanu"]
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    housing_num = housing.drop("Odległość do oceanu", axis=1)
    X = imputer.fit_transform(housing_num)
    housing1 = pd.DataFrame(X, columns=housing_num.columns)
    housing = housing1
    housing
    with st_stdout("info"):
        print('Po wypełnieniu luki w pustych miejscach znajdują się już wartości')
elif cols == "Uzupełnij dane najczęściej występującą wartością":
    strategy = SimpleImputer(strategy="most_frequent")
    housing_cat = housing["Odległość do oceanu"]
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="most_frequent")
    housing_num = housing.drop("Odległość do oceanu", axis=1)
    X = imputer.fit_transform(housing_num)
    housing1 = pd.DataFrame(X, columns=housing_num.columns)
    housing = housing1
    housing
    with st_stdout("info"):
        print('Po wypełnieniu luki w pustych miejscach znajdują się już wartości')
    # elif cols == "Pozbądź się dystryktów zawierających brakujące dane":
    # housing2 = housing.dropna(subset=["Całk. liczba sypialni"])
    # housing_cat = housing2["Odległość do oceanu"]  # na później
    # housing = housing2.drop("Odległość do oceanu", axis=1)
    # housing.reset_index(drop=True, inplace=True)
    # housing
    # with st_stdout("info"):
    # print('Możesz zauważyć, że teraz mamy już tylko 20433 wierszy z 20640 (liczymy od 0).')
    # elif cols == "Pozbądź się całych atrybutów":
    # housing3 = housing.drop("Całk. liczba sypialni", axis=1)
    # housing3 = housing3.drop("Sypialnie na pokoje", axis=1)
    # housing_cat = housing3["Odległość do oceanu"]  # na później
    # housing = housing3.drop("Odległość do oceanu", axis=1)
    # housing
    # with st_stdout("info"):
    # print('Pozbyyliśmy się atrybutu: "Całk. liczba sypialni" oraz "Sypialnie na pokoje".')


#Przygotowywanie danych
st.subheader("Mamy zbiór danych składający się z 20640 wierszy, nasz algorytm ma je przeanalizować i nauczyć się co wpływa na cenę mieszkań, a następnie po podłożeniu mu zupełnie obcych mu przykładów "
             "ma za zadanie odgadnąć cenę. Zbiór ma 12 atrybutów (kolumn). Jeden z nich to właśnie mediana cen mieszkań.")
st.subheader("Musimy teraz podzielić dane na 2 zbiory (właściwie 3): zbiór uczący i zbiór testowy, który nie będzie miał atrybutu mediany cen mieszkań. Dzięki temu będziemy mogli po wszystkim sprawdzić wydajność "
             "naszego algorytmu. Potrzebujemy jeszcze tylko zbioru poprawnych odpowiedzi do sprawdzenia poprawności wyników.")
zakres=[1,2,3,4,5,6,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,
        51,52,53,54,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,
        81,82,83,84,85,86,87,88,89,90,
        91,92,93,94,95,96,97,98,99]
wartosc = st.select_slider("Jaka częśc całego zbioru będzie stanowić zbiór uczący do zbioru testowego. (Najczęsciej 80%).", zakres)
wartosc = 100 - wartosc
wartosc = wartosc/100

housing["Mediana dochodów cat"] = np.ceil(housing["Mediana dochodów"] / 1.5)
housing["Mediana dochodów cat"].where(housing["Mediana dochodów cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=wartosc, random_state=42)
for train_index, test_index in split.split(housing, housing["Mediana dochodów cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
with st_stdout("info"):
    print("Uczące:", len(strat_train_set), ", Testowe:" ,len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("Mediana dochodów cat", axis=1, inplace=True)

#Zbiory danych
housing_copy = strat_train_set.copy()

housing = strat_train_set
housing_test = strat_test_set.drop("Mediana cen mieszkań", axis=1)
housing_labels = strat_test_set["Mediana cen mieszkań"].copy()

if st.checkbox("zbiór uczący") == True:
    housing
if st.checkbox("zbiór testowy") ==True:
    housing_test
if st.checkbox("zbiór odpowiedzi") == True:
    housing_labels

#Odległość do oceanu
st.header("Obsługa atrybutów **nie**liczbowych")
st.subheader('Bystrzejsze głowy pewnie zauważyły, że gdzieś po drodze utraciliśmy atrybut "Odległość do oceanu". Nie był to przypadek, ponieważ ma on charakter tekstowy i nie bylibyśmy w stanie'
             'obliczyć jego mediany. Jeśli nawet nie wybrałeś tej opcji (z 3 możliwych) to i tak musimy przekształcić atrybut tekstowy na liczby, ponieważ większość algorytmów "woli" pracować z wartościami liczbowymi.')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
housing_cat_encoded, housing_categories = housing_cat.factorize()

encoder = LabelEncoder()
housing_cat_encoded = encoder.fit_transform(housing_cat)
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
show = housing_cat_1hot.toarray()

if st.checkbox("Zobacz jak tłumaczone są atrybuty kategorialne") == True:
    st.subheader("Spójrzmy jeszcze raz na kategorie naszego problematycznego atrybutu:")
    housing_categories

    st.subheader("Tak jak z angielskiego tłumaczymy słowa na polski w taki sposób:")
    slowa = {0: ["BLISKO ZATOKI", "<1H DO OCEANU", "W GŁĘBI KRAJU", "BLISKO OCEANU", "NA WYSPIE"]}
    slowa = pd.DataFrame(slowa)
    slowa

    st.subheader("Tak dla algorytmu tłumaczymy je w taki sposób:")
    binary = {0:[1,0,0,0,0,], 1:[0,1,0,0,0,], 2:[0,0,1,0,0,], 3:[0,0,0,1,0,], 4:[0,0,0,0,1]}
    binary = pd.DataFrame(binary)
    binary

    if st.checkbox("Tak się to prezentuje") == True:
        show


########################################################################################################################

housing = load_housing_data()

# In[ ]:


import numpy as np

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=wartosc, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# In[ ]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# In[ ]:

housing_num = housing.drop("ocean_proximity", axis=1)

# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Pokoje_na_rodzinę = X[:, rooms_ix] / X[:, household_ix]
        Populacja_na_rodzinę = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            Sypialnie_na_pokoje = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, Pokoje_na_rodzinę, Populacja_na_rodzinę, Sypialnie_na_pokoje]
        else:
            return np.c_[X, Pokoje_na_rodzinę, Populacja_na_rodzinę]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=True).transform(housing.values)

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# In[ ]:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', strategy),
    ('attr_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('encoder', OneHotEncoder()),
])

# In[ ]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


housing_prepared = full_pipeline.fit_transform(housing)


###############################################    Wybór modelu   ###################################################
st.header("**Wybór algorytmu**")
st.subheader("Nareszcie! Możemy teraz przetestować różne algorytmy i porównywać ze sobą ich wyniki.")
algorytmy = ["Regresja liniowa"]
algorytm = st.selectbox("Wybierz algorytm", algorytmy)

if algorytm == "Regresja liniowa":
    st.subheader("Dzięki dobrze przygotowanej bazie danych, nasz algorytm mógł nauczyć się rozpoznawania wzorów i korelacji, a teraz po podłożeniu mu zupełnie obcych danych potrafi przewidzieć"
                 "medianę cen mieszkań w dystryktach Kalifornii. Sprawdźmy kilka przykładów ze zbioru testowego i porównajmy.")
    with st_stdout("code"):
        print("Więcej o regresji liniowej: https://pl.wikipedia.org/wiki/Regresja_liniowa")

    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    wyniki = {"Prognozy:": lin_reg.predict(some_data_prepared),
            "Poprawna odpowiedź:": np.array(list(some_labels))}
    wyniki = pd.DataFrame(wyniki)
    st.write("5 przykładów:")
    wyniki
    st.subheader("Działa, choć prognozy nie są zdyt dokładne. Zmierzmy błąd RMSE (Pierwiastek błędu średniokwadratowego) dla całego zbioru testowego.")
    st.write("Średni błąd rmse (im mniejszy tym lepiej):")
    from sklearn.metrics import mean_squared_error

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    with st_stdout("success"):
        print(lin_rmse)

    st.subheader("Nasz system jest gotowy. Możesz teraz prognozować medianę cen mieszkań w Kalifornii za pomocą własnych danych.")

    Dlugosc = st.number_input("Podaj długość geograficzną", min_value=-124.35, max_value=-114.31)
    Szerokosc = st.number_input("Podaj szerokość geograficzną dystryktu", min_value=32.54, max_value=41.95)
    Wiek = st.number_input("Podaj medianę wieku mieszkań", min_value=1, max_value=52)
    Pokoje = st.number_input("Podaj całkowitą liczbe pokoi", min_value=2, max_value=39320)
    Sypialnie = st.number_input("Podaj całkowitą liczbę sypialni", min_value=2, max_value=6210)
    Populacja = st.number_input("Podaj Populacje dystryktu", min_value=3, max_value=35682)
    Rodziny = st.number_input("Podaj liczbę rodzin w dystrykcie", min_value=2, max_value=5358)
    Dochody = st.number_input("Podaj medianę dochodów (w tysiącach $)", min_value=0.4999, max_value=15.0001)
    Pok_na_rodz = st.number_input("Podaj liczbę pokojów na rodzinę", min_value=1.130435, max_value=141.909091)
    Syp_na_pok = st.number_input("Podaj liczbe sypialni na pokoje", min_value=0.1, max_value=1.0)
    Pop_na_rodz = st.number_input("Podaj populacje dystryktu na rodzinę", min_value=0.692308, max_value=1243.3333)
    Ocean = st.selectbox("Wybierz odległość do oceanu", ["BLISKO ZATOKI", "<1H DO OCEANU", "W GŁĘBI KRAJU", "BLISKO OCEANU", "NA WYSPIE"])
    if Ocean == "BLISKO ZATOKI":
        Ocean = "NEAR BAY"
    if Ocean == "<1H DO OCEANU":
        Ocean = "<1H OCEAN"
    if Ocean == "W GŁĘBI KRAJU":
        Ocean = "INLAND"
    if Ocean == "BLISKO OCEANU":
        Ocean = "NEAR OCEAN"
    if Ocean == "NA WYSPIE":
        Ocean = "ISLAND"


    wyniki = {"longitude": Dlugosc,
              "latitude": Szerokosc,
              "housing_median_age": Wiek,
              "total_rooms": Pokoje,
              "total_bedrooms": Sypialnie,
              "population": Populacja,
              "households": Rodziny,
              "median_income": Dochody,
              "ocean_proximity": "INLAND", }
    wyniki = pd.DataFrame(wyniki, index=[0])
    wyniki_prepared = full_pipeline.transform(wyniki)

    with st_stdout("success"):
        print("Mediana cen mieszkań w określonym przez ciebie dystrykcie:", lin_reg.predict(wyniki_prepared), "$")



#Zakończenie
st.header("")
st.header("")
st.header("")
st.header("")

options = [0,1,2,3,5,6,7,8,9,10]
if st.select_slider("Jak oceniasz moją pracę?", options=options) > 2:
    st.balloons()
    st.header("Dziękuję za poświęcony mi czas <3")

