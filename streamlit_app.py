import streamlit as st
import shap
import altair as alt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from xgboost import XGBRegressor


def main():
    st.set_page_config(layout='wide', page_title="Производительность работников швейной индустрии")

    st.title('Конушкин Илья Федорович 2023-ФГиИБ-ПИ-1б Вариант 11 Производительность работников швейного предприятия')

    df = pd.read_csv('dataframe.csv')

    # создаю новые признаки
    df['is_idle'] = df['idle_time'] != 0
    df['is_style_change'] = df['no_of_style_change'] != 0
    df['productivity_dependence'] = (df['actual_productivity'] / df['targeted_productivity']) >= 1
    bins = range(0, int(df['no_of_workers'].max()) + 10, 10)
    df['workers_group'] = pd.cut(df['no_of_workers'], bins=bins, labels=False)

    # заполняю пропуски
    df['wip'] = df['wip'].fillna(df['wip'].median())

    # кодирую категориальные признаки
    df = pd.get_dummies(df, columns=['department', 'quarter'])
    df['date'] = pd.to_datetime(df['date'])
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    oenc = OrdinalEncoder(categories=[day_order], dtype=int)
    df['day_encoded'] = oenc.fit_transform(df[['day']])

    # было принято решение распарсить дату на компоненты из-за того что scaler не смог ее стандартизировать
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_n'] = df['date'].dt.day

    # отбросил признак day, так как уже есть его дискретное представление,
    # date, чтобы scaler не ругался,
    # year, потому что там только 2015
    df = df.drop(columns=['date', 'day', 'year'])

    @st.cache_data
    def model():
        model = XGBRegressor()
        model.load_model('model.json')
        return model

    model = model()

    X = df.drop(columns=['actual_productivity', 'productivity_dependence'])
    Y = df['actual_productivity']

    # делим данные на тестовую и обучающую выборки
    rng = np.random.RandomState(2)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rng)

    Y_pred = model.predict(X_test)

    tab1, tab2, tab3, tab4 = st.tabs(['Исходные данные', 'Графики зависимостей', 'Анализ SHAP', 'Оценка точности'])

    with tab1:
        st.subheader('Исходные данные')
        st.dataframe(pd.read_csv('dataframe.csv'))

    with tab2:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('Зависимость производительности от количества незавершенных работ')

            chart = alt.Chart(df).mark_circle().encode(
                x=alt.X('wip',
                        scale=alt.Scale(domain=[-200, 3200])),
                y='actual_productivity'
            ).interactive()

            st.altair_chart(chart)

        with col2:
            prod_by_workers = df.groupby('workers_group', observed=False)['actual_productivity'].mean().reset_index()
            st.subheader('Средняя производительность по группам работников')

            chart = alt.Chart(prod_by_workers).mark_bar(size=25).encode(
                x='workers_group',
                y='actual_productivity',
                tooltip=['workers_group', 'actual_productivity']
            ).interactive()

            st.altair_chart(chart)

        with col3:
            st.subheader("Показатель выполнения плана продуктивности")

            pivot_table1 = pd.crosstab(df['workers_group'], df['productivity_dependence'], normalize='index').reset_index()

            pivot_melted = pivot_table1.melt(
                id_vars='workers_group',
                value_vars=[False, True],
                var_name='productivity_dependence',
                value_name='Доля'
            )

            pivot_melted['productivity_dependence'] = pivot_melted['productivity_dependence'].astype(str)
            pivot_melted['workers_group'] = pivot_melted['workers_group'].astype(str)

            chart = alt.Chart(pivot_melted).mark_rect().encode(
                x='productivity_dependence',
                y='workers_group',
                color=alt.Color('Доля:Q', scale=alt.Scale(scheme='redyellowgreen'), legend=alt.Legend(title="Доля"))
            ).properties(height=400)

            text_layer = chart.mark_text(baseline='middle', fontSize=12).encode(
                text=alt.Text('Доля:Q', format='.2f'),
                color=alt.value('black')
            )

            st.altair_chart(chart + text_layer, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Общий анализ признаков')

            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            fig1 = plt.figure(figsize=(6, 4))
            shap.plots.initjs()
            shap.plots.beeswarm(shap_values, max_display=13, show=False)

            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.subheader('Значимость характеристик')

            fig2 = plt.figure(figsize=(6, 4))
            shap_values = shap.TreeExplainer(model).shap_values(X_test)

            shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=13)
            st.pyplot(fig2)

    with tab4:
        st.subheader('Метрики качества модели')
        col1, col2, col3, col4 = st.columns(4)

        col1.metric('(MAPE) Средняя абсолютная ошибка в процентах: ', f'{(mean_absolute_percentage_error(Y_test, Y_pred) * 100):.4f}')
        col2.metric('(MSE) Среднеквадратическая ошибка: ', f'{mean_squared_error(Y_test, Y_pred):.4f}')
        col3.metric('(MAE) Средняя абсолютная ошибка: ', f'{mean_absolute_error(Y_test, Y_pred):.4f}')
        col4.metric('(RMSE) Корень из среднеквадратической ошибки: ', f'{root_mean_squared_error(Y_test, Y_pred):.4f}')

if __name__ == "__main__":
    main()

