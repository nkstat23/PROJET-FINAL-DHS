# Importations nécessaires
#from jupyter_dash import JupyterDash
from dash import Dash, html,dcc
#import dash_core_components as dcc
#import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# --- Chargement et prétraitement des données avec Pandas ---
#def load_data_pandas():
    # Charger les données à partir d'un fichier CSV
    # Remplacez le chemin par le chemin réel vers votre fichier
    #df = pd.read_csv("donnees.csv")
    #return df

def preprocess_data_pandas():
    df = pd.read_csv("data.csv")
    df = df.drop('Unnamed: 0', axis=1)
    df['item_id'] = df['item_id'].astype(str)
    #df['Year'] = df['Year'].astype(str)
    #df['Month'] = df['Month'].astype(str)
    df['Customer ID'] = df['Customer ID'].astype(str)
    df = df.dropna()
    import datetime
    from datetime import datetime
    df['Working Date'] = df['Working Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['order_date'] = df['order_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return df

# Chargement et prétraitement des données
pdf = preprocess_data_pandas()

# Vérification rapide du DataFrame
print(pdf.head())

# --- Application Dash ---
# On ajoute le thème LUX et la feuille de style Bootstrap Icons
external_stylesheets = [
    dbc.themes.LUX,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
]
## Web App Layout
app = Dash(title="Wine Analysis",external_stylesheets=external_stylesheets)
server = app.server


# =============================================================================
# LAYOUT DE L'APPLICATION
# =============================================================================
app.layout = dbc.Container([
    html.H1("E-commerce Dashboard", className="text-center my-4"),
    dcc.Tabs(id="tabs", value="overview", children=[
        # Onglet Accueil / Vue d'ensemble
        dcc.Tab(label="Accueil", value="overview", children=[
            dbc.Container([
                # Filtres
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Filtres"),
                            dbc.CardBody([
                                html.Label("Période"),
                                dcc.RangeSlider(
                                    id="overview-year-slider",
                                    min=2016,
                                    max=2018,
                                    step=1,
                                    value=[2016, 2018],
                                    marks={year: str(year) for year in range(2016, 2019)}
                                ),
                                html.Br(),
                                html.Label("Catégorie"),
                                dcc.Dropdown(
                                    id="overview-category-dropdown",
                                    options=[{"label": cat, "value": cat} 
                                             for cat in sorted(pdf["category_name_1"].dropna().unique())],
                                    multi=True,
                                    placeholder="Toutes les catégories"
                                ),
                                html.Br(),
                                html.Label("Mode de paiement"),
                                dcc.Dropdown(
                                    id="overview-payment-dropdown",
                                    options=[{"label": pm, "value": pm} 
                                             for pm in sorted(pdf["payment_method"].dropna().unique())],
                                    multi=True,
                                    placeholder="Tous les modes de paiement"
                                )
                            ])
                        ])
                    ], width=12)
                ], className="mb-4"),
                # KPI : Chiffre d'affaires total
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([html.I(className="bi bi-currency-dollar me-2"), "Chiffre d'affaires total"]),
                            dbc.CardBody(html.H4(id="kpi-total-revenue", className="card-title"))
                        ], color="primary", inverse=True),
                        width=12
                    )
                ], className="mb-3"),
                # KPI : Nombre total de commandes
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([html.I(className="bi bi-bag me-2"), "Nombre total de commandes"]),
                            dbc.CardBody(html.H4(id="kpi-total-orders", className="card-title"))
                        ], color="info", inverse=True),
                        width=12
                    )
                ], className="mb-3"),
                # KPI : Panier moyen
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([html.I(className="bi bi-cart me-2"), "Panier moyen"]),
                            dbc.CardBody(html.H4(id="kpi-average-order", className="card-title"))
                        ], color="success", inverse=True),
                        width=12
                    )
                ], className="mb-3"),
                # KPI : Taux de succès
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader([html.I(className="bi bi-check-circle me-2"), "Taux de succès"]),
                            dbc.CardBody(html.H4(id="kpi-success-rate", className="card-title"))
                        ], color="warning", inverse=True),
                        width=12
                    )
                ], className="mb-3"),
                # Graphique : Meilleurs Catégories (camembert)
                dbc.Row([
                    dbc.Col(dcc.Graph(id="overview-best-category-pie"), width=12)
                ], className="mb-3"),
                # Graphique : Histogramme de l'état des commandes par année
                dbc.Row([
                    dbc.Col(dcc.Graph(id="overview-status-histogram"), width=12)
                ], className="mb-3"),
                # Graphique : Répartition des commandes annulées
                dbc.Row([
                    dbc.Col(dcc.Graph(id="overview-cancellation-pie"), width=12)
                ], className="mb-3")
            ])
        ]),
        # Onglet Analyse des ventes par période
        dcc.Tab(label="Analyse des ventes par période", value="sales", children=[
            dbc.Container([
                # Filtres
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Filtres"),
                            dbc.CardBody([
                                html.Label("Catégorie"),
                                dcc.Dropdown(
                                    id="sales-category-dropdown",
                                    options=[{"label": cat, "value": cat} 
                                             for cat in sorted(pdf["category_name_1"].dropna().unique())],
                                    multi=True,
                                    placeholder="Toutes les catégories"
                                ),
                                html.Br(),
                                html.Label("Mode de paiement"),
                                dcc.Dropdown(
                                    id="sales-payment-dropdown",
                                    options=[{"label": pm, "value": pm} 
                                             for pm in sorted(pdf["payment_method"].dropna().unique())],
                                    multi=True,
                                    placeholder="Tous les modes de paiement"
                                )
                            ])
                        ]),
                        width=12
                    )
                ], className="mb-4"),
                # Graphique : Top 10 Best-sellers par an
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sales-best-sellers"), width=12)
                ], className="mb-3"),
                # Graphique : Répartition du CA par jour
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sales-revenue-evolution"), width=12)
                ], className="mb-3"),
                # Graphique : CA par mois
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sales-revenue-month"), width=12)
                ], className="mb-3"),
                # Graphique : Nombre de commandes par jour
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sales-orders-day"), width=12)
                ], className="mb-3"),
                # Graphique : Nombre de commandes par mois
                dbc.Row([
                    dbc.Col(dcc.Graph(id="sales-orders-month"), width=12)
                ], className="mb-3")
            ])
        ]),
        # Onglet Analyse par des produits par catégories
        dcc.Tab(label="Analyse par des produits par catégories", value="product", children=[
            dbc.Container([
                # Filtres
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Filtres"),
                            dbc.CardBody([
                                html.Label("Période"),
                                dcc.RangeSlider(
                                    id="product-year-slider",
                                    min=2016,
                                    max=2018,
                                    step=1,
                                    value=[2016, 2018],
                                    marks={year: str(year) for year in range(2016, 2019)}
                                ),
                                html.Br(),
                                html.Label("Année budgétaire"),
                                dcc.Dropdown(
                                    id="product-fy-dropdown",
                                    options=[{"label": fy, "value": fy} 
                                             for fy in sorted(pdf["FY"].dropna().unique())] if "FY" in pdf.columns else [],
                                    multi=True,
                                    placeholder="Toutes les FY"
                                ),
                                html.Br(),
                                html.Label("Statut des commandes"),
                                dcc.Dropdown(
                                    id="product-status-dropdown",
                                    options=[{"label": stat, "value": stat} 
                                             for stat in sorted(pdf["status"].dropna().unique())],
                                    multi=True,
                                    placeholder="Tous les statuts"
                                )
                            ])
                        ]),
                        width=12
                    )
                ], className="mb-4"),
                # Graphique : Nombre de commandes par jour
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-orders-day"), width=12)
                ], className="mb-3"),
                # Graphique : Best Seller par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-best-seller-cat"), width=12)
                ], className="mb-3"),
                # Graphique : Commande moyenne par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-avg-order-cat"), width=12)
                ], className="mb-3"),
                # Graphique : Produits avec les plus grosses remises
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-top-discounts"), width=12)
                ], className="mb-3"),
                # Graphique : Répartition des commandes par mois et catégorie (Heatmap)
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-orders-heatmap"), width=12)
                ], className="mb-3"),
                # Graphique : Taux de remise moyen par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-avg-discount-cat"), width=12)
                ], className="mb-3"),
                # Graphique : CA moyen par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-avg-basket-cat"), width=12)
                ], className="mb-3"),
                # Graphique : Répartition des modes de paiement par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-payment-cat"), width=12)
                ], className="mb-3"),
                # Graphique : Best Seller par année
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-best-seller-year"), width=12)
                ], className="mb-3"),
                # Graphique : Produit le plus cher par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="product-most-expensive-cat"), width=12)
                ], className="mb-3")
            ])
        ]),
        # Onglet Analyse des paiements et commissions
        dcc.Tab(label="Analyse des paiements et commissions", value="payment", children=[
            dbc.Container([
                # Filtres
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Filtres"),
                            dbc.CardBody([
                                html.Label("Période"),
                                dcc.RangeSlider(
                                    id="payment-year-slider",
                                    min=2016,
                                    max=2018,
                                    step=1,
                                    value=[2016, 2018],
                                    marks={year: str(year) for year in range(2016, 2019)}
                                ),
                                html.Br(),
                                html.Label("Année budgétaire"),
                                dcc.Dropdown(
                                    id="payment-fy-dropdown",
                                    options=[{"label": fy, "value": fy} 
                                             for fy in sorted(pdf["FY"].dropna().unique())] if "FY" in pdf.columns else [],
                                    multi=True,
                                    placeholder="Toutes les FY"
                                )
                            ])
                        ]),
                        width=12
                    )
                ], className="mb-4"),
                # Graphique : Top 7 des modes de paiement
                dbc.Row([
                    dbc.Col(dcc.Graph(id="payment-top-methods"), width=12)
                ], className="mb-3"),
                # Graphique : Fréquence des modes de paiement par catégorie
                dbc.Row([
                    dbc.Col(dcc.Graph(id="payment-frequency"), width=12)
                ], className="mb-3")
            ])
        ]),
        # Onglet Profil Client
        dcc.Tab(label="Profil Client", value="client", children=[
            dbc.Container([
                # Filtres
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader("Filtres"),
                            dbc.CardBody([
                                html.Label("Période"),
                                dcc.RangeSlider(
                                    id="client-year-slider",
                                    min=2016,
                                    max=2018,
                                    step=1,
                                    value=[2016, 2018],
                                    marks={year: str(year) for year in range(2016, 2019)}
                                ),
                                html.Br(),
                                html.Label("Catégorie"),
                                dcc.Dropdown(
                                    id="client-category-dropdown",
                                    options=[{"label": cat, "value": cat} 
                                             for cat in sorted(pdf["category_name_1"].dropna().unique())],
                                    multi=True,
                                    placeholder="Toutes les catégories"
                                ),
                                html.Br(),
                                html.Label("Mode de paiement"),
                                dcc.Dropdown(
                                    id="client-payment-dropdown",
                                    options=[{"label": pm, "value": pm} 
                                             for pm in sorted(pdf["payment_method"].dropna().unique())],
                                    multi=True,
                                    placeholder="Tous les modes de paiement"
                                )
                            ])
                        ]),
                        width=12
                    )
                ], className="mb-4"),
                # Graphique : Top 10 clients par CA
                dbc.Row([
                    dbc.Col(dcc.Graph(id="client-top-customers"), width=12)
                ], className="mb-3"),
                # Tableau : Clients fidèles (ayant commandé pendant 3 années)
                dbc.Row([
                    dbc.Col(dash_table.DataTable(
                        id="client-loyal-table",
                        columns=[{"name": i, "id": i} for i in ["Customer ID", "order_year"]],
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'}
                    ), width=12)
                ], className="mb-3")
            ])
        ])
    ])
], fluid=True)

# =============================================================================
# CALLBACKS – ONGLET ACCUEIL
# =============================================================================
@app.callback(
    [Output("kpi-total-revenue", "children"),
     Output("kpi-total-orders", "children"),
     Output("kpi-average-order", "children"),
     Output("kpi-success-rate", "children")],
    [Input("overview-year-slider", "value"),
     Input("overview-category-dropdown", "value"),
     Input("overview-payment-dropdown", "value")]
)
def update_overview_kpis(year_range, selected_categories, selected_payments):
    df_filtered = pdf[(pdf["Year"].astype(int) >= year_range[0]) & 
                      (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    total_revenue = df_filtered["grand_total"].sum()
    total_orders = df_filtered["item_id"].nunique() if "item_id" in df_filtered.columns else len(df_filtered)
    average_order = total_revenue / total_orders if total_orders != 0 else 0
    successful_orders = (df_filtered[df_filtered["status"] == "complete"]["item_id"].nunique() 
                         if "item_id" in df_filtered.columns else 0)
    success_rate = (successful_orders / total_orders * 100) if total_orders != 0 else 0
    return f"{total_revenue:.2f}", total_orders, f"{average_order:.2f}", f"{success_rate:.2f}%"

@app.callback(
    Output("overview-best-category-pie", "figure"),
    [Input("overview-year-slider", "value"),
     Input("overview-category-dropdown", "value"),
     Input("overview-payment-dropdown", "value")]
)
def update_overview_best_category(year_range, selected_categories, selected_payments):
    df_filtered = pdf[(pdf["Year"].astype(int) >= year_range[0]) & 
                      (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    best_category = df_filtered["category_name_1"].value_counts().reset_index()
    best_category.columns = ["category_name_1", "count"]
    fig = px.pie(best_category, names="category_name_1", values="count",
                 title="Meilleurs Catégories des produits vendus", hole=0)
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(legend=dict(x=1.1, y=0.5))
    return fig

@app.callback(
    Output("overview-status-histogram", "figure"),
    [Input("overview-year-slider", "value"),
     Input("overview-category-dropdown", "value"),
     Input("overview-payment-dropdown", "value")]
)
def update_overview_status_histogram(year_range, selected_categories, selected_payments):
    df_filtered = pdf[(pdf["Year"].astype(int) >= year_range[0]) & 
                      (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    if "Year" in df_filtered.columns:
        fig = px.histogram(df_filtered, x="status", facet_col="Year",
                           title="État des commandes par an")
    else:
        fig = px.histogram(df_filtered, x="status", title="État des commandes")
    fig.update_xaxes(title="État des commandes", tickangle=45)
    fig.update_yaxes(title="Nombre total de commandes")
    fig.update_layout(height=500)
    return fig

@app.callback(
    Output("overview-cancellation-pie", "figure"),
    [Input("overview-year-slider", "value"),
     Input("overview-category-dropdown", "value"),
     Input("overview-payment-dropdown", "value")]
)
def update_overview_cancellation(year_range, selected_categories, selected_payments):
    df_filtered = pdf[(pdf["Year"].astype(int) >= year_range[0]) & 
                      (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    total_orders = len(df_filtered)
    canceled_orders = len(df_filtered[df_filtered["status"] == "canceled"])
    data = {"Status": ["Annulées", "Non annulées"], 
            "Count": [canceled_orders, total_orders - canceled_orders]}
    fig = px.pie(data, names="Status", values="Count",
                 title="Répartition des commandes annulées", hole=0.5)
    return fig

# =============================================================================
# CALLBACKS – ONGLET ANALYSE DES VENTES PAR PÉRIODE
# =============================================================================
@app.callback(
    Output("sales-best-sellers", "figure"),
    [Input("sales-category-dropdown", "value"),
     Input("sales-payment-dropdown", "value")]
)
def update_sales_best_sellers(selected_categories, selected_payments):
    df_filtered = pdf.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    df_completed = df_filtered[df_filtered["status"] == "complete"] if "status" in df_filtered.columns else df_filtered
    best_sellers = (df_completed.groupby(["Year", "sku"])["qty_ordered"]
                    .sum().reset_index().sort_values("qty_ordered", ascending=False)
                    .head(10))
    fig = px.bar(best_sellers, x="Year", y="qty_ordered", color="sku",
                 title="Top 10 Best-sellers par an",
                 labels={"qty_ordered": "Quantité vendue", "Year": "Année"},
                 barmode="group")
    return fig

@app.callback(
    Output("sales-revenue-evolution", "figure"),
    [Input("sales-category-dropdown", "value"),
     Input("sales-payment-dropdown", "value")]
)
def update_sales_revenue_evolution(selected_categories, selected_payments):
    df_filtered = pdf.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    revenue_day = df_filtered.groupby("order_date")["grand_total"].sum().reset_index()
    fig = px.line(revenue_day, x="order_date", y="grand_total",
                  title="Répartition du chiffre d'affaires par jour",
                  labels={"order_date": "Date", "grand_total": "Chiffre d'Affaires"})
    return fig

@app.callback(
    Output("sales-revenue-month", "figure"),
    [Input("sales-category-dropdown", "value"),
     Input("sales-payment-dropdown", "value")]
)
def update_sales_revenue_month(selected_categories, selected_payments):
    df_filtered = pdf.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    revenue_month = df_filtered.groupby(["Year", "Month"])["grand_total"].sum().reset_index()
    fig = px.bar(revenue_month, x="Month", y="grand_total", color="Year",
                 title="Chiffre d'affaire par mois", labels={"grand_total": "CA"})
    return fig

@app.callback(
    Output("sales-orders-day", "figure"),
    [Input("sales-category-dropdown", "value"),
     Input("sales-payment-dropdown", "value")]
)
def update_sales_orders_day(selected_categories, selected_payments):
    df_filtered = pdf.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    orders_day = df_filtered.groupby("order_date").size().reset_index(name="total_orders")
    fig = px.bar(orders_day, x="order_date", y="total_orders",
                 title="Nombre de commandes par jour",
                 labels={"order_date": "Date", "total_orders": "Nombre de Commandes"})
    return fig

@app.callback(
    Output("sales-orders-month", "figure"),
    [Input("sales-category-dropdown", "value"),
     Input("sales-payment-dropdown", "value")]
)
def update_sales_orders_month(selected_categories, selected_payments):
    df_filtered = pdf.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    orders_month = df_filtered.groupby(["Year", "Month"]).size().reset_index(name="total_orders")
    fig = px.bar(orders_month, x="Month", y="total_orders", color="Year",
                 title="Nombre de commandes par mois",
                 labels={"total_orders": "Nombre de Commandes"})
    return fig

# =============================================================================
# CALLBACKS – ONGLET ANALYSE PAR DES PRODUITS PAR CATÉGORIES
# =============================================================================
@app.callback(
    Output("product-orders-day", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_orders_day(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    orders_day = df_filtered.groupby("order_date").size().reset_index(name="total_orders")
    fig = px.bar(orders_day, x="order_date", y="total_orders",
                 title="Nombre de commandes par jour",
                 labels={"order_date": "Date", "total_orders": "Nombre de Commandes"})
    return fig

@app.callback(
    Output("product-best-seller-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_best_seller_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    best = df_filtered.groupby(["category_name_1", "sku"])["qty_ordered"].sum().reset_index()
    best = best.loc[best.groupby("category_name_1")["qty_ordered"].idxmax()]
    fig = px.bar(best, x="category_name_1", y="qty_ordered", color="sku",
                 title="Best Seller par catégorie",
                 labels={"qty_ordered": "Quantité vendue", "category_name_1": "Catégorie"})
    return fig

@app.callback(
    Output("product-avg-order-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_avg_order_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    avg_order = df_filtered.groupby("category_name_1")["grand_total"].mean().reset_index()
    fig = px.bar(avg_order, x="category_name_1", y="grand_total",
                 title="Commande moyenne par catégorie",
                 labels={"grand_total": "Commande moyenne", "category_name_1": "Catégorie"})
    return fig

@app.callback(
    Output("product-top-discounts", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_top_discounts(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    top_discount = df_filtered.sort_values("discount_amount", ascending=False).head(10)
    fig = px.bar(top_discount, x="sku", y="discount_amount",
                 title="Produits avec les plus grosses remises",
                 labels={"sku": "Produit", "discount_amount": "Montant de la remise"})
    return fig

@app.callback(
    Output("product-orders-heatmap", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_orders_heatmap(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    heatmap_data = df_filtered.groupby(["Month", "category_name_1"]).size().reset_index(name="order_count")
    fig = px.density_heatmap(heatmap_data, x="Month", y="category_name_1", z="order_count",
                             title="Répartition des commandes par mois et catégorie",
                             labels={"Month": "Mois", "category_name_1": "Catégorie", "order_count": "Nombre de commandes"},
                             color_continuous_scale="Viridis")
    return fig

@app.callback(
    Output("product-avg-discount-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_avg_discount_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    avg_discount = df_filtered.groupby("category_name_1")["discount_amount"].mean().reset_index()
    fig = px.bar(avg_discount, x="category_name_1", y="discount_amount",
                 title="Taux de remise moyen par catégorie",
                 labels={"category_name_1": "Catégorie", "discount_amount": "Remise moyenne"})
    return fig

@app.callback(
    Output("product-avg-basket-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_avg_basket_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    avg_basket = df_filtered.groupby("category_name_1")["grand_total"].mean().reset_index()
    fig = px.bar(avg_basket, x="grand_total", y="category_name_1", orientation="h",
                 title="CA moyen par catégorie",
                 labels={"grand_total": "Panier moyen", "category_name_1": "Catégorie"})
    return fig

@app.callback(
    Output("product-payment-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_payment_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    payment_cat = df_filtered.groupby(["category_name_1", "payment_method"]).size().reset_index(name="count")
    fig = px.bar(payment_cat, x="category_name_1", y="count", color="payment_method",
                 title="Répartition des modes de paiement par catégorie",
                 labels={"category_name_1": "Catégorie", "count": "Nombre de commandes", "payment_method": "Mode de paiement"})
    return fig

@app.callback(
    Output("product-best-seller-year", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_best_seller_year(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    best_year = df_filtered.groupby(["Year", "sku"])["qty_ordered"].sum().reset_index()
    best_year = best_year.loc[best_year.groupby("Year")["qty_ordered"].idxmax()]
    fig = px.bar(best_year, x="Year", y="qty_ordered", color="sku",
                 title="Best Seller par année",
                 labels={"qty_ordered": "Quantité vendue", "Year": "Année"})
    return fig

@app.callback(
    Output("product-most-expensive-cat", "figure"),
    [Input("product-year-slider", "value"),
     Input("product-fy-dropdown", "value"),
     Input("product-status-dropdown", "value")]
)
def update_product_most_expensive_cat(year_range, selected_fy, selected_status):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    if selected_status:
        df_filtered = df_filtered[df_filtered["status"].isin(selected_status)]
    expensive = df_filtered.loc[df_filtered.groupby("category_name_1")["price"].idxmax()]
    fig = px.bar(expensive, x="category_name_1", y="price", color="sku",
                 title="Produit le plus cher par catégorie",
                 labels={"price": "Prix", "category_name_1": "Catégorie"})
    return fig

# =============================================================================
# CALLBACKS – ONGLET ANALYSE DES PAIEMENTS ET COMMISSIONS
# =============================================================================
@app.callback(
    Output("payment-top-methods", "figure"),
    [Input("payment-year-slider", "value"),
     Input("payment-fy-dropdown", "value")]
)
def update_payment_top_methods(year_range, selected_fy):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    top_methods = df_filtered["payment_method"].value_counts().reset_index().rename(columns={"index": "payment_method", "payment_method": "count"})
    top_methods = top_methods.head(7)
    fig = px.pie(top_methods, names="payment_method", values="count",
                 title="Top 7 des modes de paiement les plus utilisés", hole=0)
    return fig

@app.callback(
    Output("payment-frequency", "figure"),
    [Input("payment-year-slider", "value"),
     Input("payment-fy-dropdown", "value")]
)
def update_payment_frequency(year_range, selected_fy):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_fy:
        df_filtered = df_filtered[df_filtered["FY"].isin(selected_fy)]
    freq = df_filtered.groupby(["category_name_1", "payment_method", "Year", "Month"]).size().reset_index(name="count")
    freq = freq.sort_values(["category_name_1", "Year", "Month", "count"], ascending=[True, True, True, False])
    freq = freq.groupby(["category_name_1", "Year", "Month"]).first().reset_index()
    fig = px.bar(freq, x="category_name_1", y="count", color="payment_method",
                 facet_col="Year", title="Mode de paiement récurrent par catégorie",
                 labels={"count": "Nombre de commandes"})
    return fig

# =============================================================================
# CALLBACKS – ONGLET PROFIL CLIENT
# =============================================================================
@app.callback(
    Output("client-top-customers", "figure"),
    [Input("client-year-slider", "value"),
     Input("client-category-dropdown", "value"),
     Input("client-payment-dropdown", "value")]
)
def update_client_top_customers(year_range, selected_categories, selected_payments):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    top_customers = df_filtered.groupby("Customer ID")["grand_total"].sum().reset_index()
    top_customers = top_customers.sort_values("grand_total", ascending=False).head(10)
    fig = px.bar(top_customers, x="Customer ID", y="grand_total",
                 title="Top 10 des clients les plus contributifs",
                 labels={"Customer ID": "ID Client", "grand_total": "Chiffre d'Affaires"})
    return fig

@app.callback(
    Output("client-loyal-table", "data"),
    [Input("client-year-slider", "value"),
     Input("client-category-dropdown", "value"),
     Input("client-payment-dropdown", "value")]
)
def update_client_loyal_table(year_range, selected_categories, selected_payments):
    df_filtered = pdf.copy()
    df_filtered = df_filtered[(pdf["Year"].astype(int) >= year_range[0]) & (pdf["Year"].astype(int) <= year_range[1])]
    if selected_categories:
        df_filtered = df_filtered[df_filtered["category_name_1"].isin(selected_categories)]
    if selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    df_filtered["order_year"] = df_filtered["order_date"].dt.year
    loyal = df_filtered.groupby("Customer ID")["order_year"].nunique().reset_index()
    loyal = loyal[loyal["order_year"] == 3]
    return loyal.to_dict("records")

# =============================================================================
# LANCEMENT DE L'APPLICATION
# =============================================================================
if __name__ == "__main__":
    app.run_server(debug=True)
