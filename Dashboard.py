import pandas as pd
import os 
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

dir=os.path.join('Datasets','BMW_Car_Sales_Classification.csv')
df=pd.read_csv(dir)

df_numerical=df[['Mileage_KM','Price_USD','Engine_Size_L','Year','Sales_Volume']]
df_categorical=df[['Model','Region','Color','Fuel_Type','Transmission','Sales_Volume']]

df_grouped=df.groupby(['Region','Model'])['Sales_Volume'].agg(['sum','mean']).reset_index()

corr_matrix = df_numerical.corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title="Matriz de Correlación"
)
fig_corr.update_layout(
    title_x=0.5,
    height=500
)

modelo_anova = ols('Sales_Volume ~ C(Model) + C(Region) + C(Color) + C(Fuel_Type) + C(Transmission)',
                   data=df_categorical).fit()

# creando anova
anova_tabla = sm.stats.anova_lm(modelo_anova, typ=2)
# Extraer sum_sq
ss_total = anova_tabla['sum_sq'].sum()
eta_squared = anova_tabla['sum_sq'] / ss_total

# Añadirlo a la tabla original
anova_tabla['eta_squared'] = eta_squared


df_dir=os.path.join('Datasets','BMW_Car_Sales_Classification.csv')
df_clasification=pd.read_csv(df_dir)
df_clasification.drop(['Sales_Volume'],axis=1,inplace=True)

df_clasification['Sales_Classification']=df_clasification['Sales_Classification'].map({'High':1,'Low':0})

categorical = ["Model","Region","Color","Fuel_Type","Transmission"]

df_clasification = pd.get_dummies(df_clasification, columns=categorical, drop_first=True)

X = df_clasification.drop("Sales_Classification", axis=1)
y = df_clasification["Sales_Classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif)),
        ("Model", LogisticRegression(max_iter=5000, class_weight='balanced'))
    ])

param_grid = {
        'selector__k': list(range(1, 11)) + ['all'],
        "Model__C":[0.001,0.01,0.1,1,10,100],
        "Model__solver":["lbfgs","liblinear","saga"]
    }
    
grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    
grid.fit(X_train, y_train)


best_model = grid.best_estimator_

# Obtener coeficientes del modelo
coefs = best_model.named_steps["Model"].coef_[0]

# Obtener las características SELECCIONADAS por SelectKBest
selector = best_model.named_steps["selector"]
selected_mask = selector.get_support()  # Máscara booleana de variables seleccionadas
selected_features = X.columns[selected_mask]

# Crear DataFrame con las variables seleccionadas
coef_df = pd.DataFrame({
    "Variable": selected_features,
    "Log_Odds": coefs
})

# Calcular Odds Ratio
coef_df["Odds_Ratio"] = np.exp(coef_df["Log_Odds"])

# Calcular cambio porcentual
coef_df["Percent_Change"] = (coef_df["Odds_Ratio"] - 1) * 100

# Redondear valores
coef_df["Odds_Ratio"] = coef_df["Odds_Ratio"].round(3)
coef_df["Percent_Change"] = coef_df["Percent_Change"].round(2)

# Ordenar por impacto absoluto (de mayor a menor)
coef_df = coef_df.sort_values("Percent_Change", key=abs, ascending=False)

# Ver resultados


# Probabilidades y predicción
probs = best_model.predict_proba(X_test)[:,1]
y_pred = best_model.predict(X_test)

# ROC
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc_value = roc_auc_score(y_test, probs)

# Accuracy
acc = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

# Crear heatmap
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicho No', 'Predicho Sí'],
    y=['Real No', 'Real Sí'],
    text=cm,
    texttemplate='%{text}',
    textfont={"size": 16},
    colorscale='Blues',
    showscale=True
))

fig_cm.update_layout(
    title='Matriz de Confusión',
    title_x=0.5,
    width=500,
    height=450,
    xaxis_title='Predicción',
    yaxis_title='Valor Real'
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

df_dir=os.path.join('Datasets','BMW_Car_Sales_Classification.csv')
df=pd.read_csv(df_dir)

y = df["Sales_Volume"]

X = df.drop(["Sales_Volume","Sales_Classification"], axis=1)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
})


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


rmse = np.sqrt(mse)

app = Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.H1("BMW", style={
            'textAlign': 'center',
            'fontFamily': 'Arial',
            'fontSize': '68px',
            'fontWeight': 'bold',
            'color': '#0066B4'
        }),
        html.Img(
            src="assets/bmw-logo.svg",
            style={'display': 'block', 'margin': 'auto', 'width': '300px'}
        ),
    ]),

    # 🔥 CONTENEDOR FLEX PARA DROPDOWN Y CHECKLIST - MODIFICADO
    html.Div([
        # Contenedor para el dropdown
        html.Div([
            dcc.Dropdown(
                id='dropdown_box',
                options=[
                    {"label": col, "value": col}
                    for col in df_categorical.columns if col != 'Sales_Volume'
                ],
                value="Region",
                clearable=False
            )
        ], style={'flex': '0.3', 'minWidth': '200px'}),  # 👈 CAMBIADO

        # Contenedor checklist
        html.Div([
            dcc.Checklist(
                id='checklist_region__vehicle',
                options=[
                    {"label": str(valor), "value": valor}
                    for valor in df_grouped['Region'].unique()
                ],
                value=df_grouped['Region'].unique().tolist(),
                inline=True
            )
        ], style={
        'flex': '0.7',  
        'minWidth': '300px',
        'marginLeft': 'auto',   
        'textAlign': 'right',  
        'display': 'flex',
        'justifyContent': 'flex-end'  
    }) 
        
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',  # 👈 AÑADIDO: permite que se envuelvan en móvil
        'gap': '20px',  # 👈 CAMBIADO: de 1200px a 20px
        'padding': '40px',
        'alignItems': 'center'  # 👈 AÑADIDO: alinea verticalmente
    }),

    # CONTENEDOR FLEX PARA LOS GRÁFICOS EN FILA
    html.Div([

        # IZQUIERDA - BOXPLOT
        html.Div([
            dcc.Graph(id='graph_box')
        ], style={
            'flex': '1',  # 👈 CAMBIADO: de 0.5 a 1 para que sea responsive
            'minWidth': '300px',  # 👈 AÑADIDO: tamaño mínimo
            'marginRight': '20px'
        }),

        # DERECHA - GRÁFICO
        html.Div([
            dcc.Graph(id='graph_region_vehicle'),  
        ], style={
            'flex': '1',  # 👈 CAMBIADO: de 0.5 a 1
            'minWidth': '300px',  # 👈 AÑADIDO
            'backgroundColor': '#f5f5f5',
            'borderRadius': '5px',
            'padding': '10px'
        })

    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',  # 👈 AÑADIDO: responsive
        'gap': '20px',
        'padding': '20px'
    }),
#----------------------------------------------------------------------------------------------------------2
# TÍTULOS EN FILA
html.Div([

    html.Div([
        html.H1("Correlación vs Tabla ANOVA", style={
            'fontFamily': 'Times New Roman',
            'fontSize': 'clamp(30px, 5vw, 60px)',
            'fontStyle': 'italic',
            'fontWeight': '300',
            'color': '#0066B4',
            'margin': '0',
            'textAlign': 'center'  # 👈 AÑADIDO: centrar título
        }),
    ], style={
        'flex': '1',
        'minWidth': '250px',
        'textAlign': 'center'  # 👈 AÑADIDO: centrar contenido
    }),

    html.Div([
        html.Img(
            src="assets/bmw-m-logo.png",
            style={
                'width': 'clamp(100px, 20vw, 200px)'
            }
        )
    ], style={
        'flex': '1',
        'textAlign': 'right',  # 👈 Mantiene logo a la derecha
        'minWidth': '100px'
    })

], style={
    'display': 'flex',
    'flexWrap': 'wrap',
    'gap': '20px',
    'padding': '40px 20px 20px 20px',  # 👈 AJUSTADO: menos padding inferior
    'alignItems': 'center'
}),

# CONTENEDOR FLEX PARA GRÁFICOS FILA 2 (Correlación + ANOVA)
html.Div([

    # IZQUIERDA: Gráfico de Correlación
    html.Div([
        dcc.Graph(
            id='graph_correlation',
            figure=fig_corr,
            config={'displayModeBar': True}
        )
    ], style={
        'flex': '1',
        'minWidth': '350px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'padding': '15px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',  # 👈 AÑADIDO: sombra suave
        'marginRight': '10px'
    }),

    # DERECHA: Tabla ANOVA mejorada y centrada
    html.Div([
        html.H3("📊 Tabla ANOVA", style={
            'textAlign': 'center',
            'color': '#0066B4',
            'fontFamily': 'Arial',
            'marginBottom': '15px',
            'fontSize': '20px',
            'fontWeight': 'bold'
        }),
        dash_table.DataTable(   
            id='table_anova',
            columns=[
                {"name": col, "id": col}
                for col in anova_tabla.reset_index().columns
            ],
            data=anova_tabla.reset_index().to_dict('records'),
            style_table={
                'overflowX': 'auto',
                'minWidth': '100%',
                'borderRadius': '8px'
            },
            style_cell={
                'textAlign': 'center',
                'fontFamily': 'Arial',
                'padding': '12px 8px',  # 👈 AUMENTADO: más espaciado
                'fontSize': '14px',
                'backgroundColor': 'white',
                'color': '#333'
            },
            style_header={
                'fontWeight': 'bold',
                'backgroundColor': '#0066B4',
                'color': 'white',
                'padding': '12px 8px',
                'fontSize': '15px',
                'textAlign': 'center'
            },
            style_data_conditional=[  # 👈 NUEVO: estilo condicional para filas
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ],
            page_size=8  # 👈 AÑADIDO: limitar filas por página
        )
    ], style={
        'flex': '1',
        'minWidth': '450px',
        'backgroundColor': '#ffffff',
        'borderRadius': '12px',  # 👈 AUMENTADO: más redondeado
        'padding': '20px',
        'boxShadow': '0 8px 16px rgba(0,0,0,0.1)',  # 👈 MEJORADO: sombra más pronunciada
        'border': '1px solid #e0e0e0',  # 👈 AÑADIDO: borde sutil
        'margin': '0 auto'  # 👈 AÑADIDO: centrado
    })

], style={
    'display': 'flex',
    'flexWrap': 'wrap',
    'gap': '25px',  # 👈 AUMENTADO: más espacio entre elementos
    'padding': '20px',
    'maxWidth': '1400px',  # 👈 AÑADIDO: ancho máximo
    'margin': '0 auto'  # 👈 AÑADIDO: centrar todo el contenedor
}),
    
#------------------------------------------------------------------------------------------------------------------------------------------ 3
    # TÍTULO REGRESIÓN
    html.H1("Evaluación Regresión Logística", style={
        'textAlign': 'center',
        'color': '#0066B4',
        'fontFamily': 'Times New Roman',
        'fontSize': 'clamp(30px, 5vw, 60px)',  # 👈 CAMBIADO
        'fontStyle': 'italic',
        'fontWeight': '300',
        'margin': '20px 0'
    }),

# MÉTRICAS Y VISUALIZACIONES DE REGRESIÓN LOGÍSTICA
html.Div([
    
    # CONTENEDOR PRINCIPAL: Tabla de Coeficientes + Matriz de Confusión
    html.Div([
        
        # IZQUIERDA: Tabla de Coeficientes
        html.Div([
            html.H3(" Coeficientes del Modelo", style={
                'textAlign': 'center',
                'color': '#0066B4',
                'marginBottom': '15px',
                'fontFamily': 'Arial'
            }),
            dash_table.DataTable(
                data=coef_df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in coef_df.columns],
                style_table={
                    'overflowX': 'auto',
                    'minWidth': '100%'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': '14px'
                },
                style_header={
                    'backgroundColor': '#0066B4',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'padding': '10px'
                },
                page_size=10
            )
        ], style={
            'flex': '1',
            'minWidth': '400px',
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # DERECHA: Matriz de Confusión
        html.Div([
            html.H3(" Matriz de Confusión", style={
                'textAlign': 'center',
                'color': '#0066B4',
                'marginBottom': '15px',
                'fontFamily': 'Arial'
            }),
            dcc.Graph(
                id='confusion_matrix',
                figure=fig_cm,
                config={'displayModeBar': True}
            )
        ], style={
            'flex': '1',
            'minWidth': '450px',
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
        
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '20px',
        'marginBottom': '30px',
        'padding': '10px'
    }),
    
    # MÉTRICAS AUC y ACCURACY (debajo, centradas)
    html.Div([
        html.Div([
            html.H4(" AUC (Área bajo la curva)", style={
                'margin': '0',
                'color': '#666',
                'fontSize': '14px'
            }),
            html.H2(f"{auc_value:.3f}", style={
                'margin': '10px 0',
                'color': '#0066B4',
                'fontWeight': 'bold'
            })
        ], style={
            'textAlign': 'center',
            'backgroundColor': '#f5f5f5',
            'padding': '20px',
            'borderRadius': '10px',
            'width': '250px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
        }),
        
        html.Div([
            html.H4(" Accuracy (Exactitud)", style={
                'margin': '0',
                'color': '#666',
                'fontSize': '14px'
            }),
            html.H2(f"{acc:.3f}", style={
                'margin': '10px 0',
                'color': '#0066B4',
                'fontWeight': 'bold'
            })
        ], style={
            'textAlign': 'center',
            'backgroundColor': '#f5f5f5',
            'padding': '20px',
            'borderRadius': '10px',
            'width': '250px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
        })
        
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '30px',
        'padding': '20px',
        'flexWrap': 'wrap'
    })
    
], style={'padding': '20px'}),

#---------------------------------------------------------------------------------------------------------------------------4

# ============================================================================
# SECCIÓN: REGRESIÓN MÚLTIPLE
# ============================================================================

html.Div([
    
    # Título principal
    html.H1("📈 Regresión Múltiple", style={
        'textAlign': 'center',
        'color': '#0066B4',
        'fontFamily': 'Times New Roman',
        'fontSize': 'clamp(35px, 5vw, 60px)',
        'fontStyle': 'italic',
        'fontWeight': '300',
        'margin': '30px 0 20px 0',
        'padding': '20px'
    }),
    
    # CONTENEDOR PRINCIPAL: Tabla de Coeficientes + Métricas
    html.Div([
        
        # COLUMNA IZQUIERDA: Tabla de Coeficientes
        html.Div([
            html.H3("🔧 Coeficientes del Modelo", style={
                'textAlign': 'center',
                'color': '#0066B4',
                'marginBottom': '20px',
                'fontFamily': 'Arial'
            }),
            
            dash_table.DataTable(
                data=coeficientes.to_dict('records'),
                columns=[
                    {"name": "Variable", "id": "Variable"},
                    {"name": "Coeficiente", "id": "Coeficiente"}
                ],
                style_table={
                    'overflowX': 'auto',
                    'borderRadius': '8px'
                },
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': 'Arial',
                    'padding': '12px 10px',
                    'fontSize': '13px',
                    'backgroundColor': 'white'
                },
                style_header={
                    'backgroundColor': '#0066B4',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'padding': '12px 10px',
                    'fontSize': '14px',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ],
                page_size=10
            )
            
        ], style={
            'flex': '1',
            'minWidth': '400px',
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
        }),
        
        # COLUMNA DERECHA: Tarjetas de Métricas
        html.Div([
            html.H3(" Métricas del Modelo", style={
                'textAlign': 'center',
                'color': '#0066B4',
                'marginBottom': '20px',
                'fontFamily': 'Arial'
            }),
            
            # Tarjeta: Intercepto
            html.Div([
                html.H4(" Intercepto", style={
                    'margin': '0',
                    'color': '#666',
                    'fontSize': '14px'
                }),
                html.H2(f"{model.intercept_:.4f}", style={
                    'margin': '10px 0',
                    'color': '#0066B4',
                    'fontWeight': 'bold'
                })
            ], style={
                'textAlign': 'center',
                'backgroundColor': '#f5f5f5',
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': '15px'
            }),
            
            # Tarjeta: MSE
            html.Div([
                html.H4("📉 MSE", style={
                    'margin': '0',
                    'color': '#666',
                    'fontSize': '14px'
                }),
                html.H2(f"{mse:.4f}", style={
                    'margin': '10px 0',
                    'color': '#0066B4',
                    'fontWeight': 'bold'
                })
            ], style={
                'textAlign': 'center',
                'backgroundColor': '#f5f5f5',
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': '15px'
            }),
            
            # Tarjeta: RMSE
            html.Div([
                html.H4(" RMSE", style={
                    'margin': '0',
                    'color': '#666',
                    'fontSize': '14px'
                }),
                html.H2(f"{rmse:.4f}", style={
                    'margin': '10px 0',
                    'color': '#0066B4',
                    'fontWeight': 'bold'
                })
            ], style={
                'textAlign': 'center',
                'backgroundColor': '#f5f5f5',
                'padding': '20px',
                'borderRadius': '10px',
                'marginBottom': '15px'
            }),
            
            # Tarjeta: R²
            html.Div([
                html.H4(" R²", style={
                    'margin': '0',
                    'color': '#666',
                    'fontSize': '14px'
                }),
                html.H2(f"{r2:.4f}", style={
                    'margin': '10px 0',
                    'color': '#0066B4',
                    'fontWeight': 'bold'
                })
            ], style={
                'textAlign': 'center',
                'backgroundColor': '#f5f5f5',
                'padding': '20px',
                'borderRadius': '10px'
            })
            
        ], style={
            'flex': '0.4',
            'minWidth': '280px',
            'padding': '20px',
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
        })
        
    ], style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'gap': '25px',
        'padding': '20px',
        'maxWidth': '1400px',
        'margin': '0 auto'
    })
    
], style={'marginBottom': '40px'}) 

])

#---------------------------------------------------------------------------------------------------------------------------------------------------
@app.callback(
    Output('graph_box', 'figure'),
    Input('dropdown_box', 'value')
)
def update_box(variable):

    fig = px.box(
        df_categorical,
        x=variable,
        y='Sales_Volume',
        color=variable,  
        color_discrete_sequence=px.colors.qualitative.Set2 
    )

    fig.update_layout(
        title=f"Distribución de las ventas en {variable}",
        title_x=0.5,
        height=500
    )

    return fig
    
@app.callback(
    Output('graph_region_vehicle', 'figure'),
    Input('checklist_region__vehicle', 'value')
)

def update_region_vehicle(selected_regions):
    
    # Filtrar el DataFrame por las regiones seleccionadas
    if selected_regions:  # Si hay al menos una región seleccionada
        df_filtered = df_grouped[df_grouped['Region'].isin(selected_regions)]
    else:
        # Si no hay ninguna seleccionada, mostrar todas o un DataFrame vacío
        df_filtered = df_grouped  # Opción: mostrar todas
    
    # Crear el gráfico con los datos filtrados
    fig = px.scatter(
        df_filtered,
        x='Model',  
        y='mean',
        color='Region',  
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        title=f"Promedio de vehículos vendidos por región",
        title_x=0.5,
        height=500
    )
    
    return fig

# Al final del archivo, modifica esta parte:
if __name__ == "__main__":
    # Para desarrollo local
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)

# Agrega esto para Gunicorn (necesario para producción)
server = app.server
