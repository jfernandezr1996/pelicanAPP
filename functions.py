import numpy as np
import base64
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from highlight_text import HighlightText, fig_text, ax_text
import os
import base64
from IPython.core.display import HTML
from matplotlib import colors
import matplotlib.image as mpimg


def get_table_download_link(file):
    data = pd.read_csv(file, sep=";")
    csv_file = data.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="metrics_description.txt">Download description metrics file</a>'


def limits(x, dict_ranges, col):
    if x > dict_ranges[col][1]:
        return dict_ranges[col][1]
    else:
        if x < dict_ranges[col][0]:
            return dict_ranges[col][0]
        else:
            return x
        

def color_background_radar_stats(x):
    palette = sns.color_palette('deep', n_colors = len(x))
    colors_radar = palette.as_hex()
    is_max = x == x.max()
    lista = []
    for c,b in zip(colors_radar, is_max):
        if b:
            lista.append('background-color: {}'.format(c))
        else:
            lista.append('background-color: #F0F2F6')
    return lista

def background_style(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def color_info(s):
    return ['background-color: lightgray' for v in s]

def color_info_2(s):
    return ['background-color: #F0F2F6' for v in s]

def radarchart_pyplot(
        data, data_filter, teams, cols, ranges_cols, 
        percentile, average): 
    
    from functions_radarchart import ComplexRadar
    
    palette = sns.color_palette('deep', n_colors = len(teams)+2)
    all_colors = palette.as_hex()
    j = 1 if average else 0
    colors_teams = all_colors[:len(teams)+j]
    
    comp = data_filter['Competition'].nunique()
    if comp > 1:
        comp_name = 'Big 5 European Leagues, 20/21'
    else:
        comp_name = list(data_filter['Competition'].unique())[0] + ', 20/21'
    
    df_filter = data_filter[['Squad'] + cols]
    data_filter = data[['Squad']+cols]
    df_teams_filter = df_filter[df_filter.Squad.isin(teams)]
    ranges_cols_selected = [ranges_cols[s] for s in cols]
    
    if percentile:
        
        df_plot = pd.DataFrame({'Squad': df_filter['Squad'].tolist()})
        
        for c in df_filter.columns[1:]:
            p_c = round((df_filter[c]-df_filter[c].min())\
                        / (df_filter[c].max()-df_filter[c].min())*100)
            p_c.reset_index(drop=True, inplace=True)
            df_plot = pd.concat([df_plot, p_c], axis=1)
            
        for c in df_plot.columns[1:]:
            df_plot[c] = df_plot[c].astype(int)
            
        df_plot = df_plot[df_plot.Squad.isin(teams)]
        teams_radar = teams
            
    else:
        
        if not average:
        
            df_plot = df_teams_filter.copy()
        
            for c in df_plot.columns[1:]:
                df_plot[c] = df_plot[c].apply(
                    lambda x: limits(x, ranges_cols, c))
            teams_radar = teams
        else:
            prom_all = ['Average ' + comp_name] + \
                [data_filter[c].mean() for c in cols]
            df_plot = pd.concat(
                [df_teams_filter, pd.DataFrame(
                    {x:[y] for x,y 
                     in zip(['Squad']+cols, prom_all)})], axis=0)
            df_plot = df_plot[['Squad']+cols]
            df_plot.reset_index(drop=True, inplace=True)
            for c in df_plot.columns[1:]:
                df_plot[c] = df_plot[c].apply(
                    lambda x: limits(x, ranges_cols, c))
            teams_radar = teams + ['Average ' + comp_name]
            
    fig, ax = plt.subplots(figsize=(6,6)) 
    ax.axis('off')
    radar = ComplexRadar(fig, tuple(cols), ranges_cols_selected)
    for t, color in zip(teams_radar, colors_teams):
        df_teams_t = list(
            df_plot[df_plot.Squad == t].\
                iloc[:,1:].values[0])
        radar.plot(tuple(df_teams_t), color, label = t)
        radar.fill(tuple(df_teams_t), color, alpha = 0.2)
        
    title_plot = ''
    for t in teams:
        title_plot = title_plot + ' <' + t + '> |'
        if [i for i in range(len(teams)) if teams[i] == t][0] in [2,5,8]:
            title_plot = title_plot[:-1] + '\n'
    title_plot = title_plot[1:-2] + '\n<' + comp_name + '>'
    
    highlight_textprops = [{'color': c , 'weight': 'bold'} 
                           for c in colors_teams[:len(teams)]] + \
        [{'color': '#303A5D', 'fontsize': 14, 'weight': 'regular'}]
    y_value = 1.3 if len(teams) < 4 else 1.4
    HighlightText(
        s = title_plot, 
        x = .55, y = y_value,
        fontsize = 18, 
        fontname = 'Roboto',
        color = '#303A5D', 
        highlight_textprops=highlight_textprops,
        textalign='center',
        va = 'center', ha = 'center', ax = ax)
    return fig, df_teams_filter, colors_teams



def calculate_stats(
        df_from_radar, df_total, 
        metrics_selection, teams_selection, select_info_table, dict_metrics, 
        select_percentiles):
    df_radar_T = df_from_radar.copy()
    df_radar_T.index = df_radar_T.Squad
    df_radar_T = df_radar_T.T.iloc[1:,:]
    for c in df_radar_T.columns:
        df_radar_T[c] = df_radar_T[c].astype(float)
    # calculo de percentiles
    df_percentiles = pd.DataFrame()
    for c in list(df_radar_T.index):
        percentil_c = round((df_total[c] - df_total[c].min())\
                / (df_total[c].max() - df_total[c].min())*100)
        df_percentiles = pd.concat([df_percentiles, 
                                    percentil_c], axis=1)
    df_percentiles.index = df_total['Squad']
    df_percentiles = df_percentiles[df_percentiles.index.isin(
        teams_selection)]
    df_percentiles_T = df_percentiles.T
    df_percentiles_T = df_percentiles_T[df_radar_T.columns]
    df_percentiles_T.rename(
        columns={c:c+ " p" for c in df_percentiles_T.columns}, 
        inplace=True)
    for c in df_percentiles_T.columns:
        df_percentiles_T[c] = df_percentiles_T[c].astype(int)
    df_full = pd.concat([df_radar_T, df_percentiles_T], axis=1)
    one_dict = {c: (c, 'Value') 
                for c in df_full.columns[:len(teams_selection)]}
    one_dict.update({c: (c[:-2], 'Percentile') 
                    for c in df_full.columns[len(teams_selection):]})
   
    df_full.rename(columns = one_dict, inplace=True)
    if select_info_table: 
        dict_keys = {y: dict_metrics[y] for y in df_percentiles.columns}
        df_full.index = df_full.index.map(dict_keys)
    # order df
    l = []
    i = 0
    while i < len(teams_selection):
        l.append(i)
        l.append(i+len(teams_selection))
        i+=1
    df_full = df_full[[df_full.columns[j] for j in l]]
    if select_percentiles:
        df_full = df_full[[c for c in df_full.columns if 'Percentile' in c]]
    else:
        df_full = df_full[[c for c in df_full.columns if 'Percentile' not in c]]
    df_full.columns = pd.MultiIndex.from_tuples(df_full.columns)
    df_full = df_full[teams_selection]
    img_list = []
    for c in [df_full.columns[i][0] for i in range(len(teams_selection))]:
        with open(os.getcwd() + "/teams/" + c + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_image = r'<img src="data:image/png;base64,%s" width="60"/>' % encoded_string
            img_list.append(encoded_image)
    df_full_image = df_full.T
    df_full_image['img'] = img_list
    df_full_image.reset_index(inplace=True)
    df_full_image.rename(columns={'level_0': 'Squad', 'level_1': ''}, inplace=True)
    df_full_image = df_full_image[['img', 'Squad'] + \
                                  [c for c in df_full_image.columns if c not in ['img', 'Squad']]]
    df_full_image.rename(columns={'img': '', 'Squad': ''}, inplace=True)
    df_stats_radar = df_full_image.T
        
    df_stats_radar.columns = df_stats_radar.iloc[0,:]
    df_stats_radar_tabla = df_stats_radar.iloc[1:,:]
    df_stats_radar_tabla.reset_index(inplace=True)
    df_stats_radar_tabla.rename(columns={'index': ''}, inplace=True)
    if not select_percentiles:
        html = (
            df_stats_radar_tabla.style
            .format("{:.2f}", subset = (df_stats_radar_tabla.index[2:], 
                                        df_stats_radar_tabla.columns[1:]))
            .apply(color_background_radar_stats, axis=1, 
                   subset = (df_stats_radar_tabla.index[2:], 
                             df_stats_radar_tabla.columns[1:]))
            .apply(color_info, axis = 1, 
                   subset = (df_stats_radar_tabla.index[:2], 
                             df_stats_radar_tabla.columns[1:]))
            .apply(color_info, axis = 0, 
                   subset = (df_stats_radar_tabla.index[2:], 
                             df_stats_radar_tabla.columns[0]))
            .apply(color_info_2, axis = 1, 
                   subset = (df_stats_radar_tabla.index[:2], 
                             df_stats_radar_tabla.columns[0]))
            .set_properties(**{'font-family': 'Calibri', 'text-align': 'center', 
                               'border-width':'thin', 'border-color':'black', 
                               'border-style':'solid', 'border-collapse':'collapse', 
                               'font-size':'11pt'})
            .hide_index().render()
            )
    else:
        html = (
            df_stats_radar_tabla.style
            .background_gradient(cmap='RdYlGn', axis = 0, low = .25, high = .25, 
                                 subset = (df_stats_radar_tabla.index[2:], 
                                           df_stats_radar_tabla.columns[1:]))
            .apply(color_info, axis = 1, 
                   subset = (df_stats_radar_tabla.index[:2], 
                             df_stats_radar_tabla.columns[1:]))
            .apply(color_info, axis = 1, 
                   subset = (df_stats_radar_tabla.index[2:], 
                             df_stats_radar_tabla.columns[0]))
            .apply(color_info_2, axis = 1, 
                   subset = (df_stats_radar_tabla.index[:2], 
                             df_stats_radar_tabla.columns[0]))
            .set_properties(**{'font-family': 'Calibri', 'text-align': 'center', 
                              'border-width':'thin', 'border-color':'black', 
                              'border-style':'solid', 'border-collapse':'collapse', 
                              'font-size': '11pt'})
            .hide_index().render()
            )

    return HTML(html)


def table_stats_scatter(df, stats, competition, dict_metrics, info, sort_metric):
    df_filter = df[df.Competition.isin(competition)]
    df_stats = df_filter[['Squad', 'Competition'] + stats]
    df_stats.sort_values(by=sort_metric, ascending=False, inplace=True)
    if info: 
        dict_stats = {y: dict_metrics[y] for y in stats}
        df_stats.rename(columns=dict_stats, inplace=True)
    img_list = []
    for c in list(df_stats['Squad'].unique()):
        with open(os.getcwd() + "/teams/" + c + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_image = r'<img src="data:image/png;base64,%s" width="30"/>' % encoded_string
            img_list.append(encoded_image)
    df_stats['img'] = img_list
    df_stats.rename(columns={'img': ''}, inplace=True)
    df_stats.index = df_stats['']
    df_stats.drop('', axis=1, inplace = True)
    df_stats_T = df_stats.T.reset_index()
    df_stats_T.rename(columns={'index': ''}, inplace=True)
    pos_float = [i for i in range(len(df_stats.columns)) if df_stats.dtypes[i] == 'float']
    html = (
        df_stats_T.style
            .format("{:.2f}", 
                    subset = (df_stats_T.index[pos_float], 
                              df_stats_T.columns[1:]))
            .apply(color_info, axis = 1, 
                   subset = (df_stats_T.index, df_stats_T.columns[0]))
            .apply(color_info, axis = 0, 
                   subset = (df_stats_T.index[:2], df_stats_T.columns))
            .background_gradient(cmap='RdYlGn', axis = 1, low = .25, high = .25, 
                                 subset = (df_stats_T.index[2:], 
                                           df_stats_T.columns[1:]))
            .set_properties(**{'font-family': 'Calibri', 'text-align': 'center', 
                               'border-width':'thin', 'border-color':'black', 
                               'border-style':'solid', 'border-collapse':'collapse', 
                               'font-size':'11pt'})
            .hide_index().render()
            )
    return HTML(html)
    
    


def scatterplot_stats(df, stats, competition, dict_metrics, tendencia, 
                      colors, minus, color_rectangle):
    import plotly.express as px
    import os
    from sklearn.linear_model import LinearRegression
    import plotly.graph_objects as go
    from sklearn.metrics import r2_score
    
    df_filter = df[df.Competition.isin(competition)]

    df_stats = df_filter[['Squad', 'Competition'] + stats]
    dict_stats = {y: dict_metrics[y] for y in stats}
    df_stats.rename(columns=dict_stats, inplace=True)
    stats = list(dict_stats.values())
    
    # linearRegression
    lr = LinearRegression()
    lr.fit(df_stats[[stats[0]]], df_stats[stats[1]])
    coef_x = lr.coef_[0]
    ct = lr.intercept_
    lr_predict = lr.predict(df_stats[[stats[0]]])
    r2 = round(r2_score(df_stats[stats[1]], lr_predict), 2)
    y_tendencia = [ct + j*coef_x for j in list(df_stats[stats[0]].values)]
    
    fig = px.scatter(df_stats, x=stats[0], y=stats[1], 
                     hover_data=['Squad'], 
                     color_discrete_sequence=['#38616d'])
    if tendencia:
        fig.add_trace(go.Scatter(x=df_stats[stats[0]], 
                                 y=y_tendencia,
                                 mode='lines', 
                                 name='Trendline (r2 = ' + str(r2) + ')', 
                                 line=dict(color='royalblue')))

    max_x = 0.1*(df_stats.iloc[:,-2].max() - df_stats.iloc[:,-2].min())
    max_y = 0.1*(df_stats.iloc[:,-1].max() - df_stats.iloc[:,-1].min())
    for i in range(len(df_stats)):
        t = df_stats.iloc[i,0]
        with open(os.getcwd() + "/teams/" + t + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        # Add the prefix that plotly will want when using the string as source
        encoded_image = "data:image/png;base64," + encoded_string
        fig.add_layout_image(
            dict(
                source = encoded_image,
                xref = "x", yref = "y", 
                x = df_stats.iloc[i,-2], y = df_stats.iloc[i,-1], 
                sizex = max_x, sizey = max_y, 
                sizing = "contain", layer = 'above',
                xanchor = "center", yanchor = "middle"))
    fig.update_traces(textposition='top center')

    
    if color_rectangle:
        
        # inferior derecha
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = list(np.repeat(df_stats[stats[1]].min()-minus, 2)), 
                                 fill = 'none', line_color = colors[1], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y=list(np.repeat(df_stats[stats[1]].mean(), 2)),
                                 fill='tonexty', line_color = colors[1], mode='lines', showlegend=False))
        # inferior izquierda
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = list(np.repeat(df_stats[stats[1]].min()-minus, 2)), 
                                 fill = 'none', line_color = colors[0], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x=[df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y=list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill='tonexty', line_color = colors[0], mode='lines', showlegend=False))
        # superior izquierda
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill = 'none', line_color = colors[2], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].min()-minus, df_stats[stats[0]].mean()], 
                                 y = [df_stats[stats[1]].max()+minus, df_stats[stats[1]].max()+minus], 
                                 fill = 'tonexty', line_color = colors[2], mode='lines', showlegend=False))
        # superior derecha
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = list(np.repeat(df_stats[stats[1]].mean(), 2)), 
                                 fill = 'none', line_color = colors[3], mode='lines', showlegend=False))
        fig.add_trace(go.Scatter(x = [df_stats[stats[0]].mean(), df_stats[stats[0]].max()+minus], 
                                 y = [df_stats[stats[1]].max()+minus, df_stats[stats[1]].max()+minus], 
                                 fill = 'tonexty', line_color = colors[3], mode='lines', showlegend=False))
                                 
        fig.update_yaxes(range=[df_stats[stats[1]].min()-minus, df_stats[stats[1]].max()+minus], row=1, col=1)
        fig.update_xaxes(range=[df_stats[stats[0]].min()-minus, df_stats[stats[0]].max()+minus], row=1, col=1)
        
    else:
        
        fig.add_shape(type='line', 
                      x0=df_stats[stats[0]].mean(), 
                      y0=df_stats[stats[1]].min(),
                      x1=df_stats[stats[0]].mean(),
                      y1=df_stats[stats[1]].max(),
                      line=dict(dash='dot', width=1))
        fig.add_shape(type='line', 
                      x0=df_stats[stats[0]].min(), 
                      y0=df_stats[stats[1]].mean(),
                      x1=df_stats[stats[0]].max(),
                      y1=df_stats[stats[1]].mean(),
                      line=dict(dash='dot', width=1))
        
    s = "<span style='font-size: 18px; font-family: IBM Plex Sans; font-weight: bold; color: #004256'>" + \
        stats[0] + " VS " + stats[1] + "</span><br>"
    call = ''
    for c in competition:
        call = call + c + ' | '
    call = call[:-3]
    s = s + "<span style='font-size: 15px; font-family: IBM Plex Sans; font-weight: bold; color: #38616d'>" + \
        call + "</span><br>"
        
    fig.update_layout(title = s, 
                      font_color = '#004256',
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1))
    
    return fig



def similar_team(
        df_similar, 
        selected_team, 
        selected_number, 
        teams_auto, 
        auto):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px
    import plotly.graph_objects as go
    import os 
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from IPython.core.display import HTML
    
    df_similar_values = df_similar.iloc[:,2:]
    scaler = StandardScaler()
    df_similar_std = scaler.fit_transform(df_similar_values)
    
    df_cosine = pd.DataFrame(cosine_similarity(pd.DataFrame(df_similar_std)))
    df_cosine.columns = df_similar['Squad']
    df_cosine.index = df_similar['Squad']
    
    for c in df_cosine.columns:
        df_cosine[c] = (df_cosine[c] - np.min(df_cosine[c])) / np.ptp(df_cosine[c])
    
    #Filter by team
    if selected_number is not None:
        n_similar_teams = df_cosine[selected_team].sort_values(
            ascending=False)[:selected_number+1]
        n_similar_teams_df = n_similar_teams.reset_index()
    else:
        n_similar_teams = df_cosine[selected_team][[selected_team] + teams_auto].\
            sort_values(ascending=False)
        n_similar_teams_df = n_similar_teams.reset_index()        
    
    n_similar_teams_df = n_similar_teams_df.merge(
        df_similar, on = 'Squad', how = 'left')

    n_similar_teams_df[selected_team] = n_similar_teams_df[selected_team].apply(
        lambda x: round(100*x, 2))
    n_similar_teams_df.rename(columns={selected_team: '% Similarity'}, inplace=True)
    n_similar_teams_df = n_similar_teams_df[['Squad', 'Competition'] + 
                                           [c for c in n_similar_teams_df.columns
                                            if c not in ['Squad', 'Competition']]]
    
    n_similar_teams_df_copy = n_similar_teams_df.copy()
        
    img_list = []
    for c in list(n_similar_teams_df_copy['Squad']):
        with open(os.getcwd() + "/teams/" + c + ".png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_image = r'<img src="data:image/png;base64,%s" width="60"/>' % encoded_string
            img_list.append(encoded_image)
            
    n_similar_teams_df_copy['img'] = img_list
    n_similar_teams_image = n_similar_teams_df_copy.copy()
    n_similar_teams_image = n_similar_teams_image[[
        'img', 'Squad', 'Competition', '% Similarity']]
    n_similar_teams_image.rename(
        columns={'img': ''}, inplace=True)
    df_similar_table = n_similar_teams_image.iloc[1:,:]
    df_similar_table.index = df_similar_table.iloc[:,0]
    df_similar_table = df_similar_table.iloc[:,1:]
        
    html = (
        df_similar_table.style
        .format("{:.2f}", subset = (df_similar_table.index, 
                                    df_similar_table.columns[-1]))
        .apply(background_style, cmap='RdYlGn', m=50, M=100, low=.25, high=.25, 
               subset = (df_similar_table.index, df_similar_table.columns[-1]))
        .apply(color_info, axis = 0, 
               subset = (df_similar_table.index, 
                         df_similar_table.columns[:2]))
        .set_properties(**{'font-family': 'Calibri', 'text-align': 'center', 
                           'border-width':'thin', 'border-color':'black', 
                           'border-style':'solid', 'border-collapse':'collapse', 
                           'font-size':'11pt'})
        .render()
        )
    
    return n_similar_teams_df, HTML(html)


def color_percentiles_zones(val):
    if val < 25:
        color = 'red'
    else:
        if val < 50:
            color = 'darkorange'
        else:
            if val < 75:
                color = 'gold'
            else:
                color = 'green'
    return color


def draw_pitch(team, list_colors, list_values, metric_selected):
    
    line = "black"
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(figsize=(10.4,6.8))
    plt.xlim(-1,121)
    plt.ylim(-4,91)
    ax.axis('off')

    # lineas campo
    ly1 = [0,0,90,90,0]
    lx1 = [0,120,120,0,0]

    plt.plot(lx1,ly1,color=line,zorder=5)

    # area grande
    ly2 = [25,25,65,65] 
    lx2 = [120,103.5,103.5,120]
    plt.plot(lx2,ly2,color=line,zorder=5)

    ly3 = [25,25,65,65] 
    lx3 = [0,16.5,16.5,0]
    plt.plot(lx3,ly3,color=line,zorder=5)

    # porteria
    ly4 = [40.5,40.7,48,48]
    lx4 = [120,120.2,120.2,120]
    plt.plot(lx4,ly4,color=line,zorder=5)

    ly5 = [40.5,40.5,48,48]
    lx5 = [0,-0.2,-0.2,0]
    plt.plot(lx5,ly5,color=line,zorder=5)

    # area pequeña
    ly6 = [36,36,54,54]
    lx6 = [120,114.5,114.5,120]
    plt.plot(lx6,ly6,color=line,zorder=5)

    ly7 = [36,36,54,54]
    lx7 = [0,5.5,5.5,0]
    plt.plot(lx7,ly7,color=line,zorder=5)

    # lineas y puntos
    vcy5 = [0,90] 
    vcx5 = [60,60]
    plt.plot(vcx5,vcy5,color=line,zorder=5)

    plt.scatter(109,45,color=line,zorder=5)
    plt.scatter(11,45,color=line,zorder=5)
    plt.scatter(60,45,color=line,zorder=5)

    # circulos
    circle1 = plt.Circle((109.5,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle2 = plt.Circle((10.5,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=1,alpha=1)
    circle3 = plt.Circle((60,45), 9.15,ls='solid',lw=1.5,color=line, fill=False, zorder=2,alpha=1)

    # rectangulos
    rec1 = plt.Rectangle((103.5,30), 16, 30,ls='-',color="white", zorder=1,alpha=1)
    rec2 = plt.Rectangle((0,30), 16.5, 30,ls='-',color="white", zorder=1,alpha=1)
    #rec3 = plt.Rectangle((-1,-1), 122, 92,color=pitch,zorder=1,alpha=1)
    
    # colors
    zone1 = plt.Rectangle((0, 0), 40, 90, color=list_colors[0], zorder=1, alpha=0.5)
    zone2 = plt.Rectangle((40, 0), 40, 90, color=list_colors[1], zorder=1, alpha=0.5)
    zone3 = plt.Rectangle((80, 0), 40, 90, color=list_colors[2], zorder=1, alpha=0.5)

    #ax.add_artist(rec3)
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(rec1)
    ax.add_artist(rec2)
    ax.add_artist(circle3)
    
    # zones
    ax.add_artist(zone1)
    ax.add_artist(zone2)
    ax.add_artist(zone3)
    
    # text
    plt.text(17, 80, str(list_values[0]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    plt.text(46, 80, str(list_values[1]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    plt.text(97, 80, str(list_values[2]) + " %", color = 'black', 
             weight = 'semibold', size = 12)
    
    # flecha
    plt.arrow(55, -3, dx = 10, dy = 0, linewidth = 1.5, head_width=1)
    
    #image
    img = mpimg.imread('teams/'+team+'.png')
    plt.imshow(img, zorder=0, extent = [35, 85, 20, 70])
    
    plt.title(metric_selected, fontweight = 'semibold')
    
    return fig

    
def color_percentiles(val):
    if val < 25:
        color = 'red'
    else:
        if val < 50:
            color = 'darkorange'
        else:
            if val < 75:
                color = 'gold'
            else:
                color = 'green'
    return color


def plot_zones(df_zone, team_zone_select, metric_selected):
    df_zone_percent = df_zone.copy()
    for c in [c for c in df_zone_percent.columns if '90' in c]:
        df_zone_percent[c] = round((df_zone_percent[c] - df_zone_percent[c].min()) \
                               / (df_zone_percent[c].max() - df_zone_percent[c].min())*100)
        df_zone_percent[c] = df_zone_percent[c].astype(int)
    df_zone_percent = df_zone_percent[['Squad'] + \
                                      [c for c in df_zone_percent.columns 
                                       if '90' in c]]
    df_zone_percent.rename(columns={c: c[:-3] 
                                    for c in df_zone_percent.columns[1:]}, 
                           inplace=True)
    df_zone_pct = df_zone[[c for c in df_zone.columns[:-3]]]

    i_team = [i for i in range(len(df_zone_percent)) 
              if df_zone_percent.iloc[i,0] == team_zone_select][0]
    colors_team = list(df_zone_percent.iloc[i_team,1:].\
                       apply(color_percentiles_zones).values)
    values_team = list(df_zone_pct.iloc[i_team,1:].values)
    fig = draw_pitch(team_zone_select, colors_team, values_team, metric_selected)
    return fig


def find_axes(n_metrics):
    if n_metrics <= 3:
        r, c = 1, n_metrics
    elif n_metrics == 4:
        r, c = 2, 2
    elif n_metrics <= 6:
        r, c = 2, 3
    else:
        r, c = 3, 3
    return r, c


def swarmplot_diagram(df, team, metrics, dict_metrics, league, additional_vars):
    if league != 'All Competitions':
        df_f = df[df.Competition == league]
        comp = league + ", 20/21"
    else:
        df_f = df
        comp = 'Big 5 European Leagues, 20/21'
    # colors
    #colors = sns.color_palette('RdYlGn', n_colors = 10)
    #colors = colors.as_hex()
    colors = ['red', 'coral', 'gold', 'lightgreen', 'forestgreen']
    palette = sns.color_palette('deep', n_colors = len(team))
    squad_colors = palette.as_hex()
    #squad_colors = ['#1C2833', '#5D6D7E', '#95A5A6']
    # find correct axes
    n_axes = find_axes(len(metrics))
    fig, axes = plt.subplots(
        nrows=n_axes[0], ncols=n_axes[1], figsize=(20,10))
    if len(metrics) in [5,8]:
        axes[-1,-1].axis('off')
    elif len(metrics) == 7:
        axes[-1,-2].axis('off')
        axes[-1,-1].axis('off')
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    # dataframe and dict description
    dict_need = {k: dict_metrics[k] for k in metrics}
    rename_metrics = list(dict_need.values())
    df_rename = df_f.rename(columns={i:j for i,j in zip(metrics, rename_metrics)})
    df_rename = df_rename[[additional_vars]+rename_metrics]
    metrics = rename_metrics
    # bucle graphs
    ejes = [axes] if len(metrics) == 1 else axes.flatten()
    for m, axis in zip(metrics, ejes):
        axis.grid(ls='dotted', lw=.5, color='lightgrey', axis='y', zorder = 1)
        for x in ['top','left', 'right']:
            axis.spines[x].set_visible(False)
        #axis.spines['bottom'].set_position(('outward', 10))
        limits = [np.percentile(df_rename[m], k) for k in range(20,120,20)]
        dict_limits = {str(j):str(k) for j,k in 
                       zip(limits, list(range(20,120,20)))}
        
        df_metric = df_rename[[m]]
        df_metric['Percentile'] = df_metric[m].apply(
            lambda x: str([j for j in limits if x <= j][0]))
        df_metric['Percentile'] = df_metric['Percentile'].map(dict_limits)
        df_metric['yaxis'] = ''
    
        sns.swarmplot(x = m, y = 'yaxis', hue = 'Percentile', 
                      palette = colors, ax = axis, 
                      zorder = 1,
                      edgecolor = 'white', 
                      data = df_metric, 
                      hue_order = [str(k) for k in range(20,120,20)])
        axis.legend('', frameon=False)
        axis.set_xlabel(f'{m}', c = 'black')
        axis.set_ylabel('')
        
        for s, h in zip(team, squad_colors):
            axis.scatter(x = df_rename.loc[df_rename[additional_vars] == s, m].values[0], 
                         y = 0, s = 200, c = h, zorder = 2, edgecolors='black')
            
        best_df = df_rename[df_rename[m] == df_rename[m].max()]
        axis.text(best_df[m].values[0],-0.07, best_df[additional_vars].values[0], 
                  fontsize=10, color='black')
    
    # legend and percentiles
    lines, labels = fig.axes[0].get_legend_handles_labels()
    labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    #labels = ['0-10', '10-20', '20-30', '30-40', '40-50', 
    #          '50-60', '60-70', '70-80', '80-90', '90-100']
    leg = fig.legend(lines, labels, loc = [0.08,0.001], ncol = 10,
                     fontsize = 'medium',
                     title = 'Percentile Colouring', title_fontsize = 'large', 
                     frameon = True, edgecolor = 'black', facecolor = '#E5E8E8')
    for text in leg.get_texts():
        text.set_color('black')
        
    # title
    title = ''
    for x in team:
        title = title + ' <' + x + '> |'
    title = title[:-2] + '\n <' + comp + '>'
        
    highlight_params = [{'color': c , 'fontsize': 22, 'fontfamily': 'IBM Plex Sans',
                         'weight': 'bold'} 
                        for c in squad_colors] + \
        [{'color': '#303A5D', 'fontfamily': 'IBM Plex Sans',
          'fontsize': 18, 'weight': 'regular'}]
        
    y_value = .88 if len(metrics) < 7 else .92
    fig_text(
        s = title, 
        x = .52, y = y_value,
        fontsize = 22, 
        fontname = 'Roboto',
        highlight_textprops = highlight_params,
        textalign='center', 
        ha = 'center', va = 'top')
    
    return fig
    

    
    
    